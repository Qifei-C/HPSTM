import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection
import sys
from pathlib import Path
import random # For selecting random NPZ files
import glob # For finding NPZ files
import shutil # For rmtree

# --- Setup Project Paths ---
# This assumes the script is in Manifold/scripts/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added '{project_root}' to sys.path")

try:
    from src.models.pose_refiner_transformer import ManifoldRefinementTransformer
    from src.kinematics.skeleton_utils import get_skeleton_parents
    from src.datasets.amass_dataset import AMASSSubsetDataset # To generate noisy data
    # infer.py contains load_model_from_checkpoint and refine_sequence_transformer
    from scripts.infer import load_model_from_checkpoint, refine_sequence_transformer
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from a location where 'src' and 'scripts' are accessible,")
    print(f"or that '{project_root}' is correctly added to PYTHONPATH.")
    sys.exit(1)

# _create_grid_animation function remains the same as in the previous version
def _create_grid_animation(
    clean_sequences_list: list[np.ndarray],
    noisy_sequences_list: list[np.ndarray],
    refined_sequences_list: list[np.ndarray],
    sequence_source_info: list[str], # e.g., NPZ filenames
    skeleton_type: str,
    suptitle_extra_info: str = "",
    fps: int = 30,
    center_around_root_in_plot: bool = True,
) -> FuncAnimation:
    num_rows = len(clean_sequences_list)
    if not (num_rows == len(noisy_sequences_list) == len(refined_sequences_list) == len(sequence_source_info)):
        raise ValueError("All sequence lists and source info must have the same length (number of rows).")
    if num_rows == 0:
        # This case should be handled before calling, but as a safeguard:
        print("Warning: _create_grid_animation called with no sequences.")
        # Create an empty figure or raise error, depending on desired behavior for no data
        fig, _ = plt.subplots()
        plt.close(fig) # Close immediately
        return fig # Or raise error

    T_common_list = [seq.shape[0] for seq in clean_sequences_list]
    if not all(t == T_common_list[0] for t in T_common_list):
        # This check might be too strict if we allow variable lengths from upstream,
        # but the script aims for args.window_size_anim for all.
        print(f"Warning: Not all clean sequences have the same number of frames. Using frames from first sequence: {T_common_list[0]}")
    T_common = T_common_list[0] # Number of frames for animation, should be consistent
    
    J_common = clean_sequences_list[0].shape[1] # Number of joints

    for i in range(num_rows):
        for seq_list_idx, (seq_list, name) in enumerate(zip(
            [clean_sequences_list, noisy_sequences_list, refined_sequences_list],
            ["Clean", "Noisy", "Refined"]
        )):
            seq = seq_list[i]
            if seq.ndim != 3 or seq.shape[2] != 3:
                raise ValueError(f"{name} sequence {i} ('{sequence_source_info[i]}') must have shape (T, J, 3), got {seq.shape}")
            if seq.shape[0] != T_common:
                # This means upstream logic didn't perfectly enforce T_common for all visualised sequences
                # For animation, FuncAnimation uses frames from the first sequence or a specified 'frames' count.
                # We ensure all data passed here has T_common frames.
                raise ValueError(f"{name} sequence {i} ('{sequence_source_info[i]}') has {seq.shape[0]} frames, expected {T_common}. Check processing logic.")
            if seq.shape[1] != J_common:
                raise ValueError(f"{name} sequence {i} ('{sequence_source_info[i]}') has {seq.shape[1]} joints, expected {J_common}")


    skeleton_parents = get_skeleton_parents(skeleton_type)

    fig, axes = plt.subplots(
        num_rows, 3, figsize=(18, 6 * num_rows), subplot_kw={"projection": "3d"}, squeeze=False
    )
    # squeeze=False ensures 'axes' is always 2D, even if num_rows is 1.

    fig.suptitle(f"Clean vs. Noisy vs. Refined ({suptitle_extra_info})", fontsize=16)

    all_sequences_for_plotting_flat = []
    data_for_subplots = [[None]*3 for _ in range(num_rows)]

    for r_idx in range(num_rows):
        data_for_subplots[r_idx][0] = clean_sequences_list[r_idx].copy()
        data_for_subplots[r_idx][1] = noisy_sequences_list[r_idx].copy()
        data_for_subplots[r_idx][2] = refined_sequences_list[r_idx].copy()
        all_sequences_for_plotting_flat.extend(data_for_subplots[r_idx])

    if center_around_root_in_plot:
        print("Centering all sequences around their respective root joints for plotting.")
        for i_seq_flat in range(len(all_sequences_for_plotting_flat)):
            # This modifies the copies in data_for_subplots as they are the same objects
            seq_data_flat = all_sequences_for_plotting_flat[i_seq_flat]
            root_trajectory = seq_data_flat[:, 0:1, :].copy()
            for t_idx in range(seq_data_flat.shape[0]):
                seq_data_flat[t_idx] -= root_trajectory[t_idx]
    
    temp_concat_for_limits = np.concatenate(all_sequences_for_plotting_flat, axis=0)
    x_min_disp, x_max_disp = temp_concat_for_limits[..., 0].min(), temp_concat_for_limits[..., 0].max()
    y_min_disp, y_max_disp = temp_concat_for_limits[..., 1].min(), temp_concat_for_limits[..., 1].max()
    z_min_disp, z_max_disp = temp_concat_for_limits[..., 2].min(), temp_concat_for_limits[..., 2].max()

    mid_x = (x_min_disp + x_max_disp) / 2
    mid_y = (y_min_disp + y_max_disp) / 2
    mid_z = (z_min_disp + z_max_disp) / 2
    range_x_disp = x_max_disp - x_min_disp
    range_y_disp = y_max_disp - y_min_disp
    range_z_disp = z_max_disp - z_min_disp
    max_total_range = max(range_x_disp, range_y_disp, range_z_disp, 0.1) 

    plot_margin_factor = 0.15
    half_span = max_total_range / 2 * (1 + plot_margin_factor)

    subplot_titles_base = ["Original Clean", "Noisy Input", "Refined Output"]
    point_colors = ("royalblue", "red", "green")
    bone_colors = ("cornflowerblue", "lightcoral", "lime")

    scatters_collection: list[list[plt.Artist]] = [[None]*3 for _ in range(num_rows)]
    bone_lines_collection: list[list[list[plt.Line2D]]] = [[[] for _ in range(3)] for _ in range(num_rows)]
    title_texts_collection: list[list[plt.Text]] = [[None]*3 for _ in range(num_rows)]

    for r_idx in range(num_rows):
        for c_idx in range(3):
            ax = axes[r_idx, c_idx]
            ax.set_xlim(mid_x - half_span, mid_x + half_span)
            ax.set_ylim(mid_y - half_span, mid_y + half_span)
            ax.set_zlim(mid_z - half_span, mid_z + half_span)
            try:
                ax.set_box_aspect((1, 1, 1))
            except AttributeError:
                ax.set_aspect("auto") 

            y_label_str = ""
            if c_idx == 0: 
                y_label_str = f"Seq {r_idx+1}\n({sequence_source_info[r_idx]})"
            ax.set_xlabel("X")
            ax.set_ylabel(y_label_str if c_idx == 0 else "Y")
            ax.set_zlabel("Z")
            
            if c_idx == 0: # Adjust label padding for the first column's Y label
                ax.yaxis.label.set_fontsize(10)
                # ax.yaxis.set_label_coords(-0.15, 0.5) # May need adjustment based on figure size

            title_obj = ax.set_title(f"{subplot_titles_base[c_idx]} — Frame 0/{T_common - 1}")
            title_texts_collection[r_idx][c_idx] = title_obj

            pose0 = data_for_subplots[r_idx][c_idx][0] 
            scat = ax.scatter(
                pose0[:, 0], pose0[:, 1], pose0[:, 2],
                c=point_colors[c_idx], s=30, marker="o",
                edgecolors="black", linewidths=0.2, depthshade=True,
            )
            scatters_collection[r_idx][c_idx] = scat

            current_ax_bone_lines = []
            for j, parent in enumerate(skeleton_parents):
                if parent == -1: continue
                (line,) = ax.plot(
                    [pose0[j, 0], pose0[parent, 0]],
                    [pose0[j, 1], pose0[parent, 1]],
                    [pose0[j, 2], pose0[parent, 2]],
                    color=bone_colors[c_idx], linewidth=1.5,
                )
                current_ax_bone_lines.append(line)
            bone_lines_collection[r_idx][c_idx] = current_ax_bone_lines
            ax.view_init(elev=15.0, azim=-75)

    def _update(frame: int):
        artists_to_return: list[plt.Artist] = []
        for r_idx_update in range(num_rows):
            for c_idx_update in range(3):
                # Ensure frame index is valid for potentially shorter actual data if T_common was from elsewhere
                current_frame_idx = min(frame, data_for_subplots[r_idx_update][c_idx_update].shape[0] - 1)
                current_pose = data_for_subplots[r_idx_update][c_idx_update][current_frame_idx]
                
                scat = scatters_collection[r_idx_update][c_idx_update]
                lines_for_current_ax = bone_lines_collection[r_idx_update][c_idx_update]
                title_obj = title_texts_collection[r_idx_update][c_idx_update]

                scat._offsets3d = (current_pose[:, 0], current_pose[:, 1], current_pose[:, 2])
                artists_to_return.append(scat)

                line_i = 0
                for j_idx, parent_idx in enumerate(skeleton_parents):
                    if parent_idx == -1: continue
                    lines_for_current_ax[line_i].set_data_3d(
                        [current_pose[j_idx, 0], current_pose[parent_idx, 0]],
                        [current_pose[j_idx, 1], current_pose[parent_idx, 1]],
                        [current_pose[j_idx, 2], current_pose[parent_idx, 2]],
                    )
                    artists_to_return.append(lines_for_current_ax[line_i])
                    line_i += 1
                
                title_obj.set_text(f"{subplot_titles_base[c_idx_update]} — Frame {frame}/{T_common - 1}")
                artists_to_return.append(title_obj)
        return artists_to_return

    plt.tight_layout(rect=[0, 0.02, 1, 0.96]) # Adjust for suptitle and bottom
    interval_ms = max(20, int(1000 / fps))
    # The 'frames' argument for FuncAnimation determines how many times _update is called.
    # This should be T_common, which is args.window_size_anim based on successful processing.
    return FuncAnimation(fig, _update, frames=T_common, interval=interval_ms, blit=False)


# parse_vis_args function remains the same
def parse_vis_args():
    parser = argparse.ArgumentParser(description="Visualize Inference of Human Pose Smoothing Model")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file)")
    parser.add_argument('--input_npz_root_dir', type=str, required=True,
                        help="Path to the root directory containing subfolders with processed AMASS .npz files.")
    parser.add_argument('--num_sequences_to_visualize', type=int, default=5,
                        help="Number of random NPZ files to select for visualization.")
    parser.add_argument('--output_anim_path', type=str, default="visualized_inference_grid.gif",
                        help="Path to save the output animation (.gif or .mp4)")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (cuda or cpu)")
    parser.add_argument('--window_size_anim', type=int, default=300,
                        help="Required number of frames from each input NPZ to use for animation. Files shorter than this will be skipped.")
    parser.add_argument('--center_plot', action='store_true',
                        help="Center poses around root joint for visualization display.")
    # Noise parameters
    parser.add_argument('--gaussian_noise_std', type=float, default=0.01) 
    parser.add_argument('--temporal_noise_type', type=str, default='filtered', choices=['none', 'filtered', 'persistent'])
    parser.add_argument('--temporal_noise_scale', type=float, default=0.01)
    parser.add_argument('--temporal_filter_window', type=int, default=7)
    parser.add_argument('--temporal_event_prob', type=float, default=0.05)
    parser.add_argument('--temporal_decay', type=float, default=0.9)
    parser.add_argument('--outlier_prob', type=float, default=0.015)
    parser.add_argument('--outlier_scale', type=float, default=0.3)
    parser.add_argument('--bonelen_noise_scale', type=float, default=0.01)

    args = parser.parse_args()
    return args

def main_visualize(args):
    device = torch.device(args.device)
    random.seed(42) 

    # 1. Load Model
    model, model_constructor_args = load_model_from_checkpoint(args.checkpoint_path, device)
    model_window_size = model_constructor_args['window_size']
    skeleton_type = model_constructor_args.get('skeleton_type', 'smpl_24')
    model_was_trained_on_centered_data = model_constructor_args.get('center_around_root_amass', True)

    # 2. Find NPZ Files
    if not os.path.isdir(args.input_npz_root_dir):
        print(f"Error: Input NPZ root directory not found: {args.input_npz_root_dir}")
        return

    all_npz_files = glob.glob(os.path.join(args.input_npz_root_dir, "**", "*.npz"), recursive=True)
    if not all_npz_files:
        print(f"Error: No .npz files found in {args.input_npz_root_dir} or its subdirectories.")
        return
    
    random.shuffle(all_npz_files) # Shuffle to try different files if some are too short

    all_clean_sequences = []
    all_noisy_sequences_for_plot = []
    all_refined_sequences = []
    all_sequence_source_names = []
    
    target_num_sequences = args.num_sequences_to_visualize
    files_attempted_idx = 0

    # Define noisy_dataset_args structure once, update specific paths later
    # Note: window_size for dataset should be args.window_size_anim as we select files of this length
    base_noisy_dataset_args = dict(
        window_size=args.window_size_anim, 
        skeleton_type=skeleton_type,
        is_train=True, # Activate noise
        center_around_root=model_was_trained_on_centered_data,
        joint_selector_indices=None,
        gaussian_noise_std=args.gaussian_noise_std,
        temporal_noise_type=args.temporal_noise_type,
        temporal_noise_scale=args.temporal_noise_scale,
        temporal_filter_window=args.temporal_filter_window,
        temporal_event_prob=args.temporal_event_prob,
        temporal_decay=args.temporal_decay,
        outlier_prob=args.outlier_prob,
        outlier_scale=args.outlier_scale,
        bonelen_noise_scale=args.bonelen_noise_scale
    )

    while len(all_clean_sequences) < target_num_sequences and files_attempted_idx < len(all_npz_files):
        npz_path_str = all_npz_files[files_attempted_idx]
        files_attempted_idx += 1
        
        npz_path = Path(npz_path_str)
        print(f"\nAttempting sequence {len(all_clean_sequences) + 1}/{target_num_sequences} (File {files_attempted_idx}/{len(all_npz_files)}): {npz_path.name}")
        
        current_file_processed_successfully = False
        temp_dir = None # Initialize temp_dir to ensure it's cleaned up if created

        try:
            data_npz = np.load(npz_path, allow_pickle=True)
            if 'poses_r3j' not in data_npz or 'bone_offsets' not in data_npz:
                print(f"  Info: 'poses_r3j' or 'bone_offsets' not found in {npz_path.name}. Skipping.")
                continue 

            clean_poses_r3j_full = data_npz['poses_r3j'].astype(np.float32)
            
            if clean_poses_r3j_full.shape[0] == 0:
                print(f"  Info: Sequence in {npz_path.name} has 0 frames. Skipping.")
                continue

            if clean_poses_r3j_full.shape[0] < args.window_size_anim:
                print(f"  Info: Sequence in {npz_path.name} has {clean_poses_r3j_full.shape[0]} frames, "
                      f"less than required window_size_anim {args.window_size_anim}. Skipping.")
                continue
                
            # Sequence length is sufficient, use args.window_size_anim frames
            current_anim_frames = args.window_size_anim 
            current_clean_sequence_np = clean_poses_r3j_full[:current_anim_frames]
            
            bone_offsets_np = data_npz['bone_offsets'].astype(np.float32)
            if bone_offsets_np.shape[0] != model.num_joints or bone_offsets_np.shape[1] != 3:
                print(f"  Info: Bone offsets shape is {bone_offsets_np.shape} for {npz_path.name}, "
                      f"expected ({model.num_joints}, 3). Skipping.")
                continue

            # 3. Generate Noisy Input
            print("  Generating noisy input sequence...")
            # Use a unique temp_dir name for each attempt to avoid conflicts if script is interrupted
            temp_dir = Path(f"temp_vis_data_{npz_path.stem}_{os.getpid()}_{len(all_clean_sequences)}")
            os.makedirs(temp_dir, exist_ok=True)
            temp_npz_path = temp_dir / "temp_clean_seq.npz"
            np.savez(temp_npz_path, poses_r3j=current_clean_sequence_np, bone_offsets=bone_offsets_np) 

            noisy_input_for_model_np = None
            try:
                current_noisy_dataset_args = base_noisy_dataset_args.copy()
                current_noisy_dataset_args['data_paths'] = [str(temp_npz_path)]
                # window_size in base_noisy_dataset_args is already args.window_size_anim
                
                noisy_data_generator = AMASSSubsetDataset(**current_noisy_dataset_args)
                if not noisy_data_generator: # Check if dataset is empty (e.g., file not found by loader)
                    raise ValueError(f"Noisy data generator is empty for {npz_path.name} (path: {temp_npz_path}).")

                noisy_input_for_model_torch, _clean_target_from_gen, _bone_offsets_from_gen = noisy_data_generator[0]
                noisy_input_for_model_np = noisy_input_for_model_torch.cpu().numpy()

            except Exception as e_noise:
                print(f"  Error generating noisy data for {npz_path.name}: {e_noise}. Skipping this file.")
                # temp_dir will be cleaned in the outer finally
                continue 
            
            if noisy_input_for_model_np is None: # Should be caught by exception, but as a safeguard
                 print(f"  Failed to generate noisy data for {npz_path.name} (returned None). Skipping this file.")
                 continue

            # Ensure noisy sequence has same T as clean sequence for this animation window
            if noisy_input_for_model_np.shape[0] != current_anim_frames:
                 print(f"  Warning: Noisy sequence length {noisy_input_for_model_np.shape[0]} for {npz_path.name} "
                       f"does not match anim frames {current_anim_frames}. Using slice.")
                 noisy_input_for_model_np = noisy_input_for_model_np[:current_anim_frames]
                 if noisy_input_for_model_np.shape[0] != current_anim_frames: 
                     print(f"  Error: Noisy sequence for {npz_path.name} too short ({noisy_input_for_model_np.shape[0]} frames) "
                           f"after adjustment. Required {current_anim_frames}. Skipping this file.")
                     continue


            # 4. Perform Inference
            print("  Performing inference...")
            current_refined_sequence_np = refine_sequence_transformer(model, noisy_input_for_model_np.copy(), model_window_size, device)
            print(f"  Inference complete for {npz_path.name}. Refined sequence shape: {current_refined_sequence_np.shape}")
            
            if current_refined_sequence_np.shape[0] != current_anim_frames:
                print(f"  Warning: Refined sequence length {current_refined_sequence_np.shape[0]} for {npz_path.name} "
                      f"does not match anim frames {current_anim_frames}. Using slice.")
                current_refined_sequence_np = current_refined_sequence_np[:current_anim_frames]
                if current_refined_sequence_np.shape[0] != current_anim_frames: 
                     print(f"  Error: Refined sequence for {npz_path.name} too short ({current_refined_sequence_np.shape[0]} frames) "
                           f"after adjustment. Required {current_anim_frames}. Skipping this file.")
                     continue

            # 5. Prepare sequences for plotting (handle potential centering by dataset)
            noisy_input_to_plot = noisy_input_for_model_np.copy()
            if model_was_trained_on_centered_data:
                original_clean_root_for_noisy = current_clean_sequence_np[:, 0:1, :].copy()
                if noisy_input_to_plot.shape[0] == original_clean_root_for_noisy.shape[0]:
                     noisy_input_to_plot += original_clean_root_for_noisy
                else: # Should not happen if previous checks passed
                    print(f"  Warning: Shape mismatch for de-centering noisy data for {npz_path.name}. "
                          f"Noisy: {noisy_input_to_plot.shape[0]}, Clean root: {original_clean_root_for_noisy.shape[0]}. "
                          "Plotting noisy as is (potentially centered).")
            
            current_file_processed_successfully = True

        except Exception as e_outer: 
            print(f"  An unexpected error occurred while processing {npz_path.name}: {e_outer}. Skipping this file.")
            # traceback.print_exc() # Uncomment for detailed debugging
            continue 
        finally:
            if temp_dir and temp_dir.exists(): # Ensure temp_dir is defined and exists
                 shutil.rmtree(temp_dir, ignore_errors=True)


        if current_file_processed_successfully:
            all_clean_sequences.append(current_clean_sequence_np)
            all_noisy_sequences_for_plot.append(noisy_input_to_plot)
            all_refined_sequences.append(current_refined_sequence_np)
            all_sequence_source_names.append(npz_path.name) # Store just filename for plot label
            print(f"  Successfully processed and added {npz_path.name}. Total sequences: {len(all_clean_sequences)}.")
        # else: (Implicitly)
            # print(f"  Skipped {npz_path.name} due to an issue during processing.") # Covered by specific info/error messages

    # After the while loop:
    if len(all_clean_sequences) == 0:
        print("\nNo sequences could be successfully processed that met all criteria (length, data integrity, processing steps). Exiting.")
        return
    elif len(all_clean_sequences) < target_num_sequences:
        print(f"\nWarning: Successfully processed {len(all_clean_sequences)} sequences, "
              f"which is less than the requested {target_num_sequences}. "
              f"Attempted {files_attempted_idx}/{len(all_npz_files)} available files.")
    else:
        print(f"\nSuccessfully processed {len(all_clean_sequences)} sequences as requested.")


    # 6. Create Grid Animation
    print("\nCreating grid animation (this may take a while)...")
    anim = _create_grid_animation(
        all_clean_sequences,
        all_noisy_sequences_for_plot,
        all_refined_sequences,
        all_sequence_source_names,
        skeleton_type,
        suptitle_extra_info=f"Model: {Path(args.checkpoint_path).name}",
        fps=30, 
        center_around_root_in_plot=args.center_plot
    )

    output_path = Path(args.output_anim_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    file_ext = output_path.suffix.lower()
    print(f"Saving animation to {output_path} ...")
    try:
        if file_ext == ".gif":
            writer = PillowWriter(fps=30, bitrate=1800) 
        elif file_ext == ".mp4":
            # Check if ffmpeg is available for FFMpegWriter
            if shutil.which("ffmpeg") is None:
                print("Error: ffmpeg not found. Cannot save as .mp4. Please install ffmpeg and ensure it's in your PATH.")
                print("Animation not saved. You can try saving as .gif instead.")
                plt.close(anim._fig)
                return            
            writer = FFMpegWriter(fps=30, metadata={"artist": "ManifoldRefinementInference"})
        else:
            print(f"Unsupported save format: {file_ext}. Use .gif or .mp4.")
            plt.close(anim._fig)
            return
        anim.save(str(output_path), writer=writer)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        # import traceback
        # traceback.print_exc()
        if file_ext == ".mp4":
            print("If saving to MP4 and ffmpeg is installed, ensure it's the correct version or try reinstalling.")
            print("You might also need to install codecs like libx264: `conda install x264 ffmpeg` or `sudo apt-get install libx264-dev ffmpeg`")
    finally:
        if 'anim' in locals() and hasattr(anim, '_fig'): # Ensure anim and fig exist
            plt.close(anim._fig)


if __name__ == "__main__":
    DEFAULT_CHECKPOINT_PATH = r"checkpoints/test/model_epoch_003_valloss_0.0157_mpjpe_28.93_B_0.0055.pth" # REPLACE
    DEFAULT_INPUT_NPZ_ROOT_DIR = r"data/" # REPLACE 
    DEFAULT_OUTPUT_ANIM = r"visualized_output/grid_comparison_animation_long.mp4" 
    DEFAULT_NUM_SEQUENCES = 3 
    DEFAULT_WINDOW_SIZE_ANIM = 150 # Files shorter than this will be skipped

    if len(sys.argv) == 1: 
        print("No command line arguments given, using default/example arguments for visualization.")
        
        if not Path(DEFAULT_CHECKPOINT_PATH).exists():
            print(f"WARNING: Default checkpoint path '{DEFAULT_CHECKPOINT_PATH}' does not exist.")
        if not Path(DEFAULT_INPUT_NPZ_ROOT_DIR).exists():
            print(f"WARNING: Default input NPZ root directory '{DEFAULT_INPUT_NPZ_ROOT_DIR}' does not exist.")
        
        os.makedirs(Path(DEFAULT_OUTPUT_ANIM).parent, exist_ok=True)

        sys.argv.extend([
            '--checkpoint_path', DEFAULT_CHECKPOINT_PATH,
            '--input_npz_root_dir', DEFAULT_INPUT_NPZ_ROOT_DIR,
            '--num_sequences_to_visualize', str(DEFAULT_NUM_SEQUENCES),
            '--output_anim_path', DEFAULT_OUTPUT_ANIM,
            '--window_size_anim', str(DEFAULT_WINDOW_SIZE_ANIM), 
            '--gaussian_noise_std', '0.008', 
            '--temporal_noise_type', 'filtered',
            '--temporal_noise_scale', '0.015',
            '--outlier_prob', '0.005',
            # '--center_plot' 
        ])
        print(f"Running with example arguments: {sys.argv[1:]}")

    args_vis = parse_vis_args()
    main_visualize(args_vis)