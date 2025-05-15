# Manifold/scripts/visualize_inference_result.py
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection
import sys
from pathlib import Path

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
    
    
def plot_covariance_ellipsoid(ax, mean_pos, cov_matrix, n_std=1.0, color='blue', alpha=0.1, **kwargs):
    """
    Plots a 3D covariance ellipsoid.
    Args:
        ax: Matplotlib 3D axis.
        mean_pos (np.ndarray): (3,) array for the center of the ellipsoid.
        cov_matrix (np.ndarray): (3, 3) covariance matrix.
        n_std (float): Number of standard deviations for the ellipsoid size.
        color (str): Color of the ellipsoid.
        alpha (float): Alpha transparency.
    Returns:
        Matplotlib surface plot object (or None if cov_matrix is invalid).
    """
    try:
        # Ensure covariance matrix is symmetric and try to make it positive definite
        # by adding a small epsilon to the diagonal if needed for eigendecomposition.
        # cov_matrix = (cov_matrix + cov_matrix.T) / 2
        # min_eig = np.min(np.linalg.eigvalsh(cov_matrix))
        # if min_eig < 0:
        #     cov_matrix -= 1.01 * min_eig * np.eye(*cov_matrix.shape) # Make positive definite

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # For symmetric matrices

        # Check for non-positive eigenvalues which can cause issues with sqrt
        if np.any(eigenvalues <= 1e-9): # Use a small threshold
            # print(f"Warning: Covariance matrix for joint at {mean_pos} has non-positive eigenvalues {eigenvalues}. Skipping ellipsoid.")
            # You might see this if the predicted Cholesky factors are very small or zero.
            return None


    except np.linalg.LinAlgError:
        # print(f"Warning: Singular covariance matrix for joint at {mean_pos}. Skipping ellipsoid.")
        return None # Cannot plot if singular or eigendecomposition fails

    # Radii are n_std * sqrt(eigenvalues)
    radii = n_std * np.sqrt(eigenvalues)
    if np.any(np.isnan(radii)) or np.any(np.isinf(radii)):
        # print(f"Warning: NaN or Inf in radii for joint at {mean_pos}. Skipping ellipsoid.")
        return None

    # Generate points on a unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 50) # Reduced for performance
    v = np.linspace(0.0, np.pi, 25)  # Reduced for performance
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Transform points to ellipsoid
    ellipsoid_points = np.stack((x, y, z), axis=-1) # (len_u, len_v, 3)
    for i in range(len(ellipsoid_points)):
        for j in range(len(ellipsoid_points[i])):
            ellipsoid_points[i, j] = radii * ellipsoid_points[i, j] # Scale
            ellipsoid_points[i, j] = eigenvectors @ ellipsoid_points[i, j] # Rotate
            ellipsoid_points[i, j] += mean_pos # Translate

    return ax.plot_surface(ellipsoid_points[...,0], ellipsoid_points[...,1], ellipsoid_points[...,2],
                           rstride=2, cstride=2, color=color, alpha=alpha, linewidth=0, **kwargs)


def _create_dual_animation(seq1_np: np.ndarray, seq2_np: np.ndarray,
                           cholesky_L_seq1: np.ndarray | None, # 新增： noisy_input 的协方差 (通常为None)
                           cholesky_L_seq2: np.ndarray | None, # 新增： refined_output 的协方差
                           title1: str, title2: str,
                           skeleton_type: str,
                           suptitle_extra_info: str = "",
                           fps: int = 30,
                           center_around_root_in_plot: bool = True,
                           n_std_dev_ellipsoid: float = 1.0
                           ) -> FuncAnimation:
    if seq1_np.shape != seq2_np.shape or seq1_np.ndim != 3:
        raise ValueError(f"Sequence 1 ({seq1_np.shape}) and Sequence 2 ({seq2_np.shape}) must both have shape (T, J, 3)")

    T, J, _ = seq1_np.shape
    skeleton_parents = get_skeleton_parents(skeleton_type)

    fig, (ax_seq1, ax_seq2) = plt.subplots(
        2, 1, figsize=(8.5, 17), subplot_kw={"projection": "3d"}
    )
    fig.suptitle(f"Input vs. Output Animation {suptitle_extra_info}", fontsize=14)

    data_to_plot = []
    current_seq1_data = seq1_np.copy()
    current_seq2_data = seq2_np.copy()

    if center_around_root_in_plot:
        print("Centering both sequences around their respective root joints for plotting.")
        for t_idx in range(T):
            current_seq1_data[t_idx] -= current_seq1_data[t_idx, 0:1, :]
            current_seq2_data[t_idx] -= current_seq2_data[t_idx, 0:1, :]
    
    data_to_plot.extend([current_seq1_data, current_seq2_data])
    cholesky_L_list = [cholesky_L_seq1, cholesky_L_seq2]

    temp_concat_for_limits = np.concatenate(data_to_plot, axis=0)
    x_min_disp, x_max_disp = temp_concat_for_limits[..., 0].min(), temp_concat_for_limits[..., 0].max()
    y_min_disp, y_max_disp = temp_concat_for_limits[..., 1].min(), temp_concat_for_limits[..., 1].max()
    z_min_disp, z_max_disp = temp_concat_for_limits[..., 2].min(), temp_concat_for_limits[..., 2].max()

    mid_x, mid_y, mid_z = (x_min_disp + x_max_disp) / 2, (y_min_disp + y_max_disp) / 2, (z_min_disp + z_max_disp) / 2
    range_x_disp, range_y_disp, range_z_disp = x_max_disp - x_min_disp, y_max_disp - y_min_disp, z_max_disp - z_min_disp
    max_total_range = max(range_x_disp, range_y_disp, range_z_disp, 0.1)

    plot_margin_factor = 0.15
    half_span = max_total_range / 2 * (1 + plot_margin_factor)

    axes = (ax_seq1, ax_seq2)
    titles_str_list = (title1, title2)
    point_colors = ("red", "green") # Noisy (red), Refined (green)
    bone_colors = ("lightcoral", "lime")
    ellipsoid_colors = ('salmon', 'lightgreen')

    scatters_list: list[plt.Artist] = []
    bone_lines_collection: list[list[plt.Line2D]] = []
    title_texts_list: list[plt.Text] = []
    ellipsoid_plots_collection: list[list[plt.Artist | None]] = [[None]*J, [None]*J] # 存储每个关节的椭球对象

    for idx, ax in enumerate(axes):
        ax.set_xlim(mid_x - half_span, mid_x + half_span)
        ax.set_ylim(mid_y - half_span, mid_y + half_span)
        ax.set_zlim(mid_z - half_span, mid_z + half_span)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            ax.set_aspect("auto")

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        title_obj = ax.set_title(f"{titles_str_list[idx]} — Frame 0/{T - 1}")
        title_texts_list.append(title_obj)

        pose0 = data_to_plot[idx][0]
        scat = ax.scatter(pose0[:, 0], pose0[:, 1], pose0[:, 2],
                          c=point_colors[idx], s=50, marker="o",
                          edgecolors="black", linewidths=0.3, depthshade=True)
        scatters_list.append(scat)

        current_ax_bone_lines = []
        for j, parent in enumerate(skeleton_parents):
            if parent == -1: continue
            (line,) = ax.plot([pose0[j, 0], pose0[parent, 0]], [pose0[j, 1], pose0[parent, 1]], [pose0[j, 2], pose0[parent, 2]],
                                color=bone_colors[idx], linewidth=2)
            current_ax_bone_lines.append(line)
        bone_lines_collection.append(current_ax_bone_lines)
        
        current_cholesky_L_seq = cholesky_L_list[idx] # (T, J, 3, 3) or None
        if current_cholesky_L_seq is not None:
            cholesky_L_frame0 = current_cholesky_L_seq[0] # (J, 3, 3)
            for j_idx in range(J):
                L_joint = cholesky_L_frame0[j_idx] # (3, 3)
                cov_matrix_joint = L_joint @ L_joint.T # (3, 3)
                mean_pos_joint = pose0[j_idx] # (3,)
                ellipsoid_plots_collection[idx][j_idx] = plot_covariance_ellipsoid(
                    ax, mean_pos_joint, cov_matrix_joint, 
                    n_std=n_std_dev_ellipsoid, color=ellipsoid_colors[idx], alpha=0.2 # 低alpha
                )
                
        ax.view_init(elev=15.0, azim=-75)

    def _update(frame: int):
        artists: list[plt.Artist] = []
        for view_idx in range(2):
            current_pose = data_to_plot[view_idx][frame]
            scat = scatters_list[view_idx]
            lines_for_current_ax = bone_lines_collection[view_idx]
            title_obj = title_texts_list[view_idx]
            current_cholesky_L_data_for_seq = cholesky_L_list[view_idx] # (T, J, 3, 3) or None
            current_ellipsoid_plots_for_ax = ellipsoid_plots_collection[view_idx] # list of J artists or Nones

            scat._offsets3d = (current_pose[:, 0], current_pose[:, 1], current_pose[:, 2])
            artists.append(scat)
            line_i = 0
            for j_idx, parent_idx in enumerate(skeleton_parents):
                if parent_idx == -1: continue
                lines_for_current_ax[line_i].set_data([current_pose[j_idx, 0], current_pose[parent_idx, 0]], [current_pose[j_idx, 1], current_pose[parent_idx, 1]])
                lines_for_current_ax[line_i].set_3d_properties([current_pose[j_idx, 2], current_pose[parent_idx, 2]])
                artists.append(lines_for_current_ax[line_i]); line_i += 1
            title_obj.set_text(f"{titles_str_list[view_idx]} — Frame {frame}/{T - 1}"); artists.append(title_obj)
            
            if current_cholesky_L_data_for_seq is not None:
                cholesky_L_current_frame = current_cholesky_L_data_for_seq[frame] # (J, 3, 3)
                for j_idx in range(J):
                    # 移除旧的椭球 (如果存在)
                    if current_ellipsoid_plots_for_ax[j_idx] is not None and \
                       hasattr(current_ellipsoid_plots_for_ax[j_idx], 'remove'):
                        current_ellipsoid_plots_for_ax[j_idx].remove()
                    
                    L_joint = cholesky_L_current_frame[j_idx] # (3,3)
                    cov_matrix_joint = L_joint @ L_joint.T   # (3,3)
                    mean_pos_joint = current_pose[j_idx]     # (3,)
                    
                    new_ellipsoid = plot_covariance_ellipsoid(
                        axes[view_idx], mean_pos_joint, cov_matrix_joint,
                        n_std=n_std_dev_ellipsoid, color=ellipsoid_colors[view_idx], alpha=0.2
                    )
                    ellipsoid_plots_collection[view_idx][j_idx] = new_ellipsoid
                    if new_ellipsoid: # plot_covariance_ellipsoid 可能返回 None
                        artists.append(new_ellipsoid)
                        
        return artists

    interval_ms = max(20, int(1000 / fps))
    return FuncAnimation(fig, _update, frames=T, interval=interval_ms, blit=False)


def parse_vis_args():
    parser = argparse.ArgumentParser(description="Visualize Inference of Human Pose Smoothing Model")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file)")
    parser.add_argument('--input_npz_path', type=str, required=True,
                        help="Path to a processed AMASS .npz file (containing 'poses_r3j' and 'bone_offsets') to use as clean input.")
    parser.add_argument('--output_anim_path', type=str, default="visualized_inference.gif",
                        help="Path to save the output animation (.gif or .mp4)")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (cuda or cpu)")
    parser.add_argument('--window_size_anim', type=int, default=150,
                        help="Number of frames from the input NPZ to use for animation.")
    parser.add_argument('--center_plot', action='store_true',
                        help="Center poses around root joint for visualization display.")
    # Noise parameters for generating noisy input (defaults from your log)
    parser.add_argument('--gaussian_noise_std', type=float, default=0.000)
    parser.add_argument('--temporal_noise_type', type=str, default='none', choices=['none', 'filtered', 'persistent'])
    parser.add_argument('--temporal_noise_scale', type=float, default=0.00)
    parser.add_argument('--temporal_filter_window', type=int, default=7)
    parser.add_argument('--temporal_event_prob', type=float, default=0.00)
    parser.add_argument('--temporal_decay', type=float, default=0.9)
    parser.add_argument('--outlier_prob', type=float, default=0.000)
    parser.add_argument('--outlier_scale', type=float, default=0.25)
    parser.add_argument('--bonelen_noise_scale', type=float, default=0.00)
    parser.add_argument('--input_cholesky_L_path', type=str, default=None,
                        help="(Optional) Path to the Cholesky L factors (.npy file) for the refined sequence. "
                             "If not provided, ellipsoids for refined sequence won't be shown. "
                             "Shape: (num_frames, num_joints, 3, 3)")
    parser.add_argument('--n_std_dev_viz', type=float, default=1.0,
                        help="Number of standard deviations to visualize for covariance ellipsoids.")

    args = parser.parse_args()
    return args

def main_visualize(args):
    device = torch.device(args.device)

    # 1. Load Model
    model, model_constructor_args = load_model_from_checkpoint(args.checkpoint_path, device)
    model_window_size = model_constructor_args['window_size']
    skeleton_type = model_constructor_args.get('skeleton_type', 'smpl_24') # Get from checkpoint
    # center_around_root_model_was_trained_with = model_constructor_args.get('center_around_root_amass', True)
    # The infer.py script's refine_sequence_transformer handles centering/decentering based on this automatically
    model_was_trained_on_centered_data = model_constructor_args.get('center_around_root_amass', True) # 默认 True
    model_predicts_covariance = model_constructor_args.get('predict_covariance_transformer', True)
    print(f"Model was configured to predict covariance: {model_predicts_covariance}") # <--- 添加打印
    
    # 2. Load Clean Data and Bone Offsets from NPZ
    if not os.path.exists(args.input_npz_path):
        print(f"Error: Input NPZ file not found: {args.input_npz_path}")
        return

    data_npz = np.load(args.input_npz_path, allow_pickle=True)
    if 'poses_r3j' not in data_npz:
        print(f"Error: 'poses_r3j' not found in {args.input_npz_path}")
        return
    if 'bone_offsets' not in data_npz:
        print(f"Error: 'bone_offsets' not found in {args.input_npz_path}. Please ensure pre-processing added this key.")
        return

    clean_poses_r3j_full = data_npz['poses_r3j'].astype(np.float32)
    bone_offsets_np = data_npz['bone_offsets'].astype(np.float32)

    if clean_poses_r3j_full.shape[0] < args.window_size_anim:
        print(f"Warning: Requested animation window {args.window_size_anim} is longer than sequence {clean_poses_r3j_full.shape[0]}. Using full sequence.")
        anim_frames_count = clean_poses_r3j_full.shape[0]
    else:
        anim_frames_count = args.window_size_anim
    
    clean_sequence_for_anim_np = clean_poses_r3j_full[:anim_frames_count]

    if bone_offsets_np.shape[0] != model.num_joints or bone_offsets_np.shape[1] != 3:
        print(f"Error: Bone offsets shape is {bone_offsets_np.shape}, expected ({model.num_joints}, 3).")
        return

    # 3. Generate Noisy Input using AMASSSubsetDataset's noise logic
    # We'll make a temporary dataset with just this one clean sequence to apply noise
    print("Generating noisy input sequence...")
    # AMASSSubsetDataset expects a list of paths. We'll save the clean sequence to a temp file.
    temp_dir = "temp_vis_data"
    os.makedirs(temp_dir, exist_ok=True)
    temp_npz_path = os.path.join(temp_dir, "temp_clean_seq.npz")
    np.savez(temp_npz_path, poses_r3j=clean_sequence_for_anim_np, bone_offsets=bone_offsets_np)

    try:
        noisy_dataset_args = dict(
            data_paths=[temp_npz_path], # Path to the temporary file
            window_size=anim_frames_count, # Use the full animation length as window
            skeleton_type=skeleton_type,
            is_train=True, # To activate noise
            center_around_root=model_was_trained_on_centered_data, # Match how data was fed to model for centering consistency
            joint_selector_indices=None, # Assuming input NPZ already has correct 24 joints
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
        noisy_data_generator = AMASSSubsetDataset(**noisy_dataset_args)
        
        if not noisy_data_generator:
            raise ValueError("Noisy data generator is empty.")

        noisy_input_for_model_torch, _clean_target_from_gen, _bone_offsets_from_gen = noisy_data_generator[0]
        noisy_input_for_model_np = noisy_input_for_model_torch.cpu().numpy()
        
        # The AMASSDataset might have centered it if center_around_root_amass was True
        # The refine_sequence_transformer expects absolute coords if model wasn't trained on centered,
        # or centered if it was. We need to ensure this matches.
        # `refine_sequence_transformer` already handles de-centering.
        # The input to `refine_sequence_transformer` should be what the model expects.
        # If `center_around_root_amass` was True for training, then `noisy_input_for_model_np` from dataset IS centered.
        # If False, it's absolute. This should align with `refine_sequence_transformer`'s internal logic.

    except Exception as e:
        print(f"Error generating noisy data: {e}")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


    # 4. Perform Inference
    print("Performing inference...")
    # refine_sequence_transformer expects absolute coords if model's center_around_root_dataset arg was False.
    # Or centered, if it was True.
    # AMASSSubsetDataset provides noisy_input_for_model_np according to its center_around_root param.
    # This should align.
    refined_sequence_np, cholesky_L_output_np = refine_sequence_transformer(
        model, noisy_input_for_model_np, model_window_size, device,
        predict_covariance=model_predicts_covariance # 传递模型是否预测协方差的标志
    )
    print(f"Inference complete. Refined sequence shape: {refined_sequence_np.shape}")
    if cholesky_L_output_np is not None:
        print(f"Cholesky L factors shape from inference: {cholesky_L_output_np.shape}")
    
    loaded_cholesky_L_for_refined = None
    
    if args.input_cholesky_L_path and os.path.exists(args.input_cholesky_L_path):
        try:
            loaded_cholesky_L_for_refined = np.load(args.input_cholesky_L_path)
            print(f"Loaded external Cholesky L factors from: {args.input_cholesky_L_path}, shape: {loaded_cholesky_L_for_refined.shape}")
            # 确保帧数匹配动画长度
            if loaded_cholesky_L_for_refined.shape[0] > anim_frames_count:
                loaded_cholesky_L_for_refined = loaded_cholesky_L_for_refined[:anim_frames_count]
            elif loaded_cholesky_L_for_refined.shape[0] < anim_frames_count:
                 print(f"Warning: Loaded Cholesky L has fewer frames ({loaded_cholesky_L_for_refined.shape[0]}) than animation ({anim_frames_count}). Will be padded implicitly by animation (might error or look odd).")
                 # For simplicity, we let the animation logic handle shorter sequences if it can, or it might error.
                 # A robust way would be to pad it here.
        except Exception as e:
            print(f"Error loading Cholesky L factors from {args.input_cholesky_L_path}: {e}")
            loaded_cholesky_L_for_refined = None
    
    # 优先使用模型直接输出的协方差，如果模型预测了的话
    cholesky_L_for_refined_anim = cholesky_L_output_np if model_predicts_covariance and cholesky_L_output_np is not None else loaded_cholesky_L_for_refined
    
    if cholesky_L_for_refined_anim is not None:
         # 确保帧数对齐
        if cholesky_L_for_refined_anim.shape[0] != refined_sequence_np.shape[0]:
            print(f"Warning: Frame count mismatch between refined poses ({refined_sequence_np.shape[0]}) and "
                  f"Cholesky L factors ({cholesky_L_for_refined_anim.shape[0]}). "
                  f"Adjusting Cholesky L factors to {refined_sequence_np.shape[0]} frames.")
            # Simple truncation or padding (e.g., repeat last frame's cov)
            if cholesky_L_for_refined_anim.shape[0] > refined_sequence_np.shape[0]:
                cholesky_L_for_refined_anim = cholesky_L_for_refined_anim[:refined_sequence_np.shape[0]]
            else: # Pad if shorter
                padding_needed = refined_sequence_np.shape[0] - cholesky_L_for_refined_anim.shape[0]
                if padding_needed > 0:
                    last_chol_L_frame = cholesky_L_for_refined_anim[-1:, ...] # Keep all J,3,3 dims
                    padding_chol_L = np.repeat(last_chol_L_frame, padding_needed, axis=0)
                    cholesky_L_for_refined_anim = np.concatenate([cholesky_L_for_refined_anim, padding_chol_L], axis=0)
    
    # 5. Visualization
    # For visualization, we want all sequences in comparable coordinate frames.
    # refine_sequence_transformer should return absolute coordinates.
    # We need to ensure noisy_input_for_model_np is also in absolute for the plot if model was trained on centered data.
    # However, if `args.center_plot` is True, the animation function will center them.

    noisy_input_to_plot = noisy_input_for_model_np
    if model_constructor_args.get('center_around_root_amass', False):
        # If model was trained on centered data, noisy_input_for_model_np is centered.
        # Add back root from the *original clean* sequence for visualization of noisy input.
        # Or, keep it centered if args.center_plot is True and let animation handle.
        # For a direct "input vs output" plot, if refine_sequence_transformer un-centers,
        # then noisy_input_for_model_np should also be un-centered for plotting.
        # Let's assume refine_sequence_transformer gives absolute.
        # If noisy_input_for_model_np was centered by dataset, add original root for fair comparison.
        original_clean_root_for_noisy = clean_sequence_for_anim_np[:, 0:1, :].copy()
        noisy_input_to_plot = noisy_input_for_model_np + original_clean_root_for_noisy


    print("Creating animation (Noisy Input vs. Refined Output)...")
    anim = _create_dual_animation(
        noisy_input_to_plot, refined_sequence_np,
        None,
        cholesky_L_for_refined_anim,
        "Noisy Input", "Refined Output",
        skeleton_type,
        suptitle_extra_info=f"Model: {Path(args.checkpoint_path).name}",
        center_around_root_in_plot=args.center_plot,
        n_std_dev_ellipsoid=args.n_std_dev_viz
    )

    output_path = Path(args.output_anim_path).expanduser().resolve()
    file_ext = output_path.suffix.lower()
    print(f"Saving animation to {output_path} (this may take a while)...")
    try:
        if file_ext == ".gif":
            writer = PillowWriter(fps=30, bitrate=1800)
        elif file_ext == ".mp4":
            writer = FFMpegWriter(fps=30, metadata={"artist": "ManifoldRefinementInference"})
        else:
            print(f"Unsupported save format: {file_ext}. Use .gif or .mp4.")
            return
        anim.save(str(output_path), writer=writer)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        # import traceback
        # traceback.print_exc()
        print("Try: `conda install ffmpeg` or `sudo apt-get install ffmpeg` if saving to MP4 and ffmpeg is missing.")

    plt.close(anim._fig) # Close the figure after saving


if __name__ == "__main__":
    # --- SET YOUR PATHS HERE ---
    # Example paths, please replace with your actual paths
    # Note: Use raw strings (r"...") or forward slashes for paths to avoid issues with backslashes.
    
    # Path to your trained model checkpoint
    DEFAULT_CHECKPOINT_PATH = r"checkpoints/test/model_epoch_010_valloss_-0.0531_mpjpe_42.88_B_0.0068_C_-7.8237.pth"
    
    # Path to an input .npz file from your PROCESSED AMASS dataset
    # This file should contain 'poses_r3j' (clean poses) and 'bone_offsets'
    DEFAULT_INPUT_NPZ = r"D:/git_repo_tidy/ESE6500/FinalProj/Manifold/data/CMU/02/02_01_poses.npz" # Example
    
    # Desired output path for the animation
    DEFAULT_OUTPUT_ANIM = r"visualized_output/test_01_comparison_10.gif" # Example output in a new folder

    # Create dummy args for testing if not run from command line
    if len(sys.argv) == 1: # No command line arguments provided
         # Ensure output directory exists
        os.makedirs(os.path.dirname(DEFAULT_OUTPUT_ANIM), exist_ok=True)

        sys.argv.extend([
            '--checkpoint_path', DEFAULT_CHECKPOINT_PATH,
            '--input_npz_path', DEFAULT_INPUT_NPZ,
            '--output_anim_path', DEFAULT_OUTPUT_ANIM,
            '--window_size_anim', '150', # Number of frames for animation
            '--center_plot', # Uncomment to center plots around root
            # You can also override default noise parameters here if needed for visualization
            # e.g., '--gaussian_noise_std', '0.01'
            '--n_std_dev_viz', '0.5'
        ])
        print(f"Running with default/example arguments: {sys.argv[1:]}")

    args_vis = parse_vis_args()
    main_visualize(args_vis)