import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import os
import sys

module_path = os.path.abspath(os.path.join('.')) 
if module_path not in sys.path:
    sys.path.insert(0, module_path)
    print(f"Added '{module_path}' to sys.path")
else:
    print(f"'{module_path}' is already in sys.path")

try:
    from src.datasets.amass_dataset import AMASSSubsetDataset
    from src.kinematics.skeleton_utils import get_skeleton_parents, get_num_joints
    print("Custom modules imported successfully.")
except ImportError as e:
    print(f"ImportError: {e}")
    print(f"Please ensure your Jupyter Notebook can find the 'src' directory from '{os.getcwd()}'")
    raise

processed_npz_path = os.path.join("..","data", "processed", "CMU", "00", "00_01_poses.npz") 
if not os.path.exists(processed_npz_path):
    print(f"ERROR: Processed data file not found at '{processed_npz_path}'")
else:
    print(f"Using processed AMASS file at: {processed_npz_path}")

skeleton_type = 'smpl_24'
window_size = 1000  
center_around_root = True 
noise_std = 0.0
dataset = None
all_frames_np = None

if os.path.exists(processed_npz_path):
    try:
        dataset = AMASSSubsetDataset(
            data_paths=[processed_npz_path],
            window_size=window_size,
            skeleton_type=skeleton_type,
            noise_std=noise_std,
            is_train=False, 
            center_around_root=center_around_root,
            joint_selector_indices=None 
        )
        print("AMASSSubsetDataset initialized.")
    except Exception as e:
        print(f"Error initializing AMASSSubsetDataset: {e}")

    if dataset and len(dataset) > 0:
        print(f"Dataset loaded successfully with {len(dataset)} windows.")
        try:
            _noisy_window_torch, clean_window_torch, _bone_offsets_torch = dataset[0]
            print("Successfully got item [0] from dataset.")
            all_frames_np = clean_window_torch.numpy()
            print(f"Shape of the animation data (frames, joints, coords): {all_frames_np.shape}")
            if np.isnan(all_frames_np).any() or np.isinf(all_frames_np).any():
                print("WARNING: NaN or Inf values found in animation data! Animation might fail or look incorrect.")
            else:
                print("Animation data appears clean (no NaN/Inf).")
        except Exception as e:
            print(f"Error getting item from dataset or processing it: {e}")
    else:
        print("Dataset could not be loaded from processed file or is empty.")
else:
    print(f"Skipping dataset loading as file was not found: {processed_npz_path}")


if all_frames_np is not None:
    num_animation_frames = all_frames_np.shape[0]
    num_joints_to_plot = all_frames_np.shape[1]

    skeleton_parents = get_skeleton_parents(skeleton_type)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = all_frames_np[..., 0].min(), all_frames_np[..., 0].max()
    y_min, y_max = all_frames_np[..., 1].min(), all_frames_np[..., 1].max()
    z_min, z_max = all_frames_np[..., 2].min(), all_frames_np[..., 2].max()
    
    margin = 0.2 
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)
    
    max_dim_range = max(x_max-x_min, y_max-y_min, z_max-z_min)
    if max_dim_range == 0 : max_dim_range = 1
    ax.set_box_aspect((x_max-x_min if (x_max-x_min)>0 else 1, 
                       y_max-y_min if (y_max-y_min)>0 else 1, 
                       z_max-z_min if (z_max-z_min)>0 else 1))


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = ax.set_title(f'Frame 0 / {num_animation_frames-1}')


    initial_pose = all_frames_np[0]
    scatter_plot = ax.scatter(initial_pose[:, 0], initial_pose[:, 1], initial_pose[:, 2], 
                              c='deepskyblue', marker='o', s=60, edgecolors='black', linewidth=0.5, depthshade=True)
    
    bone_lines = []
    for i, parent_idx in enumerate(skeleton_parents):
        if parent_idx != -1:
            line, = ax.plot([initial_pose[i, 0], initial_pose[parent_idx, 0]],
                            [initial_pose[i, 1], initial_pose[parent_idx, 1]],
                            [initial_pose[i, 2], initial_pose[parent_idx, 2]],
                            'r-', linewidth=2.5)
            bone_lines.append(line)

    def update_plot(frame_num, all_frames_data, scatter_plot, bone_lines, skeleton_parents, title_obj):
        current_pose = all_frames_data[frame_num]
        scatter_plot._offsets3d = (current_pose[:, 0], current_pose[:, 1], current_pose[:, 2])
        
        line_idx = 0
        for i, parent_idx in enumerate(skeleton_parents):
            if parent_idx != -1:
                bone_lines[line_idx].set_data([current_pose[i, 0], current_pose[parent_idx, 0]],
                                              [current_pose[i, 1], current_pose[parent_idx, 1]])
                bone_lines[line_idx].set_3d_properties([current_pose[i, 2], current_pose[parent_idx, 2]])
                line_idx += 1
        
        title_obj.set_text(f'Frame {frame_num} / {num_animation_frames-1}')
        return [scatter_plot] + bone_lines + [title_obj]

    
    print("Creating animation... This might take a moment.")
    try:
        anim = FuncAnimation(fig, update_plot, frames=num_animation_frames,
                             fargs=(all_frames_np, scatter_plot, bone_lines, skeleton_parents, title),
                             interval=50, blit=False, repeat=True)

        html_output = HTML(anim.to_jshtml())
        print("Animation created. If it doesn't display below, ensure your Jupyter environment supports jshtml.")
        
        
    except Exception as e:
        print(f"Error during animation creation or HTML conversion: {e}")
        import traceback
        traceback.print_exc()
        html_output = None

    plt.close(fig)
    
else:
    html_output = "Data not loaded, cannot create animation."

html_output