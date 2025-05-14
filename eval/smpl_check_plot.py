import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    print(f"Please check the path. Current working directory is: {os.getcwd()}")
else:
    print(f"Using processed AMASS file at: {processed_npz_path}")

skeleton_type = 'smpl_24'      
window_size = 31               
center_around_root = True      
noise_std = 0.0                

dataset = None
first_frame_pose_np = None

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
        print("AMASSSubsetDataset initialized with processed file.")
    except Exception as e:
        print(f"Error initializing AMASSSubsetDataset: {e}")

    if dataset and len(dataset) > 0:
        print(f"Dataset loaded successfully with {len(dataset)} windows.")
        try:
            _noisy_window_torch, clean_window_torch, _bone_offsets_torch = dataset[0]
            print("Successfully got item [0] from dataset.")
            
            first_frame_pose_torch = clean_window_torch[0]
            first_frame_pose_np = first_frame_pose_torch.numpy()
            print(f"Shape of the extracted first frame pose: {first_frame_pose_np.shape}")

            if np.isnan(first_frame_pose_np).any() or np.isinf(first_frame_pose_np).any():
                print("WARNING: NaN or Inf values found in first_frame_pose_np!")
            else:
                print("Data for plotting appears clean (no NaN/Inf).")

        except Exception as e:
            print(f"Error getting item from dataset or processing it: {e}")
            first_frame_pose_np = None 
    else:
        print("Dataset could not be loaded from processed file or is empty.")
else:
    print(f"Skipping dataset loading as file was not found: {processed_npz_path}")


if first_frame_pose_np is not None:
    print("Attempting to visualize frame...")
    try:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        print("1. Figure and 3D subplot created.")

        ax.scatter(first_frame_pose_np[:, 0],
                   first_frame_pose_np[:, 1],
                   first_frame_pose_np[:, 2],
                   c='deepskyblue', marker='o', s=60, edgecolors='black', linewidth=0.5)
        print("2. Joints scattered.")

        skeleton_parents = get_skeleton_parents(skeleton_type)
        for i, parent_idx in enumerate(skeleton_parents):
            if parent_idx != -1: 
                ax.plot([first_frame_pose_np[i, 0], first_frame_pose_np[parent_idx, 0]],
                        [first_frame_pose_np[i, 1], first_frame_pose_np[parent_idx, 1]],
                        [first_frame_pose_np[i, 2], first_frame_pose_np[parent_idx, 2]],
                        'r-', linewidth=2.5)
        print("3. Bones plotted.")

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title(f'Processed Frame - {skeleton_type} ({first_frame_pose_np.shape[0]} joints)')
        print("4. Labels and title set.")
        
        x_min, x_max = first_frame_pose_np[:,0].min(), first_frame_pose_np[:,0].max()
        y_min, y_max = first_frame_pose_np[:,1].min(), first_frame_pose_np[:,1].max()
        z_min, z_max = first_frame_pose_np[:,2].min(), first_frame_pose_np[:,2].max()
        
        margin = 0.2 # Add some margin
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)

        max_dim_range = max(x_max-x_min, y_max-y_min, z_max-z_min)
        ax.set_box_aspect((max_dim_range, max_dim_range, max_dim_range)) 
        print("5. Axes limits and aspect set.")

        ax.view_init(elev=20., azim=-45) 
        print("6. View initialized.")
        
        plt.show()
        print("7. Plot displayed (or attempted).")
    except Exception as e:
        print(f"Error during visualization: {e}")
else:
    print("No frame to visualize (first_frame_pose_np is None).")