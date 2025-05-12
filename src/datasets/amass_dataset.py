import torch
from torch.utils.data import Dataset # DataLoader removed as it's not used in this file
import numpy as np
import os
import collections # For deque in bone length perturbation
from scipy.ndimage import uniform_filter1d # For temporal noise

# Assuming skeleton_utils is correctly imported
from ..kinematics.skeleton_utils import get_num_joints, get_skeleton_parents

class AMASSSubsetDataset(Dataset):
    def __init__(self, data_paths, window_size, skeleton_type='smpl_24',
                 gaussian_noise_std=0.0, # Renamed from noise_std
                 is_train=True, center_around_root=True,
                 joint_selector_indices=None,
                 # Temporal Noise Parameters
                 temporal_noise_type='none', # 'none', 'filtered', 'persistent'
                 temporal_noise_scale=0.0,
                 temporal_filter_window=5,
                 temporal_event_prob=0.05,
                 temporal_decay=0.8,
                 # Outlier Noise Parameters
                 outlier_prob=0.0,
                 outlier_scale=0.0,
                 # Bone Length Noise Parameters
                 bonelen_noise_scale=0.0
                 ):
        """
        AMASS Dataset for loading motion capture data.
        Args:
            data_paths (list): List of paths to .npz files containing AMASS data.
                               Each npz file should have 'poses_r3j'.
            window_size (int): Number of frames per sequence window.
            skeleton_type (str): Defines TARGET number of joints.
            gaussian_noise_std (float): Std dev of Gaussian noise if is_train.
            is_train (bool): If True, applies all configured noises.
            center_around_root (bool): If True, normalizes poses by subtracting root joint.
            joint_selector_indices (list or np.array, optional): Indices to select target joints.
            temporal_noise_type (str): Type of temporal noise.
            temporal_noise_scale (float): Scale for temporal noise.
            temporal_filter_window (int): Window for filtered temporal noise.
            temporal_event_prob (float): Probability for persistent temporal noise events.
            temporal_decay (float): Decay for persistent temporal noise.
            outlier_prob (float): Probability of a joint being an outlier per frame.
            outlier_scale (float): Max deviation for outliers.
            bonelen_noise_scale (float): Max relative bone length perturbation.
        """
        self.data_paths = data_paths
        self.window_size = window_size
        self.skeleton_type = skeleton_type
        self.num_target_joints = get_num_joints(self.skeleton_type)
        
        # Store all noise parameters
        self.gaussian_noise_std = gaussian_noise_std
        self.is_train = is_train
        self.center_around_root = center_around_root

        self.temporal_noise_type = temporal_noise_type
        self.temporal_noise_scale = temporal_noise_scale
        self.temporal_filter_window = temporal_filter_window
        self.temporal_event_prob = temporal_event_prob
        self.temporal_decay = temporal_decay
        self.outlier_prob = outlier_prob
        self.outlier_scale = outlier_scale
        self.bonelen_noise_scale = bonelen_noise_scale

        # Get skeleton structure for bone length noise
        if self.bonelen_noise_scale > 0 and self.is_train:
            self.skeleton_parents = get_skeleton_parents(self.skeleton_type)
            self.children_map = self._get_children_map(self.skeleton_parents)
        else:
            self.skeleton_parents = None
            self.children_map = None

        if joint_selector_indices is not None:
            self.joint_selector_indices = np.array(joint_selector_indices, dtype=int)
            if len(self.joint_selector_indices) != self.num_target_joints:
                raise ValueError(f"Length of joint_selector_indices ({len(self.joint_selector_indices)}) "
                                 f"must match num_target_joints ({self.num_target_joints}).")
        else:
            self.joint_selector_indices = None

        self.sequences_r3j = []
        self.bone_offsets_list = [] # Retained, though not directly used in noise here
        self.windows = []

        self._load_data()
        self._create_windows()

    @staticmethod
    def _get_children_map(parents_array: np.ndarray) -> dict[int, list[int]]:
        """Computes a map from parent index to a list of child indices."""
        children = {i: [] for i in range(len(parents_array))}
        for i, p in enumerate(parents_array):
            if p != -1: # Root has parent -1
                if p < len(parents_array): # Ensure parent index is valid
                     children[p].append(i)
                else:
                    print(f"Warning: Invalid parent index {p} for child {i} in _get_children_map.")
        return children

    def _load_data(self):
        print(f"Loading AMASS data from {len(self.data_paths)} path(s)...")
        for path_entry in self.data_paths:
            # Adapt to how data_paths is structured. User's code implies data_paths is List[str]
            # If data_paths is List[Dict], then path = path_entry['poses_r3j_path'] (or similar)
            # Based on user's provided file, data_paths IS List[str]
            path = path_entry 
            if isinstance(path_entry, dict): # Handle if dicts are passed (like from viz script)
                path = path_entry.get('poses_r3j_path', path_entry.get('metadata_path', None))


            if path is None or not os.path.exists(path):
                print(f"Warning: Data path not found or invalid: {path_entry}")
                continue
            try:
                data = np.load(path, allow_pickle=True)
                poses_r3j_raw = data['poses_r3j']

                if self.joint_selector_indices is not None:
                    if poses_r3j_raw.shape[1] < np.max(self.joint_selector_indices) + 1:
                        print(f"Warning: Skipping {path}. poses_r3j_raw has {poses_r3j_raw.shape[1]} joints, "
                              f"but joint_selector_indices requires at least {np.max(self.joint_selector_indices) + 1} joints.")
                        continue
                    selected_poses_r3j = poses_r3j_raw[:, self.joint_selector_indices, :]
                else:
                    selected_poses_r3j = poses_r3j_raw

                if selected_poses_r3j.shape[1] != self.num_target_joints or selected_poses_r3j.shape[2] != 3:
                    print(f"Warning: Skipping {path}. Final selected pose shape is {selected_poses_r3j.shape}. "
                          f"Expected (N, {self.num_target_joints}, 3).")
                    continue

                # Bone offsets processing (simplified, as it's not used by new noise types directly)
                selected_bone_offsets = np.zeros((self.num_target_joints, 3), dtype=np.float32)
                if 'bone_offsets' in data:
                    bone_offsets_raw = data['bone_offsets']
                    if self.joint_selector_indices is not None:
                        if bone_offsets_raw.shape[0] >= np.max(self.joint_selector_indices) + 1:
                             selected_bone_offsets = bone_offsets_raw[self.joint_selector_indices, :]
                        # else: use zeros (already initialized)
                    elif bone_offsets_raw.shape[0] == self.num_target_joints :
                        selected_bone_offsets = bone_offsets_raw
                
                self.sequences_r3j.append(selected_poses_r3j.astype(np.float32))
                self.bone_offsets_list.append(selected_bone_offsets.astype(np.float32))

            except Exception as e:
                print(f"Warning: Could not load or process data from {path}: {e}")

        if not self.sequences_r3j:
            print("No sequences loaded. Check data paths and file contents.")
        else:
            print(f"Successfully loaded {len(self.sequences_r3j)} sequences.")


    def _apply_gaussian_noise(self, sequence: np.ndarray, noise_std: float) -> np.ndarray:
        noise = np.random.normal(0, noise_std, size=sequence.shape)
        return sequence + noise.astype(sequence.dtype)

    def _apply_temporal_filtered_noise(self, sequence: np.ndarray, scale: float, filter_window_size: int) -> np.ndarray:
        T, J, C = sequence.shape
        random_offsets = np.random.randn(T, J, C) * scale
        filtered_offsets = np.zeros_like(random_offsets)
        for j_idx in range(J):
            for c_idx in range(C):
                filtered_offsets[:, j_idx, c_idx] = uniform_filter1d(random_offsets[:, j_idx, c_idx], size=filter_window_size, mode='reflect')
        return sequence + filtered_offsets

    def _apply_temporal_persistent_noise(self, sequence: np.ndarray, scale: float, event_prob: float, decay: float) -> np.ndarray:
        T, J, C = sequence.shape
        output_sequence = sequence.copy() # Modify a copy
        current_perturbation = np.zeros((J, C))
        for t_idx in range(T):
            current_perturbation *= decay
            for j_idx in range(J):
                if np.random.rand() < event_prob:
                    current_perturbation[j_idx, :] += (np.random.randn(C) * scale)
            output_sequence[t_idx, :, :] += current_perturbation
        return output_sequence

    def _apply_outliers(self, sequence: np.ndarray, prob: float, scale: float) -> np.ndarray:
        T, J, C = sequence.shape
        output_sequence = sequence.copy()
        for t_idx in range(T):
            for j_idx in range(J):
                if np.random.rand() < prob:
                    offset_vector = (np.random.rand(C) - 0.5) * 2 * scale
                    output_sequence[t_idx, j_idx, :] += offset_vector
        return output_sequence
        
    def _apply_bone_length_perturbations(self, sequence: np.ndarray, parents: np.ndarray, children_map: dict, noise_scale: float) -> np.ndarray:
        perturbed_sequence = sequence.copy()
        T, J, C = sequence.shape

        processing_order = []
        q = collections.deque([i for i, p_idx in enumerate(parents) if p_idx == -1]) # Root(s)
        visited_order = set()
        while q:
            curr = q.popleft()
            if curr in visited_order: continue
            visited_order.add(curr)
            processing_order.append(curr)
            for child in children_map.get(curr, []):
                q.append(child)
        
        for t_idx in range(T):
            current_frame_original = sequence[t_idx].copy() # Base for original bone vectors
            current_frame_perturbed = perturbed_sequence[t_idx] # Apply changes here

            for joint_idx in processing_order: # Iterate in hierarchical order
                parent_idx = parents[joint_idx]
                if parent_idx == -1: continue

                parent_pos_perturbed = current_frame_perturbed[parent_idx, :]
                original_parent_pos = current_frame_original[parent_idx, :] # Original parent
                original_child_pos = current_frame_original[joint_idx, :]   # Original child

                original_bone_vector = original_child_pos - original_parent_pos
                original_length = np.linalg.norm(original_bone_vector)

                if original_length < 1e-6: continue
                
                bone_direction = original_bone_vector / original_length
                length_multiplier = 1.0 + (np.random.rand() - 0.5) * 2 * noise_scale
                new_length = original_length * length_multiplier
                
                current_frame_perturbed[joint_idx, :] = parent_pos_perturbed + bone_direction * new_length
        return perturbed_sequence


    def _create_windows(self):
        sequences_processed_count = 0
        windows_created_count = 0
        print_seq_interval = 50  # Print every 50 sequences processed
        print_window_interval = 10000  # Print every 10000 windows created
        total_sequences = len(self.sequences_r3j)
        print(f"Starting window creation for {total_sequences} sequences...")
        
        for seq_idx, seq_r3j in enumerate(self.sequences_r3j):
            T_seq = seq_r3j.shape[0]
            if seq_r3j.shape[1] != self.num_target_joints:
                print(f"Error: Seq {seq_idx} joints {seq_r3j.shape[1]} != target {self.num_target_joints}. Skipping.")
                continue

            bone_offsets_for_seq = self.bone_offsets_list[seq_idx]
            if T_seq < self.window_size: continue
            
            windows_in_current_seq = 0

            for i in range(T_seq - self.window_size + 1):
                clean_window_abs = seq_r3j[i : i + self.window_size].copy()
                
                clean_window_processed = clean_window_abs.copy()
                if self.center_around_root:
                    root_positions = clean_window_processed[:, 0:1, :].copy()
                    clean_window_processed -= root_positions
                
                # Start with the (potentially centered) clean window
                noisy_window = clean_window_processed.copy()

                if self.is_train:
                    # 1. Apply Gaussian Noise
                    if self.gaussian_noise_std > 0:
                        noisy_window = self._apply_gaussian_noise(noisy_window, self.gaussian_noise_std)

                    # 2. Apply Bone Length Perturbations
                    if self.bonelen_noise_scale > 0 and self.skeleton_parents is not None and self.children_map is not None:
                        noisy_window = self._apply_bone_length_perturbations(noisy_window, self.skeleton_parents, self.children_map, self.bonelen_noise_scale)
                    
                    # 3. Apply Temporal Noise
                    if self.temporal_noise_type == 'filtered' and self.temporal_noise_scale > 0:
                        noisy_window = self._apply_temporal_filtered_noise(noisy_window, self.temporal_noise_scale, self.temporal_filter_window)
                    elif self.temporal_noise_type == 'persistent' and self.temporal_noise_scale > 0:
                        noisy_window = self._apply_temporal_persistent_noise(noisy_window, self.temporal_noise_scale, self.temporal_event_prob, self.temporal_decay)

                    # 4. Apply Outliers
                    if self.outlier_prob > 0 and self.outlier_scale > 0:
                        noisy_window = self._apply_outliers(noisy_window, self.outlier_prob, self.outlier_scale)
                
                # `clean_window_processed` is the one that underwent centering (if any)
                # `noisy_window` has all the selected noises applied
                self.windows.append((noisy_window, clean_window_processed, bone_offsets_for_seq))
                
                # --- START: Added for progress monitoring ---
                windows_created_count += 1
                windows_in_current_seq += 1
                if windows_created_count % print_window_interval == 0:
                    print(f"  ... {windows_created_count} windows created so far (current seq: {seq_idx+1}/{total_sequences}).")
                # --- END: Added for progress monitoring ---

            # --- START: Added for progress monitoring ---
            sequences_processed_count +=1
            if sequences_processed_count % print_seq_interval == 0 or sequences_processed_count == total_sequences:
                 print(f"Processed {sequences_processed_count}/{total_sequences} sequences. Total windows from this seq: {windows_in_current_seq}. Total windows overall: {windows_created_count}")
            # --- END: Added for progress monitoring ---

        if not self.windows: print("No windows created.")
        else: print(f"Created {len(self.windows)} windows in total.")
        
        if not self.windows: print("No windows created.")
        else: print(f"Created {len(self.windows)} windows.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        noisy_window_np, clean_window_np, bone_offsets_np = self.windows[idx]
        
        noisy_window_torch = torch.from_numpy(noisy_window_np).float()
        clean_window_torch = torch.from_numpy(clean_window_np).float()
        bone_offsets_torch = torch.from_numpy(bone_offsets_np).float()
        
        return noisy_window_torch, clean_window_torch, bone_offsets_torch


if __name__ == '__main__':
    # --- Example Usage ---
    dummy_data_dir = "dummy_amass_data_noise_types"
    os.makedirs(dummy_data_dir, exist_ok=True)
    
    num_source_joints = 24 # Keep it simple, assume data is already target format
    num_target_joints = 24
    seq_len = 200
    
    dummy_poses = np.random.rand(seq_len, num_source_joints, 3).astype(np.float32) * 2 - 1 # Centered around 0
    dummy_bones = np.random.rand(num_source_joints, 3).astype(np.float32)
    np.savez(os.path.join(dummy_data_dir, "seq_24joints.npz"), poses_r3j=dummy_poses, bone_offsets=dummy_bones)

    data_paths_test = [os.path.join(dummy_data_dir, "seq_24joints.npz")]
    window_s_test = 64

    print("\n--- Testing Dataset with Multiple Noise Types ---")
    dataset_multi_noise = AMASSSubsetDataset(
        data_paths=data_paths_test,
        window_size=window_s_test,
        skeleton_type='smpl_24',
        is_train=True, # Enable all noises
        center_around_root=False, # See absolute effects
        joint_selector_indices=None,
        # Configure various noises
        gaussian_noise_std=0.01,
        temporal_noise_type='filtered',
        temporal_noise_scale=0.02,
        temporal_filter_window=5,
        outlier_prob=0.005, # Low probability for outliers
        outlier_scale=0.3,
        bonelen_noise_scale=0.05 # 5% bone length variation
    )
    
    if len(dataset_multi_noise) > 0:
        from torch.utils.data import DataLoader # Moved import here
        dataloader_multi = DataLoader(dataset_multi_noise, batch_size=2, shuffle=True)
        noisy_b, clean_b, bones_b = next(iter(dataloader_multi))
        
        print(f"Dataset target skeleton: {dataset_multi_noise.skeleton_type}, num_target_joints: {dataset_multi_noise.num_target_joints}")
        print("Noisy batch shape:", noisy_b.shape)
        print("Clean batch shape:", clean_b.shape)
        print("Bones batch shape:", bones_b.shape)
        
        # Check if noise was applied (they should not be identical)
        if torch.allclose(noisy_b, clean_b):
            print("WARNING: Noisy and Clean batches are identical! Noise might not have been applied correctly.")
        else:
            print("Noisy and Clean batches are different, noise was likely applied.")
        
        # Check for NaNs
        if torch.isnan(noisy_b).any():
            print("ERROR: NaNs found in noisy_b")
        if torch.isnan(clean_b).any():
            print("ERROR: NaNs found in clean_b")

    else:
        print("Dataset (multi-noise) is empty.")

    # Optional: Clean up dummy files
    # import shutil
    # shutil.rmtree(dummy_data_dir)