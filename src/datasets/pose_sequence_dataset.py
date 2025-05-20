# src/datasets/pose_sequence_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np


class PoseSequenceDataset(Dataset):
    def __init__(self, sequences_np, window_size, num_joints, noise_std=0.05, is_train=True):
        """
        Dataset for pose sequences, yielding sliding windows.
        Args:
            sequences_np (list of np.ndarray): List of numpy arrays, where each array
                                               represents a sequence of shape (T_seq, num_joints * 3).
            window_size (int): The size of the sliding window.
            num_joints (int): The number of joints in the pose.
            noise_std (float): Standard deviation for Gaussian noise added to inputs if is_train is True.
            is_train (bool): If True, applies noise augmentation.
        """
        self.window_size = window_size
        self.noise_std = noise_std
        self.is_train = is_train
        self.num_joints = num_joints # J
        self.joint_dim_flat = num_joints * 3 # J*3

        self.windows = []
        self.targets = []

        self._create_windows_and_targets(sequences_np)

    def _create_windows_and_targets(self, sequences_np):
        for seq_np in sequences_np: # (T_seq, J*3)
            if seq_np.shape[1] != self.joint_dim_flat:
                print(f"Warning: Sequence encountered with {seq_np.shape[1]} features, expected {self.joint_dim_flat}. Skipping.")
                continue

            T_seq = seq_np.shape[0]
            if T_seq < self.window_size:
                continue

            for i in range(T_seq - self.window_size + 1):
                window_data = seq_np[i : i + self.window_size, :] # (window_size, J*3)
                self.windows.append(window_data)
                
                center_frame_idx_in_original_seq = i + self.window_size // 2
                target_center_frame_original_shape = seq_np[center_frame_idx_in_original_seq, :] # (J*3)
                self.targets.append(target_center_frame_original_shape)
        
        if not self.windows:
            print("Warning: No windows were created. Check input sequences length and window size.")
        else:
            print(f"Created {len(self.windows)} windows for PoseSequenceDataset.")
    
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        clean_window = self.windows[idx] # (W, J*3)
        target_center_frame = self.targets[idx] # (J*3)

        noisy_window_input = clean_window.copy()
        if self.is_train and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=noisy_window_input.shape)
            noisy_window_input += noise.astype(noisy_window_input.dtype)
        
        return torch.from_numpy(noisy_window_input).float(), torch.from_numpy(target_center_frame).float()