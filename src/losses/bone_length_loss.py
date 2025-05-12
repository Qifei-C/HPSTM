# src/losses/bone_length_loss.py
import torch
import torch.nn as nn
import numpy as np

class BoneLengthMSELoss(nn.Module):
    def __init__(self, parents_list=None): # parents_list 变为可选
        """
        Calculates MSE loss for bone lengths.
        Can either compute true bone lengths from target 3D positions or
        compare against directly provided target bone lengths.

        Args:
            parents_list (list or np.array, optional): List of parent joint indices.
                            Required if calculating bone lengths from positions.
        """
        super(BoneLengthMSELoss, self).__init__()
        if parents_list is not None:
            self.parents = np.array(parents_list)
            # Ensure root parent is -1 and other indices are valid for array indexing later
            if not np.all(np.isin(self.parents[self.parents != -1], np.arange(len(self.parents)))):
                raise ValueError("Invalid parent indices found in parents_list.")
        else:
            self.parents = None
        self.mse_loss = nn.MSELoss()

    def _calculate_bone_lengths_from_positions(self, positions_3d):
        """
        Calculates bone lengths from 3D joint positions.
        Args:
            positions_3d (torch.Tensor): Shape (B, J, 3) or (B, S, J, 3)
        Returns:
            torch.Tensor: Shape (B, J) or (B, S, J).
        """
        if self.parents is None:
            raise ValueError("parents_list must be provided to BoneLengthMSELoss init "
                             "if calculating lengths from 3D positions.")

        if positions_3d.ndim == 3:  # (B, J, 3)
            batch_size, num_joints, _ = positions_3d.shape
            is_sequential = False
        elif positions_3d.ndim == 4:  # (B, S, J, 3)
            batch_size, seq_len, num_joints, _ = positions_3d.shape
            is_sequential = True
        else:
            raise ValueError(f"positions_3d has unsupported ndim: {positions_3d.ndim}. Expected 3 or 4.")

        if len(self.parents) != num_joints:
            raise ValueError(f"Length of parents_list ({len(self.parents)}) does not match "
                             f"num_joints ({num_joints}) in positions_3d.")

        if is_sequential:
            bone_lengths = torch.zeros(batch_size, seq_len, num_joints,
                                       device=positions_3d.device, dtype=positions_3d.dtype)
        else:
            bone_lengths = torch.zeros(batch_size, num_joints,
                                       device=positions_3d.device, dtype=positions_3d.dtype)

        for j_idx in range(num_joints):
            parent_idx = self.parents[j_idx]
            if parent_idx != -1:
                if is_sequential:
                    bone_vec = positions_3d[:, :, j_idx, :] - positions_3d[:, :, parent_idx, :]
                else:
                    bone_vec = positions_3d[:, j_idx, :] - positions_3d[:, parent_idx, :]
                bone_lengths_for_joint = torch.norm(bone_vec, dim=-1)

                if is_sequential:
                    bone_lengths[:, :, j_idx] = bone_lengths_for_joint
                else:
                    bone_lengths[:, j_idx] = bone_lengths_for_joint
            else:
                # Root joint has no parent, its "bone length" is 0
                if is_sequential:
                    bone_lengths[:, :, j_idx].fill_(0.0)
                else:
                    bone_lengths[:, j_idx].fill_(0.0)
        return bone_lengths

    def forward(self, predicted_bone_lengths, target_input, target_is_canonical_lengths=False):
        """
        Args:
            predicted_bone_lengths (torch.Tensor): Predicted bone lengths.
                                                   Shape (B, J) or (B, S, J).
            target_input (torch.Tensor):
                If target_is_canonical_lengths=False: Ground truth 3D joint positions.
                    Shape (B, J, 3) or (B, S, J, 3).
                If target_is_canonical_lengths=True: Ground truth canonical (fixed) bone lengths.
                    Shape (B, J).
            target_is_canonical_lengths (bool): Specifies if target_input provides direct lengths
                                                or 3D positions to calculate lengths from.
        Returns:
            torch.Tensor: Scalar MSE loss.
        """
        pred_ndim = predicted_bone_lengths.ndim

        if target_is_canonical_lengths:
            target_canonical_lengths = target_input
            if target_canonical_lengths.ndim != 2: # Expected (B, J)
                raise ValueError(f"If target_is_canonical_lengths=True, target_input (canonical_lengths) "
                                 f"must be (B, J), got {target_canonical_lengths.shape}")

            # Ensure predicted_bone_lengths Batch and NumJoints dimensions match target_canonical_lengths
            if predicted_bone_lengths.shape[0] != target_canonical_lengths.shape[0] or \
               predicted_bone_lengths.shape[-1] != target_canonical_lengths.shape[1]:
                raise ValueError(f"Shape mismatch (Batch or NumJoints) between predicted {predicted_bone_lengths.shape} "
                                 f"and target_canonical_lengths {target_canonical_lengths.shape}")

            if pred_ndim == 3: # predicted is (B, S, J), target is (B, J)
                B, S, J = predicted_bone_lengths.shape
                target_final = target_canonical_lengths.unsqueeze(1).expand(B, S, J)
            elif pred_ndim == 2: # predicted is (B, J), target is (B, J)
                target_final = target_canonical_lengths
            else:
                raise ValueError(f"predicted_bone_lengths has unsupported ndim: {pred_ndim}. Expected 2 or 3.")

        else: # target_input is target_3d_positions
            target_3d_positions = target_input
            # _calculate_bone_lengths_from_positions will return (B,J) or (B,S,J)
            # matching the ndim of predicted_bone_lengths if inputs are consistent.
            target_calculated_lengths = self._calculate_bone_lengths_from_positions(target_3d_positions)

            if predicted_bone_lengths.shape != target_calculated_lengths.shape:
                # This might happen if e.g. predicted is (B,S,J) but target_3d_positions was (B,J,3)
                # Or if predicted is (B,J) but target_3d_positions was (B,S,J,3)
                # We need them to align for direct comparison if this path is taken.
                # For now, we assume if this path is taken, shapes should naturally align or it's a usage error.
                 raise ValueError(f"Shape mismatch between predicted_bone_lengths {predicted_bone_lengths.shape} "
                                 f"and calculated true_bone_lengths {target_calculated_lengths.shape} from positions. "
                                 "Ensure predicted lengths and target positions have compatible sequential dimensions.")
            target_final = target_calculated_lengths

        return self.mse_loss(predicted_bone_lengths, target_final)

# Optional: Keep the functional wrapper if used elsewhere, but update it.
# def bone_length_loss_mse(predicted_bone_lengths, target_input, parents_list, target_is_canonical_lengths=False):
# loss_fn = BoneLengthMSELoss(parents_list if not target_is_canonical_lengths else None)
# return loss_fn(predicted_bone_lengths, target_input, target_is_canonical_lengths)