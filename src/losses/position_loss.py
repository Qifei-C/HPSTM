# src/losses/position_loss.py
import torch
import torch.nn as nn

class PositionMSELoss(nn.Module):
    def __init__(self):
        super(PositionMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_positions, target_positions):
        """
        Calculates MSE loss between predicted and target 3D joint positions.
        Args:
            predicted_positions (torch.Tensor): Shape (Batch, NumJoints, 3)
            target_positions (torch.Tensor): Shape (Batch, NumJoints, 3)
        Returns:
            torch.Tensor: Scalar MSE loss.
        """
        if predicted_positions.shape != target_positions.shape:
            raise ValueError(f"Shape mismatch: predicted_positions {predicted_positions.shape} "
                             f"vs target_positions {target_positions.shape}")
        return self.mse_loss(predicted_positions, target_positions)

def pose_loss_mse(predicted_positions, target_positions):
    loss_fn = PositionMSELoss()
    return loss_fn(predicted_positions, target_positions)