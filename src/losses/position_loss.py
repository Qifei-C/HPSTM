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

# For compatibility with your original naming if needed elsewhere, though class is preferred
def pose_loss_mse(predicted_positions, target_positions):
    loss_fn = PositionMSELoss()
    return loss_fn(predicted_positions, target_positions)

if __name__ == '__main__':
    loss_fn = PositionMSELoss()
    pred_pos = torch.randn(2, 24, 3)
    true_pos = torch.randn(2, 24, 3)
    loss = loss_fn(pred_pos, true_pos)
    print("Position MSE Loss:", loss.item())

    loss_func_style = pose_loss_mse(pred_pos, true_pos)
    print("Position MSE Loss (func style):", loss_func_style.item())