# src/losses/covariance_loss.py
import torch
import torch.nn as nn
import math

class NegativeLogLikelihoodLoss(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihoodLoss, self).__init__()

    def forward(self, y_true, y_pred_mean, pred_cholesky_L):
        """
        Calculates the Negative Log-Likelihood for a multivariate Gaussian.
        Args:
            y_true (torch.Tensor): Ground truth 3D joint positions. Shape (B, S, J, 3)
            y_pred_mean (torch.Tensor): Predicted mean 3D joint positions. Shape (B, S, J, 3)
            pred_cholesky_L (torch.Tensor): Predicted Cholesky factor L of covariance matrices.
                                           Shape (B, S, J, 3, 3)
        Returns:
            torch.Tensor: Scalar NLL loss.
        """
        if not (y_true.shape == y_pred_mean.shape and 
                y_true.shape[:-1] == pred_cholesky_L.shape[:-2] and
                pred_cholesky_L.shape[-2:] == (3,3)):
            raise ValueError("Shape mismatch in NLLLoss inputs.")

        B, S, J, D = y_true.shape # D should be 3

        log_diag_L = torch.log(pred_cholesky_L.diagonal(dim1=-2, dim2=-1)) # (B, S, J, 3)
        log_det_sigma = 2 * torch.sum(log_diag_L, dim=-1) # (B, S, J)

        L_flat = pred_cholesky_L.reshape(-1, D, D) # (B*S*J, 3, 3)
        diff_flat = (y_true - y_pred_mean).reshape(-1, D, 1) # (B*S*J, 3, 1)
        
        x, _ = torch.triangular_solve(diff_flat, L_flat, upper=False, unitriangular=False) # x shape (B*S*J, 3, 1)
        
        mahalanobis_term_flat = torch.sum(x * x, dim=(1,2)) # sum over D and K (which is 1), result (B*S*J)
        mahalanobis_term = mahalanobis_term_flat.reshape(B, S, J) # (B, S, J)

        loss_per_joint_per_frame = 0.5 * (mahalanobis_term + log_det_sigma + D * torch.log(torch.tensor(2.0 * math.pi, device=y_true.device)))
        
        return torch.mean(loss_per_joint_per_frame)