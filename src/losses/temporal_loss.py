# src/losses/temporal_loss.py
import torch
import torch.nn as nn

def compute_velocity(poses_r3j):
    """
    Computes velocity as the difference between consecutive frames.
    Args:
        poses_r3j (torch.Tensor): Pose sequence of shape (batch_size, seq_len, num_joints, 3)
    Returns:
        torch.Tensor: Velocity sequence of shape (batch_size, seq_len-1, num_joints, 3)
    """
    if poses_r3j.ndim != 4 or poses_r3j.shape[-1] != 3:
        raise ValueError("Input poses_r3j must have shape (batch, seq_len, num_joints, 3)")
    if poses_r3j.shape[1] < 2:
        return torch.empty(poses_r3j.shape[0], 0, poses_r3j.shape[2], poses_r3j.shape[3], device=poses_r3j.device, dtype=poses_r3j.dtype)
        
    velocity = poses_r3j[:, 1:] - poses_r3j[:, :-1]
    return velocity

def compute_acceleration(velocity_or_poses):
    """
    Computes acceleration as the difference between consecutive velocities.
    Can take either a velocity sequence or a pose sequence as input.
    Args:
        velocity_or_poses (torch.Tensor): 
            If velocity: (batch_size, seq_len-1, num_joints, 3)
            If poses: (batch_size, seq_len, num_joints, 3)
    Returns:
        torch.Tensor: Acceleration sequence of shape (batch_size, seq_len-2, num_joints, 3)
    """
    if velocity_or_poses.ndim != 4 or velocity_or_poses.shape[-1] != 3:
        raise ValueError("Input must have shape (batch, seq_len_dim, num_joints, 3)")

    if velocity_or_poses.shape[1] < 2:
         return torch.empty(velocity_or_poses.shape[0], 0, velocity_or_poses.shape[2], velocity_or_poses.shape[3], device=velocity_or_poses.device, dtype=velocity_or_poses.dtype)

    velocity = velocity_or_poses
    if velocity.shape[1] < 2:
        return torch.empty(velocity.shape[0], 0, velocity.shape[2], velocity.shape[3], device=velocity.device, dtype=velocity.dtype)

    acceleration = velocity[:, 1:] - velocity[:, :-1]
    return acceleration


class VelocityLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, pred_poses, target_poses):
        pred_vel = compute_velocity(pred_poses)
        target_vel = compute_velocity(target_poses)
        if pred_vel.shape[1] == 0 or target_vel.shape[1] == 0:
            return torch.tensor(0.0, device=pred_poses.device, dtype=pred_poses.dtype)
        return self.criterion(pred_vel, target_vel)

class AccelerationLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, pred_poses, target_poses):
        pred_vel = compute_velocity(pred_poses)
        target_vel = compute_velocity(target_poses)
        
        pred_accel = compute_acceleration(pred_vel)
        target_accel = compute_acceleration(target_vel)

        if pred_accel.shape[1] == 0 or target_accel.shape[1] == 0:
             return torch.tensor(0.0, device=pred_poses.device, dtype=pred_poses.dtype)
        return self.criterion(pred_accel, target_accel)


if __name__ == '__main__':
    dummy_poses_b_s_j_3 = torch.randn(2, 10, 24, 3) # Batch, SeqLen, Joints, Dims
    
    # Test velocity
    vel = compute_velocity(dummy_poses_b_s_j_3)
    print("Velocity shape:", vel.shape) # Expected (2, 9, 24, 3)

    # Test acceleration
    accel_from_vel = compute_acceleration(vel)
    print("Acceleration from velocity shape:", accel_from_vel.shape) # Expected (2, 8, 24, 3)

    # Test acceleration
    accel_from_poses = compute_acceleration(compute_velocity(dummy_poses_b_s_j_3))
    print("Acceleration from poses shape:", accel_from_poses.shape) # Expected (2, 8, 24, 3)

    # Test loss classes
    vel_loss_fn = VelocityLoss(loss_type='l1')
    accel_loss_fn = AccelerationLoss(loss_type='l1')

    dummy_pred_poses = torch.randn(2, 10, 24, 3)
    dummy_target_poses = torch.randn(2, 10, 24, 3)

    loss_v = vel_loss_fn(dummy_pred_poses, dummy_target_poses)
    loss_a = accel_loss_fn(dummy_pred_poses, dummy_target_poses)
    print("Velocity loss:", loss_v.item())
    print("Acceleration loss:", loss_a.item())

    # Test with short sequences
    short_poses = torch.randn(2,3,24,3)
    loss_v_short = vel_loss_fn(short_poses, short_poses)
    loss_a_short = accel_loss_fn(short_poses, short_poses)
    print("Short Velocity loss:", loss_v_short.item())
    print("Short Acceleration loss:", loss_a_short.item())

    very_short_poses = torch.randn(2,1,24,3)
    loss_v_vshort = vel_loss_fn(very_short_poses, very_short_poses)
    loss_a_vshort = accel_loss_fn(very_short_poses, very_short_poses)
    print("Very Short Velocity loss:", loss_v_vshort.item())
    print("Very Short Acceleration loss:", loss_a_vshort.item())