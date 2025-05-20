import torch
import numpy as np
import os
import argparse
import sys
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.signal import savgol_filter

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added '{project_root}' to sys.path for module imports.")

# NEW model
from src.models.pose_refiner_transformer import ManifoldRefinementTransformer
from scripts.infer import load_model_from_checkpoint, refine_sequence_transformer
    
# OLD model
from src.models.pose_refiner_transformer_old import ManifoldRefinementTransformerOld
from scripts.infer_old import load_model_from_checkpoint_old, refine_sequence_transformer_old
    
from src.datasets.amass_dataset import AMASSSubsetDataset
from src.kinematics.skeleton_utils import get_skeleton_parents


# Model Paths
NEW_MODEL_CHECKPOINT_PATH = "checkpoints/test/model_epoch_030_valloss_0.0023_mpjpe_26.14_C_-10.4899.pth"
OLD_MODEL_CHECKPOINT_PATH = "checkpoints/test/model_epoch_030_valloss_0.0128_mpjpe_23.00_B_0.0059.pth"


# Metric Calculation Functions
def calculate_mpjpe(predicted_seq, target_seq):
    if predicted_seq.shape != target_seq.shape:
        raise ValueError(f"Shape mismatch: Predicted {predicted_seq.shape}, Target {target_seq.shape}")
    error = np.linalg.norm(predicted_seq - target_seq, axis=-1)
    return np.mean(error) * 1000

def calculate_pa_mpjpe(predicted_seq, target_seq):
    if predicted_seq.ndim != 3 or target_seq.ndim != 3 or predicted_seq.shape != target_seq.shape:
        raise ValueError("Inputs for PA-MPJPE must be (S, J, 3) and have matching shapes.")
    num_frames, num_joints, _ = predicted_seq.shape
    pred_aligned = np.zeros_like(predicted_seq)
    for i in range(num_frames):
        mtx1, mtx2 = target_seq[i], predicted_seq[i]
        centroid1, centroid2 = np.mean(mtx1, axis=0), np.mean(mtx2, axis=0)
        mtx1_centered, mtx2_centered = mtx1 - centroid1, mtx2 - centroid2
        H = mtx2_centered.T @ mtx1_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        pred_aligned[i] = mtx2_centered @ R + centroid1
    return calculate_mpjpe(pred_aligned, target_seq)

def calculate_root_relative_mpjpe(predicted_seq, target_seq, root_joint_idx=0):
    pred_root_rel = predicted_seq - predicted_seq[:, root_joint_idx:root_joint_idx+1, :]
    target_root_rel = target_seq - target_seq[:, root_joint_idx:root_joint_idx+1, :]
    return calculate_mpjpe(pred_root_rel, target_root_rel)

def calculate_velocity(poses_seq):
    if poses_seq.shape[0] < 2: return np.array([])
    return poses_seq[1:] - poses_seq[:-1]

def calculate_acceleration(vel_seq):
    if vel_seq.shape[0] < 2: return np.array([])
    return vel_seq[1:] - vel_seq[:-1]

def calculate_jerk(accel_seq):
    if accel_seq.shape[0] < 2: return np.array([])
    return accel_seq[1:] - accel_seq[:-1]

def calculate_mean_norm_of_derivative(derivative_seq):
    if derivative_seq.size == 0: return 0.0
    norms_per_joint_per_frame = np.linalg.norm(derivative_seq, axis=2)
    mean_norm_per_frame = np.mean(norms_per_joint_per_frame, axis=1)
    return np.mean(mean_norm_per_frame)

def calculate_bone_lengths(pose_seq, skeleton_parents):
    num_frames, num_joints, _ = pose_seq.shape
    bone_lengths_seq = np.zeros((num_frames, num_joints))
    for j_idx in range(num_joints):
        parent_idx = skeleton_parents[j_idx]
        if parent_idx != -1:
            bone_vectors = pose_seq[:, j_idx, :] - pose_seq[:, parent_idx, :]
            bone_lengths_seq[:, j_idx] = np.linalg.norm(bone_vectors, axis=1)
    return bone_lengths_seq

def calculate_bone_length_error_metrics(predicted_seq, gt_bone_lengths_canonical, skeleton_parents):
    pred_bone_lengths_seq = calculate_bone_lengths(predicted_seq, skeleton_parents)
    num_frames, num_joints = predicted_seq.shape[0], predicted_seq.shape[1]
    gt_bone_lengths_expanded = np.tile(gt_bone_lengths_canonical, (num_frames, 1))
    actual_bone_indices = [j for j in range(num_joints) if skeleton_parents[j] != -1]
    if not actual_bone_indices:
        return {'mean_abs_error_mm': 0.0, 'mean_std_dev_mm': 0.0}
    errors = np.abs(pred_bone_lengths_seq[:, actual_bone_indices] - gt_bone_lengths_expanded[:, actual_bone_indices])
    mean_abs_error = np.mean(errors) * 1000
    std_dev_per_bone_over_time = np.std(pred_bone_lengths_seq[:, actual_bone_indices], axis=0)
    mean_std_dev = np.mean(std_dev_per_bone_over_time) * 1000
    return {'mean_abs_error_mm': mean_abs_error, 'mean_std_dev_mm': mean_std_dev}

class SimpleParticleFilter:
    def __init__(self, num_particles, num_joints, process_noise_std_pos=0.005, 
                 process_noise_std_vel=0.002, measurement_noise_std=0.05, 
                 initial_pos_noise_std=0.005):
        self.num_particles = num_particles
        self.num_joints = num_joints
        self.pos_dim = num_joints * 3
        self.vel_dim = num_joints * 3
        self.state_dim = self.pos_dim + self.vel_dim
        self.process_noise_pos = process_noise_std_pos
        self.process_noise_vel = process_noise_std_vel
        self.measurement_noise_std = measurement_noise_std
        self.dt = 1/30.0 
        self.initial_pos_noise_std = initial_pos_noise_std
        self.particles = np.zeros((num_particles, self.state_dim))
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles(self, initial_measurement_r3j):
        initial_pos_flat = initial_measurement_r3j.flatten()
        self.particles[:, :self.pos_dim] = np.tile(initial_pos_flat, (self.num_particles, 1)) + \
                                           np.random.randn(self.num_particles, self.pos_dim) * self.initial_pos_noise_std
        self.particles[:, self.pos_dim:] = np.random.randn(self.num_particles, self.vel_dim) * 0.001

    def predict(self):
        self.particles[:, :self.pos_dim] += self.particles[:, self.pos_dim:] * self.dt
        process_noise_sample = np.zeros_like(self.particles)
        process_noise_sample[:, :self.pos_dim] = np.random.randn(self.num_particles, self.pos_dim) * self.process_noise_pos
        process_noise_sample[:, self.pos_dim:] = np.random.randn(self.num_particles, self.vel_dim) * self.process_noise_vel
        self.particles += process_noise_sample

    def update(self, measurement_r3j):
        measurement_flat = measurement_r3j.flatten()
        particle_positions_flat = self.particles[:, :self.pos_dim]
        errors_sq = np.sum((particle_positions_flat - measurement_flat)**2, axis=1)
        likelihood = np.exp(-0.5 * errors_sq / (self.measurement_noise_std**2))
        self.weights *= likelihood
        self.weights += 1e-300 
        if np.sum(self.weights) < 1e-100: # Adjusted threshold
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(self.num_particles, size=self.num_particles, replace=True, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def estimate(self):
        mean_state = np.average(self.particles, weights=self.weights, axis=0)
        return mean_state[:self.pos_dim].reshape(self.num_joints, 3), \
               mean_state[self.pos_dim:].reshape(self.num_joints, 3)

    def smooth_sequence(self, noisy_seq_r3j):
        num_frames = noisy_seq_r3j.shape[0]
        smoothed_seq_r3j = np.zeros_like(noisy_seq_r3j)
        if num_frames == 0: return smoothed_seq_r3j
        self.initialize_particles(noisy_seq_r3j[0])
        for t in range(num_frames):
            self.predict()
            self.update(noisy_seq_r3j[t])
            neff = 1.0 / np.sum(self.weights**2) if np.sum(self.weights**2) > 1e-100 else 0
            if neff < self.num_particles / 2.0:
                self.resample()
            smoothed_pos, _ = self.estimate()
            smoothed_seq_r3j[t] = smoothed_pos
        return smoothed_seq_r3j

def savgol_smooth_sequence(noisy_seq_r3j, window_length=15, polyorder=3):
    if noisy_seq_r3j.shape[0] < window_length:
        print(f"  SavGol: Sequence length {noisy_seq_r3j.shape[0]} < window_length {window_length}. Returning original.")
        return noisy_seq_r3j.copy()
    if window_length % 2 == 0: window_length +=1 
    if polyorder >= window_length: polyorder = max(1, window_length -1)
    num_frames, num_joints, num_coords = noisy_seq_r3j.shape
    smoothed_seq = np.zeros_like(noisy_seq_r3j)
    for j in range(num_joints):
        for c in range(num_coords):
            smoothed_seq[:, j, c] = savgol_filter(noisy_seq_r3j[:, j, c], window_length, polyorder, mode='mirror')
    return smoothed_seq

def parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluate pose smoothing models with fixed checkpoints.")
    parser.add_argument('--gt_npz_path', type=str, required=True,
                        help="Path to the input NPZ file containing ground truth 'poses_r3j' and 'bone_offsets'.")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--gaussian_noise_std', type=float, default=0.03) # Example, match your training
    parser.add_argument('--temporal_noise_type', type=str, default='filtered', choices=['none', 'filtered', 'persistent'])
    parser.add_argument('--temporal_noise_scale', type=float, default=0.015)
    parser.add_argument('--temporal_filter_window', type=int, default=7)
    parser.add_argument('--outlier_prob', type=float, default=0.005)
    parser.add_argument('--outlier_scale', type=float, default=0.25)
    parser.add_argument('--bonelen_noise_scale', type=float, default=0.01)
    parser.add_argument('--pf_particles', type=int, default=100)
    parser.add_argument('--pf_proc_noise_pos', type=float, default=0.005)
    parser.add_argument('--pf_proc_noise_vel', type=float, default=0.002)
    parser.add_argument('--pf_meas_noise', type=float, default=0.05) # Crucial
    parser.add_argument('--pf_init_noise_pos', type=float, default=0.005)
    parser.add_argument('--savgol_window', type=int, default=15)
    parser.add_argument('--savgol_poly', type=int, default=3)

    args = parser.parse_args()
    return args

def main_evaluation(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    current_project_root = Path(__file__).resolve().parent.parent 
    new_model_ckpt_rel_path = NEW_MODEL_CHECKPOINT_PATH.replace("./Manifold/", "")
    old_model_ckpt_rel_path = OLD_MODEL_CHECKPOINT_PATH.replace("./Manifold/", "")
    new_model_abs_path = current_project_root / new_model_ckpt_rel_path
    old_model_abs_path = current_project_root / old_model_ckpt_rel_path
    
    # NEW HPSTM
    print(f"Loading NEW HPSTM model from: {new_model_abs_path}")
    if not new_model_abs_path.exists():
        print(f"Error: NEW Model Checkpoint file not found at {new_model_abs_path}"); return
    hpstm_model_new, model_args_new = load_model_from_checkpoint(str(new_model_abs_path), device)
    hpstm_model_new.eval()
    mw_new = model_args_new['window_size']
    st_new = model_args_new.get('skeleton_type', 'smpl_24')
    center_new = model_args_new.get('center_around_root_amass', True)
    cov_new = model_args_new.get('predict_covariance_transformer', True)
    parents_new = get_skeleton_parents(st_new)

    # OLD HPSTM
    old_hpstm_smoothed_seq_np = None
    if old_model_abs_path.exists():
        print(f"Loading OLD HPSTM model from: {old_model_abs_path}")
        hpstm_model_old, model_args_old = load_model_from_checkpoint_old(str(old_model_abs_path), device)
        hpstm_model_old.eval()
        mw_old = model_args_old['window_size']
        st_old = model_args_old.get('skeleton_type', 'smpl_24')
        center_old = model_args_old.get('center_around_root_amass', True)
    else:
        print(f"Warning: OLD Model Checkpoint file not found at {old_model_abs_path}. Skipping old model eval.")


    # 3. Ground Truth
    print(f"Loading Ground Truth NPZ data from: {args.gt_npz_path}")
    if not os.path.exists(args.gt_npz_path):
        print(f"Error: GT NPZ file not found: {args.gt_npz_path}"); return
    data_gt_npz = np.load(args.gt_npz_path, allow_pickle=True)
    if 'poses_r3j' not in data_gt_npz or 'bone_offsets' not in data_gt_npz:
        print(f"Error: 'poses_r3j' or 'bone_offsets' not in {args.gt_npz_path}"); return
    gt_poses_r3j = data_gt_npz['poses_r3j'].astype(np.float32)
    gt_bone_offsets = data_gt_npz['bone_offsets'].astype(np.float32)
    gt_canonical_bone_lengths = np.linalg.norm(gt_bone_offsets, axis=1)
    
    if gt_poses_r3j.shape[0] == 0: print("Error: GT sequence has 0 frames."); return
    if gt_poses_r3j.shape[1] != hpstm_model_new.num_joints :
         print(f"Error: GT num_joints ({gt_poses_r3j.shape[1]}) != model num_joints ({hpstm_model_new.num_joints})")
         return

    # Generate Noisy Input using AMASSSubsetDataset
    print("Generating noisy input sequence...")
    temp_dir = Path("temp_eval_data_optimized")
    temp_dir.mkdir(exist_ok=True)
    temp_npz_path = temp_dir / "temp_gt_seq_for_noise.npz"
    np.savez(temp_npz_path, poses_r3j=gt_poses_r3j, bone_offsets=gt_bone_offsets)
    
    noisy_input_for_models_np = None 
    try:
        dataset_window_size = gt_poses_r3j.shape[0]
        noise_gen_args = dict(
            data_paths=[str(temp_npz_path)], window_size=dataset_window_size,
            skeleton_type=st_new, is_train=True, 
            center_around_root=center_new, 
            gaussian_noise_std=args.gaussian_noise_std,
            temporal_noise_type=args.temporal_noise_type,
            temporal_noise_scale=args.temporal_noise_scale,
            temporal_filter_window=args.temporal_filter_window,
            outlier_prob=args.outlier_prob,
            outlier_scale=args.outlier_scale,
            bonelen_noise_scale=args.bonelen_noise_scale
        )
        noisy_data_gen = AMASSSubsetDataset(**noise_gen_args)
        if not noisy_data_gen: raise ValueError("Noisy data generator is empty.")
        noisy_input_for_models_torch, _, _ = noisy_data_gen[0]
        noisy_input_for_models_np = noisy_input_for_models_torch.cpu().numpy()
    except Exception as e:
        print(f"Error generating noisy data: {e}"); import shutil; shutil.rmtree(temp_dir, ignore_errors=True); return
    finally:
        import shutil; shutil.rmtree(temp_dir, ignore_errors=True)

    noisy_input_absolute_np = noisy_input_for_models_np.copy()
    root_traj_gt_for_decenter = gt_poses_r3j[:, 0:1, :]
    if center_new: 
        if noisy_input_absolute_np.shape[0] == root_traj_gt_for_decenter.shape[0]:
            noisy_input_absolute_np += root_traj_gt_for_decenter
        else:
            print("Warning: Frame mismatch for de-centering noisy input. Using potentially centered noisy input for baselines.")


    # NEW HPSTM Model Inference
    print("Performing inference with NEW HPSTM model...")
    new_hpstm_smoothed_seq_np, _ = refine_sequence_transformer(
        hpstm_model_new, noisy_input_for_models_np,
        mw_new, device, predict_covariance=cov_new
    )

    # OLD HPSTM Model Inference
    if old_model_abs_path.exists():
        print("Performing inference with OLD HPSTM model...")
        input_for_old_model_abs = noisy_input_absolute_np.copy()
        root_pos_for_old_decentering = None
        if center_old:
            root_pos_for_old_decentering = input_for_old_model_abs[:,0:1,:].copy()
        
        old_hpstm_smoothed_seq_np = refine_sequence_transformer_old(
            hpstm_model_old, 
            input_for_old_model_abs,
            mw_old, 
            device,
            center_input_if_model_expects_it=center_old,
            root_positions_original_for_decentering=root_pos_for_old_decentering
        )
    
    # Baselines
    print("Running Particle Filter baseline...")
    pf_smoother = SimpleParticleFilter(num_particles=args.pf_particles, num_joints=gt_poses_r3j.shape[1],
                                       process_noise_std_pos=args.pf_proc_noise_pos,
                                       process_noise_std_vel=args.pf_proc_noise_vel,
                                       measurement_noise_std=args.pf_meas_noise,
                                       initial_pos_noise_std=args.pf_init_noise_pos)
    pf_smoothed_seq_np = pf_smoother.smooth_sequence(noisy_input_absolute_np)

    print("Running Savitzky-Golay Filter baseline...")
    savgol_smoothed_seq_np = savgol_smooth_sequence(noisy_input_absolute_np, args.savgol_window, args.savgol_poly)


    # Metrics Printer
    results = {}
    sequences_to_eval = {
        "Noisy_Input": noisy_input_absolute_np,
        "PF_Smoothed": pf_smoothed_seq_np,
        "SavGol_Smoothed": savgol_smoothed_seq_np,
    }
    if old_hpstm_smoothed_seq_np is not None:
        sequences_to_eval["Old_HPSTM_Smoothed"] = old_hpstm_smoothed_seq_np
    sequences_to_eval["New_HPSTM_Smoothed"] = new_hpstm_smoothed_seq_np
    
    current_skeleton_parents = parents_new 

    for name, pred_seq in sequences_to_eval.items():
        if pred_seq is None: 
            print(f"\n--- Metrics for: {name} --- SKIPPED (no data) ---")
            continue
        print(f"\n--- Metrics for: {name} ---")
        min_len = min(pred_seq.shape[0], gt_poses_r3j.shape[0])
        if min_len == 0 : print(f"  Skipping {name}, zero length."); continue
        current_pred_seq, current_gt_seq = pred_seq[:min_len], gt_poses_r3j[:min_len]

        mpjpe = calculate_mpjpe(current_pred_seq, current_gt_seq)
        pa_mpjpe = calculate_pa_mpjpe(current_pred_seq, current_gt_seq)
        rr_mpjpe = calculate_root_relative_mpjpe(current_pred_seq, current_gt_seq)
        print(f"  MPJPE: {mpjpe:.4f} mm")
        print(f"  PA-MPJPE: {pa_mpjpe:.4f} mm")
        print(f"  Root-Relative MPJPE: {rr_mpjpe:.4f} mm")
        results[name+'_MPJPE'] = mpjpe; results[name+'_PA_MPJPE'] = pa_mpjpe; results[name+'_RR_MPJPE'] = rr_mpjpe

        vel = calculate_velocity(current_pred_seq); accel = calculate_acceleration(vel); jerk = calculate_jerk(accel)
        mean_accel = calculate_mean_norm_of_derivative(accel)
        mean_jerk = calculate_mean_norm_of_derivative(jerk)
        print(f"  Mean Acceleration Norm: {mean_accel:.4f}")
        print(f"  Mean Jerk Norm: {mean_jerk:.4f}")
        results[name+'_MeanAccel'] = mean_accel; results[name+'_MeanJerk'] = mean_jerk
        
        bone_metrics = calculate_bone_length_error_metrics(current_pred_seq, gt_canonical_bone_lengths, current_skeleton_parents)
        print(f"  Bone Length MAE: {bone_metrics['mean_abs_error_mm']:.4f} mm")
        print(f"  Bone Length StdDev: {bone_metrics['mean_std_dev_mm']:.4f} mm")
        results[name+'_BoneMAE'] = bone_metrics['mean_abs_error_mm']; results[name+'_BoneStdDev'] = bone_metrics['mean_std_dev_mm']

    print("\n\n--- Summary of Results ---")
    eval_keys_ordered = ["Noisy_Input", "PF_Smoothed", "SavGol_Smoothed"]
    if "Old_HPSTM_Smoothed" in sequences_to_eval: eval_keys_ordered.append("Old_HPSTM_Smoothed")
    eval_keys_ordered.append("New_HPSTM_Smoothed")
    
    header = "| Metric                | " + " | ".join([name for name in eval_keys_ordered if sequences_to_eval.get(name) is not None]) + " |"
    max_name_len = max(len(name) for name in eval_keys_ordered if sequences_to_eval.get(name) is not None)
    separator = "|-----------------------|-" + "-|-".join(["-" * max(len(name),7) for name in eval_keys_ordered if sequences_to_eval.get(name) is not None]) + "-|" # Min width 7 for values
    
    print(header)
    print(separator)
    metric_keys_display = [
        ("MPJPE (mm)", "_MPJPE"), ("PA-MPJPE (mm)", "_PA_MPJPE"), ("RR-MPJPE (mm)", "_RR_MPJPE"),
        ("MeanAccel", "_MeanAccel"), ("MeanJerk", "_MeanJerk"),
        ("BoneMAE (mm)", "_BoneMAE"), ("BoneStdDev (mm)", "_BoneStdDev")
    ]
    for display_name, key_suffix in metric_keys_display:
        row = f"| {display_name:<21} |"
        for seq_name in eval_keys_ordered:
            if sequences_to_eval.get(seq_name) is not None:
                value = results.get(seq_name + key_suffix, float('nan'))
                row += f" {value:<{max(len(seq_name),7)}.4f} |"
        print(row)

if __name__ == "__main__":
    args_eval = parse_eval_args()
    main_evaluation(args_eval)