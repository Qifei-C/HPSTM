import torch
import numpy as np
import os
import argparse
import sys
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.signal import savgol_filter
import shutil

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added '{project_root}' to sys.path for module imports.")

from src.models.pose_refiner_transformer import ManifoldRefinementTransformer
from scripts.infer import load_model_from_checkpoint, refine_sequence_transformer
from src.models.pose_refiner_transformer_old import ManifoldRefinementTransformerOld
from scripts.infer_old import load_model_from_checkpoint_old, refine_sequence_transformer_old
from src.datasets.amass_dataset import AMASSSubsetDataset
from src.kinematics.skeleton_utils import get_skeleton_parents, get_num_joints


# Metric
def calculate_mpjpe(predicted_seq, target_seq):
    if predicted_seq.shape[0] == 0 or target_seq.shape[0] == 0: return float('nan')
    min_frames = min(predicted_seq.shape[0], target_seq.shape[0])
    predicted_seq, target_seq = predicted_seq[:min_frames], target_seq[:min_frames]
    if predicted_seq.shape != target_seq.shape:
        print(f"Warning: MPJPE shape mismatch after frame alignment. Pred: {predicted_seq.shape}, GT: {target_seq.shape}")
        return float('nan')
    error = np.linalg.norm(predicted_seq - target_seq, axis=-1)
    return np.mean(error) * 1000

def calculate_pa_mpjpe(predicted_seq, target_seq):
    if predicted_seq.shape[0] == 0 or target_seq.shape[0] == 0: return float('nan')
    min_frames = min(predicted_seq.shape[0], target_seq.shape[0])
    predicted_seq, target_seq = predicted_seq[:min_frames], target_seq[:min_frames]

    if predicted_seq.ndim != 3 or target_seq.ndim != 3 or predicted_seq.shape[1:] != target_seq.shape[1:]:
        print(f"Warning: PA-MPJPE shape error. Pred: {predicted_seq.shape}, GT: {target_seq.shape}")
        return float('nan')
    
    num_frames = predicted_seq.shape[0]
    pred_aligned = np.zeros_like(predicted_seq)
    for i in range(num_frames):
        mtx1, mtx2 = target_seq[i], predicted_seq[i] # target, predicted
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
    if predicted_seq.shape[0] == 0 or target_seq.shape[0] == 0: return float('nan')
    min_frames = min(predicted_seq.shape[0], target_seq.shape[0])
    predicted_seq, target_seq = predicted_seq[:min_frames], target_seq[:min_frames]
    if predicted_seq.size == 0: return float('nan')
    pred_root_rel = predicted_seq - predicted_seq[:, root_joint_idx:root_joint_idx+1, :]
    target_root_rel = target_seq - target_seq[:, root_joint_idx:root_joint_idx+1, :]
    return calculate_mpjpe(pred_root_rel, target_root_rel)

def calculate_velocity(poses_seq):
    if poses_seq.shape[0] < 2: return np.array([]).reshape(0, poses_seq.shape[1], poses_seq.shape[2])
    return poses_seq[1:] - poses_seq[:-1]

def calculate_acceleration(vel_seq):
    if vel_seq.shape[0] < 2: return np.array([]).reshape(0, vel_seq.shape[1], vel_seq.shape[2])
    return vel_seq[1:] - vel_seq[:-1]

def calculate_jerk(accel_seq):
    if accel_seq.shape[0] < 2: return np.array([]).reshape(0, accel_seq.shape[1], accel_seq.shape[2])
    return accel_seq[1:] - accel_seq[:-1]

def calculate_mean_norm_of_derivative(derivative_seq):
    if derivative_seq.size == 0: return float(0.0) # Return float for consistency
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

def calculate_bone_length_error_metrics(predicted_seq, gt_canonical_bone_lengths, skeleton_parents):
    if predicted_seq.shape[0] == 0: return {'mean_abs_error_mm': float('nan'), 'mean_std_dev_mm': float('nan')}
    pred_bone_lengths_seq = calculate_bone_lengths(predicted_seq, skeleton_parents)
    num_frames, num_joints = pred_bone_lengths_seq.shape
    gt_bone_lengths_expanded = np.tile(gt_canonical_bone_lengths, (num_frames, 1))
    actual_bone_indices = [j for j in range(num_joints) if skeleton_parents[j] != -1]
    if not actual_bone_indices:
        return {'mean_abs_error_mm': 0.0, 'mean_std_dev_mm': 0.0}
    pred_actual_bones = pred_bone_lengths_seq[:, actual_bone_indices]
    gt_actual_bones = gt_bone_lengths_expanded[:, actual_bone_indices]
    errors = np.abs(pred_actual_bones - gt_actual_bones)
    mean_abs_error = np.mean(errors) * 1000
    std_dev_per_bone_over_time = np.std(pred_actual_bones, axis=0)
    mean_std_dev = np.mean(std_dev_per_bone_over_time) * 1000
    return {'mean_abs_error_mm': mean_abs_error, 'mean_std_dev_mm': mean_std_dev}

# Configurations
GT_NPZ_PATH = ".\Trail\joints_drc_smooth.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = [
    {"path": "checkpoints/test/model_epoch_022_valloss_-0.0893_mpjpe_33.43_C_-10.6224.pth", "type": "new", "name": "HPSTM_New_Cov_Ep22"},
    {"path": "checkpoints/test/model_epoch_030_valloss_0.0023_mpjpe_26.14_C_-10.4899.pth", "type": "new", "name": "HPSTM_New_Cov_Ep30"},
    {"path": "checkpoints/test/model_epoch_030_valloss_0.0128_mpjpe_23.00_B_0.0059.pth", "type": "old", "name": "HPSTM_Old_NoCov_Ep30"},
    {"path": "checkpoints/test/model_epoch_023_valloss_0.0156_mpjpe_28.09_B_0.0056.pth", "type": "old", "name": "HPSTM_Old_NoCov_Ep23"}
]

NOISE_CONFIGS = [
    {
        "name": "NoNoise", 
        "params": {
            "gaussian_noise_std": 0.0, "temporal_noise_type": "none", 
            "temporal_noise_scale": 0.0, "temporal_filter_window": 7,
            "outlier_prob": 0.0, "outlier_scale": 0.0, 
            "bonelen_noise_scale": 0.0
        }
    },
    {
        "name": "ComplexNoise", 
        "params": {
            "gaussian_noise_std": 0.03, "temporal_noise_type": "filtered",
            "temporal_noise_scale": 0.03, "temporal_filter_window": 7,
            "bonelen_noise_scale": 0.08, "outlier_prob": 0.0025,
            "outlier_scale": 0.25
        }
    }
]
TEMP_EVAL_DIR = Path("temp_eval_data_ablation_study_v2")

def run_single_evaluation(model_config, noise_name, noise_config_params, gt_npz_path, device_str):
    device = torch.device(device_str)
    print(f"\nRunning Evaluation for: {model_config['name']} with {noise_name}")

    results_for_this_hpstm_run = {} 
    model_abs_path = project_root / model_config["path"]
    print(f"  Loading model from: {model_abs_path}")
    if not model_abs_path.exists():
        print(f"  Error: Model Checkpoint file not found at {model_abs_path}"); return None
    
    hpstm_model, model_args, mw, st, skeleton_parents_np, predict_covariance_flag, model_expects_centered_input = [None]*7
    try:
        if model_config["type"] == "new":
            hpstm_model, model_args = load_model_from_checkpoint(str(model_abs_path), device)
            predict_covariance_flag = model_args.get('predict_covariance_transformer', False)
        elif model_config["type"] == "old":
            hpstm_model, model_args = load_model_from_checkpoint_old(str(model_abs_path), device)
            predict_covariance_flag = False 
        else:
            print(f"  Error: Unknown model type: {model_config['type']}"); return None
        
        hpstm_model.eval()
        mw = model_args['window_size']
        st = model_args.get('skeleton_type', 'smpl_24')
        skeleton_parents_np = get_skeleton_parents(st)
        model_expects_centered_input = model_args.get('center_around_root_amass', model_args.get('center_around_root', True))
    except Exception as e:
        print(f"  Error during model loading for {model_config['name']}: {e}"); return None

    # Ground Truth
    print(f"  Loading Ground Truth NPZ data from: {gt_npz_path}")
    if not os.path.exists(gt_npz_path):
        print(f"  Error: GT NPZ file not found: {gt_npz_path}"); return None
    try:
        data_gt_npz = np.load(gt_npz_path, allow_pickle=True)
        gt_poses_r3j = data_gt_npz['poses_r3j'].astype(np.float32)
        gt_bone_offsets = data_gt_npz['bone_offsets'].astype(np.float32)
    except Exception as e:
        print(f"  Error loading data from GT NPZ '{gt_npz_path}': {e}"); return None
        
    if 'poses_r3j' not in data_gt_npz or 'bone_offsets' not in data_gt_npz:
        print(f"  Error: 'poses_r3j' or 'bone_offsets' not in {gt_npz_path}"); return None
    
    gt_canonical_bone_lengths = np.linalg.norm(gt_bone_offsets, axis=1)
    if gt_poses_r3j.shape[0] == 0: print("  Error: GT sequence has 0 frames."); return None
    
    num_joints_model = hpstm_model.num_joints
    if gt_poses_r3j.shape[1] != num_joints_model:
        print(f"  Error: GT num_joints ({gt_poses_r3j.shape[1]}) != model num_joints ({num_joints_model}) for skeleton '{st}'")
        return None

    # Noisy Input
    print(f"  Generating noisy input sequence with profile: {noise_name}")
    TEMP_EVAL_DIR.mkdir(exist_ok=True)
    temp_npz_path = TEMP_EVAL_DIR / f"temp_gt_{Path(gt_npz_path).stem}_noise_{noise_name}.npz" # More specific temp file
    np.savez(temp_npz_path, poses_r3j=gt_poses_r3j, bone_offsets=gt_bone_offsets)
    
    noisy_input_generated_np = None
    amass_dataset_center_around_root = True # Assume noise is applied to centered data by AMASSDataset
    try:
        dataset_window_size = gt_poses_r3j.shape[0] 
        noise_gen_args = dict(
            data_paths=[str(temp_npz_path)], window_size=dataset_window_size,
            skeleton_type=st, is_train=True, 
            center_around_root=amass_dataset_center_around_root, 
            **noise_config_params 
        )
        noisy_data_gen = AMASSSubsetDataset(**noise_gen_args)
        if not noisy_data_gen or len(noisy_data_gen) == 0: raise ValueError("Noisy data generator is empty.")
        noisy_input_generated_torch, _, _ = noisy_data_gen[0] 
        noisy_input_generated_np = noisy_input_generated_torch.cpu().numpy() 
    except Exception as e:
        print(f"  Error generating noisy data: {e}"); return None

    noisy_input_absolute_np = noisy_input_generated_np.copy()
    gt_root_trajectory = gt_poses_r3j[:, 0:1, :] 
    if amass_dataset_center_around_root:
        noisy_input_absolute_np += gt_root_trajectory
    
    # HPSTM Inference
    print(f"  Performing inference with {model_config['name']}...")
    hpstm_smoothed_seq_np = None
    try:
        if model_config["type"] == "new":
            hpstm_smoothed_seq_np, _ = refine_sequence_transformer(
                hpstm_model, noisy_input_absolute_np, 
                mw, device, predict_covariance=predict_covariance_flag
            )
        elif model_config["type"] == "old":
            root_for_old_decenter = noisy_input_absolute_np[:, 0:1, :].copy() if model_expects_centered_input else None
            hpstm_smoothed_seq_np = refine_sequence_transformer_old(
                hpstm_model, noisy_input_absolute_np, 
                mw, device,
                center_input_if_model_expects_it=model_expects_centered_input,
                root_positions_original_for_decentering=root_for_old_decenter
            )
    except Exception as e:
        print(f"  Error during HPSTM model inference for {model_config['name']}: {e}")

    # Metrics for HPSTM
    method_name_key = model_config["name"] # Use the unique model name as the key
    print(f"  Calculating metrics for: {method_name_key}")

    if hpstm_smoothed_seq_np is None: 
        print(f"    Skipping metrics for {method_name_key} (inference failed or produced no data).")
        metrics_to_calculate = ["_MPJPE", "_PA_MPJPE", "_RR_MPJPE", "_MeanAccel", "_MeanJerk", "_BoneMAE", "_BoneStdDev"]
        for suffix in metrics_to_calculate:
             results_for_this_hpstm_run[method_name_key + suffix] = float('nan')
        return results_for_this_hpstm_run # Return dict with NaNs

    min_len = min(hpstm_smoothed_seq_np.shape[0], gt_poses_r3j.shape[0])
    if min_len == 0 : 
        print(f"    Skipping metrics for {method_name_key}, zero length common sequence."); return None
            
    current_pred_seq, current_gt_seq = hpstm_smoothed_seq_np[:min_len], gt_poses_r3j[:min_len]

    results_for_this_hpstm_run[method_name_key + '_MPJPE'] = calculate_mpjpe(current_pred_seq, current_gt_seq)
    results_for_this_hpstm_run[method_name_key + '_PA_MPJPE'] = calculate_pa_mpjpe(current_pred_seq, current_gt_seq)
    results_for_this_hpstm_run[method_name_key + '_RR_MPJPE'] = calculate_root_relative_mpjpe(current_pred_seq, current_gt_seq)
    
    vel = calculate_velocity(current_pred_seq); accel = calculate_acceleration(vel); jerk = calculate_jerk(accel)
    results_for_this_hpstm_run[method_name_key + '_MeanAccel'] = calculate_mean_norm_of_derivative(accel)
    results_for_this_hpstm_run[method_name_key + '_MeanJerk'] = calculate_mean_norm_of_derivative(jerk)
    
    bone_metrics = calculate_bone_length_error_metrics(current_pred_seq, gt_canonical_bone_lengths, skeleton_parents_np)
    results_for_this_hpstm_run[method_name_key + '_BoneMAE'] = bone_metrics['mean_abs_error_mm']
    results_for_this_hpstm_run[method_name_key + '_BoneStdDev'] = bone_metrics['mean_std_dev_mm']
        
    return results_for_this_hpstm_run


def format_and_print_results(all_run_results_list):
    print("\n\n--- Ablation Study Summary (HPSTM Models Only) ---")
    
    metric_display_config = [
        ("MPJPE (mm)", "_MPJPE", 12, ".3f"), 
        ("PA-MPJPE (mm)", "_PA_MPJPE", 15, ".3f"),
        ("RR-MPJPE (mm)", "_RR_MPJPE", 15, ".3f"),
        ("MeanAccel", "_MeanAccel", 16, ".7f"), 
        ("MeanJerk", "_MeanJerk", 16, ".7f"),
        ("BoneMAE (mm)", "_BoneMAE", 14, ".3f"),
        ("BoneStdDev (mm)", "_BoneStdDev", 16, ".3f")
    ]

    header_model_col_width = 25
    header_noise_col_width = 15
    header_model_col = f"{'HPSTM Model':<{header_model_col_width}}"
    header_noise_col = f"{'Noise':<{header_noise_col_width}}"
    
    header_metric_cols_str_parts = []
    for name, _, width, _ in metric_display_config:
        header_metric_cols_str_parts.append(f"{name:>{width}}")
    header_metrics_str = " | ".join(header_metric_cols_str_parts)
    
    print(f"{header_model_col} | {header_noise_col} | {header_metrics_str}")
    
    separator_line_parts = [f"{'-'*header_model_col_width}", f"{'-'*header_noise_col_width}"]
    for _, _, width, _ in metric_display_config:
        separator_line_parts.append(f"{'-'*width}")
    num_cols_for_sep = 2 + len(metric_display_config)
    separator_str = ("-" * 3).join(separator_line_parts)
    print(separator_str)


    for run_data in all_run_results_list:
        model_name_display = run_data["model_name"]
        noise_name_display = run_data["noise_name"]
        metrics = run_data["metrics_dict"]

        row_str_parts = [
            f"{model_name_display:<{header_model_col_width}}",
            f"{noise_name_display:<{header_noise_col_width}}"
        ]

        if metrics is None:
            failed_metrics_placeholders = " | ".join([f"{'FAILED RUN':>{width}}" for _, _, width, _ in metric_display_config])
            print(f"{model_name_display:<{header_model_col_width}} | {noise_name_display:<{header_noise_col_width}} | {failed_metrics_placeholders}")
            continue
        
        method_prefix_in_key = model_name_display

        for _, metric_suffix, width, fmt_spec in metric_display_config:
            full_metric_key = method_prefix_in_key + metric_suffix
            value = metrics.get(full_metric_key)
            
            if isinstance(value, float) and not np.isnan(value):
                format_string = "{:>" + str(width) + fmt_spec + "}"
                row_str_parts.append(format_string.format(value))
            else: 
                row_str_parts.append(f"{'N/A':>{width}}")
        
        print(" | ".join(row_str_parts))
    print(separator_str)


def ablation_main():
    all_overall_results = []
    if not Path(GT_NPZ_PATH).exists():
        print(f"FATAL ERROR: Ground truth file not found at '{GT_NPZ_PATH}'.")
        print("Please set the 'GT_NPZ_PATH' variable at the top of the script to a valid .npz file")
        print("containing 'poses_r3j' and 'bone_offsets'.")
        return

    if TEMP_EVAL_DIR.exists():
        print(f"Cleaning up existing temporary directory: {TEMP_EVAL_DIR}")
        shutil.rmtree(TEMP_EVAL_DIR)
    TEMP_EVAL_DIR.mkdir(exist_ok=True)


    for model_c in MODEL_CONFIGS:
        for noise_c in NOISE_CONFIGS:
            print(f"\n{'='*90}")
            print(f"Processing: Model '{model_c['name']}' with Noise '{noise_c['name']}'")
            print(f"{'='*90}")
            
            current_run_metrics_dict = run_single_evaluation(
                model_config=model_c,
                noise_name=noise_c["name"],
                noise_config_params=noise_c["params"],
                gt_npz_path=GT_NPZ_PATH,
                device_str=DEVICE
            )
            
            all_overall_results.append({
                "model_name": model_c["path"],
                "noise_name": noise_c["name"],
                "metrics_dict": current_run_metrics_dict 
            })
            if current_run_metrics_dict is None:
                 print(f"!!! Evaluation FAILED for Model: {model_c['name']}, Noise: {noise_c['name']} !!!")

    if TEMP_EVAL_DIR.exists(): 
        print(f"Cleaning up temporary directory: {TEMP_EVAL_DIR}")
        shutil.rmtree(TEMP_EVAL_DIR)

    format_and_print_results(all_overall_results)

if __name__ == "__main__":
    try:
        ablation_main()
    except Exception as e:
        print(f"An uncaught error occurred during the ablation study: {e}")
        import traceback
        traceback.print_exc()
        if TEMP_EVAL_DIR.exists(): 
            print(f"Attempting to clean up temporary directory on error: {TEMP_EVAL_DIR}")
            shutil.rmtree(TEMP_EVAL_DIR, ignore_errors=True)