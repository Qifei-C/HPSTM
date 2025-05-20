# smooth_noisy_npz.py
import torch
import numpy as np
import os
import argparse
import sys
from pathlib import Path

try:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added '{project_root}' to sys.path for module imports.")

    from src.models.pose_refiner_transformer import ManifoldRefinementTransformer
    from scripts.infer import load_model_from_checkpoint, refine_sequence_transformer
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"Ensure this script is run from a location where 'src' and 'scripts' are accessible from '{project_root}'.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Smooth a noisy NPZ file using ManifoldRefinementTransformer.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument('--input_noisy_npz_path', type=str, required=True,
                        help="Path to the input noisy NPZ file (must contain 'poses_r3j').")
    parser.add_argument('--output_smoothed_npz_path', type=str, required=True,
                        help="Path to save the output smoothed NPZ file.")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (cuda or cpu).")
    args = parser.parse_args()
    return args

def main_smooth_npz(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return
        
    model, model_constructor_args = load_model_from_checkpoint(args.checkpoint_path, device)
    model.eval()

    model_window_size = model_constructor_args.get('window_size', 150)
    skeleton_type = model_constructor_args.get('skeleton_type', 'smpl_24')
    model_was_trained_on_centered_data = model_constructor_args.get('center_around_root_amass', True)
    model_predicts_covariance = model_constructor_args.get('predict_covariance_transformer', True)

    print(f"Model loaded. Skeleton type: {skeleton_type}, Model window size: {model_window_size}, "
          f"Trained on centered data: {model_was_trained_on_centered_data}, Predicts covariance: {model_predicts_covariance}")

    print(f"Loading noisy NPZ data from: {args.input_noisy_npz_path}")
    if not os.path.exists(args.input_noisy_npz_path):
        print(f"Error: Input noisy NPZ file not found at {args.input_noisy_npz_path}")
        return

    try:
        data_npz = np.load(args.input_noisy_npz_path, allow_pickle=True)
    except ValueError as e:
        if "allow_pickle" in str(e):
            print(f"Failed to load NPZ with allow_pickle=False. Retrying with allow_pickle=True. "
                  f"Ensure the file source is trusted: {args.input_noisy_npz_path}")
            data_npz = np.load(args.input_noisy_npz_path, allow_pickle=True)
        else:
            raise e


    if 'poses_r3j' not in data_npz:
        print(f"Error: 'poses_r3j' key not found in the input NPZ file: {args.input_noisy_npz_path}")
        return
    
    noisy_poses_r3j_abs = data_npz['poses_r3j'].astype(np.float32)
    print(f"Loaded noisy 'poses_r3j' with shape: {noisy_poses_r3j_abs.shape}")

    bone_offsets_input = None
    if 'bone_offsets' in data_npz:
        bone_offsets_input = data_npz['bone_offsets'].astype(np.float32)
        print(f"Loaded 'bone_offsets' with shape: {bone_offsets_input.shape}")
    else:
        print("No 'bone_offsets' found in input NPZ. This key will not be in the output unless generated differently.")
    
    input_for_transformer = noisy_poses_r3j_abs.copy()
    root_positions_original_for_decentering = None

    if model_was_trained_on_centered_data:
        print("Model was trained on centered data. Centering input poses_r3j around the root joint.")
        # (B, T, J, C) -> (T, J, C)
        if input_for_transformer.ndim == 3: # (T, J, C)
            root_positions_original_for_decentering = input_for_transformer[:, 0:1, :].copy()
            input_for_transformer -= root_positions_original_for_decentering
        elif input_for_transformer.ndim == 4: # (B, T, J, C), B=1
             root_positions_original_for_decentering = input_for_transformer[:, :, 0:1, :].copy()
             input_for_transformer -= root_positions_original_for_decentering
        else:
            print(f"Error: Unexpected shape for noisy_poses_r3j_abs: {noisy_poses_r3j_abs.shape}. Expected 3D or 4D.")
            return
        print("Input data centered.")

    print("Performing inference (smoothing/denoising)...")
    smoothed_sequence_np, cholesky_L_output_np = refine_sequence_transformer(
        model,
        input_for_transformer,
        model_window_size,
        device,
        predict_covariance=model_predicts_covariance
    )
    print(f"Inference complete. Smoothed sequence shape: {smoothed_sequence_np.shape}")
    if cholesky_L_output_np is not None:
        print(f"Cholesky L factors predicted with shape: {cholesky_L_output_np.shape}")

    output_data_dict = {}
    output_data_dict['poses_r3j'] = smoothed_sequence_np.astype(np.float32)

    if bone_offsets_input is not None:
        output_data_dict['bone_offsets'] = bone_offsets_input
    
    if cholesky_L_output_np is not None:
        output_data_dict['cholesky_L'] = cholesky_L_output_np.astype(np.float32)
    
    output_data_dict['metadata'] = np.array({
        'source_file': str(Path(args.input_noisy_npz_path).name),
        'checkpoint_used': str(Path(args.checkpoint_path).name),
        'skeleton_type': skeleton_type,
        'processed_with_centering': model_was_trained_on_centered_data,
        'model_window_size': model_window_size
    })

    output_dir = Path(args.output_smoothed_npz_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving smoothed data to: {args.output_smoothed_npz_path}")
    try:
        np.savez_compressed(args.output_smoothed_npz_path, **output_data_dict)
        print("Successfully saved smoothed NPZ file.")
    except Exception as e:
        print(f"Error saving output NPZ file: {e}")

if __name__ == "__main__":
    args = parse_args()
    
    if len(sys.argv) == 1:
        print("No command line arguments provided, using placeholder default arguments for testing.")
        print("Please provide arguments via command line, e.g.:")
        print("python smooth_noisy_npz.py --checkpoint_path <path_to_ckpt> --input_noisy_npz_path <path_to_noisy.npz> --output_smoothed_npz_path <path_to_output.npz>")

    main_smooth_npz(args)