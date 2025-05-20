# scripts/infer.py
import torch
import numpy as np
import os
import argparse

from src.models.pose_refiner_transformer_old import ManifoldRefinementTransformerOld # MODIFIED: Added import for old model
from src.models.pose_refiner_simple import PoseRefinerSimple
from src.kinematics.forward_kinematics import ForwardKinematics
from src.kinematics.skeleton_utils import get_skeleton_parents, get_rest_directions_dict 

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Human Pose Smoothing Model")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file)")
    parser.add_argument('--input_pose_path', type=str, required=True,
                        help="Path to the input noisy pose sequence (.npy file). "
                             "Expected shape: (num_frames, num_joints, 3) for transformer model, "
                             "or (num_frames, num_joints*3) for simple model.")
    parser.add_argument('--output_path', type=str, default="refined_sequence.npy",
                        help="Path to save the refined pose sequence (.npy file)")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (cuda or cpu)")
    
    args = parser.parse_args()
    return args

def load_model_from_checkpoint_old(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_constructor_args' not in checkpoint:
        raise KeyError("Checkpoint must contain 'model_constructor_args' for model instantiation.")
    model_args = checkpoint['model_constructor_args']
    
    model_type = model_args.get('model_type', 'transformer')

    if model_type == 'transformer':
        required_transformer_args = ['num_joints', 'joint_dim', 'window_size', 'd_model', 'nhead', 
                                     'num_encoder_layers', 'num_decoder_layers', 'dim_feedforward', 
                                     'dropout', 'smpl_parents', 'use_quaternions']
        for req_arg in required_transformer_args:
            if req_arg not in model_args:
                raise KeyError(f"Missing argument '{req_arg}' in checkpoint's model_constructor_args for transformer model.")

        model = ManifoldRefinementTransformerOld(
            num_joints=model_args['num_joints'],
            joint_dim=model_args['joint_dim'],
            window_size=model_args['window_size'],
            d_model=model_args['d_model'],
            nhead=model_args['nhead'],
            num_encoder_layers=model_args['num_encoder_layers'],
            num_decoder_layers=model_args['num_decoder_layers'],
            dim_feedforward=model_args['dim_feedforward'],
            dropout=model_args['dropout'],
            smpl_parents=model_args['smpl_parents'], # Should be list
            use_quaternions=model_args['use_quaternions']
        )
    # elif model_type == 'simple':
    #     # fk_module_instance = ForwardKinematics(
    #     #     parents_list=model_args['smpl_parents'], # Assuming these are saved
    #     #     rest_directions_dict=get_rest_directions_dict(model_args.get('skeleton_type', 'smpl'))
    #     # ).to(device)
    #     # model = PoseModel(
    #     #     J=model_args['num_joints'],
    #     #     window_size=model_args['window_size'],
    #     #     # Add other PoseModel specific args from model_args
    #     #     fk_module=fk_module_instance
    #     # )
    #     print("Simple model loading not fully implemented yet.")
    #     raise NotImplementedError("Simple model loading needs to be completed.")
    else:
        raise ValueError(f"Unsupported model type '{model_type}' in checkpoint.")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model '{model_type}' loaded from {checkpoint_path}.")
    print(f"Trained for {checkpoint.get('epoch', 'N/A')} epochs. Val MPJPE: {checkpoint.get('val_mpjpe', 'N/A'):.2f} mm")
    
    return model, model_args

def refine_sequence_transformer_old(model, noisy_r3j_sequence_np, window_size, device, center_input_if_model_expects_it = True, root_positions_original_for_decentering = True):
    num_frames, num_joints_data, _ = noisy_r3j_sequence_np.shape
    
    if num_joints_data != model.num_joints:
        raise ValueError(f"Input sequence has {num_joints_data} joints, model expects {model.num_joints}.")

    noisy_r3j_sequence = torch.from_numpy(noisy_r3j_sequence_np).float().to(device)

    original_num_frames = num_frames
    padding_start_len = 0
    padding_end_len = 0

    if num_frames < window_size:
        padding_needed = window_size - num_frames
        padding_start_len = padding_needed // 2 + padding_needed % 2
        padding_end_len = padding_needed // 2
        
        padding_frames_start = noisy_r3j_sequence[0:1].repeat(padding_start_len, 1, 1)
        padding_frames_end = noisy_r3j_sequence[-1:].repeat(padding_end_len, 1, 1)
        
        noisy_r3j_sequence = torch.cat([padding_frames_start, noisy_r3j_sequence, padding_frames_end], dim=0)
        print(f"Padded input sequence from {original_num_frames} to {noisy_r3j_sequence.shape[0]} frames.")
        num_frames = noisy_r3j_sequence.shape[0]
        
    refined_frames_aggregated = torch.zeros_like(noisy_r3j_sequence)
    counts = torch.zeros(num_frames, device=device) 

    with torch.no_grad():
        for i in range(num_frames - window_size + 1):
            window_input = noisy_r3j_sequence[i : i + window_size].unsqueeze(0) # (1, W, J, 3)

            refined_r3j_seq_window, _ = model(window_input)
            refined_window = refined_r3j_seq_window.squeeze(0) # (W, J, 3)

            refined_frames_aggregated[i : i + window_size] += refined_window
            counts[i : i + window_size] += 1
    
    counts = counts.unsqueeze(-1).unsqueeze(-1).clamp(min=1)
    refined_sequence_padded = refined_frames_aggregated / counts
    
    # Remove padding if applied
    if original_num_frames < window_size:
        refined_sequence = refined_sequence_padded[padding_start_len : padding_start_len + original_num_frames]
    else:
        refined_sequence = refined_sequence_padded

    return refined_sequence.cpu().numpy()


def main(args):
    device = torch.device(args.device)
    model, model_constructor_args = load_model_from_checkpoint(args.checkpoint_path, device)
    
    model_type = model_constructor_args.get('model_type', 'transformer')
    window_size = model_constructor_args['window_size']
    num_joints_model = model_constructor_args['num_joints']
    center_around_root_dataset = model_constructor_args.get('center_around_root', True if model_type=='transformer' else False)


    if not os.path.exists(args.input_pose_path):
        raise FileNotFoundError(f"Input pose file not found: {args.input_pose_path}")
    noisy_input_np = np.load(args.input_pose_path)
    print(f"Loaded input pose sequence from {args.input_pose_path} with shape {noisy_input_np.shape}")

    refined_output_sequence = None

    if model_type == 'transformer':
        if noisy_input_np.ndim != 3 or noisy_input_np.shape[-1] != 3:
            raise ValueError(f"Transformer model expects input shape (frames, joints, 3), got {noisy_input_np.shape}")
        if noisy_input_np.shape[1] != num_joints_model:
             raise ValueError(f"Input sequence has {noisy_input_np.shape[1]} joints, but model was trained with {num_joints_model} joints.")

        input_for_model = noisy_input_np.copy()
        root_positions_original_seq = None
        if center_around_root_dataset:
            print("Centering input sequence around root joint (joint 0) as per model training.")
            root_positions_original_seq = noisy_input_np[:, 0:1, :].copy() # (N, 1, 3)
            input_for_model = noisy_input_np - root_positions_original_seq
        
        refined_output_centered = refine_sequence_transformer(model, input_for_model, window_size, device)
        
        if center_around_root_dataset and root_positions_original_seq is not None:
            print("Adding back original root joint positions.")
            if refined_output_centered.shape[0] != root_positions_original_seq.shape[0]:
                 raise ValueError("Mismatch in frames between refined output and original root positions after padding/unpadding.")
            refined_output_sequence = refined_output_centered + root_positions_original_seq
        else:
            refined_output_sequence = refined_output_centered

    else:
        raise ValueError(f"Unsupported model type '{model_type}' for inference.")

    if refined_output_sequence is not None:
        np.save(args.output_path, refined_output_sequence)
        print(f"Refined sequence saved to {args.output_path} with shape {refined_output_sequence.shape}")
    else:
        print("Inference did not produce an output.")

if __name__ == "__main__":
    args = parse_args()
    main(args)