# scripts/infer.py
import torch
import numpy as np
import os
import argparse

# Adjust imports based on your project structure
from src.models.pose_refiner_transformer import ManifoldRefinementTransformer
from src.models.pose_refiner_simple import PoseModel
from src.kinematics.forward_kinematics import ForwardKinematics
from src.kinematics.skeleton_utils import get_skeleton_parents, get_rest_directions_dict # For simple model FK

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
    
    # Optional: if bone offsets are needed and not part of checkpoint or derivable
    parser.add_argument('--bone_offsets_path', type=str, default=None,
                        help="Path to subject-specific T-pose bone offsets (.npy, shape (num_joints, 3)). "
                             "Required by transformer model if not using canonical. "
                             "If not provided, canonical/average offsets might be attempted if model supports.")
    args = parser.parse_args()
    return args

def load_model_from_checkpoint(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # --- Load model constructor arguments ---
    # These should have been saved during training
    if 'model_constructor_args' not in checkpoint:
        raise KeyError("Checkpoint must contain 'model_constructor_args' for model instantiation.")
    model_args = checkpoint['model_constructor_args']
    
    model_type = model_args.get('model_type', 'transformer') # Default to transformer if not specified

    if model_type == 'transformer':
        # Ensure all required args for ManifoldRefinementTransformer are present
        required_transformer_args = ['num_joints', 'joint_dim', 'window_size', 'd_model', 'nhead', 
                                     'num_encoder_layers', 'num_decoder_layers', 'dim_feedforward', 
                                     'dropout', 'smpl_parents', 'use_quaternions']
        for req_arg in required_transformer_args:
            if req_arg not in model_args:
                raise KeyError(f"Missing argument '{req_arg}' in checkpoint's model_constructor_args for transformer model.")

        model = ManifoldRefinementTransformer(
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
    
    return model, model_args # Return model_args as it contains window_size etc.


def refine_sequence_transformer(model, noisy_r3j_sequence_np, bone_offsets_np, window_size, device):
    """
    Refines a single pose sequence using the ManifoldRefinementTransformer model.
    Args:
        model: The trained PyTorch model.
        noisy_r3j_sequence_np (np.ndarray): (num_frames, num_joints, 3)
        bone_offsets_np (np.ndarray): (num_joints, 3) T-pose bone offsets.
        window_size (int): The window size the model was trained with.
        device: Torch device.
    Returns:
        refined_sequence_np (np.ndarray): (num_frames, num_joints, 3)
    """
    num_frames, num_joints_data, _ = noisy_r3j_sequence_np.shape
    
    # num_joints_model comes from model.num_joints which is set during its __init__
    if num_joints_data != model.num_joints:
        raise ValueError(f"Input sequence has {num_joints_data} joints, model expects {model.num_joints}.")

    noisy_r3j_sequence = torch.from_numpy(noisy_r3j_sequence_np).float().to(device)
    bone_offsets = torch.from_numpy(bone_offsets_np).float().to(device) # (J, 3)

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
        num_frames = noisy_r3j_sequence.shape[0] # Update num_frames after padding

    refined_frames_aggregated = torch.zeros_like(noisy_r3j_sequence)
    counts = torch.zeros(num_frames, device=device) 

    with torch.no_grad():
        for i in range(num_frames - window_size + 1):
            window_input = noisy_r3j_sequence[i : i + window_size].unsqueeze(0) # (1, W, J, 3)
            batched_bone_offsets = bone_offsets.unsqueeze(0) # (1, J, 3)

            refined_window = model(window_input, batched_bone_offsets).squeeze(0) # (W, J, 3)

            refined_frames_aggregated[i : i + window_size] += refined_window
            counts[i : i + window_size] += 1
    
    counts = counts.unsqueeze(-1).unsqueeze(-1).clamp(min=1) # Add J and D dims for broadcasting
    refined_sequence_padded = refined_frames_aggregated / counts
    
    # Remove padding if applied
    if original_num_frames < window_size:
        refined_sequence = refined_sequence_padded[padding_start_len : padding_start_len + original_num_frames]
    else:
        refined_sequence = refined_sequence_padded

    return refined_sequence.cpu().numpy()

# def refine_sequence_simple(model, pose_sequence_flat_np, window_size, device):
#     """
#     Refines a sequence using the simple PoseModel (center frame refinement).
#     Args:
#         model: Trained PoseModel instance.
#         pose_sequence_flat_np (np.ndarray): (num_frames, num_joints*3)
#         window_size (int): Window size used during training.
#         device: Torch device.
#     Returns:
#         refined_sequence_flat_np (np.ndarray): (num_frames, num_joints*3)
#     """
#     model.eval()
#     half = window_size // 2
#     T, J_flat_dim = pose_sequence_flat_np.shape
#     num_joints = model.J # Get num_joints from the model instance
    
#     # Pad sequence (replication padding)
#     pad_left = np.repeat(pose_sequence_flat_np[0:1], half, axis=0)
#     pad_right = np.repeat(pose_sequence_flat_np[-1:], half, axis=0) # half or half-1 depending on even/odd window
#     padded_seq_np = np.vstack([pad_left, pose_sequence_flat_np, pad_right])
    
#     refined_seq_list = []

#     for t_center_original in range(T): # Iterate through original frames
#         # The window is centered at t_center_original in the original sequence
#         # which corresponds to t_center_original + half in the padded sequence
#         window_start_idx = t_center_original # t_center_original + half - half
#         window_np = padded_seq_np[window_start_idx : window_start_idx + window_size] 
        
#         inp_tensor = torch.from_numpy(window_np).float().unsqueeze(0).to(device) # (1, W, J*3)
        
#         with torch.no_grad():
#             # PoseModel outputs: pred_positions (B, J, 3), pred_bones (B, J)
#             pred_positions_tensor, _ = model(inp_tensor) # (1, J, 3)
        
#         # Reshape to (J*3) for the refined center frame
#         refined_frame_flat = pred_positions_tensor.cpu().numpy().reshape(J_flat_dim) 
#         refined_seq_list.append(refined_frame_flat)
        
#     return np.array(refined_seq_list)


def main(args):
    device = torch.device(args.device)
    model, model_constructor_args = load_model_from_checkpoint(args.checkpoint_path, device)
    
    model_type = model_constructor_args.get('model_type', 'transformer')
    window_size = model_constructor_args['window_size']
    num_joints_model = model_constructor_args['num_joints']
    center_around_root_dataset = model_constructor_args.get('center_around_root', True if model_type=='transformer' else False)


    # Load input pose sequence
    if not os.path.exists(args.input_pose_path):
        raise FileNotFoundError(f"Input pose file not found: {args.input_pose_path}")
    noisy_input_np = np.load(args.input_pose_path)
    print(f"Loaded input pose sequence from {args.input_pose_path} with shape {noisy_input_np.shape}")

    refined_output_sequence = None

    if model_type == 'transformer':
        # Expected input: (num_frames, num_joints, 3)
        if noisy_input_np.ndim != 3 or noisy_input_np.shape[-1] != 3:
            raise ValueError(f"Transformer model expects input shape (frames, joints, 3), got {noisy_input_np.shape}")
        if noisy_input_np.shape[1] != num_joints_model:
             raise ValueError(f"Input sequence has {noisy_input_np.shape[1]} joints, but model was trained with {num_joints_model} joints.")

        if args.bone_offsets_path:
            bone_offsets_np = np.load(args.bone_offsets_path)
            if bone_offsets_np.shape != (num_joints_model, 3):
                raise ValueError(f"Bone offsets shape mismatch. Expected ({num_joints_model}, 3), got {bone_offsets_np.shape}")
        else:
            # This is a placeholder. A proper canonical offset should be defined or loaded.
            print("Warning: bone_offsets_path not provided. Using zero bone offsets. This is likely incorrect.")
            bone_offsets_np = np.zeros((num_joints_model, 3), dtype=np.float32)

        # If dataset was centered around root, apply the same transformation to input
        input_for_model = noisy_input_np.copy()
        root_positions_original_seq = None
        if center_around_root_dataset:
            print("Centering input sequence around root joint (joint 0) as per model training.")
            root_positions_original_seq = noisy_input_np[:, 0:1, :].copy() # (N, 1, 3)
            input_for_model = noisy_input_np - root_positions_original_seq
        
        refined_output_centered = refine_sequence_transformer(model, input_for_model, bone_offsets_np, window_size, device)
        
        if center_around_root_dataset and root_positions_original_seq is not None:
            print("Adding back original root joint positions.")
            # Ensure refined_output_centered matches original length before adding root
            if refined_output_centered.shape[0] != root_positions_original_seq.shape[0]:
                 raise ValueError("Mismatch in frames between refined output and original root positions after padding/unpadding.")
            refined_output_sequence = refined_output_centered + root_positions_original_seq
        else:
            refined_output_sequence = refined_output_centered


    # elif model_type == 'simple':
    #     # Expected input: (num_frames, num_joints*3)
    #     if noisy_input_np.ndim != 2 or noisy_input_np.shape[1] != num_joints_model * 3:
    #         raise ValueError(f"Simple model expects input shape (frames, joints*3), got {noisy_input_np.shape}")
    #     refined_output_sequence = refine_sequence_simple(model, noisy_input_np, window_size, device)
    #     # Output is (num_frames, num_joints*3), might need reshaping to (num_frames, num_joints, 3)
    #     refined_output_sequence = refined_output_sequence.reshape(-1, num_joints_model, 3)

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