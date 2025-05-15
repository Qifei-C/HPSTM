# Manifold/scripts/train_transfer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import os
import argparse
from datetime import datetime
import glob
import random

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.amass_dataset import AMASSSubsetDataset
from src.datasets.pose_sequence_dataset import PoseSequenceDataset
from src.models.pose_refiner_transformer import ManifoldRefinementTransformer
from src.models.pose_refiner_simple import PoseRefinerSimple
from src.kinematics.forward_kinematics import ForwardKinematics
from src.kinematics.skeleton_utils import get_num_joints, get_skeleton_parents, get_rest_directions_tensor
from src.losses.temporal_loss import VelocityLoss, AccelerationLoss
from src.losses.position_loss import PositionMSELoss
from src.losses.bone_length_loss import BoneLengthMSELoss
from src.losses.covariance_loss import NegativeLogLikelihoodLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Human Pose Smoothing Model with Transfer Learning for Covariance")

    # --- Arguments from the original train.py (excluding predict_covariance_transformer) ---
    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'simple'],
                        help="Type of model to train ('transformer' or 'simple') - Should be 'transformer' for this script")
    # '--predict_covariance_transformer' is implicitly True in this script, so no need for CLI arg.

    parser.add_argument('--amass_root_dir', type=str, default=None,
                        help="Root directory containing AMASS .npz files. Will be split into train/val.")
    parser.add_argument('--val_split_ratio', type=float, default=0.1,
                        help="Fraction of AMASS data to use for validation.")
    parser.add_argument('--split_seed', type=int, default=None,
                        help="Random seed for train/val split.")

    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints_transfer", # Suggest different dir
                        help="Directory to save model checkpoints for this transfer learning run")
    parser.add_argument('--log_dir', type=str, default="results/logs_transfer", # Suggest different dir
                        help="Directory to save training logs for this transfer learning run")

    # Dataset parameters
    parser.add_argument('--skeleton_type', type=str, default='smpl_24', help="Skeleton type")
    parser.add_argument('--window_size', type=int, default=31, help="Number of frames per sequence window")
    parser.add_argument('--center_around_root_amass', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Center poses around root joint for AMASS dataset (True/False)")
    parser.add_argument('--gaussian_noise_std_train', type=float, default=0.03, help="Gaussian noise std for training")
    parser.add_argument('--temporal_noise_type', type=str, default='none', choices=['none', 'filtered', 'persistent'])
    parser.add_argument('--temporal_noise_scale', type=float, default=0.0)
    parser.add_argument('--temporal_filter_window', type=int, default=5)
    parser.add_argument('--temporal_event_prob', type=float, default=0.05)
    parser.add_argument('--temporal_decay', type=float, default=0.8)
    parser.add_argument('--outlier_prob', type=float, default=0.0)
    parser.add_argument('--outlier_scale', type=float, default=0.0)
    parser.add_argument('--bonelen_noise_scale', type=float, default=0.0)

    # Transformer Model hyperparameters (will be used for the new model instance)
    parser.add_argument('--d_model_transformer', type=int, default=256)
    parser.add_argument('--nhead_transformer', type=int, default=8)
    parser.add_argument('--num_encoder_layers_transformer', type=int, default=4)
    parser.add_argument('--num_decoder_layers_transformer', type=int, default=4)
    parser.add_argument('--dim_feedforward_transformer', type=int, default=1024)
    parser.add_argument('--dropout_transformer', type=float, default=0.1)
    parser.add_argument('--use_quaternions_transformer', type=lambda x: (str(x).lower() == 'true'), default=True)


    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4) # May need adjustment for fine-tuning
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--num_workers', type=int, default=4)
    
    # --- Crucial for transfer learning: path to the PRE-TRAINED model ---
    parser.add_argument('--pretrained_checkpoint_path', type=str, required=True,
                        help="Path to the pre-trained model checkpoint (without covariance head) to load weights from.")

    # Loss weights (covariance loss weight is now always active)
    parser.add_argument('--w_tf_loss_pose', type=float, default=1.0)
    parser.add_argument('--w_tf_loss_vel', type=float, default=0.5)
    parser.add_argument('--w_tf_loss_accel', type=float, default=1.0)
    parser.add_argument('--w_tf_loss_bone', type=float, default=0.1)
    parser.add_argument('--w_tf_loss_cov', type=float, default=0.01,
                        help="Weight for Transformer model's covariance (NLL) loss component.")

    parser.add_argument('--run_name', type=str, default=f"transfer_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="A name for this transfer training run")

    args = parser.parse_args()

    if args.model_type != 'transformer':
        parser.error("This script is intended for transfer learning with the 'transformer' model type.")
    if not (0.0 < args.val_split_ratio < 1.0):
        parser.error("--val_split_ratio must be between 0.0 and 1.0 (exclusive).")

    return args

def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    run_checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    log_file_path = os.path.join(args.log_dir, f"{args.run_name}_train_transfer.log")

    def log_message(message):
        print(message)
        with open(log_file_path, 'a') as f:
            f.write(message + '\n')

    log_message(f"Starting transfer learning run for covariance prediction: {args.run_name}")
    log_message("Arguments: " + str(vars(args)))

    device = torch.device(args.device)
    num_joints = get_num_joints(args.skeleton_type)
    skeleton_parents_np = get_skeleton_parents(args.skeleton_type)
    smplh_to_smpl_body_indices = None # Logic for this remains the same as in train.py

    # --- Dataset and Dataloader (same as train.py for AMASS) ---
    log_message(f"Loading AMASS dataset for Transformer model from root: {args.amass_root_dir}")
    # ... (Copy AMASS dataset loading and splitting logic from your train.py) ...
    if not os.path.isdir(args.amass_root_dir):
        log_message(f"Error: AMASS root directory not found: {args.amass_root_dir}")
        return
    all_npz_files = glob.glob(os.path.join(args.amass_root_dir, '**', '*.npz'), recursive=True)
    if not all_npz_files:
        log_message(f"Error: No .npz files found in {args.amass_root_dir}. Exiting.")
        return
    if args.split_seed is not None: random.seed(args.split_seed)
    random.shuffle(all_npz_files)
    split_idx = int(len(all_npz_files) * (1 - args.val_split_ratio))
    train_files, val_files = all_npz_files[:split_idx], all_npz_files[split_idx:]
    # ... (rest of dataset init from train.py, ensure AMASSSubsetDataset is used)
    train_dataset = AMASSSubsetDataset(
        data_paths=train_files, window_size=args.window_size,
        skeleton_type=args.skeleton_type, is_train=True,
        center_around_root=args.center_around_root_amass,
        joint_selector_indices=smplh_to_smpl_body_indices, # Define this if needed
        gaussian_noise_std=args.gaussian_noise_std_train,
        temporal_noise_type=args.temporal_noise_type,
        temporal_noise_scale=args.temporal_noise_scale,
        temporal_filter_window=args.temporal_filter_window,
        temporal_event_prob=args.temporal_event_prob,
        temporal_decay=args.temporal_decay,
        outlier_prob=args.outlier_prob,
        outlier_scale=args.outlier_scale,
        bonelen_noise_scale=args.bonelen_noise_scale
    )
    val_dataset = None
    if val_files:
        val_dataset = AMASSSubsetDataset(
            data_paths=val_files, window_size=args.window_size,
            skeleton_type=args.skeleton_type, is_train=False,
            center_around_root=args.center_around_root_amass,
            joint_selector_indices=smplh_to_smpl_body_indices, # Define this
            gaussian_noise_std=0.0, temporal_noise_type='none',
            temporal_noise_scale=0.0, outlier_prob=0.0, bonelen_noise_scale=0.0
        )
    # ... (Dataloader creation) ...
    if train_dataset is None or len(train_dataset) == 0:
         log_message("Error: Training dataset is empty. Exiting.")
         return
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        log_message(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    else:
        log_message(f"Training dataset size: {len(train_dataset)}, Validation dataset is empty or not created.")


    # --- Model and Loss Criteria ---
    # Model is ALWAYS instantiated WITH covariance prediction enabled for this script
    model_constructor_args_transfer = {
        'num_joints': num_joints, 'joint_dim': 3, 'window_size': args.window_size,
        'd_model': args.d_model_transformer, 'nhead': args.nhead_transformer,
        'num_encoder_layers': args.num_encoder_layers_transformer,
        'num_decoder_layers': args.num_decoder_layers_transformer,
        'dim_feedforward': args.dim_feedforward_transformer, 'dropout': args.dropout_transformer,
        'smpl_parents': skeleton_parents_np.tolist(),
        'use_quaternions': args.use_quaternions_transformer, # Ensure this comes from args
        'skeleton_type': args.skeleton_type,
        'predict_covariance': True, # <- Crucial: Always True for this transfer script
        # Include other args from your original train.py's model_constructor_args if ManifoldRefinementTransformer needs them
        'center_around_root_amass': args.center_around_root_amass # Store for potential reference if needed
    }

    model = ManifoldRefinementTransformer(**model_constructor_args_transfer).to(device)

    criterion_pose = nn.L1Loss().to(device)
    criterion_vel = VelocityLoss(loss_type='l1').to(device)
    criterion_accel = AccelerationLoss(loss_type='l1').to(device)
    criterion_bone = BoneLengthMSELoss(parents_list=skeleton_parents_np.tolist()).to(device)
    criterion_cov = NegativeLogLikelihoodLoss().to(device) # Covariance loss is always used

    log_message(f"Instantiated Transformer model FOR TRANSFER LEARNING (covariance head enabled).")
    log_message(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Load Pre-trained Weights (MODIFIED LOGIC) ---
    start_epoch = 1
    best_val_loss = float('inf')

    if not os.path.isfile(args.pretrained_checkpoint_path):
        log_message(f"Error: Pre-trained checkpoint file not found at {args.pretrained_checkpoint_path}. Exiting.")
        return

    log_message(f"Loading pre-trained weights from: {args.pretrained_checkpoint_path}")
    pretrained_checkpoint = torch.load(args.pretrained_checkpoint_path, map_location=device)
    
    pretrained_state_dict = pretrained_checkpoint['model_state_dict']
    current_model_dict = model.state_dict()

    # Filter out parameters that are not in the current model OR have size mismatch
    # (Mainly for `output_head_covariance` which won't be in pretrained_state_dict)
    weights_to_load = {k: v for k, v in pretrained_state_dict.items() 
                       if k in current_model_dict and v.size() == current_model_dict[k].size()}
    
    current_model_dict.update(weights_to_load) # Update current model's dict with loaded weights
    model.load_state_dict(current_model_dict)   # Load the merged dict

    num_loaded_params = sum(p.numel() for p in weights_to_load.values())
    log_message(f"Loaded {len(weights_to_load)} matching Tensors (total {num_loaded_params} parameters) from pre-trained checkpoint.")

    missing_keys = [k for k in current_model_dict.keys() if k not in weights_to_load]
    if missing_keys:
        log_message(f"Weights for these layers were NOT in the pre-trained checkpoint and are RANDOMLY INITIALIZED: {missing_keys}")
    
    # Optimizer: It's generally safer to re-initialize the optimizer when layers change
    # or when transfer learning, especially if learning rates for new layers should differ.
    # For simplicity, we re-initialize here.
    # You could try to load and adapt if needed, but it's more complex.
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    log_message("Optimizer and Scheduler re-initialized for transfer learning.")

    # --- Training Loop (mostly same as train.py for transformer) ---
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        total_train_loss_epoch = 0
        total_train_loss_p_epoch, total_train_loss_v_epoch, total_train_loss_a_epoch = 0,0,0
        total_train_loss_b_epoch, total_train_loss_c_epoch = 0,0

        for batch_idx, (noisy_seq, clean_seq, bone_offsets_at_rest) in enumerate(train_loader):
            noisy_seq, clean_seq, bone_offsets_at_rest = noisy_seq.to(device), clean_seq.to(device), bone_offsets_at_rest.to(device)
            optimizer.zero_grad()
            refined_seq, predicted_bone_lengths_seq, pred_cholesky_L = model(noisy_seq)

            loss_p = criterion_pose(refined_seq, clean_seq)
            loss_v = criterion_vel(refined_seq, clean_seq)
            loss_a = criterion_accel(refined_seq, clean_seq)
            target_canonical_bone_lengths = torch.norm(bone_offsets_at_rest, dim=-1)
            loss_b = criterion_bone(predicted_bone_lengths_seq, target_canonical_bone_lengths, target_is_canonical_lengths=True)
            loss_c = criterion_cov(clean_seq, refined_seq, pred_cholesky_L) # This is always calculated now

            loss = (args.w_tf_loss_pose * loss_p +
                    args.w_tf_loss_vel * loss_v +
                    args.w_tf_loss_accel * loss_a +
                    args.w_tf_loss_bone * loss_b +
                    args.w_tf_loss_cov * loss_c)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss_epoch += loss.item()
            total_train_loss_p_epoch += loss_p.item()
            total_train_loss_v_epoch += loss_v.item()
            total_train_loss_a_epoch += loss_a.item()
            total_train_loss_b_epoch += loss_b.item()
            total_train_loss_c_epoch += loss_c.item()

            if batch_idx % 50 == 0:
                log_message(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f} "
                            f"(P: {loss_p.item():.4f} V: {loss_v.item():.4f} A: {loss_a.item():.4f} "
                            f"B: {loss_b.item():.4f} C: {loss_c.item():.4f})")

        avg_train_loss_epoch = total_train_loss_epoch / len(train_loader)
        avg_train_p = total_train_loss_p_epoch / len(train_loader)
        avg_train_v = total_train_loss_v_epoch / len(train_loader)
        avg_train_a = total_train_loss_a_epoch / len(train_loader)
        avg_train_b = total_train_loss_b_epoch / len(train_loader)
        avg_train_c = total_train_loss_c_epoch / len(train_loader)
        log_message(f"Epoch {epoch}/{args.num_epochs} | Avg Train Loss: {avg_train_loss_epoch:.4f} "
                    f"(P: {avg_train_p:.4f} V: {avg_train_v:.4f} A: {avg_train_a:.4f} "
                    f"B: {avg_train_b:.4f} C: {avg_train_c:.4f})")

        # --- Validation Loop (mostly same as train.py for transformer) ---
        if val_loader is None:
            # ... (save checkpoint based on train loss if no val_loader) ...
            if avg_train_loss_epoch < best_val_loss: # Using train loss if no val
                best_val_loss = avg_train_loss_epoch
                checkpoint_name = f"model_epoch_{epoch:03d}_trainloss_{avg_train_loss_epoch:.4f}.pth"
                checkpoint_path = os.path.join(run_checkpoint_dir, checkpoint_name)
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss_epoch, # Save train loss here
                    'args': vars(args),
                    'model_constructor_args': model_constructor_args_transfer # Use the transfer constructor args
                }, checkpoint_path)
                log_message(f"Best model checkpoint (based on train loss) saved to {checkpoint_path}")
            continue
            
        model.eval()
        total_val_loss_epoch = 0
        running_mpjpe_sum, running_mpjpe_count = 0.0, 0
        total_val_loss_p_epoch, total_val_loss_v_epoch, total_val_loss_a_epoch = 0,0,0
        total_val_loss_b_epoch, total_val_loss_c_epoch = 0,0

        with torch.no_grad():
            for noisy_seq_val, clean_seq_val, bone_offsets_at_rest_val in val_loader:
                noisy_seq_val, clean_seq_val, bone_offsets_at_rest_val = \
                    noisy_seq_val.to(device), clean_seq_val.to(device), bone_offsets_at_rest_val.to(device)
                
                refined_seq_val, predicted_bone_lengths_seq_val, pred_cholesky_L_val = model(noisy_seq_val)
                
                loss_p_val = criterion_pose(refined_seq_val, clean_seq_val)
                loss_v_val = criterion_vel(refined_seq_val, clean_seq_val)
                loss_a_val = criterion_accel(refined_seq_val, clean_seq_val)
                target_canonical_bone_lengths_val = torch.norm(bone_offsets_at_rest_val, dim=-1)
                loss_b_val = criterion_bone(predicted_bone_lengths_seq_val, target_canonical_bone_lengths_val, target_is_canonical_lengths=True)
                loss_c_val = criterion_cov(clean_seq_val, refined_seq_val, pred_cholesky_L_val)

                val_loss_batch = (args.w_tf_loss_pose * loss_p_val +
                                  args.w_tf_loss_vel * loss_v_val +
                                  args.w_tf_loss_accel * loss_a_val +
                                  args.w_tf_loss_bone * loss_b_val +
                                  args.w_tf_loss_cov * loss_c_val)
                
                total_val_loss_epoch += val_loss_batch.item() * noisy_seq_val.size(0)
                error_vectors = refined_seq_val - clean_seq_val
                per_joint_errors = torch.norm(error_vectors, p=2, dim=-1)
                running_mpjpe_sum += torch.sum(per_joint_errors).item()
                running_mpjpe_count += per_joint_errors.numel()
                
                total_val_loss_p_epoch += loss_p_val.item() * noisy_seq_val.size(0)
                total_val_loss_v_epoch += loss_v_val.item() * noisy_seq_val.size(0)
                total_val_loss_a_epoch += loss_a_val.item() * noisy_seq_val.size(0)
                total_val_loss_b_epoch += loss_b_val.item() * noisy_seq_val.size(0)
                total_val_loss_c_epoch += loss_c_val.item() * noisy_seq_val.size(0)

        num_val_samples = len(val_dataset)
        avg_val_loss = total_val_loss_epoch / num_val_samples
        avg_val_mpjpe = (running_mpjpe_sum / running_mpjpe_count) * 1000 if running_mpjpe_count > 0 else 0.0
        
        avg_val_p = total_val_loss_p_epoch / num_val_samples
        avg_val_v = total_val_loss_v_epoch / num_val_samples
        avg_val_a = total_val_loss_a_epoch / num_val_samples
        avg_val_b = total_val_loss_b_epoch / num_val_samples
        avg_val_c = total_val_loss_c_epoch / num_val_samples
        log_message(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} MPJPE: {avg_val_mpjpe:.2f}mm "
                    f"(P: {avg_val_p:.4f} V: {avg_val_v:.4f} A: {avg_val_a:.4f} "
                    f"B: {avg_val_b:.4f} C: {avg_val_c:.4f})")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_payload = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss, 'val_mpjpe': avg_val_mpjpe,
                'args': vars(args), # Current args for this transfer run
                'model_constructor_args': model_constructor_args_transfer, # Args used to create this model
                'val_loss_components': {'P': avg_val_p, 'V': avg_val_v, 'A': avg_val_a, 'B': avg_val_b, 'C': avg_val_c}
            }
            checkpoint_name = f"model_epoch_{epoch:03d}_valloss_{avg_val_loss:.4f}_mpjpe_{avg_val_mpjpe:.2f}_C_{avg_val_c:.4f}.pth"
            checkpoint_path = os.path.join(run_checkpoint_dir, checkpoint_name)
            torch.save(checkpoint_payload, checkpoint_path)
            log_message(f"Best model checkpoint saved to {checkpoint_path}")

    log_message(f"Transfer learning training finished for run: {args.run_name}")

if __name__ == "__main__":
    args = parse_args()
    main(args)