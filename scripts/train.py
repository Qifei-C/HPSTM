# scripts/train.py
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
from src.kinematics.forward_kinematics import ForwardKinematics # Keep if simple model uses it directly
from src.kinematics.skeleton_utils import get_num_joints, get_skeleton_parents, get_rest_directions_tensor
from src.losses.temporal_loss import VelocityLoss, AccelerationLoss
from src.losses.position_loss import PositionMSELoss
from src.losses.bone_length_loss import BoneLengthMSELoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Human Pose Smoothing Model")

    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'simple'],
                        help="Type of model to train ('transformer' or 'simple')")

    parser.add_argument('--amass_root_dir', type=str, default=None,
                        help="Root directory containing AMASS .npz files (for 'transformer' model). Will be split into train/val.")
    parser.add_argument('--val_split_ratio', type=float, default=0.1,
                        help="Fraction of AMASS data to use for validation (e.g., 0.1 for 10%).")
    parser.add_argument('--split_seed', type=int, default=None,
                        help="Random seed for train/val split to ensure reproducibility.")

    parser.add_argument('--simple_data_paths_train', type=str, nargs='+', default=None,
                        help="List of paths to .npy sequence files for training 'simple' model (each T, J*3)")
    parser.add_argument('--simple_data_paths_val', type=str, nargs='+', default=None,
                        help="List of paths to .npy sequence files for validating 'simple' model (each T, J*3)")

    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument('--log_dir', type=str, default="results/logs",
                        help="Directory to save training logs")

    # --- Dataset parameters ---
    parser.add_argument('--skeleton_type', type=str, default='smpl_24',
                        help="Skeleton type (e.g., 'smpl_24') from skeleton_utils.py")
    parser.add_argument('--window_size', type=int, default=31,
                        help="Number of frames per sequence window")
    parser.add_argument('--center_around_root_amass', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Center poses around root joint for AMASS dataset (True/False)")
    parser.add_argument('--gaussian_noise_std_train', type=float, default=0.03,
                        help="Gaussian noise std for training data augmentation (for AMASS dataset)")
    parser.add_argument('--temporal_noise_type', type=str, default='none',
                        choices=['none', 'filtered', 'persistent'],
                        help="Type of temporal noise for AMASS dataset")
    parser.add_argument('--temporal_noise_scale', type=float, default=0.0,
                        help="Scale for temporal noise for AMASS dataset")
    parser.add_argument('--temporal_filter_window', type=int, default=5,
                        help="Window size for filtered temporal noise for AMASS dataset")
    parser.add_argument('--temporal_event_prob', type=float, default=0.05,
                        help="Event probability for persistent temporal noise for AMASS dataset")
    parser.add_argument('--temporal_decay', type=float, default=0.8,
                        help="Decay factor for persistent temporal noise for AMASS dataset")
    parser.add_argument('--outlier_prob', type=float, default=0.0,
                        help="Probability of a joint being an outlier per frame for AMASS dataset")
    parser.add_argument('--outlier_scale', type=float, default=0.0,
                        help="Maximum deviation scale for outliers for AMASS dataset")
    parser.add_argument('--bonelen_noise_scale', type=float, default=0.0,
                        help="Maximum relative bone length perturbation scale for AMASS dataset")

    # --- Transformer Model hyperparameters ---
    parser.add_argument('--d_model_transformer', type=int, default=256, help="Transformer: d_model")
    parser.add_argument('--nhead_transformer', type=int, default=8, help="Transformer: nhead")
    parser.add_argument('--num_encoder_layers_transformer', type=int, default=4, help="Transformer: num_encoder_layers")
    parser.add_argument('--num_decoder_layers_transformer', type=int, default=4, help="Transformer: num_decoder_layers")
    parser.add_argument('--dim_feedforward_transformer', type=int, default=1024, help="Transformer: dim_feedforward")
    parser.add_argument('--dropout_transformer', type=float, default=0.1, help="Transformer: dropout")
    parser.add_argument('--use_quaternions_transformer', type=bool, default=True, help="Transformer: use quaternions")

    # --- Simple Model hyperparameters ---
    parser.add_argument('--d_model_simple', type=int, default=96, help="SimpleRefiner: d_model")
    parser.add_argument('--nhead_simple', type=int, default=8, help="SimpleRefiner: nhead")
    parser.add_argument('--num_encoder_layers_simple', type=int, default=4, help="SimpleRefiner: num_encoder_layers")
    parser.add_argument('--dim_feedforward_simple', type=int, default=256, help="SimpleRefiner: dim_feedforward")
    parser.add_argument('--dropout_simple', type=float, default=0.1, help="SimpleRefiner: dropout")

    # --- Training hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers")

    # --- Loss weights ---
    parser.add_argument('--w_tf_loss_pose', type=float, default=1.0)
    parser.add_argument('--w_tf_loss_vel', type=float, default=0.5)
    parser.add_argument('--w_tf_loss_accel', type=float, default=1.0)
    parser.add_argument('--w_tf_loss_bone', type=float, default=0.1, # <<< ADDED: Weight for Transformer bone length loss
                        help="Weight for Transformer model's bone length loss component.")
    parser.add_argument('--w_simple_loss_pose', type=float, default=1.0)
    parser.add_argument('--w_simple_loss_bone', type=float, default=0.1)

    parser.add_argument('--run_name', type=str, default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="A name for this training run")

    args = parser.parse_args()

    if args.model_type == 'transformer' and not (0.0 < args.val_split_ratio < 1.0):
        parser.error("--val_split_ratio must be between 0.0 and 1.0 (exclusive) for 'transformer' model type.")

    return args

def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    run_checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    log_file_path = os.path.join(args.log_dir, f"{args.run_name}_train.log")

    def log_message(message):
        print(message)
        with open(log_file_path, 'a') as f:
            f.write(message + '\n')

    log_message(f"Starting training run: {args.run_name}")
    log_message("Arguments: " + str(vars(args)))

    device = torch.device(args.device)
    num_joints = get_num_joints(args.skeleton_type)
    skeleton_parents_np = get_skeleton_parents(args.skeleton_type)

    smplh_to_smpl_body_indices = None
    if args.skeleton_type == 'smpl_24':
        smplh_to_smpl_body_indices = list(range(22)) + [20, 21]
        log_message(f"Using SMPL+H to smpl_24 joint mapping: {smplh_to_smpl_body_indices}")
        if len(smplh_to_smpl_body_indices) != num_joints:
            log_message(f"Error: smplh_to_smpl_body_indices length ({len(smplh_to_smpl_body_indices)}) "
                        f"does not match num_joints ({num_joints}) for skeleton_type '{args.skeleton_type}'. "
                        "Disabling joint selection.")
            smplh_to_smpl_body_indices = None
    else:
        log_message(f"Warning: skeleton_type is '{args.skeleton_type}'. Specific joint selection from SMPL+H is not applied.")

    # --- Dataset and Dataloader ---
    train_dataset, val_dataset = None, None
    # ... (Dataset loading logic for AMASS or Simple, remains mostly the same) ...
    if args.model_type == 'transformer':
        log_message(f"Loading AMASS dataset for Transformer model from root: {args.amass_root_dir}")
        if not os.path.isdir(args.amass_root_dir):
            log_message(f"Error: AMASS root directory not found: {args.amass_root_dir}")
            return

        all_npz_files = glob.glob(os.path.join(args.amass_root_dir, '**', '*.npz'), recursive=True)
        log_message(f"Found {len(all_npz_files)} .npz files in {args.amass_root_dir} and its subdirectories.")

        if not all_npz_files:
            log_message(f"Error: No .npz files found in {args.amass_root_dir}. Exiting.")
            return

        if args.split_seed is not None:
            random.seed(args.split_seed)
        random.shuffle(all_npz_files)

        split_idx = int(len(all_npz_files) * (1 - args.val_split_ratio))
        train_files = all_npz_files[:split_idx]
        val_files = all_npz_files[split_idx:]

        if not train_files:
            log_message("Error: No files allocated for training after split. Check val_split_ratio or dataset size.")
            return
        if not val_files:
            log_message("Warning: No files allocated for validation after split. Consider a smaller val_split_ratio or larger dataset.")

        log_message(f"Splitting AMASS data: {len(train_files)} training files, {len(val_files)} validation files.")

        train_dataset = AMASSSubsetDataset(
            data_paths=train_files, window_size=args.window_size,
            skeleton_type=args.skeleton_type,
            is_train=True,
            center_around_root=args.center_around_root_amass,
            joint_selector_indices=smplh_to_smpl_body_indices,
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
        if val_files:
            val_dataset = AMASSSubsetDataset(
                data_paths=val_files, window_size=args.window_size,
                skeleton_type=args.skeleton_type,
                is_train=False,
                center_around_root=args.center_around_root_amass,
                joint_selector_indices=smplh_to_smpl_body_indices,
                gaussian_noise_std=0.0,
                temporal_noise_type='none',
                temporal_noise_scale=0.0,
                outlier_prob=0.0,
                bonelen_noise_scale=0.0
            )
        else:
            val_dataset = None
    # ... (Simple model dataset loading - unchanged) ...
    elif args.model_type == 'simple':
        # ... (your existing simple model dataset loading code) ...
        log_message("Loading PoseSequenceDataset for Simple model...")
        def load_sequences_from_paths(paths):
            loaded_seqs = []
            if paths is None: return loaded_seqs
            for p in paths:
                try:
                    seq = np.load(p)
                    if seq.ndim == 2 and seq.shape[1] == num_joints * 3:
                        loaded_seqs.append(seq.astype(np.float32))
                    else:
                        log_message(f"Warning: Skipping file {p}, unexpected shape {seq.shape}. Expected (T, {num_joints*3})")
                except Exception as e:
                    log_message(f"Warning: Could not load sequence from {p}: {e}")
            return loaded_seqs

        train_seqs_np = load_sequences_from_paths(args.simple_data_paths_train)
        val_seqs_np = load_sequences_from_paths(args.simple_data_paths_val)

        if not train_seqs_np:
            log_message("Error: No training sequences loaded for simple model. Exiting.")
            return
        
        train_dataset = PoseSequenceDataset(
            sequences_np=train_seqs_np, window_size=args.window_size,
            num_joints=num_joints, noise_std=args.gaussian_noise_std_train, is_train=True
        )
        if val_seqs_np:
            val_dataset = PoseSequenceDataset(
                sequences_np=val_seqs_np, window_size=args.window_size,
                num_joints=num_joints, noise_std=0.0, is_train=False
            )
        else:
            val_dataset = None


    if train_dataset is None or len(train_dataset) == 0:
         log_message("Error: Training dataset is empty. Exiting.")
         return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        log_message(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    else:
        log_message(f"Training dataset size: {len(train_dataset)}, Validation dataset is empty or not created.")


    # --- Model and Loss Criteria ---
    model_constructor_args = {'model_type': args.model_type, 'num_joints': num_joints,
                              'window_size': args.window_size, 'skeleton_type': args.skeleton_type}
    criterion_bone = None # <<< ADDED: Initialize criterion_bone

    if args.model_type == 'transformer':
        model = ManifoldRefinementTransformer(
            num_joints=num_joints, # joint_dim will be 3 internally for R3J
            joint_dim=3,
            window_size=args.window_size,
            d_model=args.d_model_transformer, nhead=args.nhead_transformer,
            num_encoder_layers=args.num_encoder_layers_transformer,
            num_decoder_layers=args.num_decoder_layers_transformer,
            dim_feedforward=args.dim_feedforward_transformer,
            dropout=args.dropout_transformer,
            smpl_parents=skeleton_parents_np.tolist(),
            use_quaternions=args.use_quaternions_transformer
        ).to(device)
        criterion_pose = nn.L1Loss().to(device) # Or PositionMSELoss if you prefer
        criterion_vel = VelocityLoss(loss_type='l1').to(device)
        criterion_accel = AccelerationLoss(loss_type='l1').to(device)
        criterion_bone = BoneLengthMSELoss(parents_list=skeleton_parents_np.tolist()).to(device) # <<< ADDED: Init for Transformer
        model_constructor_args.update({
            'joint_dim': 3, 'd_model': args.d_model_transformer, 'nhead': args.nhead_transformer,
            'num_encoder_layers': args.num_encoder_layers_transformer,
            'num_decoder_layers': args.num_decoder_layers_transformer,
            'dim_feedforward': args.dim_feedforward_transformer, 'dropout': args.dropout_transformer,
            'smpl_parents': skeleton_parents_np.tolist(),
            'use_quaternions': args.use_quaternions_transformer,
            # 'center_around_root': args.center_around_root_amass # This is a dataset param, not model constr.
        })

    elif args.model_type == 'simple':
        fk_module_instance = ForwardKinematics( # Ensure FK init matches your FK class
            parents_list=skeleton_parents_np.tolist(),
            rest_directions_dict_or_tensor=get_rest_directions_tensor(args.skeleton_type, use_placeholder=True)
        ).to(device)

        model = PoseRefinerSimple(
            num_joints=num_joints, window_size=args.window_size,
            fk_module=fk_module_instance,
            d_model=args.d_model_simple, nhead=args.nhead_simple,
            num_encoder_layers=args.num_encoder_layers_simple,
            dim_feedforward=args.dim_feedforward_simple,
            dropout=args.dropout_simple
        ).to(device)
        criterion_pose = PositionMSELoss().to(device)
        criterion_bone = BoneLengthMSELoss(parents_list=skeleton_parents_np.tolist()).to(device) # Already here for simple
        model_constructor_args.update({
            'd_model': args.d_model_simple, 'nhead': args.nhead_simple,
            'num_encoder_layers': args.num_encoder_layers_simple,
            'dim_feedforward': args.dim_feedforward_simple, 'dropout': args.dropout_simple,
        })
    else:
        log_message(f"Error: Unsupported model type: {args.model_type}")
        raise ValueError(f"Unsupported model type: {args.model_type}")

    log_message(f"Training model: {args.model_type} on device: {device} with {num_joints} joints.")
    log_message(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_train_loss_epoch = 0
        total_train_loss_p_epoch = 0 # <<< ADDED for detailed logging
        total_train_loss_v_epoch = 0 # <<< ADDED
        total_train_loss_a_epoch = 0 # <<< ADDED
        total_train_loss_b_epoch = 0 # <<< ADDED

        if args.model_type == 'transformer':
            # Dataloader for AMASS returns: noisy_seq, clean_seq, bone_offsets_at_rest
            for batch_idx, (noisy_seq, clean_seq, bone_offsets_at_rest) in enumerate(train_loader): # <<< MODIFIED: var name bone_offsets -> bone_offsets_at_rest
                noisy_seq = noisy_seq.to(device)
                clean_seq = clean_seq.to(device) # clean_seq shape is (B, S, J, 3)
                bone_offsets_at_rest = bone_offsets_at_rest.to(device) # shape (B, J, 3)
                
                optimizer.zero_grad()
                
                # <<< MODIFIED: Model now returns two outputs >>>
                refined_seq, predicted_bone_lengths_seq = model(noisy_seq, bone_offsets_at_rest)
                # predicted_bone_lengths_seq shape is (B, S, J)
                
                loss_p = criterion_pose(refined_seq, clean_seq)
                loss_v = criterion_vel(refined_seq, clean_seq)
                loss_a = criterion_accel(refined_seq, clean_seq)

                # <<< ADDED: Calculate bone length loss >>>
                target_canonical_bone_lengths = torch.norm(bone_offsets_at_rest, dim=-1) # Shape (B, J)
                loss_b = criterion_bone(
                    predicted_bone_lengths_seq, 
                    target_canonical_bone_lengths, 
                    target_is_canonical_lengths=True # Use the flag from modified BoneLengthMSELoss
                )
                
                loss = args.w_tf_loss_pose * loss_p + \
                       args.w_tf_loss_vel * loss_v + \
                       args.w_tf_loss_accel * loss_a + \
                       args.w_tf_loss_bone * loss_b # <<< MODIFIED: Added bone loss with its weight
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_train_loss_epoch += loss.item()
                total_train_loss_p_epoch += loss_p.item() # <<< ADDED
                total_train_loss_v_epoch += loss_v.item() # <<< ADDED
                total_train_loss_a_epoch += loss_a.item() # <<< ADDED
                total_train_loss_b_epoch += loss_b.item() # <<< ADDED

                if batch_idx % 50 == 0:
                    log_message(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f} "
                                f"(P: {loss_p.item():.4f} V: {loss_v.item():.4f} A: {loss_a.item():.4f} B: {loss_b.item():.4f})") # <<< MODIFIED: Log B

        elif args.model_type == 'simple':
            # ... (simple model training loop - unchanged) ...
            for batch_idx, (noisy_window_flat, target_center_flat) in enumerate(train_loader):
                noisy_window_flat, target_center_flat = noisy_window_flat.to(device), target_center_flat.to(device)
                optimizer.zero_grad()
                pred_positions, pred_bone_lengths = model(noisy_window_flat)
                target_center_positions_3d = target_center_flat.view(-1, num_joints, 3) # B*S_center, J, 3
                loss_p = criterion_pose(pred_positions, target_center_positions_3d)
                loss_b = criterion_bone(pred_bone_lengths, target_center_positions_3d) # Existing BoneLengthMSELoss takes (B,J) and (B,J,3)
                loss = args.w_simple_loss_pose * loss_p + args.w_simple_loss_bone * loss_b
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_train_loss_epoch += loss.item()
                if batch_idx % 50 == 0:
                    log_message(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f} (Pose: {loss_p.item():.4f} Bone: {loss_b.item():.4f})")


        avg_train_loss_epoch = total_train_loss_epoch / len(train_loader)
        if args.model_type == 'transformer': # <<< ADDED: Detailed average loss logging
            avg_train_p = total_train_loss_p_epoch / len(train_loader)
            avg_train_v = total_train_loss_v_epoch / len(train_loader)
            avg_train_a = total_train_loss_a_epoch / len(train_loader)
            avg_train_b = total_train_loss_b_epoch / len(train_loader)
            log_message(f"Epoch {epoch}/{args.num_epochs} | Avg Train Loss: {avg_train_loss_epoch:.4f} "
                        f"(P: {avg_train_p:.4f} V: {avg_train_v:.4f} A: {avg_train_a:.4f} B: {avg_train_b:.4f})")
        else:
            log_message(f"Epoch {epoch}/{args.num_epochs} | Average Training Loss: {avg_train_loss_epoch:.4f}")


        # --- Validation ---
        if val_loader is None:
            log_message(f"Epoch {epoch} | No validation data. Skipping validation.")
            if avg_train_loss_epoch < best_val_loss:
                best_val_loss = avg_train_loss_epoch
                checkpoint_name = f"model_epoch_{epoch:03d}_trainloss_{avg_train_loss_epoch:.4f}.pth"
                # ... (save checkpoint) ...
                checkpoint_path = os.path.join(run_checkpoint_dir, checkpoint_name)
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss_epoch,
                    'args': vars(args),
                    'model_constructor_args': model_constructor_args
                }, checkpoint_path)
                log_message(f"Best model checkpoint (based on train loss) saved to {checkpoint_path}")

            continue

        model.eval()
        total_val_loss_epoch = 0
        total_val_mpjpe_epoch = 0
        total_val_loss_p_epoch = 0 # <<< ADDED for detailed logging
        total_val_loss_v_epoch = 0 # <<< ADDED
        total_val_loss_a_epoch = 0 # <<< ADDED
        total_val_loss_b_epoch = 0 # <<< ADDED

        with torch.no_grad():
            if args.model_type == 'transformer':
                for noisy_seq_val, clean_seq_val, bone_offsets_at_rest_val in val_loader: # <<< MODIFIED: var name
                    noisy_seq_val = noisy_seq_val.to(device)
                    clean_seq_val = clean_seq_val.to(device)
                    bone_offsets_at_rest_val = bone_offsets_at_rest_val.to(device) # <<< MODIFIED: var name
                    
                    # <<< MODIFIED: Model now returns two outputs >>>
                    refined_seq_val, predicted_bone_lengths_seq_val = model(noisy_seq_val, bone_offsets_at_rest_val)
                    
                    loss_p_val = criterion_pose(refined_seq_val, clean_seq_val)
                    loss_v_val = criterion_vel(refined_seq_val, clean_seq_val)
                    loss_a_val = criterion_accel(refined_seq_val, clean_seq_val)

                    # <<< ADDED: Calculate bone length loss for validation >>>
                    target_canonical_bone_lengths_val = torch.norm(bone_offsets_at_rest_val, dim=-1)
                    loss_b_val = criterion_bone(
                        predicted_bone_lengths_seq_val,
                        target_canonical_bone_lengths_val,
                        target_is_canonical_lengths=True
                    )
                    
                    val_loss_batch = args.w_tf_loss_pose * loss_p_val + \
                                     args.w_tf_loss_vel * loss_v_val + \
                                     args.w_tf_loss_accel * loss_a_val + \
                                     args.w_tf_loss_bone * loss_b_val # <<< MODIFIED: Added bone loss
                                     
                    total_val_loss_epoch += val_loss_batch.item() * noisy_seq_val.size(0)
                    mpjpe_batch = torch.norm(refined_seq_val - clean_seq_val, dim=(-1,-2)).mean() # MPJPE over joints and then mean over frames/batch
                    total_val_mpjpe_epoch += mpjpe_batch.item() * noisy_seq_val.size(0)

                    total_val_loss_p_epoch += loss_p_val.item() * noisy_seq_val.size(0) # <<< ADDED
                    total_val_loss_v_epoch += loss_v_val.item() * noisy_seq_val.size(0) # <<< ADDED
                    total_val_loss_a_epoch += loss_a_val.item() * noisy_seq_val.size(0) # <<< ADDED
                    total_val_loss_b_epoch += loss_b_val.item() * noisy_seq_val.size(0) # <<< ADDED


            elif args.model_type == 'simple':
                # ... (simple model validation loop - unchanged for its bone loss calculation) ...
                for noisy_window_flat_val, target_center_flat_val in val_loader:
                    noisy_window_flat_val, target_center_flat_val = noisy_window_flat_val.to(device), target_center_flat_val.to(device)
                    pred_positions_val, pred_bone_lengths_val = model(noisy_window_flat_val)
                    target_center_positions_3d_val = target_center_flat_val.view(-1, num_joints, 3)
                    loss_p_val = criterion_pose(pred_positions_val, target_center_positions_3d_val)
                    loss_b_val = criterion_bone(pred_bone_lengths_val, target_center_positions_3d_val) # Uses original mode of BoneLengthMSELoss
                    val_loss_batch = args.w_simple_loss_pose * loss_p_val + args.w_simple_loss_bone * loss_b_val
                    total_val_loss_epoch += val_loss_batch.item() * noisy_window_flat_val.size(0)
                    mpjpe_batch = torch.norm(pred_positions_val - target_center_positions_3d_val, dim=(-1,-2)).mean()
                    total_val_mpjpe_epoch += mpjpe_batch.item() * noisy_window_flat_val.size(0)


        num_val_samples = len(val_dataset) if val_dataset else 0
        if num_val_samples == 0:
             log_message(f"Epoch {epoch} | Validation dataset is empty. Cannot compute average validation loss.")
             continue

        avg_val_loss = total_val_loss_epoch / num_val_samples
        avg_val_mpjpe = (total_val_mpjpe_epoch / num_val_samples) * 1000 # Convert to mm

        if args.model_type == 'transformer': # <<< ADDED: Detailed average val loss logging
            avg_val_p = total_val_loss_p_epoch / num_val_samples
            avg_val_v = total_val_loss_v_epoch / num_val_samples
            avg_val_a = total_val_loss_a_epoch / num_val_samples
            avg_val_b = total_val_loss_b_epoch / num_val_samples
            log_message(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} MPJPE: {avg_val_mpjpe:.2f}mm "
                        f"(P: {avg_val_p:.4f} V: {avg_val_v:.4f} A: {avg_val_a:.4f} B: {avg_val_b:.4f})")
        else:
            log_message(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f} | Val MPJPE: {avg_val_mpjpe:.2f} mm")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # <<< MODIFIED: Include individual val losses in checkpoint if desired, e.g. for transformer >>>
            checkpoint_payload = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss, 'val_mpjpe': avg_val_mpjpe,
                'args': vars(args),
                'model_constructor_args': model_constructor_args
            }
            if args.model_type == 'transformer':
                checkpoint_payload['val_loss_components'] = {
                    'P': avg_val_p, 'V': avg_val_v, 'A': avg_val_a, 'B': avg_val_b
                }
                checkpoint_name = f"model_epoch_{epoch:03d}_valloss_{avg_val_loss:.4f}_mpjpe_{avg_val_mpjpe:.2f}_B_{avg_val_b:.4f}.pth"
            else:
                checkpoint_name = f"model_epoch_{epoch:03d}_valloss_{avg_val_loss:.4f}_mpjpe_{avg_val_mpjpe:.2f}.pth"
            
            checkpoint_path = os.path.join(run_checkpoint_dir, checkpoint_name)
            torch.save(checkpoint_payload, checkpoint_path)
            log_message(f"Best model checkpoint saved to {checkpoint_path}")

    log_message(f"Training finished for run: {args.run_name}")

if __name__ == "__main__":
    args = parse_args()
    main(args)