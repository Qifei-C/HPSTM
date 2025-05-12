# src/models/pose_refiner_transformer.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.kinematics.forward_kinematics import ForwardKinematics
from src.kinematics.conversions import rodrigues_batch, quaternion_to_matrix #
# 确保有 matrix_to_quaternion 或 axis_angle_to_quaternion 如果 use_quaternions=False
# from src.kinematics.conversions import matrix_to_quaternion, axis_angle_to_quaternion # 假设这些存在
from src.models.components.positional_encoding import PositionalEncoding
import torch.nn.functional as F


class ManifoldRefinementTransformer(nn.Module):
    def __init__(self, num_joints, joint_dim, window_size,
                 d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1,
                 smpl_parents=None,
                 use_quaternions=True):
        super().__init__()
        self.num_joints = num_joints
        # joint_dim is the input dimension for each joint (e.g., 3 for R3J)
        # self.joint_dim = joint_dim # This was in your provided code, but input_embedding takes flattened
        self.window_size = window_size
        self.d_model = d_model
        self.use_quaternions = use_quaternions

        # Input embedding: num_joints * 3 (for R3J) -> d_model
        self.input_embedding = nn.Linear(num_joints * 3, d_model) # Assuming joint_dim is 3 for R3J input

        # Positional Encoding: Ensure batch_first consistency
        # Based on previous fixes, PositionalEncoding is batch_first=False,
        # and Transformer layers below are batch_first=False too (as per Solution 1).
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=window_size, batch_first=False)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False) # Changed to False
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False) # Changed to False
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.decoder_query_embed = nn.Embedding(window_size, d_model)

        self.num_orient_params = 4 if use_quaternions else 3
        
        # Define dimensions for each part of the pose
        self.dim_root_trans = 3
        self.dim_root_orient = self.num_orient_params
        self.dim_joint_rotations = self.num_joints * self.num_orient_params # Predicting for all joints including root
        self.dim_bone_lengths = self.num_joints # Predicting length for each bone originating from a joint

        self.total_pose_and_length_params = self.dim_root_trans + \
                                            self.dim_root_orient + \
                                            self.dim_joint_rotations + \
                                            self.dim_bone_lengths
                                            
        self.output_head = nn.Linear(d_model, self.total_pose_and_length_params)

        if smpl_parents is None:
            raise ValueError("smpl_parents must be provided for FK.")
        self.smpl_parents = smpl_parents
        # FK layer can be instantiated once if canonical rest directions are used,
        # or if rest directions are passed to its forward method.
        # Current FK init takes canonical rest_directions.

    def forward(self, noisy_r3j_seq, bone_offsets_at_rest):
        """
        Args:
            noisy_r3j_seq (torch.Tensor): (batch_size, window_size, num_joints, 3)
            bone_offsets_at_rest (torch.Tensor): (batch_size, num_joints, 3) T-pose FULL bone offset vectors
        Returns:
            refined_r3j_seq (torch.Tensor): (batch_size, window_size, num_joints, 3)
            predicted_bone_lengths_for_loss (torch.Tensor): (batch_size, window_size, num_joints) for loss calculation
        """
        batch_size, seq_len, num_j_input, _ = noisy_r3j_seq.shape
        if seq_len != self.window_size:
            raise ValueError(f"Input sequence length {seq_len} does not match model window_size {self.window_size}")
        if num_j_input != self.num_joints:
            raise ValueError(f"Input num_joints {num_j_input} does not match model num_joints {self.num_joints}")

        # 1. Embed Input
        embedded_input = noisy_r3j_seq.reshape(batch_size, seq_len, -1) # (B, S, J*3)
        embedded_input = self.input_embedding(embedded_input) # (B, S, d_model)

        # Transpose for batch_first=False PositionalEncoding and Transformer layers
        seq_first_embedded_input = embedded_input.transpose(0, 1) # (S, B, d_model)
        seq_first_embedded_input = self.pos_encoder(seq_first_embedded_input) # Output: (S, B, d_model)

        # 2. Transformer Encoder
        memory = self.transformer_encoder(seq_first_embedded_input) # Output: (S, B, d_model)

        # 3. Transformer Decoder
        tgt_queries = self.decoder_query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) # (S, B, d_model) - standard query init
        # Note: Original code had .unsqueeze(0).repeat(batch_size,1,1) which made it (B,S,D) then needed transpose.
        # If decoder_query_embed.weight is (window_size, d_model), then unsqueeze(1) -> (S,1,D) -> repeat to (S,B,D) is direct.

        seq_first_tgt_queries = self.pos_encoder(tgt_queries) # Apply pos encoding, output: (S, B, d_model)
        decoder_output = self.transformer_decoder(seq_first_tgt_queries, memory) # Output: (S, B, d_model)

        # Transpose back to (B, S, D) for the output head
        batch_first_decoder_output = decoder_output.transpose(0, 1) # (B, S, d_model)

        # 4. Output Head: Predict Pose Parameters and Bone Lengths
        all_params = self.output_head(batch_first_decoder_output) # (B, S, total_pose_and_length_params)

        # Split params
        current_offset = 0
        root_trans = all_params[..., current_offset : current_offset + self.dim_root_trans]
        current_offset += self.dim_root_trans
        root_orient_params = all_params[..., current_offset : current_offset + self.dim_root_orient]
        current_offset += self.dim_root_orient
        joint_rotations_params_flat = all_params[..., current_offset : current_offset + self.dim_joint_rotations]
        current_offset += self.dim_joint_rotations
        predicted_bone_lengths = all_params[..., current_offset : current_offset + self.dim_bone_lengths] # (B, S, J)
        
        # Ensure predicted bone lengths are positive
        predicted_bone_lengths = F.relu(predicted_bone_lengths) + 1e-6 

        # Reshape joint rotations
        joint_rotations_params = joint_rotations_params_flat.reshape(
            batch_size, seq_len, self.num_joints, self.num_orient_params
        )

        # 5. Prepare inputs for Forward Kinematics (per frame, as FK takes single pose inputs)
        # Reshape all per-frame params to be (B*S, ...)
        root_trans_flat = root_trans.reshape(-1, 3) # (B*S, 3)
        
        if self.use_quaternions:
            root_orient_quat_flat = F.normalize(root_orient_params.reshape(-1, 4), p=2, dim=-1) # (B*S, 4)
            local_joint_rot_quat_flat = F.normalize(
                joint_rotations_params.reshape(-1, self.num_joints, 4), p=2, dim=-1 # Normalize per-joint quat
            ).reshape(-1, self.num_joints, 4) # (B*S, J, 4)
        else:
            # Axis-angle to Quaternion conversion needed here
            # Placeholder - this will require a utility function
            # For example, convert axis-angle to matrix using rodrigues_batch, then matrix to quaternion
            # root_orient_mat_flat = rodrigues_batch(root_orient_params.reshape(-1, 3)) # (B*S, 3, 3)
            # joint_rot_mat_flat = rodrigues_batch(joint_rotations_params.reshape(-1, 3)).reshape(batch_size*seq_len, self.num_joints, 3, 3)
            # root_orient_quat_flat = matrix_to_quaternion(root_orient_mat_flat) # (B*S, 4)
            # local_joint_rot_quat_flat = matrix_to_quaternion(joint_rot_mat_flat.reshape(-1,3,3)).reshape(batch_size*seq_len, self.num_joints, 4)
            raise NotImplementedError("use_quaternions=False requires axis_angle_to_quaternion or matrix_to_quaternion, which is not readily available in your conversions.py")

        # Use predicted bone lengths for FK.
        # FK's bone_lengths argument is (Batch_fk, NumJoints). Our Batch_fk is B*S.
        # predicted_bone_lengths is (B, S, J). Reshape to (B*S, J).
        fk_bone_lengths = predicted_bone_lengths.reshape(-1, self.num_joints) # (B*S, J)

        # Prepare rest directions for FK initialization. These are canonical T-pose unit vectors.
        # We assume they are the same for all items in the batch for this single FK layer instance.
        # Taking from the first item in the batch.
        canonical_offsets = bone_offsets_at_rest[0] # (J, 3) - full offsets for the first subject
        canonical_unit_rest_dirs = F.normalize(canonical_offsets, p=2, dim=-1) # (J, 3) - unit vectors

        # Instantiate FK layer (once per forward pass is fine if rest dirs are canonical)
        fk_layer = ForwardKinematics(
            parents_list=self.smpl_parents,
            rest_directions_dict_or_tensor=canonical_unit_rest_dirs
        )
        
        # Perform batched FK for all frames in the batch
        refined_r3j_flat = fk_layer(
            root_orientation_quat=root_orient_quat_flat,    # (B*S, 4)
            root_position=root_trans_flat,                  # (B*S, 3)
            local_joint_rotations_quat=local_joint_rot_quat_flat, # (B*S, J, 4)
            bone_lengths=fk_bone_lengths                    # (B*S, J)
        ) # Output: (B*S, J, 3)

        # Reshape back to (B, S, J, 3)
        refined_r3j_seq = refined_r3j_flat.reshape(batch_size, seq_len, self.num_joints, 3)

        # Return both refined sequence and predicted lengths (for loss calculation)
        return refined_r3j_seq, predicted_bone_lengths