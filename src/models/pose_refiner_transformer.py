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
from src.models.components.positional_encoding import PositionalEncoding
import torch.nn.functional as F
from src.kinematics.skeleton_utils import get_rest_directions_tensor, get_skeleton_parents

class ManifoldRefinementTransformer(nn.Module):
    def __init__(self, num_joints, joint_dim, window_size,
                 d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1,
                 smpl_parents=None,
                 use_quaternions=True,
                 skeleton_type='smpl_24',
                 predict_covariance=True,
                 center_around_root_amass = True):
        super().__init__()
        self.num_joints = num_joints
        self.window_size = window_size
        self.d_model = d_model
        self.use_quaternions = use_quaternions
        self.skeleton_type = skeleton_type
        self.predict_covariance = predict_covariance
        self.input_embedding = nn.Linear(num_joints * 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=window_size, batch_first=False)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.decoder_query_embed = nn.Embedding(window_size, d_model)

        self.num_orient_params = 4 if use_quaternions else 3
        self.dim_root_trans = 3
        self.dim_root_orient = self.num_orient_params
        self.dim_joint_rotations = self.num_joints * self.num_orient_params
        self.dim_bone_lengths = self.num_joints

        self.total_pose_and_length_params = self.dim_root_trans + \
                                            self.dim_root_orient + \
                                            self.dim_joint_rotations + \
                                            self.dim_bone_lengths
                                            
        self.output_head_pose = nn.Linear(d_model, self.total_pose_and_length_params)
        
        self.dim_covariance_params_per_joint = 6 # L11, L21, L22, L31, L32, L33
        self.total_covariance_params_dim = 0
        if self.predict_covariance:
            self.total_covariance_params_dim = self.num_joints * self.dim_covariance_params_per_joint
            self.output_head_covariance = nn.Linear(d_model, self.total_covariance_params_dim)

        if smpl_parents is None:
            loaded_parents = get_skeleton_parents(self.skeleton_type)
            self.smpl_parents = loaded_parents.tolist()
        else:
            self.smpl_parents = smpl_parents if isinstance(smpl_parents, list) else smpl_parents.tolist()
        
        standard_rest_dirs = get_rest_directions_tensor(self.skeleton_type, use_placeholder=False)
        if standard_rest_dirs.shape[0] != self.num_joints:
            raise ValueError(f"Standard rest directions for skeleton_type '{self.skeleton_type}' "
                             f"has {standard_rest_dirs.shape[0]} joints, but model is configured for {self.num_joints} joints.")
        self.register_buffer('standard_unit_rest_dirs', standard_rest_dirs)
        

    def forward(self, noisy_r3j_seq):
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

        # Embed Input
        embedded_input = noisy_r3j_seq.reshape(batch_size, seq_len, -1) # (B, S, J*3)
        embedded_input = self.input_embedding(embedded_input) # (B, S, d_model)

        seq_first_embedded_input = embedded_input.transpose(0, 1) # (S, B, d_model)
        seq_first_embedded_input = self.pos_encoder(seq_first_embedded_input) # Output: (S, B, d_model)

        memory = self.transformer_encoder(seq_first_embedded_input) # Output: (S, B, d_model)
        tgt_queries = self.decoder_query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) # (S, B, d_model) - standard query init

        seq_first_tgt_queries = self.pos_encoder(tgt_queries) # Output: (S, B, d_model)
        decoder_output = self.transformer_decoder(seq_first_tgt_queries, memory) # Output: (S, B, d_model)

        batch_first_decoder_output = decoder_output.transpose(0, 1) # (B, S, d_model)
        pose_params = self.output_head_pose(batch_first_decoder_output) # (B, S, pose_params_dim)
        current_offset = 0
        
        root_trans = pose_params[..., current_offset : current_offset + self.dim_root_trans]
        current_offset += self.dim_root_trans
        root_orient_params = pose_params[..., current_offset : current_offset + self.dim_root_orient]
        current_offset += self.dim_root_orient
        joint_rotations_params_flat = pose_params[..., current_offset : current_offset + self.dim_joint_rotations]
        current_offset += self.dim_joint_rotations
        
        raw_predicted_bone_lengths = pose_params[..., current_offset : current_offset + self.dim_bone_lengths] # (B, S, J)
        predicted_bone_lengths = F.softplus(raw_predicted_bone_lengths) + 1e-6 
        
        joint_rotations_params = joint_rotations_params_flat.reshape(
            batch_size, seq_len, self.num_joints, self.num_orient_params
        )

        root_trans_flat = root_trans.reshape(-1, 3) # (B*S, 3)
        
        if self.use_quaternions:
            '''Previous Calculation
            root_orient_quat_flat = F.normalize(root_orient_params.reshape(-1, 4), p=2, dim=-1) # (B*S, 4)
            local_joint_rot_quat_flat = F.normalize(
                joint_rotations_params.reshape(-1, self.num_joints, 4), p=2, dim=-1 # Normalize per-joint quat
            ).reshape(-1, self.num_joints, 4) # (B*S, J, 4)
            '''
            root_orient_quat_flat = F.normalize(root_orient_params.reshape(-1, 4), p=2, dim=-1)
            local_joint_rot_quat_flat = F.normalize(
                joint_rotations_params.reshape(-1, self.num_joints, 4), p=2, dim=-1
            ).reshape(-1, self.num_joints, 4) 
        
        fk_bone_lengths = predicted_bone_lengths.reshape(-1, self.num_joints)
        
        fk_layer = ForwardKinematics(
            parents_list=self.smpl_parents,
            rest_directions_dict_or_tensor=self.standard_unit_rest_dirs
        )
        
        refined_r3j_flat = fk_layer(
            root_orientation_quat=root_orient_quat_flat,    # (B*S, 4)
            root_position=root_trans_flat,                  # (B*S, 3)
            local_joint_rotations_quat=local_joint_rot_quat_flat, # (B*S, J, 4)
            bone_lengths=fk_bone_lengths                    # (B*S, J)
        ) # Output: (B*S, J, 3)

        # Reshape back to (B, S, J, 3)
        refined_r3j_seq = refined_r3j_flat.reshape(batch_size, seq_len, self.num_joints, 3)
        
        cholesky_factors = None
        if self.predict_covariance:
            covariance_params_flat = self.output_head_covariance(batch_first_decoder_output) # (B, S, num_joints * 6)
            
            # (B, S, J, 6)
            covariance_params_per_joint = covariance_params_flat.reshape(
                batch_size, seq_len, self.num_joints, self.dim_covariance_params_per_joint
            )
            
            
            cholesky_L = torch.zeros(batch_size, seq_len, self.num_joints, 3, 3, device=noisy_r3j_seq.device)
            
            cholesky_L[..., 0, 0] = F.softplus(covariance_params_per_joint[..., 0]) + 1e-6
            cholesky_L[..., 1, 0] = covariance_params_per_joint[..., 1]                
            cholesky_L[..., 1, 1] = F.softplus(covariance_params_per_joint[..., 2]) + 1e-6
            cholesky_L[..., 2, 0] = covariance_params_per_joint[..., 3]                 
            cholesky_L[..., 2, 1] = covariance_params_per_joint[..., 4] 
            cholesky_L[..., 2, 2] = F.softplus(covariance_params_per_joint[..., 5]) + 1e-6
            
            cholesky_factors = cholesky_L # (B, S, J, 3, 3)

        return refined_r3j_seq, predicted_bone_lengths, cholesky_factors