# src/models/pose_refiner_simple.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming forward_kinematics.py is in src.kinematics
# Adjust import if necessary based on how you run scripts
from ..kinematics.forward_kinematics import ForwardKinematics 

class PoseRefinerSimple(nn.Module):
    def __init__(self, num_joints, window_size, fk_module, 
                 d_model=96, nhead=8, num_encoder_layers=4, dim_feedforward=256, dropout=0.1):
        """
        Simple pose refiner model using a Transformer encoder.
        Refines the center frame of an input window of poses.
        Args:
            num_joints (int): Number of joints (J).
            window_size (int): Number of frames in the input window (L).
            fk_module (nn.Module): An instance of the ForwardKinematics module.
            d_model (int): Dimension of the Transformer model.
            nhead (int): Number of attention heads in the Transformer.
            num_encoder_layers (int): Number of layers in the Transformer encoder.
            dim_feedforward (int): Dimension of the feedforward network in Transformer.
            dropout (float): Dropout rate.
        """
        super(PoseRefinerSimple, self).__init__()
        self.J = num_joints
        self.window_size = window_size # L
        self.d_model = d_model
        
        self.input_proj = nn.Linear(num_joints * 3, d_model) # Input is (B, L, J*3)
        
        # Learnable positional embedding for the window sequence
        self.pos_emb = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.uniform_(self.pos_emb, -0.1, 0.1) # Initialize positional embeddings

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Important: input is (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Output heads from the features of the center frame's representation
        # Predicts root orientation (quat), local joint rotations (quat), and bone lengths
        self.root_orient_head = nn.Linear(d_model, 4) # Quaternion (w,x,y,z) for root
        self.local_rot_head = nn.Linear(d_model, num_joints * 4) # Quaternions for all J joints
        self.bone_len_head = nn.Linear(d_model, num_joints) # Scalar length for each of J bones/segments
        
        self.fk = fk_module # Store the passed FK module instance

        # Initialize output head biases and weights
        # For rotations, small weights initially to be close to identity if input features are zero
        nn.init.xavier_uniform_(self.root_orient_head.weight, gain=0.01)
        self.root_orient_head.bias.data.zero_()
        # Set bias for w component of quaternion to 1 for root_orient to encourage identity
        # self.root_orient_head.bias.data[0] = 1.0 # If predicting (w,x,y,z) and want identity

        nn.init.xavier_uniform_(self.local_rot_head.weight, gain=0.01)
        self.local_rot_head.bias.data.zero_()
        # For local_rot_head, initialize such that 'w' component of quaternions is favored for identity
        # for b in range(num_joints):
        #    self.local_rot_head.bias.data[b*4] = 1.0 # If predicting (w,x,y,z) and want identity

        # For bone lengths, initialize bias for softplus to output reasonable positive values (e.g., 1.0)
        # softplus(0) approx 0.69. To get ~1.0, bias should be log(exp(1)-1) approx 0.54
        # For now, using zero bias.
        nn.init.constant_(self.bone_len_head.bias, 0.0) 
        nn.init.xavier_uniform_(self.bone_len_head.weight, gain=0.01)

    def forward(self, pose_window_flat):
        """
        Forward pass of the model.
        Args:
            pose_window_flat (torch.Tensor): Input tensor of shape (Batch, WindowSize, NumJoints*3).
                                             This is a window of flattened 3D joint positions.
        Returns:
            pred_positions (torch.Tensor): Predicted 3D joint positions for the center frame,
                                           shape (Batch, NumJoints, 3).
            pred_bone_lengths (torch.Tensor): Predicted bone lengths for the center frame,
                                              shape (Batch, NumJoints).
        """
        B, L, _ = pose_window_flat.shape # L should be self.window_size
        if L != self.window_size:
            # This can happen if dataset doesn't strictly enforce window_size or during inference with padding
            # For simplicity here, we assume L == self.window_size for pos_emb.
            # A more robust solution might involve interpolating/slicing pos_emb or masking.
            # For now, we'll slice pos_emb if L is smaller, or raise error if L is larger.
            if L > self.window_size:
                 raise ValueError(f"Input window length {L} is greater than model's window_size {self.window_size}")
            pos_embedding_to_add = self.pos_emb[:, :L, :]
        else:
            pos_embedding_to_add = self.pos_emb

        x = self.input_proj(pose_window_flat) # (B, L, d_model)
        x = x + pos_embedding_to_add # Add positional embedding
        
        encoder_output = self.transformer_encoder(x) # (B, L, d_model)
        
        # Extract features for the center frame of the window
        center_frame_feature_idx = L // 2
        center_frame_features = encoder_output[:, center_frame_feature_idx, :] # (B, d_model)
        
        # Predict pose parameters from center frame features
        pred_root_orient_quat = self.root_orient_head(center_frame_features) # (B, 4)
        pred_local_rotations_flat = self.local_rot_head(center_frame_features) # (B, J*4)
        pred_bone_lengths_raw = self.bone_len_head(center_frame_features) # (B, J)

        # Reshape local rotations and normalize quaternions
        pred_local_rotations_quat = pred_local_rotations_flat.view(B, self.J, 4)
        # Normalize all predicted quaternions (root and local)
        pred_root_orient_quat = pred_root_orient_quat / (torch.norm(pred_root_orient_quat, dim=-1, keepdim=True) + 1e-8)
        pred_local_rotations_quat = pred_local_rotations_quat / (torch.norm(pred_local_rotations_quat, dim=-1, keepdim=True) + 1e-8)
        
        # Ensure bone lengths are positive using softplus
        pred_bone_lengths = F.softplus(pred_bone_lengths_raw) # (B, J)
        
        # The root position for FK is assumed to be the root position from the input center frame.
        # Input pose_window_flat is (B, L, J*3).
        # Center frame input: pose_window_flat[:, center_frame_feature_idx, :] -> (B, J*3)
        # Root position of center frame: first 3 values.
        input_center_frame_root_pos = pose_window_flat[:, center_frame_feature_idx, :3] # (B, 3)

        # Use FK to get 3D joint positions
        # The FK module expects:
        # root_orientation_quat (B, 4), root_position (B, 3),
        # local_joint_rotations_quat (B, J, 4), bone_lengths (B, J)
        pred_positions = self.fk(
            pred_root_orient_quat,
            input_center_frame_root_pos,
            pred_local_rotations_quat,
            pred_bone_lengths
        ) # Output: (B, J, 3)
        
        return pred_positions, pred_bone_lengths


if __name__ == '__main__':
    # Example Usage
    batch_s = 4
    num_j_example = 24
    window_s_example = 31
    d_model_example = 96

    # 1. Setup FK module
    parents_example = get_skeleton_parents('smpl_24')
    # CRITICAL: Use actual rest directions, not placeholder, for real results
    rest_dirs_example = get_rest_directions_tensor('smpl_24', use_placeholder=True) 
    fk_instance = ForwardKinematics(parents_list=parents_example, rest_directions_dict_or_tensor=rest_dirs_example)

    # 2. Instantiate the model
    model = PoseRefinerSimple(
        num_joints=num_j_example,
        window_size=window_s_example,
        fk_module=fk_instance,
        d_model=d_model_example
    )
    model.eval() # Set to evaluation mode for testing

    # 3. Create dummy input
    # (Batch, WindowSize, NumJoints*3)
    dummy_input_window = torch.randn(batch_s, window_s_example, num_j_example * 3)

    # 4. Forward pass
    with torch.no_grad():
        pred_pos, pred_bones = model(dummy_input_window)

    print("Model instantiated successfully.")
    print("Input shape:", dummy_input_window.shape)
    print("Predicted positions shape:", pred_pos.shape) # Expected: (Batch, NumJoints, 3)
    print("Predicted bone lengths shape:", pred_bones.shape) # Expected: (Batch, NumJoints)

    # Check a sample output
    print("\nSample predicted position (first joint, first batch item):", pred_pos[0, 0].numpy())
    print("Sample predicted bone length (first bone, first batch item):", pred_bones[0, 0].item())