# src/kinematics/forward_kinematics.py
import torch
import torch.nn as nn
# Assuming skeleton_utils.py is in the same directory or PYTHONPATH is set up
from .skeleton_utils import get_rest_directions_tensor # Or adjust import based on execution

class ForwardKinematics(nn.Module):
    def __init__(self, parents_list, rest_directions_dict_or_tensor, skeleton_type_for_default_tensor='smpl_24'):
        """
        Initializes the ForwardKinematics module.
        Args:
            parents_list (list or np.array): List of parent joint indices.
            rest_directions_dict_or_tensor (dict or torch.Tensor):
                - If dict: Maps joint_idx to its unit rest direction vector [x,y,z] from its parent in T-pose.
                - If torch.Tensor: A tensor of shape (num_joints, 3) with rest directions.
                                   Joint 0's direction is typically [0,0,0] if it's the root.
            skeleton_type_for_default_tensor (str): Used if rest_directions_dict_or_tensor is None,
                                                    to fetch a default tensor from skeleton_utils.
                                                    A warning will be issued.
        """
        super(ForwardKinematics, self).__init__()
        
        self.parents = torch.tensor(parents_list, dtype=torch.long)
        self.num_joints = len(parents_list)

        if isinstance(rest_directions_dict_or_tensor, dict):
            _rest_dirs_tensors = []
            for i in range(self.num_joints):
                if i in rest_directions_dict_or_tensor:
                    _rest_dirs_tensors.append(torch.tensor(rest_directions_dict_or_tensor[i], dtype=torch.float32))
                else: # e.g., for root or joints without defined rest_dir in this dict
                    _rest_dirs_tensors.append(torch.zeros(3, dtype=torch.float32))
            self.register_buffer('rest_directions', torch.stack(_rest_dirs_tensors)) # (J, 3)
        elif isinstance(rest_directions_dict_or_tensor, torch.Tensor):
            if rest_directions_dict_or_tensor.shape != (self.num_joints, 3):
                raise ValueError(f"Provided rest_directions tensor has shape {rest_directions_dict_or_tensor.shape}, "
                                 f"expected ({self.num_joints}, 3)")
            self.register_buffer('rest_directions', rest_directions_dict_or_tensor.clone())
        elif rest_directions_dict_or_tensor is None:
            print(f"WARNING: rest_directions_dict_or_tensor is None. "
                  f"Attempting to use default placeholder tensor for skeleton type '{skeleton_type_for_default_tensor}'. "
                  "FK results will likely be incorrect without accurate rest directions.")
            self.register_buffer('rest_directions', get_rest_directions_tensor(skeleton_type_for_default_tensor, use_placeholder=True))
        else:
            raise TypeError("rest_directions_dict_or_tensor must be a dict, torch.Tensor, or None.")

        # Precompute children map for efficient traversal (BFS)
        self.children_map = {i: [] for i in range(self.num_joints)}
        for j_idx, p_idx in enumerate(self.parents):
            if p_idx != -1: # If not the root joint
                if p_idx.item() < self.num_joints: # Basic sanity check
                    self.children_map[p_idx.item()].append(j_idx)
                else:
                    print(f"Warning: Parent index {p_idx.item()} for joint {j_idx} is out of bounds for num_joints {self.num_joints}")
        
        # Generate a BFS order for processing joints
        self._bfs_order = []
        q = [0] # Start with root joint
        visited = {0}
        head = 0
        while head < len(q):
            curr = q[head]
            head += 1
            self._bfs_order.append(curr)
            for child_node in self.children_map.get(curr, []):
                if child_node not in visited:
                    visited.add(child_node)
                    q.append(child_node)
        if len(self._bfs_order) != self.num_joints:
            print(f"Warning: BFS order ({len(self._bfs_order)}) does not include all joints ({self.num_joints}). Check parents list for disconnected components.")


    def forward(self, root_orientation_quat, root_position, local_joint_rotations_quat, bone_lengths):
        """
        Computes 3D joint positions from root pose, local joint rotations, and bone lengths.
        Args:
            root_orientation_quat (torch.Tensor): (Batch, 4) tensor of root orientation quaternions (w, x, y, z).
            root_position (torch.Tensor): (Batch, 3) tensor of root joint positions.
            local_joint_rotations_quat (torch.Tensor): (Batch, NumJoints, 4) tensor of local joint orientation
                                                       quaternions (w, x, y, z) relative to parent.
                                                       The root's local rotation (local_joint_rotations_quat[:, 0, :])
                                                       is typically an identity quaternion if root_orientation_quat
                                                       is already its global orientation. Or, it can be combined.
                                                       This FK expects local_joint_rotations_quat[:, child_idx, :] to be the child's rotation.
            bone_lengths (torch.Tensor): (Batch, NumJoints) tensor of bone lengths.
                                         bone_lengths[:, child_idx] is the length of the bone connecting parent to child.
                                         bone_lengths[:, 0] (for root) is typically unused or 0.
        Returns:
            torch.Tensor: (Batch, NumJoints, 3) tensor of calculated 3D joint positions.
        """
        batch_size = root_position.shape[0]
        device = root_position.device
        
        if torch.rand(1).item() < 0.001: # 极小概率打印
             print(f"--- FK Input Debug (ForwardKinematics.forward) ---")
             print(f"  root_orientation_quat shape: {root_orientation_quat.shape}") # Expected (B*S, 4)
             print(f"  root_position shape: {root_position.shape}")             # Expected (B*S, 3)
             print(f"  local_joint_rotations_quat shape: {local_joint_rotations_quat.shape}") # Expected (B*S, J, 4)
             print(f"  bone_lengths shape: {bone_lengths.shape}")                 # Expected (B*S, J)
             if batch_size > 0:
                 print(f"  root_orientation_quat sample [0]: {root_orientation_quat[0].detach().cpu().numpy()}")
                 print(f"  root_position sample [0]: {root_position[0].detach().cpu().numpy()}")
                 print(f"  local_joint_rotations_quat sample [0,0]: {local_joint_rotations_quat[0,0,:].detach().cpu().numpy()}")
                 print(f"  bone_lengths sample [0,0-5]: {bone_lengths[0,:6].detach().cpu().numpy()}")

        # Normalize input quaternions to be safe
        root_orient_norm = root_orientation_quat / (torch.norm(root_orientation_quat, dim=-1, keepdim=True) + 1e-8)
        local_rots_norm = local_joint_rotations_quat / (torch.norm(local_joint_rotations_quat, dim=-1, keepdim=True) + 1e-8)

        # Initialize output tensors
        global_positions = torch.zeros(batch_size, self.num_joints, 3, device=device, dtype=root_position.dtype)
        global_orientations_quat = torch.zeros(batch_size, self.num_joints, 4, device=device, dtype=root_orientation_quat.dtype)

        # Set root joint's global position and orientation
        global_positions[:, 0, :] = root_position
        global_orientations_quat[:, 0, :] = root_orient_norm # Global orientation of the root

        # Iterate through joints in BFS order (already computed global transform for parent)
        for joint_idx in self._bfs_order:
            if self.parents[joint_idx] == -1: # Root joint, already handled
                continue

            parent_idx = self.parents[joint_idx].item()
            
            # Get parent's global orientation and position
            parent_global_orient_q = global_orientations_quat[:, parent_idx, :].clone()
            parent_global_pos = global_positions[:, parent_idx, :]

            # Get child's local rotation and corresponding bone length and rest direction
            child_local_rot_q = local_rots_norm[:, joint_idx, :]
            # Rest direction of the bone vector in parent's local frame (pointing to child)
            # This should be a unit vector; bone_lengths provides the magnitude.
            rest_dir_vec_for_child = self.rest_directions[joint_idx].unsqueeze(0).expand(batch_size, -1).to(device) # (B, 3)
            current_bone_length = bone_lengths[:, joint_idx].unsqueeze(-1) # (B, 1)

            # Step 1: Calculate child's global orientation
            # Global_child_orient = Global_parent_orient * Local_child_orient (quaternion multiplication)
            qw_p, qx_p, qy_p, qz_p = parent_global_orient_q.unbind(dim=-1)
            qw_c, qx_c, qy_c, qz_c = child_local_rot_q.unbind(dim=-1)

            # Hamilton product (w, x, y, z) components
            new_qw = qw_p * qw_c - qx_p * qx_c - qy_p * qy_c - qz_p * qz_c
            new_qx = qw_p * qx_c + qx_p * qw_c + qy_p * qz_c - qz_p * qy_c # 原报错行对应的计算
            new_qy = qw_p * qy_c - qx_p * qz_c + qy_p * qw_c + qz_p * qx_c
            new_qz = qw_p * qz_c + qx_p * qy_c - qy_p * qx_c + qz_p * qw_c
            
            # 将计算出的新分量堆叠成一个新的四元数张量
            # new_qw, new_qx, new_qy, new_qz 的形状都是 (batch_size)
            # torch.stack([...], dim=-1) 会创建一个形状为 (batch_size, 4) 的张量
            calculated_global_orientation_for_joint = torch.stack(
                [new_qw, new_qx, new_qy, new_qz], dim=-1
            )
            
            # 将这个新计算的完整四元数张量赋给 global_orientations_quat 的相应切片
            global_orientations_quat[:, joint_idx, :] = calculated_global_orientation_for_joint
            
            # Step 2: Calculate child's global position
            # The rest_dir_vector is defined in parent's local coordinate system (if it's T-Pose like).
            # We need to rotate this vector by parent's global orientation.
            # offset_in_world_space = rotate_vector_by_quaternion(parent_global_orient_q, rest_dir_vec_for_child) * current_bone_length
            
            # Quaternion rotation of vector v by q: qvq* (where q* is conjugate)
            # More efficiently: v' = v + 2 * cross(q_vec, cross(q_vec, v) + q_scalar * v)
            q_vec_parent = parent_global_orient_q[:, 1:] # (B, 3)
            q_scalar_parent = parent_global_orient_q[:, 0:1] # (B, 1)
            
            # t = 2 * cross(q_vec_parent, rest_dir_vec_for_child)
            # rotated_offset_unit = rest_dir_vec_for_child + q_scalar_parent * t + torch.cross(q_vec_parent, t)
            # Your original kinematics had a slightly different but common formula for qvq*:
            t_original = 2 * torch.cross(q_vec_parent, rest_dir_vec_for_child, dim=1)
            rotated_offset_unit = rest_dir_vec_for_child + q_scalar_parent * t_original + torch.cross(q_vec_parent, t_original, dim=1)

            offset_in_world = rotated_offset_unit * current_bone_length
            global_positions[:, joint_idx, :] = parent_global_pos + offset_in_world
            
        return global_positions

if __name__ == '__main__':
    # Test the ForwardKinematics module
    batch_s = 2
    num_j = get_num_joints('smpl_24')
    parents = get_skeleton_parents('smpl_24')
    
    # Using placeholder rest directions from skeleton_utils
    rest_dirs_tensor = get_rest_directions_tensor('smpl_24', use_placeholder=True)
    
    fk = ForwardKinematics(parents_list=parents, rest_directions_dict_or_tensor=rest_dirs_tensor)
    print("BFS order for FK:", fk._bfs_order)

    # Dummy inputs
    root_q = torch.zeros(batch_s, 4)
    root_q[:, 0] = 1.0 # Identity quaternion (w,x,y,z)
    root_p = torch.zeros(batch_s, 3)
    
    local_q = torch.zeros(batch_s, num_j, 4)
    local_q[..., 0] = 1.0 # All local rotations are identity

    # Dummy bone lengths (e.g., all bones are length 1, except root's "bone")
    bone_ls = torch.ones(batch_s, num_j)
    bone_ls[:, 0] = 0.0 # Root has no incoming bone from a parent

    # Test with a slight rotation on one joint, e.g., joint 1 (left hip)
    # Rotate 90 degrees around Z-axis for joint 1 local rotation
    # angle = torch.tensor(torch.pi / 2.0)
    # axis = torch.tensor([0.0, 0.0, 1.0])
    # q_rot_joint1 = torch.cat([torch.cos(angle/2).unsqueeze(0), torch.sin(angle/2) * axis])
    # local_q[:, 1, :] = q_rot_joint1.unsqueeze(0).repeat(batch_s, 1)

    positions = fk(root_q, root_p, local_q, bone_ls)
    print("\nCalculated positions (all identity rotations, unit bone lengths, placeholder rest dirs):")
    for i in range(num_j):
        print(f"Joint {i}: {positions[0, i].numpy()}")

    # Example: If rest_directions[1] = [1,0,0] and bone_lengths[1]=1,
    # and all rotations are identity, joint 1 should be at [1,0,0] relative to root.
    # Given placeholder rest_directions[1] = [1.0, 0.0, 0.0]
    print(f"\nBased on placeholder rest_directions[1]={rest_dirs_tensor[1].numpy()}, "
          f"bone_lengths[:,1]=1, and identity rotations, Joint 1 position should be {rest_dirs_tensor[1].numpy()}.")
    print(f"Actual Joint 1 position from FK: {positions[0, 1].numpy()}")