# src/kinematics/forward_kinematics.py
import torch
import torch.nn as nn
from .skeleton_utils import get_rest_directions_tensor

class ForwardKinematics(nn.Module):
    def __init__(self, parents_list, rest_directions_dict_or_tensor, skeleton_type_for_default_tensor='smpl_24'):
        super(ForwardKinematics, self).__init__()
        
        self.parents = torch.tensor(parents_list, dtype=torch.long)
        self.num_joints = len(parents_list)

        if isinstance(rest_directions_dict_or_tensor, dict):
            _rest_dirs_tensors = []
            for i in range(self.num_joints):
                if i in rest_directions_dict_or_tensor:
                    _rest_dirs_tensors.append(torch.tensor(rest_directions_dict_or_tensor[i], dtype=torch.float32))
                else:
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

        self.children_map = {i: [] for i in range(self.num_joints)}
        for j_idx, p_idx in enumerate(self.parents):
            if p_idx != -1:
                if p_idx.item() < self.num_joints:
                    self.children_map[p_idx.item()].append(j_idx)
                else:
                    print(f"Warning: Parent index {p_idx.item()} for joint {j_idx} is out of bounds for num_joints {self.num_joints}")
        
        self._bfs_order = []
        q = [0]
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
        batch_size = root_position.shape[0]
        device = root_position.device
        
        root_orient_norm = root_orientation_quat / (torch.norm(root_orientation_quat, dim=-1, keepdim=True) + 1e-8)
        local_rots_norm = local_joint_rotations_quat / (torch.norm(local_joint_rotations_quat, dim=-1, keepdim=True) + 1e-8)

        global_positions = torch.zeros(batch_size, self.num_joints, 3, device=device, dtype=root_position.dtype)
        global_orientations_quat = torch.zeros(batch_size, self.num_joints, 4, device=device, dtype=root_orientation_quat.dtype)
        global_positions[:, 0, :] = root_position
        global_orientations_quat[:, 0, :] = root_orient_norm

        for joint_idx in self._bfs_order:
            if self.parents[joint_idx] == -1:
                continue

            parent_idx = self.parents[joint_idx].item()
            parent_global_orient_q = global_orientations_quat[:, parent_idx, :].clone()
            parent_global_pos = global_positions[:, parent_idx, :]

            child_local_rot_q = local_rots_norm[:, joint_idx, :]
            rest_dir_vec_for_child = self.rest_directions[joint_idx].unsqueeze(0).expand(batch_size, -1).to(device) # (B, 3)
            current_bone_length = bone_lengths[:, joint_idx].unsqueeze(-1) # (B, 1)

            qw_p, qx_p, qy_p, qz_p = parent_global_orient_q.unbind(dim=-1)
            qw_c, qx_c, qy_c, qz_c = child_local_rot_q.unbind(dim=-1)
            new_qw = qw_p * qw_c - qx_p * qx_c - qy_p * qy_c - qz_p * qz_c
            new_qx = qw_p * qx_c + qx_p * qw_c + qy_p * qz_c - qz_p * qy_c
            new_qy = qw_p * qy_c - qx_p * qz_c + qy_p * qw_c + qz_p * qx_c
            new_qz = qw_p * qz_c + qx_p * qy_c - qy_p * qx_c + qz_p * qw_c
            
            calculated_global_orientation_for_joint = torch.stack(
                [new_qw, new_qx, new_qy, new_qz], dim=-1
            )
            
            global_orientations_quat[:, joint_idx, :] = calculated_global_orientation_for_joint
            q_vec_parent = parent_global_orient_q[:, 1:] # (B, 3)
            q_scalar_parent = parent_global_orient_q[:, 0:1] # (B, 1)
            
            t_original = 2 * torch.cross(q_vec_parent, rest_dir_vec_for_child, dim=1)
            rotated_offset_unit = rest_dir_vec_for_child + q_scalar_parent * t_original + torch.cross(q_vec_parent, t_original, dim=1)

            offset_in_world = rotated_offset_unit * current_bone_length
            global_positions[:, joint_idx, :] = parent_global_pos + offset_in_world
            
        return global_positions