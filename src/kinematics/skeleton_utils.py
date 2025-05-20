# src/kinematics/skeleton_utils.py
import torch
import numpy as np

# Standard SMPL-like skeleton (24 joints)
SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)
SMPL_NUM_JOINTS = len(SMPL_PARENTS)
SMPL_REST_DIRECTIONS_DICT = {
    1: [0.0, -1.0, 0.0], 2: [0.0, -1.0, 0.0], 3: [0.0, 1.0, 0.0],
    4: [0.0, -1.0, 0.0], 5: [0.0, -1.0, 0.0], 6: [0.0, 1.0, 0.0],
    7: [0.0, -1.0, 0.0], 8: [0.0, -1.0, 0.0], 9: [0.0, 1.0, 0.0],
    10: [0.0, 0.0, -1.0], 11: [0.0, 0.0, -1.0], 12: [0.0, 1.0, 0.0],
    13: [1.0, 0.0, 0.0], 14: [-1.0, 0.0, 0.0], 15: [0.0, 0.0, 0.0],
    16: [1.0, 0.0, 0.0], 17: [-1.0, 0.0, 0.0], 18: [1.0, 0.0, 0.0],
    19: [-1.0, 0.0, 0.0], 20: [1.0, 0.0, 0.0], 21: [-1.0, 0.0, 0.0],
    22: [1.0, 0.0, 0.0], 23: [-1.0, 0.0, 0.0]
}
SMPL_REST_DIRECTIONS_TENSOR = torch.zeros((SMPL_NUM_JOINTS, 3), dtype=torch.float32)
for i in range(SMPL_NUM_JOINTS):
    if i in SMPL_REST_DIRECTIONS_DICT:
        SMPL_REST_DIRECTIONS_TENSOR[i] = torch.tensor(SMPL_REST_DIRECTIONS_DICT[i], dtype=torch.float32)

SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR = torch.zeros((SMPL_NUM_JOINTS, 3), dtype=torch.float32)

for i in range(1, SMPL_NUM_JOINTS):
    parent = SMPL_PARENTS[i]
    if parent != -1:
        if i in [1, 2, 4, 5, 7, 8]: # Legs
            SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR[i] = torch.tensor([0.0, -0.3, 0.0])
        elif i in [18, 19, 20, 21, 22, 23]: # Arms
             SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR[i] = torch.tensor([0.3, 0.0, 0.0])
        elif i in [3, 6, 9, 12]: # Spine, Neck, Head
             SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR[i] = torch.tensor([0.0, 0.3, 0.0])
        else: # Collars etc.
             SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR[i] = torch.tensor([0.1, 0.1, 0.0])


def get_skeleton_parents(skeleton_type='smpl_24'):
    if skeleton_type == 'smpl' or skeleton_type == 'smpl_24':
        return SMPL_PARENTS.copy()
    else:
        raise ValueError(f"Unknown skeleton type: {skeleton_type}")

def get_num_joints(skeleton_type='smpl_24'):
    if skeleton_type == 'smpl' or skeleton_type == 'smpl_24':
        return SMPL_NUM_JOINTS
    else:
        raise ValueError(f"Unknown skeleton type: {skeleton_type}")

def get_rest_directions_dict(skeleton_type='smpl_24'):
    if skeleton_type == 'smpl' or skeleton_type == 'smpl_24':
        return SMPL_REST_DIRECTIONS_DICT.copy()
    else:
        raise ValueError(f"Unknown skeleton type: {skeleton_type}")

def get_rest_directions_tensor(skeleton_type='smpl_24', use_placeholder=False):
    if skeleton_type == 'smpl' or skeleton_type == 'smpl_24':
        if use_placeholder:
            print("Warning: Using PLACEHOLDER rest directions for FK. Results may be inaccurate.")
            return SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR.clone()
        return SMPL_REST_DIRECTIONS_TENSOR.clone()
    else:
        raise ValueError(f"Unknown skeleton type: {skeleton_type}")