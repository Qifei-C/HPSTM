# src/kinematics/skeleton_utils.py
import torch
import numpy as np

# Standard SMPL-like skeleton (24 joints)
SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)
SMPL_NUM_JOINTS = len(SMPL_PARENTS)

# Example rest directions (canonical T-pose offsets, normalized)
# These should ideally be derived from your specific dataset/T-pose.
SMPL_REST_DIRECTIONS_DICT = {
    1: [0.0, -1.0, 0.0], 2: [0.0, -1.0, 0.0], 3: [0.0, 1.0, 0.0],
    4: [0.0, -1.0, 0.0], 5: [0.0, -1.0, 0.0], 6: [0.0, 1.0, 0.0],
    7: [0.0, -1.0, 0.0], 8: [0.0, -1.0, 0.0], 9: [0.0, 1.0, 0.0],
    10: [0.0, 0.0, -1.0], 11: [0.0, 0.0, -1.0], 12: [0.0, 1.0, 0.0], # Head top often not direct bone
    13: [1.0, 0.0, 0.0], 14: [-1.0, 0.0, 0.0], 15: [0.0, 0.0, 0.0], # Placeholder
    16: [1.0, 0.0, 0.0], 17: [-1.0, 0.0, 0.0], 18: [1.0, 0.0, 0.0],
    19: [-1.0, 0.0, 0.0], 20: [1.0, 0.0, 0.0], 21: [-1.0, 0.0, 0.0],
    22: [1.0, 0.0, 0.0], 23: [-1.0, 0.0, 0.0]
}
SMPL_REST_DIRECTIONS_TENSOR = torch.zeros((SMPL_NUM_JOINTS, 3), dtype=torch.float32)
for i in range(SMPL_NUM_JOINTS):
    if i in SMPL_REST_DIRECTIONS_DICT: # Joint 0 (root) won't be in here
        SMPL_REST_DIRECTIONS_TENSOR[i] = torch.tensor(SMPL_REST_DIRECTIONS_DICT[i], dtype=torch.float32)

# Placeholder for rest directions if true values are not available
# This creates a very basic set of directions, mostly unit vectors along axes or zero
# It's better to have actual T-pose bone vectors for your specific skeleton.
SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR = torch.zeros((SMPL_NUM_JOINTS, 3), dtype=torch.float32)
# For non-root joints, set a default direction (e.g., along Y or X axis from parent)
# This is a very rough placeholder and might not be kinematically correct for a specific T-Pose.
# For ForwardKinematics to work meaningfully, these should be actual T-pose bone vectors (child_pos - parent_pos in T-pose).
for i in range(1, SMPL_NUM_JOINTS): # Skip root
    parent = SMPL_PARENTS[i]
    if parent != -1: # Should always be true for i > 0
        # Simplistic placeholder: if child of root hips, go down; if child of spine, go up etc.
        if i in [1, 2, 4, 5, 7, 8]: # Legs
            SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR[i] = torch.tensor([0.0, -0.3, 0.0]) # Small vector down
        elif i in [18, 19, 20, 21, 22, 23]: # Arms
             SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR[i] = torch.tensor([0.3, 0.0, 0.0]) # Small vector along X
        elif i in [3, 6, 9, 12]: # Spine, Neck, Head
             SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR[i] = torch.tensor([0.0, 0.3, 0.0]) # Small vector up
        else: # Collars etc.
             SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR[i] = torch.tensor([0.1, 0.1, 0.0])


def get_skeleton_parents(skeleton_type='smpl_24'): # Changed default for clarity
    if skeleton_type == 'smpl' or skeleton_type == 'smpl_24': # MODIFIED
        return SMPL_PARENTS.copy() # Return a copy
    # elif skeleton_type == 'h36m':
    #     return H36M_PARENTS
    else:
        raise ValueError(f"Unknown skeleton type: {skeleton_type}")

def get_num_joints(skeleton_type='smpl_24'): # Changed default for clarity
    if skeleton_type == 'smpl' or skeleton_type == 'smpl_24': # MODIFIED
        return SMPL_NUM_JOINTS
    # elif skeleton_type == 'h36m':
    #     return H36M_NUM_JOINTS
    else:
        raise ValueError(f"Unknown skeleton type: {skeleton_type}")

def get_rest_directions_dict(skeleton_type='smpl_24'): # Changed default for clarity
    if skeleton_type == 'smpl' or skeleton_type == 'smpl_24': # MODIFIED
        return SMPL_REST_DIRECTIONS_DICT.copy() # Return a copy
    else:
        raise ValueError(f"Unknown skeleton type: {skeleton_type}")

def get_rest_directions_tensor(skeleton_type='smpl_24', use_placeholder=False): # Changed default & added use_placeholder
    if skeleton_type == 'smpl' or skeleton_type == 'smpl_24': # MODIFIED
        if use_placeholder:
            print("Warning: Using PLACEHOLDER rest directions for FK. Results may be inaccurate.")
            return SMPL_REST_DIRECTIONS_PLACEHOLDER_TENSOR.clone()
        return SMPL_REST_DIRECTIONS_TENSOR.clone()
    else:
        raise ValueError(f"Unknown skeleton type: {skeleton_type}")

if __name__ == '__main__':
    print("SMPL Parents (smpl_24):", get_skeleton_parents('smpl_24'))
    print("SMPL Num Joints (smpl_24):", get_num_joints('smpl_24'))
    print("SMPL Rest Dirs Dict (smpl_24, joint 1):", get_rest_directions_dict('smpl_24').get(1))
    print("SMPL Rest Dirs Tensor (smpl_24, joint 1):", get_rest_directions_tensor('smpl_24')[1])
    print("SMPL Rest Dirs Placeholder Tensor (smpl_24, joint 1):", get_rest_directions_tensor('smpl_24', use_placeholder=True)[1])
    try:
        get_num_joints('unknown_skeleton')
    except ValueError as e:
        print(f"Caught expected error: {e}")