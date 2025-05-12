import numpy as np
import torch
import smplx # Make sure this is installed: pip install smplx
import os

def generate_poses_r3j_from_amass_params(
    source_npz_path, 
    output_npz_path, 
    smpl_model_folder, # Path to your downloaded SMPL/SMPL+H model files
    model_type='smplh', # 'smplh' or 'smpl' or 'smplx'
    target_joint_indices=None, # e.g., list(range(22)) + [20, 21] for your 24-joint body
    device_str='cpu' # 'cuda' if you have a GPU and want to speed it up
    ):
    """
    Loads an AMASS .npz file, computes 3D joint locations using smplx,
    selects target joints, and saves a new .npz file with 'poses_r3j'.
    """
    try:
        data = np.load(source_npz_path, allow_pickle=True)
        print(f"Processing: {source_npz_path}")

        # Required keys for SMPL forward pass
        required_keys = ['poses', 'betas', 'trans', 'gender']
        for key in required_keys:
            if key not in data:
                print(f"Error: Required key '{key}' not found in {source_npz_path}. Skipping.")
                return

        # Determine gender for model loading
        gender_str = str(data['gender'])
        if isinstance(gender_str, np.ndarray): # If gender is an array (e.g. from some datasets)
            gender_str = gender_str.item()
            if isinstance(gender_str, bytes):
                 gender_str = gender_str.decode('utf-8')
        
        # Workaround for gender strings that might be 'male ' or 'female '
        gender_str = gender_str.strip().lower()
        if gender_str not in ['male', 'female', 'neutral']:
            print(f"Warning: Unknown gender '{gender_str}' in {source_npz_path}. Defaulting to 'neutral'.")
            gender_str = 'neutral'

        # Load SMPL/SMPL+H model using smplx
        # Ensure num_betas matches what's in the file if possible, or a default
        num_betas_from_file = data['betas'].shape[-1]
        
        body_model = smplx.create(
            model_path=smpl_model_folder,
            model_type=model_type,
            gender=gender_str,
            num_betas=num_betas_from_file,
            ext='pkl' if model_type == 'smplh' or model_type == 'smpl' else 'npz', # model file extension
            # These settings are often used for AMASS data with smplx
            use_pca=False, 
            flat_hand_mean=True if model_type in ['smplh', 'smplx'] else False, 
        ).to(device_str)

        # Prepare parameters for smplx model
        # Ensure tensors have the correct batch dimension and data type
        poses_params = torch.tensor(data['poses'], dtype=torch.float32, device=device_str)
        betas_params = torch.tensor(data['betas'], dtype=torch.float32, device=device_str)
        trans_params = torch.tensor(data['trans'], dtype=torch.float32, device=device_str)
        
        num_frames = poses_params.shape[0]

        # Ensure betas have batch dimension for smplx model
        # AMASS betas can be (num_betas,) or (1, num_betas) or (N, num_betas)
        if betas_params.ndim == 1: # (num_betas,)
            betas_params = betas_params.unsqueeze(0).repeat(num_frames, 1)
        elif betas_params.shape[0] == 1 and num_frames > 1: # (1, num_betas) -> (N, num_betas)
            betas_params = betas_params.repeat(num_frames, 1)
        elif betas_params.shape[0] != num_frames:
            print(f"Warning: Beta shape {betas_params.shape} mismatch with num_frames {num_frames} in {source_npz_path}. "
                  f"Using first beta for all frames.")
            betas_params = betas_params[0:1].repeat(num_frames, 1)

        # SMPL+H pose parameters: global_orient (3), body_pose (63), left_hand_pose (45), right_hand_pose (45)
        # Total 3+63+45+45 = 156
        global_orient = poses_params[:, 0:3]
        body_pose = poses_params[:, 3:66] # 21 body joints * 3
        
        left_hand_pose = None
        right_hand_pose = None
        expression_params = None # For SMPL-X

        if model_type == 'smplh':
            if poses_params.shape[1] >= 111: # Enough params for left hand
                 left_hand_pose = poses_params[:, 66:111]
            if poses_params.shape[1] >= 156: # Enough params for right hand
                 right_hand_pose = poses_params[:, 111:156]
        elif model_type == 'smplx':
            # SMPL-X: global_orient(3) + body_pose(63) + jaw_pose(3) + leye_pose(3) + reye_pose(3) = 75
            # left_hand_pose(45), right_hand_pose(45)
            # expression(10)
            # Total: 75+45+45+10 = 175 (actually more like 55 joints * 3 = 165 for pose without expression)
            # Check smplx documentation for exact slicing for SMPL-X poses from AMASS
            body_pose = poses_params[:, 3:66] # This is standard
            # You'll need to adjust slicing for hands and expression if using SMPL-X based on AMASS file spec
            # For simplicity, let's assume AMASS 'poses' for SMPL-X can be fed similarly for body part
            if poses_params.shape[1] > 66: # Placeholder, adjust based on SMPL-X AMASS spec
                 left_hand_pose = poses_params[:, 75:120] if poses_params.shape[1] >=120 else None # Example
                 right_hand_pose = poses_params[:, 120:165] if poses_params.shape[1] >=165 else None # Example
            # expression_params = data['expression'] # if available and model_type='smplx'

        with torch.no_grad(): # No need for gradients during inference
            model_output = body_model(
                betas=betas_params,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                transl=trans_params,
                # dmpls=dmpls_params, # dmpls might require specific handling or be part of betas in some models
                return_verts=False, # We only need joints
                return_full_pose=False 
            )
        
        # Get all 3D joint locations from the model output
        # For SMPL+H, this will be 52 joints. For SMPL, 24 joints.
        all_joints_3d_model = model_output.joints.detach().cpu().numpy() 

        if target_joint_indices:
            if all_joints_3d_model.shape[1] < np.max(target_joint_indices) + 1:
                 print(f"Error: Model produced {all_joints_3d_model.shape[1]} joints, but target_joint_indices "
                       f"requires at least {np.max(target_joint_indices) + 1}. Skipping {source_npz_path}.")
                 return
            poses_r3j_selected = all_joints_3d_model[:, target_joint_indices, :]
        else:
            # If no specific indices, assume the model output is already what's desired
            # (e.g., if you used model_type='smpl' and it outputs 24 joints)
            poses_r3j_selected = all_joints_3d_model

        # Prepare data for saving (copy existing keys, add/overwrite 'poses_r3j')
        save_data_dict = {key: data[key] for key in data.files}
        save_data_dict['poses_r3j'] = poses_r3j_selected # This is what your dataset script expects

        # Also process bone_offsets if they exist and need selection
        if 'bone_offsets' in data and target_joint_indices:
            bone_offsets_raw = data['bone_offsets']
            if bone_offsets_raw.shape[0] < np.max(target_joint_indices) + 1:
                print(f"Warning: bone_offsets_raw in {source_npz_path} has {bone_offsets_raw.shape[0]} entries, "
                      f"but target_joint_indices requires {np.max(target_joint_indices) + 1}. Not saving selected bone_offsets.")
            else:
                selected_bone_offsets = bone_offsets_raw[target_joint_indices, :]
                save_data_dict['bone_offsets'] = selected_bone_offsets


        os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
        np.savez(output_npz_path, **save_data_dict)
        print(f"Successfully processed and saved to {output_npz_path}. 'poses_r3j' shape: {poses_r3j_selected.shape}")

    except Exception as e:
        print(f"ERROR processing file {source_npz_path}: {e}")
        import traceback
        traceback.print_exc()

# --- Example Usage of the Pre-processing Function ---
if __name__ == '__main__':
    # !!! USER: YOU NEED TO SET THESE PATHS CORRECTLY !!!
    SMPL_MODELS_PATH = "/path/to/your/body_models/" # Directory containing SMPLH_MALE.pkl, SMPLH_FEMALE.pkl, SMPLH_NEUTRAL.pkl etc.
    SOURCE_AMASS_DATA_ROOT = "/path/to/your/downloaded/AMASS_data/" # e.g., containing CMU/01, ACCAD/Female1_ مزیدار_poses, etc.
    PROCESSED_AMASS_DATA_ROOT = "/path/to/your/processed_AMASS_data/" # Where the new .npz files will be saved

    # Your target mapping from SMPL+H (52 joints) to your SMPL (24 joints)
    # This assumes your 24-joint SMPL uses joints 0-21 from SMPL+H (body to wrists)
    # and then SMPL+H joint 20 (L_Wrist) for target joint 22 (LHand)
    # and SMPL+H joint 21 (R_Wrist) for target joint 23 (RHand).
    # !!! VERIFY THIS MAPPING IS CORRECT FOR YOUR DEFINITIONS !!!
    smplh_to_smpl24_body_indices = list(range(22)) + [20, 21] 

    # Example: Process one specific dataset like CMU
    # You would typically loop through all datasets and subjects you want to process
    dataset_name = "CMU" # Example
    subject_name = "01"  # Example
    
    source_folder = os.path.join(SOURCE_AMASS_DATA_ROOT, dataset_name, subject_name)
    output_folder = os.path.join(PROCESSED_AMASS_DATA_ROOT, dataset_name, subject_name)

    if not os.path.exists(SMPL_MODELS_PATH) or not os.path.isdir(SMPL_MODELS_PATH):
        print(f"ERROR: SMPL_MODELS_PATH '{SMPL_MODELS_PATH}' does not exist or is not a directory.")
        print("Please download SMPL-H models and set the correct path.")
        exit()

    if os.path.exists(source_folder):
        import glob
        for npz_file in glob.glob(os.path.join(source_folder, "*.npz")):
            base_name = os.path.basename(npz_file)
            output_file_path = os.path.join(output_folder, base_name)
            
            generate_poses_r3j_from_amass_params(
                source_npz_path=npz_file,
                output_npz_path=output_file_path,
                smpl_model_folder=SMPL_MODELS_PATH,
                model_type='smplh', # Assuming your AMASS files are based on SMPL+H
                target_joint_indices=smplh_to_smpl24_body_indices,
                device_str='cuda' if torch.cuda.is_available() else 'cpu' 
            )
    else:
        print(f"Source folder not found: {source_folder}")