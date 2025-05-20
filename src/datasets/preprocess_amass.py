import numpy as np
import torch
import smplx
import os

def generate_poses_r3j_from_amass_params(
    source_npz_path, 
    output_npz_path, 
    smpl_model_folder,
    model_type='smplh',
    target_joint_indices=None,
    device_str='cpu'
    ):
    
    try:
        data = np.load(source_npz_path, allow_pickle=True)
        print(f"Processing: {source_npz_path}")

        required_keys = ['poses', 'betas', 'trans', 'gender']
        for key in required_keys:
            if key not in data:
                print(f"Error: Required key '{key}' not found in {source_npz_path}. Skipping.")
                return

        gender_str = str(data['gender'])
        if isinstance(gender_str, np.ndarray):
            gender_str = gender_str.item()
            if isinstance(gender_str, bytes):
                 gender_str = gender_str.decode('utf-8')
        
        gender_str = gender_str.strip().lower()
        if gender_str not in ['male', 'female', 'neutral']:
            print(f"Warning: Unknown gender '{gender_str}' in {source_npz_path}. Defaulting to 'neutral'.")
            gender_str = 'neutral'
            
        num_betas_from_file = data['betas'].shape[-1]
        
        body_model = smplx.create(
            model_path=smpl_model_folder,
            model_type=model_type,
            gender=gender_str,
            num_betas=num_betas_from_file,
            ext='pkl' if model_type == 'smplh' or model_type == 'smpl' else 'npz',
            use_pca=False, 
            flat_hand_mean=True if model_type in ['smplh', 'smplx'] else False, 
        ).to(device_str)

        poses_params = torch.tensor(data['poses'], dtype=torch.float32, device=device_str)
        betas_params = torch.tensor(data['betas'], dtype=torch.float32, device=device_str)
        trans_params = torch.tensor(data['trans'], dtype=torch.float32, device=device_str)
        
        num_frames = poses_params.shape[0]

        if betas_params.ndim == 1: # (num_betas,)
            betas_params = betas_params.unsqueeze(0).repeat(num_frames, 1)
        elif betas_params.shape[0] == 1 and num_frames > 1: # (1, num_betas) -> (N, num_betas)
            betas_params = betas_params.repeat(num_frames, 1)
        elif betas_params.shape[0] != num_frames:
            print(f"Warning: Beta shape {betas_params.shape} mismatch with num_frames {num_frames} in {source_npz_path}. "
                  f"Using first beta for all frames.")
            betas_params = betas_params[0:1].repeat(num_frames, 1)

        # SMPL+H pose parameters: global_orient (3), body_pose (63), left_hand_pose (45), right_hand_pose (45)
        global_orient = poses_params[:, 0:3]
        body_pose = poses_params[:, 3:66] # 21 body joints * 3
        
        left_hand_pose = None
        right_hand_pose = None
        expression_params = None # For SMPL-X

        if model_type == 'smplh':
            if poses_params.shape[1] >= 111: 
                 left_hand_pose = poses_params[:, 66:111]
            if poses_params.shape[1] >= 156: 
                 right_hand_pose = poses_params[:, 111:156]
        elif model_type == 'smplx':
            # SMPL-X: global_orient(3) + body_pose(63) + jaw_pose(3) + leye_pose(3) + reye_pose(3) = 75
            # left_hand_pose(45), right_hand_pose(45)
            # expression(10)
            # Total: 75+45+45+10 = 175
            body_pose = poses_params[:, 3:66]
            if poses_params.shape[1] > 66:
                 left_hand_pose = poses_params[:, 75:120] if poses_params.shape[1] >=120 else None
                 right_hand_pose = poses_params[:, 120:165] if poses_params.shape[1] >=165 else None

        with torch.no_grad():
            model_output = body_model(
                betas=betas_params,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                transl=trans_params,
                return_verts=False,
                return_full_pose=False 
            )
        
        all_joints_3d_model = model_output.joints.detach().cpu().numpy() 

        if target_joint_indices:
            if all_joints_3d_model.shape[1] < np.max(target_joint_indices) + 1:
                 print(f"Error: Model produced {all_joints_3d_model.shape[1]} joints, but target_joint_indices "
                       f"requires at least {np.max(target_joint_indices) + 1}. Skipping {source_npz_path}.")
                 return
            poses_r3j_selected = all_joints_3d_model[:, target_joint_indices, :]
        else:
            poses_r3j_selected = all_joints_3d_model

        save_data_dict = {key: data[key] for key in data.files}
        save_data_dict['poses_r3j'] = poses_r3j_selected

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


if __name__ == '__main__':
    SMPL_MODELS_PATH = "/path/to/your/body_models/"
    SOURCE_AMASS_DATA_ROOT = "/path/to/your/downloaded/AMASS_data/"
    PROCESSED_AMASS_DATA_ROOT = "/path/to/your/processed_AMASS_data/"

    smplh_to_smpl24_body_indices = list(range(22)) + [20, 21] 
    dataset_name = "CMU"
    subject_name = "01"
    
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
                model_type='smplh',
                target_joint_indices=smplh_to_smpl24_body_indices,
                device_str='cuda' if torch.cuda.is_available() else 'cpu' 
            )
    else:
        print(f"Source folder not found: {source_folder}")