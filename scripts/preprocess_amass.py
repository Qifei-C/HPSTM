# In preprocess_amass.py

import numpy as np
import torch
import smplx
# from smplx.vertex_ids import vertex_ids as SMPLX_VERTEX_IDS # Not strictly needed if we pass {}
import os
import glob
import sys 
import traceback

def generate_poses_r3j_from_amass_params(
    source_npz_path,
    output_npz_path,
    smpl_model_base_path,
    smpl_body_joint_indices_for_bone_offsets, # Still needed for bone_offsets slicing
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

        gender_from_file = data['gender']
        if isinstance(gender_from_file, np.ndarray): gender_str = gender_from_file.item()
        else: gender_str = str(gender_from_file)
        if isinstance(gender_str, bytes): gender_str = gender_str.decode('utf-8')
        gender_str = gender_str.strip().lower()
        model_gender_for_path = gender_str
        if gender_str not in ['male', 'female', 'neutral']:
            print(f"Warning: Unknown gender '{gender_str}' in {source_npz_path}. Using 'neutral'.")
            model_gender_for_path = 'neutral'
            
        smpl_model_dir = os.path.join(smpl_model_base_path, 'smpl')
        specific_model_file_name = f'SMPL_{model_gender_for_path.upper()}.pkl'
        specific_model_file_path = os.path.join(smpl_model_dir, specific_model_file_name)

        if not os.path.exists(specific_model_file_path):
            print(f"Error: Basic SMPL model file not found at '{specific_model_file_path}'. Skipping.")
            return

        num_betas_from_file = data['betas'].shape[-1]

        # --- MODIFICATION: Pass an empty dictionary for vertex_ids ---
        # This tells VertexJointSelector not to add any extra joints from vertices.
        # It should then default to using only the joints derived from J_regressor via LBS.
        # For a standard SMPL model, this should be 24 joints.
        vertex_ids_for_smpl_constructor = {} 
        print("INFO: Passing empty dict for vertex_ids to smplx.SMPL constructor to aim for LBS joints only.")
        # --- END MODIFICATION ---

        # --- MODIFICATION: Pass flags to control VertexJointSelector ---
        body_model = smplx.SMPL( 
            model_path=specific_model_file_path,
            gender=gender_str, 
            num_betas=10, 
            ext='pkl',
            vertex_ids={},                 # Pass empty dict to avoid any vertex_id based selection
            create_global_orient=True,     
            create_body_pose=True,       
            create_betas=True,             
            create_transl=True,            
            use_hands=False,               # Ensure no hand tips are added
            use_feet_keypoints=False,      # Ensure no extra feet keypoints from vertices are added
                                           # LBS joints should include feet for SMPL 24
            dtype=torch.float32
        ).to(device_str)
        # --- END MODIFICATION ---
        
        poses_params_all = torch.tensor(data['poses'], dtype=torch.float32, device=device_str)
        betas_input = torch.tensor(data['betas'], dtype=torch.float32, device=device_str)
        trans_params = torch.tensor(data['trans'], dtype=torch.float32, device=device_str)
        
        num_frames = poses_params_all.shape[0]

        if betas_input.ndim == 1:
            betas_input = betas_input.unsqueeze(0)
        if betas_input.shape[0] == 1 and num_frames > 1:
            betas_params = betas_input.repeat(num_frames, 1)
        elif betas_input.shape[0] == num_frames:
            betas_params = betas_input
        else:
            print(f"Warning: Beta shape {betas_input.shape} mismatch with num_frames {num_frames}. Using first beta.")
            betas_params = betas_input[0:1].repeat(num_frames, 1)

        if betas_params.shape[1] != body_model.num_betas:
            print(f"Warning: AMASS betas has {betas_params.shape[1]} components, SMPL model expects {body_model.num_betas}. Slicing/padding betas.")
            if betas_params.shape[1] > body_model.num_betas:
                betas_params = betas_params[:, :body_model.num_betas]
            else:
                padding = torch.zeros((num_frames, body_model.num_betas - betas_params.shape[1]), dtype=torch.float32, device=device_str)
                betas_params = torch.cat([betas_params, padding], dim=1)

        if poses_params_all.shape[1] < 72:
            print(f"Error: 'poses' in {source_npz_path} have only {poses_params_all.shape[1]} params. Need 72 for SMPL. Skipping.")
            return
            
        global_orient_params = poses_params_all[:, 0:3]
        body_pose_params = poses_params_all[:, 3:72]

        with torch.no_grad():
            model_output = body_model(
                betas=betas_params,
                global_orient=global_orient_params,
                body_pose=body_pose_params,
                transl=trans_params,
                return_verts=False,
            )
        
        poses_r3j_body = model_output.joints.detach().cpu().numpy()

        if poses_r3j_body.shape[1] != 24:
            print(f"Error: smplx.SMPL model output {poses_r3j_body.shape[1]} joints, expected 24. "
                  f"This configuration (empty vertex_ids) did not yield 24 joints. "
                  f"The J_regressor in your SMPL .pkl model might be non-standard, or VertexJointSelector has other defaults. "
                  f"Skipping {source_npz_path}.")
            return

        save_data_dict = {key: data[key] for key in data.files}
        save_data_dict['poses_r3j'] = poses_r3j_body

        if 'bone_offsets' in data and smpl_body_joint_indices_for_bone_offsets:
            bone_offsets_raw = data['bone_offsets']
            if bone_offsets_raw.ndim == 2 and \
               bone_offsets_raw.shape[0] >= np.max(smpl_body_joint_indices_for_bone_offsets) + 1 and \
               len(smpl_body_joint_indices_for_bone_offsets) == 24:
                selected_bone_offsets = bone_offsets_raw[smpl_body_joint_indices_for_bone_offsets, :]
                save_data_dict['bone_offsets'] = selected_bone_offsets
            else:
                 print(f"Warning: bone_offsets_raw in {source_npz_path} has shape {bone_offsets_raw.shape}. Cannot select for 24 joints. Original kept if any.")
        
        os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
        np.savez(output_npz_path, **save_data_dict)
        print(f"Successfully processed (as SMPL body) and saved to {output_npz_path}. 'poses_r3j' shape: {poses_r3j_body.shape}")

    except Exception as e:
        print(f"ERROR processing file {source_npz_path}: {e}")
        traceback.print_exc()

# Ensure the __main__ block is the same as the previous version,
# especially the paths (SMPL_MODEL_BASE_PATH, etc.) and the loop.
# Make sure SMPL_MODEL_BASE_PATH points to the directory containing the 'smpl' subfolder,
# which contains your SMPL_NEUTRAL.pkl, SMPL_MALE.pkl, SMPL_FEMALE.pkl.
if __name__ == '__main__':
    SMPL_MODEL_BASE_PATH = r"D:\git_repo_tidy\ESE6500\FinalProj\Manifold\data\models" 
    SOURCE_AMASS_DATA_ROOT = r"D:\git_repo_tidy\ESE6500\FinalProj\Manifold\data"
    PROCESSED_AMASS_DATA_ROOT = r"D:\git_repo_tidy\ESE6500\FinalProj\Manifold\data\processed"
    smplh_indices_for_bone_offsets = list(range(22)) + [20, 21] 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    print(f"SMPL Model Base Path (expects 'smpl/SMPL_GENDER.pkl' inside): {SMPL_MODEL_BASE_PATH}")
    print(f"Source AMASS Data Root: {SOURCE_AMASS_DATA_ROOT}")
    print(f"Processed AMASS Data Root: {PROCESSED_AMASS_DATA_ROOT}")

    cmu_source_root = os.path.join(SOURCE_AMASS_DATA_ROOT, 'CMU')
    if not os.path.isdir(cmu_source_root):
        print(f"ERROR: CMU source folder not found at {cmu_source_root}")
        sys.exit()

    # 遍历 CMU 下所有子目录（每个 subject），并处理其中的 .npz
    for subject_name in os.listdir(cmu_source_root):
        source_subject_folder = os.path.join(cmu_source_root, subject_name)
        if not os.path.isdir(source_subject_folder):
            continue
        print(f"\nProcessing CMU subject: {subject_name}")

        output_subject_folder = os.path.join(PROCESSED_AMASS_DATA_ROOT, 'CMU', subject_name)
        os.makedirs(output_subject_folder, exist_ok=True)

        for npz_path in glob.glob(os.path.join(source_subject_folder, '*.npz')):
            base_name = os.path.basename(npz_path)
            output_path = os.path.join(output_subject_folder, base_name)

            # 如果已经处理过且结果正确，可跳过
            try:
                if os.path.exists(output_path):
                    existing = np.load(output_path)
                    if 'poses_r3j' in existing and existing['poses_r3j'].shape[1] == 24:
                        print(f"  → Skipping {base_name}, already processed.")
                        continue
            except Exception as e_check:
                print(f"  ! Check failed for {base_name}, will reprocess. Error: {e_check}")

            # 调用主函数处理
            generate_poses_r3j_from_amass_params(
                source_npz_path       = npz_path,
                output_npz_path       = output_path,
                smpl_model_base_path  = SMPL_MODEL_BASE_PATH,
                smpl_body_joint_indices_for_bone_offsets = smplh_indices_for_bone_offsets,
                device_str            = DEVICE
            )
    print("\nPre-processing finished.")