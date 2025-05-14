# In preprocess_amass.py

import numpy as np
import torch
import smplx
# from smplx.vertex_ids import vertex_ids as SMPLX_VERTEX_IDS # Not strictly needed if we pass {}
import os
import glob
import sys
import traceback

# 假设 skeleton_utils.py 在 ../src/kinematics/ 路径下
# 如果您的项目结构不同，请调整此导入路径
# from src.kinematics.skeleton_utils import get_skeleton_parents # 如果需要，但 body_model.parents 通常可用

def generate_poses_r3j_from_amass_params(
    source_npz_path,
    output_npz_path,
    smpl_model_base_path, # Renamed from smpl_model_folder for clarity with your original script
    smpl_body_joint_indices_for_bone_offsets, # Still used if bone_offsets are in source
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
            gender_str = 'neutral' # Ensure gender_str for model loading is valid

        smpl_model_dir = os.path.join(smpl_model_base_path, 'smpl')
        specific_model_file_name = f'SMPL_{model_gender_for_path.upper()}.pkl'
        specific_model_file_path = os.path.join(smpl_model_dir, specific_model_file_name)

        if not os.path.exists(specific_model_file_path):
            print(f"Error: Basic SMPL model file not found at '{specific_model_file_path}'. Skipping.")
            return

        # num_betas_from_file = data['betas'].shape[-1] # Not directly used for model creation here

        print("INFO: Passing empty dict for vertex_ids to smplx.SMPL constructor to aim for LBS joints only.")

        body_model = smplx.SMPL(
            model_path=specific_model_file_path, # Corrected: use specific_model_file_path
            gender=gender_str,
            num_betas=10, # AMASS betas typically have 10 or 16 components, SMPL model uses 10.
            ext='pkl',
            vertex_ids={},
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_transl=True,
            use_hands=False,
            use_feet_keypoints=False,
            dtype=torch.float32
        ).to(device_str)

        poses_params_all = torch.tensor(data['poses'], dtype=torch.float32, device=device_str)
        betas_input_original = torch.tensor(data['betas'], dtype=torch.float32, device=device_str) # Keep original for T-pose
        trans_params = torch.tensor(data['trans'], dtype=torch.float32, device=device_str)

        num_frames = poses_params_all.shape[0]

        # Prepare betas_params for posed sequence (can be per-frame or static)
        betas_params_posed = betas_input_original.clone()
        if betas_params_posed.ndim == 1:
            betas_params_posed = betas_params_posed.unsqueeze(0)
        if betas_params_posed.shape[0] == 1 and num_frames > 1:
            betas_params_posed = betas_params_posed.repeat(num_frames, 1)
        elif betas_params_posed.shape[0] != num_frames and num_frames > 0 : # num_frames > 0 check added
            print(f"Warning: Beta shape {betas_params_posed.shape} mismatch with num_frames {num_frames}. Using first beta for posed sequence.")
            betas_params_posed = betas_params_posed[0:1].repeat(num_frames, 1)
        elif num_frames == 0: # Handle empty sequence case
            print(f"Warning: num_frames is 0 for {source_npz_path}. Cannot process betas for posed sequence.")
            return # Or handle as appropriate

        # Adjust beta dimensions for the model (posed sequence)
        if betas_params_posed.shape[1] != body_model.num_betas:
            # print(f"Warning: AMASS betas has {betas_params_posed.shape[1]} components for posed seq, SMPL model expects {body_model.num_betas}. Slicing/padding.")
            if betas_params_posed.shape[1] > body_model.num_betas:
                betas_params_posed = betas_params_posed[:, :body_model.num_betas]
            else: # betas_params_posed.shape[1] < body_model.num_betas
                padding_posed = torch.zeros((betas_params_posed.shape[0], body_model.num_betas - betas_params_posed.shape[1]), dtype=torch.float32, device=device_str)
                betas_params_posed = torch.cat([betas_params_posed, padding_posed], dim=1)


        # --- Calculate T-pose Bone Offsets ---
        calculated_bone_offsets = None
        try:
            betas_for_tpose = betas_input_original.clone() # Start with original betas from file
            if betas_for_tpose.ndim == 1:
                betas_for_tpose = betas_for_tpose.unsqueeze(0) # Make it (1, num_betas_file)
            
            # If file has more than one frame of betas, use the first one for T-pose
            if betas_for_tpose.shape[0] > 1:
                betas_for_tpose = betas_for_tpose[0:1]

            # Adjust num_betas for T-pose calculation to match model
            if betas_for_tpose.shape[1] != body_model.num_betas:
                # print(f"Adjusting betas for T-pose from {betas_for_tpose.shape[1]} to {body_model.num_betas} components.")
                if betas_for_tpose.shape[1] > body_model.num_betas:
                    betas_for_tpose_adjusted = betas_for_tpose[:, :body_model.num_betas]
                else: # betas_for_tpose.shape[1] < body_model.num_betas
                    padding_tpose = torch.zeros((1, body_model.num_betas - betas_for_tpose.shape[1]), dtype=torch.float32, device=device_str)
                    betas_for_tpose_adjusted = torch.cat([betas_for_tpose, padding_tpose], dim=1)
            else:
                betas_for_tpose_adjusted = betas_for_tpose
            
            with torch.no_grad():
                t_pose_output = body_model(
                    betas=betas_for_tpose_adjusted, # Use adjusted (1, model_num_betas)
                    global_orient=torch.zeros(1, 3, device=device_str, dtype=torch.float32),
                    body_pose=torch.zeros(1, 69, device=device_str, dtype=torch.float32), # 23 joints * 3 params
                    transl=torch.zeros(1, 3, device=device_str, dtype=torch.float32),
                    return_verts=False
                )
            t_pose_joints_model_output_np = t_pose_output.joints.detach().cpu().numpy().squeeze()

            if t_pose_joints_model_output_np.shape != (24, 3): # SMPL model should output 24 joints
                print(f"Error: T-pose joints from SMPL model have shape {t_pose_joints_model_output_np.shape}, expected (24, 3).")
                raise ValueError("T-pose joint calculation did not yield (24, 3) joints.")

            smpl_model_parents = body_model.parents.cpu().numpy()
            calculated_bone_offsets = np.zeros((24, 3), dtype=np.float32)
            for j_idx in range(24):
                parent_idx = smpl_model_parents[j_idx]
                if parent_idx != -1:
                    calculated_bone_offsets[j_idx] = t_pose_joints_model_output_np[j_idx] - t_pose_joints_model_output_np[parent_idx]
            print(f"INFO: Calculated T-pose bone_offsets with shape {calculated_bone_offsets.shape}")

        except Exception as e_tpose:
            print(f"Error calculating T-pose bone offsets for {source_npz_path}: {e_tpose}")
            traceback.print_exc()
            calculated_bone_offsets = None
        # --- End T-pose Bone Offset Calculation ---


        if poses_params_all.shape[1] < 72: # Standard SMPL pose params (root_orient + 23 body joints)
            print(f"Error: 'poses' in {source_npz_path} have only {poses_params_all.shape[1]} params. Need at least 72 for SMPL. Skipping.")
            return

        global_orient_params = poses_params_all[:, 0:3]
        body_pose_params = poses_params_all[:, 3:72] # 23 joints * 3 params each = 69

        with torch.no_grad():
            model_output = body_model(
                betas=betas_params_posed, # Use per-frame or static betas for posed sequence
                global_orient=global_orient_params,
                body_pose=body_pose_params,
                transl=trans_params,
                return_verts=False,
            )

        poses_r3j_body = model_output.joints.detach().cpu().numpy() # (num_frames, 24, 3)

        if poses_r3j_body.shape[1] != 24:
            print(f"Error: smplx.SMPL model output {poses_r3j_body.shape[1]} joints for posed sequence, expected 24. Skipping {source_npz_path}.")
            return

        save_data_dict = {key: data[key] for key in data.files}
        save_data_dict['poses_r3j'] = poses_r3j_body

        # Add calculated bone_offsets if available, otherwise try to use existing from file
        if calculated_bone_offsets is not None and calculated_bone_offsets.shape == (24, 3):
            save_data_dict['bone_offsets'] = calculated_bone_offsets
            print(f"INFO: Added calculated 'bone_offsets' (shape {calculated_bone_offsets.shape}) to {output_npz_path}")
        elif 'bone_offsets' in data and smpl_body_joint_indices_for_bone_offsets:
            # Fallback to original logic if calculation failed AND bone_offsets are in source
            print(f"INFO: Using existing 'bone_offsets' from source file as calculation failed or was not prioritized.")
            bone_offsets_raw = data['bone_offsets']
            if bone_offsets_raw.ndim == 2 and \
               bone_offsets_raw.shape[0] >= np.max(smpl_body_joint_indices_for_bone_offsets) + 1 and \
               len(smpl_body_joint_indices_for_bone_offsets) == 24: # Ensure we select 24
                selected_bone_offsets = bone_offsets_raw[smpl_body_joint_indices_for_bone_offsets, :]
                save_data_dict['bone_offsets'] = selected_bone_offsets
                print(f"INFO: Copied and selected existing 'bone_offsets' to output (shape {selected_bone_offsets.shape}).")
            else:
                print(f"Warning: Existing bone_offsets_raw in {source_npz_path} has shape {bone_offsets_raw.shape}. "
                      f"Cannot select for 24 joints. 'bone_offsets' will not be in the output.")
        else:
            print(f"Warning: 'bone_offsets' could not be calculated and were not found in source {source_npz_path}. "
                  "Output .npz will not contain 'bone_offsets'.")


        os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
        np.savez(output_npz_path, **save_data_dict)
        print(f"Successfully processed (as SMPL body) and saved to {output_npz_path}. 'poses_r3j' shape: {poses_r3j_body.shape}")

    except Exception as e:
        print(f"ERROR processing file {source_npz_path}: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    SMPL_MODEL_BASE_PATH = r"D:\git_repo_tidy\ESE6500\FinalProj\data\models"
    SOURCE_AMASS_DATA_ROOT = r"D:\git_repo_tidy\ESE6500\FinalProj\data"
    PROCESSED_AMASS_DATA_ROOT = r"D:\git_repo_tidy\ESE6500\FinalProj\data\processed"
    # This index list was for selecting from a larger set (e.g., SMPL+H's 52 offsets)
    # If we are generating for SMPL 24 joints directly, it might not be strictly needed
    # for selection, but the function expects it. We will use it for the fallback.
    smplh_indices_for_bone_offsets_selection = list(range(22)) + [20, 21] # Corresponds to 24 joints
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")
    print(f"SMPL Model Base Path (expects 'smpl/SMPL_GENDER.pkl' inside): {SMPL_MODEL_BASE_PATH}")
    print(f"Source AMASS Data Root: {SOURCE_AMASS_DATA_ROOT}")
    print(f"Processed AMASS Data Root: {PROCESSED_AMASS_DATA_ROOT}")

    # Example: Process a specific subset, e.g., CMU
    # You can adapt this loop to process all datasets you need.
    cmu_source_root = os.path.join(SOURCE_AMASS_DATA_ROOT, 'CMU')
    if not os.path.isdir(cmu_source_root):
        print(f"ERROR: CMU source folder not found at {cmu_source_root}")
        sys.exit()

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

            try:
                if os.path.exists(output_path):
                    existing_data = np.load(output_path, allow_pickle=True)
                    # Check if both poses_r3j and bone_offsets (if expected) are present and valid
                    if 'poses_r3j' in existing_data and existing_data['poses_r3j'].shape[1] == 24 and \
                       'bone_offsets' in existing_data and existing_data['bone_offsets'].shape == (24,3) : # Added check for bone_offsets
                        print(f"  → Skipping {base_name}, already processed with poses_r3j and bone_offsets.")
                        continue
                    else:
                         print(f"  ! Re-processing {base_name} as existing file is incomplete or has wrong shapes.")
            except Exception as e_check:
                print(f"  ! Check failed for {base_name}, will reprocess. Error: {e_check}")

            generate_poses_r3j_from_amass_params(
                source_npz_path=npz_path,
                output_npz_path=output_path,
                smpl_model_base_path=SMPL_MODEL_BASE_PATH,
                smpl_body_joint_indices_for_bone_offsets=smplh_indices_for_bone_offsets_selection,
                device_str=DEVICE
            )
    print("\nPre-processing finished.")