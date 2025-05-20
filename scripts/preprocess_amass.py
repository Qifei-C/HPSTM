# In preprocess_amass.py

import numpy as np
import torch
import smplx
import os
import glob
import sys
import traceback


def generate_poses_r3j_from_amass_params(
    source_npz_path,
    output_npz_path,
    smpl_model_base_path, 
    smpl_body_joint_indices_for_bone_offsets,
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
            gender_str = 'neutral'
            
        smpl_model_dir = os.path.join(smpl_model_base_path, 'smpl')
        specific_model_file_name = f'SMPL_{model_gender_for_path.upper()}.pkl'
        specific_model_file_path = os.path.join(smpl_model_dir, specific_model_file_name)

        if not os.path.exists(specific_model_file_path):
            print(f"Error: Basic SMPL model file not found at '{specific_model_file_path}'. Skipping.")
            return

        print("INFO: Passing empty dict for vertex_ids to smplx.SMPL constructor to aim for LBS joints only.")

        body_model = smplx.SMPL(
            model_path=specific_model_file_path,
            gender=gender_str,
            num_betas=10,
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
        betas_input_original = torch.tensor(data['betas'], dtype=torch.float32, device=device_str)
        trans_params = torch.tensor(data['trans'], dtype=torch.float32, device=device_str)

        num_frames = poses_params_all.shape[0]
        
        betas_params_posed = betas_input_original.clone()
        if betas_params_posed.ndim == 1:
            betas_params_posed = betas_params_posed.unsqueeze(0)
        if betas_params_posed.shape[0] == 1 and num_frames > 1:
            betas_params_posed = betas_params_posed.repeat(num_frames, 1)
        elif betas_params_posed.shape[0] != num_frames and num_frames > 0:
            print(f"Warning: Beta shape {betas_params_posed.shape} mismatch with num_frames {num_frames}. Using first beta for posed sequence.")
            betas_params_posed = betas_params_posed[0:1].repeat(num_frames, 1)
        elif num_frames == 0:
            print(f"Warning: num_frames is 0 for {source_npz_path}. Cannot process betas for posed sequence.")
            return

        if betas_params_posed.shape[1] != body_model.num_betas:
            if betas_params_posed.shape[1] > body_model.num_betas:
                betas_params_posed = betas_params_posed[:, :body_model.num_betas]
            else:
                padding_posed = torch.zeros((betas_params_posed.shape[0], body_model.num_betas - betas_params_posed.shape[1]), dtype=torch.float32, device=device_str)
                betas_params_posed = torch.cat([betas_params_posed, padding_posed], dim=1)


        # T-pose Bone Offsets
        calculated_bone_offsets = None
        try:
            betas_for_tpose = betas_input_original.clone()
            if betas_for_tpose.ndim == 1:
                betas_for_tpose = betas_for_tpose.unsqueeze(0) # (1, num_betas_file)
            if betas_for_tpose.shape[0] > 1:
                betas_for_tpose = betas_for_tpose[0:1]
            if betas_for_tpose.shape[1] != body_model.num_betas:
                if betas_for_tpose.shape[1] > body_model.num_betas:
                    betas_for_tpose_adjusted = betas_for_tpose[:, :body_model.num_betas]
                else:
                    padding_tpose = torch.zeros((1, body_model.num_betas - betas_for_tpose.shape[1]), dtype=torch.float32, device=device_str)
                    betas_for_tpose_adjusted = torch.cat([betas_for_tpose, padding_tpose], dim=1)
            else:
                betas_for_tpose_adjusted = betas_for_tpose
            
            with torch.no_grad():
                t_pose_output = body_model(
                    betas=betas_for_tpose_adjusted, # (1, model_num_betas)
                    global_orient=torch.zeros(1, 3, device=device_str, dtype=torch.float32),
                    body_pose=torch.zeros(1, 69, device=device_str, dtype=torch.float32), # 23 joints * 3 params
                    transl=torch.zeros(1, 3, device=device_str, dtype=torch.float32),
                    return_verts=False
                )
            t_pose_joints_model_output_np = t_pose_output.joints.detach().cpu().numpy().squeeze()

            if t_pose_joints_model_output_np.shape != (24, 3):
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


        if poses_params_all.shape[1] < 72:
            print(f"Error: 'poses' in {source_npz_path} have only {poses_params_all.shape[1]} params. Need at least 72 for SMPL. Skipping.")
            return

        global_orient_params = poses_params_all[:, 0:3]
        body_pose_params = poses_params_all[:, 3:72]

        with torch.no_grad():
            model_output = body_model(
                betas=betas_params_posed,
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

        if calculated_bone_offsets is not None and calculated_bone_offsets.shape == (24, 3):
            save_data_dict['bone_offsets'] = calculated_bone_offsets
            print(f"INFO: Added calculated 'bone_offsets' (shape {calculated_bone_offsets.shape}) to {output_npz_path}")
        elif 'bone_offsets' in data and smpl_body_joint_indices_for_bone_offsets:
            print(f"INFO: Using existing 'bone_offsets' from source file as calculation failed or was not prioritized.")
            bone_offsets_raw = data['bone_offsets']
            if bone_offsets_raw.ndim == 2 and \
               bone_offsets_raw.shape[0] >= np.max(smpl_body_joint_indices_for_bone_offsets) + 1 and \
               len(smpl_body_joint_indices_for_bone_offsets) == 24:
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

    smplh_indices_for_bone_offsets_selection = list(range(22)) + [20, 21]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")
    print(f"SMPL Model Base Path (expects 'smpl/SMPL_GENDER.pkl' inside): {SMPL_MODEL_BASE_PATH}")
    print(f"Source AMASS Data Root: {SOURCE_AMASS_DATA_ROOT}")
    print(f"Processed AMASS Data Root: {PROCESSED_AMASS_DATA_ROOT}")

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
                    if 'poses_r3j' in existing_data and existing_data['poses_r3j'].shape[1] == 24 and \
                       'bone_offsets' in existing_data and existing_data['bone_offsets'].shape == (24,3):
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