#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.amass_dataset import AMASSSubsetDataset
from src.kinematics.skeleton_utils import get_skeleton_parents


def _create_animation(clean: np.ndarray,
                      noisy: np.ndarray,
                      skeleton_type: str,
                      title_noise_info: str,
                      fps: int = 30,
                      center_around_root_in_plot: bool = False
                      ) -> FuncAnimation:
    if clean.shape != noisy.shape or clean.ndim != 3:
        raise ValueError(f"Clean ({clean.shape}) and noisy ({noisy.shape}) sequences must both have shape (T, J, 3)")

    T, J, _ = clean.shape
    skeleton_parents = get_skeleton_parents(skeleton_type)

    fig, (ax_clean, ax_noisy) = plt.subplots(
        1, 2, figsize=(17, 8.5), subplot_kw={"projection": "3d"}
    )
    fig.suptitle(f"Clean vs. Noisy ({title_noise_info})", fontsize=16)
    
    data_to_plot = []
    current_clean_data = clean.copy()
    current_noisy_data = noisy.copy()

    if center_around_root_in_plot:
        for t_idx in range(T):
            current_clean_data[t_idx] -= current_clean_data[t_idx, 0:1, :]
            current_noisy_data[t_idx] -= current_noisy_data[t_idx, 0:1, :]
    data_to_plot.extend([current_clean_data, current_noisy_data])

    xyz_all_display_data = np.concatenate(data_to_plot, axis=0)
    x_min_disp, x_max_disp = xyz_all_display_data[..., 0].min(), xyz_all_display_data[..., 0].max()
    y_min_disp, y_max_disp = xyz_all_display_data[..., 1].min(), xyz_all_display_data[..., 1].max()
    z_min_disp, z_max_disp = xyz_all_display_data[..., 2].min(), xyz_all_display_data[..., 2].max()

    mid_x, mid_y, mid_z = (x_min_disp + x_max_disp) / 2, (y_min_disp + y_max_disp) / 2, (z_min_disp + z_max_disp) / 2
    range_x_disp, range_y_disp, range_z_disp = x_max_disp - x_min_disp, y_max_disp - y_min_disp, z_max_disp - z_min_disp
    max_total_range = max(range_x_disp, range_y_disp, range_z_disp, 0.1) 

    plot_margin_factor = 0.15
    half_span = max_total_range / 2 * (1 + plot_margin_factor)

    axes = (ax_clean, ax_noisy)
    titles = ("Clean Pose", "Noisy Pose")
    point_colors = ("green", "red")
    bone_colors = ("lime", "lightcoral")

    scatters_list: list[plt.Artist] = []
    bone_lines_collection: list[list[plt.Line2D]] = []
    title_texts_list: list[plt.Text] = []

    for idx, ax in enumerate(axes):
        ax.set_xlim(mid_x - half_span, mid_x + half_span)
        ax.set_ylim(mid_y - half_span, mid_y + half_span)
        ax.set_zlim(mid_z - half_span, mid_z + half_span)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            ax.set_aspect("auto")

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        title_obj = ax.set_title(f"{titles[idx]} — Frame 0/{T - 1}")
        title_texts_list.append(title_obj)

        pose0 = data_to_plot[idx][0]
        scat = ax.scatter(pose0[:, 0], pose0[:, 1], pose0[:, 2],
                          c=point_colors[idx], s=50, marker="o",
                          edgecolors="black", linewidths=0.3, depthshade=True)
        scatters_list.append(scat)

        current_ax_bone_lines = []
        for j, parent in enumerate(skeleton_parents):
            if parent == -1: continue
            (line,) = ax.plot([pose0[j, 0], pose0[parent, 0]], [pose0[j, 1], pose0[parent, 1]], [pose0[j, 2], pose0[parent, 2]],
                                color=bone_colors[idx], linewidth=2)
            current_ax_bone_lines.append(line)
        bone_lines_collection.append(current_ax_bone_lines)
        ax.view_init(elev=15.0, azim=-75)

    def _update(frame: int):
        artists: list[plt.Artist] = []
        for view_idx in range(2):
            current_pose = data_to_plot[view_idx][frame]
            scat = scatters_list[view_idx]
            lines_for_current_ax = bone_lines_collection[view_idx]
            title_obj = title_texts_list[view_idx]

            scat._offsets3d = (current_pose[:, 0], current_pose[:, 1], current_pose[:, 2])
            artists.append(scat)
            line_i = 0
            for j_idx, parent_idx in enumerate(skeleton_parents):
                if parent_idx == -1: continue
                lines_for_current_ax[line_i].set_data([current_pose[j_idx, 0], current_pose[parent_idx, 0]], [current_pose[j_idx, 1], current_pose[parent_idx, 1]])
                lines_for_current_ax[line_i].set_3d_properties([current_pose[j_idx, 2], current_pose[parent_idx, 2]])
                artists.append(lines_for_current_ax[line_i]); line_i += 1
            title_obj.set_text(f"{titles[view_idx]} — Frame {frame}/{T - 1}"); artists.append(title_obj)
        return artists

    interval_ms = max(20, int(1000 / fps))
    return FuncAnimation(fig, _update, frames=T, interval=interval_ms, blit=False)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AMASS clean vs. various noisy types visualization.")
    p.add_argument("--amass_npz", required=True, help="Path to processed AMASS .npz file")
    p.add_argument("--window_size", type=int, default=150, help="Animation length")
    p.add_argument("--skeleton_type", default="smpl_24", help="Skeleton type")
    p.add_argument("--center_root_in_dataset", action="store_true", help="Dataset centers poses around root")
    p.add_argument("--center_root_in_plot", action="store_true", help="Plot centers poses around root (display only)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save", metavar="OUTPUT_FILE.[gif|mp4]", help="Save animation to file")
    p.add_argument("--fps", type=int, default=30, help="Animation FPS")

    g_gauss = p.add_argument_group("Gaussian Noise")
    g_gauss.add_argument("--gaussian_noise_std", type=float, default=0.0, help="Gaussian noise σ (meters)")

    g_temporal = p.add_argument_group("Temporal Noise")
    g_temporal.add_argument("--temporal_noise_type", choices=['none', 'filtered', 'persistent'], default='none', help="Type of temporal noise")
    g_temporal.add_argument("--temporal_noise_scale", type=float, default=0.0, help="Scale for temporal noise events/random signal") # Default 0.0
    g_temporal.add_argument("--temporal_filter_window", type=int, default=5, help="Window size for filtered temporal noise")
    g_temporal.add_argument("--temporal_event_prob", type=float, default=0.05, help="Probability of new event for persistent temporal noise")
    g_temporal.add_argument("--temporal_decay", type=float, default=0.8, help="Decay factor for persistent temporal noise")

    g_outlier = p.add_argument_group("Outlier Noise")
    g_outlier.add_argument("--outlier_prob", type=float, default=0.0, help="Probability of a joint being an outlier per frame")
    g_outlier.add_argument("--outlier_scale", type=float, default=0.0, help="Max deviation for outliers (meters)") # Default 0.0
    
    g_bonelen = p.add_argument_group("Bone Length Noise")
    g_bonelen.add_argument("--bonelen_noise_scale", type=float, default=0.0, help="Max relative bone length perturbation (e.g., 0.05 for +/-5%%)")
    
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    amass_path = Path(args.amass_npz).expanduser().resolve()
    if not amass_path.is_file(): sys.exit(f"[ERROR] AMASS file not found: {amass_path}")

    active_noise_descriptions = []
    data_paths_list = [{'poses_r3j_path': str(amass_path), 
                        'metadata_path': str(amass_path), 
                        'seq_name': amass_path.stem}]

    print("Loading base clean sequence...")
    clean_np_base = None
    dataset_clean_args = dict(
        data_paths=data_paths_list,
        window_size=args.window_size, skeleton_type=args.skeleton_type,
        center_around_root=args.center_root_in_dataset, joint_selector_indices=None,
        # Explicitly turn off all noises for the clean loader
        gaussian_noise_std=0.0,
        temporal_noise_type='none', temporal_noise_scale=0.0,
        outlier_prob=0.0, outlier_scale=0.0,
        bonelen_noise_scale=0.0
    )
    dataset_clean = AMASSSubsetDataset(**dataset_clean_args, is_train=False)
    
    if len(dataset_clean) == 0: sys.exit("[ERROR] Clean dataset loader is empty.")
    clean_window_from_loader, _, _ = dataset_clean[0] 
    clean_np_base = clean_window_from_loader.numpy()
    print(f"Base clean sequence loaded, shape: {clean_np_base.shape}")

    print("Loading noisy sequence with specified parameters...")
    noisy_np_to_animate = None
    dataset_noisy_args = dict(
        data_paths=data_paths_list,
        window_size=args.window_size, skeleton_type=args.skeleton_type,
        center_around_root=args.center_root_in_dataset, joint_selector_indices=None,
        gaussian_noise_std=args.gaussian_noise_std,
        temporal_noise_type=args.temporal_noise_type,
        temporal_noise_scale=args.temporal_noise_scale,
        temporal_filter_window=args.temporal_filter_window,
        temporal_event_prob=args.temporal_event_prob,
        temporal_decay=args.temporal_decay,
        outlier_prob=args.outlier_prob,
        outlier_scale=args.outlier_scale,
        bonelen_noise_scale=args.bonelen_noise_scale
    )
    dataset_noisy = AMASSSubsetDataset(**dataset_noisy_args, is_train=True)
    if len(dataset_noisy) == 0: sys.exit("[ERROR] Noisy dataset loader is empty.")
    noisy_window_from_loader, _, _ = dataset_noisy[0]
    noisy_np_to_animate = noisy_window_from_loader.numpy()
    print(f"Noisy sequence generated by dataset, shape: {noisy_np_to_animate.shape}")
    
    # noise description constructor
    if args.gaussian_noise_std > 0: active_noise_descriptions.append(f"Gauss σ={args.gaussian_noise_std:.2f}")
    if args.temporal_noise_type != 'none' and args.temporal_noise_scale > 0:
        if args.temporal_noise_type == 'filtered':
            active_noise_descriptions.append(f"TemporalFilt(s={args.temporal_noise_scale:.2f},w={args.temporal_filter_window})")
        elif args.temporal_noise_type == 'persistent':
             active_noise_descriptions.append(f"TemporalPers(s={args.temporal_noise_scale:.2f},p={args.temporal_event_prob:.2f},d={args.temporal_decay:.2f})")
    if args.outlier_prob > 0 and args.outlier_scale > 0: active_noise_descriptions.append(f"Outlier(p={args.outlier_prob:.3f},s={args.outlier_scale:.2f})")
    if args.bonelen_noise_scale > 0: active_noise_descriptions.append(f"BoneLen(s={args.bonelen_noise_scale:.2f})")
    if not active_noise_descriptions: active_noise_descriptions.append("None (should match clean)")

    # check animate shape
    if clean_np_base.shape != noisy_np_to_animate.shape or clean_np_base.ndim != 3:
        print(f"[ERROR] Final clean shape {clean_np_base.shape} and noisy shape {noisy_np_to_animate.shape} are incompatible.")
        sys.exit(1)
    
    # MPJPE
    if not np.allclose(clean_np_base, noisy_np_to_animate):
        diff = clean_np_base - noisy_np_to_animate
        mpjpe_val = np.mean(np.sqrt(np.sum(diff**2, axis=2)))
        print(f"Average MPJPE (clean vs. final noisy) across sequence: {mpjpe_val:.4f} m")

    # animation
    animation_title_noise_info = "; ".join(active_noise_descriptions) if active_noise_descriptions else "No specific noise requested"
    anim = _create_animation(clean_np_base, noisy_np_to_animate, args.skeleton_type, 
                             animation_title_noise_info, fps=args.fps, 
                             center_around_root_in_plot=args.center_root_in_plot)

    if args.save:
        output_path = Path(args.save).expanduser().resolve()
        file_ext = output_path.suffix.lower()
        print(f"Saving animation to {output_path} (this may take a while)…")
        
        if file_ext == ".gif": writer = PillowWriter(fps=args.fps, bitrate=1800) # Added bitrate for Pillow
        elif file_ext == ".mp4": writer = FFMpegWriter(fps=args.fps, metadata={"artist": __file__})
        else: 
            print(f"[ERROR] Unsupported save format: {file_ext}. Use .gif or .mp4."); sys.exit(1)
        anim.save(str(output_path), writer=writer)
        print("Done.")
    else:
        plt.show()


if __name__ == "__main__":
    main()