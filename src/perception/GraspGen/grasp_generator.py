# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import glob
import os
import time

import numpy as np
import open3d as o3d
import torch
import trimesh.transformations as tra
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import (
    point_cloud_outlier_removal,
    point_cloud_outlier_removal_with_color,
    filter_colliding_grasps,
)
from grasp_gen.robot import get_gripper_info


# ============================================================================
# Helper Functions
# ============================================================================

def pose_to_7d_list(pose_matrix):
    """
    Convert 4x4 transformation matrix to 7D list [x, y, z, qx, qy, qz, qw].
    
    Args:
        pose_matrix: (4, 4) transformation matrix
        
    Returns:
        list: [x, y, z, qx, qy, qz, qw]
    """
    # Extract position
    position = pose_matrix[:3, 3]
    
    # Extract rotation matrix and convert to quaternion
    rotation_matrix = pose_matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # Returns [qx, qy, qz, qw]
    
    # Combine position and quaternion
    pose_7d = [
        position[0], position[1], position[2],  # x, y, z
        quaternion[0], quaternion[1], quaternion[2], quaternion[3]  # qx, qy, qz, qw
    ]
    
    return pose_7d


def pose_7d_to_matrix(pose_7d):
    """
    Convert 7D pose [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix.
    
    Args:
        pose_7d: list or array [x, y, z, qx, qy, qz, qw]
        
    Returns:
        np.ndarray: (4, 4) transformation matrix
    """
    pose_7d = np.array(pose_7d)
    position = pose_7d[:3]
    quaternion = pose_7d[3:]  # [qx, qy, qz, qw]
    
    # Convert quaternion to rotation matrix
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    
    # Construct 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position
    
    return T


def transform_pose_to_world(grasp_pose_camera_7d, camera_pose_world_7d):
    """
    Transform grasp pose from camera frame to world frame.
    
    Args:
        grasp_pose_camera_7d: Grasp pose in camera frame [x, y, z, qx, qy, qz, qw]
        camera_pose_world_7d: Camera pose in world frame [x, y, z, qx, qy, qz, qw]
        
    Returns:
        grasp_pose_world_7d: Grasp pose in world frame [x, y, z, qx, qy, qz, qw]
    """
    # Convert 7D poses to 4x4 matrices
    T_world_camera = pose_7d_to_matrix(camera_pose_world_7d)
    T_camera_grasp = pose_7d_to_matrix(grasp_pose_camera_7d)
    
    # Transform: T_world_grasp = T_world_camera @ T_camera_grasp
    T_world_grasp = T_world_camera @ T_camera_grasp
    
    # Convert back to 7D
    grasp_pose_world_7d = pose_to_7d_list(T_world_grasp)
    
    return grasp_pose_world_7d


def analyze_point_cloud_coordinate_system(pc):
    """
    Analyze and print point cloud coordinate system information.
    Helps understand the reference frame of the point cloud.
    """
    print(f"\n  📊 Point Cloud Coordinate Analysis:")
    print(f"     Bounding box:")
    print(f"       X: [{pc[:, 0].min():.4f}, {pc[:, 0].max():.4f}]  range: {pc[:, 0].max() - pc[:, 0].min():.4f}m")
    print(f"       Y: [{pc[:, 1].min():.4f}, {pc[:, 1].max():.4f}]  range: {pc[:, 1].max() - pc[:, 1].min():.4f}m")
    print(f"       Z: [{pc[:, 2].min():.4f}, {pc[:, 2].max():.4f}]  range: {pc[:, 2].max() - pc[:, 2].min():.4f}m")
    print(f"     Centroid: [{pc[:, 0].mean():.4f}, {pc[:, 1].mean():.4f}, {pc[:, 2].mean():.4f}]")
    
    # Try to infer coordinate system
    z_min, z_max = pc[:, 2].min(), pc[:, 2].max()
    y_min, y_max = pc[:, 1].min(), pc[:, 1].max()
    
    print(f"\n  🔍 Coordinate System Inference:")
    if z_min > 0 and z_max > 0:
        print(f"     ✓ All Z > 0 → Likely CAMERA frame (Z = depth, points in front of camera)")
        print(f"       Camera origin is at (0, 0, 0), object is {z_min:.3f}m to {z_max:.3f}m away")
    elif abs(z_max - abs(z_min)) < 0.1:  # Z is roughly symmetric
        print(f"     ✓ Z is centered → Likely WORLD/ROBOT frame (Z could be up/down)")
    
    if y_min < 0 and y_max < 0:
        print(f"     ✓ All Y < 0 → Object is below camera/origin")
    elif y_min > 0 and y_max > 0:
        print(f"     ✓ All Y > 0 → Object is above origin")
    
    # Distance from origin
    distances = np.linalg.norm(pc, axis=1)
    print(f"     Average distance from origin: {distances.mean():.4f}m")
    print(f"     Distance range: [{distances.min():.4f}, {distances.max():.4f}]m")


def load_ply_file(ply_path):
    """
    Load point cloud from PLY file.
    
    Returns:
        pc: (N, 3) point cloud coordinates
        pc_color: (N, 3) point cloud colors [0-255]
    """
    print(f"  Loading PLY: {os.path.basename(ply_path)}")
    
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Extract coordinates
    pc = np.asarray(pcd.points)
    
    # Handle colors
    if pcd.has_colors():
        pc_color = np.asarray(pcd.colors) * 255.0  # Convert to [0, 255]
    else:
        pc_color = np.ones((len(pc), 3)) * 128.0  # Default gray
    
    print(f"    Loaded {len(pc)} points")
    
    # Analyze coordinate system
    analyze_point_cloud_coordinate_system(pc)
    
    return pc, pc_color


def center_point_cloud(pc):
    """Center point cloud around origin."""
    T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, T_subtract_pc_mean)
    return pc_centered, T_subtract_pc_mean


def filter_vertical_grasps(grasps, grasp_conf, angle_threshold=30.0):
    """
    Filter grasps to keep only those with gripper pointing downward (vertical).
    
    Args:
        grasps: (N, 4, 4) array of grasp poses
        grasp_conf: (N,) array of grasp confidences
        angle_threshold: Maximum angle (in degrees) from vertical to consider valid
    
    Returns:
        filtered_grasps: Grasps pointing downward
        filtered_conf: Corresponding confidences
        vertical_mask: Boolean mask of vertical grasps
    """
    # Extract z-axis direction from rotation matrix (third column)
    # For a gripper, the z-axis typically points in the approach direction
    z_axes = grasps[:, :3, 2]  # Shape: (N, 3)
    
    # Target direction is downward: [0, 0, -1]
    target_direction = np.array([0, 0, -1])
    
    # Calculate angles between each z-axis and downward direction
    # Using dot product: cos(angle) = z_axis · target / (|z_axis| * |target|)
    z_axes_normalized = z_axes / (np.linalg.norm(z_axes, axis=1, keepdims=True) + 1e-8)
    cos_angles = np.dot(z_axes_normalized, target_direction)
    angles_deg = np.rad2deg(np.arccos(np.clip(cos_angles, -1.0, 1.0)))
    
    # Filter grasps within angle threshold
    vertical_mask = angles_deg <= angle_threshold
    
    return grasps[vertical_mask], grasp_conf[vertical_mask], vertical_mask, angles_deg


def get_best_vertical_grasp(grasps, grasp_conf, angle_threshold=30.0, verticality_weight=0.5):
    """
    Get the single best grasp that points vertically downward.
    
    Args:
        grasps: (N, 4, 4) array of grasp poses
        grasp_conf: (N,) array of grasp confidences
        angle_threshold: Maximum angle (in degrees) from vertical
        verticality_weight: Weight for verticality vs confidence (0-1)
                          0 = only confidence, 1 = only verticality
    
    Returns:
        best_grasp: (4, 4) best vertical grasp pose, or None
        best_conf: Confidence of best grasp, or None
        num_vertical: Number of vertical grasps found
        angles: All angles from vertical
    """
    filtered_grasps, filtered_conf, vertical_mask, angles = filter_vertical_grasps(
        grasps, grasp_conf, angle_threshold
    )
    
    if len(filtered_grasps) == 0:
        return None, None, 0, angles
    
    # Get angles for filtered grasps
    filtered_angles = angles[vertical_mask]
    
    # Calculate composite score:
    # - Normalize confidence to [0, 1]
    # - Normalize verticality: 0° = score 1.0, angle_threshold° = score 0.0
    conf_normalized = (filtered_conf - filtered_conf.min() + 1e-6) / (filtered_conf.max() - filtered_conf.min() + 1e-6)
    verticality_score = 1.0 - (filtered_angles / angle_threshold)
    
    # Composite score (higher is better)
    composite_score = (1 - verticality_weight) * conf_normalized + verticality_weight * verticality_score
    
    # Get the grasp with highest composite score
    best_idx = np.argmax(composite_score)
    
    print(f"    Best grasp - Confidence: {filtered_conf[best_idx]:.3f}, "
          f"Angle: {filtered_angles[best_idx]:.1f}°, "
          f"Composite score: {composite_score[best_idx]:.3f}")
    
    return filtered_grasps[best_idx], filtered_conf[best_idx], len(filtered_grasps), angles


def process_grasps_for_visualization(pc, grasps, grasp_conf):
    """
    Process grasps and point cloud for visualization by centering them.
    (From official demo_scene_pc.py)
    """
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    print(f"  Confidence range: [{grasp_conf.min():.3f}, {grasp_conf.max():.3f}]")

    # Ensure grasps have correct homogeneous coordinate
    grasps[:, 3, 3] = 1

    # Center point cloud and grasps
    T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, T_subtract_pc_mean)
    grasps_centered = np.array(
        [T_subtract_pc_mean @ np.array(g) for g in grasps.tolist()]
    )

    return pc_centered, grasps_centered, scores, T_subtract_pc_mean


# ============================================================================
# Mode: Object - Simple object point cloud grasp generation
# ============================================================================

def run_object_mode(args, grasp_sampler, gripper_name, vis):
    """
    Object mode: Process single object point clouds.
    Based on demo_object_pc.py
    """
    print("\n" + "="*80)
    print("MODE: OBJECT (Simple object point cloud processing)")
    print("="*80)
    
    ply_files = glob.glob(os.path.join(args.sample_data_dir, "*.ply"))
    
    if len(ply_files) == 0:
        raise ValueError(f"No PLY files found in {args.sample_data_dir}")
    
    print(f"Found {len(ply_files)} PLY files\n")
    
    for ply_file in ply_files:
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(ply_file)}")
        print(f"{'='*80}")
        vis.delete()

        # Load object point cloud
        obj_pc, obj_pc_color = load_ply_file(ply_file)

        # Center point cloud
        obj_pc_centered, T_center = center_point_cloud(obj_pc)

        # Visualize original point cloud
        visualize_pointcloud(vis, "pc_original", obj_pc_centered, obj_pc_color, size=0.0025)

        # Filter outliers
        print("  Filtering outliers...")
        pc_filtered, pc_removed = point_cloud_outlier_removal(
            torch.from_numpy(obj_pc_centered)
        )
        pc_filtered = pc_filtered.cpu().numpy()
        pc_removed = pc_removed.cpu().numpy()
        
        print(f"    Kept: {len(pc_filtered)} points")
        print(f"    Removed: {len(pc_removed)} points")
        
        # Visualize removed points
        if len(pc_removed) > 0:
            visualize_pointcloud(vis, "pc_removed", pc_removed, [255, 0, 0], size=0.003)

        # Run inference
        print(f"\n  Running GraspGen inference...")
        grasps, grasp_conf = GraspGenSampler.run_inference(
            pc_filtered,
            grasp_sampler,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk_num_grasps,
        )

        if len(grasps) > 0:
            grasp_conf = grasp_conf.cpu().numpy()
            grasps = grasps.cpu().numpy()
            grasps[:, 3, 3] = 1
            
            scores = get_color_from_score(grasp_conf, use_255_scale=True)
            
            print(f"\n  ✓ Generated {len(grasps)} grasps")
            print(f"  Confidence range: [{grasp_conf.min():.3f}, {grasp_conf.max():.3f}]")

            if args.vertical_first:
                # Mode 1: Vertical-first mode (only show best vertical grasp)
                print(f"\n  [VERTICAL FIRST MODE]")
                print(f"  Filtering for vertical (downward) grasps...")
                print(f"    Verticality weight: {args.verticality_weight:.1f} (0=conf only, 1=vertical only)")
                best_grasp, best_conf, num_vertical, all_angles = get_best_vertical_grasp(
                    grasps, grasp_conf, 
                    angle_threshold=args.vertical_angle_threshold,
                    verticality_weight=args.verticality_weight
                )
                
                if best_grasp is not None:
                    print(f"  Found {num_vertical} vertical grasps (within {args.vertical_angle_threshold}°)")
                    print(f"  Best vertical grasp confidence: {best_conf:.3f}")
                    
                    # Transform grasp back to original coordinate system
                    grasp_original = np.linalg.inv(T_center) @ best_grasp
                    
                    # Convert to 7D list and print
                    pose_7d = pose_to_7d_list(grasp_original)
                    print(f"\n  📍 Best Grasp Pose (Camera Frame): {pose_7d}")
                    
                    # Transform to world frame if camera pose is provided
                    if args.camera_pose_world is not None:
                        pose_7d_world = transform_pose_to_world(pose_7d, args.camera_pose_world)
                        print(f"  📍 Best Grasp Pose (World Frame):  {pose_7d_world}")
                    
                    # Visualize only the best vertical grasp
                    visualize_grasp(
                        vis,
                        f"best_vertical_grasp",
                        best_grasp,
                        color=[0, 255, 0],  # Green for best grasp
                        gripper_name=gripper_name,
                        linewidth=2.0,
                    )
                    
                    # Optional: Save results
                    if args.output_dir is not None:
                        os.makedirs(args.output_dir, exist_ok=True)
                        output_file = os.path.join(
                            args.output_dir,
                            os.path.basename(ply_file).replace('.ply', '_best_vertical_grasp.npz')
                        )
                        
                        np.savez(
                            output_file,
                            best_grasp=grasp_original,
                            best_grasp_7d=pose_7d,
                            best_conf=best_conf,
                            pc_original=obj_pc,
                            pc_color=obj_pc_color,
                            pc_filtered=tra.transform_points(pc_filtered, np.linalg.inv(T_center)),
                            num_vertical_grasps=num_vertical,
                            all_angles=all_angles,
                        )
                        print(f"  Saved to: {output_file}")
                else:
                    print(f"  ⚠️  No vertical grasps found within {args.vertical_angle_threshold}° threshold!")
                    print(f"  Min angle from vertical: {all_angles.min():.1f}°")
            
            else:
                # Mode 2: Show all grasps mode
                print(f"\n  [SHOW ALL GRASPS MODE]")
                print(f"  Visualizing all {len(grasps)} grasps with color-coded confidence...")
                
                # Visualize all grasps with color based on confidence
                for idx in range(len(grasps)):
                    visualize_grasp(
                        vis,
                        f"grasp_{idx}",
                        grasps[idx],
                        color=scores[idx],
                        gripper_name=gripper_name,
                        linewidth=1.0,
                    )
                
                # Highlight the best grasp
                best_idx = np.argmax(grasp_conf)
                best_grasp = grasps[best_idx]
                best_conf = grasp_conf[best_idx]
                
                print(f"  Best grasp (highest confidence): #{best_idx}")
                print(f"  Best grasp confidence: {best_conf:.3f}")
                
                # Transform grasp back to original coordinate system
                grasp_original = np.linalg.inv(T_center) @ best_grasp
                
                # Convert to 7D list and print
                pose_7d = pose_to_7d_list(grasp_original)
                print(f"\n  📍 Best Grasp Pose (Camera Frame): {pose_7d}")
                
                # Transform to world frame if camera pose is provided
                if args.camera_pose_world is not None:
                    pose_7d_world = transform_pose_to_world(pose_7d, args.camera_pose_world)
                    print(f"  📍 Best Grasp Pose (World Frame):  {pose_7d_world}")
                
                # Highlight best grasp with green and thicker line
                visualize_grasp(
                    vis,
                    f"best_grasp_highlight",
                    best_grasp,
                    color=[0, 255, 0],
                    gripper_name=gripper_name,
                    linewidth=3.0,
                )
                
                # Optional: Save results
                if args.output_dir is not None:
                    os.makedirs(args.output_dir, exist_ok=True)
                    output_file = os.path.join(
                        args.output_dir,
                        os.path.basename(ply_file).replace('.ply', '_all_grasps.npz')
                    )
                    
                    # Transform all grasps back to original coordinate system
                    grasps_original = np.array([
                        np.linalg.inv(T_center) @ g for g in grasps
                    ])
                    
                    np.savez(
                        output_file,
                        all_grasps=grasps_original,
                        all_confidences=grasp_conf,
                        best_grasp_idx=best_idx,
                        best_grasp=grasp_original,
                        best_grasp_7d=pose_7d,
                        best_conf=best_conf,
                        pc_original=obj_pc,
                        pc_color=obj_pc_color,
                        pc_filtered=tra.transform_points(pc_filtered, np.linalg.inv(T_center)),
                    )
                    print(f"  Saved to: {output_file}")

        else:
            print("  ⚠️  No grasps found!")

        print(f"{'='*80}")
        input("\nPress Enter to continue to next object...")


# ============================================================================
# Mode: Scene - Scene with collision detection support
# ============================================================================

def run_scene_mode(args, grasp_sampler, gripper_name, vis):
    """
    Scene mode: Process scene + object point clouds with collision detection.
    Based on demo_scene_pc.py
    """
    print("\n" + "="*80)
    print("MODE: SCENE (Scene-aware with collision detection)")
    print("="*80)
    
    # Get gripper collision mesh for collision filtering
    gripper_info = None
    gripper_collision_mesh = None
    if args.filter_collisions:
        gripper_info = get_gripper_info(gripper_name)
        gripper_collision_mesh = gripper_info.collision_mesh
        print(f"Using gripper: {gripper_name}")
        print(f"Gripper collision mesh: {len(gripper_collision_mesh.vertices)} vertices\n")
    
    # Find object PLY files
    obj_ply_files = glob.glob(os.path.join(args.sample_data_dir, "*_object.ply"))
    
    if len(obj_ply_files) == 0:
        raise ValueError(
            f"No object PLY files found in {args.sample_data_dir}\n"
            f"For scene mode, files should be named: *_object.ply and *_scene.ply"
        )
    
    print(f"Found {len(obj_ply_files)} object PLY files\n")
    
    for obj_ply_file in obj_ply_files:
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(obj_ply_file)}")
        print(f"{'='*80}")
        vis.delete()

        # Load object point cloud
        obj_pc, obj_pc_color = load_ply_file(obj_ply_file)
        
        # Load corresponding scene point cloud
        scene_ply_file = obj_ply_file.replace('_object.ply', '_scene.ply')
        if not os.path.exists(scene_ply_file):
            print(f"  ⚠️  Warning: Scene file not found: {scene_ply_file}")
            print(f"  Falling back to object-only mode")
            scene_pc = None
            scene_pc_color = None
        else:
            scene_pc, scene_pc_color = load_ply_file(scene_ply_file)

        # Filter outliers from object point cloud
        print("  Filtering object outliers...")
        obj_pc_filtered, pc_removed, obj_pc_color_filtered, obj_pc_color_removed = (
            point_cloud_outlier_removal_with_color(
                torch.from_numpy(obj_pc), torch.from_numpy(obj_pc_color)
            )
        )
        obj_pc_filtered = obj_pc_filtered.cpu().numpy()
        pc_removed = pc_removed.cpu().numpy()
        obj_pc_color_filtered = obj_pc_color_filtered.cpu().numpy()
        
        print(f"    Kept: {len(obj_pc_filtered)} points")
        print(f"    Removed: {len(pc_removed)} points")

        # Visualize object point cloud
        visualize_pointcloud(vis, "pc_obj", obj_pc_filtered, obj_pc_color_filtered, size=0.005)
        
        # Visualize scene point cloud if available
        if scene_pc is not None:
            visualize_pointcloud(vis, "pc_scene", scene_pc, scene_pc_color, size=0.0025)

        # Run inference on object point cloud
        print(f"\n  Running GraspGen inference...")
        grasps, grasp_conf = GraspGenSampler.run_inference(
            obj_pc_filtered,
            grasp_sampler,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk_num_grasps,
        )

        if len(grasps) > 0:
            grasp_conf = grasp_conf.cpu().numpy()
            grasps = grasps.cpu().numpy()
            grasps[:, 3, 3] = 1

            print(f"\n  ✓ Generated {len(grasps)} grasps")

            # Process grasps for visualization (centering)
            obj_pc_centered, grasps_centered, scores, T_center = (
                process_grasps_for_visualization(
                    obj_pc_filtered, grasps, grasp_conf
                )
            )

            # Apply collision filtering if requested and scene is available
            collision_free_grasps = grasps_centered
            collision_free_conf = grasp_conf
            collision_free_scores = scores
            colliding_grasps = None
            
            if args.filter_collisions and scene_pc is not None:
                print("\n  Applying collision filtering...")
                collision_start = time.time()

                # Center scene point cloud using same transformation
                scene_pc_centered = tra.transform_points(scene_pc, T_center)

                # Downsample scene for faster collision checking
                if len(scene_pc_centered) > args.max_scene_points:
                    indices = np.random.choice(
                        len(scene_pc_centered), args.max_scene_points, replace=False
                    )
                    scene_pc_downsampled = scene_pc_centered[indices]
                    print(f"    Downsampled scene: {len(scene_pc_centered)} → {len(scene_pc_downsampled)} points")
                else:
                    scene_pc_downsampled = scene_pc_centered

                # Filter colliding grasps
                collision_free_mask = filter_colliding_grasps(
                    scene_pc=scene_pc_downsampled,
                    grasp_poses=grasps_centered,
                    gripper_collision_mesh=gripper_collision_mesh,
                    collision_threshold=args.collision_threshold,
                )

                collision_time = time.time() - collision_start
                print(f"    Collision detection: {collision_time:.2f}s")

                # Separate grasps
                collision_free_grasps = grasps_centered[collision_free_mask]
                collision_free_conf = grasp_conf[collision_free_mask]
                colliding_grasps = grasps_centered[~collision_free_mask]
                collision_free_scores = scores[collision_free_mask]

                print(f"    Collision-free: {len(collision_free_grasps)}/{len(grasps_centered)}")

            if args.vertical_first:
                # Mode 1: Vertical-first mode (only show best vertical grasp)
                print(f"\n  [VERTICAL FIRST MODE]")
                print(f"  Filtering for vertical (downward) grasps...")
                print(f"    Verticality weight: {args.verticality_weight:.1f} (0=conf only, 1=vertical only)")
                best_grasp, best_conf, num_vertical, all_angles = get_best_vertical_grasp(
                    collision_free_grasps, collision_free_conf, 
                    angle_threshold=args.vertical_angle_threshold,
                    verticality_weight=args.verticality_weight
                )
                
                if best_grasp is not None:
                    print(f"  Found {num_vertical} vertical grasps (within {args.vertical_angle_threshold}°)")
                    print(f"  Best vertical grasp confidence: {best_conf:.3f}")
                    
                    # Transform best grasp back to original coordinate system
                    best_grasp_original = tra.inverse_matrix(T_center) @ best_grasp
                    
                    # Convert to 7D list and print
                    pose_7d = pose_to_7d_list(best_grasp_original)
                    print(f"\n  📍 Best Grasp Pose (Camera Frame): {pose_7d}")
                    
                    # Transform to world frame if camera pose is provided
                    if args.camera_pose_world is not None:
                        pose_7d_world = transform_pose_to_world(pose_7d, args.camera_pose_world)
                        print(f"  📍 Best Grasp Pose (World Frame):  {pose_7d_world}")
                    
                    # Visualize only the best vertical grasp
                    visualize_grasp(
                        vis,
                        f"best_vertical_grasp",
                        best_grasp_original,
                        color=[0, 255, 0],  # Green for best grasp
                        gripper_name=gripper_name,
                        linewidth=2.5,
                    )
                else:
                    print(f"  ⚠️  No vertical grasps found within {args.vertical_angle_threshold}° threshold!")
                    if len(all_angles) > 0:
                        print(f"  Min angle from vertical: {all_angles.min():.1f}°")

                # Optional: Save results
                if args.output_dir is not None and best_grasp is not None:
                    os.makedirs(args.output_dir, exist_ok=True)
                    output_file = os.path.join(
                        args.output_dir,
                        os.path.basename(obj_ply_file).replace('_object.ply', '_best_vertical_grasp.npz')
                    )
                    
                    save_dict = {
                        'best_grasp': best_grasp_original,
                        'best_grasp_7d': pose_7d,
                        'best_conf': best_conf,
                        'obj_pc': obj_pc_filtered,
                        'obj_pc_color': obj_pc_color_filtered,
                        'num_vertical_grasps': num_vertical,
                        'all_angles': all_angles,
                    }
                    
                    if scene_pc is not None:
                        save_dict['scene_pc'] = scene_pc
                        save_dict['scene_pc_color'] = scene_pc_color
                    
                    np.savez(output_file, **save_dict)
                    print(f"    Saved to: {output_file}")
            
            else:
                # Mode 2: Show all grasps mode
                print(f"\n  [SHOW ALL GRASPS MODE]")
                print(f"  Visualizing {len(collision_free_grasps)} collision-free grasps with color-coded confidence...")
                
                # Visualize all collision-free grasps with color based on confidence
                for idx in range(len(collision_free_grasps)):
                    visualize_grasp(
                        vis,
                        f"collision_free_grasp_{idx}",
                        collision_free_grasps[idx],
                        color=collision_free_scores[idx],
                        gripper_name=gripper_name,
                        linewidth=1.0,
                    )
                
                # Optionally visualize colliding grasps in red
                if colliding_grasps is not None and len(colliding_grasps) > 0:
                    print(f"  Visualizing {len(colliding_grasps)} colliding grasps (red, thin)...")
                    for idx in range(len(colliding_grasps)):
                        visualize_grasp(
                            vis,
                            f"colliding_grasp_{idx}",
                            colliding_grasps[idx],
                            color=[255, 0, 0],
                            gripper_name=gripper_name,
                            linewidth=0.5,
                        )
                
                # Highlight the best grasp
                best_idx = np.argmax(collision_free_conf)
                best_grasp = collision_free_grasps[best_idx]
                best_conf = collision_free_conf[best_idx]
                
                print(f"  Best grasp (highest confidence): #{best_idx}")
                print(f"  Best grasp confidence: {best_conf:.3f}")
                
                # Transform best grasp back to original coordinate system
                best_grasp_original = tra.inverse_matrix(T_center) @ best_grasp
                
                # Convert to 7D list and print
                pose_7d = pose_to_7d_list(best_grasp_original)
                print(f"\n  📍 Best Grasp Pose (Camera Frame): {pose_7d}")
                
                # Transform to world frame if camera pose is provided
                if args.camera_pose_world is not None:
                    pose_7d_world = transform_pose_to_world(pose_7d, args.camera_pose_world)
                    print(f"  📍 Best Grasp Pose (World Frame):  {pose_7d_world}")
                
                # Highlight best grasp with green and thicker line
                visualize_grasp(
                    vis,
                    f"best_grasp_highlight",
                    best_grasp_original,
                    color=[0, 255, 0],
                    gripper_name=gripper_name,
                    linewidth=3.0,
                )
                
                # Optional: Save results
                if args.output_dir is not None:
                    os.makedirs(args.output_dir, exist_ok=True)
                    output_file = os.path.join(
                        args.output_dir,
                        os.path.basename(obj_ply_file).replace('_object.ply', '_all_grasps.npz')
                    )
                    
                    # Transform all collision-free grasps back to original coordinate system
                    collision_free_grasps_original = np.array([
                        tra.inverse_matrix(T_center) @ g for g in collision_free_grasps
                    ])
                    
                    save_dict = {
                        'all_collision_free_grasps': collision_free_grasps_original,
                        'all_confidences': collision_free_conf,
                        'best_grasp_idx': best_idx,
                        'best_grasp': best_grasp_original,
                        'best_grasp_7d': pose_7d,
                        'best_conf': best_conf,
                        'obj_pc': obj_pc_filtered,
                        'obj_pc_color': obj_pc_color_filtered,
                    }
                    
                    if scene_pc is not None:
                        save_dict['scene_pc'] = scene_pc
                        save_dict['scene_pc_color'] = scene_pc_color
                    
                    np.savez(output_file, **save_dict)
                    print(f"    Saved to: {output_file}")

        else:
            print("  ⚠️  No grasps found!")

        print(f"{'='*80}")
        input("\nPress Enter to continue to next scene...")


def load_camera_pose(json_path: str, frame_index: int):
    """
    Load camera pose from robot state JSON file.
    Camera pose = End effector pose at given frame.
    
    Args:
        json_path: Path to robot_state.json
        frame_index: Frame index to extract pose from
        
    Returns:
        list: [x, y, z, qx, qy, qz, qw]
    """
    import json
    
    with open(json_path, 'r') as f:
        robot_state = json.load(f)
    
    # gripper_pose shape: (num_frames, 7)
    camera_pose = robot_state['gripper_pose'][frame_index]
    
    return camera_pose

# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="GraspGen: Generate grasps from PLY point clouds (Object or Scene mode)"
    )
    
    # === Mode Selection ===
    parser.add_argument(
        "--mode",
        type=str,
        choices=["object", "scene"],
        default="object",
        help="Processing mode: 'object' for simple object PCs, 'scene' for scene-aware with collision detection"
    )
    
    # === Input/Output ===
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        required=True,
        help="Directory containing PLY files. Object mode: *.ply; Scene mode: *_object.ply and *_scene.ply"
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        required=True,
        help="Path to gripper configuration YAML file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save grasp results (optional)"
    )
    
    # === Grasp Generation Parameters ===
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.8,
        help="Threshold for valid grasps"
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasps to generate"
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Whether to return only the top k grasps"
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=-1,
        help="Number of top grasps to return"
    )
    parser.add_argument(
        "--vertical_angle_threshold",
        type=float,
        default=30.0,
        help="Maximum angle (degrees) from vertical (downward) direction to consider grasp as vertical"
    )
    parser.add_argument(
        "--verticality_weight",
        type=float,
        default=0.7,
        help="Weight for verticality when selecting best grasp (0-1). "
             "0 = only confidence, 1 = only verticality, 0.7 = prefer vertical grasps"
    )
    parser.add_argument(
        "--vertical_first",
        action="store_true",
        help="If set, only show best vertical grasp. If not set, show all grasps with color-coded confidence"
    )
    parser.add_argument(
        "--camera_pose_world",
        type=float,
        nargs=7,
        default=None,
        metavar=('X', 'Y', 'Z', 'QX', 'QY', 'QZ', 'QW'),
        help="Camera pose in world frame: [x, y, z, qx, qy, qz, qw]"
    )
    parser.add_argument(
        "--robot_state_json",
        type=str,
        default="/home/yuzhench/Desktop/Research/Media_lab/RLBench_tasks/Paper_Demo/OpenJar/OpenJar_Demo_004/robot_state/robot_state.json",
        help="Path to robot_state.json file (alternative to --camera_pose_world)"
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=50,
        help="Frame index to extract camera pose from robot_state.json"
    )
    
    # === Scene Mode: Collision Detection ===
    parser.add_argument(
        "--filter_collisions",
        action="store_true",
        help="[Scene mode only] Filter grasps based on collision detection"
    )
    parser.add_argument(
        "--collision_threshold",
        type=float,
        default=0.02,
        help="[Scene mode only] Distance threshold for collision detection (meters)"
    )
    parser.add_argument(
        "--max_scene_points",
        type=int,
        default=8192,
        help="[Scene mode only] Max scene points for collision checking"
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    args = parse_args()

    # Validate inputs
    if not os.path.exists(args.gripper_config):
        raise ValueError(f"Gripper config not found: {args.gripper_config}")
    
    if not os.path.exists(args.sample_data_dir):
        raise ValueError(f"Sample data directory not found: {args.sample_data_dir}")

    # Handle return_topk logic
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    # Load camera pose from robot_state.json if provided
    if args.robot_state_json is not None:
        args.camera_pose_world = load_camera_pose(args.robot_state_json, args.frame_index)
        print(f"Loaded camera pose from frame {args.frame_index}: {args.camera_pose_world}")

    # Load gripper config
    print("\n" + "="*80)
    print("Initializing GraspGen")
    print("="*80)
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    print(f"Gripper: {gripper_name}")
    print(f"Config: {args.gripper_config}")

    # Initialize GraspGenSampler
    grasp_sampler = GraspGenSampler(grasp_cfg)
    print("✓ Model loaded")

    # Create visualizer
    vis = create_visualizer()

    # Run appropriate mode
    if args.mode == "object":
        run_object_mode(args, grasp_sampler, gripper_name, vis)
    else:  # scene
        run_scene_mode(args, grasp_sampler, gripper_name, vis)
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)
