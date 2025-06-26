"""
3D mapping and camera tracking module for ORB-SLAM2 Python implementation.

This module handles triangulation of 3D points from matched features
and tracks camera position over time.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


class Triangulator:
    """
    Class for triangulating 3D points from matched features in stereo or consecutive frames.
    """
    
    def __init__(self, camera_matrix, dist_coeffs=None):
        """
        Initialize the triangulator.
        
        Args:
            camera_matrix (np.array): 3x3 camera intrinsic matrix.
            dist_coeffs (np.array, optional): Distortion coefficients.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.K_inv = np.linalg.inv(camera_matrix)
        
    def triangulate_points(self, pts1, pts2, pose1, pose2):
        """
        Triangulate 3D points from matched points in two views.
        
        Args:
            pts1 (np.array): 2D points in first view (Nx2).
            pts2 (np.array): 2D points in second view (Nx2).
            pose1 (np.array): First camera pose as 3x4 [R|t] matrix.
            pose2 (np.array): Second camera pose as 3x4 [R|t] matrix.
            
        Returns:
            np.array: Triangulated 3D points (Nx3).
        """
        # Convert to homogeneous coordinates
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), 
                                       self.camera_matrix, 
                                       self.dist_coeffs)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), 
                                       self.camera_matrix, 
                                       self.dist_coeffs)
        
        # Flatten points
        pts1_norm = pts1_norm.reshape(-1, 2)
        pts2_norm = pts2_norm.reshape(-1, 2)
        
        # Calculate projection matrices
        P1 = self.camera_matrix @ pose1
        P2 = self.camera_matrix @ pose2
        
        # Triangulate points
        pts_4d = cv2.triangulatePoints(P1, P2, 
                                     pts1_norm.T, 
                                     pts2_norm.T)
        
        # Convert from homogeneous to 3D coordinates
        pts_3d = pts_4d[:3, :] / pts_4d[3, :]
        pts_3d = pts_3d.T
        
        return pts_3d
    
    def compute_fundamental_matrix(self, pts1, pts2):
        """
        Compute fundamental matrix between two views.
        
        Args:
            pts1 (np.array): 2D points in first view (Nx2).
            pts2 (np.array): 2D points in second view (Nx2).
            
        Returns:
            tuple: (F, mask) - Fundamental matrix and inlier mask.
        """
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
        return F, mask.ravel().astype(bool)
    
    def estimate_pose_from_essential(self, pts1, pts2):
        """
        Estimate relative camera pose from matched points.
        
        Args:
            pts1 (np.array): 2D points in first view (Nx2).
            pts2 (np.array): 2D points in second view (Nx2).
            
        Returns:
            tuple: (R, t, mask) - Rotation, translation and inlier mask.
        """
        # Normalize points
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), 
                                       self.camera_matrix, 
                                       self.dist_coeffs)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), 
                                       self.camera_matrix, 
                                       self.dist_coeffs)
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, cv2.RANSAC, 0.999, 1.0)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        
        return R, t, mask.ravel().astype(bool)
    
    def filter_points_by_triangulation_angle(self, pts_3d, pose1, pose2, min_angle_deg=1.0):
        """
        Filter 3D points based on triangulation angle.
        
        Args:
            pts_3d (np.array): 3D points (Nx3).
            pose1 (np.array): First camera pose as 3x4 [R|t] matrix.
            pose2 (np.array): Second camera pose as 3x4 [R|t] matrix.
            min_angle_deg (float): Minimum triangulation angle in degrees.
            
        Returns:
            np.array: Boolean mask of valid points.
        """
        # Extract camera centers (last column of inverse pose matrix)
        C1 = -np.linalg.inv(pose1[:3, :3]) @ pose1[:3, 3]
        C2 = -np.linalg.inv(pose2[:3, :3]) @ pose2[:3, 3]
        
        # Calculate angles
        valid_mask = np.ones(len(pts_3d), dtype=bool)
        min_angle_rad = np.deg2rad(min_angle_deg)
        
        for i, pt in enumerate(pts_3d):
            v1 = pt - C1
            v2 = pt - C2
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            if angle < min_angle_rad:
                valid_mask[i] = False
                
        return valid_mask


class SLAMVisualizer:
    """
    Class for visualizing 3D map and camera trajectory.
    """
    
    def __init__(self):
        """
        Initialize the visualizer.
        """
        self.point_cloud = o3d.geometry.PointCloud()
        self.camera_poses = []
        self.trajectory_lines = o3d.geometry.LineSet()
        
    def update_point_cloud(self, points, colors=None):
        """
        Update the 3D point cloud.
        
        Args:
            points (np.array): 3D points (Nx3).
            colors (np.array, optional): RGB colors for points (Nx3).
        """
        if len(points) == 0:
            return
            
        # Convert to Open3D format
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            # Ensure colors are in range [0, 1]
            if np.max(colors) > 1:
                colors = colors / 255.0
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Default color: white
            self.point_cloud.paint_uniform_color([1, 1, 1])
    
    def add_camera_pose(self, pose):
        """
        Add a camera pose to the trajectory.
        
        Args:
            pose (np.array): 3x4 camera pose [R|t].
        """
        # Extract camera center (last column of inverse pose matrix)
        R = pose[:3, :3]
        t = pose[:3, 3]
        C = -R.T @ t
        
        self.camera_poses.append(C)
        
        # Update trajectory lines if we have at least two poses
        if len(self.camera_poses) > 1:
            points = np.array(self.camera_poses)
            lines = [[i, i+1] for i in range(len(points)-1)]
            
            self.trajectory_lines.points = o3d.utility.Vector3dVector(points)
            self.trajectory_lines.lines = o3d.utility.Vector2iVector(lines)
            self.trajectory_lines.paint_uniform_color([1, 0, 0])  # Red trajectory
    
    def create_camera_frustum(self, pose, scale=0.1):
        """
        Create a camera frustum mesh for visualization.
        
        Args:
            pose (np.array): 3x4 camera pose [R|t].
            scale (float): Scale for the frustum.
            
        Returns:
            o3d.geometry.LineSet: Camera frustum as line set.
        """
        # Define canonical frustum vertices
        frustum_pts = np.array([
            [0, 0, 0],               # Camera center
            [scale, scale, scale],   # Top-right-front
            [scale, -scale, scale],  # Bottom-right-front
            [-scale, -scale, scale], # Bottom-left-front
            [-scale, scale, scale]   # Top-left-front
        ])
        
        # Define lines connecting frustum vertices
        frustum_lines = np.array([
            [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Front face
        ])
        
        # Transform vertices according to pose
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        transformed_pts = []
        C = -R.T @ t  # Camera center
        
        for pt in frustum_pts:
            if np.array_equal(pt, np.zeros(3)):
                transformed_pts.append(C)
            else:
                transformed_pts.append(C + R.T @ pt)
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(transformed_pts)
        line_set.lines = o3d.utility.Vector2iVector(frustum_lines)
        line_set.paint_uniform_color([0, 1, 0])  # Green frustum
        
        return line_set
    
    def visualize(self, show_cameras=True):
        """
        Visualize the point cloud and camera trajectory.
        
        Args:
            show_cameras (bool): Whether to show camera frustums.
        """
        # Create visualization elements
        geometries = [self.point_cloud, self.trajectory_lines]
        
        # Add camera frustums if needed
        if show_cameras and len(self.camera_poses) > 0:
            for i, pose in enumerate(self.camera_poses):
                # Create full pose matrix from position (assuming identity rotation for simplicity)
                full_pose = np.eye(3, 4)
                full_pose[:3, 3] = pose
                frustum = self.create_camera_frustum(full_pose)
                geometries.append(frustum)
        
        # Visualize using Open3D
        o3d.visualization.draw_geometries(geometries)
    
    def visualize_with_matplotlib(self):
        """
        Visualize using matplotlib (alternative to Open3D).
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D points
        points = np.asarray(self.point_cloud.points)
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                     s=1, c='blue', alpha=0.5)
        
        # Plot camera trajectory
        if len(self.camera_poses) > 1:
            poses = np.array(self.camera_poses)
            ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], 
                  'r-', linewidth=2)
            ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2], 
                     c='red', s=20)
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        max_range = 0
        if len(points) > 0:
            for dim in range(3):
                ptp = np.ptp(points[:, dim])
                if ptp > max_range:
                    max_range = ptp
                    
        if max_range == 0:
            max_range = 1
            
        # Set axis limits
        mid_x = np.mean(points[:, 0]) if len(points) > 0 else 0
        mid_y = np.mean(points[:, 1]) if len(points) > 0 else 0
        mid_z = np.mean(points[:, 2]) if len(points) > 0 else 0
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        plt.tight_layout()
        plt.show()


class VisualSLAM:
    """
    Main Visual SLAM class combining feature extraction, matching, 
    triangulation and visualization.
    """
    
    def __init__(self, camera_matrix, dist_coeffs=None, extractor=None, matcher=None):
        """
        Initialize the Visual SLAM system.
        
        Args:
            camera_matrix (np.array): 3x3 camera intrinsic matrix.
            dist_coeffs (np.array, optional): Distortion coefficients.
            extractor (ORBExtractor, optional): Feature extractor instance.
            matcher (DescriptorMatcher, optional): Feature matcher instance.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        
        # Initialize components
        if extractor is None:
            from .extractor import ORBExtractor
            self.extractor = ORBExtractor(n_features=2000)
        else:
            self.extractor = extractor
            
        if matcher is None:
            from .matcher import DescriptorMatcher
            self.matcher = DescriptorMatcher(matcher_type='bruteforce-hamming')
        else:
            self.matcher = matcher
            
        self.triangulator = Triangulator(camera_matrix, dist_coeffs)
        self.visualizer = SLAMVisualizer()
        
        # Store state
        self.keyframes = []
        self.curr_pose = np.eye(3, 4)  # Identity rotation, zero translation [R|t]
        self.prev_keyframe = None
        self.map_points = []
        self.map_colors = []
    
    def _compose_poses(self, pose1, pose2):
        """
        Compose two 3x4 camera poses.
        
        Args:
            pose1 (np.array): First camera pose as 3x4 [R|t] matrix.
            pose2 (np.array): Second camera pose as 3x4 [R|t] matrix.
            
        Returns:
            np.array: Composed camera pose as 3x4 [R|t] matrix.
        """
        # Extract rotation and translation from first pose
        R1 = pose1[:3, :3]
        t1 = pose1[:3, 3]
        
        # Extract rotation and translation from second pose
        R2 = pose2[:3, :3]
        t2 = pose2[:3, 3]
        
        # Compose: R = R1 @ R2, t = R1 @ t2 + t1
        R_result = R1 @ R2
        t_result = R1 @ t2 + t1
        
        # Create result pose matrix
        result = np.zeros((3, 4))
        result[:3, :3] = R_result
        result[:3, 3] = t_result
        
        return result
    
    def process_frame(self, frame, is_keyframe=None):
        """
        Process a new frame.
        
        Args:
            frame (np.array): Input image.
            is_keyframe (bool, optional): Force frame to be a keyframe or not.
                If None, automatic selection is used.
                
        Returns:
            tuple: (pose, points_3d, is_keyframe)
        """
        # Extract features
        keypoints, descriptors = self.extractor.extract_features(frame, distributed=True)
        
        # Skip if no features found
        if len(keypoints) == 0:
            return self.curr_pose, [], False
        
        new_points_3d = []
        
        # If we have a previous keyframe, match and update map
        if self.prev_keyframe is not None:
            prev_kf = self.prev_keyframe
            
            # Match features
            matches = self.matcher.match(prev_kf['descriptors'], descriptors)
            
            # Filter matches by fundamental matrix
            filtered_matches, mask = self.matcher.filter_matches_by_fundamental(
                prev_kf['keypoints'], keypoints, matches)
            
            print("N matches filtrados:")
            print(len(filtered_matches))
            # Skip if not enough matches
            if len(filtered_matches) < 20:
                return self.curr_pose, [], False
            
            # Extract matched points
            pts1 = np.float32([prev_kf['keypoints'][m.queryIdx].pt for m in filtered_matches])
            pts2 = np.float32([keypoints[m.trainIdx].pt for m in filtered_matches])
            
            # Estimate pose
            R, t, pose_mask = self.triangulator.estimate_pose_from_essential(pts1, pts2)
            
            # Form current camera pose (relative to previous keyframe)
            new_pose = np.hstack((R, t))
            
            # Update global pose (using proper pose composition)
            self.curr_pose = self._compose_poses(prev_kf['pose'], new_pose)
            
            # Decide if this is a keyframe
            if is_keyframe is None:
                # Auto keyframe selection based on number of inliers
                is_keyframe = len(filtered_matches) < 0.7 * len(prev_kf['keypoints'])
            
            # If it's a keyframe, triangulate and add points to map
            if is_keyframe:
                # Keep only pose inliers
                pts1 = pts1[pose_mask]
                pts2 = pts2[pose_mask]
                
                # Get pose matrices
                pose1 = prev_kf['pose']
                pose2 = self.curr_pose
                
                # Triangulate points
                points_3d = self.triangulator.triangulate_points(pts1, pts2, pose1, pose2)
                
                # Filter points by triangulation angle
                angle_mask = self.triangulator.filter_points_by_triangulation_angle(
                    points_3d, pose1, pose2)
                points_3d = points_3d[angle_mask]
                
                # Get colors for these points (from current frame)
                if len(points_3d) > 0:
                    # Project points to current frame to get colors
                    proj_pts, _ = cv2.projectPoints(
                        points_3d, 
                        self.curr_pose[:3, :3], 
                        self.curr_pose[:3, 3], 
                        self.camera_matrix, 
                        self.dist_coeffs
                    )
                    proj_pts = proj_pts.reshape(-1, 2)
                    
                    # Get colors from image
                    colors = []
                    frame_gray = frame
                    if len(frame.shape) == 3:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                    for pt in proj_pts:
                        x, y = int(pt[0]), int(pt[1])
                        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                            if len(frame.shape) == 3:
                                color = frame[y, x][::-1] / 255.0  # BGR to RGB, normalize to [0,1]
                            else:
                                gray = frame_gray[y, x] / 255.0
                                color = np.array([gray, gray, gray])  # Convert grayscale to RGB
                            colors.append(color)
                        else:
                            colors.append(np.array([1.0, 1.0, 1.0]))  # White for out-of-bound points
                    
                    # Add points and colors to map
                    self.map_points.extend(points_3d.tolist())
                    self.map_colors.extend(colors)
                    new_points_3d = points_3d
                
                # Store keyframe
                self.keyframes.append({
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'pose': self.curr_pose.copy(),
                    'frame': frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                })
                self.prev_keyframe = self.keyframes[-1]
                
                # Update visualizer
                self.visualizer.update_point_cloud(np.array(self.map_points), np.array(self.map_colors))
                self.visualizer.add_camera_pose(self.curr_pose)
        else:
            # First frame, just initialize
            is_keyframe = True
            self.keyframes.append({
                'keypoints': keypoints,
                'descriptors': descriptors,
                'pose': self.curr_pose.copy(),
                'frame': frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            })
            self.prev_keyframe = self.keyframes[-1]
            self.visualizer.add_camera_pose(self.curr_pose)
            
        return self.curr_pose, new_points_3d, is_keyframe
    
    def process_video(self, video_path, keyframe_interval=None, visualize_freq=10):
        """
        Process a video file.
        
        Args:
            video_path (str): Path to video file.
            keyframe_interval (int, optional): Force keyframe every N frames. 
                If None, automatic selection is used.
            visualize_freq (int): Update visualization every N frames.
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        plt.figure(figsize=(16, 8))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale for feature extraction
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame
                
            # Process frame
            is_keyframe = None
            if keyframe_interval is not None:
                is_keyframe = (frame_count % keyframe_interval == 0)
                
            pose, new_points, is_kf = self.process_frame(frame, is_keyframe)
            
            # Visualization
            if frame_count % visualize_freq == 0:
                plt.clf()
                
                # Left: Input video with features
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.title(f"Frame {frame_count}" + (" (Keyframe)" if is_kf else ""))
                plt.axis('off')
                
                # Right: 3D map (top view)
                plt.subplot(1, 2, 2)
                if len(self.map_points) > 0:
                    points = np.array(self.map_points)
                    plt.scatter(points[:, 0], points[:, 2], s=1, alpha=0.5)
                    
                    # Plot camera trajectory
                    if len(self.keyframes) > 1:
                        trajectory = np.array([kf['pose'][:3, 3] for kf in self.keyframes])
                        plt.plot(trajectory[:, 0], trajectory[:, 2], 'r-', linewidth=2)
                        plt.scatter(trajectory[:, 0], trajectory[:, 2], c='red', s=20)
                        
                    plt.title("3D Map (Top View)")
                    plt.axis('equal')
                
                plt.tight_layout()
                plt.pause(0.01)
            
            frame_count += 1
            
        cap.release()
        
    def visualize_map(self):
        """
        Visualize the final 3D map and camera trajectory.
        """
        self.visualizer.visualize()
    
    def visualize_map_matplotlib(self):
        """
        Visualize using matplotlib instead of Open3D.
        """
        self.visualizer.visualize_with_matplotlib()