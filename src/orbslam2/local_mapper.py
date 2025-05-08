"""
Local mapping module for ORB-SLAM2 Python implementation.

This module handles keyframe processing, map point creation, and map refinement.
"""

import numpy as np
import cv2
from collections import defaultdict
from .utils import triangulate_points, convert_to_3d_points, compute_projection_matrix, create_point_cloud_ply


class LocalMapper:
    """
    Class for local mapping, creating and optimizing the map.
    """
    
    def __init__(self, camera_matrix, output_path=None):
        """
        Initialize the local mapper.
        
        Args:
            camera_matrix (np.array): 3x3 camera intrinsic matrix.
            output_path (str): Path to save PLY map file.
        """
        self.camera_matrix = camera_matrix
        self.keyframes = []  # List of keyframes (image, keypoints, descriptors, pose)
        self.map_points = []  # List of map points (3D position, observations)
        self.output_path = output_path if output_path else "map.ply"
        
        # For co-visibility graph
        self.co_visibility_graph = defaultdict(lambda: defaultdict(int))
        
        # Parameters for map maintenance
        self.redundancy_threshold = 0.9  # Threshold for keyframe redundancy
        self.min_observations = 2  # Minimum observations for a valid map point
        self.culling_threshold = 0.05  # Threshold for map point culling (reprojection error)
    
    def set_output_path(self, output_path):
        """
        Set the output path for the PLY map file.
        
        Args:
            output_path (str): Path to save PLY map file.
        """
        self.output_path = output_path
    
    def add_keyframe(self, image, keypoints, descriptors, pose):
        """
        Add a new keyframe to the map.
        
        Args:
            image (np.array): Keyframe image.
            keypoints (list): List of keypoints.
            descriptors (np.array): Descriptors.
            pose (np.array): 4x4 camera pose (world to camera).
        """
        keyframe_id = len(self.keyframes)
        
        # Store keyframe
        keyframe = {
            'id': keyframe_id,
            'image': image.copy(),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose.copy(),
            'map_points': []  # List of map point references visible in this keyframe
        }
        
        self.keyframes.append(keyframe)
        
        # Process new keyframe: triangulate new map points, etc.
        if len(self.keyframes) > 1:
            self._process_new_keyframe(keyframe)
            
        # Update map if we have enough keyframes
        if len(self.keyframes) >= 2:
            self._update_map()
    
    def update_map_points(self, new_map_points):
        """
        Update map with new map points (used during initialization).
        
        Args:
            new_map_points (list): List of new map points.
        """
        # Merge new map points with existing ones (during initialization)
        for mp in new_map_points:
            self.map_points.append(mp)
    
    def _process_new_keyframe(self, new_keyframe):
        """
        Process a new keyframe to create new map points.
        
        Args:
            new_keyframe (dict): New keyframe dictionary.
        """
        # Get the latest two keyframes
        current_kf = new_keyframe
        prev_kf_idx = -2  # Second to last keyframe
        if abs(prev_kf_idx) > len(self.keyframes):
            return
        
        prev_kf = self.keyframes[prev_kf_idx]
        
        # Get camera poses
        pose1 = prev_kf['pose']
        pose2 = current_kf['pose']
        
        # Compute projection matrices
        P1 = compute_projection_matrix(pose1[:3, :3], pose1[:3, 3], self.camera_matrix)
        P2 = compute_projection_matrix(pose2[:3, :3], pose2[:3, 3], self.camera_matrix)
        
        # Match features between keyframes (in a real implementation, 
        # this would be more sophisticated with local bundle adjustment)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(prev_kf['descriptors'], current_kf['descriptors'], k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) >= 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
        
        # Check if we have enough matches
        if len(good_matches) < 8:
            return
        
        # Create 3D points by triangulation
        points1 = np.float32([prev_kf['keypoints'][m.queryIdx].pt for m in good_matches])
        points2 = np.float32([current_kf['keypoints'][m.trainIdx].pt for m in good_matches])
        
        # Calculate fundamental matrix to filter outliers
        F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 3.0)
        
        if mask is None or F is None:
            return
            
        # Keep inliers only
        mask = mask.ravel() > 0
        points1 = points1[mask]
        points2 = points2[mask]
        matches_filtered = [m for m, keep in zip(good_matches, mask) if keep]
        
        # Triangulate points
        points_4d = triangulate_points(points1, points2, P1, P2)
        points_3d = convert_to_3d_points(points_4d)
        
        # Create new map points
        for i, (point_3d, match) in enumerate(zip(points_3d, matches_filtered)):
            # Extract color from image
            x, y = int(prev_kf['keypoints'][match.queryIdx].pt[0]), int(prev_kf['keypoints'][match.queryIdx].pt[1])
            color = None
            
            if 0 <= x < prev_kf['image'].shape[1] and 0 <= y < prev_kf['image'].shape[0]:
                if len(prev_kf['image'].shape) == 3:  # Color image
                    color = prev_kf['image'][y, x, :]
                else:  # Grayscale
                    color = np.array([prev_kf['image'][y, x]] * 3)
            
            if color is None:
                color = np.array([0, 0, 255])  # Default blue
            
            # Create map point
            map_point = {
                'id': len(self.map_points),
                'position': point_3d,
                'color': color,
                'observed_keyframes': {
                    prev_kf['id']: match.queryIdx,  # Keyframe ID -> keypoint ID
                    current_kf['id']: match.trainIdx
                },
                'descriptor': prev_kf['descriptors'][match.queryIdx]  # Reference descriptor
            }
            
            # Add map point
            self.map_points.append(map_point)
            
            # Update keyframe references
            prev_kf['map_points'].append(map_point['id'])
            current_kf['map_points'].append(map_point['id'])
            
            # Update co-visibility graph
            self.co_visibility_graph[prev_kf['id']][current_kf['id']] += 1
            self.co_visibility_graph[current_kf['id']][prev_kf['id']] += 1
    
    def _update_map(self):
        """
        Update the map by removing outliers, optimizing positions, etc.
        """
        # In a full SLAM system, this would include:
        # 1. Local bundle adjustment
        # 2. Map point culling
        # 3. Keyframe culling
        
        if len(self.map_points) > 0:
            self._cull_map_points()
        
        if len(self.keyframes) > 3:
            self._cull_keyframes()
        
        # For this simplified version, just save the current map
        self._save_map()
    
    def _cull_map_points(self):
        """
        Remove low-quality map points.
        """
        # Criteria for removing points:
        # 1. Points observed in fewer than min_observations keyframes
        # 2. Points with high reprojection error
        
        valid_map_points = []
        for mp in self.map_points:
            # Check observation count
            if len(mp['observed_keyframes']) < self.min_observations:
                continue
            
            # Check reprojection error (simplified)
            valid = True
            reprojection_errors = []
            
            for kf_id, kp_id in mp['observed_keyframes'].items():
                kf = self.keyframes[kf_id]
                kp = kf['keypoints'][kp_id]
                
                # Project 3D point to image
                P = compute_projection_matrix(kf['pose'][:3, :3], kf['pose'][:3, 3], self.camera_matrix)
                point_3d_homogeneous = np.append(mp['position'], 1.0)
                point_proj_homogeneous = P @ point_3d_homogeneous
                point_proj = point_proj_homogeneous[:2] / point_proj_homogeneous[2]
                
                # Calculate reprojection error
                kp_pt = np.array(kp.pt)
                error = np.linalg.norm(point_proj - kp_pt)
                reprojection_errors.append(error)
                
                # Check if error is too large
                if error > 5.0:  # Threshold in pixels
                    valid = False
                    break
            
            # Keep the point if it's valid
            if valid:
                valid_map_points.append(mp)
        
        # Update map points
        self.map_points = valid_map_points
        
        # Update keyframe references to map points
        for kf in self.keyframes:
            kf['map_points'] = [mp['id'] for mp in self.map_points if kf['id'] in mp['observed_keyframes']]
    
    def _cull_keyframes(self):
        """
        Remove redundant keyframes.
        """
        # Criteria for removing keyframes:
        # 1. Not the most recent keyframe
        # 2. Most map points are observed by other keyframes
        # 3. Has high co-visibility with other keyframes
        
        # Don't remove the first two keyframes (needed for initialization) 
        # or the most recent one (needed for tracking)
        if len(self.keyframes) <= 3:
            return
        
        # Check keyframes from 1 to N-2 (skip first and last two)
        keyframes_to_remove = []
        
        for i in range(1, len(self.keyframes) - 2):
            kf = self.keyframes[i]
            
            # Skip if this keyframe doesn't have enough map points
            if len(kf['map_points']) < 20:
                continue
            
            # Check redundancy: percentage of map points visible in other keyframes
            redundant_count = 0
            for mp_id in kf['map_points']:
                # Find map point
                mp = next((p for p in self.map_points if p['id'] == mp_id), None)
                if mp is None:
                    continue
                
                # Count observations in other keyframes
                other_observations = sum(1 for kf_id in mp['observed_keyframes'] if kf_id != kf['id'])
                
                if other_observations >= 3:  # Visible in at least 3 other keyframes
                    redundant_count += 1
            
            # Calculate redundancy ratio
            redundancy_ratio = redundant_count / len(kf['map_points']) if kf['map_points'] else 0
            
            # If redundancy is high, consider removing this keyframe
            if redundancy_ratio > self.redundancy_threshold:
                keyframes_to_remove.append(i)
        
        # Sort indices in descending order to avoid index shifting
        keyframes_to_remove.sort(reverse=True)
        
        # Remove keyframes
        for idx in keyframes_to_remove:
            # Update co-visibility graph
            kf_id = self.keyframes[idx]['id']
            for other_kf_id in self.co_visibility_graph[kf_id]:
                if other_kf_id != kf_id:
                    del self.co_visibility_graph[other_kf_id][kf_id]
            del self.co_visibility_graph[kf_id]
            
            # Remove keyframe
            self.keyframes.pop(idx)
        
        # Update keyframe IDs if needed
        for i, kf in enumerate(self.keyframes):
            kf['id'] = i
    
    def _filter_map_points(self):
        """
        Filter map points to remove outliers.
        
        Returns:
            list: Filtered map points.
        """
        filtered_points = []
        filtered_colors = []
        
        for mp in self.map_points:
            # Simple filtering: keep points observed in at least 2 keyframes
            if len(mp['observed_keyframes']) >= self.min_observations:
                filtered_points.append(mp['position'])
                filtered_colors.append(mp['color'])
        
        return np.array(filtered_points), np.array(filtered_colors)
    
    def _save_map(self):
        """
        Save the map as a PLY file.
        """
        if not self.map_points:
            return
            
        # Filter points
        points, colors = self._filter_map_points()
        
        if len(points) == 0:
            return
            
        # Save to PLY
        create_point_cloud_ply(points, colors, self.output_path)
        print(f"Saved map with {len(points)} points to {self.output_path}")
    
    def get_map_statistics(self):
        """
        Get statistics about the current map.
        
        Returns:
            dict: Map statistics.
        """
        stats = {
            'num_keyframes': len(self.keyframes),
            'num_map_points': len(self.map_points),
            'num_filtered_points': len(self._filter_map_points()[0]),
            'avg_observations_per_point': np.mean([len(mp['observed_keyframes']) for mp in self.map_points]) if self.map_points else 0,
            'map_density': len(self.map_points) / len(self.keyframes) if len(self.keyframes) > 0 else 0
        }
        return stats
    
    def visualize_map(self, width=800, height=600):
        """
        Create a simple visualization of the map and camera trajectory.
        
        Args:
            width (int): Image width.
            height (int): Image height.
            
        Returns:
            np.array: Visualization image.
        """
        # Create black image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get map points
        points, colors = self._filter_map_points()
        
        if len(points) == 0:
            return img
        
        # Find bounds of points for scaling
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        extent = max_vals - min_vals
        
        # Scale points to fit in image (use X and Z coordinates for top-down view)
        margin = 50
        scale_x = (width - 2 * margin) / extent[0] if extent[0] > 0 else 1
        scale_z = (height - 2 * margin) / extent[2] if extent[2] > 0 else 1
        scale = min(scale_x, scale_z)
        
        # Draw points
        for point, color in zip(points, colors):
            # Project to 2D (top-down view)
            x = int(margin + (point[0] - min_vals[0]) * scale)
            y = int(margin + (point[2] - min_vals[2]) * scale)  # Use Z as Y in top-down view
            
            # Draw point
            cv2.circle(img, (x, y), 1, color.tolist(), -1)
        
        # Draw camera trajectory
        if len(self.keyframes) > 1:
            for i in range(1, len(self.keyframes)):
                # Get camera centers
                pose1 = self.keyframes[i-1]['pose']
                pose2 = self.keyframes[i]['pose']
                
                # Camera center is -R^T * t
                R1, t1 = pose1[:3, :3], pose1[:3, 3]
                R2, t2 = pose2[:3, :3], pose2[:3, 3]
                
                c1 = -R1.T @ t1
                c2 = -R2.T @ t2
                
                # Project to 2D
                x1 = int(margin + (c1[0] - min_vals[0]) * scale)
                y1 = int(margin + (c1[2] - min_vals[2]) * scale)
                x2 = int(margin + (c2[0] - min_vals[0]) * scale)
                y2 = int(margin + (c2[2] - min_vals[2]) * scale)
                
                # Draw trajectory
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            # Draw current camera position
            pose = self.keyframes[-1]['pose']
            R, t = pose[:3, :3], pose[:3, 3]
            c = -R.T @ t
            x = int(margin + (c[0] - min_vals[0]) * scale)
            y = int(margin + (c[2] - min_vals[2]) * scale)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        
        # Add legend and info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Map Points: {len(points)}", (10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Keyframes: {len(self.keyframes)}", (10, 40), font, 0.5, (255, 255, 255), 1)
        
        return img
    
    def get_camera_trajectory(self):
        """
        Get the camera trajectory for visualization.
        
        Returns:
            np.array: Array of camera positions.
        """
        trajectory = []
        for kf in self.keyframes:
            pose = kf['pose']
            R, t = pose[:3, :3], pose[:3, 3]
            # Camera center is -R^T * t
            c = -R.T @ t
            trajectory.append(c)
        
        return np.array(trajectory) if trajectory else np.array([])
    
    def get_keyframe_by_id(self, keyframe_id):
        """
        Get a keyframe by its ID.
        
        Args:
            keyframe_id (int): Keyframe ID.
            
        Returns:
            dict: Keyframe or None if not found.
        """
        for kf in self.keyframes:
            if kf['id'] == keyframe_id:
                return kf
        return None
    
    def get_map_point_by_id(self, map_point_id):
        """
        Get a map point by its ID.
        
        Args:
            map_point_id (int): Map point ID.
            
        Returns:
            dict: Map point or None if not found.
        """
        for mp in self.map_points:
            if mp['id'] == map_point_id:
                return mp
        return None
    
    def find_connected_keyframes(self, keyframe_id, min_connections=15):
        """
        Find keyframes connected to the given keyframe in the co-visibility graph.
        
        Args:
            keyframe_id (int): Keyframe ID.
            min_connections (int): Minimum number of common observations.
            
        Returns:
            list: List of connected keyframe IDs.
        """
        if keyframe_id not in self.co_visibility_graph:
            return []
        
        # Get connections from co-visibility graph
        connections = self.co_visibility_graph[keyframe_id]
        
        # Filter by minimum connections
        connected_kfs = [kf_id for kf_id, conn in connections.items() if conn >= min_connections]
        
        return connected_kfs