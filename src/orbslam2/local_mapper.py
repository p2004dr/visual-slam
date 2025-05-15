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
        self.min_observations = 2  # Minimum observations for a valid map point
    
    def set_output_path(self, output_path):
        """
        Set the output path for the PLY map file.
        
        Args:
            output_path (str): Path to save PLY map file.
        """
        self.output_path = output_path
    
    def add_keyframe(self, image, keypoints, descriptors, pose):
        """
        Add a new keyframe, triangulate new points, and perform optional local bundle adjustment.

        Args:
            image (np.array): Keyframe image.
            keypoints (list of cv2.KeyPoint): Extracted keypoints.
            descriptors (np.ndarray): Corresponding descriptors.
            pose (np.ndarray): 4x4 camera pose (world to camera).
        """
        # Assign ID and store keyframe
        kf_id = len(self.keyframes)
        self.keyframes.append({
            'id': kf_id,
            'image': image.copy(),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose.copy(),
            'map_points': []  # IDs of observed map points
        })

        # Triangulate new map points against previous keyframe
        if len(self.keyframes) >= 2:
            self._triangulate_new_points()

        # Save map after adding new points
        if self.output_path:
            self._save_map()
    
    def _triangulate_new_points(self):
        """
        Triangulate new map points between the two most recent keyframes.
        """
        # Get the two most recent keyframes
        kf1, kf2 = self.keyframes[-2], self.keyframes[-1]
        
        # Compute projection matrices
        P1 = compute_projection_matrix(kf1['pose'][:3,:3], kf1['pose'][:3,3], self.camera_matrix)
        P2 = compute_projection_matrix(kf2['pose'][:3,:3], kf2['pose'][:3,3], self.camera_matrix)
        
        # Match features between keyframes
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(kf1['descriptors'], kf2['descriptors'])
        
        # Get matched points
        pts1 = np.float32([kf1['keypoints'][m.queryIdx].pt for m in matches])
        pts2 = np.float32([kf2['keypoints'][m.trainIdx].pt for m in matches])
        
        # Apply fundamental matrix to filter outliers
        if len(matches) >= 8:
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
            if mask is not None:
                mask = mask.ravel() > 0
                pts1 = pts1[mask]
                pts2 = pts2[mask]
                matches = [m for i, m in enumerate(matches) if mask[i]]
        
        # Triangulate points
        pts4d = triangulate_points(pts1, pts2, P1, P2)
        pts3d = convert_to_3d_points(pts4d)
        
        # Filter points by checking reprojection error
        good_points = []
        for i, (point_3d, match) in enumerate(zip(pts3d, matches)):
            # Project the 3D point back to 2D in both keyframes
            p1 = P1 @ np.append(point_3d, 1.0)
            p1 = p1[:2] / p1[2]
            
            p2 = P2 @ np.append(point_3d, 1.0)
            p2 = p2[:2] / p2[2]
            
            # Compute reprojection error
            error1 = np.linalg.norm(p1 - pts1[i])
            error2 = np.linalg.norm(p2 - pts2[i])
            
            # Accept points with low reprojection error
            if error1 < 5.0 and error2 < 5.0:
                good_points.append((point_3d, match))
        
        # Create map points
        for point_3d, match in good_points:
            # Create map point and add observations
            mp = {
                'id': len(self.map_points),
                'position': point_3d,
                'observations': {
                    kf1['id']: match.queryIdx,
                    kf2['id']: match.trainIdx
                }
            }
            
            # Add map point to the list
            self.map_points.append(mp)
            
            # Update keyframe references
            kf1['map_points'].append(mp['id'])
            kf2['map_points'].append(mp['id'])
            
            # Update co-visibility graph
            self.co_visibility_graph[kf1['id']][kf2['id']] += 1
            self.co_visibility_graph[kf2['id']][kf1['id']] += 1
    
    def _cull_map_points(self):
        """
        Remove low-quality map points.
        """
        valid_map_points = []
        for mp in self.map_points:
            # Get the observations
            obs = mp.get('observations', {})
            
            # Discard points with too few observations
            if len(obs) < self.min_observations:
                continue

            # Check reprojection error
            valid = True
            for kf_id, kp_id in obs.items():
                if kf_id >= len(self.keyframes):
                    valid = False
                    break
                    
                kf = self.keyframes[kf_id]
                if kp_id >= len(kf['keypoints']):
                    valid = False
                    break
                    
                kp = kf['keypoints'][kp_id]

                # Project point
                P = compute_projection_matrix(kf['pose'][:3, :3],
                                             kf['pose'][:3, 3],
                                             self.camera_matrix)
                point_h = np.append(mp['position'], 1.0)
                proj = P @ point_h
                proj = proj[:2] / proj[2]

                # Calculate error
                err = np.linalg.norm(proj - np.array(kp.pt))
                if err > 5.0:
                    valid = False
                    break

            if valid:
                valid_map_points.append(mp)

        # Update map points
        self.map_points = valid_map_points

        # Update map point references in keyframes
        for kf in self.keyframes:
            kf['map_points'] = [
                mp['id'] for mp in self.map_points 
                if kf['id'] in mp.get('observations', {})
            ]
    
    def _save_map(self):
        """
        Save the map as a PLY file.
        """
        if not self.map_points:
            print(f"No map points to save to {self.output_path}")
            return
            
        pts = np.array([mp['position'] for mp in self.map_points])
        cols = np.full_like(pts, 255)  # Default white color
        
        create_point_cloud_ply(pts, cols, self.output_path)
        print(f"Saved map with {len(pts)} points to {self.output_path}")
    
    def visualize_map(self, width=800, height=600):
        """
        Create a simple visualization of the map and camera trajectory.
        
        Args:
            width (int): Image width.
            height (int): Image height.
            
        Returns:
            np.array: Visualization image.
        """
        # Create blank image
        img = np.zeros((height, width, 3), np.uint8)
        
        if not self.map_points:
            return img
            
        # Get map points
        pts = np.array([mp['position'] for mp in self.map_points])
        
        # Get min/max values for scaling
        min_vals = pts.min(0)
        max_vals = pts.max(0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        if np.any(range_vals == 0):
            return img
            
        # Calculate scaling factor
        x_scale = (width - 40) / range_vals[0]
        z_scale = (height - 40) / range_vals[2]
        scale = min(x_scale, z_scale)
        
        # Draw points (top-down view: x-z plane)
        for p in pts:
            x = int((p[0] - min_vals[0]) * scale + 20)
            y = int((p[2] - min_vals[2]) * scale + 20)
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1)
        
        # Draw camera trajectory
        if len(self.keyframes) > 1:
            trajectory = self.get_camera_trajectory()
            
            for i in range(1, len(trajectory)):
                x1 = int((trajectory[i-1][0] - min_vals[0]) * scale + 20)
                y1 = int((trajectory[i-1][2] - min_vals[2]) * scale + 20)
                x2 = int((trajectory[i][0] - min_vals[0]) * scale + 20)
                y2 = int((trajectory[i][2] - min_vals[2]) * scale + 20)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
            # Draw current camera position
            x = int((trajectory[-1][0] - min_vals[0]) * scale + 20)
            y = int((trajectory[-1][2] - min_vals[2]) * scale + 20)
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        
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