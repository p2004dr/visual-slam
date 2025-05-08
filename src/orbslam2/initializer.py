"""
Map initialization module for ORB-SLAM2 Python implementation.

This module handles the initialization of the map from the first two frames.
"""

import numpy as np
import cv2
from .utils import triangulate_points, convert_to_3d_points, calculate_essential_matrix, recover_pose, compute_projection_matrix


class MapInitializer:
    """
    Class for initializing the map from the first two frames.
    """
    
    def __init__(self, camera_matrix, min_matches=100, min_inliers_ratio=0.9):
        """
        Initialize the map initializer.
        
        Args:
            camera_matrix (np.array): 3x3 camera intrinsic matrix.
            min_matches (int): Minimum number of matches required for initialization.
            min_inliers_ratio (float): Minimum ratio of inliers for initialization.
        """
        self.camera_matrix = camera_matrix
        self.min_matches = min_matches
        self.min_inliers_ratio = min_inliers_ratio
        self.initialization_done = False
        self.first_frame_keypoints = None
        self.first_frame_descriptors = None
        self.first_frame_image = None
        
    def set_first_frame(self, keypoints, descriptors, image):
        """
        Set the first frame keypoints and descriptors.
        
        Args:
            keypoints (list): List of keypoints from the first frame.
            descriptors (np.array): Descriptors corresponding to keypoints.
            image (np.array): First frame image for later triangulation.
        """
        self.first_frame_keypoints = keypoints
        self.first_frame_descriptors = descriptors
        self.first_frame_image = image.copy()
        self.initialization_done = False
        
    def initialize(self, current_keypoints, current_descriptors, matcher, current_image):
        """
        Initialize the map using the first and current frame.
        
        Args:
            current_keypoints (list): List of keypoints from the current frame.
            current_descriptors (np.array): Descriptors corresponding to current keypoints.
            matcher (DescriptorMatcher): Descriptor matcher instance.
            current_image (np.array): Current frame image.
            
        Returns:
            tuple: (success, R, t, initial_map_points, matches)
        """
        # Check if we have the first frame
        if self.first_frame_keypoints is None or self.first_frame_descriptors is None:
            return False, None, None, None, None
        
        # Match features between first and current frame
        matches = matcher.match(self.first_frame_descriptors, current_descriptors)
        
        # Check if we have enough matches
        if len(matches) < self.min_matches:
            print(f"Not enough matches for initialization: {len(matches)} < {self.min_matches}")
            return False, None, None, None, matches
        
        # Get matched points
        points1 = np.float32([self.first_frame_keypoints[m.queryIdx].pt for m in matches])
        points2 = np.float32([current_keypoints[m.trainIdx].pt for m in matches])
        
        # Calculate essential matrix
        E, mask = calculate_essential_matrix(points1, points2, self.camera_matrix)
        
        # Check if we have enough inliers
        inliers_count = np.sum(mask)
        inliers_ratio = inliers_count / len(matches)
        
        if inliers_ratio < self.min_inliers_ratio:
            print(f"Not enough inliers for initialization: {inliers_ratio:.2f} < {self.min_inliers_ratio}")
            return False, None, None, None, matches
        
        # Recover pose (R,t) from essential matrix
        R, t, mask_pose = recover_pose(E, points1, points2, self.camera_matrix, mask)
        
        # Create projection matrices
        P1 = compute_projection_matrix(np.eye(3), np.zeros((3, 1)), self.camera_matrix)
        P2 = compute_projection_matrix(R, t, self.camera_matrix)
        
        # Filter points based on pose recovery mask
        valid_matches = [m for m, valid in zip(matches, mask_pose.ravel().astype(bool)) if valid]
        points1 = np.float32([self.first_frame_keypoints[m.queryIdx].pt for m in valid_matches])
        points2 = np.float32([current_keypoints[m.trainIdx].pt for m in valid_matches])
        
        # Triangulate points to get 3D map points
        points_4d = triangulate_points(points1, points2, P1, P2)
        points_3d = convert_to_3d_points(points_4d)
        
        # Extract colors from the first frame for visualization
        colors = []
        for point, match in zip(points_3d, valid_matches):
            x, y = int(self.first_frame_keypoints[match.queryIdx].pt[0]), int(self.first_frame_keypoints[match.queryIdx].pt[1])
            if 0 <= x < self.first_frame_image.shape[1] and 0 <= y < self.first_frame_image.shape[0]:
                if len(self.first_frame_image.shape) == 3:  # Color image
                    color = self.first_frame_image[y, x, :]
                else:  # Grayscale image
                    color = np.array([self.first_frame_image[y, x]] * 3)
                colors.append(color)
            else:
                colors.append(np.array([0, 0, 255]))  # Default blue
        
        # Create initial map points with keypoint references
        initial_map_points = []
        for i, (point_3d, match) in enumerate(zip(points_3d, valid_matches)):
            map_point = {
                'position': point_3d,
                'color': colors[i] if i < len(colors) else np.array([0, 0, 255]),
                'keypoint_references': {
                    0: match.queryIdx,  # First frame keypoint index
                    1: match.trainIdx    # Current frame keypoint index
                },
                'observed_frames': [0, 1]  # Frame indices where this point was observed
            }
            initial_map_points.append(map_point)
        
        self.initialization_done = True
        
        return True, R, t, initial_map_points, valid_matches
    
    def draw_initialization(self, first_image, current_image, matches):
        """
        Draw initialization results for visualization.
        
        Args:
            first_image (np.array): First frame image.
            current_image (np.array): Current frame image.
            matches (list): List of matches.
            
        Returns:
            np.array: Image with matches drawn.
        """
        # Ensure images are color for drawing
        if len(first_image.shape) == 2:
            first_image_color = cv2.cvtColor(first_image, cv2.COLOR_GRAY2BGR)
        else:
            first_image_color = first_image.copy()
            
        if len(current_image.shape) == 2:
            current_image_color = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
        else:
            current_image_color = current_image.copy()
        
        # Draw matches
        result = cv2.drawMatches(
            first_image_color, self.first_frame_keypoints,
            current_image_color, None,  # We'll fill in current keypoints below
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return result