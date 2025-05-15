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
    
    def __init__(self, camera_matrix, min_matches=10, min_inliers_ratio=0.9):
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
        self.current_frame_keypoints = None 
        
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
        
        # Calculate essential matrix with higher threshold
        E, mask = calculate_essential_matrix(points1, points2, self.camera_matrix, threshold=3.0)
        
        # Recover pose (R,t) from essential matrix
        _, R, t, mask_pose, *_ = recover_pose(E, points1, points2, self.camera_matrix, mask)
        t = t.reshape(3, 1) if t.ndim == 1 else t
        
        # Create projection matrices
        P1 = compute_projection_matrix(np.eye(3), np.zeros((3, 1)), self.camera_matrix)
        P2 = compute_projection_matrix(R, t, self.camera_matrix)
        
        # Filter matches using pose recovery mask
        valid_matches = [m for m, valid in zip(matches, mask_pose.ravel().astype(bool)) if valid]
        points1 = np.float32([self.first_frame_keypoints[m.queryIdx].pt for m in valid_matches])
        points2 = np.float32([current_keypoints[m.trainIdx].pt for m in valid_matches])
        # Triangulate points
        points_4d = triangulate_points(points1, points2, P1, P2)
        points_3d = convert_to_3d_points(points_4d)

        
        print("\n=== Debug Triangulación ===")
        print("Shape points_4d:", points_4d.shape)  # Debería ser (4, N)
        print("Ejemplo punto 4D:", points_4d[:,0])  # Debería tener 4 componentes
        print("Shape points_3d:", points_3d.shape)  # Debería ser (3, N) o (N,3)
        print("Ejemplo punto 3D:", points_3d[0])    # Debería tener 3 componentes

        # Filter points behind cameras
        valid_indices = []
        for i, pt in enumerate(points_3d):
            # Depth in first camera (P1 is identity)
            depth1 = pt[2]
            # Depth in second camera (transformed by R and t)
            #print("\n=== Debug Geometría ===")
            #print("Shape R:", R.shape)  # Debe ser (3,3)
            #print("Shape t:", t.shape)  # Debe ser (3,1) o (3,)
            #print("Primer punto pt:", points_3d[0].shape)  # Debe ser (3,)
            pt_cam2 = R @ pt + t.ravel()
            depth2 = pt_cam2[2]
            if depth1 > 0 and depth2 > 0:
                valid_indices.append(i)
        
        points_3d = points_3d[valid_indices]
        valid_matches = [valid_matches[i] for i in valid_indices]
        
        # Check if we have sufficient valid points
        if len(points_3d) < self.min_matches//2:
            print(f"Insufficient valid 3D points after filtering: {len(points_3d)}")
            return False, None, None, None, valid_matches
        
        # Extract colors for visualization
        colors = []
        for match in valid_matches:
            kp = self.first_frame_keypoints[match.queryIdx]
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < self.first_frame_image.shape[1] and 0 <= y < self.first_frame_image.shape[0]:
                if self.first_frame_image.ndim == 3:  # Color image
                    color = self.first_frame_image[y, x, :]
                else:  # Grayscale
                    color = np.array([self.first_frame_image[y, x]]*3)
            else:
                color = np.array([0, 0, 255])
            colors.append(color)
        
        # Create map points
        initial_map_points = [{
            'position': pt,
            'color': colors[i],
            'keypoint_references': {0: m.queryIdx, 1: m.trainIdx},
            'observed_frames': [0, 1]
        } for i, (pt, m) in enumerate(zip(points_3d, valid_matches))]
        
        self.initialization_done = True
        
        self.current_frame_keypoints = current_keypoints
        return True, R, t, initial_map_points, valid_matches
        
    def draw_initialization(self, first_image, current_image, matches):
        """
        Dibuja los matches entre el primer frame y el frame actual.
        
        Args:
            first_image (np.array): Imagen del primer frame.
            current_image (np.array): Imagen del frame actual.
            matches (list): Lista de objetos DMatch.
            
        Returns:
            np.array: Imagen con los matches dibujados.
        """
        # Convertir a color si es necesario
        first_img = cv2.cvtColor(first_image, cv2.COLOR_GRAY2BGR) if len(first_image.shape) == 2 else first_image.copy()
        current_img = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR) if len(current_image.shape) == 2 else current_image.copy()

        # Usar TODOS los keypoints del frame actual, no solo los filtrados
        return cv2.drawMatches(
            first_img, 
            self.first_frame_keypoints,
            current_img, 
            self.current_frame_keypoints,  # Keypoints completos del segundo frame
            matches, 
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0),  # Color verde para los matches
            singlePointColor=(255, 0, 0)  # Color azul para keypoints no emparejados
        )