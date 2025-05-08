"""
Tracking module for ORB-SLAM2 Python implementation.

This module handles frame-to-frame tracking and pose estimation.
"""

import numpy as np
import cv2
from .extractor import ORBExtractor
from .matcher import DescriptorMatcher
from .initializer import MapInitializer
from .utils import compute_projection_matrix


class Tracker:
    """
    Class for tracking camera motion and features across frames.
    """
    
    def __init__(self, camera_matrix, distortion_coeffs=None, 
                 n_features=2000, local_mapper=None):
        """
        Initialize the tracker.
        
        Args:
            camera_matrix (np.array): 3x3 camera intrinsic matrix.
            distortion_coeffs (np.array): Camera distortion coefficients.
            n_features (int): Number of features to extract per frame.
            local_mapper (LocalMapper): Local mapping module instance.
        """
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        
        # Create feature extractor
        self.extractor = ORBExtractor(n_features=n_features)
        
        # Create feature matcher
        self.matcher = DescriptorMatcher(matcher_type='bruteforce-hamming')
        
        # Map initializer
        self.initializer = MapInitializer(camera_matrix)
        
        # Reference to local mapper
        self.local_mapper = local_mapper
        
        # Tracking state
        self.state = "NOT_INITIALIZED"
        self.current_frame_idx = 0
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.reference_pose = None  # [R|t]
        self.poses = []  # Store all camera poses (for trajectory visualization)
        
        # Map points
        self.map_points = []
        
        # Last frame info
        self.last_frame = None
        self.last_keypoints = None
        self.last_descriptors = None
        
    def set_local_mapper(self, local_mapper):
        """
        Set the local mapper reference.
        
        Args:
            local_mapper (LocalMapper): Local mapping module instance.
        """
        self.local_mapper = local_mapper
        
    def process_frame(self, frame):
        """
        Process a new frame for tracking.
        
        Args:
            frame (np.array): Input frame.
            
        Returns:
            tuple: (transformation_matrix, keypoints, visualization_image)
        """
        # Make a copy of the frame
        current_frame = frame.copy()
        
        # Extract keypoints and descriptors
        keypoints, descriptors = self.extractor.extract_features(current_frame)
        
        # Create visualization image
        vis_image = current_frame.copy()
        if len(vis_image.shape) == 2:  # Convert to color for visualization
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw keypoints
        vis_image = self.extractor.draw_keypoints(vis_image, keypoints)
        
        # Handle different tracking states
        if self.state == "NOT_INITIALIZED":
            success = self._initialize(current_frame, keypoints, descriptors)
            if success:
                self.state = "TRACKING"
                # Add initial pose (identity)
                self.poses.append(np.eye(4))
                
                # Draw initialization result
                vis_image = self.initializer.draw_initialization(self.last_frame, current_frame, [])
        
        elif self.state == "TRACKING":
            # Track from last frame
            success, T, vis_image_matches = self._track_from_last_frame(current_frame, keypoints, descriptors)
            
            if success:
                # Update pose and map
                self._update_pose(T)
                
                # Check if we need to create a new keyframe
                if self._need_new_keyframe():
                    self._create_keyframe(current_frame, keypoints, descriptors)
                    
                # Update visualization
                if vis_image_matches is not None:
                    vis_image = vis_image_matches
            else:
                # Tracking failure
                self.state = "LOST"
                print("Tracking lost!")
        
        elif self.state == "LOST":
            # Try to relocalize
            success = self._relocalize(current_frame, keypoints, descriptors)
            
            if success:
                self.state = "TRACKING"
            else:
                # Draw failure text
                cv2.putText(vis_image, "TRACKING LOST", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Store current frame info for next iteration
        self.last_frame = current_frame
        self.last_keypoints = keypoints
        self.last_descriptors = descriptors
        self.current_frame_idx += 1
        
        # Return current transformation matrix (camera pose)
        return self.reference_pose, keypoints, vis_image
    
    def _initialize(self, frame, keypoints, descriptors):
        """
        Initialize the map with the first two frames.
        
        Args:
            frame (np.array): Current frame.
            keypoints (list): List of keypoints.
            descriptors (np.array): Descriptors.
            
        Returns:
            bool: Success flag.
        """
        if self.last_frame is None:
            # Set the first frame
            self.initializer.set_first_frame(keypoints, descriptors, frame)
            self.reference_keypoints = keypoints
            self.reference_descriptors = descriptors
            return False
        else:
            # Initialize map with the second frame
            success, R, t, initial_map_points, matches = self.initializer.initialize(
                keypoints, descriptors, self.matcher, frame
            )
            
            if success:
                # Set reference pose as identity
                self.reference_pose = np.eye(4)
                self.reference_pose[:3, :3] = np.eye(3)
                self.reference_pose[:3, 3] = np.zeros(3)
                
                # Set up map points
                self.map_points = initial_map_points
                
                # Pass map points to local mapper if available
                if self.local_mapper is not None:
                    self.local_mapper.add_keyframe(
                        self.last_frame, self.reference_keypoints, 
                        self.reference_descriptors, self.reference_pose
                    )
                    self.local_mapper.add_keyframe(
                        frame, keypoints, descriptors, 
                        self._compose_transform(R, t)
                    )
                    self.local_mapper.update_map_points(self.map_points)
                
                print(f"Initialization successful with {len(initial_map_points)} map points")
                return True
            
            return False

    def _track_from_last_frame(self, frame, keypoints, descriptors):
        """
        Track camera motion from the last frame using PnP 2D–3D against the global map.

        Args:
            frame (np.array): Current grayscale frame.
            keypoints (list): Keypoints detected in `frame`.
            descriptors (np.array): Descriptors for those keypoints.

        Returns:
            tuple: (success: bool, transformation_matrix: np.array 4x4, visualization_image: np.array or None)
        """
        # 0) Chequeos básicos
        if not self.local_mapper.map_points or descriptors is None:
            return False, None, None

        # 1) Filtrar solo map points con descriptor
        valid_mps = [mp for mp in self.local_mapper.map_points if 'descriptor' in mp]
        if not valid_mps:
            return False, None, None

        # 2) Sacar arrays de posiciones 3D y descriptores de mapa
        pts3d = np.array([mp['position'] for mp in valid_mps], dtype=np.float32)
        des3d = np.array([mp['descriptor'] for mp in valid_mps], dtype=np.uint8)

        # 3) Emparejar 3D->2D (map-points contra frame actual)
        matches = self.matcher.match(des3d, descriptors, ratio_test=True)
        min_tracked = getattr(self, 'min_tracked_points', 15)
        if len(matches) < min_tracked:
            return False, None, None

        # 4) Construir correspondencias
        object_points = np.float32([pts3d[m.queryIdx] for m in matches]).reshape(-1, 3)
        image_points  = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 5) PnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points,
            self.camera_matrix, self.distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=8.0,
            confidence=0.99
        )
        if not success or inliers is None or len(inliers) < min_tracked:
            return False, None, None

        # 6) Convertir rvec/tvec a matriz 4x4
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()

        # 7) Visualizar solo inliers
        inlier_matches = [matches[i[0]] for i in inliers]
        vis_image = cv2.drawMatches(
            getattr(self, 'last_frame', frame),
            getattr(self, 'last_keypoints', []),
            frame, keypoints,
            inlier_matches[:50],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # 8) Actualizar estado interno
        self._update_pose(T)
        self.last_frame = frame.copy()
        self.last_keypoints = keypoints
        self.last_descriptors = descriptors

        return True, T, vis_image


    def _update_pose(self, transformation_matrix):
        """
        Update the camera pose with a new transformation.
        
        Args:
            transformation_matrix (np.array): 4x4 transformation matrix.
        """
        # Update reference pose (T_w_c = T_w_r * T_r_c)
        self.reference_pose = self.reference_pose @ transformation_matrix
        
        # Store pose for trajectory visualization
        self.poses.append(self.reference_pose.copy())
    
    def _need_new_keyframe(self):
        """
        Determine if a new keyframe should be created.
        
        Returns:
            bool: True if a new keyframe is needed.
        """
        # Simple heuristic: create a keyframe every 20 frames
        # In a real system, this would be based on parallax, tracking quality, etc.
        return self.current_frame_idx % 20 == 0
    
    def _create_keyframe(self, frame, keypoints, descriptors):
        """
        Create a new keyframe and pass it to the local mapper.
        
        Args:
            frame (np.array): Frame image.
            keypoints (list): List of keypoints.
            descriptors (np.array): Descriptors.
        """
        if self.local_mapper is not None:
            self.local_mapper.add_keyframe(
                frame, keypoints, descriptors, self.reference_pose
            )
            print(f"Created new keyframe at frame {self.current_frame_idx}")
    
    def _relocalize(self, frame, keypoints, descriptors):
        """
        Try to relocalize the camera when tracking is lost.
        
        Args:
            frame (np.array): Current frame.
            keypoints (list): List of keypoints.
            descriptors (np.array): Descriptors.
            
        Returns:
            bool: Success flag.
        """
        # In a full SLAM system, this would match against keyframes
        # For this simple implementation, just try to restart tracking
        self.state = "NOT_INITIALIZED"
        self.last_frame = None
        self.last_keypoints = None
        self.last_descriptors = None
        return False
    
    def _compose_transform(self, R, t):
        """
        Compose a 4x4 transformation matrix from rotation and translation.
        
        Args:
            R (np.array): 3x3 rotation matrix.
            t (np.array): 3x1 translation vector.
            
        Returns:
            np.array: 4x4 transformation matrix.
        """
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.reshape(3)
        return T
    
    def get_trajectory(self):
        """
        Get the camera trajectory.
        
        Returns:
            np.array: Array of camera positions.
        """
        # Extract camera centers from poses
        centers = []
        for pose in self.poses:
            # Camera center C = -R^T * t
            R = pose[:3, :3]
            t = pose[:3, 3]
            center = -R.T @ t
            centers.append(center)
        
        return np.array(centers)