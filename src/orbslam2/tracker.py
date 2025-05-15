"""
Tracking module for ORB-SLAM2 Python implementation with improved initialization.

This module handles frame-to-frame tracking and pose estimation with more robust initialization.
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
                 n_features=2000, local_mapper=None, ratio_threshold=0.75):
        """
        Initialize the tracker.
        
        Args:
            camera_matrix (np.array): 3x3 camera intrinsic matrix.
            distortion_coeffs (np.array): Camera distortion coefficients.
            n_features (int): Number of features to extract per frame.
            local_mapper (LocalMapper): Local mapping module instance.
            ratio_threshold (float): Ratio threshold for feature matching.
        """
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.extractor = ORBExtractor(n_features=n_features)
        self.matcher = DescriptorMatcher(
            matcher_type='bruteforce-hamming',
            ratio_threshold=ratio_threshold
        )
        self.initializer = MapInitializer(camera_matrix)
        self.local_mapper = local_mapper
        self.state = "NOT_INITIALIZED"
        self.current_frame_idx = 0
        self.reference_pose = np.eye(4)
        self.poses = [np.eye(4)]  # list of 4x4 poses

        # Last frame data
        self.last_frame = None
        self.last_keypoints = None
        self.last_descriptors = None
        
        # Initialization parameters
        self.init_min_matches = 60     # Minimum matches required for initialization
        self.init_min_inliers = 30      # Minimum inliers for initialization
        self.init_min_parallax = 0.5    # Minimum parallax angle in degrees
        self.init_attempts = 0          # Count of initialization attempts
        self.init_max_attempts = 5      # Maximum attempts before resetting reference frame
        
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
            frame (np.array): Input frame image.
            
        Returns:
            tuple: (pose, keypoints, visualization)
        """
        # Optional undistort
        if self.distortion_coeffs is not None:
            frame = cv2.undistort(frame, self.camera_matrix, self.distortion_coeffs)

        # Extract features
        kps, desc = self.extractor.extract_features(frame)
        vis = cv2.drawKeypoints(frame, kps, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if self.state == "NOT_INITIALIZED":
            success = self._initialize(frame, kps, desc)
            if success:
                self.state = "TRACKING"
                if hasattr(self.initializer, 'matches') and self.initializer.matches is not None:
                    vis = self._draw_initialization(
                        self.initializer.first_frame,
                        frame,
                        self.initializer.matches
                    )
                else:
                    print("Warning: Matches not available for visualization")
        else:
            success, T, matches_vis = self._track_with_pnp(frame, kps, desc)
            if success:
                self.reference_pose = T  # Replace with absolute pose instead of concatenating
                self.poses.append(self.reference_pose.copy())
                vis = matches_vis
                if self._need_new_keyframe():
                    self._create_keyframe(frame, kps, desc)
            else:
                self.state = "LOST"
                print("Tracking lost!")

        # Update last frame data
        self.last_frame = frame
        self.last_keypoints = kps
        self.last_descriptors = desc
        self.current_frame_idx += 1
        
        return self.reference_pose, kps, vis
   
    def _initialize(self, frame, kps, desc):
        """
        Initialize the map with first two frames.
        
        Args:
            frame (np.array): Current frame image.
            kps (list): List of keypoints.
            desc (np.array): Descriptors.
            
        Returns:
            bool: Success flag.
        """
        if self.last_frame is None:
            # Store first frame
            self.initializer.first_frame = frame.copy()
            self.initializer.first_kps = kps
            self.initializer.first_desc = desc
            self.reference_pose = np.eye(4)
            self.init_attempts = 0
            print("First frame stored for initialization")
            return False

        # Check if we have enough motion between frames
        if len(kps) < self.init_min_matches or len(self.initializer.first_kps) < self.init_min_matches:
            print(f"Not enough features for initialization: {len(kps)} vs {len(self.initializer.first_kps)}")
            # Reset reference frame if too few features
            if self.current_frame_idx % 10 == 0:
                self.initializer.first_frame = frame.copy()
                self.initializer.first_kps = kps
                self.initializer.first_desc = desc
                self.init_attempts = 0
                print("Resetting reference frame due to low feature count")
            return False

        # Try to initialize map from first and second frames
        success, R, t, matches = self.initializer.initialize(
            self.initializer.first_kps, self.initializer.first_desc,
            kps, desc, self.matcher
        )
        
        # Check if we have enough matches
        if not matches or len(matches) < self.init_min_matches:
            self.init_attempts += 1
            print(f"Initialization failed: {len(matches) if matches else 0} matches (attempt {self.init_attempts})")
            
            # Check if we've tried too many times with this reference frame
            if self.init_attempts >= self.init_max_attempts:
                print(f"Too many failed attempts ({self.init_attempts}), resetting reference frame")
                self.initializer.first_frame = frame.copy()
                self.initializer.first_kps = kps
                self.initializer.first_desc = desc
                self.init_attempts = 0
            return False
        
        # If we got matches but initialization still failed, analyze why
        if not success:
            self.init_attempts += 1
            print(f"Geometric model validation failed with {len(matches)} matches (attempt {self.init_attempts})")
            
            # Estimate average feature motion to check for camera movement
            if matches and len(matches) > 10:
                pts1 = np.float32([self.initializer.first_kps[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kps[m.trainIdx].pt for m in matches])
                motion = np.mean(np.linalg.norm(pts2 - pts1, axis=1))
                
                print(f"Average feature motion: {motion:.2f} pixels")
                
                # If motion is too small, wait for more movement
                if motion < 5.0:
                    print("Camera motion too small, waiting for more movement")
                    return False
            
            # Reset reference frame after several attempts
            if self.init_attempts >= self.init_max_attempts:
                print(f"Too many failed attempts ({self.init_attempts}), resetting reference frame")
                self.initializer.first_frame = frame.copy()
                self.initializer.first_kps = kps
                self.initializer.first_desc = desc
                self.init_attempts = 0
            return False

        # Store successful matches for visualization
        self.initializer.matches = matches
        
        # Create initial keyframes and map points
        if self.local_mapper:
            # First keyframe with identity pose
            self._add_keyframe(self.initializer.first_frame, 
                              self.initializer.first_kps, 
                              self.initializer.first_desc, 
                              np.eye(4))
            
            # Second keyframe with relative pose
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()
            self._add_keyframe(frame, kps, desc, T)
            
            # Triangulate initial points
            self.local_mapper._triangulate_new_points()
            
        print(f"Map initialized successfully with {len(matches)} matches")
        return True
  
    def _track_with_pnp(self, frame, kps, desc):
        """
        Track current frame using PnP with 3D-2D correspondences.
        
        Args:
            frame (np.array): Current frame image.
            kps (list): List of keypoints.
            desc (np.array): Descriptors.
            
        Returns:
            tuple: (success, transformation, visualization)
        """
        if not self.local_mapper or not self.local_mapper.map_points:
            return False, None, None
            
        # Get all map points
        map_points = self.local_mapper.map_points
        
        # Match current frame descriptors with keyframe descriptors to establish 2D-3D correspondences
        obj_pts = []  # 3D points
        img_pts = []  # 2D points in current frame
        
        # Loop through keyframes to find matches
        for kf in self.local_mapper.keyframes[-3:]:  # Use last 3 keyframes for efficiency
            matches = self.matcher.match(kf['descriptors'], desc)
            
            if not matches:
                continue
                
            for m in matches:
                # Get keypoint index in keyframe
                kp_idx_kf = m.queryIdx
                kp_idx_curr = m.trainIdx
                
                # Find map points that observed this keypoint
                for mp in map_points:
                    observations = mp.get('observations', {})
                    
                    # If this keypoint in this keyframe is associated with this map point
                    if kf['id'] in observations and observations[kf['id']] == kp_idx_kf:
                        obj_pts.append(mp['position'])
                        img_pts.append(kps[kp_idx_curr].pt)
                        break
        
        # Ensure we have enough correspondences
        if len(obj_pts) < 6:
            print(f"Not enough point correspondences for PnP: {len(obj_pts)}")
            return False, None, None

        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)
        
        # Solve PnP
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts,
                self.camera_matrix, self.distortion_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=8.0,  # Higher threshold for more robustness
                iterationsCount=100     # More iterations for better results
            )
        except Exception as e:
            print(f"PnP error: {e}")
            return False, None, None
        
        if not ok or inliers is None or len(inliers) < 6:
            print(f"PnP failed or insufficient inliers: {len(inliers) if inliers is not None else 0}")
            return False, None, None
            
        # Convert to transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = tvec.ravel()

        # Create visualization with inliers
        vis = frame.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            
        for idx in inliers.flatten():
            x, y = img_pts[idx]
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Add text showing inlier count
        cv2.putText(vis, f"Inliers: {len(inliers)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        return True, T, vis

    def _need_new_keyframe(self):
        """
        Determine if a new keyframe should be created.
        
        Returns:
            bool: True if a new keyframe is needed.
        """
        # Simple heuristic: create a keyframe every 15 frames (reduced from 20)
        # In a real system, this would be based on parallax, tracking quality, etc.
        return self.current_frame_idx % 15 == 0
    
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
            
    def _add_keyframe(self, image, kps, desc, pose):
        """
        Add a keyframe to the local mapper.
        
        Args:
            image (np.array): Keyframe image.
            kps (list): List of keypoints.
            desc (np.array): Descriptors.
            pose (np.array): 4x4 camera pose.
        """
        if self.local_mapper:
            self.local_mapper.add_keyframe(image, kps, desc, pose)
    
    def _draw_initialization(self, first_frame, current_frame, matches):
        """
        Draw initialization matches between first and current frame.
        
        Args:
            first_frame (np.array): First frame image.
            current_frame (np.array): Current frame image.
            matches (list): List of DMatch objects.
            
        Returns:
            np.array: Visualization image.
        """
        # Ensure images are in BGR
        if first_frame.ndim == 2:
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
        if current_frame.ndim == 2:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
            
        # Draw matches
        try:
            vis = cv2.drawMatches(
                first_frame, self.initializer.first_kps,
                current_frame, self.last_keypoints,
                matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            # Add text showing match count
            cv2.putText(vis, f"Matches: {len(matches)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            return vis
        except Exception as e:
            print(f"Error drawing matches: {e}")
            return np.hstack((first_frame, current_frame))
    
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