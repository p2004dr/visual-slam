"""
Feature extraction module for ORB-SLAM2 Python implementation.

This module handles the extraction of ORB features from images,
which are used for tracking and mapping.
"""

import numpy as np
import cv2


class ORBExtractor:
    """
    ORB feature extractor class.
    
    Extracts ORB features and descriptors from images for tracking and mapping.
    """
    
    def __init__(self, n_features=2000, scale_factor=1.2, n_levels=8, 
                 ini_threshold=20, min_threshold=7):
        """
        Initialize the ORB extractor.
        
        Args:
            n_features (int): Number of features to extract.
            scale_factor (float): Scale factor between pyramid levels.
            n_levels (int): Number of pyramid levels.
            ini_threshold (int): Initial FAST threshold.
            min_threshold (int): Minimum FAST threshold.
        """
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.ini_threshold = ini_threshold
        self.min_threshold = min_threshold
        
        # Create ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=min_threshold
        )
        
    def detect_and_compute(self, image):
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image (np.array): Input grayscale image.
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        # Make sure image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        return keypoints, descriptors
    
    def distribute_keypoints(self, image, n_features=None):
        """
        Distribute keypoints evenly across the image using grid-based detection.
        
        Args:
            image (np.array): Input grayscale image.
            n_features (int, optional): Override number of features.
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        if n_features is None:
            n_features = self.n_features
            
        # Make sure image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Define grid parameters
        height, width = image.shape
        grid_rows = 8
        grid_cols = 8
        cell_height = height // grid_rows
        cell_width = width // grid_cols
        
        features_per_cell = n_features // (grid_rows * grid_cols)
        
        all_keypoints = []
        
        # Create a mask for each cell
        for i in range(grid_rows):
            for j in range(grid_cols):
                # Create mask for current cell
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[i*cell_height:(i+1)*cell_height, 
                     j*cell_width:(j+1)*cell_width] = 255
                
                # Detect keypoints in the cell
                cell_keypoints = cv2.goodFeaturesToTrack(
                    image, 
                    maxCorners=features_per_cell,
                    qualityLevel=0.01,
                    minDistance=10,
                    mask=mask
                )
                
                if cell_keypoints is not None:
                    # Convertir CADA punto a cv2.KeyPoint ✅
                    for pt in cell_keypoints:
                        x, y = pt.ravel()
                        kp = cv2.KeyPoint(float(x), float(y), 31)
                        all_keypoints.append(kp)
        
        # Compute descriptors for all keypoints
        if all_keypoints:
            _, descriptors = self.orb.compute(image, all_keypoints)
        else:
            descriptors = None
            
        return all_keypoints, descriptors

    def extract_features(self, image, distributed=True):
        """
        Extract features from an image.
        
        Args:
            image (np.array): Input image.
            distributed (bool): Whether to use grid-based feature distribution.
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        if distributed:
            return self.distribute_keypoints(image)
        else:
            return self.detect_and_compute(image)
            
    def draw_keypoints(self, image, keypoints):
        """
        Draw keypoints on an image.
        
        Args:
            image (np.array): Input image.
            keypoints (list): List of cv2.KeyPoint objects.
            
        Returns:
            np.array: Image with keypoints drawn.
        """
        return cv2.drawKeypoints(image, keypoints, None, 
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)