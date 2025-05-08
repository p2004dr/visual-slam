"""
Feature matching module for ORB-SLAM2 Python implementation.

This module handles matching features between frames using ORB descriptors.
"""

import numpy as np
import cv2


class DescriptorMatcher:
    """
    Class for matching ORB descriptors between frames.
    """
    
    def __init__(self, matcher_type='bruteforce-hamming', ratio_threshold=0.75):
        """
        Initialize the descriptor matcher.
        
        Args:
            matcher_type (str): Type of matcher ('bruteforce-hamming' or 'flann').
            ratio_threshold (float): Lowe's ratio test threshold.
        """
        self.ratio_threshold = ratio_threshold
        
        if matcher_type == 'bruteforce-hamming':
            # ORB descriptors are binary, so we use Hamming distance
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif matcher_type == 'flann':
            # For FLANN with binary descriptors
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matcher type: {matcher_type}")
    
    def match(self, descriptors1, descriptors2, ratio_test=True):
        """
        Match descriptors between two sets.
        
        Args:
            descriptors1 (np.array): First set of descriptors.
            descriptors2 (np.array): Second set of descriptors.
            ratio_test (bool): Whether to apply Lowe's ratio test.
            
        Returns:
            list: List of good matches.
        """
        # Check if we have valid descriptors
        if descriptors1 is None or descriptors2 is None:
            return []
        
        if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
            return []
        
        # Ensure descriptors are in the right format (uint8)
        if descriptors1.dtype != np.uint8:
            descriptors1 = np.uint8(descriptors1)
        if descriptors2.dtype != np.uint8:
            descriptors2 = np.uint8(descriptors2)
        
        # Match descriptors
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        # Filter matches using Lowe's ratio test
        good_matches = []
        for match in matches:
            if len(match) >= 2:
                m, n = match[:2]
                if not ratio_test or m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
            elif len(match) == 1:
                # If only one match found, add it (less reliable)
                good_matches.append(match[0])
        
        return good_matches
    
    def match_with_mask(self, descriptors1, descriptors2, mask):
        """
        Match descriptors using a mask to restrict the search area.
        
        Args:
            descriptors1 (np.array): First set of descriptors.
            descriptors2 (np.array): Second set of descriptors.
            mask (np.array): Binary mask where 1 indicates valid match region.
            
        Returns:
            list: List of good matches.
        """
        # Check if we have valid descriptors
        if descriptors1 is None or descriptors2 is None:
            return []
        
        # Match descriptors with mask
        matches = self.matcher.match(descriptors1, descriptors2, mask=mask)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches
    
    def filter_matches_by_distance(self, matches, distance_threshold=None):
        """
        Filter matches by distance.
        
        Args:
            matches (list): List of matches.
            distance_threshold (float, optional): Distance threshold. If None,
                a threshold based on median distance is used.
            
        Returns:
            list: Filtered matches.
        """
        if not matches:
            return []
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        if distance_threshold is None:
            # Calculate threshold based on median distance
            distances = [m.distance for m in matches]
            median_distance = np.median(distances)
            distance_threshold = median_distance * 2.0
        
        # Filter matches
        return [m for m in matches if m.distance < distance_threshold]
    
    def filter_matches_by_fundamental(self, keypoints1, keypoints2, matches, threshold=3.0):
        """
        Filter matches using the fundamental matrix and RANSAC.
        
        Args:
            keypoints1 (list): First set of keypoints.
            keypoints2 (list): Second set of keypoints.
            matches (list): List of matches.
            threshold (float): RANSAC threshold.
            
        Returns:
            tuple: (filtered_matches, inlier_mask)
        """
        if len(matches) < 8:
            return matches, np.ones(len(matches), dtype=bool)
        
        # Extract matched points
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        
        # Find fundamental matrix
        F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, threshold, 0.99)
        
        # Convert mask to boolean array
        mask = mask.ravel().astype(bool)
        
        # Filter matches by mask
        filtered_matches = [m for m, inlier in zip(matches, mask) if inlier]
        
        return filtered_matches, mask
    
    def draw_matches(self, img1, keypoints1, img2, keypoints2, matches, flags=0):
        """
        Draw matches between two images.
        
        Args:
            img1 (np.array): First image.
            keypoints1 (list): First set of keypoints.
            img2 (np.array): Second image.
            keypoints2 (list): Second set of keypoints.
            matches (list): List of matches.
            flags (int): Drawing flags.
            
        Returns:
            np.array: Image with matches drawn.
        """
        return cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=flags)