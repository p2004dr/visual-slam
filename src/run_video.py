#!/usr/bin/env python
"""
Main script to run monocular visual SLAM on a video file.
"""

import os
import sys
import argparse
import yaml
import cv2
import numpy as np
import time

from orbslam2.extractor import ORBExtractor
from orbslam2.matcher import DescriptorMatcher
from orbslam2.initializer import Initializer
from orbslam2.tracker import Tracker
from orbslam2.local_mapper import LocalMapper
from orbslam2.utils import load_config, load_camera_intrinsics, undistort_image


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run monocular visual SLAM on a video file.')
    parser.add_argument('--config', type=str, default='configs/monocular.yaml',
                        help='Path to configuration file')
    parser.add_argument('--video', type=str, help='Path to video file (overrides config)')
    parser.add_argument('--output', type=str, help='Path to output map file (overrides config)')
    parser.add_argument('--display', action='store_true', help='Display tracking visualization')
    parser.add_argument('--skip', type=int, default=0, help='Skip N frames at the beginning')
    parser.add_argument('--max-frames', type=int, default=0, help='Maximum number of frames to process (0 for all)')
    
    return parser.parse_args()


def run_slam(config_path, video_path=None, output_path=None, display=False, skip_frames=0, max_frames=0):
    """
    Run monocular visual SLAM on a video file.
    
    Args:
        config_path (str): Path to configuration file.
        video_path (str): Path to video file (overrides config).
        output_path (str): Path to output map file (overrides config).
        display (bool): Whether to display tracking visualization.
        skip_frames (int): Number of frames to skip at the beginning.
        max_frames (int): Maximum number of frames to process (0 for all).
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override video path if provided
    if video_path:
        config['video_path'] = video_path
    
    # Override output path if provided
    if output_path:
        config['output_path'] = output_path
    
    # Load camera intrinsics
    camera_matrix, distortion_coeffs = load_camera_intrinsics(config)
    
    # Create components
    orb_extractor = ORBExtractor(
        n_features=config.get('n_features', 2000),
        scale_factor=config.get('scale_factor', 1.2),
        n_levels=config.get('n_levels', 8)
    )
    
    matcher = DescriptorMatcher(
        matcher_type=config.get('matcher_type', 'bruteforce-hamming'),
        ratio_threshold=config.get('ratio_threshold', 0.75)
    )
    
    initializer = Initializer(
        matcher=matcher,
        camera_matrix=camera_matrix,
        dist_coeffs=distortion_coeffs
    )
    
    local_mapper = LocalMapper(
        camera_matrix=camera_matrix,
        output_path=config.get('output_path', 'map.ply')
    )
    
    # Create tracker
    tracker = Tracker(
        orb_extractor=orb_extractor,
        matcher=matcher,
        initializer=initializer,
        local_mapper=local_mapper,
        camera_matrix=camera_matrix,
        dist_coeffs=distortion_coeffs
    )
    
    # Open video
    video_path = config['video_path']
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Skip frames if requested
    if skip_frames > 0:
        print(f"Skipping {skip_frames} frames...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    
    # Set maximum frames
    if max_frames <= 0:
        max_frames = total_frames
    else:
        max_frames = min(max_frames, total_frames - skip_frames)
    
    print(f"Processing {max_frames} frames...")
    
    # Initialize variables
    frame_count = 0
    tracking_lost = False
    start_time = time.time()
    
    # Define output windows if display is enabled
    if display:
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Map', cv2.WINDOW_NORMAL)
    
    # Main loop
    while frame_count < max_frames:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("End of video reached")
            break
        
        # Process frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Undistort image if necessary
        if np.any(distortion_coeffs):
            frame_gray = undistort_image(frame_gray, camera_matrix, distortion_coeffs)
        
        # Track frame
        tracking_result = tracker.track_frame(frame_gray)
        
        # Check if tracking is lost
        if tracking_result['status'] == 'LOST':
            print(f"Frame {frame_count + skip_frames}: Tracking lost!")
            tracking_lost = True
        else:
            tracking_lost = False
        
        # Display results if requested
        if display:
            # Prepare tracking visualization
            tracking_vis = tracking_result['visualization']
            
            # Add frame number and status
            cv2.putText(tracking_vis, f"Frame: {frame_count + skip_frames}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(tracking_vis, f"Status: {tracking_result['status']}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show tracking visualization
            cv2.imshow('Tracking', tracking_vis)
            
            # Show map visualization
            map_vis = local_mapper.visualize_map()
            cv2.imshow('Map', map_vis)
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                print("User terminated")
                break
        
        # Print progress
        if (frame_count + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            fps_est = (frame_count + 1) / elapsed_time if elapsed_time > 0 else 0
            print(f"Processed {frame_count + 1}/{max_frames} frames ({fps_est:.1f} fps)")
            
            # Print map statistics
            stats = local_mapper.get_map_statistics()
            print(f"Map: {stats['num_keyframes']} keyframes, {stats['num_map_points']} map points")
        
        frame_count += 1
    
    # Clean up
    cap.release()
    
    if display:
        cv2.destroyAllWindows()
    
    # Save final map
    print("Saving map...")
    local_mapper._save_map()
    
    # Print final statistics
    elapsed_time = time.time() - start_time
    fps_est = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"Processed {frame_count} frames in {elapsed_time:.1f} seconds ({fps_est:.1f} fps)")
    
    stats = local_mapper.get_map_statistics()
    print(f"Final map: {stats['num_keyframes']} keyframes, {stats['num_map_points']} map points")
    print(f"Map saved to {config['output_path']}")


if __name__ == "__main__":
    args = parse_arguments()
    run_slam(
        config_path=args.config,
        video_path=args.video,
        output_path=args.output,
        display=args.display,
        skip_frames=args.skip,
        max_frames=args.max_frames
    )