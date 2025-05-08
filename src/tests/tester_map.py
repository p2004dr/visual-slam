#!/usr/bin/env python
"""
End-to-end map generation tester for Visual SLAM.
Processes the entire video through the pipeline:
 1) Load configuration and camera parameters
 2) Initialize Tracker and LocalMapper
 3) Iterate over video frames (with optional skipping)
 4) Process each frame for tracking and mapping
 5) After processing, visualize the 2D top-down map and save a PLY point cloud
"""
import os
import sys
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Paths ---
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_DIR = os.path.join(BASEDIR, 'src')
CONFIG_PATH = os.path.join(BASEDIR, 'configs', 'monocular.yaml')
VIDEO_PATH = os.path.join(BASEDIR, 'data', 'hospital_video.mp4')
OUTPUT_PLY = os.path.join(BASEDIR, 'data', 'map.ply')

# --- Add src to PYTHONPATH ---
sys.path.append(SRC_DIR)
from orbslam2.utils import load_config, load_camera_intrinsics
from orbslam2.local_mapper import LocalMapper
from orbslam2.tracker import Tracker


def main():
    # Load configuration
    config = load_config(CONFIG_PATH)
    K, D = load_camera_intrinsics(config)

    # Initialize mapping and tracking modules
    local_mapper = LocalMapper(camera_matrix=K, output_path=OUTPUT_PLY)
    tracker = Tracker(camera_matrix=K,
                      distortion_coeffs=D,
                      n_features=config['orb']['n_features'],
                      local_mapper=local_mapper)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open video at {VIDEO_PATH}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = config.get('skip_frames', 0)
    max_frames = config.get('max_frames', 0) or total
    frame_idx = 0

    print(f"Processing up to {max_frames} frames (skip={skip}) from video with {total} total frames...")

    while frame_idx < total and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if skip and (frame_idx % (skip + 1) != 0):
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tf, kps, vis = tracker.process_frame(gray)

        # Optional: display tracking overlay (uncomment to see live)
        # if vis is not None:
        #     cv2.imshow('Tracking', vis)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        frame_idx += 1

    cap.release()
    # cv2.destroyAllWindows()

    # After processing, visualize and save map
    print("--- Map Statistics ---")
    stats = local_mapper.get_map_statistics()
    for k, v in stats.items():
        print(f"{k}: {v}")

    # 2D top-down visualization
    map_img = local_mapper.visualize_map(width=800, height=600)
    plt.figure(figsize=(8,6))
    plt.imshow(map_img[..., ::-1])  # BGR to RGB
    plt.title('Top-Down Map View')
    plt.axis('off')
    plt.show()

    # Debug before saving PLY
    print("--- Debug PLY Save ---")
    # Extract filtered points/colors
    points, colors = local_mapper._filter_map_points()
    print(f"Filtered points shape: {getattr(points, 'shape', None)}")
    print(f"Colors shape: {getattr(colors, 'shape', None)}")
    # Ensure colors array is Nx3
    if colors.ndim == 2 and colors.shape[1] == 4:
        print("Detected RGBA colors, converting to RGB by dropping alpha")
        colors = colors[:, :3]
    try:
        local_mapper._save_map()
    except Exception as e:
        print(f"Error saving PLY: {e}")
    else:
        print(f"Point cloud saved to: {OUTPUT_PLY}")


if __name__ == '__main__':
    main()
