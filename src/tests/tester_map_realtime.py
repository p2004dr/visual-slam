#!/usr/bin/env python
"""
Real-time SLAM visualization tester.
Generates a side-by-side video: original frames on the left, evolving top-down map on the right.
Handles missing map-point metadata by falling back to a blank map view.
"""
import os
import sys
import yaml
import cv2
import numpy as np

# --- Paths ---
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_DIR = os.path.join(BASEDIR, 'src')
CONFIG_PATH = os.path.join(BASEDIR, 'configs', 'monocular.yaml')
VIDEO_PATH = os.path.join(BASEDIR, 'data', 'hospital_video.mp4')
OUTPUT_VIDEO = os.path.join(BASEDIR, 'data', 'realtime_slam_output.mp4')
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
    tracker = Tracker(
        camera_matrix=K,
        distortion_coeffs=D,
        n_features=config['orb']['n_features'],
        local_mapper=local_mapper
    )

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open video at {VIDEO_PATH}")
        return

    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, sample_frame = cap.read()
    if not ret:
        print("Error: cannot read first frame")
        return
    frame_h, frame_w = sample_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Prepare output writer
    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_w * 2, frame_h)
    )

    print(f"Processing {total_frames} frames, writing to {OUTPUT_VIDEO}...")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # SLAM processing
        tf, kps, vis = tracker.process_frame(gray)

        # Left pane: tracking visualization
        left_vis = vis if vis is not None else frame
        if left_vis.ndim == 2:
            left_vis = cv2.cvtColor(left_vis, cv2.COLOR_GRAY2BGR)

        # Right pane: map visualization (handle missing metadata)
        try:
            map_img = local_mapper.visualize_map(width=frame_w, height=frame_h)
            if map_img is None:
                raise ValueError
            if map_img.ndim == 2:
                map_img = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
        except (KeyError, ValueError):
            # Before map init or missing fields, show blank
            map_img = np.zeros_like(left_vis)

        # Combine and write
        combined = np.hstack((left_vis, map_img))
        out.write(combined)

        # Display in real time
        cv2.imshow('Real-time SLAM', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Video generation complete.")

    # Save final map PLY
    try:
        local_mapper._save_map()
        print(f"Point cloud saved to: {OUTPUT_PLY}")
    except Exception as e:
        print(f"Error saving PLY: {e}")


if __name__ == '__main__':
    main()
