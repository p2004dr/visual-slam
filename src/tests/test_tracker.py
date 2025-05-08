#!/usr/bin/env python
"""
Standalone test script for the Tracker module.
Loads three consecutive frames from the video (first two for map initialization + third for frame-to-frame tracking),
runs the tracker, prints the resulting transformation matrices and displays the visualization images.
"""
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Configurable paths
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # points to src directory where orbslam2 lives
VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data', 'hospital_video.mp4'))
FRAME_OFFSET_INIT = 0   # starting frame index for init

# Add src directory to path to import modules
sys.path.append(SRC_DIR)
from orbslam2.local_mapper import LocalMapper
from orbslam2.tracker import Tracker


def grab_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to grab frame at index {idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def main():
    # Dummy camera intrinsics (fx=fy=320, cx=320, cy=240)
    K = np.array([[320.,   0., 320.],
                  [  0., 320., 240.],
                  [  0.,   0.,   1.]])
    D = np.zeros(5)

    # Initialize modules
    local_mapper = LocalMapper(camera_matrix=K)
    tracker = Tracker(camera_matrix=K, distortion_coeffs=D, n_features=1000, local_mapper=local_mapper)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open video at {VIDEO_PATH}")
        return

    # Determine three frame indices
    idx1 = FRAME_OFFSET_INIT
    idx2 = idx1 + 1
    idx3 = idx2 + 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if idx3 >= total:
        print(f"Error: video too short (only {total} frames)")
        cap.release()
        return

    # Grab frames
    gray1 = grab_frame(cap, idx1)
    gray2 = grab_frame(cap, idx2)
    gray3 = grab_frame(cap, idx3)
    cap.release()

    # Process first frame (initialization)
    print("--- Processing first frame (initialization) ---")
    tf1, kps1, vis1 = tracker.process_frame(gray1)
    print(f"First transform:\n{tf1}")
    if vis1 is not None:
        plt.figure(figsize=(6,6))
        plt.imshow(vis1, cmap='gray')
        plt.title('Keypoints Frame 1')
        plt.axis('off')

    # Process second frame (map init)
    print("\n--- Processing second frame (map initialization) ---")
    tf2, kps2, vis2 = tracker.process_frame(gray2)
    print(f"Second transform:\n{tf2}")
    if vis2 is not None:
        plt.figure(figsize=(6,6))
        plt.imshow(vis2, cmap='gray')
        plt.title('Initialization Matches')
        plt.axis('off')

    # Process third frame (frame-to-frame tracking)
    print("\n--- Processing third frame (frame-to-frame tracking) ---")
    tf3, kps3, vis3 = tracker.process_frame(gray3)
    print(f"Third transform:\n{tf3}")
    if vis3 is not None:
        plt.figure(figsize=(6,6))
        plt.imshow(vis3, cmap='gray')
        plt.title('Tracking Matches')
        plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()
