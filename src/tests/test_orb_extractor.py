#!/usr/bin/env python
"""
Test script for ORB extractor with fixed parameters.
Loads a video, extracts ORB features from random frames, and displays them.
"""
import os
import sys
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Fixed parameters (no argparse needed)
CONFIG_PATH = 'configs/monocular.yaml'
VIDEO_PATH = 'data/hospital_video.mp4'  # Ruta fija al vídeo
NUM_FRAMES = 2  # Número de frames aleatorios a procesar
SAVE_OUTPUT = False  # True para guardar imágenes, False para mostrar
OUTPUT_DIR = 'tests/output'
FIGURE_WIDTH = 20         # ancho global de las figuras
FIGURE_HEIGHT_PER_ROW = 6 # alto por fila de resultados

# Add src directory to path to import orbslam2 modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orbslam2.extractor import ORBExtractor
from orbslam2.utils import load_config, load_camera_intrinsics, undistort_image

def main():
    # Load configuration
    config = load_config(CONFIG_PATH)

    # Load camera intrinsics
    camera_matrix, distortion_coeffs = load_camera_intrinsics(config)

    # Create ORB extractor
    orb_extractor = ORBExtractor(
        n_features=config.get('orb', {}).get('n_features', 2000),
        scale_factor=config.get('orb', {}).get('scale_factor', 1.2),
        n_levels=config.get('orb', {}).get('n_levels', 8),
        ini_threshold=config.get('orb', {}).get('ini_threshold', 20),
        min_threshold=config.get('orb', {}).get('min_threshold', 7)
    )

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {VIDEO_PATH}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

    # Select random frames ONCE for both extractions
    if total_frames <= NUM_FRAMES:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = sorted(random.sample(range(total_frames), NUM_FRAMES))
    print(f"Processing {len(frame_indices)} frames: {frame_indices}")

    # Process each selected frame
    results = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {frame_idx}")
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.any(distortion_coeffs):
            frame_gray = undistort_image(frame_gray, camera_matrix, distortion_coeffs)

        # Standard ORB extraction
        keypoints1, descriptors1 = orb_extractor.detect_and_compute(frame_gray)
        frame_kp1 = orb_extractor.draw_keypoints(frame.copy(), keypoints1)

        # Distributed ORB extraction
        keypoints2, descriptors2 = orb_extractor.distribute_keypoints(frame_gray)
        frame_kp2 = orb_extractor.draw_keypoints(frame.copy(), keypoints2)

        print(f"Frame {frame_idx}:")
        print(f"  Standard extraction: {len(keypoints1)} keypoints")
        print(f"  Distributed extraction: {len(keypoints2)} keypoints")

        results.append({
            'frame_idx': frame_idx,
            'frame': frame,
            'frame_kp1': frame_kp1,
            'frame_kp2': frame_kp2,
            'keypoints1': keypoints1,
            'keypoints2': keypoints2
        })

    cap.release()

    # Display results for each frame: only one original + standard + distributed per row
    if SAVE_OUTPUT:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for result in results:
            idx = result['frame_idx']
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{idx}.png"), result['frame'])
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{idx}_kp1.png"), result['frame_kp1'])
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{idx}_kp2.png"), result['frame_kp2'])
        print(f"Results saved to {OUTPUT_DIR}")
    else:
        rows = len(results)
        plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT_PER_ROW * rows))
        gs = GridSpec(rows, 3, figure=plt.gcf(), hspace=0.2, wspace=0.1)
        for i, result in enumerate(results):
            # Original frame
            ax0 = plt.subplot(gs[i, 0])
            ax0.imshow(cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB))
            ax0.set_title(f"Frame {result['frame_idx']}")
            ax0.axis('off')

            # Standard extraction
            ax1 = plt.subplot(gs[i, 1])
            ax1.imshow(cv2.cvtColor(result['frame_kp1'], cv2.COLOR_BGR2RGB))
            ax1.set_title(f"Standard: {len(result['keypoints1'])} kpts")
            ax1.axis('off')

            # Distributed extraction
            ax2 = plt.subplot(gs[i, 2])
            ax2.imshow(cv2.cvtColor(result['frame_kp2'], cv2.COLOR_BGR2RGB))
            ax2.set_title(f"Distributed: {len(result['keypoints2'])} kpts")
            ax2.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
