import os
import sys
import random
import math
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt

# Fixed parameters
CONFIG_PATH = 'configs/monocular.yaml'
VIDEO_PATH = 'data/video_gerard_1.mp4'
FRAME_OFFSET = 1         # Difference between frames to match (1 = consecutive)
GEOM_THRESHOLD = 0.055   # Percentage (0–1) of (width+height)/2 para filtrar desplazamientos grandes
SAVE_OUTPUT = False      # True to save images instead of showing
OUTPUT_DIR = 'tests/output'
FIGURE_WIDTH = 12        # Width of match figure
FIGURE_HEIGHT = 8        # Height of match figure

# Add src directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orbslam2.extractor import ORBExtractor
from orbslam2.matcher import DescriptorMatcher
from orbslam2.utils import load_config, load_camera_intrinsics, undistort_image

def main():
    # Load config
    config = load_config(CONFIG_PATH)
    camera_matrix, distortion_coeffs = load_camera_intrinsics(config)

    # Create ORB extractor
    orb = ORBExtractor(
        n_features=config['orb']['n_features'],
        scale_factor=config['orb']['scale_factor'],
        n_levels=config['orb']['n_levels'],
        ini_threshold=config['orb']['ini_threshold'],
        min_threshold=config['orb']['min_threshold']
    )

    # Create matcher
    matcher = DescriptorMatcher(
        matcher_type=config['matcher']['matcher_type'],
        ratio_threshold=config['matcher']['ratio_threshold']
    )

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open {VIDEO_PATH}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = random.randint(0, total - FRAME_OFFSET - 1)
    idx2 = idx + FRAME_OFFSET

    # Read frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx2)
    ret2, frame2 = cap.read()
    cap.release()

    if not ret or not ret2:
        print("Error reading frames.")
        return

    # Preprocess: grayscale + undistort
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    if np.any(distortion_coeffs):
        gray1 = undistort_image(gray1, camera_matrix, distortion_coeffs)
        gray2 = undistort_image(gray2, camera_matrix, distortion_coeffs)

    # Extract features
    kp1, des1 = orb.detect_and_compute(gray1)
    kp2, des2 = orb.detect_and_compute(gray2)

    # Match descriptors (Lowe's ratio test)
    matches = matcher.match(des1, des2, ratio_test=True)
    print(f"Raw matches Frame {idx} → {idx2}: {len(matches)}")

    # Geometric-distance filtering
    img_h, img_w = gray1.shape
    filtered_matches = matcher.filter_matches_by_geometric_distance(
        kp1, kp2, matches,
        threshold_percent=GEOM_THRESHOLD,
        image_shape=(img_h, img_w)
    )
    print(f"Filtered matches (geom ≤ {GEOM_THRESHOLD*100:.1f}% of (w+h)/2): {len(filtered_matches)}")

    # Draw filtered matches
    match_img = matcher.draw_matches(frame1, kp1, frame2, kp2, filtered_matches)

    # Show or save
    if SAVE_OUTPUT:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"match_{idx}_{idx2}.png")
        cv2.imwrite(out_path, match_img)
        print(f"Saved match image to {out_path}")
    else:
        plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Matches: {len(filtered_matches)} between frames {idx} & {idx2}")
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()