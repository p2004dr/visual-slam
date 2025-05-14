#!/usr/bin/env python
import os
import glob
import cv2
import numpy as np
import yaml
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

BASEDIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_DIR   = os.path.join(BASEDIR, 'src')
sys.path.append(SRC_DIR)

from orbslam2.extractor import ORBExtractor
from orbslam2.matcher   import DescriptorMatcher
from orbslam2.utils      import load_groundtruth_matcher

GT_DIR     = 'data/groundtruth_matches'
RATIO_TEST = True
SHOW_PREDICTIONS = False


def show_groundtruth(ax1, ax2, gt_pts1, gt_pts2, gt_matches, img1, img2):
    ax1.imshow(img1); ax1.set_title('GT: Image 1')
    ax2.imshow(img2); ax2.set_title('GT: Image 2')
    for (x, y) in gt_pts1:
        ax1.plot(x, y, 'go', markersize=4)
    for (u, v) in gt_pts2:
        ax2.plot(u, v, 'go', markersize=4)
    for idx1, idx2 in gt_matches:
        x, y = gt_pts1[idx1]
        u, v = gt_pts2[idx2]
        con = ConnectionPatch(
            xyA=(x, y), xyB=(u, v), coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2, color="r", linewidth=1
        )
        ax2.add_artist(con)
    ax1.axis('off'); ax2.axis('off')


def show_predictions(ax1, ax2, kp1, kp2, pred, img1, img2):
    ax1.imshow(img1); ax1.set_title('Pred: Image 1')
    ax2.imshow(img2); ax2.set_title('Pred: Image 2')
    for kp in kp1:
        x, y = kp.pt; ax1.plot(x, y, 'bx', markersize=4)
    for kp in kp2:
        u, v = kp.pt; ax2.plot(u, v, 'bx', markersize=4)
    for m in pred:
        x, y = kp1[m.queryIdx].pt
        u, v = kp2[m.trainIdx].pt
        con = ConnectionPatch(
            xyA=(x, y), xyB=(u, v), coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2, color="b", linewidth=1
        )
        ax2.add_artist(con)
    ax1.axis('off'); ax2.axis('off')

def evaluate_pair(pair_path, orb, matcher):
    # 1) Carga del ground‑truth
    gt_pts1, gt_pts2, gt_matches = load_groundtruth_matcher(
        os.path.join(pair_path, 'gt.yaml')
    )
    img1_color = cv2.cvtColor(
        cv2.imread(os.path.join(pair_path, 'img1.png')),
        cv2.COLOR_BGR2RGB
    )
    img2_color = cv2.cvtColor(
        cv2.imread(os.path.join(pair_path, 'img2.png')),
        cv2.COLOR_BGR2RGB
    )

    # 2) Mostrar ground‑truth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    show_groundtruth(ax1, ax2,
                     gt_pts1, gt_pts2, gt_matches,
                     img1_color, img2_color)
    plt.tight_layout()
    plt.show()

    # 3) Preparamos las imágenes en gris
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_RGB2GRAY)

    # 4) Creamos los KeyPoint en las posiciones del GT
    #    (nota: el tercer parámetro es 'size', no '_size')
    kp1 = [cv2.KeyPoint(float(x), float(y), 31) for (x, y) in gt_pts1]
    kp2 = [cv2.KeyPoint(float(u), float(v), 31) for (u, v) in gt_pts2]

    # 5) Computamos descriptores SOLO en esos keypoints
    kp1, des1 = orb.compute(img1_gray, kp1)
    kp2, des2 = orb.compute(img2_gray, kp2)

    # 6) Hacemos el matching con esos descriptores
    pred = matcher.match(des1, des2, ratio_test=RATIO_TEST)

    # 7) Mostrar predicciones
    if SHOW_PREDICTIONS:
        fig, (p1, p2) = plt.subplots(1, 2, figsize=(10,5))
        show_predictions(p1, p2, kp1, kp2, pred, img1_color, img2_color)
        plt.tight_layout()
        plt.show()

    # 8) Cálculo de métricas (precision, recall, F1)
    pred_pairs = {(m.queryIdx, m.trainIdx) for m in pred}
    gt_set     = set(gt_matches)
    tp = len(gt_set & pred_pairs)
    fp = len(pred_pairs - gt_set)
    fn = len(gt_set - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)

    return {'precision': precision,
            'recall':    recall,
            'f1':        f1}

def main():
    orb     = ORBExtractor(n_features=2000, scale_factor=1.2,
                           n_levels=8, ini_threshold=20, min_threshold=7)
    matcher = DescriptorMatcher(matcher_type='bruteforce-hamming', ratio_threshold=0.85)
    results = []
    pairs = sorted(glob.glob(os.path.join(GT_DIR, 'pair*')))
    for pair_path in pairs:
        print(f"\n=== Evaluating {os.path.basename(pair_path)} ===")
        res = evaluate_pair(pair_path, orb, matcher)
        print(f"Precision={res['precision']:.2f}, Recall={res['recall']:.2f}, F1={res['f1']:.2f}")
        results.append(res)

    avg = {k: np.mean([r[k] for r in results]) for k in ('precision','recall','f1')}
    print(f"\nAverage: Precision={avg['precision']:.2f}, Recall={avg['recall']:.2f}, F1={avg['f1']:.2f}")

    # Show only first pair GT and predictions again
    if pairs:
        print(f"\n--- First pair summaries ({os.path.basename(pairs[0])}) ---")
        evaluate_pair(pairs[0], orb, matcher)

if __name__ == '__main__':
    main()
