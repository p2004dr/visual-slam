#!/usr/bin/env python
"""
Test script for MapInitializer.
Carga dos frames consecutivos, extrae ORB features, realiza matching,
e inicializa el mapa comprobando los outputs y la visualización.
"""
import os
import sys
import random
import numpy as np
import cv2
import yaml
import pytest

# Paths fijos
CONFIG_PATH   = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'monocular.yaml')
VIDEO_PATH    = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'hospital_video.mp4')
FRAME_OFFSET  = 1   # fotogramas consecutivos
MIN_MATCHES   = 50  # umbral mínimo de matches para el test
MIN_MAPPOINTS = 20  # mínimo de puntos 3D esperados

# Añadir src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from orbslam2.extractor import ORBExtractor
from orbslam2.matcher   import DescriptorMatcher
from orbslam2.initializer import MapInitializer
from orbslam2.utils      import load_config, load_camera_intrinsics, undistort_image

@pytest.fixture(scope="module")
def frames_and_params():
    # Carga config y cámara
    config = load_config(CONFIG_PATH)
    K, dist = load_camera_intrinsics(config)

    # Abrir vídeo y seleccionar dos índices aleatorios consecutivos
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"No se pudo abrir el vídeo {VIDEO_PATH}"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 4
    idx2 = idx + FRAME_OFFSET

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx);  ret, f1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx2); ret2, f2 = cap.read()
    cap.release()
    assert ret and ret2, "Error leyendo los frames del vídeo"

    # Pasar a gris y corregir distorsión
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    if np.any(dist):
        g1 = undistort_image(g1, K, dist)
        g2 = undistort_image(g2, K, dist)

    return {
        'config': config,
        'K': K,
        'frame1': g1,
        'frame2': g2,
        'orig1': f1,
        'orig2': f2
    }

def test_initializer_success_and_mappoints(frames_and_params):
    cfg   = frames_and_params['config']
    K     = frames_and_params['K']
    f1, f2 = frames_and_params['frame1'], frames_and_params['frame2']
    o1, o2 = frames_and_params['orig1'],  frames_and_params['orig2']

    # Crear extractor y matcher
    orb = ORBExtractor(
        n_features=cfg['orb']['n_features'],
        scale_factor=cfg['orb']['scale_factor'],
        n_levels=cfg['orb']['n_levels'],
        ini_threshold=cfg['orb']['ini_threshold'],
        min_threshold=cfg['orb']['min_threshold']
    )
    matcher = DescriptorMatcher(
        matcher_type=cfg['matcher']['matcher_type'],
        ratio_threshold=cfg['matcher']['ratio_threshold']
    )

    # Extraer features
    kp1, des1 = orb.detect_and_compute(f1)
    
    print(type(kp1[0]))  # <class 'cv2.KeyPoint'>
    print(kp1[0].pt)     # (x, y) como tupla
    kp2, des2 = orb.detect_and_compute(f2)
    assert len(kp1) > 0 and len(kp2) > 0, "No se detectaron keypoints en alguno de los frames"

    # Matching
    matches = matcher.match(des1, des2, ratio_test=True)
    assert len(matches) >= MIN_MATCHES, f"Pocos matches ({len(matches)}) para inicializar"

    # Inicialización
    initializer = MapInitializer(camera_matrix=K,
                                 min_matches=MIN_MATCHES,
                                 min_inliers_ratio=0.8)
    initializer.set_first_frame(kp1, des1, f1)
    success, R, t, map_pts, inlier_matches = initializer.initialize(kp2, des2, matcher, f2)

    # Comprobaciones básicas
    assert success is True, "La inicialización falló"
    assert R.shape == (3,3), "R no tiene tamaño 3×3"
    assert t.shape in [(3,), (3,1)], "t no es un vector 3×1"
    assert isinstance(map_pts, list) and len(map_pts) >= MIN_MAPPOINTS, \
        f"Demasiados pocos puntos 3D triangulados: {len(map_pts)}"
    assert isinstance(inlier_matches, list) and len(inlier_matches) > 0, \
        "No se devolvieron matches inliers"

    # Test de dibujo de inicialización
    init_vis = initializer.draw_initialization(o1, o2, inlier_matches)
    assert isinstance(init_vis, np.ndarray), "La visualización no es imagen numpy"
    h, w = init_vis.shape[:2]
    assert h > 0 and w > 0, "Imagen de inicialización vacía"

if __name__ == '__main__':
    pytest.main([__file__])
