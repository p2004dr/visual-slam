import cv2
import numpy as np
import pytest
from orbslam2.tracker import Tracker
from orbslam2.local_mapper import LocalMapper

@pytest.fixture
def dummy_camera():
    # fx=fy=320, cx=320, cy=240
    K = np.array([[320.,   0., 320.],
                  [  0., 320., 240.],
                  [  0.,   0.,   1.]])
    D = np.zeros(5)
    return K, D

@pytest.fixture
def tracker(dummy_camera):
    K, D = dummy_camera
    lm = LocalMapper(camera_matrix=K)
    tr = Tracker(camera_matrix=K, distortion_coeffs=D, n_features=500, local_mapper=lm)
    return tr

def load_gray(path):
    img = cv2.imread(path)
    assert img is not None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def test_initialize_map(tracker, tmp_path):
    # Primer y segundo frame
    img1 = load_gray("data/frame_000.png")
    img2 = load_gray("data/frame_001.png")
    # Processing first frame
    tf1, kps1, vis1 = tracker.process_frame(img1)
    # tf1 debe ser identidad y vis1 debe ser None o keypoints dibujados
    assert tf1.shape == (4,4)
    # Segundo frame: initialize
    tf2, kps2, vis2 = tracker.process_frame(img2)
    assert tracker._initialized is True
    assert tf2.shape == (4,4)
    assert isinstance(vis2, np.ndarray)

def test_frame_to_frame_tracking(tracker):
    # Suponiendo que ya está inicializado con dos imágenes
    # Generamos un frame 3 ligeramente movido (p. ej. traslación en x)
    img3 = np.roll(load_gray("data/frame_000.png"), shift=5, axis=1)
    success, tf3, vis3 = tracker._track_from_last_frame(img3, *tracker.last_features)
    assert success is True
    assert tf3.shape == (4,4)
    assert vis3 is not None

def test_need_new_keyframe(tracker):
    # simulamos distintos índices
    tracker.current_frame_idx = 19
    assert tracker._need_new_keyframe() is False
    tracker.current_frame_idx = 20
    assert tracker._need_new_keyframe() is True

def test_relocalize(tracker):
    # Forzamos estado de tracking perdido
    tracker._tracking_lost = True
    img = load_gray("data/frame_005.png")
    # extraemos features
    success = tracker._relocalize(img, *tracker.extractor.extract_features(img))
    assert isinstance(success, bool)
