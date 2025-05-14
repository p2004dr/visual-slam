# visual-slam


## Estructura del proyecto
visual-slam/
├── configs/
│   └── monocular.yaml       # Parámetros de cámara y SLAM (ORB, matcher, tracking, mapping, I/O y opciones de ejecución)
│
├── data/
│   └── hospital_video.mp4   # Vídeo de entrada sobre el que se ejecuta el SLAM
│
├── src/
│   ├── orbslam2/            # Implementación de un SLAM monocular inspirado en ORB-SLAM2
│   │   ├── __init__.py      # Inicializa el paquete y contiene la versión
│   │   ├── extractor.py     # ORBExtractor: detecta y calcula descriptores ORB, con distribución por grillas y dibujo de keypoints
│   │   ├── initializer.py   # MapInitializer: inicializa el mapa a partir de los dos primeros fotogramas (E, recuperación de pose, triangulación)
│   │   ├── matcher.py       # DescriptorMatcher: empareja descriptores ORB (Brute-Force, FLANN), ratio test, filtrado por distancia y fundamental
│   │   ├── tracker.py       # Tracker: lógica de tracking frame-a-frame, inicialización, relocalización, creación de keyframes y paso de poses al mapeador
│   │   ├── local_mapper.py  # LocalMapper: añade keyframes, crea y refina puntos de mapa, poda/outliers, exporta PLY, estadísticas y visualización
│   │   └── utils.py         # Funciones auxiliares: carga de config, intrínsecos, corrección de distorsión, triangulación, cálculo de E, P, generación de PLY…
│   │
│   └── tests/               # Conjunto de tests unitarios para validar cada módulo
│       ├── test_matcher.py      # Pruebas de DescriptorMatcher (matching, filtrado, enmascarado)
│       └── test_orb_extractor.py# Pruebas de ORBExtractor (detección, computación de descriptores, distribución de keypoints)
│
└── README.md                # (Opcional) Guía de uso, instalación y ejemplos de ejecución


## Descripción de cada componente
configs/monocular.yaml
    Define todos los parámetros configurables del sistema:

    camera: matriz intrínseca y coeficientes de distorsión.

    orb: número de features, niveles de pirámide, thresholds FAST.

    matcher: tipo de matcher y umbral de Lowe.

    tracker: condiciones para keyframes y tracking mínimo.

    mapper: tamaño de ventana local, reproyección máxima, número mínimo de observaciones.

    I/O: rutas de vídeo y de salida del mapa PLY.

    runtime: frames a saltar, máximo de frames a procesar, visualización en pantalla.

data/hospital_video.mp4
    Vídeo de prueba monocular utilizado como input para extraer características, trackear la cámara y generar el mapa 3D.

src/orbslam2/
    Implementación núcleo:

    extractor.py:

    ORBExtractor: encapsula la extracción de keypoints y descriptores ORB, incluye métodos para distribuirlos uniformemente y visualizarlos.

    initializer.py:

    MapInitializer: toma los dos primeros frames, calcula la matriz esencial, recupera R, t, triangula puntos 3D y genera el conjunto inicial de map points.

    matcher.py:

    DescriptorMatcher: ofrece matching BF-Hamming o FLANN, aplica ratio test de Lowe, filtrado por distancia y por prueba de matriz fundamental (RANSAC), y dibuja matches.

    tracker.py:

    Tracker: coordina el extractor, matcher e initializer. Se encarga de la lógica de iniciar el SLAM, trackear cada frame, decidir keyframes, relocalizar y pasar keyframes al mapeador.

    local_mapper.py:

    LocalMapper: gestiona la incorporación de nuevos keyframes, creación y refinamiento de puntos de mapa, poda de outliers y redundancias, guarda el mapa en PLY y ofrece estadísticas/visualización.

    utils.py:

    Funciones comunes: lectura de YAML, carga/calibración de cámara, corrección de distorsión, triangulación, conversión a puntos 3D, cálculo de E y P, exportación de nubes de puntos.

src/tests/
    Pruebas unitarias basadas en pytest para asegurar que:

    test_orb_extractor.py cubre detección, cómputo de descriptores y distribución uniforme.

    test_matcher.py verifica diversas estrategias de matching, ratio test y filtrado.

# Camera parameters
camera:
  camera_matrix: [320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0]  # fx, 0, cx, 0, fy, cy, 0, 0, 1
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]  # k1, k2, p1, p2, k3

# ORB parameters
orb:
  n_features: 2000
  scale_factor: 1.2
  n_levels: 8
  ini_threshold: 20
  min_threshold: 7

# Matching parameters
matcher:
  matcher_type: "bruteforce-hamming"
  ratio_threshold: 0.75

# Tracking parameters
tracker:
  min_keypoints: 100
  max_frame_to_keyframe_dist: 50
  min_tracked_points: 50

# Mapping parameters
mapper:
  local_window_size: 10
  min_observations: 3
  max_reprojection_error: 2.0

# Input/Output
video_path: "data/hospital_video.mp4"
output_path: "data/map.ply"

# Runtime options
skip_frames: 0
max_frames: 0  # 0 means process all frames
display: true
Y tenemos una carpeta data con hospital_video.mp4
Y tenemos la carpeta src dnde tenemos todo el codigo, dentro de src tenemos una carpeta llamada orbslam2 que contiene:
__init.py__
"""
Python implementation of a simplified ORB-SLAM2 system.

This package implements a monocular visual SLAM system inspired by ORB-SLAM2.
It extracts ORB features from video frames, tracks them across frames,
and builds a 3D map of the environment.
"""

__version__ = '0.1.0'
extractor.py
"""
Feature extraction module for ORB-SLAM2 Python implementation.

This module handles the extraction of ORB features from images,
which are used for tracking and mapping.
"""

class ORBExtractor:
    """
    ORB feature extractor class.
    
    Extracts ORB features and descriptors from images for tracking and mapping.
    """
    
    def __init__(self, n_features=2000, scale_factor=1.2, n_levels=8, 
                 ini_threshold=20, min_threshold=7):
        """
        Initialize the ORB extractor.
        
        Args:
            n_features (int): Number of features to extract.
            scale_factor (float): Scale factor between pyramid levels.
            n_levels (int): Number of pyramid levels.
            ini_threshold (int): Initial FAST threshold.
            min_threshold (int): Minimum FAST threshold.
        """
    def detect_and_compute(self, image):
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image (np.array): Input grayscale image.
            
        Returns:
            tuple: (keypoints, descriptors)
        """
    def distribute_keypoints(self, image, n_features=None):
        """
        Distribute keypoints evenly across the image using grid-based detection.
        
        Args:
            image (np.array): Input grayscale image.
            n_features (int, optional): Override number of features.
            
        Returns:
            tuple: (keypoints, descriptors)
        """
    def extract_features(self, image, distributed=True):
        """
        Extract features from an image.
        
        Args:
            image (np.array): Input image.
            distributed (bool): Whether to use grid-based feature distribution.
            
        Returns:
            tuple: (keypoints, descriptors)
        """
    def draw_keypoints(self, image, keypoints):
y tambien tenemos initializer.py:
"""
Map initialization module for ORB-SLAM2 Python implementation.

This module handles the initialization of the map from the first two frames.
"""

import numpy as np
import cv2
from .utils import triangulate_points, convert_to_3d_points, calculate_essential_matrix, recover_pose, compute_projection_matrix

class MapInitializer:
    """
    Class for initializing the map from the first two frames.
    """
    
    def __init__(self, camera_matrix, min_matches=100, min_inliers_ratio=0.9):
        """
        Initialize the map initializer.
        
        Args:
            camera_matrix (np.array): 3x3 camera intrinsic matrix.
            min_matches (int): Minimum number of matches required for initialization.
            min_inliers_ratio (float): Minimum ratio of inliers for initialization.
        """
    def set_first_frame(self, keypoints, descriptors, image):
        """
        Set the first frame keypoints and descriptors.
        
        Args:
            keypoints (list): List of keypoints from the first frame.
            descriptors (np.array): Descriptors corresponding to keypoints.
            image (np.array): First frame image for later triangulation.
        """
    def initialize(self, current_keypoints, current_descriptors, matcher, current_image):
        """
        Initialize the map using the first and current frame.
        
        Args:
            current_keypoints (list): List of keypoints from the current frame.
            current_descriptors (np.array): Descriptors corresponding to current keypoints.
            matcher (DescriptorMatcher): Descriptor matcher instance.
            current_image (np.array): Current frame image.
            
        Returns:
            tuple: (success, R, t, initial_map_points, matches)
        """
    def draw_initialization(self, first_image, current_image, matches):
        """
        Draw initialization results for visualization.
        
        Args:
            first_image (np.array): First frame image.
            current_image (np.array): Current frame image.
            matches (list): List of matches.
            
        Returns:
            np.array: Image with matches drawn.
        """
tb tenemos local_mapper.py:
"""
Local mapping module for ORB-SLAM2 Python implementation.

This module handles keyframe processing, map point creation, and map refinement.
"""
from .utils import triangulate_points, convert_to_3d_points, compute_projection_matrix, create_point_cloud_ply


class LocalMapper:
    """
    Class for local mapping, creating and optimizing the map.
    """
    
    def __init__(self, camera_matrix, output_path=None):
        """
        Initialize the local mapper.
        
        Args:
            camera_matrix (np.array): 3x3 camera intrinsic matrix.
            output_path (str): Path to save PLY map file.
        """
    def set_output_path(self, output_path):
        """
        Set the output path for the PLY map file.
        
        Args:
            output_path (str): Path to save PLY map file.
        """
    def add_keyframe(self, image, keypoints, descriptors, pose):
        """
        Add a new keyframe to the map.
        
        Args:
            image (np.array): Keyframe image.
            keypoints (list): List of keypoints.
            descriptors (np.array): Descriptors.
            pose (np.array): 4x4 camera pose (world to camera).
        """
    def update_map_points(self, new_map_points):
        """
        Update map with new map points (used during initialization).
        
        Args:
            new_map_points (list): List of new map points.
        """
        # Merge new map points with existing ones (during initialization)
        for mp in new_map_points:
            self.map_points.append(mp)
    def _process_new_keyframe(self, new_keyframe):
        """
        Process a new keyframe to create new map points.
        
        Args:
            new_keyframe (dict): New keyframe dictionary.
        """
    def _update_map(self):
        """
        Update the map by removing outliers, optimizing positions, etc.
        """
    def _cull_map_points(self):
        """
        Remove low-quality map points.
        """
    def _cull_keyframes(self):
        """
        Remove redundant keyframes.
        """
    def _filter_map_points(self):
        """
        Filter map points to remove outliers.
        
        Returns:
            list: Filtered map points.
        """
    def _save_map(self):
        """
        Save the map as a PLY file.
        """
    def get_map_statistics(self):
        """
        Get statistics about the current map.
        
        Returns:
            dict: Map statistics.
        """
    def visualize_map(self, width=800, height=600):
        """
        Create a simple visualization of the map and camera trajectory.
        
        Args:
            width (int): Image width.
            height (int): Image height.
            
        Returns:
            np.array: Visualization image.
        """
    def get_camera_trajectory(self):
        """
        Get the camera trajectory for visualization.
        
        Returns:
            np.array: Array of camera positions.
        """
    def get_keyframe_by_id(self, keyframe_id):
        """
        Get a keyframe by its ID.
        
        Args:
            keyframe_id (int): Keyframe ID.
            
        Returns:
            dict: Keyframe or None if not found.
        """
    def get_map_point_by_id(self, map_point_id):
        """
        Get a map point by its ID.
        
        Args:
            map_point_id (int): Map point ID.
            
        Returns:
            dict: Map point or None if not found.
        """
    def find_connected_keyframes(self, keyframe_id, min_connections=15):
        """
        Find keyframes connected to the given keyframe in the co-visibility graph.
        
        Args:
            keyframe_id (int): Keyframe ID.
            min_connections (int): Minimum number of common observations.
            
        Returns:
            list: List of connected keyframe IDs.
        """
tb tenemos matcher.py:
"""
Feature matching module for ORB-SLAM2 Python implementation.

This module handles matching features between frames using ORB descriptors.
"""
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
tb tenemos tracker.py:
"""
Tracking module for ORB-SLAM2 Python implementation.

This module handles frame-to-frame tracking and pose estimation.
"""

import numpy as np
import cv2
from .extractor import ORBExtractor
from .matcher import DescriptorMatcher
from .initializer import MapInitializer
from .utils import compute_projection_matrix


class Tracker:
    """
    Class for tracking camera motion and features across frames.
    """
    
    def __init__(self, camera_matrix, distortion_coeffs=None, 
                 n_features=2000, local_mapper=None):
        """
        Initialize the tracker.
        
        Args:
            camera_matrix (np.array): 3x3 camera intrinsic matrix.
            distortion_coeffs (np.array): Camera distortion coefficients.
            n_features (int): Number of features to extract per frame.
            local_mapper (LocalMapper): Local mapping module instance.
        """
    def set_local_mapper(self, local_mapper):
        """
        Set the local mapper reference.
        
        Args:
            local_mapper (LocalMapper): Local mapping module instance.
        """
        self.local_mapper = local_mapper
    def process_frame(self, frame):
        """
        Process a new frame for tracking.
        
        Args:
            frame (np.array): Input frame.
            
        Returns:
            tuple: (transformation_matrix, keypoints, visualization_image)
        """
    def _initialize(self, frame, keypoints, descriptors):
        """
        Initialize the map with the first two frames.
        
        Args:
            frame (np.array): Current frame.
            keypoints (list): List of keypoints.
            descriptors (np.array): Descriptors.
            
        Returns:
            bool: Success flag.
        """
    def _track_from_last_frame(self, frame, keypoints, descriptors):
        """
        Track camera motion from the last frame.
        
        Args:
            frame (np.array): Current frame.
            keypoints (list): List of keypoints.
            descriptors (np.array): Descriptors.
            
        Returns:
            tuple: (success, transformation_matrix, visualization_image)
        """
    def _update_pose(self, transformation_matrix):
        """
        Update the camera pose with a new transformation.
        
        Args:
            transformation_matrix (np.array): 4x4 transformation matrix.
        """
    def _need_new_keyframe(self):
        """
        Determine if a new keyframe should be created.
        
        Returns:
            bool: True if a new keyframe is needed.
        """
        # Simple heuristic: create a keyframe every 20 frames
        # In a real system, this would be based on parallax, tracking quality, etc.
        return self.current_frame_idx % 20 == 0
    def _create_keyframe(self, frame, keypoints, descriptors):
        """
        Create a new keyframe and pass it to the local mapper.
        
        Args:
            frame (np.array): Frame image.
            keypoints (list): List of keypoints.
            descriptors (np.array): Descriptors.
        """
    def _relocalize(self, frame, keypoints, descriptors):
        """
        Try to relocalize the camera when tracking is lost.
        
        Args:
            frame (np.array): Current frame.
            keypoints (list): List of keypoints.
            descriptors (np.array): Descriptors.
            
        Returns:
            bool: Success flag.
        """
    def _compose_transform(self, R, t):
        """
        Compose a 4x4 transformation matrix from rotation and translation.
        
        Args:
            R (np.array): 3x3 rotation matrix.
            t (np.array): 3x1 translation vector.
            
        Returns:
            np.array: 4x4 transformation matrix.
        """
    def get_trajectory(self):
        """
        Get the camera trajectory.
        
        Returns:
            np.array: Array of camera positions.
        """
y por ultimo tenemos utils.py con funciones varias como def load_config(config_path): , def load_camera_intrinsics(config): , def undistort_image(image, camera_matrix, distortion): , def triangulate_points(kp1, kp2, P1, P2): , def convert_to_3d_points(points_4d): , def create_point_cloud_ply(points_3d, colors, output_path): , def calculate_essential_matrix(points1, points2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0): , def recover_pose(E, points1, points2, camera_matrix, mask=None):  y def compute_projection_matrix(R, t, camera_matrix):

Finalmente, tenemos otra carpeta dentro de src llamada tests, donde tenemos test_matcher.py y test_orb_extractor.py que se encargan de testear los scripts previamente definidos de la libreria q hemos creado orbslam2 