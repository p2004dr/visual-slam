import numpy as np
import cv2
import yaml


def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_camera_intrinsics(config):
    """
    Extract camera intrinsic parameters from configuration.
    Reads 'camera_matrix' (flattened row-major 9 floats) and 'distortion_coeffs' (5 floats).

    Args:
        config (dict): Configuration parameters loaded from YAML.

    Returns:
        tuple: (camera_matrix (3x3 np.array), distortion_coeffs (1x5 np.array))
    """
    cam = config.get('camera', {})
    if 'camera_matrix' not in cam or 'distortion_coeffs' not in cam:
        raise KeyError("'camera_matrix' or 'distortion_coeffs' missing in 'camera' section of config")
    # Reshape flattened list into 3x3 matrix
    camera_matrix = np.array(cam['camera_matrix'], dtype=float).reshape(3, 3)
    distortion_coeffs = np.array(cam['distortion_coeffs'], dtype=float)
    return camera_matrix, distortion_coeffs


def undistort_image(image, camera_matrix, distortion):
    """
    Undistort an image using camera intrinsics and distortion.

    Args:
        image (np.array): Input image (grayscale or BGR).
        camera_matrix (np.array): 3x3 camera intrinsic matrix.
        distortion (np.array): Distortion coefficients.

    Returns:
        np.array: Undistorted image.
    """
    return cv2.undistort(image, camera_matrix, distortion)


# En utils.py (corregido)
def triangulate_points(points1, points2, P1, P2):
    points1 = points1.T  # Convierte (N,2) → (2,N)
    points2 = points2.T
    points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
    return points_4d  # Shape (4,N)

def convert_to_3d_points(points_4d):
    """Convierte puntos homogéneos 4D (x,y,z,w) a 3D euclídeos (x/w, y/w, z/w)."""
    # Asegurar que el array es (4, N)
    if points_4d.shape[0] != 4:
        points_4d = points_4d.T
    
    # Dividir por w y transponer para obtener (N, 3)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d


def create_point_cloud_ply(points_3d, colors, output_path):
    """
    Save 3D points and corresponding colors to a PLY file.

    Args:
        points_3d (np.array): 3xN array of 3D points.
        colors (np.array): N×3 array of RGB values (0-255).
        output_path (str): File path for the PLY output.
    """
    points_3d = points_3d.T
    colors = colors.astype(int)
    num_points = points_3d.shape[0]
    with open(output_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(points_3d, colors):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def calculate_essential_matrix(points1, points2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0):
    """
    Compute the Essential matrix from corresponding points.
    """
    E, mask = cv2.findEssentialMat(points1, points2, camera_matrix,
                                   method=method, prob=prob, threshold=threshold)
    return E, mask


def recover_pose(E, points1, points2, camera_matrix, mask=None):
    """
    Recover relative pose (R, t) from the Essential matrix.
    """
    _, R, t, pose_mask = cv2.recoverPose(E, points1, points2, camera_matrix, mask=mask)
    return _, R, t, pose_mask


def compute_projection_matrix(R, t, camera_matrix):
    """
    Compute projection matrix P = K [R|t].
    """
    Rt = np.hstack((R, t))
    return camera_matrix @ Rt