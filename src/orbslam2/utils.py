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
    Create a PLY file from 3D points and colors.

    Args:
        points_3d (np.array): Nx3 array of 3D points.
        colors (np.array): Nx3 array of RGB colors or Nx1 grayscale.
        output_path (str): File path to save the PLY file.
    """
    # Ensure numpy arrays
    pts = np.array(points_3d)
    cols = np.array(colors)

    # Debug shapes
    print(f"DEBUG create_point_cloud_ply: pts.shape={pts.shape}, cols.shape={cols.shape}")

    # Ensure colors have shape (N,3)
    if cols.ndim == 1:
        # Grayscale values, replicate to RGB
        cols = np.tile(cols.reshape(-1,1), (1,3))
    elif cols.ndim == 2 and cols.shape[1] == 4:
        # RGBA, drop alpha
        cols = cols[:, :3]
    elif cols.ndim == 2 and cols.shape[1] != 3:
        # Unexpected, trim or pad
        cols = cols[:, :3] if cols.shape[1] > 3 else np.tile(cols, (1, 3 // cols.shape[1] + 1))[:,:3]

    # Write PLY header
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(output_path, 'w') as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(pts, cols):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    print(f"Saved PLY to {output_path}")

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
    Compute 3x4 projection matrix P = K * [R | t]
    Ensures that t tiene forma (3,1) para poder hacer hstack.
    """
    import numpy as np

    # Convertir a array y asegurar 2D
    t = np.array(t)
    # DEBUG: imprimimos dimensiones para verificar
    # print(f"DEBUG compute_projection_matrix: R.shape={R.shape}, t.shape(before)={t.shape}")

    if t.ndim == 1:
        t = t.reshape(3, 1)
    elif t.ndim == 2 and t.shape[0] == 1 and t.shape[1] == 3:
        # si llega como (1,3), convertir a (3,1)
        t = t.reshape(3, 1)
    # tras este reshape, t debe ser (3,1)
    # print(f"DEBUG compute_projection_matrix: t.shape(after)={t.shape}")

    Rt = np.hstack((R, t))  # ahora R: (3,3), t: (3,1) → Rt: (3,4)
    P = camera_matrix @ Rt

    return P
