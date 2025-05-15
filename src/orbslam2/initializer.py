"""
Map initialization module for ORB-SLAM2 Python implementation.
"""

import numpy as np
import cv2


class MapInitializer:
    """
    Class for initializing the map from two frames.
    """
    
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.first_frame = None
        self.first_kps = None
        self.first_desc = None
        self.matches = None
        self.last_matches = None

    def initialize(self, kps1, desc1, kps2, desc2, matcher):
        matches = matcher.match(desc1, desc2)
        self.last_matches = matches

        # Reducir mínimo de matches requeridos
        if len(matches) < 80:  # Cambiado de 100 a 80
            return False, None, None, matches
        
        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

        # Aumentar umbral RANSAC para Essential Matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=3.0  # Aumentado de 1.0 a 3.0
        )
        
        if E is None or mask.sum() < 40:  # Reducido de 50 a 40
            return False, None, None, matches
            
        inliers_mask = mask.ravel() > 0
        pts1_inliers = pts1[inliers_mask]
        pts2_inliers = pts2[inliers_mask]
        matches_inliers = [m for i, m in enumerate(matches) if inliers_mask[i]]
        
        # Recuperar pose con chequeo mejorado
        _, R, t, mask_pose = cv2.recoverPose(
            E, pts1_inliers, pts2_inliers,
            self.camera_matrix,
            mask=mask[inliers_mask].ravel()  # Pasar máscara actualizada
        )

        # Validación 3D de puntos triangulados
        if mask_pose.sum() >= 40:  # Reducido de 50 a 40
            valid = mask_pose.ravel() > 0
            pts1_valid = pts1_inliers[valid]
            pts2_valid = pts2_inliers[valid]
            
            # Matrices de proyección
            P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = np.hstack([R, t.reshape(3, 1)])
            
            # Triangulación directa
            points_4d = cv2.triangulatePoints(P1, P2, pts1_valid.T, pts2_valid.T)
            points_3d = (points_4d[:3] / points_4d[3]).T
            
            # Verificar profundidades positivas en ambos sistemas de coordenadas
            depths1 = points_3d[:, 2]
            depths2 = (points_3d @ R.T + t.reshape(1, 3))[:, 2]
            valid_depths = (depths1 > 0.01) & (depths2 > 0.01)  # Evitar puntos muy cercanos
            
            if valid_depths.sum() < 0.5 * len(points_3d):
                print(f"Puntos 3D inválidos: {valid_depths.sum()}/{len(points_3d)}")
                return False, None, None, matches_inliers
        else:
            return False, None, None, matches_inliers

        # Chequeo de paralaje ajustado
        if not self._check_parallax(R, t, pts1_valid, pts2_valid):
            return False, None, None, matches_inliers
            
        return True, R, t, matches_inliers

    def _check_parallax(self, R, t, pts1, pts2, min_angle_deg=1.0):
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.camera_matrix, None)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.camera_matrix, None)
        
        # Corregir sintaxis de hstack
        rays1 = np.hstack((pts1_norm.reshape(-1, 2), np.ones((len(pts1), 1))))  # <-- Paréntesis adicional
        rays2 = np.hstack((pts2_norm.reshape(-1, 2), np.ones((len(pts2), 1))))  # <-- Paréntesis adicional
        
        rays1 /= np.linalg.norm(rays1, axis=1, keepdims=True)
        rays2_in_1 = (rays2 @ R.T) / np.linalg.norm(rays2, axis=1, keepdims=True)
        
        cos_angles = np.sum(rays1 * rays2_in_1, axis=1)
        angles_deg = np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))
        
        sufficient_parallax = np.mean(angles_deg > min_angle_deg) > 0.3
        print(f"Paralaje promedio: {np.mean(angles_deg):.1f}°, Buenas: {np.sum(angles_deg > min_angle_deg)}/{len(angles_deg)}")
        
        return sufficient_parallax