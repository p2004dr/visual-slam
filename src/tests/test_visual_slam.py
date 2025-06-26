import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src directory to path
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_DIR = os.path.join(BASEDIR, 'src')
sys.path.append(SRC_DIR)
from orbslam2.extractor import ORBExtractor
from orbslam2.matcher import DescriptorMatcher
from orbslam2.mapping import VisualSLAM
from orbslam2.utils import load_config, load_camera_intrinsics

# Configuration constants
CONFIG_PATH = os.path.join(BASEDIR, 'configs', 'monocular_gerard_low.yaml')
VIDEO_PATH = os.path.join(BASEDIR, 'data/videos', 'video_gerard_1_l.mp4')
KEYFRAME_INTERVAL = 5     # Force keyframe every N frames, None for automatic
VISUALIZE_FREQ = 2        # Update visualization every N framesq
SAVE_OUTPUT = False       # True to save images instead of showing
OUTPUT_DIR = 'tests/output'
SHOW_FINAL_MAP = True     # Show final 3D map after processing
MANUAL_MODE = False       # True: process frames on keypress, False: automatic
SHOW_MATCHES = True       # Show matches between consecutive frames

# Global variables for event handling
process_frame = False
paused = False

class CameraVisualizer:
    """Handles camera frustum visualization for SLAM"""
    
    @staticmethod
    def create_frustum(pose, scale=0.1, fov_degrees=60):
        """Create camera frustum for visualization"""
        # Camera center
        R = pose[:3, :3]
        t = pose[:3, 3]
        C = -R.T @ t
        
        # Calculate frustum corners based on FOV
        fov_rad = np.deg2rad(fov_degrees / 2)
        aspect_ratio = 4/3
        
        # Tangent of half FOV
        tan_fov_h = np.tan(fov_rad)
        tan_fov_v = tan_fov_h / aspect_ratio
        
        # Define canonical frustum vertices
        depth = scale * 2
        width = depth * tan_fov_h
        height = depth * tan_fov_v
        
        canonical_pts = np.array([
            [0, 0, 0],                    # Camera center
            [width, height, depth],       # Top-right-front
            [width, -height, depth],      # Bottom-right-front
            [-width, -height, depth],     # Bottom-left-front
            [-width, height, depth],      # Top-left-front
        ])
        
        # Transform vertices to world coordinates
        world_pts = []
        for pt in canonical_pts:
            if np.array_equal(pt, np.zeros(3)):
                world_pts.append(C)
            else:
                world_pts.append(C + R.T @ pt)
        
        # Define edges
        edges = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Front face
        ]
        
        return np.array(world_pts), edges

    @staticmethod
    def create_viewing_direction(pose, scale=0.2):
        """Create line representing camera's viewing direction"""
        # Camera center and direction
        R = pose[:3, :3]
        t = pose[:3, 3]
        C = -R.T @ t
        direction = R.T @ np.array([0, 0, 1])
        
        return C, C + direction * scale

    @staticmethod
    def create_fov_plane(pose, scale=0.2, fov_degrees=60, num_rays=8):
        """Create FOV visualization as rays"""
        R = pose[:3, :3]
        t = pose[:3, 3]
        C = -R.T @ t
        
        fov_rad = np.deg2rad(fov_degrees / 2)
        aspect_ratio = 4/3
        
        lines = []
        
        # Horizontal FOV rays
        for i in range(num_rays):
            angle = -fov_rad + i * (2 * fov_rad) / (num_rays - 1)
            ray = np.array([np.sin(angle), 0, np.cos(angle)])
            world_ray = R.T @ ray
            end_point = C + world_ray * scale
            lines.append((C, end_point))
        
        # Vertical FOV rays
        v_fov_rad = np.arctan(np.tan(fov_rad) / aspect_ratio)
        for i in range(num_rays):
            angle = -v_fov_rad + i * (2 * v_fov_rad) / (num_rays - 1)
            ray = np.array([0, np.sin(angle), np.cos(angle)])
            world_ray = R.T @ ray
            end_point = C + world_ray * scale
            lines.append((C, end_point))
        
        return lines


class SLAMVisualizer:
    """Main visualization class for SLAM processing"""
    
    def __init__(self, config_path, video_path):
        self.config_path = config_path
        self.video_path = video_path
        self.cam_visualizer = CameraVisualizer()
        self.slam = None
        self.cap = None
        self.frame_count = 0
        self.camera_frustums = []
        self.prev_frame = None
        self.current_frame = None
        self.matches = None
        
        # Matplotlib figure and axes
        self.fig = None
        self.axes = None
        self.setup_visualization()
    
    def setup_slam(self):
        """Initialize SLAM system"""
        # Load config
        config = load_config(self.config_path)
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

        # Create Visual SLAM system
        self.slam = VisualSLAM(
            camera_matrix=camera_matrix,
            dist_coeffs=distortion_coeffs
        )
        
        # Set extractor and matcher
        self.slam.extractor = orb
        self.slam.matcher = matcher
    
    def setup_visualization(self):
        """Setup matplotlib for visualization"""
        if SHOW_MATCHES:
            # Create 2x2 grid with video top left, matches top right, map bottom
            self.fig = plt.figure(figsize=(16, 12))
            self.axes = {
                'video': self.fig.add_subplot(2, 2, 1),
                'matches': self.fig.add_subplot(2, 2, 2),
                'map': self.fig.add_subplot(2, 1, 2, projection='3d')
            }
        else:
            # Create 1x2 grid with video left, map right
            self.fig = plt.figure(figsize=(16, 8))
            self.axes = {
                'video': self.fig.add_subplot(1, 2, 1),
                'map': self.fig.add_subplot(1, 2, 2, projection='3d')
            }
        
        plt.ion()  # Turn on interactive mode
        
        # Connect keyboard event
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        global MANUAL_MODE, process_frame, paused
        
        if event.key == ' ':  # Space key
            if MANUAL_MODE:
                process_frame = True
                print("Processing next frame...")
        elif event.key == 'q':
            print("Exiting...")
            plt.close('all')
        elif event.key == 'p':
            if not MANUAL_MODE:
                paused = not paused
                print("Playback", "paused" if paused else "resumed")
        elif event.key == 'm' or event.key == 'a':
            MANUAL_MODE = not MANUAL_MODE
            paused = False  # Reset pause when changing mode
            print(f"Switched to {'MANUAL' if MANUAL_MODE else 'AUTOMATIC'} mode")
    
    def update_video_view(self, frame, is_keyframe=False):
        """Update video frame visualization"""
        ax = self.axes['video']
        ax.clear()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw keypoints on keyframes
        if is_keyframe and hasattr(self.slam, 'prev_keyframe') and len(self.slam.prev_keyframe.get('keypoints', [])) > 0:
            kp = self.slam.prev_keyframe['keypoints'][:100]
            frame_rgb = cv2.drawKeypoints(frame_rgb, kp, None, color=(0, 255, 0))
        
        ax.imshow(frame_rgb)
        ax.set_title(f"Frame {self.frame_count}" + (" (Keyframe)" if is_keyframe else ""))
        ax.axis('off')
    
    def update_matches_view(self):
        """Update feature matches visualization"""
        if not SHOW_MATCHES or 'matches' not in self.axes:
            return
            
        ax = self.axes['matches']
        ax.clear()
        
        if self.prev_frame is not None and self.current_frame is not None and self.matches is not None:
            # Draw matches between consecutive frames
            matches_img = cv2.drawMatches(
                self.prev_frame, 
                self.slam.prev_frame.get('keypoints', []), 
                self.current_frame,
                self.slam.curr_frame.get('keypoints', []),
                self.matches, None,
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            matches_img = cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB)
            ax.imshow(matches_img)
            ax.set_title(f"Matches: {len(self.matches)}")
            ax.axis('off')
    
    def update_map_view(self):
        """Update 3D map visualization"""
        ax = self.axes['map']
        ax.clear()
        
        if len(self.slam.map_points) == 0:
            return
            
        points = np.array(self.slam.map_points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5, c='blue')
        
        # Plot camera trajectory
        if len(self.slam.keyframes) > 1:
            trajectory = np.array([kf['pose'][:3, 3] for kf in self.slam.keyframes])
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-', linewidth=2)
            ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='red', s=20)
        
        # Create and plot camera frustum for current pose
        vertices, edges = self.cam_visualizer.create_frustum(self.slam.curr_pose, scale=0.15)
        for edge in edges:
            ax.plot([vertices[edge[0], 0], vertices[edge[1], 0]],
                   [vertices[edge[0], 1], vertices[edge[1], 1]],
                   [vertices[edge[0], 2], vertices[edge[1], 2]],
                   'g-', linewidth=2)
        
        # Plot viewing direction
        start, end = self.cam_visualizer.create_viewing_direction(self.slam.curr_pose, scale=0.3)
        ax.quiver(start[0], start[1], start[2], 
                 end[0] - start[0], end[1] - start[1], end[2] - start[2],
                 color='yellow', arrow_length_ratio=0.2, linewidth=3)
        
        # Plot field of view rays
        fov_lines = self.cam_visualizer.create_fov_plane(self.slam.curr_pose, scale=0.25, num_rays=5)
        for start, end in fov_lines:
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   'g-', alpha=0.3, linewidth=0.8)
        
        # Plot camera frustums for keyframes
        for frustum in self.camera_frustums[-5:]:  # Show only the last 5 keyframe frustums
            frustum_vertices, frustum_edges = frustum
            for edge in frustum_edges:
                ax.plot([frustum_vertices[edge[0], 0], frustum_vertices[edge[1], 0]],
                        [frustum_vertices[edge[0], 1], frustum_vertices[edge[1], 1]],
                        [frustum_vertices[edge[0], 2], frustum_vertices[edge[1], 2]],
                        'g-', alpha=0.3, linewidth=1)
        
        # Add camera position label
        camera_pos = -self.slam.curr_pose[:3, :3].T @ self.slam.curr_pose[:3, 3]
        ax.text(camera_pos[0], camera_pos[1], camera_pos[2], 
               f"Camera", color='green', fontsize=8)
        
        ax.set_title(f"3D Map - {len(points)} points")
        
        # Set equal aspect ratio and adjust view
        self._adjust_map_view(ax, points)
        
        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Rotate view slowly
        ax.view_init(elev=30, azim=self.frame_count % 360)
    
    def _adjust_map_view(self, ax, points):
        """Helper to adjust map view proportions"""
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        z_range = np.max(points[:, 2]) - np.min(points[:, 2])
        max_range = max(x_range, y_range, z_range) / 2
        
        mid_x = np.mean(points[:, 0])
        mid_y = np.mean(points[:, 1])
        mid_z = np.mean(points[:, 2])
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    def update_visualization(self, is_keyframe):
        """Update all visualization components"""
        self.update_video_view(self.current_frame, is_keyframe)
        self.update_matches_view()
        self.update_map_view()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def show_final_map(self):
        """Display final 3D map visualization"""
        if not SHOW_FINAL_MAP or len(self.slam.map_points) == 0:
            return
            
        print("Showing final 3D map...")
        plt.ioff()  # Turn off interactive mode
        
        # Create new figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        points = np.array(self.slam.map_points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5, c='blue')
        
        # Plot camera trajectory
        if len(self.slam.keyframes) > 1:
            trajectory = np.array([kf['pose'][:3, 3] for kf in self.slam.keyframes])
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-', linewidth=2)
            ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='red', s=20)
        
        # Plot camera frustums for every 5th keyframe
        for i, kf in enumerate(self.slam.keyframes[::5]):
            vertices, edges = self.cam_visualizer.create_frustum(kf['pose'], scale=0.2)
            for edge in edges:
                ax.plot([vertices[edge[0], 0], vertices[edge[1], 0]],
                        [vertices[edge[0], 1], vertices[edge[1], 1]],
                        [vertices[edge[0], 2], vertices[edge[1], 2]],
                        'g-', linewidth=1)
            
            # Plot viewing direction
            start, end = self.cam_visualizer.create_viewing_direction(kf['pose'], scale=0.3)
            ax.quiver(start[0], start[1], start[2], 
                     end[0] - start[0], end[1] - start[1], end[2] - start[2],
                     color='yellow', arrow_length_ratio=0.2, linewidth=2)
            
            # Add keyframe label
            ax.text(vertices[0, 0], vertices[0, 1], vertices[0, 2], 
                   f"KF {i*5}", color='green', fontsize=8)
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Final 3D Map - {len(points)} points")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Map Points',
                  markerfacecolor='blue', markersize=5),
            Line2D([0], [0], color='red', lw=2, label='Camera Trajectory'),
            Line2D([0], [0], color='green', lw=2, label='Camera Frustum'),
            Line2D([0], [0], color='yellow', lw=2, label='Viewing Direction')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set equal aspect ratio
        self._adjust_map_view(ax, points)
        
        plt.tight_layout()
        plt.show()
    
    def process_video(self):
        """Main function to process video frames"""
        global process_frame, paused
        
        self.setup_slam()
        
        print(f"Processing video: {self.video_path}")
        if MANUAL_MODE:
            print("MANUAL MODE: Press SPACE to process frame, 'q' to exit, 'a' for automatic mode")
        else:
            print("AUTOMATIC MODE: Press 'q' to exit, 'p' to pause/resume, 'm' for manual mode")
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: cannot open {self.video_path}")
            return
        
        # Read first frame
        ret, self.current_frame = self.cap.read()
        if not ret:
            print("Error reading first frame")
            return
        
        # Show first frame
        self.axes['video'].imshow(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
        self.axes['video'].set_title("Frame 0 (Not processed)")
        self.axes['video'].axis('off')
        if SHOW_MATCHES and 'matches' in self.axes:
            self.axes['matches'].set_title("No matches yet")
            self.axes['matches'].axis('off')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        
        running = True
        while running:
            # Check if figure still exists
            if not plt.fignum_exists(self.fig.number):
                break
            
            # Determine if we process the current frame
            process_this_frame = False
            if MANUAL_MODE:
                if process_frame:
                    process_this_frame = True
                    process_frame = False
            else:
                if not paused:
                    process_this_frame = True
            
            if process_this_frame and self.cap.isOpened():
                # Save previous frame for matches visualization
                self.prev_frame = self.current_frame.copy()
                
                # Process frame
                is_keyframe = None
                if KEYFRAME_INTERVAL is not None:
                    is_keyframe = (self.frame_count % KEYFRAME_INTERVAL == 0)
                
                # Process frame with SLAM
                pose, new_points, is_kf = self.slam.process_frame(self.current_frame, is_keyframe)
                
                # Get matches for visualization
                if SHOW_MATCHES and hasattr(self.slam, 'prev_frame') and hasattr(self.slam, 'curr_frame'):
                    self.matches = self.slam.get_matches() if hasattr(self.slam, 'get_matches') else []
                
                # Update visualization
                if self.frame_count % VISUALIZE_FREQ == 0:
                    self.update_visualization(is_kf)
                
                # Update camera frustums for keyframes
                if is_kf:
                    vertices, edges = self.cam_visualizer.create_frustum(self.slam.curr_pose, scale=0.15)
                    self.camera_frustums.append((vertices, edges))
                
                print(f"Processed frame {self.frame_count}, keyframe: {is_kf}, new points: {len(new_points)}")
                self.frame_count += 1
                
                # Read next frame
                if self.cap.isOpened():
                    ret, next_frame = self.cap.read()
                    if ret:
                        self.current_frame = next_frame
                    else:
                        print("End of video")
                        if self.cap.isOpened():
                            self.cap.release()
                        running = False
                else:
                    print("Video closed")
                    running = False
            
            # Display current frame when paused or manual mode
            if (MANUAL_MODE and not process_this_frame) or paused:
                if self.current_frame is not None:
                    self.axes['video'].clear()
                    frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                    self.axes['video'].imshow(frame_rgb)
                    status = "paused" if paused else "current"
                    self.axes['video'].set_title(f"Frame {status} (Press SPACE to process)")
                    self.axes['video'].axis('off')
                    plt.tight_layout()
                    plt.draw()
            
            # Short pause to allow matplotlib to process events
            plt.pause(0.1)
        
        # Clean up
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {self.frame_count} frames, created {len(self.slam.keyframes)} keyframes")
        print(f"Final map contains {len(self.slam.map_points)} 3D points")
        
        # Show final map
        self.show_final_map()


def main():
    """Main function"""
    # Create visualizer and process video
    visualizer = SLAMVisualizer(CONFIG_PATH, VIDEO_PATH)
    visualizer.process_video()

if __name__ == '__main__':
    main()