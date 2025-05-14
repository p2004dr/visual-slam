import os
import cv2
import yaml
import random
import glob
import numpy as np
from collections import deque

# Global parameters
VIDEO_PATH = 'data/video_gerard_1.mp4'
FRAME_OFFSET = 1        # Difference between frames
N_ORB_POINTS = 200      # Number of ORB keypoints to detect
OUTPUT_DIR = 'data/groundtruth_matches'
MIN_KP_DIST = 10        # Minimum pixel distance between keypoints

# Instructions text and font settings
INSTR_TEXT = 'g: save | c: clear | r: undo | +: zoom in | -: zoom out | wasd: pan | q: quit'
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_THICK = 1
BAR_HEIGHT = 20
BAR_COLOR = (50, 50, 50)
TEXT_COLOR = (200, 200, 200)

current_selection = None
history_stack = deque(maxlen=100)

COLOR_FULL_GREEN = (0, 255, 0)    # Verde completo
COLOR_FADED_GREEN = (0, 76, 0)    # Verde al 30%
COLOR_FULL_RED = (0, 0, 255)      # Rojo completo 
COLOR_FADED_RED = (0, 0, 76)      # Rojo al 30%

KP_SIZE_NORMAL = 5
KP_SIZE_SELECTED = 7
LINE_THICKNESS = 1

# Zoom and pan settings
offset_x = 0
offset_y = 0
zoom_scale = 1.0
zoom_step = 0.25
pan_step = 50  # pixels per arrow press

# Initialize ORB detector
detector = cv2.ORB_create(nfeatures=N_ORB_POINTS)

# State variables
keypoints1 = []
keypoints2 = []
matches = []           # list of (idx1, idx2)
sel1 = None            # index selected in img1
frame1 = None
frame2 = None
img_base = None         # image before zoom + bar
img_zoomed = None       # full zoomed image
win_w = None            # window display width
win_h = None            # window display height
img_display = None      # final cropped/padded image to show

# History stack for undo operations
history_stack = deque(maxlen=20)  # Limit history to prevent memory issues

# Utility: remove keypoints too close to each other
def filter_close_keypoints(kps, min_dist=MIN_KP_DIST):
    pts = []
    filtered = []
    for kp in kps:
        x, y = kp.pt
        if all((x - px)**2 + (y - py)**2 >= min_dist**2 for px, py in pts):
            filtered.append(kp)
            pts.append((x, y))
    return filtered

# Save current state for undo
def save_state():
    state = {
        'matches': matches.copy(),
        'current_selection': tuple(current_selection) if current_selection else None
    }
    history_stack.append(state)
    print(f"Estado guardado. Historial: {len(history_stack)}")

# Restore previous state (undo)
def undo():
    global matches, current_selection
    if not history_stack:
        print("No hay acciones para deshacer")
        return
    
    # Obtener estado anterior
    prev_state = history_stack.pop()
    
    # Restaurar valores
    matches = prev_state['matches'].copy()
    current_selection = tuple(prev_state['current_selection']) if prev_state['current_selection'] else None
    
    print(f"Deshecho. Restaurado estado con {len(matches)} matches")
    update_base_image()
    update_display()

# Read and prepare a random frame pair
def load_random_pair():
    global keypoints1, keypoints2, matches, sel1, frame1, frame2, zoom_scale, offset_x, offset_y, win_w, win_h
    # Clear history stack when loading a new pair
    history_stack.clear()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = random.randint(0, total - FRAME_OFFSET - 1)
    idx2 = idx + FRAME_OFFSET

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret1, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx2)
    ret2, frame2 = cap.read()
    cap.release()
    if not ret1 or not ret2:
        raise RuntimeError("Error reading video frames")

    # detect and filter keypoints
    raw1 = detector.detect(frame1, None)
    raw2 = detector.detect(frame2, None)
    keypoints1 = filter_close_keypoints(raw1)
    keypoints2 = filter_close_keypoints(raw2)
    matches.clear()
    sel1 = None

    # reset view
    zoom_scale = 1.0
    offset_x = 0
    offset_y = 0

    update_base_image()
    # set window dims on first load
    h, w = img_base.shape[:2]
    if win_h is None or win_w is None:
        win_h, win_w = h, w
        cv2.namedWindow('GT Annotator', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('GT Annotator', win_w, win_h)
    update_display()

# create base image with keypoints and match lines
def update_base_image():
    global img_base
    h1, w1 = frame1.shape[:2]
    disp1 = frame1.copy()
    disp2 = frame2.copy()
    
    # Dibujar keypoints imagen 1
    matched_indices1 = {m[0] for m in matches}
    for i, kp in enumerate(keypoints1):
        x, y = map(int, kp.pt)
        color = COLOR_FADED_GREEN if i in matched_indices1 else COLOR_FULL_GREEN
        size = KP_SIZE_NORMAL
        
        if current_selection and current_selection[0] == 1 and i == current_selection[1]:
            color = COLOR_FULL_RED
            size = KP_SIZE_SELECTED
            
        cv2.circle(disp1, (x, y), size, color, -1)
    
    # Dibujar keypoints imagen 2
    matched_indices2 = {m[1] for m in matches}
    for i, kp in enumerate(keypoints2):
        x, y = map(int, kp.pt)
        color = COLOR_FADED_GREEN if i in matched_indices2 else COLOR_FULL_GREEN
        size = KP_SIZE_NORMAL
        
        if current_selection and current_selection[0] == 2 and i == current_selection[1]:
            color = COLOR_FULL_RED
            size = KP_SIZE_SELECTED
            
        cv2.circle(disp2, (x, y), size, color, -1)
    
    # Dibujar líneas de matches con transparencia
    disp_combined = cv2.hconcat([disp1, disp2])
    for i1, i2 in matches:
        pt1 = tuple(map(int, keypoints1[i1].pt))
        pt2 = (int(keypoints2[i2].pt[0] + w1), int(keypoints2[i2].pt[1]))
        
        # Crear capa transparente
        overlay = disp_combined.copy()
        cv2.line(overlay, pt1, pt2, COLOR_FADED_RED, LINE_THICKNESS)
        disp_combined = cv2.addWeighted(overlay, 0.3, disp_combined, 0.7, 0)
    
    img_base = disp_combined
    
# update zoomed and display image with pan and zoom
def update_display():
    global img_zoomed, img_display, actual_offset_x, actual_offset_y
    
    # 1. Calcular área principal sin la barra
    main_win_h = win_h - BAR_HEIGHT if win_h > BAR_HEIGHT else win_h
    
    # 2. Aplicar zoom y pan solo a la imagen base
    h, w = img_base.shape[:2]
    img_zoomed = cv2.resize(img_base, None, fx=zoom_scale, fy=zoom_scale)
    zh, zw = img_zoomed.shape[:2]
    
    # 3. Calcular región visible (manteniendo aspect ratio)
    img_aspect = w / h
    win_aspect = win_w / main_win_h
    
    if win_aspect > img_aspect:
        fit_h = main_win_h
        fit_w = int(main_win_h * img_aspect)
        place_x = (win_w - fit_w) // 2
        place_y = 0
    else:
        fit_w = win_w
        fit_h = int(win_w / img_aspect)
        place_x = 0
        place_y = (main_win_h - fit_h) // 2
    
    # 4. Calcular offsets reales
    max_offset_x = max(0, zw - fit_w)
    max_offset_y = max(0, zh - fit_h)
    actual_offset_x = min(max_offset_x, max(0, int(offset_x)))
    actual_offset_y = min(max_offset_y, max(0, int(offset_y)))
    
    # 5. Crear canvas con barra de instrucciones
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    visible_img = img_zoomed[actual_offset_y:actual_offset_y+fit_h, actual_offset_x:actual_offset_x+fit_w]
    canvas[place_y:place_y+fit_h, place_x:place_x+fit_w] = visible_img
    
    # 6. Añadir barra de instrucciones
    bar = np.full((BAR_HEIGHT, win_w, 3), BAR_COLOR, dtype=np.uint8)
    cv2.putText(bar, INSTR_TEXT, (5, 15), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICK)
    canvas[main_win_h:main_win_h+BAR_HEIGHT, :] = bar
    
    cv2.imshow('GT Annotator', canvas)
# mouse click handler
def on_mouse(event, x, y, flags, param):
    global current_selection
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    
    # Calcular dimensiones principales sin la barra de instrucciones
    main_win_h = win_h - BAR_HEIGHT if win_h > BAR_HEIGHT else win_h
    h, w = img_base.shape[:2]
    img_aspect = w / h
    win_aspect = win_w / main_win_h
    
    # Calcular posición de la imagen en la ventana
    if win_aspect > img_aspect:
        fit_h = main_win_h
        fit_w = int(main_win_h * img_aspect)
        place_x = (win_w - fit_w) // 2
        place_y = 0
    else:
        fit_w = win_w
        fit_h = int(win_w / img_aspect)
        place_x = 0
        place_y = (main_win_h - fit_h) // 2
    
    # Ignorar clics fuera de la imagen o en la barra
    if (x < place_x or x >= place_x + fit_w or 
        y < place_y or y >= place_y + fit_h or 
        y >= main_win_h):
        return
    
    # Ajustar coordenadas relativas a la imagen
    rel_x = x - place_x
    rel_y = y - place_y
    
    # Convertir a coordenadas del zoomed image usando offsets reales
    zoomed_x = rel_x + actual_offset_x
    zoomed_y = rel_y + actual_offset_y
    
    # Convertir a coordenadas originales con redondeo
    orig_x = int(round(zoomed_x / zoom_scale))
    orig_y = int(round(zoomed_y / zoom_scale))
    
    print(f"Click at ({x}, {y}) -> image coords: ({orig_x}, {orig_y})")
    
    # Guardar estado actual para undo
    save_state()
    
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    # Obtener índices de keypoints ya emparejados
    matched_indices1 = {m[0] for m in matches}
    matched_indices2 = {m[1] for m in matches}
    
    if orig_x < w1 and orig_y < h1:  # Click en imagen 1
        idx = find_nearest_keypoint(keypoints1, (orig_x, orig_y))
        if idx in matched_indices1:
            print("Keypoint ya emparejado, selección ignorada")
            return
        
        # Lógica de selección bidireccional
        if current_selection and current_selection[0] == 2:
            matches.append((current_selection[1], idx))
            print(f"Match añadido: {current_selection[1]} -> {idx}")
            current_selection = None
        else:
            current_selection = (1, idx)
            print(f"Keypoint {idx} seleccionado en imagen 1")
            
    elif w1 <= orig_x < w1 + w2 and orig_y < h2:  # Click en imagen 2
        idx = find_nearest_keypoint(keypoints2, (orig_x - w1, orig_y))
        if idx in matched_indices2:
            print("Keypoint ya emparejado, selección ignorada")
            return
        
        if current_selection and current_selection[0] == 1:
            matches.append((current_selection[1], idx))
            print(f"Match añadido: {current_selection[1]} -> {idx}")
            current_selection = None
        else:
            current_selection = (2, idx)
            print(f"Keypoint {idx} seleccionado en imagen 2")
    
    update_base_image()
    update_display()

def draw_with_alpha(img, elements, color, alpha, thickness=2):
    overlay = img.copy()
    for elem in elements:
        if isinstance(elem, tuple):  # Para líneas
            cv2.line(overlay, elem[0], elem[1], color, thickness)
        else:  # Para keypoints
            cv2.drawKeypoints(overlay, [elem], overlay, color, 
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# find nearest keypoint index by location distance
def find_nearest_keypoint(kps, pt):
    best, best_dist = None, float('inf')
    for i, kp in enumerate(kps):
        d = (kp.pt[0]-pt[0])**2 + (kp.pt[1]-pt[1])**2
        if d < best_dist:
            best, best_dist = i, d
    return best

# save ground truth
def save_pair():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(OUTPUT_DIR, 'pair*'))
    idx = len(existing) + 1
    pair_folder = os.path.join(OUTPUT_DIR, f'pair{idx:02d}')
    os.makedirs(pair_folder)
    cv2.imwrite(os.path.join(pair_folder, 'img1.png'), frame1)
    cv2.imwrite(os.path.join(pair_folder, 'img2.png'), frame2)
    data = {
        'keypoints1': [[float(kp.pt[0]), float(kp.pt[1])] for kp in keypoints1],
        'keypoints2': [[float(kp.pt[0]), float(kp.pt[1])] for kp in keypoints2],
        'matches': [[int(i1), int(i2)] for i1, i2 in matches]
    }
    with open(os.path.join(pair_folder, 'gt.yaml'), 'w') as f:
        yaml.dump(data, f)
    print(f"Saved ground-truth to {pair_folder}")

# clear matches
def clear_matches():
    global matches, sel1
    # Save current state before clearing
    save_state()
    matches.clear()
    sel1 = None
    update_base_image()
    update_display()

# zoom and pan helpers
def zoom_in():
    global zoom_scale, offset_x, offset_y
    prev_scale = zoom_scale
    zoom_scale = min(5.0, zoom_scale + zoom_step)
    
    # Adjust offset to maintain center of image
    # Calculate current center
    center_x = int(offset_x) + (win_w // 2) // prev_scale
    center_y = int(offset_y) + (win_h // 2) // prev_scale
    
    # Calculate new offset to maintain center
    offset_x = center_x - (win_w // 2) / zoom_scale
    offset_y = center_y - (win_h // 2) / zoom_scale
    
    # Ensure offsets are not negative
    offset_x = max(0, offset_x)
    offset_y = max(0, offset_y)
    
    print(f"Zoom in: {zoom_scale:.2f}, offset_x={offset_x:.2f}, offset_y={offset_y:.2f}")
    update_display()

def zoom_out():
    global zoom_scale, offset_x, offset_y
    prev_scale = zoom_scale
    zoom_scale = max(1.0, zoom_scale - zoom_step)
    
    # Adjust offset to maintain center of image
    # Calculate current center
    center_x = int(offset_x) + (win_w // 2) // prev_scale
    center_y = int(offset_y) + (win_h // 2) // prev_scale
    
    # Calculate new offset to maintain center
    offset_x = center_x - (win_w // 2) / zoom_scale
    offset_y = center_y - (win_h // 2) / zoom_scale
    
    # Ensure offsets are not negative
    offset_x = max(0, offset_x)
    offset_y = max(0, offset_y)
    
    print(f"Zoom out: {zoom_scale:.2f}, offset_x={offset_x:.2f}, offset_y={offset_y:.2f}")
    update_display()

def pan(direction):
    global offset_x, offset_y
    
    # Calculate movement amount according to current zoom
    # Higher zoom requires more precision
    move_amount = pan_step / zoom_scale
    
    original_x, original_y = offset_x, offset_y
    
    if direction == 'left':
        offset_x = max(0, offset_x - move_amount)
    elif direction == 'right':
        offset_x += move_amount
    elif direction == 'up':
        offset_y = max(0, offset_y - move_amount)
    elif direction == 'down':
        offset_y += move_amount
    
    print(f"Pan {direction}: from ({original_x:.2f},{original_y:.2f}) to ({offset_x:.2f},{offset_y:.2f}), moved by {move_amount:.2f}")
    update_display()

# main loop
def main():
    global win_w, win_h
    load_random_pair()
    cv2.setMouseCallback('GT Annotator', on_mouse)
    print("Keys: s=save, c=clear, r=undo, +=zoom in, -=zoom out, arrows: pan, q=quit")
    
    # Arrow key codes - fixed for cross-platform compatibility
    UP_KEY = 0
    DOWN_KEY = 1
    LEFT_KEY = 2 
    RIGHT_KEY = 3
    
    cv2.setWindowProperty('GT Annotator', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    
    while True:
        key = cv2.waitKey(30) & 0xFF  # Use & 0xFF for compatibility
        
        # Check if window size has changed
        try:
            current_rect = cv2.getWindowImageRect('GT Annotator')
            current_w, current_h = current_rect[2], current_rect[3]
            if current_w != win_w or current_h != win_h:
                print(f"Window resized: {win_w}x{win_h} -> {current_w}x{current_h}")
                win_w, win_h = current_w, current_h
                update_display()
        except Exception as e:
            print(f"Error getting window size: {e}")
        
        if key == -1:
            continue
        if key == ord('g'):
            save_pair()
        elif key == ord('c'):
            clear_matches()
        elif key == ord('r'):  # Changed to undo function
            undo()
        elif key in [ord('+'), ord('=')]:
            zoom_in()
        elif key == ord('-'):
            zoom_out()
        # Fixed arrow key handling
        elif key == ord('w'):
            pan('up')
        elif key == ord('s'):
            pan('down')
        elif key == ord('a'):
            pan('left')
        elif key == ord('d'):
            pan('right')
        elif key == ord('q'):
            break
            
        # Print key code for debug (optional)
        if key != -1 and key != 255:
            print(f"Key pressed: {key}")
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()