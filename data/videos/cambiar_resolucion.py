import cv2, os, sys

def reducir_resolucion(video_entrada, video_salida):
    video_entrada = os.path.abspath(video_entrada)
    video_salida  = os.path.abspath(video_salida)
    print(f"Abrir vídeo: {video_entrada}")
    cap = cv2.VideoCapture(video_entrada)
    if not cap.isOpened():
        print(f"Error: no se puede abrir {video_entrada}", file=sys.stderr); return
    ancho  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  // 2)
    alto   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    codec  = cv2.VideoWriter_fourcc(*'XVID')  # más compatible
    out    = cv2.VideoWriter(video_salida, codec, fps, (ancho, alto))
    if not out.isOpened():
        print("Error: VideoWriter no inicializado", file=sys.stderr); cap.release(); return
    print(f"Guardando copia en: {video_salida} ({ancho}x{alto} @ {fps} FPS)")

    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(cv2.resize(frame, (ancho, alto)))

    cap.release(); out.release()
    print("Proceso completado.")

# Uso
reducir_resolucion("video_gerard_1_h.mp4", "video_gerard_1_l.mp4")
