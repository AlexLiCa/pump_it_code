import cv2
import mediapipe as mp
import numpy as np

# Inicializa los módulos de Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define la ruta del video
video_path = "videos_ejercicios/PullUp.mov"
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("No se puede acceder al video")
    exit()

with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        model_complexity=2) as pose:

    while True:
        ret, frame = video.read()

        w,h,c = frame.shape

        frameImg = np.zeros([w,h,c])

        if not ret:
            print("Fin del video o error al leer el frame")
            break

        # Procesa el frame para detectar poses
        results = pose.process(frame)

        # Verifica si se han detectado pose landmarks antes de dibujar
        if results.pose_landmarks:
            # Dibuja los landmarks y conexiones
            mp_drawing.draw_landmarks(
                #frame,
                frameImg,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        #cv2.imshow("Frame", frame)
        cv2.imshow("Frame", frameImg )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Libera el video y destruye las ventanas
video.release()
cv2.destroyAllWindows()
