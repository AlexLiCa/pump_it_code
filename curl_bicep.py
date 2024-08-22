import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple


def calculate_angle(start: Tuple[float, float], middle: Tuple[float, float], end: Tuple[float, float]) -> float:
    """
    Calcula el ángulo entre tres puntos (start, middle, end).
    
    :param start: Coordenadas del primer punto (x, y).
    :param middle: Coordenadas del punto medio (x, y).
    :param end: Coordenadas del tercer punto (x, y).
    :return: Ángulo en grados.
    """
    a: np.ndarray = np.array(start)  # First
    b: np.ndarray = np.array(middle)  # Mid
    c: np.ndarray = np.array(end)    # End

    radians: float = np.arctan2(
        c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle: float = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def draw_landmarks_with_numbers(frame: np.ndarray, pose_landmarks) -> None:
    """
    Dibuja los números correspondientes a los landmarks en la imagen.
    
    :param frame: Frame de video en el que se dibujan los landmarks.
    :param pose_landmarks: Lista de landmarks normalizados detectados por Mediapipe.
    """
    h: int
    w: int
    h, w, _ = frame.shape
    for idx, landmark in enumerate(pose_landmarks.landmark):
        x: int = int(landmark.x * w)
        y: int = int(landmark.y * h)
        cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)


def extract_coordinates(landmarks, indices: List[int]) -> List[Tuple[float, float]]:
    """
    Extrae las coordenadas (x, y) de una lista de landmarks para los índices especificados.
    
    :param landmarks: Lista de landmarks normalizados detectados por Mediapipe.
    :param indices: Índices de los landmarks cuyas coordenadas se desean extraer.
    :return: Lista de coordenadas (x, y) correspondientes a los índices especificados.
    """
    return [(landmarks[i].x, landmarks[i].y) for i in indices]


# Inicializa los módulos de Mediapipe
mp_pose: mp.solutions.pose = mp.solutions.pose
mp_drawing: mp.solutions.drawing_utils = mp.solutions.drawing_utils
mp_drawing_styles: mp.solutions.drawing_styles = mp.solutions.drawing_styles

# Intenta abrir la cámara utilizando el backend AVFoundation
video: cv2.VideoCapture = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not video.isOpened():
    print("No se puede acceder a la cámara")
else:
    print("Cámara accedida correctamente")

counter: int = 0
stage: str = None

with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        model_complexity=2) as pose:

    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = video.read()

        if not ret:
            print("No se pudo leer el frame")
            break

        # Procesa el frame para detectar poses
        results = pose.process(frame)

        # Verifica si se han detectado pose landmarks antes de dibujar
        if results.pose_landmarks:
            # Dibuja los landmarks y conexiones
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Dibujar los números de los landmarks
            draw_landmarks_with_numbers(frame, results.pose_landmarks)

            # Extraer coordenadas de los landmarks
            start: Tuple[float, float]
            mid: Tuple[float, float]
            end: Tuple[float, float]
            start, mid, end = extract_coordinates(
                results.pose_landmarks.landmark, [12, 14, 16])

            # Calcular el ángulo entre los puntos
            angle: float = calculate_angle(start, mid, end)

            # Contar las repeticiones basadas en el ángulo
            if angle > 120:
                stage = "down"
            elif angle < 60 and stage == "down":
                counter += 1
                stage = "up"
                print(counter)

            # Mostrar el contador y el estado en el video
            cv2.putText(frame, f'Counter: {counter}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Stage: {stage}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Libera la cámara y destruye las ventanas
video.release()
cv2.destroyAllWindows()