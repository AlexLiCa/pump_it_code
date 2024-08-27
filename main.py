# main.py

import cv2
import mediapipe as mp
import numpy as np
from ejercicio import Ejercicio
from dibujar_mediapipe import DibujarMediaPipe


# TODO: Considerar si las rodillas son importantes para el futuro, en caso de serlas considerar otro metodo
def esta_de_frente(landmarks) -> bool:
    """
    Verifica si el cuerpo del usuario está de frente basándose en las distancias 3D de los landmarks del torso y las piernas.

    :param landmarks: Los puntos de referencia proporcionados por MediaPipe.
    :return: True si el cuerpo está de frente, False si está de lado.
    """
    tolerancia = 0.12

    left_shoulder_z = landmarks[11].z
    right_shoulder_z = landmarks[12].z
    left_hip_z = landmarks[23].z
    right_hip_z = landmarks[24].z
    left_knee_z = landmarks[25].z
    right_knee_z = landmarks[26].z

    # Calcula la diferencia en profundidad entre los lados izquierdo y derecho
    shoulder_diff = abs(left_shoulder_z - right_shoulder_z)
    hip_diff = abs(left_hip_z - right_hip_z)
    knee_diff = abs(left_knee_z - right_knee_z)

    # Si las diferencias en profundidad son pequeñas, el cuerpo está de frente
    if shoulder_diff < tolerancia and hip_diff < tolerancia and knee_diff < tolerancia:  
        return True
    return False



# Inicializa los módulos de MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Intenta abrir la cámara utilizando el backend AVFoundation
video = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not video.isOpened():
    print("No se puede acceder a la cámara")
else:
    print("Cámara accedida correctamente")

counter = 0

# Ejercicio específico con múltiples ángulos
curl_bicep = Ejercicio(
    nombre="Curl de Barra",
    angulos_objetivo={
        (11, 13, 15): (60.0, 120.0),  # Ángulo para el brazo izquierdo
        (12, 14, 16): (60.0, 120.0),   # Ángulo para el brazo derecho
    },
    tolerancia=10
)

# Instancia de la clase DibujarMediaPipe
dibujar_mediapipe = DibujarMediaPipe(
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1,
    font_color=(0, 255, 0),
    thickness=2
)

with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        model_complexity=2) as pose:

    while True:
        ret, frame = video.read()

        if not ret:
            print("No se pudo leer el frame")
            break

        # Procesa el frame para detectar poses
        results = pose.process(frame)

        # Verifica si se han detectado pose landmarks antes de continuar
        if results.pose_landmarks:
            # Verifica si el cuerpo del usuario está de frente
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            #print(esta_de_frente(results.pose_landmarks.landmark))

            if esta_de_frente(results.pose_landmarks.landmark):


                # Dibujar los números de los landmarks
                dibujar_mediapipe.draw_landmarks(frame, results.pose_landmarks)

                # Verificar el ejercicio con múltiples ángulos
                if curl_bicep.verificar_ejercicio(results.pose_landmarks.landmark):
                    counter += 1
                    print(f'Repeticiones: {counter}')

                # Mostrar mensaje en la imagen indicando que el usuario está de frente
                dibujar_mediapipe.put_multiline_text(
                    frame, "Cuerpo de frente", (10, 30))
            else:
                # Mostrar mensaje de que el usuario debe estar de frente
                dibujar_mediapipe.put_multiline_text(
                    frame, "Por favor, coloquese de frente", (10, 30))

        stages_text = '\n'.join(
            [f'{key}: {value}' for key, value in curl_bicep.stage.items()])

        # Mostrar el contador y el estado en el video con múltiples líneas para "Stages"
        dibujar_mediapipe.put_multiline_text(
            frame, f'Counter: {counter}', (10, 60))
        dibujar_mediapipe.put_multiline_text(
            frame, f'Stages:\n{stages_text}', (10, 100))

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video.release()
cv2.destroyAllWindows()
