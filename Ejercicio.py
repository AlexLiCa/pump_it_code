import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Dict


# Función para calcular ángulos a partir de tres puntos


def calculate_angle(start: Tuple[float, float], middle: Tuple[float, float], end: Tuple[float, float]) -> float:
    a: np.ndarray = np.array(start)  # First
    b: np.ndarray = np.array(middle)  # Mid
    c: np.ndarray = np.array(end)    # End

    radians: float = np.arctan2(
        c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle: float = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


# Clase Ejercicio para manejar ejercicios y verificar múltiples ángulos
class Ejercicio:
    def __init__(self, nombre: str, angulos_objetivo: Dict[Tuple[int, int, int], Tuple[float, float]], tolerancia: float = 10.0):
        """
        Clase para definir un ejercicio basado en múltiples ángulos y puntos de referencia.

        :param nombre: El nombre del ejercicio.
        :param angulos_objetivo: Un diccionario donde las claves son tuplas de puntos (start, mid, end) y los valores son tuplas con ángulos (inicial, final).
        :param tolerancia: La tolerancia permitida para considerar cada ángulo como válido.
        """
        self.nombre = nombre
        self.angulos_objetivo = angulos_objetivo
        self.tolerancia = tolerancia
        # Etapa del ejercicio para cada conjunto de puntos
        self.stage = {puntos: None for puntos in angulos_objetivo}

    def _verificar_angulo(self, puntos: Tuple[int, int, int], landmarks) -> float:
        """
        Función auxiliar para calcular el ángulo.

        :param puntos: Tupla de índices (start, mid, end) para calcular el ángulo.
        :param landmarks: Los puntos de referencia proporcionados por MediaPipe.
        :return: El ángulo calculado.
        """
        start, mid, end = extract_coordinates(landmarks, puntos)
        return calculate_angle(start, mid, end)

    def verificar_ejercicio(self, landmarks) -> bool:
        """
        Verifica si todos los ángulos necesarios para el ejercicio están dentro de los márgenes de tolerancia,
        siguiendo la secuencia de etapas.

        :param landmarks: Los puntos de referencia proporcionados por MediaPipe.
        :return: True si se completa una repetición correctamente para todos los ángulos, False en caso contrario.
        """
        ejercicio_completado = True

        for puntos, (angulo_inicial, angulo_final) in self.angulos_objetivo.items():
            angulo_actual = self._verificar_angulo(puntos, landmarks)

            #print(angulo_actual)

            # Verificar si se ha alcanzado la posición inicial (down)
            if angulo_actual >= angulo_final - self.tolerancia:
                self.stage[puntos] = "down"
            # Verificar si se ha alcanzado la posición final (up) desde la posición "down"
            elif angulo_actual <= angulo_inicial + self.tolerancia and self.stage[puntos] == "down":
                self.stage[puntos] = "up"
            else:
                ejercicio_completado = False

        # Si todos los ángulos han pasado por las etapas "down" y "up", la repetición se cuenta
        return ejercicio_completado and all(stage == "up" for stage in self.stage.values())


# Función para dibujar landmarks con números
def draw_landmarks(frame: np.ndarray, pose_landmarks) -> None:
    h, w, _ = frame.shape
    for idx, landmark in enumerate(pose_landmarks.landmark):
        x: int = int(landmark.x * w)
        y: int = int(landmark.y * h)
        cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)


# Función para extraer coordenadas
def extract_coordinates(landmarks, indices: Tuple[int, int, int]) -> List[Tuple[float, float]]:
    return [(landmarks[i].x, landmarks[i].y) for i in indices]


# Función para agregar texto con fondo a la imagen
def put_multiline_text(image, text, position, font, font_scale, font_color, thickness, line_spacing=10):
    # Divide el texto en varias líneas si es demasiado largo
    lines = text.splitlines()
    x, y = position

    # Dibuja cada línea de texto sin fondo
    for i, line in enumerate(lines):
        y_line = y + \
            (i * (cv2.getTextSize(line, font, font_scale,
                                  thickness)[0][1] + line_spacing))
        cv2.putText(image, line, (x, y_line), font, font_scale,
                    font_color, thickness, cv2.LINE_AA)


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

# Ejercicio específico con múltiples ángulos
curl_bicep = Ejercicio(
    nombre="Curl de Barra",
    angulos_objetivo={
        (11, 13, 15): (60.0, 120.0),  # Ángulo para el brazo izquierdo
        (12, 14, 16): (60.0, 120.0),   # Ángulo para el brazo derecho
        #(14, 12, 24): (60.0, 40.0),   # Ángulo para el brazo-cadera derecha
        #(13, 11, 23): (60.0, 40.0)   # Ángulo para brazo-cadera iznquierda
    },
    tolerancia=20
)

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
            draw_landmarks(frame, results.pose_landmarks)

            # Verificar el ejercicio con múltiples ángulos
            if curl_bicep.verificar_ejercicio(results.pose_landmarks.landmark):
                counter += 1
                print(f'Repeticiones: {counter}')

        # Formatear la información de "Stages" para que se divida en múltiples líneas
        stages_text = '\n'.join(
            [f'{key}: {value}' for key, value in curl_bicep.stage.items()])

        # Mostrar el contador y el estado en el video con múltiples líneas para "Stages"
        put_multiline_text(frame, f'Counter: {counter}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        put_multiline_text(frame, f'Stages:\n{stages_text}', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Libera la cámara y destruye las ventanas
video.release()
cv2.destroyAllWindows()
