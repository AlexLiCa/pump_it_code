# dibujar_mediapipe.py

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple  # Importación añadida


class DibujarMediaPipe:
    def __init__(self, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_color=(255, 0, 0), thickness=1, line_spacing=10):
        """
        Clase para manejar operaciones comunes relacionadas con MediaPipe y OpenCV.

        :param font: El tipo de fuente utilizada para el texto (por defecto, cv2.FONT_HERSHEY_SIMPLEX).
        :param font_scale: El tamaño de la fuente (por defecto, 0.5).
        :param font_color: El color de la fuente, en formato BGR (por defecto, azul).
        :param thickness: El grosor de las letras (por defecto, 1).
        :param line_spacing: El espaciado entre líneas en píxeles (por defecto, 10 píxeles).
        """
        self.font = font
        self.font_scale = font_scale
        self.font_color = font_color
        self.thickness = thickness
        self.line_spacing = line_spacing

    def draw_landmarks(self, frame: np.ndarray, pose_landmarks) -> None:
        """
        Dibuja los landmarks de pose en un frame dado, numerándolos de acuerdo con su índice.

        :param frame: El frame de la imagen donde se dibujarán los landmarks.
        :param pose_landmarks: Los landmarks de la pose detectada, proporcionados por MediaPipe.
        """
        h, w, _ = frame.shape
        for idx, landmark in enumerate(pose_landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.putText(frame, str(idx), (x, y), self.font, self.font_scale,
                        self.font_color, self.thickness, cv2.LINE_AA)

    def put_multiline_text(self, image: np.ndarray, text: str, position: Tuple[int, int]) -> None:
        """
        Dibuja texto multilínea en una imagen, posicionando cada línea una debajo de la otra.

        :param image: La imagen donde se dibujará el texto.
        :param text: El texto que se va a dibujar, donde cada línea se separa con '\n'.
        :param position: La posición inicial (x, y) donde se comenzará a dibujar el texto.
        """
        lines = text.splitlines()
        x, y = position

        for i, line in enumerate(lines):
            y_line = y + (i * (cv2.getTextSize(line, self.font,
                          self.font_scale, self.thickness)[0][1] + self.line_spacing))
            cv2.putText(image, line, (x, y_line), self.font, self.font_scale,
                        self.font_color, self.thickness, cv2.LINE_AA)



