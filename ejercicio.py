# ejercicio.py

import numpy as np
from typing import List, Tuple, Dict


# Clase Ejercicio para manejar ejercicios y verificar ángulos
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
        self.stage = {puntos: None for puntos in angulos_objetivo}

    def calcular_angulo(self, puntos: Tuple[int, int, int], landmarks) -> float:
        """
        Función para calcular el ángulo entre tres puntos dados los landmarks.

        :param puntos: Tupla de índices (start, mid, end) para calcular el ángulo.
        :param landmarks: Los puntos de referencia proporcionados por MediaPipe.
        :return: El ángulo calculado.
        """
        start, mid, end = self.extract_coordinates(landmarks, puntos)
        a = np.array(start)  # First
        b = np.array(mid)    # Mid
        c = np.array(end)    # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
            np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360.0 - angle

        return angle

    def verificar_ejercicio(self, landmarks) -> bool:
        """
        Verifica si todos los ángulos necesarios para el ejercicio están dentro de los márgenes de tolerancia,
        siguiendo la secuencia de etapas.

        :param landmarks: Los puntos de referencia proporcionados por MediaPipe.
        :return: True si se completa una repetición correctamente para todos los ángulos, False en caso contrario.
        """
        ejercicio_completado = True

        for puntos, (angulo_inicial, angulo_final) in self.angulos_objetivo.items():
            angulo_actual = self.calcular_angulo(puntos, landmarks)

            if angulo_actual >= angulo_final - self.tolerancia:
                self.stage[puntos] = "down"
            elif angulo_actual <= angulo_inicial + self.tolerancia and self.stage[puntos] == "down":
                self.stage[puntos] = "up"
            else:
                ejercicio_completado = False

        return ejercicio_completado and all(stage == "up" for stage in self.stage.values())

    @staticmethod
    def extract_coordinates(landmarks, indices: Tuple[int, int, int]) -> List[Tuple[float, float]]:
        """
        Función auxiliar para extraer las coordenadas de los landmarks dados los índices.

        :param landmarks: Los puntos de referencia proporcionados por MediaPipe.
        :param indices: Tupla de índices para extraer las coordenadas (start, mid, end).
        :return: Lista de tuplas con las coordenadas (x, y) de los puntos.
        """
        return [(landmarks[i].x, landmarks[i].y) for i in indices]
