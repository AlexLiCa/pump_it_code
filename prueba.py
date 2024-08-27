import numpy as np
from typing import List, Tuple, Dict, Optional
import concurrent.futures


class Ejercicio:
    def __init__(self, nombre: str, angulos_objetivo: Dict[Tuple[int, int, int], Tuple[float, float]],
                 tolerancia: float = 10.0, angulos_adicionales: Optional[Dict[Tuple[int, int, int], float]] = None,
                 tolerancia_adicional: float = 5.0):
        """
        Clase para definir un ejercicio basado en múltiples ángulos y puntos de referencia.

        :param nombre: El nombre del ejercicio.
        :param angulos_objetivo: Un diccionario donde las claves son tuplas de puntos (start, mid, end) y los valores son tuplas con ángulos (inicial, final).
        :param tolerancia: La tolerancia permitida para considerar cada ángulo como válido.
        :param angulos_adicionales: Un diccionario donde las claves son tuplas de puntos (start, mid, end) y los valores son los ángulos objetivos adicionales.
        :param tolerancia_adicional: La tolerancia permitida para considerar los ángulos adicionales como válidos.
        """
        self.nombre = nombre
        self.angulos_objetivo = angulos_objetivo
        self.tolerancia = tolerancia
        self.stage = {puntos: None for puntos in angulos_objetivo}

        self.angulos_adicionales = angulos_adicionales or {}
        self.tolerancia_adicional = tolerancia_adicional

    def calcular_angulo(self, puntos: Tuple[int, int, int], landmarks) -> float:
        """
        Función para calcular el ángulo entre tres puntos dados los landmarks.

        :param puntos: Tupla de índices (start, mid, end) para calcular el ángulo.
        :param landmarks: Los puntos de referencia proporcionados por MediaPipe.
        :return: El ángulo calculado.
        """
        start, mid, end = self.extraer_coordenadas(landmarks, puntos)
        a = np.array(start)  # First
        b = np.array(mid)    # Mid
        c = np.array(end)    # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
            np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360.0 - angle

        return angle

    def verificar_angulo_principal(self, puntos: Tuple[int, int, int], angulo_inicial: float, angulo_final: float, landmarks) -> bool:
        """
        Verifica un ángulo principal específico y actualiza su estado en la secuencia.

        :param puntos: Tupla de puntos (start, mid, end).
        :param angulo_inicial: El ángulo inicial esperado.
        :param angulo_final: El ángulo final esperado.
        :param landmarks: Los puntos de referencia proporcionados por MediaPipe.
        :return: True si el ángulo cumple con los criterios, False en caso contrario.
        """
        angulo_actual = self.calcular_angulo(puntos, landmarks)

        if angulo_actual >= angulo_final - self.tolerancia:
            self.stage[puntos] = "down"
        elif angulo_actual <= angulo_inicial + self.tolerancia and self.stage[puntos] == "down":
            self.stage[puntos] = "up"
        else:
            return False

        return True

    def verificar_angulo_adicional(self, puntos: Tuple[int, int, int], angulo_objetivo: float, landmarks) -> bool:
        """
        Verifica un ángulo adicional específico.

        :param puntos: Tupla de puntos (start, mid, end).
        :param angulo_objetivo: El ángulo objetivo esperado.
        :param landmarks: Los puntos de referencia proporcionados por MediaPipe.
        :return: True si el ángulo está dentro de la tolerancia, False en caso contrario.
        """
        angulo_actual = self.calcular_angulo(puntos, landmarks)
    
        return angulo_objetivo - self.tolerancia_adicional <= angulo_actual <= angulo_objetivo + self.tolerancia_adicional

    def verificar_ejercicio(self, landmarks) -> bool:
        """
        Verifica si todos los ángulos necesarios para el ejercicio están dentro de los márgenes de tolerancia,
        siguiendo la secuencia de etapas. También verifica los ángulos adicionales si están definidos.

        :param landmarks: Los puntos de referencia proporcionados por MediaPipe.
        :return: True si se completa una repetición correctamente para todos los ángulos, False en caso contrario.
        """
        ejercicio_completado = True

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Verificar ángulos principales
            futuros_principales = [
                executor.submit(self.verificar_angulo_principal,
                                puntos, angulo_inicial, angulo_final, landmarks)
                for puntos, (angulo_inicial, angulo_final) in self.angulos_objetivo.items()
            ]

            # Verificar ángulos adicionales
            futuros_adicionales = [
                executor.submit(self.verificar_angulo_adicional,
                                puntos, angulo_objetivo, landmarks)
                for puntos, angulo_objetivo in self.angulos_adicionales.items()
            ]

            # Recolectar resultados
            for futuro in concurrent.futures.as_completed(futuros_principales + futuros_adicionales):
                if not futuro.result():
                    ejercicio_completado = False

        return ejercicio_completado and all(stage == "up" for stage in self.stage.values())

    @staticmethod
    def extraer_coordenadas(landmarks, indices: Tuple[int, int, int]) -> List[Tuple[float, float]]:
        """
        Función auxiliar para extraer las coordenadas de los landmarks dados los índices.

        :param landmarks: Los puntos de referencia proporcionados por MediaPipe.
        :param indices: Tupla de índices para extraer las coordenadas (start, mid, end).
        :return: Lista de tuplas con las coordenadas (x, y) de los puntos.
        """
        return [(landmarks[i].x, landmarks[i].y) for i in indices]
