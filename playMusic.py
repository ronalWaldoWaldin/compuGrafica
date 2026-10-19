# SC3321 Vision Computacional : 584431 / 696641 : Xavier Maximiliano López Ibarra / Ronal Waldo Mora Aybar
"""Control de Musica por Gestos.

Usage:
  control_gestos.py <ruta_audio> [--camara=<idx>] [--sensibilidad=<val>]
  control_gestos.py -h | --help

Options:
  -h --help             Muestra esta ayuda.
  <ruta_audio>          Ruta al archivo de música (ej. cancion.mp3).
  --camara=<idx>        Índice de la cámara web [default: 0].
  --sensibilidad=<val>  Píxeles blancos mínimos para activar el botón [default: 1500].
"""

import cv2
import numpy as np
import pygame
from docopt import docopt

def main():
    # 1. Parsear los argumentos de la terminal con docopt
    args = docopt(__doc__)
    ruta_audio = args['<ruta_audio>']
    idx_camara = int(args['--camara'])
    umbral_movimiento = int(args['--sensibilidad'])

    # 2. Inicializar reproductor y cargar el audio especificado
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(ruta_audio)
        print(f"Audio cargado exitosamente: {ruta_audio}")
    except pygame.error as e:
        print(f"Error al cargar el audio: {e}")
        return

    # 3. Iniciar la webcam
    cap = cv2.VideoCapture(idx_camara)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    ret, frame_prev = cap.read()
    frame_prev = cv2.flip(frame_prev, 1)
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)

    # Coordenadas de los botones (x1, y1, x2, y2)
    boton_play = (50, 50, 200, 300)
    boton_stop = (1750, 50,1900 , 300)

    cooldown = 0
    reproduciendo = False

    print("\nControles iniciados. Pasa la mano por los recuadros. Presiona 'q' en la ventana para salir.")

    while True:
        ret, frame_actual = cap.read()
        if not ret: break
        
        frame_actual = cv2.flip(frame_actual, 1)
        gray_actual = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
        gray_actual = cv2.GaussianBlur(gray_actual, (21, 21), 0)

        # Derivada temporal y umbralización
        diferencia = cv2.absdiff(gray_prev, gray_actual)
        _, umbral = cv2.threshold(diferencia, 25, 255, cv2.THRESH_BINARY)

        # Apertura morfológica (Erosión + Dilatación)
        kernel = np.ones((5, 5), np.uint8)
        umbral = cv2.erode(umbral, kernel, iterations=1)
        umbral = cv2.dilate(umbral, kernel, iterations=3)

        # Zonas de interés
        zona_play = umbral[boton_play[1]:boton_play[3], boton_play[0]:boton_play[2]]
        zona_stop = umbral[boton_stop[1]:boton_stop[3], boton_stop[0]:boton_stop[2]]

        movimiento_play = cv2.countNonZero(zona_play)
        movimiento_stop = cv2.countNonZero(zona_stop)

        # Lógica usando la variable de docopt
        if movimiento_play > umbral_movimiento and cooldown <= 0:
            print("¡Gesto PLAY detectado!")
            if not reproduciendo:
                pygame.mixer.music.play()
                reproduciendo = True
            cooldown = 30

        elif movimiento_stop > umbral_movimiento and cooldown <= 0:
            print("¡Gesto STOP detectado!")
            if reproduciendo:
                pygame.mixer.music.stop()
                reproduciendo = False
            cooldown = 30

        if cooldown > 0:
            cooldown -= 1

      
        gray_prev = gray_actual.copy()
        cv2.imshow("Control de Musica - Camara", frame_actual)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == '__main__':
    main()