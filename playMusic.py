import cv2
import numpy as np
import pygame

# 1. Inicializar el reproductor de música
pygame.mixer.init()
# Cargar la pista de audio (asegúrate de tener un archivo con este nombre)
pygame.mixer.music.load('music.mp3') 

# Iniciar la webcam
cap = cv2.VideoCapture(0)

# Leer el primer frame para establecer el fondo inicial
ret, frame_prev = cap.read()
# Espejar el frame para que la interacción sea natural (como verse en un espejo)
frame_prev = cv2.flip(frame_prev, 1)
gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
# CONVOLUCIÓN: Suavizar la imagen inicial
gray_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)

# Coordenadas de nuestros "Botones Virtuales" (x1, y1, x2, y2)
boton_play = (50, 50, 200, 300)   # Lado Izquierdo
boton_stop = (1750, 50,1900 , 300)  # Lado Derecho

# Variables de control
cooldown = 0
reproduciendo = False

print("Pasa la mano por los recuadros para controlar la música. Presiona 'q' para salir.")

while True:
    ret, frame_actual = cap.read()
    if not ret: 
        break
    
    # Espejar la cámara
    frame_actual = cv2.flip(frame_actual, 1)
    
    # Preprocesamiento del frame actual
    gray_actual = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
    gray_actual = cv2.GaussianBlur(gray_actual, (21, 21), 0)

    # 2. DERIVADA TEMPORAL Y UMBRALIZACIÓN
    diferencia = cv2.absdiff(gray_prev, gray_actual)
    _, umbral = cv2.threshold(diferencia, 25, 255, cv2.THRESH_BINARY)

    # 3. APERTURA MORFOLÓGICA 
    kernel = np.ones((5, 5), np.uint8)
    # Primero EROSIÓN: Elimina el ruido visual y luces falsas
    umbral = cv2.erode(umbral, kernel, iterations=1)
    # Luego DILATACIÓN: Recupera y hace más denso el bloque de tu mano
    umbral = cv2.dilate(umbral, kernel, iterations=3)

    # ZONAS DE INTERÉS 
    # Recortar la matriz binaria en las coordenadas de los botones
    zona_play = umbral[boton_play[1]:boton_play[3], boton_play[0]:boton_play[2]]
    zona_stop = umbral[boton_stop[1]:boton_stop[3], boton_stop[0]:boton_stop[2]]

    # Contar píxeles blancos (movimiento real filtrado) en cada zona
    movimiento_play = cv2.countNonZero(zona_play)
    movimiento_stop = cv2.countNonZero(zona_stop)

    # Lógica del botón PLAY
    if movimiento_play > 1500 and cooldown <= 0:
        print("¡Gesto PLAY detectado!")
        if not reproduciendo:
            pygame.mixer.music.play() # Descomenta esto para que suene
            reproduciendo = True
        cooldown = 30 # Bloquear entradas por ~1 segundo

    # Lógica del botón STOP
    elif movimiento_stop > 1500 and cooldown <= 0:
        print("¡Gesto STOP detectado!")
        if reproduciendo:
            pygame.mixer.music.stop() # Descomenta esto para detener el audio
            reproduciendo = False
        cooldown = 30

    # Reducir el cooldown frame a frame
    if cooldown > 0:
        cooldown -= 1

    # --- INTERFAZ VISUAL ---
    # Cambiar colores según el estado de la música
    color_play = (0, 255, 0) if reproduciendo else (150, 255, 150)
    color_stop = (0, 0, 255) if not reproduciendo else (150, 150, 255)


    # Actualizar frame previo para el siguiente ciclo
    gray_prev = gray_actual.copy()

    # Mostrar ventanas
    cv2.imshow("Control de Musica - Camara", frame_actual)
    # cv2.imshow("Filtro Morfologico", umbral) # Descomenta para ver cómo actúan la erosión y dilatación

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
pygame.quit()