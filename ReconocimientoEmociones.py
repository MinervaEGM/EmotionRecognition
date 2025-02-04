import cv2
import os
from deepface import DeepFace

print("Cargando modelo de reconocimiento de emociones...")

# Cargar el modelo pre entrenado
modeloPath = 'haarcascade_frontalface_default.xml'
if not os.path.exists(modeloPath):
    print(f'Error: El archivo del modelo {modeloPath} no existe. Entrena el modelo primero.')
    exit()


# Inicializar la cámara y el clasificador Haar Cascade
cap = cv2.VideoCapture(0)

#Se crea el clasificador
clasificador = cv2.CascadeClassifier(modeloPath)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el video.")
        break
    #Se captura el vídeo en escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Se captura el vídeo en RGBP
    rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    #Se recorta la región del rostro para analizar 
    caras = clasificador.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in caras:
        rostro = rgb_frame[y:y + h, x:x + w]

        # Se pasa el rostro al modelo de detección
        result = DeepFace.analyze(rostro, actions=['emotion'], enforce_detection=False)

        # Se determina la emoción
        emotion = result[0]['dominant_emotion']
        print(f"Resultado: {emotion}")

        # Se dibuja el rectángulo alrededor del rostro con la etiqueta de la emoción identificada
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #Nombre de la ventana
    cv2.imshow('Reconocimiento de emociones', frame)

    # Salir con la tecla ESC
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
