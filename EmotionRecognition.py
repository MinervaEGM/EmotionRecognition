import cv2
import os
from deepface import DeepFace

print("Loading emotion recognition algorithm...")

# Loading pretrained model
modelPath = 'haarcascade_frontalface_default.xml'
if not os.path.exists(modelPath):
    print(f'Error: Model file doesn\'t exist {modelPath}. Train the model first.')
    exit()


# Initializing camera and Haar Cascade classifier
cap = cv2.VideoCapture(0)

#Creating classifier
classifier = cv2.CascadeClassifier(modelPath)

while True:
    ret, frame = cap.read()
    if not ret:
        print("It wasn\'t possible to capture video.")
        break
    #Capturing video in grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Capturing video in 2RGB
    rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    #Trimming the face region
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = rgb_frame[y:y + h, x:x + w]

        # Passing face to the detection model
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

        # Determining the emotion
        emotion = result[0]['dominant_emotion']
        print(f"Result: {emotion}")

        # Draw a rectangle enclosing the face and adding the emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #Window name
    cv2.imshow('Emotion recognition', frame)

    # Exit using Esc
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
