import cv2
import dlib
import numpy as np
import time
from imutils import face_utils

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(frame):
    #change face internally to gray scale (not to user)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    return []

# while True:
#     ret, frame = cap.read()
#     landmarks = get_landmarks(frame)
#     for (x, y) in landmarks:
#         cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#     cv2.imshow('Webcam', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

while True:
    ret, frame = cap.read()
    landmarks = get_landmarks(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    if landmarks:
      left_eye = np.array([landmarks[i] for i in LEFT_EYE_POINTS])
      right_eye = np.array([landmarks[i] for i in RIGHT_EYE_POINTS])
      left_ear = calculate_ear(left_eye)
      right_ear = calculate_ear(right_eye)
      ear = (left_ear + right_ear) / 2.0
      if ear < 0.15:
        print("Blink detected")
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

blink_count = 0
blink_start_time = time.time()
blink_interval = []

while True:
    ret, frame = cap.read()
    landmarks = get_landmarks(frame)
    if landmarks:
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        if ear < 0.2:
            if not blink_detected:
                blink_detected = True
                blink_count += 1
                blink_interval.append(time.time() - blink_start_time)
                blink_start_time = time.time()
        else:
            blink_detected = False
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

reminder_interval = 1200  # 20 minutes in seconds
last_reminder_time = time.time()

while True:
    ret, frame = cap.read()
    current_time = time.time()
    if current_time - last_reminder_time > reminder_interval:
        cv2.putText(frame, "Take a break!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        last_reminder_time = current_time
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = 0  # simple brightness control
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted

while True:
    ret, frame = cap.read()
    frame = adjust_brightness_contrast(frame, brightness=30, contrast=1.2)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

