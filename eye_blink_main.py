import cv2
import dlib
import numpy as np
import time
from imutils import face_utils
import math

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        return face, [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    return None, []

def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_face_orientation(landmarks):
    # Get the key points for face orientation
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left Mouth corner
        landmarks[54]   # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    return success, rotation_vector, translation_vector

def get_angle_from_rotation_vector(rotation_vector):
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, np.zeros((3, 1)))))[-1]
    pitch, yaw, roll = [math.radians(x) for x in angles]
    return pitch, yaw, roll

# Define constants for eye landmarks
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))
SHOULDER_POINTS = list(range(0, 17))

# Variables to track blink detection and time intervals
blink_detected = False
last_blink_time = time.time()
blink_intervals = []

# Variables to track face detection and prompt timer
face_detected = True
face_not_detected_start_time = None
prompt_interval = 10  # 20 minutes in seconds
last_prompt_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    face, landmarks = get_landmarks(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if landmarks:
        if face_detected is False:
            face_detected = True
            face_not_detected_start_time = None
            last_prompt_time = time.time()  # Reset the prompt timer

        left_eye = np.array([landmarks[i] for i in LEFT_EYE_POINTS])
        right_eye = np.array([landmarks[i] for i in RIGHT_EYE_POINTS])
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < 0.2:
            if blink_detected is False:
                blink_detected = True
                current_time = time.time()
                blink_interval = current_time - last_blink_time
                blink_intervals.append(blink_interval)
                last_blink_time = current_time
                print(f"Blink detected. Time since last blink: {blink_interval:.2f} seconds")
        else:
            blink_detected = False

        # Check face orientation
        success, rotation_vector, _ = get_face_orientation(landmarks)
        if success:
            _, yaw, _ = get_angle_from_rotation_vector(rotation_vector)
            yaw_degrees = math.degrees(yaw)
            if abs(yaw_degrees) > 30:
                print("looked away for more than 30 degrees")
                face_detected = False
                face_not_detected_start_time = time.time()

        # Check if it's time to show the prompt
        current_time = time.time()
        if current_time - last_prompt_time > prompt_interval:
            print("Time to take a break and look away!")
            last_prompt_time = current_time  # Reset the prompt timer

    else:
        if face_detected:
            face_detected = False
            face_not_detected_start_time = time.time()
        else:
            if face_not_detected_start_time and time.time() - face_not_detected_start_time < 20:
                print("Look away for longer")
                last_prompt_time = time.time()  # Reset the prompt timer if face not detected for 20 seconds
                face_not_detected_start_time = None

    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
