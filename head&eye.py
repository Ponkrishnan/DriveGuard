import cv2
import time
import mediapipe as mp
import numpy as np

# Load the Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize MediaPipe FaceMesh for head pose detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize video capture
cap = cv2.VideoCapture(0)
eye_closed_start_time = None
eye_closed_duration = 0
CLOSED_THRESHOLD = 3  # Time in seconds to trigger fatigue warning

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Convert frame to grayscale for eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # Eye closed detection
    if len(eyes) == 0:
        cv2.putText(frame, "The eyes are closed. Be alert!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if eye_closed_start_time is None:
            eye_closed_start_time = time.time()
        else:
            eye_closed_duration = time.time() - eye_closed_start_time
        if eye_closed_duration >= CLOSED_THRESHOLD:
            cv2.putText(frame, "You are fatigued!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        eye_closed_start_time = None
        eye_closed_duration = 0
        cv2.putText(frame, "Eyes are open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Head pose detection
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:  # Points for head orientation
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
                    if idx == 1:
                        nose_2d = (lm.x * frame.shape[1], lm.y * frame.shape[0])
                        nose_3d = (lm.x * frame.shape[1], lm.y * frame.shape[0], lm.z * 3000)
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * frame.shape[1]
            cam_matrix = np.array([[focal_length, 0, frame.shape[1] / 2],
                                   [0, focal_length, frame.shape[0] / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

            # Determine head position
            if y < -10:
                head_text = "Looking Left"
            elif y > 10:
                head_text = "Looking Right"
            elif x < -10:
                head_text = "Looking Down"
            elif x > 10:
                head_text = "Looking Up"
            else:
                head_text = "Looking Forward"

            cv2.putText(frame, head_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(frame, p1, p2, (255, 0, 0), 3)

    # Display the resulting frame
    cv2.imshow('Drowsiness and Head Pose Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
