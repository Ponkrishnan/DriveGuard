import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def calculate_mar(mouth):
    if len(mouth) < 10:
        return None
    A = dist.euclidean(mouth[4], mouth[5])  
    B = dist.euclidean(mouth[6], mouth[7])  
    C = dist.euclidean(mouth[8], mouth[9])  
    mar = (A + B) / (2.0 * C)
    return mar

MAR_THRESHOLD = 0.8 
CONSECUTIVE_FRAMES = 20  
frame_count = 0
yawn_count = 0
baseline_mars = []
CALIBRATION_FRAMES = 30  
ALERT_DURATION = 3  
alert_start_time = None  

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            indices = [61, 291, 78, 308, 13, 14, 15, 16, 11, 12]
            mouth_landmarks = [
                (face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0])
                for i in indices if i < len(face_landmarks.landmark)
            ]

            if len(mouth_landmarks) == 10:
                mar = calculate_mar(mouth_landmarks)

                if len(baseline_mars) < CALIBRATION_FRAMES:
                    if mar:
                        baseline_mars.append(mar)
                    MAR_THRESHOLD = np.mean(baseline_mars) + 0.4  
                else:
                    if mar and mar > MAR_THRESHOLD:
                        frame_count += 1 
                    else:
                        frame_count = 0  

                    if frame_count >= CONSECUTIVE_FRAMES:
                        yawn_count += 1
                        alert_start_time = time.time()  
                        frame_count = 0 

                for (x, y) in mouth_landmarks:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                cv2.putText(frame, f"MAR: {mar:.2f}" if mar else "MAR: N/A", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, f"Threshold: {MAR_THRESHOLD:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if alert_start_time and (time.time() - alert_start_time < ALERT_DURATION):
        cv2.putText(frame, "Yawning Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        alert_start_time = None  
    cv2.imshow("Yawn Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()