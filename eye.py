import cv2
import time


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


cap = cv2.VideoCapture(0)
eye_closed_start_time = None
eye_closed_duration = 0
CLOSED_THRESHOLD = 3  

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    
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

    
    cv2.imshow('Eye Detection', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()