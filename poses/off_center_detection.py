import cv2
import mediapipe as mp
import numpy as np

def calculate_deviation(head, center_hips):
    """Calculate deviation of the head from the vertical centerline passing through the hips."""
    return abs(head[0] - center_hips[0])

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Extract key joint coordinates
        head = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y]
        hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        
        # Compute center of hips (midpoint between left and right hips)
        center_hips_x = (hip_l[0] + hip_r[0]) / 2
        center_hips = [center_hips_x, (hip_l[1] + hip_r[1]) / 2]
        
        # Compute head deviation from centerline
        head_deviation = calculate_deviation(head, center_hips)
        
        # Check if body is off-center (threshold can be adjusted as needed)
        if head_deviation >= 0.05:  # Threshold for detecting off-center movement
            cv2.putText(image, "OFF-CENTER", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow("Off-Center Detector", image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
