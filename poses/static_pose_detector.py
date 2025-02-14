import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize drawing specs with thicker lines
drawing_spec = mp_drawing.DrawingSpec(thickness=4, circle_radius=3, color=(0, 255, 0))
connection_spec = mp_drawing.DrawingSpec(thickness=4, color=(255, 255, 255))

# Read image
image_path = 'path/to/your/image.jpg'  # Replace with your image path
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Could not load image from {image_path}")

# Convert image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

if results.pose_landmarks:
    # Extract key joint coordinates
    landmarks = results.pose_landmarks.landmark
    
    # Extract coordinates
    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
    wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
    
    # Calculate angles
    elbow_angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
    elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
    
    # Draw pose landmarks with thicker lines
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=connection_spec
    )
    
    # Add angle annotations
    h, w, _ = image.shape
    cv2.putText(image, f"L Elbow: {int(elbow_angle_l)}°", 
                (int(w*0.1), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"R Elbow: {int(elbow_angle_r)}°", 
                (int(w*0.6), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Check if pose is detected (both elbows bent < 150 degrees)
    if elbow_angle_l < 150 and elbow_angle_r < 150:
        cv2.putText(image, "POSE DETECTED!", 
                    (int(w*0.3), int(h*0.9)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 3)

# Display result
cv2.imshow("Pose Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Clean up
pose.close() 
