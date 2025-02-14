import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,  # Changed to False for video
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5  # Added for video
)
mp_drawing = mp.solutions.drawing_utils

# Initialize drawing specs with thicker lines
drawing_spec = mp_drawing.DrawingSpec(thickness=4, circle_radius=3, color=(0, 255, 0))
connection_spec = mp_drawing.DrawingSpec(thickness=4, color=(255, 255, 255))

# Read video
video_path = 'videos/untouchable.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Get video properties for output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
output_path = 'videos/output_dance.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Extract key joint coordinates
        landmarks = results.pose_landmarks.landmark
        
        # Extract coordinates for shoulders and head
        shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        head = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y]
        
        # Calculate shoulder tilt (vertical difference between shoulders)
        shoulder_tilt = abs(shoulder_l[1] - shoulder_r[1]) * frame.shape[0]  # Convert to pixels
        
        # Calculate head offset from shoulder center
        shoulder_center_x = (shoulder_l[0] + shoulder_r[0]) / 2
        head_offset = abs(head[0] - shoulder_center_x) * frame.shape[1]  # Convert to pixels
        
        # Draw pose landmarks with thicker lines
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=connection_spec
        )
        
        # Add measurements
        h, w, _ = frame.shape
        cv2.putText(frame, f"Shoulder Tilt: {int(shoulder_tilt)}px", 
                    (int(w*0.1), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Head Offset: {int(head_offset)}px", 
                    (int(w*0.6), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Check if pose is detected (significant tilt or head offset)
        if shoulder_tilt > 20 or head_offset > 30:
            cv2.putText(frame, "POSE DETECTED!", 
                        (int(w*0.3), int(h*0.9)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 3)

    # Write the frame to output video
    out.write(frame)
    
    # Display the frame
    cv2.imshow('Pose Detection', frame)
    
    # Break on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()
