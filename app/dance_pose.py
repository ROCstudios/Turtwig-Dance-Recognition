import cv2
import mediapipe as mp
import numpy as np
import argparse
import os

def calculate_angle(a, b, c):
    """ Calculate angle between three points (a, b, c). """
    a = np.array(a)  # First joint
    b = np.array(b)  # Middle joint
    c = np.array(c)  # End joint

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle

def create_capture(source):
    """Create video capture object from source (0 for webcam or video path)"""
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Error: Video file '{source}' not found")
        exit(1)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open {'webcam' if source == 0 else 'video file'}")
        exit(1)
        
    return cap

def process_frame(frame, pose, mp_pose, mp_drawing):
    """Process a single frame and return the annotated frame"""
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe Pose
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract keypoints
        landmarks = results.pose_landmarks.landmark

        # Get required keypoints for dance pose
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

        # Calculate angles
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Calculate shoulder tilt (vertical difference between shoulders)
        shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) * frame.shape[0]  # Convert to pixels

        # Define dance pose criteria:
        # 1. One arm up (angle > 150 degrees)
        # 2. One arm down (angle < 60 degrees)
        # 3. Shoulders tilted (difference in y coordinates > 20 pixels)
        is_dance_pose = (
            ((left_arm_angle > 150 and right_arm_angle < 60) or
             (right_arm_angle > 150 and left_arm_angle < 60)) and
            shoulder_tilt > 20
        )

        # Display measurements and result
        if is_dance_pose:
            cv2.putText(frame, "DANCE POSE DETECTED!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, f"Shoulder tilt: {shoulder_tilt:.1f}px", (50, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NOT A DANCE POSE", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, f"Shoulder tilt: {shoulder_tilt:.1f}px", (50, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Dance Pose Detection')
    parser.add_argument('--source', default='0', help='Source (0 for webcam, or path to video file)')
    args = parser.parse_args()
    
    # Convert source to int if it's webcam
    source = 0 if args.source == '0' else args.source
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize video capture
    cap = create_capture(source)
    
    # Set resolution for webcam (optional)
    if source == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting pose detection... Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video file or error reading frame")
            break

        # Process the frame
        annotated_frame = process_frame(frame, pose, mp_pose, mp_drawing)

        # Show output
        cv2.imshow('Dance Pose Detection', annotated_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == '__main__':
    main()
