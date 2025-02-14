import cv2
import mediapipe as mp
import numpy as np
import argparse
import os

class DancePoseDetector:
    def __init__(self, source='0'):
      """Initialize the Dance Pose Detector"""
      # Convert source to int if it's webcam
      self.source = 0 if source == '0' else source
      
      # Initialize MediaPipe Pose
      self.mp_pose = mp.solutions.pose
      self.pose = self.mp_pose.Pose(
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5
      )
      self.mp_drawing = mp.solutions.drawing_utils
      
      # Initialize video capture
      self.cap = self.create_capture()
      
      # Set resolution for webcam
      if self.source == 0:
          self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
          self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
          
      # Initialize position history
      self.position_history = {
          'left_shoulder': [],
          'right_shoulder': [],
          'timestamp': []
      }
      self.last_frame_time = None
      self.recent_dynamic_movement = False
      self.dynamic_timeout = 5.0  # seconds to remember dynamic movement
      
    def calculate_speed_and_acceleration(self, current_pos, position_history, frame_time):
      """Calculate speed and acceleration from position history"""
      if len(position_history) < 3:
          return 0, 0
      
      # Calculate speed (first derivative)
      speed = abs(current_pos - position_history[-1]) / frame_time
      
      # Calculate acceleration (second derivative)
      prev_speed = abs(position_history[-1] - position_history[-2]) / frame_time
      acceleration = abs(speed - prev_speed) / frame_time
      
      return speed, acceleration
      
    def create_capture(self):
      """Create video capture object from source"""
      if isinstance(self.source, str) and self.source != '0' and not os.path.exists(self.source):
          raise FileNotFoundError(f"Video file '{self.source}' not found")
      
      source = int(self.source) if self.source == '0' else self.source
      cap = cv2.VideoCapture(source)
      
      if not cap.isOpened():
          raise RuntimeError(f"Could not open {'webcam' if self.source == '0' else 'video file'}")
          
      return cap 

    def check_dynamic_movement(self, frame_time):
        """Check if there's been significant dynamic movement"""
        if len(self.position_history['left_shoulder']) < 3:
            return False
            
        # Calculate speed and acceleration for both shoulders
        left_speed, left_accel = self.calculate_speed_and_acceleration(
            self.position_history['left_shoulder'][-1],
            self.position_history['left_shoulder'],
            frame_time
        )
        right_speed, right_accel = self.calculate_speed_and_acceleration(
            self.position_history['right_shoulder'][-1],
            self.position_history['right_shoulder'],
            frame_time
        )
        
        # Print debug info
        print(f"Speed L: {left_speed:.1f}, R: {right_speed:.1f}")
        print(f"Accel L: {left_accel:.1f}, R: {right_accel:.1f}")
        
        # Adjust these thresholds as needed
        speed_threshold = 100  # Lowered from 200
        accel_threshold = 500  # Lowered from 1000
        
        return (max(left_speed, right_speed) > speed_threshold and 
                max(left_accel, right_accel) > accel_threshold)

    def update_position_history(self, landmarks, frame_shape, current_time):
        """Update position history for shoulders"""
        # Get shoulder positions in pixels
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_shape[0]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_shape[0]
        
        # Update history
        self.position_history['left_shoulder'].append(left_shoulder)
        self.position_history['right_shoulder'].append(right_shoulder)
        self.position_history['timestamp'].append(current_time)
        
        # Keep only last 3 frames
        if len(self.position_history['left_shoulder']) > 3:
            self.position_history['left_shoulder'].pop(0)
            self.position_history['right_shoulder'].pop(0)
            self.position_history['timestamp'].pop(0)

    def process_frame(self, frame):
        """Process a single frame and return the annotated frame"""
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        frame_time = current_time - self.last_frame_time if self.last_frame_time else 0.033
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe Pose
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            # Extract keypoints
            landmarks = results.pose_landmarks.landmark

            # Update position history
            self.update_position_history(landmarks, frame.shape, current_time)

            # Check for dynamic movement
            is_dynamic = self.check_dynamic_movement(frame_time)
            if is_dynamic:
                self.recent_dynamic_movement = True
                self.last_dynamic_time = current_time
            elif hasattr(self, 'last_dynamic_time'):
                # Clear dynamic flag if too much time has passed
                if current_time - self.last_dynamic_time > self.dynamic_timeout:
                    self.recent_dynamic_movement = False

            # Get shoulder points
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y

            # Calculate shoulder tilt in pixels
            shoulder_tilt = abs(left_shoulder - right_shoulder) * frame.shape[0]

            # Check if shoulders are tilted enough
            is_pose_position = shoulder_tilt > 30  # Increased threshold for more obvious tilt

            # Only consider it a dance pose if we got there through dynamic movement
            is_dance_pose = is_pose_position and self.recent_dynamic_movement

            # Display results
            if is_dance_pose:
                cv2.putText(frame, "DANCE POSE DETECTED!", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(frame, f"Shoulder tilt: {shoulder_tilt:.1f}px", (50, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if is_dynamic:
                    cv2.putText(frame, "ACTIVE MOVEMENT!", (50, 130),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                status = "MOVING..." if is_dynamic else "NOT A DANCE POSE"
                color = (0, 255, 255) if is_dynamic else (0, 0, 255)
                cv2.putText(frame, status, (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                cv2.putText(frame, f"Shoulder tilt: {shoulder_tilt:.1f}px", (50, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self.last_frame_time = current_time
        return frame

    def run(self):
        """Run the dance pose detection"""
        print("Starting pose detection... Press 'q' to quit")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("End of video file or error reading frame")
                break

            # Process the frame
            annotated_frame = self.process_frame(frame)

            # Show output
            cv2.imshow('Dance Pose Detection', annotated_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

        # Clean up
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Dynamic Dance Pose Detection')
    parser.add_argument('--source', default='0', help='Source (0 for webcam, or path to video file)')
    args = parser.parse_args()
    
    # Create and run detector
    detector = DancePoseDetector(args.source)
    detector.run()

if __name__ == '__main__':
    main()
