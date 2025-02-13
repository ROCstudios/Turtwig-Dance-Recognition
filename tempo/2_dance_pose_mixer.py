import cv2
import pygame
import time
import librosa
import numpy as np
import mediapipe as mp

class DancePoseDetector:
    def __init__(self, audio_file):
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Audio setup
        self.y, self.sr = librosa.load(audio_file, sr=44100)
        self.tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        
        # Calculate subdivisions
        self.eighth_note_times = []
        self.sixteenth_note_times = []
        for i in range(len(self.beat_times)-1):
            beat_interval = self.beat_times[i+1] - self.beat_times[i]
            self.eighth_note_times.append(self.beat_times[i] + beat_interval/2)
            self.sixteenth_note_times.append(self.beat_times[i] + beat_interval/4)
            self.sixteenth_note_times.append(self.beat_times[i] + 3*beat_interval/4)

        # Initialize counters and tracking
        self.eighth_count = 0
        self.sixteenth_count = 0
        self.is_in_pose = False
        self.pose_start_time = None
        self.last_movement_state = False
        self.last_hit_time = 0
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def is_on_any_beat(self, current_time, tolerance=0.05):
        """Check if current time is on any beat (8th or 16th)"""
        return (any(abs(current_time - eighth_time) < tolerance for eighth_time in self.eighth_note_times) or
                any(abs(current_time - sixteenth_time) < tolerance for sixteenth_time in self.sixteenth_note_times))

    def check_pose(self, landmarks):
        """Check if arms are in a pose position (elbows bent)"""
        # Get relevant landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Calculate angles
        left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Consider it a pose if both elbows are bent (angle < 150 degrees)
        return left_angle < 150 and right_angle < 150

    def process_frame(self, frame, elapsed_time):
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Check if arms are in pose position
            is_pose_position = self.check_pose(results.pose_landmarks.landmark)
            
            # Check for beats
            is_beat = self.is_on_any_beat(elapsed_time)
            
            # Detect pose transitions
            if is_pose_position:  # In pose position
                if not self.is_in_pose and is_beat:  # Starting pose on beat
                    self.is_in_pose = True
                    self.pose_start_time = elapsed_time
            else:  # Not in pose position
                if self.is_in_pose and is_beat:  # Ending pose on beat
                    self.last_hit_time = elapsed_time  # Record when we hit the pose
                self.is_in_pose = False
                self.pose_start_time = None

            # Check for eighth notes and increment counters
            if any(abs(elapsed_time - eighth_time) < 0.05 for eighth_time in self.eighth_note_times):
                self.eighth_count += 1

            # Check for sixteenth notes and increment counters
            if any(abs(elapsed_time - sixteenth_time) < 0.05 for sixteenth_time in self.sixteenth_note_times):
                self.sixteenth_count += 1

            # Display counters in top right corner
            cv2.putText(frame, f"8th: {self.eighth_count}", (frame.shape[1]-200, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"16th: {self.sixteenth_count}", (frame.shape[1]-200, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display "HIT POSE!" for 2 seconds with bolder font and darker green
            if hasattr(self, 'last_hit_time') and elapsed_time - self.last_hit_time < 2.0:
                cv2.putText(frame, "HIT POSE!", 
                           (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_TRIPLEX, 3.0, (0, 150, 0), 5)

            # Display current pose status
            if self.is_in_pose:
                cv2.putText(frame, "IN POSE", (50, frame.shape[0]-50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

def main():
    audio_file = "tempo/rottenrose.mp3"
    detector = DancePoseDetector(audio_file)
    
    # Initialize Pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    # Start webcam capture
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        frame = detector.process_frame(frame, elapsed_time)
        
        cv2.imshow("Dance Pose Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()
