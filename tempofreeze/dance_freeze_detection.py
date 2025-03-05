import cv2
import mediapipe as mp
import numpy as np
import librosa
from typing import List, Dict, Tuple
from moviepy import VideoFileClip
import os


###############################################################################
# 1. Audio Extraction, Beat Detection, and 1/8 Note Generation
###############################################################################
def extract_audio_from_video(video_path: str, output_audio_path: str) -> None:
    """
    Extract the audio track from the video and save it as a WAV file.
    """
    print(f"[AUDIO] Extracting audio from {video_path} to {output_audio_path}...")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, logger=None)
    print("[AUDIO] Audio extraction completed.")


def detect_beats(audio_file_path: str) -> Tuple[float, np.ndarray]:
    """
    Load the audio, detect the overall tempo (BPM) and beat frames using librosa,
    and convert those frames to timestamps in seconds.

    Returns:
        tempo (float): Estimated BPM.
        beat_times (np.ndarray): Timestamps (in seconds) for each detected beat.
    """
    print(f"[BEAT] Loading audio file: {audio_file_path} ...")
    y, sr = librosa.load(audio_file_path, sr=None)  # Preserve native sample rate
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print(f"[BEAT] Beat times (first 10): {beat_times[:10]}")
    return tempo, beat_times


def get_eighth_note_times(beat_times: np.ndarray) -> np.ndarray:
    """
    Given an array of beat times (typically quarter-note times in a 4/4 track),
    subdivide each beat interval into two parts to create 1/8 note times.

    Returns:
        np.ndarray: Array of timestamps representing 1/8 note positions.
    """
    eighth_times = []
    for i in range(len(beat_times) - 1):
        start = beat_times[i]
        end = beat_times[i + 1]
        midpoint = (start + end) / 2.0
        eighth_times.append(start)  # the beat (quarter-note)
        eighth_times.append(midpoint)  # the subdivision ("and")
    if len(beat_times) > 0:
        eighth_times.append(beat_times[-1])
    eighth_array = np.array(eighth_times)
    print(f"[BEAT] Generated 1/8-note times (first 10): {eighth_array[:10]}")
    return eighth_array


###############################################################################
# 2. Pose Data Velocity Computation & Pose Detection Functions
###############################################################################
def compute_pose_velocities(pose_data: List[Dict]) -> List[Dict]:
    """
    Compute the average velocity for each frame based on differences in keypoint positions.
    (Velocity is computed here but is no longer used for counting lock-ins.)
    """
    print("[POSE] Computing velocities for collected pose data...")
    pose_data_with_vel = []
    prev_keypoints = None
    prev_time = None

    for i, frame in enumerate(pose_data):
        if i == 0 or not frame["keypoints"]:
            frame["velocity"] = 0.0
        else:
            dt = frame["timestamp"] - prev_time
            if dt <= 0:
                frame["velocity"] = 0.0
            else:
                total_speed = 0.0
                num_joints = 0
                for joint_name, (x2, y2) in frame["keypoints"].items():
                    if joint_name in prev_keypoints:
                        x1, y1 = prev_keypoints[joint_name]
                        speed = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / dt
                        total_speed += speed
                        num_joints += 1
                frame["velocity"] = total_speed / max(num_joints, 1)
        pose_data_with_vel.append(frame)
        prev_time = frame["timestamp"]
        prev_keypoints = frame["keypoints"]
    print("[POSE] Velocity computation completed.")
    return pose_data_with_vel


def is_pose_detected_near_timestamp(
    pose_data: List[Dict], target_time: float, window_ms: float
) -> bool:
    """
    Check whether a pose is detected (i.e. non-empty keypoints) near the target timestamp.
    """
    window_s = window_ms / 1000.0
    start_t = target_time - window_s
    end_t = target_time + window_s
    for frame in pose_data:
        if start_t <= frame["timestamp"] <= end_t:
            if frame["keypoints"]:
                return True
    return False


def compute_pose_detection_ratio_on_eighths(
    pose_data: List[Dict], eighth_times: np.ndarray, window_ms: float = 50
) -> float:
    """
    For each 1/8 note timestamp, determine if a pose is detected around that time.
    Compute the ratio as the number of detections divided by the total number of 1/8 notes.
    """
    if len(eighth_times) == 0:
        return 0.0
    hits = 0
    for t in eighth_times:
        if is_pose_detected_near_timestamp(pose_data, t, window_ms):
            hits += 1
    return hits / len(eighth_times)


###############################################################################
# 3. MediaPipe Pose Setup and Video Processing
###############################################################################
print("[VIDEO] Initializing MediaPipe Pose...")
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,  # For video processing
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# Drawing specs for landmarks
drawing_spec = mp_drawing.DrawingSpec(thickness=4, circle_radius=3, color=(0, 255, 0))
connection_spec = mp_drawing.DrawingSpec(thickness=4, color=(255, 255, 255))

# Video input/output paths
video_path = "videos/untouchable.mp4"  # Replace with your video path
output_video_path = "videos/output_dance.mp4"
audio_path = "temp_audio.wav"  # Temporary audio file

###############################################################################
# 4. Audio Analysis (Runs First)
###############################################################################
extract_audio_from_video(video_path, audio_path)
tempo, beat_times = detect_beats(audio_path)
eighth_note_times = get_eighth_note_times(beat_times)

# Define binding parameters for pose detection on beats
window_s = 0.05  # 50 ms window (0.05 sec)

###############################################################################
# 5. Video Processing: Pose Tracking, Beat Binding, and Annotation
###############################################################################
print("[VIDEO] Opening video capture...")
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(
    f"[VIDEO] Video properties: width={width}, height={height}, fps={fps}, total_frames={frame_count}"
)

# Initialize video writer for annotated output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Prepare to collect pose data for post-analysis and real-time processing
pose_data_all = []  # List to store per-frame {timestamp, keypoints, velocity}
lock_in_count = 0  # Counter for poses detected on 1/8 notes
current_beat_index = 0  # Index into eighth_note_times

prev_keypoints = {}  # For computing instantaneous velocity (for logging only)
prev_time = None

frame_idx = 0

global_shoulder_tilt = 0
global_head_offset = 0

print("[VIDEO] Starting video processing loop...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[VIDEO] No more frames to read. Exiting loop.")
        break

    # Current timestamp (seconds)
    current_time = frame_idx / fps

    # Convert frame to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    # Dictionary to hold keypoints for this frame
    frame_keypoints = {}

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=connection_spec,
        )

        # Extract keypoints and convert normalized coordinates to pixels
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            frame_keypoints[f"{idx}"] = (landmark.x * width, landmark.y * height)

        # Compute shoulder tilt and head offset (for demonstration)
        landmarks = results.pose_landmarks.landmark
        shoulder_l = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        shoulder_r = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        ]
        head = [
            landmarks[mp_pose.PoseLandmark.NOSE.value].x,
            landmarks[mp_pose.PoseLandmark.NOSE.value].y,
        ]
        shoulder_tilt = abs(shoulder_l[1] - shoulder_r[1]) * height
        global_shoulder_tilt = max(global_shoulder_tilt, shoulder_tilt)
        shoulder_center_x = (shoulder_l[0] + shoulder_r[0]) / 2
        head_offset = abs(head[0] - shoulder_center_x) * width
        global_head_offset = max(global_head_offset, head_offset)

        cv2.putText(
            frame,
            f"Shoulder Tilt: {int(shoulder_tilt)}px",
            (int(width * 0.1), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Head Offset: {int(head_offset)}px",
            (int(width * 0.6), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        if shoulder_tilt > 20 or head_offset > 30:
            cv2.putText(
                frame,
                "POSE DETECTED!",
                (int(width * 0.3), int(height * 0.9)),
                cv2.FONT_HERSHEY_TRIPLEX,
                1.5,
                (0, 255, 0),
                3,
            )

    # Compute instantaneous velocity for logging/post-analysis (optional)
    current_velocity = 0.0
    if prev_time is not None and prev_keypoints and frame_keypoints:
        dt = current_time - prev_time
        total_speed = 0.0
        count = 0
        for joint, (x2, y2) in frame_keypoints.items():
            if joint in prev_keypoints:
                x1, y1 = prev_keypoints[joint]
                speed = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / dt
                total_speed += speed
                count += 1
        if count > 0:
            current_velocity = total_speed / count

    if frame_idx % 30 == 0:
        print(
            f"[VIDEO] Frame {frame_idx}: timestamp={current_time:.2f}s, velocity={current_velocity:.2f}"
        )

    # Store current frame's pose data for post-analysis
    pose_data_all.append(
        {
            "timestamp": current_time,
            "keypoints": frame_keypoints,
            "velocity": current_velocity,
        }
    )

    prev_keypoints = frame_keypoints.copy()
    prev_time = current_time

    # --- Lock-In Condition: Check if it's an 1/8 note AND a pose is detected ---
    if current_beat_index < len(eighth_note_times):
        beat_time = eighth_note_times[current_beat_index]
        if (current_time >= beat_time - window_s) and (
            current_time <= beat_time + window_s
        ):
            # Lock-in if a pose is detected in this window.
            if frame_keypoints and shoulder_tilt > 20 or head_offset > 30:
                lock_in_count += 1
                print(
                    f"[BEAT] Lock-in detected at beat index {current_beat_index} (time: {beat_time:.2f}s)"
                )
                current_beat_index += 1  # Move to next beat once detected
        elif current_time > beat_time + window_s:
            print(f"[BEAT] Missed beat at time {beat_time:.2f}s. Moving to next beat.")
            current_beat_index += 1

    # Draw the lock-in count on the frame
    cv2.putText(
        frame,
        f"Lock-in Count: {lock_in_count}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    out.write(frame)
    cv2.imshow("Pose & Beat Binding", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[VIDEO] 'q' pressed. Exiting video loop.")
        break

    frame_idx += 1

print("[VIDEO] Video processing loop finished. Cleaning up...")
cap.release()
out.release()
cv2.destroyAllWindows()
pose_detector.close()

###############################################################################
# 6. Post-Processing: Compute Overall Pose Detection Ratio (Optional)
###############################################################################
print("[POST] Computing overall pose detection ratio from collected pose data...")
pose_data_with_velocity = compute_pose_velocities(pose_data_all)
overall_pose_detection_ratio = compute_pose_detection_ratio_on_eighths(
    pose_data=pose_data_with_velocity,
    eighth_times=eighth_note_times,
    window_ms=50,  # +/- 50 ms window
)
print(f"[POST] Final Lock-in Count: {lock_in_count}")
print(
    f"[POST] Overall Pose Detection Ratio on 1/8 Notes: {overall_pose_detection_ratio:.2f}"
)
print(
    "[POST] Processing complete. A higher count/ratio indicates the dancer consistently 'locks in' on the 1/8-note beats."
)

if os.path.exists(audio_path):
    os.remove(audio_path)
    print(f"[CLEANUP] Removed temporary audio file: {audio_path}")
