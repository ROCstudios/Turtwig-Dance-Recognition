import cv2
import pygame
import time
import librosa
import numpy as np

# Load the audio file
audio_file = "tempo/rottenrose.mp3"
y, sr = librosa.load(audio_file, sr=44100)

# Extract BPM and beat times
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# Calculate eighth and sixteenth note times
eighth_note_times = []
sixteenth_note_times = []

for i in range(len(beat_times)-1):
    beat_interval = beat_times[i+1] - beat_times[i]
    # Add 8th notes (halfway between beats)
    eighth_note_times.append(beat_times[i] + beat_interval/2)
    # Add 16th notes (quarter and three-quarters between beats)
    sixteenth_note_times.append(beat_times[i] + beat_interval/4)
    sixteenth_note_times.append(beat_times[i] + 3*beat_interval/4)

print(f"Estimated BPM: {tempo}")
print(f"Number of 8th notes: {len(eighth_note_times)}")
print(f"Number of 16th notes: {len(sixteenth_note_times)}")

# Initialize Pygame mixer
pygame.mixer.init()
pygame.mixer.music.load(audio_file)

# Start playing the song
pygame.mixer.music.play()

# Start webcam capture
cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get current time elapsed
    elapsed_time = time.time() - start_time

    # Check for eighth notes (yellow, top of screen)
    if any(abs(elapsed_time - eighth_time) < 0.05 for eighth_time in eighth_note_times):
        cv2.putText(frame, "8th", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

    # Check for sixteenth notes (blue, bottom of screen)
    if any(abs(elapsed_time - sixteenth_time) < 0.05 for sixteenth_time in sixteenth_note_times):
        cv2.putText(frame, "16th", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    # Display the webcam feed
    cv2.imshow("Webcam with BPM Sync", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
