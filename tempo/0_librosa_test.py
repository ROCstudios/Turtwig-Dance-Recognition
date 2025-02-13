import librosa
import numpy as np

def get_tempo(audio_file):
    # Load the audio file
    audio_file = "tempo/rottenrose.mp3"
    y, sr = librosa.load(audio_file, sr=44100)

    # Extract BPM and beat times
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print(f"Estimated BPM: {tempo}")
    print(f"Beat times: {beat_times}")

    return tempo, beat_times

if __name__ == "__main__":
    get_tempo("tempo/rottenrose.mp3")
