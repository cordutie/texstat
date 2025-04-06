import os
import numpy as np
import soundfile as sf
import resampy
import torch
import random

# Function that get all paths for all .waav files in a path
def get_all_wav_paths(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                all_files.append(os.path.join(root, file))
    return all_files

# Function that segmentate a file and make a numpy array from it
def segment_audio(file_path, segment_size, sr, torch_type=False):
    """
    Segments an audio file into chunks of segment_size seconds.
    
    Parameters:
        file_path (str): Path to the audio file.
        segment_size (int): Size of each segment.
        sr (int): Sampling rate (default 44100).
    
    Returns:
        List[np.ndarray]: List of segmented audio clips.
    """
    # Read the audio file
    audio, sr_orig = sf.read(file_path, dtype='int16')

    # Check if audio is stereo and convert to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to sr
    target_sample_rate = sr
    if sr_orig != target_sample_rate:
        audio = resampy.resample(audio, sr_orig, target_sample_rate)
    # Normalize
    audio = audio / 32768.0 

    # Segment the audio
    segments = [audio[i:i + segment_size] for i in range(0, len(audio), segment_size)]
    segments.pop()  # Remove the last segment since it might be too short

    # Convert to torch tensor if needed
    if torch_type:
        segments = [torch.tensor(segment) for segment in segments]
    return segments

def segment_audio_from_signal(signal, segment_size, torch_type=False):
    # Transform to int16 and normalize
    signal = signal / np.max(np.abs(signal))
    audio = (signal * 32768).astype(np.int16)
    audio = audio / 32768.0 

    # Segment the audio
    segments = [audio[i:i + segment_size] for i in range(0, len(audio), segment_size)]
    segments.pop()  # Remove the last segment since it might be too short

    # Convert to torch tensor if needed
    if torch_type:
        segments = [torch.tensor(segment) for segment in segments]
    return segments

# Function that segmentate all files in a path and make a numpy array from it
def segmentate_from_path(path, sampling_rate = 44100, segment_length=44100, segments_number = None, torch_type=False):
    all_files = get_all_wav_paths(path)
    print(f"Found {len(all_files)} files in {path}")
    all_segments = []
    for file in all_files:
        segments = segment_audio(file, segment_length, sampling_rate, torch_type)
        print(f"    Segmented {len(segments)} segments from {file}")
        all_segments.extend(segments)
    print(f"Total segments: {len(all_segments)}")
    if segments_number is not None:
        # Randomly select segments without repetition
        if len(all_segments) < segments_number:
            # If not enough unique segments, repeat some of them
            all_segments = random.choices(all_segments, k=segments_number)
        else:
            # Otherwise, sample without repeating
            all_segments = random.sample(all_segments, k=segments_number)        
        print(f"Selected {len(all_segments)} segments after random selection")
    if torch_type:
        all_segments = torch.stack(all_segments)
    else:
        all_segments = np.vstack(all_segments)
    return all_segments