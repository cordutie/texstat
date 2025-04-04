# Add the parent directory to the Python path
import sys
import os
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import loss function
from texstat.functions import *
import texstat.torch_filterbanks.filterbanks as fb

# Other imports
import torch
import torchaudio
import librosa
import soundfile as sf
import resampy
import numpy as np
import pickle
import itertools
from scipy.linalg import sqrtm

def segment_audio(file_path, segment_size, sr):
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
    # Resample to 16 kHz if the sample rate is not already 16 kHz
    target_sample_rate = sr
    if sr_orig != target_sample_rate:
        audio = resampy.resample(audio, sr_orig, target_sample_rate)
    # Normalize
    audio = audio / 32768.0 

    # Segment the audio
    segments = [audio[i:i + segment_size] for i in range(0, len(audio), segment_size)]
    segments.pop()  # Remove the last segment since it might be too short
    return segments

def stats_model(segment_np, coch_fb, mod_fb, downsampler, N_moments, alpha):
    segment_torch = torch.tensor(segment_np)
    return statistics_mcds_feature_vector(segment_torch, coch_fb, mod_fb, downsampler, N_moments, alpha).detach().numpy()

# Function to extract embeddings from folder
def extract_embeddings_from_folder(folder_path, feature_extractor, segment_size, sample_rate, *model_args, **model_kwargs):
    """
    Computes embeddings for all .wav files in the given folder, segmenting long files.
    
    Parameters:
        folder_path (str): Path to the folder containing .wav files.
        feature_extractor (callable): Function that extracts embeddings.
        segment_size (float): Size of each segment in seconds.
        *model_args: Additional model arguments.
        **model_kwargs: Additional keyword arguments.
    
    Returns:
        np.ndarray: Stacked embeddings from all files.
    """
    folder_name = os.path.basename(folder_path)
    print(f"Processing folder: {folder_name}")

    embeddings_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            print(f"    Segmentating file: {file_name}")
            segments  = segment_audio(file_path, segment_size=segment_size, sr=sample_rate)
            print(f"        Computing feature vector for {len(segments)} segments.")
            for segment in segments:
                embedding_local = feature_extractor(segment, *model_args, **model_kwargs)
                if np.isnan(embedding_local).any():
                    continue
                else:
                    embeddings_list.append(embedding_local)
    print(f"Processed {len(embeddings_list)} files in {folder_path}")
    return np.vstack(embeddings_list)  # Stack into a single array

def compute_fad(embeddings_real, embeddings_fake):
    mu_real, sigma_real = np.mean(embeddings_real, axis=0), np.cov(embeddings_real, rowvar=False)
    mu_fake, sigma_fake = np.mean(embeddings_fake, axis=0), np.cov(embeddings_fake, rowvar=False)
    diff = mu_real - mu_fake
    sigma_mean = (sigma_real + sigma_fake) / 2
    fad = np.trace(sigma_real + sigma_fake - 2 * sqrtm(sigma_mean)) + np.dot(diff, diff)
    return np.real(fad)

def load_embeddings(pkl_path):
    """Load embeddings from a .pkl file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)  # Assuming stored as a NumPy array

def compute_fad_folders_pair(folder_path_1, folder_path_2, segment_size, sample_rate, save=False, *model_args, **model_kwargs):
    embeddings_1 = extract_embeddings_from_folder(folder_path_1, stats_model, segment_size, sample_rate, *model_args, **model_kwargs)
    embeddings_2 = extract_embeddings_from_folder(folder_path_2, stats_model, segment_size, sample_rate, *model_args, **model_kwargs)
    if save:
        with open(f"{folder_path_1}_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_1, f)
        with open(f"{folder_path_2}_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_2, f)
    # Compute FAD
    fad_score = compute_fad(embeddings_1, embeddings_2)
    return fad_score  