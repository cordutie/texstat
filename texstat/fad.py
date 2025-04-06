# Add the parent directory to the Python path
import sys
import os
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import loss function
from texstat.functions import *
from texstat.segmentation import *
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

def stats_model(segment_np, coch_fb, mod_fb, downsampler, N_moments=4, alpha=torch.tensor([10, 1, 1/10, 1/100])):
    segment_torch = torch.tensor(segment_np)
    return statistics_mcds_feature_vector(segment_torch, coch_fb, mod_fb, downsampler, N_moments, alpha).detach().numpy()

# Function to extract embeddings from folder
def extract_embeddings_from_folder(folder_path, segment_size, sample_rate, segments_number=None, *model_args, **model_kwargs):
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
    # Get all segments in the folder
    segments = segmentate_from_path(folder_path, sample_rate, segment_size, segments_number)
    for segment in segments:
        # compute embedding/feature vector for each segment
        embedding_local = stats_model(segment, *model_args, **model_kwargs)
        if np.isnan(embedding_local).any():
            continue
        else:
            embeddings_list.append(embedding_local)
    print(f"Processed {len(embeddings_list)} segments in {folder_path}")
    return np.vstack(embeddings_list)  # Stack into a single array


# Function to extract embeddings from folder
def extract_embeddings_from_signal(signal, segment_size, *model_args, **model_kwargs):
    embeddings_list = []
    segments = segment_audio_from_signal(signal, segment_size)
    for segment in segments:
        # compute embedding/feature vector for each segment
        embedding_local = stats_model(segment, *model_args, **model_kwargs)
        if np.isnan(embedding_local).any():
            continue
        else:
            embeddings_list.append(embedding_local)
    return np.vstack(embeddings_list)  # Stack into a single array

def compute_fad_from_embeddings(embeddings_real, embeddings_fake):
    mu_real, sigma_real = np.mean(embeddings_real, axis=0), np.cov(embeddings_real, rowvar=False)
    mu_fake, sigma_fake = np.mean(embeddings_fake, axis=0), np.cov(embeddings_fake, rowvar=False)
    diff = mu_real - mu_fake
    sigma_mean = (sigma_real + sigma_fake) / 2
    fad = np.trace(sigma_real + sigma_fake - 2 * sqrtm(sigma_mean)) + np.dot(diff, diff)
    return np.real(fad)

def compute_fad_from_folders(folder_path_1, folder_path_2, segment_size, sample_rate, segments_number=None, save=False, *model_args, **model_kwargs):
    embeddings_1 = extract_embeddings_from_folder(folder_path_1, segment_size, sample_rate, segments_number, *model_args, **model_kwargs)
    embeddings_2 = extract_embeddings_from_folder(folder_path_2, segment_size, sample_rate, segments_number, *model_args, **model_kwargs)
    if save:
        with open(f"{folder_path_1}_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_1, f)
        with open(f"{folder_path_2}_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_2, f)
    # Compute FAD
    fad_score = compute_fad_from_embeddings(embeddings_1, embeddings_2)
    return fad_score

def compute_fad_from_signals(signal_1, signal_2, segment_size, *model_args, **model_kwargs):
    embeddings_1 = extract_embeddings_from_signal(signal_1, segment_size, *model_args, **model_kwargs)
    embeddings_2 = extract_embeddings_from_signal(signal_2, segment_size, *model_args, **model_kwargs)
    # Compute FAD
    fad_score = compute_fad_from_embeddings(embeddings_1, embeddings_2)
    return fad_score
