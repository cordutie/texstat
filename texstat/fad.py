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
from scipy import linalg

class FAD_wrapper():
    def __init__(self, frame_size, 
                 N_filter_bank = 16, 
                 M_filter_bank = 6, 
                 N_moments     = 4, 
                 sampling_rate = 44100, 
                 downsampling_factor = 4, 
                 alpha               = torch.tensor([1, 1/10, 1/100, 1/1000]),
                 spectrum_lower_bound = 20, 
                 spectrum_higher_bound = 16000, 
                 device='cpu'):
        self.frame_size          = frame_size
        self.N_filter_bank       = N_filter_bank
        self.M_filter_bank       = M_filter_bank
        self.N_moments           = N_moments
        self.sr                  = sampling_rate
        self.alpha               = alpha.to(device)
        self.downsampling_factor = downsampling_factor
        self.new_sr              = self.sr // self.downsampling_factor
        self.new_frame_size      = self.frame_size // self.downsampling_factor
        self.downsampler = torchaudio.transforms.Resample(self.sr, self.new_sr).to(device)
        self.coch_fb     = fb.EqualRectangularBandwidth(self.frame_size, self.sr, self.N_filter_bank, spectrum_lower_bound, spectrum_higher_bound)
        self.mod_fb      = fb.Logarithmic(self.new_frame_size, self.new_sr, self.M_filter_bank,        10, self.new_sr // 4)

    def score(self, folder_path_1, folder_path_2, save_embeddings=False, segments_number=None):
        return compute_fad_from_folders(folder_path_1=folder_path_1, 
                                        folder_path_2=folder_path_2, 
                                        segment_size=self.frame_size, 
                                        sample_rate=self.sr, 
                                        segments_number=segments_number, 
                                        save=save_embeddings, 
                                        coch_fb = self.coch_fb, 
                                        mod_fb = self.mod_fb, 
                                        downsampler = self.downsampler,
                                        N_moments = self.N_moments,
                                        alpha = self.alpha)

    # def score_from_signals(self, signal_1, signal_2):
    #     return compute_fad_from_signals(signal_1=signal_1, signal_2=signal_2, segment_size=self.frame_size, coch_fb = self.coch_fb, mod_fb = self.mod_fb, downsampler = self.downsampler)

def stats_model(segment_np, coch_fb, mod_fb, downsampler, N_moments, alpha):
    segment_torch = torch.tensor(segment_np)
    return sub_statistics_mcds_feature_vector(segment_torch, coch_fb, mod_fb, downsampler, N_moments, alpha).detach().numpy()

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

# # Function to extract embeddings from folder
# def extract_embeddings_from_signal(signal, segment_size, *model_args, **model_kwargs):
#     embeddings_list = []
#     segments = segment_audio_from_signal(signal, segment_size)
#     for segment in segments:
#         # compute embedding/feature vector for each segment
#         embedding_local = stats_model(segment, *model_args, **model_kwargs)
#         if np.isnan(embedding_local).any():
#             continue
#         else:
#             embeddings_list.append(embedding_local)
#     return np.vstack(embeddings_list)  # Stack into a single array

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Code snippet taken from: https://github.com/gudgud96/frechet-audio-distance/blob/main/frechet_audio_distance/fad.py
    which was adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
            representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def compute_fad_from_embeddings(embeddings_real, embeddings_fake):
    mu_real, sigma_real = np.mean(embeddings_real, axis=0), np.cov(embeddings_real, rowvar=False)
    mu_fake, sigma_fake = np.mean(embeddings_fake, axis=0), np.cov(embeddings_fake, rowvar=False)
    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

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

# def compute_fad_from_signals(signal_1, signal_2, segment_size, *model_args, **model_kwargs):
#     embeddings_1 = extract_embeddings_from_signal(signal_1, segment_size, *model_args, **model_kwargs)
#     embeddings_2 = extract_embeddings_from_signal(signal_2, segment_size, *model_args, **model_kwargs)
#     # Compute FAD
#     fad_score = compute_fad_from_embeddings(embeddings_1, embeddings_2)
#     return fad_score