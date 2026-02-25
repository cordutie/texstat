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
                 hop_size = None,
                 device='cpu'):
        self.frame_size          = frame_size
        self.N_filter_bank       = N_filter_bank
        self.M_filter_bank       = M_filter_bank
        self.N_moments           = N_moments
        self.sr                  = sampling_rate
        self.alpha               = alpha.to(device)
        self.downsampling_factor = downsampling_factor
        self.hop_size            = hop_size if hop_size is not None else frame_size
        self.device              = device
        self.new_sr              = self.sr // self.downsampling_factor
        self.new_frame_size      = self.frame_size // self.downsampling_factor
        self.downsampler = torchaudio.transforms.Resample(self.sr, self.new_sr).to(device)
        self.coch_fb     = fb.EqualRectangularBandwidth(self.frame_size, self.sr, self.N_filter_bank, spectrum_lower_bound, spectrum_higher_bound)
        self.mod_fb      = fb.Logarithmic(self.new_frame_size, self.new_sr, self.M_filter_bank,        10, self.new_sr // 4)

    def score(self, folder_path_1, folder_path_2, segments_number=None, 
              cache_dir=None, force_recompute=False):
        """
        Compute FAD score between two folders.
        
        Args:
            folder_path_1: Path to first audio folder
            folder_path_2: Path to second audio folder
            segments_number: Maximum number of segments to use per folder (None = use all)
            cache_dir: Custom directory for caching embeddings (None = use default)
            force_recompute: If True, ignore cache and recompute embeddings
        
        Returns:
            FAD score (float)
        """
        return compute_fad_from_folders(folder_path_1=folder_path_1, 
                                        folder_path_2=folder_path_2, 
                                        segment_size=self.frame_size, 
                                        sample_rate=self.sr,
                                        hop_size=self.hop_size,
                                        segments_number=segments_number,
                                        cache_dir=cache_dir,
                                        force_recompute=force_recompute,
                                        coch_fb = self.coch_fb, 
                                        mod_fb = self.mod_fb, 
                                        downsampler = self.downsampler,
                                        N_moments = self.N_moments,
                                        alpha = self.alpha,
                                        device = self.device)

    # def score_from_signals(self, signal_1, signal_2):
    #     return compute_fad_from_signals(signal_1=signal_1, signal_2=signal_2, segment_size=self.frame_size, coch_fb = self.coch_fb, mod_fb = self.mod_fb, downsampler = self.downsampler)

def stats_model(segment_np, coch_fb, mod_fb, downsampler, N_moments, alpha, device='cpu'):
    segment_torch = torch.tensor(segment_np, device=device)
    return sub_statistics_mcds_feature_vector(segment_torch, coch_fb, mod_fb, downsampler, N_moments, alpha).detach().cpu().numpy()

# Function to extract embeddings from folder with caching
def extract_embeddings_from_folder(folder_path, segment_size, sample_rate, hop_size=None, segments_number=None, 
                                   cache_dir=None, force_recompute=False, *model_args, **model_kwargs):
    """
    Computes embeddings for all .wav files in the given folder, segmenting long files.
    Processes files one at a time to avoid memory issues and caches results.
    
    Parameters:
        folder_path (str): Path to the folder containing .wav files.
        segment_size (int): Size of each segment in samples.
        sample_rate (int): Sample rate for audio.
        hop_size (int): Hop size between segments. If None, defaults to segment_size.
        segments_number (int): Number of segments to randomly select. If None, uses all.
        cache_dir (str): Directory to cache embeddings. If None, uses folder_path/embeddings.
        force_recompute (bool): If True, recompute even if cache exists.
        *model_args: Additional model arguments.
        **model_kwargs: Additional keyword arguments.
    
    Returns:
        np.ndarray: Stacked embeddings from all files.
    """
    folder_name = os.path.basename(folder_path)
    
    # Set up cache directory
    if cache_dir is None:
        cache_dir = os.path.join(folder_path, "embeddings_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, "embeddings.npy")
    
    # Check if cached embeddings exist
    if os.path.exists(cache_file) and not force_recompute:
        print(f"Loading cached embeddings from {cache_file}")
        return np.load(cache_file)
    
    print(f"Processing folder: {folder_name}")
    
    # Get all audio files
    all_files = get_all_wav_paths(folder_path)
    print(f"Found {len(all_files)} files in {folder_path}")
    
    embeddings_list = []
    
    # Process each file individually to save memory
    for file_idx, file_path in enumerate(all_files):
        print(f"  [{file_idx + 1}/{len(all_files)}] Processing {os.path.basename(file_path)}...")
        
        # Segment this file
        segments = segment_audio(file_path, segment_size, sample_rate, hop_size, torch_type=False)
        print(f"    Extracted {len(segments)} segments")
        
        # Compute embeddings for segments from this file
        for segment in segments:
            embedding_local = stats_model(segment, *model_args, **model_kwargs)
            if np.isnan(embedding_local).any():
                continue
            else:
                embeddings_list.append(embedding_local)
        
        # Clear segments from memory
        del segments
    
    print(f"Total embeddings computed: {len(embeddings_list)}")
    
    # Stack all embeddings
    all_embeddings = np.vstack(embeddings_list)
    
    # Randomly select segments if specified
    if segments_number is not None and segments_number < len(all_embeddings):
        print(f"Randomly selecting {segments_number} embeddings from {len(all_embeddings)}")
        indices = np.random.choice(len(all_embeddings), size=segments_number, replace=False)
        all_embeddings = all_embeddings[indices]
    
    # Save to cache
    print(f"Saving embeddings to {cache_file}")
    np.save(cache_file, all_embeddings)
    
    print(f"Processed {len(all_embeddings)} embeddings for {folder_path}")
    return all_embeddings

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
    print("Computing FAD (Fréchet Audio Distance)...")
    mu_real, sigma_real = np.mean(embeddings_real, axis=0), np.cov(embeddings_real, rowvar=False)
    mu_fake, sigma_fake = np.mean(embeddings_fake, axis=0), np.cov(embeddings_fake, rowvar=False)
    fad_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    print(f"FAD computed: {fad_score:.4f}")
    return fad_score

def compute_fad_from_folders(folder_path_1, folder_path_2, segment_size, sample_rate, hop_size=None, 
                             segments_number=None, cache_dir=None, force_recompute=False, device='cpu', *model_args, **model_kwargs):
    """
    Compute FAD between two folders of audio files.
    
    Args:
        folder_path_1: Path to first folder
        folder_path_2: Path to second folder
        segment_size: Size of audio segments in samples
        sample_rate: Sample rate for audio
        hop_size: Hop size between segments (None = no overlap)
        segments_number: Number of segments to use (None for all)
        cache_dir: Directory to cache embeddings (None = auto)
        force_recompute: Force recomputation even if cache exists
        device: Device for feature computation
        *model_args: Arguments for the feature extraction model
        **model_kwargs: Keyword arguments for the feature extraction model
    
    Returns:
        FAD score (float)
    """
    print("\n" + "="*60)
    print("Starting FAD computation")
    print("="*60)
    
    # Add device to model_kwargs
    model_kwargs['device'] = device
    
    embeddings_1 = extract_embeddings_from_folder(folder_path_1, segment_size, sample_rate, hop_size, 
                                                  segments_number, cache_dir, force_recompute, *model_args, **model_kwargs)
    embeddings_2 = extract_embeddings_from_folder(folder_path_2, segment_size, sample_rate, hop_size, 
                                                  segments_number, cache_dir, force_recompute, *model_args, **model_kwargs)
    
    # Compute FAD
    fad_score = compute_fad_from_embeddings(embeddings_1, embeddings_2)
    
    print("="*60)
    print(f"FAD computation complete: {fad_score:.4f}")
    print("="*60 + "\n")
    
    return fad_score

# def compute_fad_from_signals(signal_1, signal_2, segment_size, *model_args, **model_kwargs):
#     embeddings_1 = extract_embeddings_from_signal(signal_1, segment_size, *model_args, **model_kwargs)
#     embeddings_2 = extract_embeddings_from_signal(signal_2, segment_size, *model_args, **model_kwargs)
#     # Compute FAD
#     fad_score = compute_fad_from_embeddings(embeddings_1, embeddings_2)
#     return fad_score