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

class KAD_wrapper():
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
                 bandwidth = None,
                 kernel = 'gaussian',
                 device='cpu'):
        self.frame_size          = frame_size
        self.N_filter_bank       = N_filter_bank
        self.M_filter_bank       = M_filter_bank
        self.N_moments           = N_moments
        self.sr                  = sampling_rate
        self.alpha               = alpha.to(device)
        self.downsampling_factor = downsampling_factor
        self.hop_size            = hop_size if hop_size is not None else frame_size
        self.bandwidth           = bandwidth
        self.kernel              = kernel
        self.device              = device
        self.new_sr              = self.sr // self.downsampling_factor
        self.new_frame_size      = self.frame_size // self.downsampling_factor
        self.downsampler = torchaudio.transforms.Resample(self.sr, self.new_sr).to(device)
        self.coch_fb     = fb.EqualRectangularBandwidth(self.frame_size, self.sr, self.N_filter_bank, spectrum_lower_bound, spectrum_higher_bound)
        self.mod_fb      = fb.Logarithmic(self.new_frame_size, self.new_sr, self.M_filter_bank,        10, self.new_sr // 4)

    def score(self, folder_path_1, folder_path_2, segments_number=None, 
              cache_dir=None, force_recompute=False):
        """
        Compute KAD score between two folders.
        
        Args:
            folder_path_1: Path to first audio folder
            folder_path_2: Path to second audio folder
            segments_number: Maximum number of segments to use per folder (None = use all)
            cache_dir: Custom directory for caching embeddings (None = use default)
            force_recompute: If True, ignore cache and recompute embeddings
        
        Returns:
            KAD score (float)
        """
        return compute_kad_from_folders(folder_path_1=folder_path_1, 
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
                                        bandwidth = self.bandwidth,
                                        kernel = self.kernel,
                                        device = self.device)

def stats_model(segment_np, coch_fb, mod_fb, downsampler, N_moments, alpha):
    segment_torch = torch.tensor(segment_np)
    return sub_statistics_mcds_feature_vector(segment_torch, coch_fb, mod_fb, downsampler, N_moments, alpha).detach().numpy()

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

def median_pairwise_distance(x, subsample=None):
    """
    Compute the median pairwise distance of an embedding set.
    
    Args:
        x: torch.Tensor of shape (n_samples, embedding_dim)
        subsample: int, number of random pairs to consider (optional)
    
    Returns:
        The median pairwise distance between points in x.
    """
    x = torch.tensor(x, dtype=torch.float32)
    n_samples = x.shape[0]
    
    if subsample is not None and subsample < n_samples * (n_samples - 1) / 2:
        # Randomly select pairs of indices
        idx1 = torch.randint(0, n_samples, (subsample,))
        idx2 = torch.randint(0, n_samples, (subsample,))
        
        # Ensure idx1 != idx2
        mask = idx1 == idx2
        idx2[mask] = (idx2[mask] + 1) % n_samples
        
        # Compute distances for selected pairs
        distances = torch.sqrt(torch.sum((x[idx1] - x[idx2])**2, dim=1))
    else:
        # Compute all pairwise distances
        distances = torch.pdist(x)
        
    return torch.median(distances).item()

def calculate_kernel_audio_distance(embeddings_1, embeddings_2, bandwidth=None, kernel='gaussian', device='cpu', eps=1e-8):
    """
    Compute the Kernel Audio Distance (KAD) between two samples using PyTorch.
    
    This implements the Maximum Mean Discrepancy (MMD) with kernel methods.
    KAD = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    
    where k is a kernel function (e.g., Gaussian RBF).

    Args:
        embeddings_1: The first set of embeddings of shape (m, embedding_dim).
        embeddings_2: The second set of embeddings of shape (n, embedding_dim).
        bandwidth: The bandwidth value for the kernel. If None, uses median heuristic.
        kernel: Kernel function to use ('gaussian', 'iq', 'imq').
        device: Device to run computation on ('cpu' or 'cuda').
        eps: Small value to prevent division by zero.

    Returns:
        The KAD between the two embedding sets (scaled by 100).
    """
    SCALE_FACTOR = 100
    
    # Convert to tensors and move to device
    x = torch.tensor(embeddings_1, dtype=torch.float32, device=device)
    y = torch.tensor(embeddings_2, dtype=torch.float32, device=device)
    
    # Use median distance heuristic if bandwidth not provided
    if bandwidth is None:
        bandwidth = median_pairwise_distance(y)
    
    m, n = x.shape[0], y.shape[0]
    
    # Define kernel functions
    gamma = 1 / (2 * bandwidth**2 + eps)
    if kernel == 'gaussian':    # Gaussian Kernel (RBF)
        kernel_fn = lambda a: torch.exp(-gamma * a)
    elif kernel == 'iq':        # Inverse Quadratic Kernel
        kernel_fn = lambda a: 1 / (1 + gamma * a)
    elif kernel == 'imq':       # Inverse Multiquadric Kernel
        kernel_fn = lambda a: 1 / torch.sqrt(1 + gamma * a)
    else:
        raise ValueError("Invalid kernel type. Valid kernels: 'gaussian', 'iq', 'imq'")
    
    # Compute kernel for x vs x (k_xx)
    xx = x @ x.T
    x_sqnorms = torch.diagonal(xx)
    d2_xx = x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0) - 2 * xx  # shape (m, m)
    k_xx = kernel_fn(d2_xx)
    k_xx = k_xx - torch.diag(torch.diagonal(k_xx))  # Remove diagonal (unbiased estimator)
    k_xx_mean = k_xx.sum() / (m * (m - 1))
    
    # Compute kernel for y vs y (k_yy)
    yy = y @ y.T
    y_sqnorms = torch.diagonal(yy)
    d2_yy = y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * yy  # shape (n, n)
    k_yy = kernel_fn(d2_yy)
    k_yy = k_yy - torch.diag(torch.diagonal(k_yy))  # Remove diagonal (unbiased estimator)
    k_yy_mean = k_yy.sum() / (n * (n - 1))
    
    # Compute kernel for x vs y (k_xy)
    xy = x @ y.T
    d2_xy = x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * xy  # shape (m, n)
    k_xy = kernel_fn(d2_xy)
    k_xy_mean = k_xy.mean()
    
    # Compute MMD (Maximum Mean Discrepancy)
    result = k_xx_mean + k_yy_mean - 2 * k_xy_mean
    
    return (result * SCALE_FACTOR).item()

def compute_kad_from_embeddings(embeddings_real, embeddings_fake, bandwidth=None, kernel='gaussian', device='cpu'):
    """
    Compute KAD from pre-computed embeddings.
    
    Args:
        embeddings_real: Embeddings from the first set (numpy array)
        embeddings_fake: Embeddings from the second set (numpy array)
        bandwidth: Kernel bandwidth (None for automatic)
        kernel: Kernel type ('gaussian', 'iq', 'imq')
        device: Computation device
    
    Returns:
        KAD score (float)
    """
    print("Computing KAD (Kernel Audio Distance)...")
    kad_score = calculate_kernel_audio_distance(embeddings_real, embeddings_fake, bandwidth, kernel, device)
    print(f"KAD computed: {kad_score:.4f}")
    return kad_score

def compute_kad_from_folders(folder_path_1, folder_path_2, segment_size, sample_rate, hop_size=None,
                             segments_number=None, cache_dir=None, force_recompute=False, 
                             bandwidth=None, kernel='gaussian', device='cpu', *model_args, **model_kwargs):
    """
    Compute KAD between two folders of audio files.
    
    Args:
        folder_path_1: Path to first folder
        folder_path_2: Path to second folder
        segment_size: Size of audio segments in samples
        sample_rate: Sample rate for audio
        hop_size: Hop size between segments (None = no overlap)
        segments_number: Number of segments to use (None for all)
        cache_dir: Directory to cache embeddings (None = auto)
        force_recompute: Force recomputation even if cache exists
        bandwidth: Kernel bandwidth (None for automatic)
        kernel: Kernel type ('gaussian', 'iq', 'imq')
        device: Computation device
        *model_args: Arguments for the feature extraction model
        **model_kwargs: Keyword arguments for the feature extraction model
    
    Returns:
        KAD score (float)
    """
    print("\n" + "="*60)
    print("Starting KAD computation")
    print("="*60)
    
    embeddings_1 = extract_embeddings_from_folder(folder_path_1, segment_size, sample_rate, hop_size,
                                                  segments_number, cache_dir, force_recompute, *model_args, **model_kwargs)
    embeddings_2 = extract_embeddings_from_folder(folder_path_2, segment_size, sample_rate, hop_size,
                                                  segments_number, cache_dir, force_recompute, *model_args, **model_kwargs)
    
    # Compute KAD
    kad_score = compute_kad_from_embeddings(embeddings_1, embeddings_2, bandwidth, kernel, device)
    
    print("="*60)
    print(f"KAD computation complete: {kad_score:.4f}")
    print("="*60 + "\n")
    
    return kad_score
