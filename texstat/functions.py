import torch
import texstat.torch_filterbanks as fb

def hilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform on PyTorch tensors.

    Parameters
    ----------
    x : torch.Tensor
        Signal data. Must be real.
    N : int, optional
        Number of Fourier components. Default: `x.shape[axis]`
    axis : int, optional
        Axis along which to do the transformation. Default: -1.

    Returns
    -------
    xa : torch.Tensor
        Analytic signal of `x`, of each 1-D array along `axis`
    """
    if torch.is_complex(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = torch.fft.fft(x, n=N, dim=axis)
    h  = torch.zeros(N, dtype=Xf.dtype, device=x.device)

    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [None] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]

    x_analytic = torch.fft.ifft(Xf * h, dim=axis)
    return x_analytic

def correlation_coefficient(x, y):
    """
    Compute the Pearson correlation between two tensors along the last dimension. Useful for correlation computing in batches.

    Parameters
    ----------
    x,y : torch.Tensor
        Signal data. Must be real.

    Returns
    -------
    rho(x,y) : torch.Tensor
        Pearson correlation 
    """
    mean1 = x.mean(dim=-1, keepdim=True)
    mean2 = y.mean(dim=-1, keepdim=True)
    
    x = x - mean1
    y = y - mean2
    
    std1 = x.norm(dim=-1) / (x.shape[-1] ** 0.5)  # Equivalent to std but avoids computing mean again
    std2 = y.norm(dim=-1) / (y.shape[-1] ** 0.5)
    
    corr = (x * y).mean(dim=-1) / (std1 * std2)
    return corr

# Before using, make a cochlear filterbank (e.g. ERB Filterbank), a modulation filterbank (e.g. Log Filterbank) and a downsampler.
# Example:
#   coch_fb  = fb.EqualRectangularBandwidth(frame_size, sample_rate, N_filter_bank, low_lim, high_lim)
#   log_bank = fb.Logarithmic(new_size, new_sample_rate, M_filter_bank, 10, new_sample_rate // 4)
#   new_size = size // 4 and new_sample_rate = sample_rate // 4
#   downsampler = torchaudio.transforms.Resample(sample_rate, new_sample_rate)
def statistics_mcds(signals, coch_fb, mod_fb, downsampler, N_moments = 4, alpha = torch.tensor([100, 1, 1/10, 1/100])):
    """
    Compute the summary statistics introduced by McDermott and Simoncelli. These statistics correspond to
    
    S_1 = Statistical Moments of amplitude envelopes of cochlear subband decomposition
    S_2 = Correlation coefficients of amplitude envelopes of cochlear subband decomposition
    S_3 = Energy portion of each modulation band
    S_4 = Correlation coefficients of modulation bands of same cochlear subband decomposition
    S_5 = Correlation coefficients of modulation bands of different cochlear subband decomposition
    
    For a detailed description, see TexStat paper.

    Parameters
    ----------
    signals : torch.Tensor
        Signal data. Must be real.
    coch_fb : filterbank object
        Cochlear filterbank object
    mod_fb : filterbank object
        Modulation filterbank object
    downsampler : torch.nn.Module
        Downsampler object
    N_moments : int
        Number of statistical moments to compute. Default: 4
    alpha : torch.Tensor
        weigths for S_1 statistics. Default: [100, 1, 1/10, 1/100]

    Returns
    -------
    [S_1, S_2, S_3, S_4, S_5] : list of torch.Tensor
                                list of summary statistics with S_1 being weighted by alpha.
    """
    # Check N_moments is integer bigger or equal to 2
    if not isinstance(N_moments, int) or N_moments < 2:
        raise ValueError("N_moments must be an integer greater than or equal to 2.")
    # Check alpha is a tensor of size N_moments
    if not isinstance(alpha, torch.Tensor) or alpha.size(0) != N_moments:
        raise ValueError("alpha must be a tensor of size N_moments.")

    # retrieve device of the signal tensor
    device = signals.device

    # retrieve size of the filterbanks
    N_filter_bank = coch_fb.N
    M_filter_bank = mod_fb.N

    # alpha to device
    alpha = alpha.to(device)

    # check if the input is a single signal or a batch
    if signals.dim() == 1:  # Single signal case
        signals = signals.unsqueeze(0)  # Add batch dimension (1, Size)
        was_single = True
    else:
        was_single = False
    
    # Define the batch size
    batch_size = signals.shape[0]

    # Compute the cochlear subband decomposition
    erb_subbands = coch_fb.generate_subbands(signals)[:, 1:-1, :]

    # Compute the amplitude envelope of the cochlear subband decomposition
    env_subbands = torch.abs(hilbert(erb_subbands))

    # Downsample the amplitude envelopes for modulation analysis
    env_subbands_downsampled = downsampler(env_subbands.float())

    # Compute the length of the downsampled signal
    length_downsampled       = env_subbands_downsampled.shape[-1]

    # Compute the modulation subband decomposition
    subenvelopes = torch.zeros((batch_size, N_filter_bank, M_filter_bank, length_downsampled), device=device)
    for i in range(N_filter_bank):
        banda     = env_subbands_downsampled[:, i, :]
        subbandas = mod_fb.generate_subbands(banda)[:, 1:-1, :]
        subenvelopes[:, i, :, :] = subbandas

    # Defien the matrix S_1
    stats_1 = torch.zeros(batch_size, N_filter_bank, N_moments, device=device)

    # Compute the first two statistical moments
    epsilon = 1e-8
    mu = env_subbands.mean(dim=-1)
    sigma = env_subbands.std(dim=-1)
    stats_1[:, :, 0] = mu * alpha[0]
    stats_1[:, :, 1] = ((sigma ** 2) / (mu ** 2 + epsilon) ) * alpha[1]
    
    # If N_moments is bigger than 2 keep computing
    if N_moments > 2:
        # Compute normalized envelopes for faster computation of statistical moments.
        normalized_env_subbands = (env_subbands - mu.unsqueeze(-1))
        for j in range(2, N_moments):
            stats_1[:, :, j] = ((normalized_env_subbands ** j ).mean(dim=-1) / (sigma ** j + epsilon)) * alpha[j]

    # Compute the second set of statistics
    corr_pairs = torch.triu_indices(N_filter_bank, N_filter_bank, 1)
    stats_2 = correlation_coefficient(env_subbands[:, corr_pairs[0]], env_subbands[:, corr_pairs[1]])

    # Compute the third set of statistics
    subenv_sigma = subenvelopes.std(dim=-1)
    stats_3 = (subenv_sigma / (env_subbands_downsampled.std(dim=-1, keepdim=True))).view(batch_size, -1)

    # Compute the fourth set of statistics
    cross_corr_across_subbands = correlation_coefficient(subenvelopes[:, None, :, :, :], subenvelopes[:, :, None, :, :])
    stats_4 = cross_corr_across_subbands[:, torch.triu_indices(N_filter_bank, N_filter_bank, 1)[0], torch.triu_indices(N_filter_bank, N_filter_bank, 1)[1]].view(batch_size, -1)

    # Compute the fifth set of statistics
    cross_corr_subenvs = correlation_coefficient(subenvelopes[:, :, None, :, :], subenvelopes[:, :, :, None, :])
    stats_5 = cross_corr_subenvs[:, :, torch.triu_indices(M_filter_bank, M_filter_bank, 1)[0], torch.triu_indices(M_filter_bank, M_filter_bank, 1)[1]]
    stats_5 = stats_5.permute(0, 2, 1).contiguous().view(batch_size, -1)

    # If the input was a single signal, remove the batch dimension on each statistic
    if was_single:
        stats_1 = stats_1.squeeze(0)
        stats_2 = stats_2.squeeze(0)
        stats_3 = stats_3.squeeze(0)
        stats_4 = stats_4.squeeze(0)
        stats_5 = stats_5.squeeze(0)

    # Return the statistics
    return [stats_1, stats_2, stats_3, stats_4, stats_5]

def statistics_mcds_feature_vector(signal, coch_fb, mod_fb, downsampler, N_moments = 4, alpha = torch.tensor([100, 1, 1/10, 1/100])):
    stats_1, stats_2, stats_3, stats_4, stats_5 = statistics_mcds(signal, coch_fb, mod_fb, downsampler, N_moments, alpha)
    # transform to 1D
    stats_1 = stats_1.view(-1)
    stats_2 = stats_2.view(-1)
    stats_3 = stats_3.view(-1)
    stats_4 = stats_4.view(-1)
    stats_5 = stats_5.view(-1)

    # concatenate all stats
    stats = torch.cat((stats_1, stats_2, stats_3, stats_4, stats_5), dim=0)
    return stats

def texstat_loss(x, y, coch_fb, mod_fb, downsampler, N_moments = 4, alpha = torch.tensor([100,1,1/10,1/100]), beta=torch.tensor([1, 20, 20, 20, 20])):
    """
    Compute the TexStat loss between two signals. The loss is computed as the weighted sum of the differences between the summary statistics of the two signals.

    Parameters
    ----------
    x,y : torch.Tensor
        Signals to be compared. Batches allowed.
    coch_fb : filterbank object
        Cochlear filterbank object
    mod_fb : filterbank object
        Modulation filterbank object
    downsampler : torch.nn.Module
        Downsampler object
    N_moments : int
        Number of statistical moments to compute. Default: 4
    alpha : torch.Tensor
        weigths for S_1 statistics. Default: [100, 1, 1/10, 1/100]
    beta : torch.Tensor
        Weights for the summary statistics. Default: [1, 20, 20, 20, 20]

    Returns
    -------
    L : torch.Tensor
        Loss computed over the batches of signals x and y.
    """

    # check if the input is a single signal or a batch
    if x.dim() == 1:  # Single signal case
        x = x.unsqueeze(0)  # Add batch dimension (1, Size)
        y = y.unsqueeze(0)  # Add batch dimension (1, Size)
    
    # Compute the summary statistics
    original_stats      = statistics_mcds(x, coch_fb, mod_fb, downsampler, N_moments, alpha)
    reconstructed_stats = statistics_mcds(y, coch_fb, mod_fb, downsampler, N_moments, alpha)

    # Compute per-statistic loss and take mean over feature dimensions to ensure scalars
    losses = [torch.sqrt(torch.mean((o - r) ** 2, dim=list(range(1, o.dim())))) for o, r in zip(original_stats, reconstructed_stats)]

    # Stack and take batch mean
    losses = torch.stack(losses, dim=-1).mean(dim=0)  

    # Apply weighting
    return (losses * beta.to(losses.device)).sum()