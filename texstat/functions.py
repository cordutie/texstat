import torch
from auxiliar import *

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

# Before using, make both an erb bank and a log bank:
# erb_bank = fb.EqualRectangularBandwidth(size, sample_rate, N_filter_bank, low_lim, high_lim)
# new_size = size // 4 and new_sample_rate = sample_rate // 4
# log_bank = fb.Logarithmic(new_size, new_sample_rate, M_filter_bank, 10, new_sample_rate // 4)
# downsampler = torchaudio.transforms.Resample(sample_rate, new_sample_rate).to(device)  # Move downsampler to device
def statistics_mcds(signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha=torch.tensor([100, 1, 1/10, 1/100])):
    device = signals.device

    # alpha to device
    alpha = alpha.to(device)

    if signals.dim() == 1:  # Single signal case
        signals = signals.unsqueeze(0)  # Add batch dimension (1, Size)
        was_single = True
    else:
        was_single = False
    batch_size = signals.shape[0]

    erb_subbands = erb_bank.generate_subbands(signals)[:, 1:-1, :]

    N_filter_bank = erb_subbands.shape[1]

    env_subbands = torch.abs(hilbert(erb_subbands))

    env_subbands_downsampled = downsampler(env_subbands.float())

    length_downsampled       = env_subbands_downsampled.shape[-1]

    subenvelopes = torch.zeros((batch_size, N_filter_bank, M_filter_bank, length_downsampled), device=device)
    for i in range(N_filter_bank):
        banda     = env_subbands_downsampled[:, i, :]
        subbandas = log_bank.generate_subbands(banda)[:, 1:-1, :]
        subenvelopes[:, i, :, :] = subbandas

    mu = env_subbands.mean(dim=-1)
    sigma = env_subbands.std(dim=-1)

    stats_1 = torch.zeros(batch_size, N_filter_bank, 4, device=device)
    stats_1[:, :, 0] = mu * alpha[0]
    stats_1[:, :, 1] = ((sigma ** 2) / (mu ** 2) ) * alpha[1]
    normalized_env_subbands = (env_subbands - mu.unsqueeze(-1))
    stats_1[:, :, 2] = ((normalized_env_subbands ** 3).mean(dim=-1) / (sigma ** 3)) * alpha[2]
    stats_1[:, :, 3] = ((normalized_env_subbands ** 4).mean(dim=-1) / (sigma ** 4)) * alpha[3]

    corr_pairs = torch.triu_indices(N_filter_bank, N_filter_bank, 1)
    stats_2 = correlation_coefficient(env_subbands[:, corr_pairs[0]], env_subbands[:, corr_pairs[1]])

    subenv_sigma = subenvelopes.std(dim=-1)
    stats_3 = (subenv_sigma / (env_subbands_downsampled.std(dim=-1, keepdim=True))).view(batch_size, -1)

    cross_corr_across_subbands = correlation_coefficient(subenvelopes[:, None, :, :, :], subenvelopes[:, :, None, :, :])
    stats_4 = cross_corr_across_subbands[:, torch.triu_indices(N_filter_bank, N_filter_bank, 1)[0], torch.triu_indices(N_filter_bank, N_filter_bank, 1)[1]].view(batch_size, -1)

    cross_corr_subenvs = correlation_coefficient(subenvelopes[:, :, None, :, :], subenvelopes[:, :, :, None, :])
    stats_5 = cross_corr_subenvs[:, :, torch.triu_indices(M_filter_bank, M_filter_bank, 1)[0], torch.triu_indices(M_filter_bank, M_filter_bank, 1)[1]]
    stats_5 = stats_5.permute(0, 2, 1).contiguous().view(batch_size, -1)

    return [stats_1, stats_2, stats_3, stats_4, stats_5]

# alpha=torch.tensor([0.3, 0.15, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1])
# alpha = torch.tensor([0.0070, 0.0035, 0.8993, 0.0049, 0.0431, 0.0265, 0.0067, 0.0089])
# alpha_old = torch.tensor([1000, 1, 0.01, 0.0001, 20, 20, 20, 20]) # actually no lol
def texstat_loss(original_signals, reconstructed_signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha = torch.tensor([100,1,1/10,1/100]), beta=torch.tensor([1, 20, 20, 20, 20])):
    if original_signals.dim() == 1:  # Single signal case
        original_signals      = original_signals.unsqueeze(0)  # Add batch dimension (1, Size)
        reconstructed_signals = reconstructed_signals.unsqueeze(0)

    original_stats      = statistics_mcds(original_signals,      N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha)
    reconstructed_stats = statistics_mcds(reconstructed_signals, N_filter_bank, M_filter_bank, erb_bank, log_bank, downsampler, alpha)

    # Compute per-statistic loss and take mean over feature dimensions to ensure scalars
    losses = [torch.sqrt(torch.mean((o - r) ** 2, dim=list(range(1, o.dim())))) for o, r in zip(original_stats, reconstructed_stats)]

    # Stack and take batch mean
    losses = torch.stack(losses, dim=-1).mean(dim=0)  

    # Apply weighting
    return (losses * beta.to(losses.device)).sum()