import torch

class FilterBank:
    def __init__(self, leny, fs, N, low_lim, high_lim):
        self.leny = leny
        self.fs = fs
        self.N = N
        self.low_lim = low_lim
        self.high_lim, self.freqs, self.nfreqs = self.check_limits(leny, fs, high_lim)

    def check_limits(self, leny, fs, high_lim):
        if leny % 2 == 0:
            nfreqs = leny // 2
            max_freq = fs / 2
        else:
            nfreqs = (leny - 1) // 2
            max_freq = fs * (leny - 1) / 2 / leny
        freqs = torch.linspace(0, max_freq, nfreqs + 1)
        if high_lim > fs / 2:
            high_lim = max_freq
        return high_lim, freqs, nfreqs

    # def generate_subbands(self, signal):
    #     device = signal.device  # Get the device of the input signal tensor
    #     self.filters = self.filters.to(device)  # Move filters to the same device
        
    #     if signal.shape[0] == 1:  # turn into column vector
    #         signal = signal.T.to(device)
    #     else:
    #         signal = signal.to(device)
        
    #     N = self.filters.shape[1] - 2
    #     signal_length = signal.shape[0]
    #     filt_length = self.filters.shape[0]
        
    #     fft_sample = torch.fft.fft(signal, dim=0).to(device)
        
    #     if signal_length % 2 == 0:
    #         fft_filts = torch.cat([self.filters, torch.flipud(self.filters[1:filt_length - 1, :])]).to(device)
    #     else:
    #         fft_filts = torch.cat([self.filters, torch.flipud(self.filters[1:filt_length, :])]).to(device)
        
    #     tile = fft_sample.unsqueeze(1) * torch.ones(1, N + 2, device=device)
    #     fft_subbands = fft_filts * tile
    #     # self.subbands = torch.fft.ifft(fft_subbands, dim=0).real.to(device)
    #     return torch.fft.ifft(fft_subbands, dim=0).real.to(device).transpose(0, 1)

    def generate_subbands(self, signal):
        device = signal.device
        self.filters = self.filters.to(device)  # Move filters to the same device
        
        if signal.dim() == 1:  # Single signal case
            signal = signal.unsqueeze(0)  # Add batch dimension (1, Size)
            was_single = True
        else:
            was_single = False
        
        batch_size, signal_length = signal.shape
        N = self.filters.shape[1] - 2
        filt_length = self.filters.shape[0]

        fft_sample = torch.fft.fft(signal, dim=-1).to(device)

        if signal_length % 2 == 0:
            fft_filts = torch.cat([self.filters, torch.flipud(self.filters[1:filt_length - 1, :])]).to(device)
        else:
            fft_filts = torch.cat([self.filters, torch.flipud(self.filters[1:filt_length, :])]).to(device)

        # Expand filters to match batch size
        fft_filts = fft_filts.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, freq_bins, N+2)
        fft_sample = fft_sample.unsqueeze(2).expand(-1, -1, N + 2)  # (batch_size, Size, N+2)

        fft_subbands = fft_filts * fft_sample
        subbands = torch.fft.ifft(fft_subbands, dim=-2).real.to(device)
        subbands = subbands.transpose(1, 2)  # (batch_size, N+2, Size)

        if was_single:
            return subbands.squeeze(0)  # Remove batch dimension
        return subbands


class EqualRectangularBandwidth(FilterBank):
    def __init__(self, leny, fs, N, low_lim, high_lim):
        super(EqualRectangularBandwidth, self).__init__(leny, fs, N, low_lim, high_lim)
        erb_low = self.freq2erb(torch.tensor(self.low_lim, dtype=torch.float32))
        erb_high = self.freq2erb(torch.tensor(self.high_lim, dtype=torch.float32))
        erb_lims = torch.linspace(erb_low, erb_high, self.N + 2)
        self.cutoffs = self.erb2freq(erb_lims)
        self.filters = self.make_filters(self.N, self.nfreqs, self.freqs, self.cutoffs)

    def freq2erb(self, freq_Hz):
        return 9.265 * torch.log(1 + freq_Hz / (24.7 * 9.265))

    def erb2freq(self, n_erb):
        return 24.7 * 9.265 * (torch.exp(n_erb / 9.265) - 1)

    def make_filters(self, N, nfreqs, freqs, cutoffs):
        cos_filts = torch.zeros(nfreqs + 1, N)
        for k in range(N):
            l_k = cutoffs[k]
            h_k = cutoffs[k + 2]
            l_ind = torch.min(torch.where(freqs > l_k)[0])
            h_ind = torch.max(torch.where(freqs < h_k)[0])
            avg = (self.freq2erb(l_k) + self.freq2erb(h_k)) / 2
            rnge = self.freq2erb(h_k) - self.freq2erb(l_k)
            cos_filts[l_ind:h_ind + 1, k] = torch.cos((self.freq2erb(freqs[l_ind:h_ind + 1]) - avg) / rnge * torch.pi)
        filters = torch.zeros(nfreqs + 1, N + 2)
        filters[:, 1:N + 1] = cos_filts
        h_ind = torch.max(torch.where(freqs < cutoffs[1])[0])
        filters[:h_ind + 1, 0] = torch.sqrt(1 - filters[:h_ind + 1, 1] ** 2)
        l_ind = torch.min(torch.where(freqs > cutoffs[N])[0])
        filters[l_ind:nfreqs + 1, N + 1] = torch.sqrt(1 - filters[l_ind:nfreqs + 1, N] ** 2)
        return filters

class Linear(FilterBank):
    def __init__(self, leny, fs, N, low_lim, high_lim):
        super(Linear, self).__init__(leny, fs, N, low_lim, high_lim)
        self.cutoffs = torch.linspace(self.low_lim, self.high_lim, self.N + 2)
        self.filters = self.make_filters(self.N, self.nfreqs, self.freqs, self.cutoffs)

    def make_filters(self, N, nfreqs, freqs, cutoffs):
        cos_filts = torch.zeros(nfreqs + 1, N)
        for k in range(N):
            l_k = cutoffs[k]
            h_k = cutoffs[k + 2]
            l_ind = torch.min(torch.where(freqs > l_k)[0])
            h_ind = torch.max(torch.where(freqs < h_k)[0])
            avg = (l_k + h_k) / 2
            rnge = h_k - l_k
            cos_filts[l_ind:h_ind + 1, k] = torch.cos((freqs[l_ind:h_ind + 1] - avg) / rnge * torch.pi)
        filters = torch.zeros(nfreqs + 1, N + 2)
        filters[:, 1:N + 1] = cos_filts
        h_ind = torch.max(torch.where(freqs < cutoffs[1])[0])
        filters[:h_ind + 1, 0] = torch.sqrt(1 - filters[:h_ind + 1, 1] ** 2)
        l_ind = torch.min(torch.where(freqs > cutoffs[N])[0])
        filters[l_ind:nfreqs + 1, N + 1] = torch.sqrt(1 - filters[l_ind:nfreqs + 1, N] ** 2)
        return filters

class Logarithmic(FilterBank):
    def __init__(self, leny, fs, N, low_lim, high_lim):
        super(Logarithmic, self).__init__(leny, fs, N, low_lim, high_lim)
        self.cutoffs = torch.logspace(
            torch.log10(torch.tensor(self.low_lim, dtype=torch.float32)),
            torch.log10(torch.tensor(self.high_lim, dtype=torch.float32)),
            self.N + 2
        )
        self.filters = self.make_filters(self.N, self.nfreqs, self.freqs, self.cutoffs)

    def make_filters(self, N, nfreqs, freqs, cutoffs):
        cos_filts = torch.zeros(nfreqs + 1, N)
        for k in range(N):
            l_k = cutoffs[k]
            h_k = cutoffs[k + 2]
            l_ind = torch.where(freqs >= l_k)[0][0]
            h_ind = torch.where(freqs <= h_k)[0][-1]
            avg = (torch.log10(l_k) + torch.log10(h_k)) / 2
            rnge = torch.log10(h_k) - torch.log10(l_k)
            cos_filts[l_ind:h_ind + 1, k] = torch.cos((torch.log10(freqs[l_ind:h_ind + 1]) - avg) / rnge * torch.pi)
        filters = torch.zeros(nfreqs + 1, N + 2)
        filters[:, 1:N + 1] = cos_filts
        h_ind = torch.where(freqs <= cutoffs[1])[0][-1]
        filters[:h_ind + 1, 0] = torch.sqrt(1 - filters[:h_ind + 1, 1] ** 2)
        l_ind = torch.where(freqs >= cutoffs[N])[0][0]
        filters[l_ind:nfreqs + 1, N + 1] = torch.sqrt(1 - filters[l_ind:nfreqs + 1, N] ** 2)
        return filters