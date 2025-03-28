import torch

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
    h = torch.zeros(N, dtype=Xf.dtype, device=x.device)

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