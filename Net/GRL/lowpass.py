import torch 
import torch.nn as nn
import torch.nn.functional as F


# -------- FFT-based learnable low-pass filter --------
class LearnableLowPass2D(nn.Module):
    """
    Applies FFT -> multiply by Gaussian low-pass mask -> IFFT.
    The mask's sigma is a learnable parameter (per channel or global).
    The forward computes the mask on-the-fly for the given spatial size.
    Inputs: tensor (B, C, H, W)
    """
    def __init__(self, channels, per_channel=True, init_sigma=0.2, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.per_channel = per_channel
        self.eps = eps
        if per_channel:
            # store a parameter in unconstrained space, convert via softplus to ensure >0
            self._log_sigma = nn.Parameter(torch.log(torch.ones(channels) * init_sigma + eps))
        else:
            self._log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma) + eps))

    def forward(self, x):
        """
        x: (B, C, H, W) float tensor
        returns: (B, C, H, W) real tensor after low-pass filtering
        """
        B, C, H, W = x.shape
        device = x.device
        # compute frequency grid normalized to [-0.5, 0.5)
        fy = torch.fft.fftfreq(H, d=1.0, device=device)  # shape (H,)
        fx = torch.fft.fftfreq(W, d=1.0, device=device)  # shape (W,)
        # meshgrid with shape (H, W)
        fyv, fxv = torch.meshgrid(fy, fx, indexing='ij')
        # radius squared
        r2 = (fxv**2 + fyv**2)  # shape H x W

        # compute sigma
        sigma = F.softplus(self._log_sigma)  # if per_channel -> (C,), else scalar
        # ensure broadcastable into (1, C, H, W)
        if self.per_channel:
            sigma = sigma.view(1, C, 1, 1)
        else:
            sigma = sigma.view(1, 1, 1, 1)

        # Gaussian low-pass mask: exp(- (freq^2) / (2 * sigma^2))
        # But r2 are values in cycles/unit, scale appropriately - we'll use r2 as-is
        # add small eps for numerical stability
        denom = 2.0 * (sigma**2).clamp(min=self.eps)
        # r2 broadcasted to (1,1,H,W)
        mask = torch.exp(- r2.view(1, 1, H, W) / denom)  # complex multiply will broadcast
        # perform FFT per channel
        # torch.fft.fft2 handles (B,C,H,W) directly
        Xf = torch.fft.fft2(x)
        # multiply by mask (mask real -> broadcast to complex)
        Xf_filtered = Xf * mask.to(x.device)
        x_filtered = torch.fft.ifft2(Xf_filtered).real  # discard tiny imaginary parts
        return x_filtered
