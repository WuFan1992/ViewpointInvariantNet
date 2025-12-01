
import torch
import torch.nn as nn

# ---------------------------
# Reconstruction Head (Decoder)
# ---------------------------
class ReconstructionHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 简单两层 conv decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
    def forward(self, x):
        return self.decoder(x)