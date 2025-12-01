import torch.nn as nn

from .GRL.encoder import SingleConvEncoder
from .GRL.lowpass import LearnableLowPass2D
from .GRL.grl_layer import grl

from .Pose.reconstruction import ReconstructionHead


# -------- Full Model --------
class InvariantFeatureNet(nn.Module):
    """
    - encoder1: single-layer encoder (image -> raw features)
    - FFT -> learnable low-pass -> IFFT
    - encoder2: single-layer encoder to produce final invariant embedding
    - an external pose_regressor MLP takes GRL(invariant) flattened and regresses pose
    """
    def __init__(self, feat_extract, in_channels=64, enc1_ch=256, enc2_ch=64, per_channel_sigma=True):
        super().__init__()
        self.encoder1 = SingleConvEncoder(in_channels, enc1_ch, kernel_size=3, padding=1, pool=False)
        self.lowpass = LearnableLowPass2D(enc1_ch, per_channel=per_channel_sigma, init_sigma=0.08)
        # second encoder: we keep kernel=1 conv to be a "one-layer encoder" in spec
        self.encoder2 = SingleConvEncoder(enc1_ch, enc2_ch, kernel_size=1, padding=0, pool=False)
        # the dimension of invariant embedding vector (after pooling)
        self.embedding_dim = enc2_ch * 8 * 8  # due to AdaptiveAvgPool in encoder2
        
        self.decoder = ReconstructionHead(enc2_ch, in_channels)

    def forward(self, x, grl_lambda=1.0):
        """
        x: (B, C, H, W) Feature Map
        returns:
          invariant_embedding: (B, D) flattened embedding (before GRL)
          adv_input: (B, D) flattened embedding after GRL (to feed adversary)
        """
  
        f1 = self.encoder1(x)               # (B, enc1_ch, H, W)
        f1_filtered = self.lowpass(f1)     # keep shape (B, enc1_ch, H, W)
        f2 = self.encoder2(f1_filtered)    # (B, enc2_ch, H2, W2) e.g. (B, enc2_ch, 8,8)
        recon = self.decoder(f2)
        # adv input passes though GRL when training the encoder vs pose regressor adversarially
        adv_emb = grl(f2, grl_lambda)
        return recon, f2, adv_emb
