

import torch.nn as nn


class SingleConvEncoder(nn.Module):
    """One-layer conv encoder (conv -> nonlinearity -> optional pooling)"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.pool = pool
        if pool:
            self.pool_layer = nn.AdaptiveAvgPool2d((8, 8))  # reduce spatial, optional

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.pool:
            x = self.pool_layer(x)
        return x 