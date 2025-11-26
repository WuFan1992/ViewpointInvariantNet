
import torch
import torch.nn as nn
import torch.nn.functional as F


from .poseregression import PoseRegressorMLP

class PoseNetMLP(nn.Module):
    def __init__(self,  poseregressionmlp: PoseRegressorMLP, mlp_dim=1024):
        super().__init__()

        # 1) Global feature extraction
        # 将 H×W 的 feature map 压缩成 64 维的全局向量
        self.global_pool = nn.AdaptiveAvgPool2d(1)   # -> (B, 64, 1, 1)

        # 2) 3-layer MLP for pose regression
        self.mlp = poseregressionmlp

       
    def forward(self, x):
        # x: (B, 64, H, W)
        # Global pooled feature
        x = self.global_pool(x)  
        x = x.unsqueeze(0) # -> (B, 64, 1, 1)
        x = x.view(x.size(0), -1)
        
        translation, rotation = self.mlp(x)       # -> (B, mlp_dim)

        return translation, rotation
    

        
    
