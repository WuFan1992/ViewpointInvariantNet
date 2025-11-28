import torch.nn as nn

from .Pose.poseregression import PoseRegressorMLP
from .transformer.transformer import XFeatTransformerAggregator



# -------------------------------------------------------------
# Full Model (XFeat feature map → Pose)
# -------------------------------------------------------------
class XFeatPoseNet(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.agg = XFeatTransformerAggregator(
            in_dim=feat_dim,
            embed_dim=256,
            depth=4,
            num_heads=8,
        )
        self.pose = PoseRegressorMLP(in_dim=256)

    def forward(self, feat_map):
        """
        feat_map: XFeat 输出 (B, 64, H, W)
        """
        global_feat = self.agg(feat_map)  # (B, 256)
        t, r = self.pose(global_feat)
        return t, r