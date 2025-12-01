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
            in_dim=feat_dim
        )
        self.pose = PoseRegressorMLP(in_dim=768)

    def forward(self, feat_map):
        """
        feat_map: XFeat 输出 (B, 64, H, W)
        """
        global_feat = self.agg(feat_map)  # (B, 768)
        t, r = self.pose(global_feat)
        return t, r