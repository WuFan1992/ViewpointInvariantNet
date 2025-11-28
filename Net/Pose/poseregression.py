import torch
import torch.nn as nn

class PoseRegressorMLP(nn.Module):
    """3-layer MLP for pose regression/classification. Input is a flattened feature vector."""
    def __init__(self, in_dim, hidden=512, out_dim=128, dropout=0.1):
        # out_dim could be 6 (e.g., 3 for translation + 3 for rotation params) or num bins
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, out_dim)
        )
        
        self.fc_translation = nn.Linear(out_dim, 3)
        self.fc_rotation = nn.Linear(out_dim, 4)
        

    def forward(self, x):
        # x: (B, D)
        
        feat = self.net(x)
         # Pose prediction
        translation = self.fc_translation(feat)  # -> (B, 3)
        rotation = self.fc_rotation(feat)        # -> (B, 4)

        # Normalize quaternion
        rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)

        return translation, rotation
    
    
    



