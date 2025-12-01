import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# -------------------------------------------------------------
# Transformer Aggregator
# -------------------------------------------------------------
class XFeatTransformerAggregator(nn.Module):
    def __init__(self, in_dim=64, vit_model="vit_base_patch16_224"):
        super().__init__()

        # 载入预训练 ViT backbone（ImageNet 预训练）
        self.vit = timm.create_model(vit_model, pretrained=True)

        # ViT 的嵌入维度
        self.embed_dim = self.vit.embed_dim

        # 把 XFeat channel 64 映射到 ViT 的 embed_dim（通常是 768）
        self.input_proj = nn.Linear(in_dim, self.embed_dim)

        # 使用 ViT 的位置编码（可学习）
        self.pos_embed = self.vit.pos_embed  # (1, 1+N, 768)

        # 使用 ViT 的 Transformer blocks
        self.blocks = self.vit.blocks

        # 使用 ViT 的层归一化
        self.norm = self.vit.norm

        # class token
        self.cls_token = self.vit.cls_token

    def forward(self, x):
        """
        x: XFeat feature map, shape (B, 64, H, W)
        """

        B, C, H, W = x.shape

        # (B, C, H, W) → (B, H*W, C)
        x = x.reshape(B, C, H * W).transpose(1, 2)

        # 投影到 ViT 的 embed_dim，例如 768
        x = self.input_proj(x)  # (B, N, 768)
        N = x.shape[1]

        # 构造 cls token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, 768)

        # 使用 ViT 的可学习位置编码（如果 N 不同，插值）
        if N + 1 != self.pos_embed.shape[1]:
            pos = self._interpolate_pos_embed(H,W, x.device)
        else:
            pos = self.pos_embed

        x = x + pos

        # 通过 ViT 的 transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # 返回 cls token 作为全局特征
        global_feat = x[:, 0]  # (B, 768)
        return global_feat

    def _interpolate_pos_embed(self, H,W, device):
        pos_embed = self.pos_embed  # (1, 1+old_N, C)
        cls_pos = pos_embed[:, 0:1, :]
        patch_pos = pos_embed[:, 1:, :]  # (1, old_N, C)

        old_N = patch_pos.shape[1]
        old_H = old_W = int(old_N ** 0.5)

        # reshape to 2D
        patch_pos = patch_pos.reshape(1, old_H, old_W, -1).permute(0, 3, 1, 2)

        # interpolate to (H, W)
        patch_pos = F.interpolate(
            patch_pos, size=(H, W), mode="bicubic", align_corners=False
        )

        # back to sequence
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, H * W, -1)

        # combine cls + patch_pos
        return torch.cat([cls_pos, patch_pos], dim=1)