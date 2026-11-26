"""3D mask decoder for SACM: a compact 3D conv head on top of the frozen 2D
SAM image encoder. Takes a stack of K consecutive slice embeddings
(B, 256, K, 64, 64) and outputs K consistent mask logits (B, 1, K, 64, 64).
"""
import torch
import torch.nn as nn


class MaskDecoder3D(nn.Module):
    def __init__(self, in_ch=256, dims=(128, 64, 32)):
        super().__init__()
        layers = []
        prev = in_ch
        for d in dims:
            layers += [nn.Conv3d(prev, d, 3, padding=1, bias=False),
                       nn.GroupNorm(min(8, d), d),
                       nn.GELU()]
            prev = d
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv3d(prev, 1, 1)

    def forward(self, embeddings):
        """embeddings: (B, 256, D, 64, 64) -> logits: (B, 1, D, 64, 64)"""
        return self.head(self.body(embeddings))
