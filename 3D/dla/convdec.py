"""3D convolutional decoder replacing SAM-Med3D's transformer mask decoder.

Why: the transformer decoder emits masks at 128/4 = 32^3, i.e. 3.2 mm voxels at
0.8 mm spacing -- thinner than the vessels we want. A 128^3 output also needs
features finer than stride 16, which the ViT never produces (patch_embed is a
stride-16 conv), so a ConvStem3D on the input feeds the fine scales.

    image 128^3
      ├── ConvStem3D            : 1 -> 16 (64^3) -> 32 (32^3) -> 64 (16^3)
      └── ViT + SACM adapters    : neck 384@8^3, external adapters 768@8^3 x5
                                    ↓ 1x1x1 proj + softmax depth weighting
              decoder 8^3 -> 16^3 -> 32^3 -> 64^3 -> 128^3 (skips: stem + input)
                                    ↓
              logits 1 x 128^3
"""
from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



class ConvBlock3D(nn.Module):
    """2 x (3^3 conv + GroupNorm + GELU)."""

    def __init__(self, cin: int, cout: int, groups: int = 8) -> None:
        super().__init__()
        groups = min(groups, cout)
        while cout % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ConvStem3D(nn.Module):
    """Fine-scale branch: stride 2 / 4 / 8 features straight from the image."""

    def __init__(self, ch: Tuple[int, int, int] = (16, 32, 64)) -> None:
        super().__init__()
        c1, c2, c3 = ch
        self.d1 = nn.Sequential(nn.Conv3d(1, c1, 3, stride=2, padding=1, bias=False),
                                nn.GroupNorm(min(8, c1), c1), nn.GELU())
        self.d2 = nn.Sequential(nn.Conv3d(c1, c2, 3, stride=2, padding=1, bias=False),
                                nn.GroupNorm(min(8, c2), c2), nn.GELU())
        self.d3 = nn.Sequential(nn.Conv3d(c2, c3, 3, stride=2, padding=1, bias=False),
                                nn.GroupNorm(min(8, c3), c3), nn.GELU())

    def forward(self, x, use_stem: bool = True):
        if not use_stem:
            return None, None, None
        f1 = self.d1(x)      # 64^3
        f2 = self.d2(f1)     # 32^3
        f3 = self.d3(f2)     # 16^3
        return f1, f2, f3


class AdapterFusion3D(nn.Module):
    """SACM's softmax-over-depth fusion, applied at the 8^3 level."""

    def __init__(self, adapter_dim: int = 768, out_ch: int = 64,
                 extra_ch: int = 384) -> None:
        super().__init__()
        self.level_norm = nn.LayerNorm(adapter_dim)
        self.level_score = nn.Sequential(nn.Linear(adapter_dim, adapter_dim // 4),
                                         nn.GELU(), nn.Linear(adapter_dim // 4, 1))
        self.proj_adapters = nn.Conv3d(adapter_dim, out_ch, 1)
        self.proj_neck = nn.Conv3d(extra_ch, out_ch, 1)

    def forward(self, neck: torch.Tensor, feats: List[torch.Tensor]) -> torch.Tensor:
        y = self.proj_neck(neck)
        if feats:
            stacked = torch.stack(feats, dim=0)                       # L,B,C,D,H,W
            pooled = stacked.mean(dim=(3, 4, 5))
            w = F.softmax(self.level_score(self.level_norm(pooled)).squeeze(-1), dim=0)
            agg = (stacked * w.view(w.shape[0], w.shape[1], 1, 1, 1, 1)).sum(dim=0)
            y = y + self.proj_adapters(agg)
        return y


class ConvDecoder3D(nn.Module):
    def __init__(self, adapter_dim: int = 768, neck_ch: int = 384,
                 stem_ch: Tuple[int, int, int] = (16, 32, 64),
                 use_stem: bool = True) -> None:
        super().__init__()
        self.use_stem = use_stem
        c1, c2, c3 = stem_ch
        self.fusion = AdapterFusion3D(adapter_dim, out_ch=64, extra_ch=neck_ch)

        # 8^3 -> 16^3
        self.up1 = nn.ConvTranspose3d(64, 64, 2, stride=2)
        self.blk1 = ConvBlock3D(64 + (c3 if use_stem else 0), 64)
        # 16^3 -> 32^3
        self.up2 = nn.ConvTranspose3d(64, 48, 2, stride=2)
        self.blk2 = ConvBlock3D(48 + (c2 if use_stem else 0), 48)
        # 32^3 -> 64^3
        self.up3 = nn.ConvTranspose3d(48, 32, 2, stride=2)
        self.blk3 = ConvBlock3D(32 + (c1 if use_stem else 0), 32)
        # 64^3 -> 128^3
        self.up4 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.blk4 = ConvBlock3D(16, 16)
        self.head = nn.Conv3d(16, 1, 1)

        self.stem = ConvStem3D(stem_ch) if use_stem else None

    def forward(self, neck: torch.Tensor, feats: List[torch.Tensor],
                image: torch.Tensor) -> torch.Tensor:
        pre = self.up1(self.fusion(neck, feats))           # 16^3
        return self.decode_from_16(pre, image)

    def decode_from_16(self, pre: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Rest of the decoder, from a [B,64,16,16,16] feature volcume."""
        s1 = s2 = s3 = None
        if self.stem is not None:
            s1, s2, s3 = self.stem(image, self.use_stem)

        x = pre                                            # 16^3
        if s3 is not None:
            x = torch.cat([x, s3], dim=1)
        x = self.blk1(x)

        x = self.up2(x)                                    # 32^3
        if s2 is not None:
            x = torch.cat([x, s2], dim=1)
        x = self.blk2(x)

        x = self.up3(x)                                    # 64^3
        if s1 is not None:
            x = torch.cat([x, s1], dim=1)
        x = self.blk3(x)

        x = self.up4(x)                                    # 128^3
        x = self.blk4(x)
        return self.head(x)


class Sam3D_ConvDec(nn.Module):
    """ViT encoder (frozen) + SACM adapters + 3D conv decoder."""

    def __init__(self, image_encoder: ImageEncoderViT3D_DLA, decoder: ConvDecoder3D,
                 neck_ch: int = 384) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.decoder = decoder
        self.neck_ch = neck_ch

    def set_dla(self, enabled: bool) -> None:
        self.image_encoder.dla_enabled = bool(enabled)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        neck, feats = self.image_encoder(image)             # [B,384,8,8,8], list
        return self.decoder(neck, feats, image)


def build_sam3d_convdec(checkpoint: str | None = None, *, adapter_ratio: float = 0.25,
                        ext_levels: Sequence[int] = (0, 3, 6, 9, 11),
                        use_stem: bool = True, img_size: int = 128,
                        embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                        patch_size: int = 16, out_chans: int = 384,
                        window_size: int = 14,
                        global_attn_indexes=(2, 5, 8, 11)) -> Sam3D_ConvDec:
    from functools import partial
    from .model import ImageEncoderViT3D_DLA
    enc = ImageEncoderViT3D_DLA(
        depth=depth, embed_dim=embed_dim, img_size=img_size, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_heads=num_heads,
        patch_size=patch_size, qkv_bias=True, use_rel_pos=True,
        global_attn_indexes=tuple(global_attn_indexes), window_size=window_size,
        out_chans=out_chans, adapter_ratio=adapter_ratio, ext_levels=ext_levels)
    dec = ConvDecoder3D(adapter_dim=embed_dim, neck_ch=out_chans, use_stem=use_stem)
    model = Sam3D_ConvDec(enc, dec, neck_ch=out_chans)

    if checkpoint is not None:
        ck = torch.load(checkpoint, map_location="cpu")
        if isinstance(ck, dict) and "model_state_dict" in ck:
            ck = ck["model_state_dict"]
        enc_sd = {k[len("image_encoder."):]: v for k, v in ck.items()
                  if k.startswith("image_encoder.")}
        missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
        new = ("internal_adapters", "external_adapters", "external_norms")
        bad = [k for k in missing if not any(n in k for n in new)]
        print(f"[build-conv] encoder tensors={len(enc_sd)} missing(new)={len(missing)} "
              f"unexpected={len(unexpected)}")
        assert not bad, f"unexpected missing encoder keys: {bad[:6]}"
    return model


def trainable_conv(model: Sam3D_ConvDec, *, adapters: bool = True,
                   full: bool = False) -> Tuple[List[nn.Parameter], int, int]:
    if full:
        params = list(model.parameters())
        for p in params:
            p.requires_grad = True
        n = sum(p.numel() for p in params)
        return params, n, n
    for p in model.parameters():
        p.requires_grad = False
    params = []
    for name, p in model.named_parameters():
        keep = name.startswith("decoder.")
        if adapters and ("internal_adapters" in name or "external_adapters" in name
                         or "external_norms" in name):
            keep = True
        if keep:
            p.requires_grad = True
            params.append(p)
    total = sum(p.numel() for p in model.parameters())
    return params, sum(p.numel() for p in params), total
