"""Native-3D attribute decoder: SAM-Med3D ViT + dual-level adapters + Conv3d
centreline / radius / boundary-SDF heads.

The mask is produced at 128^3.  Attribute heads sit on the same 128^3 feature
that feeds the mask head, so attributes and mask share a resolution (as in 2D).
The mask becomes ``region + learned residual`` whose inputs include the soft 3D
distance to the predicted centreline and an analytic tube render; both residual
paths are zero-initialised, so at step 0 the decoder is exactly the plain Conv3d
decoder.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convdec import ConvBlock3D, ConvDecoder3D, ConvStem3D, Sam3D_ConvDec


def soft_distance_3d(center_logits: torch.Tensor, iters: int = 6) -> torch.Tensor:
    p = torch.sigmoid(center_logits)
    dist = torch.zeros_like(p)
    covered = p.clamp(0.0, 1.0)
    cur = p
    for i in range(1, iters + 1):
        nxt = F.max_pool3d(cur, 3, 1, 1)
        newly = (nxt - cur).clamp(min=0.0)
        dist = dist + newly * float(i)
        covered = (covered + newly).clamp(max=1.0)
        cur = nxt
    cov = (covered > 0.5).to(dist.dtype)
    return dist * cov + (1.0 - cov) * float(iters)


class ConvDecoder3DAttr(ConvDecoder3D):
    def __init__(self, *args, attr_iters: int = 6, tau: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attr_iters = attr_iters
        self.tau = tau
        c = 16  # blk4 output channels
        self.attr_stem = ConvBlock3D(c, c)
        self.branch_center = nn.Conv3d(c, 1, 1)
        self.branch_width = nn.Conv3d(c, 1, 1)
        self.branch_sdf = nn.Sequential(ConvBlock3D(c + 4, c), nn.Conv3d(c, 1, 1))
        self.geo_fuse = nn.Conv3d(4, c, 1)
        self.attr_fuse = nn.Conv3d(2 * c + 4, c, 1)
        for m in (self.geo_fuse, self.attr_fuse):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def decode_from_16(self, pre: torch.Tensor, image: torch.Tensor):
        s1 = s2 = s3 = None
        if self.stem is not None:
            s1, s2, s3 = self.stem(image, self.use_stem)

        x = pre
        if s3 is not None:
            x = torch.cat([x, s3], dim=1)
        x = self.blk1(x)

        x = self.up2(x)
        if s2 is not None:
            x = torch.cat([x, s2], dim=1)
        x = self.blk2(x)

        x = self.up3(x)
        if s1 is not None:
            x = torch.cat([x, s1], dim=1)
        x = self.blk3(x)

        x = self.up4(x)          # 128^3
        x = self.blk4(x)         # 128^3, 16 ch

        feat = self.attr_stem(x)
        center = self.branch_center(feat)
        width = self.branch_width(feat)              # log1p(radius in voxels)
        radius = F.softplus(width)
        dist = soft_distance_3d(center, self.attr_iters)
        render = torch.sigmoid((radius - dist) / self.tau)

        geo = self.geo_fuse(torch.cat([center, width, dist, render], 1))
        sdf = self.branch_sdf(torch.cat([feat, center, width, dist, render], 1))
        corr = self.attr_fuse(torch.cat([feat, center, width, dist, render, geo], 1))
        x = x + corr

        return {"masks": self.head(x), "skeleton": center,
                "width": width, "sdf": sdf}


class Sam3D_ConvDecAttr(Sam3D_ConvDec):
    def forward(self, image: torch.Tensor):
        neck, feats = self.image_encoder(image)
        return self.decoder(neck, feats, image)


def build_sam3d_convdec_attr(checkpoint: str | None = None, *, adapter_ratio: float = 0.25,
                             ext_levels: Sequence[int] = (0, 3, 6, 9, 11),
                             use_stem: bool = True, img_size: int = 128,
                             embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                             patch_size: int = 16, out_chans: int = 384,
                             window_size: int = 14,
                             global_attn_indexes=(2, 5, 8, 11)) -> Sam3D_ConvDecAttr:
    from functools import partial

    from .model import ImageEncoderViT3D_DLA
    enc = ImageEncoderViT3D_DLA(
        depth=depth, embed_dim=embed_dim, img_size=img_size, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_heads=num_heads,
        patch_size=patch_size, qkv_bias=True, use_rel_pos=True,
        global_attn_indexes=tuple(global_attn_indexes), window_size=window_size,
        out_chans=out_chans, adapter_ratio=adapter_ratio, ext_levels=ext_levels)
    dec = ConvDecoder3DAttr(adapter_dim=embed_dim, neck_ch=out_chans, use_stem=use_stem)
    model = Sam3D_ConvDecAttr(enc, dec, neck_ch=out_chans)

    if checkpoint is not None:
        ck = torch.load(checkpoint, map_location="cpu")
        if isinstance(ck, dict) and "model_state_dict" in ck:
            ck = ck["model_state_dict"]
        enc_sd = {k[len("image_encoder."):]: v for k, v in ck.items()
                  if k.startswith("image_encoder.")}
        missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
        new = ("internal_adapters", "external_adapters", "external_norms")
        bad = [k for k in missing if not any(n in k for n in new)]
        print(f"[build-attr] encoder tensors={len(enc_sd)} missing(new)={len(missing)} "
              f"unexpected={len(unexpected)}")
        assert not bad, f"unexpected missing encoder keys: {bad[:6]}"
    return model
