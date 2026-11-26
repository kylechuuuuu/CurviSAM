"""Dual-Level Adapter (DLA) for SAM-Med3D.

Port of the 2D "dual level adapter" design from SACM (kylechuuuuu/SACM,
`segment_anything/modeling/image_encoder.py` + `mask_decoder.py`) to the 3D
SAM-Med3D ViT backbone.

Level 1 -- *internal* adapters: one bottleneck residual adapter after every
           transformer block (tokens are [B, D, H, W, C]).
Level 2 -- *external* adapters: a parallel multi-depth chain.  At a few block
           depths an external adapter produces a residual feature that is both
           (a) added back to the main stream and (b) collected.  The mask
           decoder fuses the collected features with learned, softmax-normalised
           depth weights and adds the result to its up-sampled embedding.

Both levels are zero-initialised (the last linear / conv of every adapter is
zero), so at step 0 the network is *exactly* the pretrained SAM-Med3D.

Upstream code is imported from a pristine SAM-Med3D checkout; this module only
adds modules around it (no upstream file is modified).
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from functools import partial
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.environ.get("SAMMED3D_REPO")
if _REPO is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.join(os.path.dirname(_here), "third_party", "SAM-Med3D")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from segment_anything.modeling.image_encoder3D import (  # noqa: E402
    ImageEncoderViT3D,
    LayerNorm3d,
)
from segment_anything.modeling.mask_decoder3D import MaskDecoder3D  # noqa: E402
from segment_anything.modeling.prompt_encoder3D import PromptEncoder3D  # noqa: E402
from segment_anything.modeling.sam3D import Sam3D  # noqa: E402


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
class Adapter3D(nn.Module):
    """Bottleneck residual adapter on ViT tokens ([B, D, H, W, C]).

    Returns the *delta* only (caller adds it).  Last layer is zero-initialised.
    """

    def __init__(self, dim: int, ratio: float = 0.25, act: nn.Module = nn.GELU) -> None:
        super().__init__()
        hidden = max(8, int(round(dim * ratio)))
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.act = act()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(x))))


class AdapterSkipFusion3D(nn.Module):
    """Level-2 fusion inside the mask decoder (SACM `adapter_skip` port).

    `pending` holds the list of external adapter features
    ([B, adapter_dim, d, h, w]); the module is appended to `output_upscaling`
    so it is applied at exactly the right spatial resolution.
    """

    def __init__(self, adapter_dim: int, in_ch: int, inner: int | None = None,
                 num_levels: int = 5, act: nn.Module = nn.GELU) -> None:
        super().__init__()
        inner = inner if inner is not None else in_ch
        # NOTE: level weights operate on *pooled vectors* -> plain LayerNorm,
        # while the feature aggregation is channel-first -> LayerNorm3d.
        self.level_norm = nn.LayerNorm(adapter_dim)
        self.level_score = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim // 4), act(), nn.Linear(adapter_dim // 4, 1)
        )
        self.reduce = nn.Sequential(
            nn.Conv3d(adapter_dim, inner, kernel_size=1, bias=False),
            LayerNorm3d(inner),
            act(),
            nn.Conv3d(inner, in_ch, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.reduce[-1].weight)
        nn.init.zeros_(self.reduce[-1].bias)
        self.pending: List[torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.pending
        self.pending = None
        if not feats:
            return x
        stacked = torch.stack(feats, dim=0)                       # L,B,C,D,H,W
        pooled = stacked.mean(dim=(3, 4, 5))                      # L,B,C
        w = F.softmax(self.level_score(self.level_norm(pooled)).squeeze(-1), dim=0)  # L,B
        agg = (stacked * w.view(w.shape[0], w.shape[1], 1, 1, 1, 1)).sum(dim=0)     # B,C,D,H,W
        y = self.reduce(agg)
        if y.shape[2:] != x.shape[2:]:
            y = F.interpolate(y, size=x.shape[2:], mode="trilinear", align_corners=False)
        assert y.shape == x.shape, f"adapter-fusion shape mismatch {y.shape} vs {x.shape}"
        return x + y


# --------------------------------------------------------------------------- #
# encoder / decoder
# --------------------------------------------------------------------------- #
class ImageEncoderViT3D_DLA(ImageEncoderViT3D):
    """SAM-Med3D image encoder + internal adapters + external adapter chain."""

    def __init__(self, *, adapter_ratio: float = 0.25,
                 ext_levels: Sequence[int] = (0, 3, 6, 9, 11), **kwargs) -> None:
        super().__init__(**kwargs)
        dim, depth = kwargs["embed_dim"], kwargs["depth"]
        self.dla_enabled = True
        self.ext_levels = tuple(int(i) for i in ext_levels)
        # level 1: internal, every block
        self.internal_adapters = nn.ModuleList([Adapter3D(dim, adapter_ratio) for _ in range(depth)])
        # level 2: external, selected depths
        self.external_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.ext_levels])
        self.external_adapters = nn.ModuleList([Adapter3D(dim, adapter_ratio) for _ in self.ext_levels])

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        feats: List[torch.Tensor] = []
        ext = {lvl: k for k, lvl in enumerate(self.ext_levels)}
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.dla_enabled:
                x = x + self.internal_adapters[i](x)
                if i in ext:
                    k = ext[i]
                    delta = self.external_adapters[k](self.external_norms[k](x))
                    x = x + delta
                    feats.append(delta.permute(0, 4, 1, 2, 3))  # B,C,D,H,W
        x = self.neck(x.permute(0, 4, 1, 2, 3))
        return x, feats


class MaskDecoder3D_DLA(MaskDecoder3D):
    """SAM-Med3D mask decoder that consumes the level-2 features."""

    def __init__(self, *, adapter_dim: int = 768, num_levels: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        inner = kwargs["transformer_dim"] // 4
        in_ch = kwargs["transformer_dim"] // 8
        self.fusion = AdapterSkipFusion3D(adapter_dim, in_ch, inner=inner,
                                          num_levels=num_levels)
        # append the fusion to the up-sampling stack under a *named* key so that
        # its parameters are recognisable as "new" when loading the checkpoint
        named = OrderedDict((str(i), m) for i, m in enumerate(self.output_upscaling))
        named["adapter_fusion"] = self.fusion
        self.output_upscaling = nn.Sequential(named)

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                      dense_prompt_embeddings, adapter_features=None):
        self.fusion.pending = adapter_features  # consumed inside output_upscaling
        return super().predict_masks(image_embeddings, image_pe,
                                     sparse_prompt_embeddings, dense_prompt_embeddings)

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output, adapter_features=None):
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            adapter_features=adapter_features,
        )
        sl = slice(1, None) if multimask_output else slice(0, 1)
        return masks[:, sl], iou_pred[:, sl]


class Sam3D_DLA(Sam3D):
    """Thin wrapper so callers can do `emb, feats = model.image_encoder(x)`."""

    def set_dla(self, enabled: bool) -> None:
        self.image_encoder.dla_enabled = bool(enabled)

    @torch.no_grad()
    def predict_logits(self, image: torch.Tensor, point_coords: torch.Tensor,
                       point_labels: torch.Tensor) -> torch.Tensor:
        """image: [1,1,D,H,W] (already normalised), points: [1,N,3] / [1,N]."""
        self.eval()
        emb, feats = self.image_encoder(image)
        sparse, dense = self.prompt_encoder(
            points=[point_coords, point_labels], boxes=None, masks=None
        )
        low_res, _ = self.mask_decoder(
            image_embeddings=emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            adapter_features=feats,
        )
        return low_res


# --------------------------------------------------------------------------- #
# builder -- mirrors SAM-Med3D `_build_sam3D_ori` (the released turbo config)
# --------------------------------------------------------------------------- #
def build_sam3d_dla(
    checkpoint: str | None = None,
    *,
    img_size: int = 128,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    patch_size: int = 16,
    out_chans: int = 384,
    window_size: int = 14,
    global_attn_indexes: Iterable[int] = (2, 5, 8, 11),
    adapter_ratio: float = 0.25,
    ext_levels: Sequence[int] = (0, 3, 6, 9, 11),
    verbose: bool = True,
) -> Sam3D_DLA:
    prompt_embed_dim = out_chans
    image_embedding_size = img_size // patch_size
    model = Sam3D_DLA(
        image_encoder=ImageEncoderViT3D_DLA(
            depth=depth,
            embed_dim=embed_dim,
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=num_heads,
            patch_size=patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=tuple(global_attn_indexes),
            window_size=window_size,
            out_chans=out_chans,
            adapter_ratio=adapter_ratio,
            ext_levels=ext_levels,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size,) * 3,
            input_image_size=(img_size,) * 3,
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3D_DLA(
            transformer_dim=prompt_embed_dim,
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            adapter_dim=embed_dim,
            num_levels=len(ext_levels),
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if checkpoint is not None:
        ck = torch.load(checkpoint, map_location="cpu")
        if isinstance(ck, dict) and "model_state_dict" in ck:
            ck = ck["model_state_dict"]
        missing, unexpected = model.load_state_dict(ck, strict=False)
        new = ("internal_adapters", "external_adapters", "external_norms", "fusion")
        bad_missing = [k for k in missing if not any(n in k for n in new)]
        bad_unexpected = [k for k in unexpected if not any(n in k for n in new)]
        if verbose:
            print(f"[build] loaded {len(ck)} tensors | missing(new)={len(missing)} "
                  f"unexpected={len(unexpected)}")
            if bad_missing:
                print("[build] !! unexpected missing:", bad_missing[:8])
            if bad_unexpected:
                print("[build] !! unexpected ckpt keys:", bad_unexpected[:8])
        assert not bad_missing and not bad_unexpected, "checkpoint / architecture mismatch"
    return model


def trainable_parameters(model: nn.Module, *, adapters: bool = True,
                         decoder: bool = True, full: bool = False
                         ) -> Tuple[List[nn.Parameter], int, int]:
    """Freeze everything, then unfreeze adapters/decoder (or everything with `full`)."""
    if full:
        params = [p for p in model.parameters()]
        total = sum(p.numel() for p in params)
        for p in params:
            p.requires_grad = True
        return params, total, total

    for p in model.parameters():
        p.requires_grad = False

    def _on(name: str) -> bool:
        if adapters and ("internal_adapters" in name or "external_adapters" in name
                         or "external_norms" in name or ".fusion." in name):
            return True
        if decoder and name.startswith("mask_decoder"):
            return True
        return False

    params = []
    total = trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if _on(name):
            p.requires_grad = True
            params.append(p)
            trainable += p.numel()
    return params, trainable, total


@torch.no_grad()
def verify_identity(model: Sam3D_DLA, x: torch.Tensor) -> float:
    """Adapter delta must be ~0 at init: max |DLA(feat) - base(feat)| on the neck."""
    model.eval()
    out_with, _ = model.image_encoder(x)
    model.image_encoder.dla_enabled = False
    out_base, _ = model.image_encoder(x)
    model.image_encoder.dla_enabled = True
    return float((out_with - out_base).abs().max())
