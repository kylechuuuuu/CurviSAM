# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Mask decoder v17 - attribute-faithful, resolution-decoupled curve decoder.

What changed vs v16
-------------------
1. **Resolution decoupling.** The attribute heads (centerline / radius / boundary
   SDF) and the final mask now live at 512, driven by the raw-image DetailStem.
   v16 decoded attributes at 256 and its 512 refiner was never executed, so thin
   curves were sub-pixel at the prediction grid.
2. **Attribute triple.** Besides centerline and radius, the decoder regresses a
   truncated **signed boundary distance** (SDF). All three attributes are
   domain-invariant and derived from the mask itself, so they act as an inductive
   bias for unseen domains rather than as a dataset-specific trick.
3. **Explicit geometric rendering.** A differentiable soft distance field to the
   predicted centerline is combined with the predicted radius into an analytic
   tube render `sigmoid((r - dist) / tau)`, which is given to the fusion head.
   The mask therefore is *a rendering of decoded attributes* plus a learned,
   zero-initialised residual instead of a free-form conv on features.
4. **IBN normalisation** (InstanceNorm + LayerNorm) in the raw-image stem and the
   attribute fusion path, to survive unknown appearance statistics.
5. Every residual (geo_fuse / fusion_head / refiner) is **zero-initialised**, so
   at init the output equals the region head: the model can never start worse.
"""

from typing import Dict, List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d

ATTR_SIZE = 512


# --------------------------------------------------------------------------- #
# normalisation / blocks
# --------------------------------------------------------------------------- #
class IBNorm2d(nn.Module):
    """InstanceNorm (appearance-invariant) + LayerNorm (content) concatenated.

    IBN-Net style: the instance branch strips per-image contrast/scale
    statistics, which is precisely the nuisance factor when the target domain
    is unknown.
    """

    def __init__(self, ch: int, use_ibn: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.use_ibn = use_ibn
        self.ln = LayerNorm2d(ch, eps)
        if use_ibn:
            self.in_norm = nn.InstanceNorm2d(ch, affine=True, eps=eps)
            self.proj = nn.Conv2d(ch * 2, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_ibn:
            return self.ln(x)
        return self.proj(torch.cat([self.in_norm(x), self.ln(x)], dim=1))


class ResBlock(nn.Module):
    def __init__(self, ch: int, act: Type[nn.Module] = nn.GELU, ibnorm: bool = False) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = IBNorm2d(ch, ibnorm)
        self.a1 = act()
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = IBNorm2d(ch, ibnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.n2(self.c2(self.a1(self.n1(self.c1(x)))))


class UpBlock(nn.Module):
    """resize-conv: bilinear x2 upsample + conv stack (no transposed-conv checkerboard)."""

    def __init__(self, in_ch: int, out_ch: int, act: Type[nn.Module] = nn.GELU,
                 ibnorm: bool = False) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n1 = IBNorm2d(out_ch, ibnorm)
        self.a1 = act()
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n2 = IBNorm2d(out_ch, ibnorm)
        self.a2 = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.a2(self.n2(self.c2(self.a1(self.n1(self.c1(x))))))


class ConvHead(nn.Module):
    """conv3x3 stack + 1x1 projection; optionally zero-initialised (identity res.)."""

    def __init__(self, in_ch: int, hidden: int, out_ch: int = 1, depth: int = 2,
                 act: Type[nn.Module] = nn.GELU, ibnorm: bool = False,
                 zero_init: bool = False) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        c_in = in_ch
        for _ in range(depth):
            layers += [nn.Conv2d(c_in, hidden, 3, padding=1), IBNorm2d(hidden, ibnorm), act()]
            c_in = hidden
        layers.append(nn.Conv2d(c_in, out_ch, 1))
        self.net = nn.Sequential(*layers)
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DetailStem(nn.Module):
    """High-frequency image stem: raw-image features at 1/4 (256) and 1/2 (512).

    The ViT embedding lives at 1/16 (64): thin curves are already lost there, so
    the decoder is given the raw-image bandwidth it needs to decode them.
    """

    def __init__(self, in_ch: int = 3, width: int = 32, out256: int = 64,
                 out512: int = 32, act: Type[nn.Module] = nn.GELU,
                 ibnorm: bool = True) -> None:
        super().__init__()
        self.d2 = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1),
            IBNorm2d(width, ibnorm), act(),
        )
        self.d4 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            IBNorm2d(width * 2, ibnorm), act(),
        )
        self.r256 = nn.Sequential(
            ResBlock(width * 2, act, ibnorm),
            ResBlock(width * 2, act, ibnorm),
            nn.Conv2d(width * 2, out256, 3, padding=1),
            IBNorm2d(out256, ibnorm), act(),
        )
        self.p512 = nn.Sequential(
            nn.Conv2d(width, out512, 3, padding=1),
            IBNorm2d(out512, ibnorm), act(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.d2(x)          # 1/2
        b = self.d4(a)          # 1/4
        return self.r256(b), self.p512(a)


def soft_distance(center_logits: torch.Tensor, iters: int = 16) -> torch.Tensor:
    """Differentiable approximation of the (chessboard) distance to the centerline.

    Iterated max-pooling dilations accumulate the arrival step index, so a pixel
    covered at dilation step ``i`` gets distance ``i``. Uncovered pixels saturate
    at ``iters``. This is a *non-local* geometric quantity that a stack of 3x3
    convs cannot compute, which is why it is handed to the fusion head.
    """
    p = torch.sigmoid(center_logits)
    dist = torch.zeros_like(p)
    covered = p.clamp(0.0, 1.0)          # step-0 coverage: the centerline itself is d=0
    cur = p
    for i in range(1, iters + 1):
        nxt = F.max_pool2d(cur, 3, 1, 1)
        newly = (nxt - cur).clamp(min=0.0)
        dist = dist + newly * float(i)
        covered = (covered + newly).clamp(max=1.0)
        cur = nxt
    cov = (covered > 0.5).to(dist.dtype)  # hard indicator: gradient flows via `dist`
    return dist * cov + (1.0 - cov) * float(iters)


class MaskDecoder(nn.Module):
    """Coarse-to-fine, attribute-faithful curve decoder (v17)."""

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_adapter_skip: bool = True,
        adapter_embed_dim: int = 768,
        use_curve_heads: bool = True,
        curve_head_dim: int = 32,
        use_detail: bool = True,
        detail_dim: int = 64,
        detail512_dim: int = 32,
        attr_dim: int = 32,
        detach_aux: bool = True,
        use_sdf: bool = True,
        use_ibn: bool = True,
        max_radius: float = 16.0,
        tau: float = 0.5,
        learn_tau: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.use_adapter_skip = use_adapter_skip
        self.use_curve_heads = use_curve_heads
        self.use_detail = use_detail
        self.use_sdf = use_sdf
        self.use_ibn = use_ibn
        self.detach_aux = detach_aux
        self.max_radius = max_radius
        self.tau = tau
        self.learn_tau = learn_tau
        if learn_tau:
            # The render temperature sets the boundary softness, and softness
            # directly controls the predicted area. The right value is domain
            # dependent, so let the model learn it instead of fixing it.
            self.tau_raw = nn.Parameter(torch.tensor(-0.55))
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(self.num_mask_tokens)]
        )
        self.output_hypernetworks_mlps_stage2 = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(self.num_mask_tokens)]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        if self.use_adapter_skip:
            # adapter features are token grids (B,64,64,C): pool per layer, gate, aggregate
            self.adapter_aggregator = nn.Sequential(
                nn.LayerNorm(adapter_embed_dim),
                nn.Linear(adapter_embed_dim, transformer_dim // 4),
                activation(),
                nn.Linear(transformer_dim // 4, transformer_dim // 8),
            )
            self.adapter_attention = nn.Sequential(
                nn.LayerNorm(adapter_embed_dim),
                nn.Linear(adapter_embed_dim, 1),
                nn.Sigmoid(),
            )

        feat_dim = transformer_dim // 8  # 32
        self.feat_dim = feat_dim
        self.attr_dim = attr_dim

        if self.use_detail:
            self.detail_stem = DetailStem(3, width=32, out256=detail_dim,
                                          out512=detail512_dim, act=activation,
                                          ibnorm=use_ibn)
            self.detail_proj256 = nn.Sequential(
                nn.Conv2d(detail_dim, attr_dim, 1), IBNorm2d(attr_dim, use_ibn), activation(),
            )
            self.detail_proj512 = nn.Sequential(
                nn.Conv2d(detail512_dim, attr_dim, 1), IBNorm2d(attr_dim, use_ibn), activation(),
            )

        if self.use_curve_heads:
            # low-res ViT evidence for the attribute path (detached: aux losses must
            # not reshape the frozen trunk)
            self.src_proj = nn.Sequential(
                nn.Conv2d(transformer_dim, attr_dim, 1), IBNorm2d(attr_dim, use_ibn), activation(),
            )
            # attribute feature pyramid: 256 -> 512 (IBN only where it is cheap)
            f256_in = feat_dim + attr_dim + attr_dim
            self.attr_fuse256 = nn.Sequential(
                nn.Conv2d(f256_in, attr_dim, 3, padding=1), IBNorm2d(attr_dim, use_ibn), activation(),
                nn.Conv2d(attr_dim, attr_dim, 3, padding=1), IBNorm2d(attr_dim, use_ibn), activation(),
            )
            # 256 -> 512 fusion with the raw-image lateral (plain conv: the x2
            # upsample already happened, so no UpBlock here)
            self.attr_up512 = nn.Sequential(
                nn.Conv2d(attr_dim * 2, attr_dim, 3, padding=1), LayerNorm2d(attr_dim), activation(),
                nn.Conv2d(attr_dim, attr_dim, 3, padding=1), activation(),
            )

            # One shared, lean stem at 512 followed by thin attribute heads.
            # A 512^2 activation costs 4x a 256^2 one, so norm-concats / wide
            # hidden layers are deliberately avoided on this side of the decoder.
            self.head_stem = nn.Sequential(
                nn.Conv2d(attr_dim, curve_head_dim, 3, padding=1), activation(),
            )
            self.branch_center = nn.Conv2d(curve_head_dim, 1, 1)
            self.branch_width = nn.Conv2d(curve_head_dim, 1, 1)
            if self.use_sdf:
                self.branch_sdf = nn.Sequential(
                    nn.Conv2d(curve_head_dim + 3, curve_head_dim, 3, padding=1), activation(),
                    nn.Conv2d(curve_head_dim, 1, 1),
                )

            # --- geometric reconstruction: [center, log r, dist, render] -> correction
            self.geo_fuse = ConvHead(4, curve_head_dim, 1, 1, activation, False, zero_init=True)

            # --- fusion: explicit geometric evidence + features -> residual mask
            self.fusion_head = ConvHead(attr_dim + 6, curve_head_dim, 1, 1, activation,
                                        False, zero_init=True)

            # --- high-res refiner: detail + geometry -> residual mask
            ref_in = attr_dim + attr_dim + 3 + (1 if self.use_sdf else 0)
            self.refiner = ConvHead(ref_in, curve_head_dim, 1, 1, activation, False,
                                    zero_init=True)

    # ------------------------------------------------------------------ #
    @property
    def tau_value(self):
        if self.learn_tau:
            return 0.05 + F.softplus(self.tau_raw)
        return self.tau

    def encode_detail(self, images: torch.Tensor):
        if not self.use_detail:
            return None, None
        return self.detail_stem(images)

    def _attr_features(self, src, upscaled_embedding, detail256, detail512):
        """Build the 512-resolution attribute feature map."""
        b, c, h, w = src.shape
        parts = [upscaled_embedding.detach()]
        if self.use_detail and detail256 is not None:
            d256 = self.detail_proj256(detail256)
            if d256.shape[2:] != upscaled_embedding.shape[2:]:
                d256 = F.interpolate(d256, size=upscaled_embedding.shape[2:],
                                     mode="bilinear", align_corners=False)
        else:
            d256 = torch.zeros_like(upscaled_embedding)  # keep the channel layout fixed
        parts.append(d256)
        src_feat = self.src_proj(src)
        if src_feat.shape[2:] != upscaled_embedding.shape[2:]:
            src_feat = F.interpolate(src_feat, size=upscaled_embedding.shape[2:],
                                     mode="bilinear", align_corners=False)
        parts.append(src_feat)
        feat256 = self.attr_fuse256(torch.cat(parts, dim=1))

        # 256 -> 512: upsample the attribute feature, then fuse the raw-image
        # lateral at 512 (the lateral is already at 1/2 resolution)
        hi = F.interpolate(feat256, scale_factor=2, mode="bilinear", align_corners=False)
        if self.use_detail and detail512 is not None:
            d512 = self.detail_proj512(detail512)
            if d512.shape[2:] != hi.shape[2:]:
                d512 = F.interpolate(d512, size=hi.shape[2:], mode="bilinear",
                                     align_corners=False)
            hi = torch.cat([hi, d512], dim=1)
        else:
            hi = torch.cat([hi, torch.zeros_like(hi)], dim=1)
        return self.attr_up512(hi)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        adapter_features: List[torch.Tensor] = None,
        detail_features=None,
    ) -> Dict[str, torch.Tensor]:
        out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            adapter_features=adapter_features,
            detail_features=detail_features,
        )
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        out["region"] = out["region"][:, mask_slice, :, :]
        out["iou_pred"] = out["iou_pred"][:, mask_slice]
        return out

    # ------------------------------------------------------------------ #
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        adapter_features: List[torch.Tensor] = None,
        detail_features=None,
    ) -> Dict[str, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src_tok = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        src_tok = src_tok.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src_tok)

        if self.use_adapter_skip and adapter_features is not None:
            weights = []
            for feat in adapter_features:
                weights.append(self.adapter_attention(feat.mean(dim=[1, 2])))
            weights = F.softmax(torch.stack(weights, dim=0), dim=0)
            weighted = torch.zeros_like(adapter_features[0])
            for feat, wt in zip(adapter_features, weights):
                weighted = weighted + feat * wt.unsqueeze(-1).unsqueeze(-1)
            processed = self.adapter_aggregator(weighted).permute(0, 3, 1, 2)
            if processed.shape[2:] != upscaled_embedding.shape[2:]:
                processed = F.interpolate(processed, size=upscaled_embedding.shape[2:],
                                          mode="bilinear", align_corners=False)
            upscaled_embedding = upscaled_embedding + processed

        detail256 = detail512 = None
        if self.use_detail and detail_features is not None:
            detail256, detail512 = detail_features

        # ---- region head (SAM hypernetworks, stage1 gates stage2 ordering) ----
        hyper1 = torch.stack([
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)], dim=1)
        bb, cc, hh, ww = upscaled_embedding.shape
        masks_stage1 = (hyper1 @ upscaled_embedding.view(bb, cc, hh * ww)).view(bb, -1, hh, ww)
        gating = F.softmax(masks_stage1.view(bb, self.num_mask_tokens, -1).max(dim=-1).values, dim=1)
        sorted_indices = torch.argsort(gating, dim=1, descending=True)

        hyper2 = torch.stack([
            self.output_hypernetworks_mlps_stage2[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)], dim=1)
        masks_stage2 = (hyper2 @ upscaled_embedding.view(bb, cc, hh * ww)).view(bb, -1, hh, ww)
        gather = sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hh, ww)
        masks_ordered = torch.gather(masks_stage2, 1, gather)
        iou_pred = self.iou_prediction_head(iou_token_out)
        iou_ordered = torch.gather(iou_pred, 1, sorted_indices)

        out: Dict[str, torch.Tensor] = {
            "region": masks_ordered,
            "iou_pred": iou_ordered,
            "skeleton": None,
            "width": None,
            "sdf": None,
            "coarse": None,
            "masks": masks_ordered[:, 0:1],
        }
        if not self.use_curve_heads:
            return out

        # ---- attribute path (detached from the trunk) ----
        feat512 = self._attr_features(src_tok if not self.detach_aux else src_tok.detach(),
                                      upscaled_embedding, detail256, detail512)
        head_in = self.head_stem(feat512)                                 # shared lean stem
        center_logits = self.branch_center(head_in)                       # (B,1,512,512)
        width_log = self.branch_width(head_in)                            # log1p(px radius)
        radius_px = torch.expm1(width_log).clamp(min=0.0, max=self.max_radius)
        dist = soft_distance(center_logits, iters=int(self.max_radius))
        render = torch.sigmoid((radius_px - dist) / self.tau_value)

        geo = self.geo_fuse(torch.cat([center_logits, width_log, dist, render], dim=1))
        sdf_logits = None
        if self.use_sdf:
            sdf_logits = self.branch_sdf(
                torch.cat([head_in, center_logits, width_log, render], dim=1))

        # ---- mask assembly: region (upsampled) + learned geometric residual ----
        region_primary = masks_ordered[:, 0:1]
        region_up = F.interpolate(region_primary, size=feat512.shape[2:], mode="bilinear",
                                  align_corners=False)
        coarse = region_up + self.fusion_head(torch.cat(
            [feat512, region_up, center_logits, width_log, dist, render, geo], dim=1))

        ref_parts = [feat512, coarse, center_logits, dist]
        if self.use_detail and detail512 is not None:
            d512 = self.detail_proj512(detail512)
            if d512.shape[2:] != coarse.shape[2:]:
                d512 = F.interpolate(d512, size=coarse.shape[2:], mode="bilinear",
                                     align_corners=False)
            ref_parts.append(d512)
        else:
            ref_parts.append(torch.zeros_like(feat512))
        if self.use_sdf and sdf_logits is not None:
            ref_parts.append(sdf_logits)
        masks = coarse + self.refiner(torch.cat(ref_parts, dim=1))

        out["skeleton"] = center_logits
        out["width"] = width_log
        out["sdf"] = sdf_logits
        out["coarse"] = coarse
        out["masks"] = masks
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
