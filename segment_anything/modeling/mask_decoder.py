# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Dict, List, Tuple, Type

from .common import LayerNorm2d


class ResBlock(nn.Module):
    def __init__(self, ch: int, act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = LayerNorm2d(ch)
        self.a1 = act()
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = LayerNorm2d(ch)
        self.a2 = act()

    def forward(self, x):
        return x + self.a2(self.n2(self.c2(self.a1(self.n1(self.c1(x))))))


class BranchDecoder(nn.Module):
    """Independent upsampling decoder path for one geometry branch."""

    def __init__(self, in_ch: int, out_ch: int, hidden: int = 32,
                 act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 4, 2, 2), LayerNorm2d(in_ch // 4), act(),
            nn.ConvTranspose2d(in_ch // 4, hidden, 2, 2), LayerNorm2d(hidden), act(),
            nn.Conv2d(hidden, hidden, 3, padding=1), act(),
            nn.Conv2d(hidden, hidden, 3, padding=1), act(),
            nn.Conv2d(hidden, out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


class DetailStem(nn.Module):
    """High-frequency image stem producing features at 1/4 (256) and 1/2 (512).

    The ViT embedding lives at 1/16 (64). Thin curves are lost there, so these
    raw-image features give the decoder the bandwidth it needs.
    """

    def __init__(self, in_ch: int = 3, width: int = 32, out256: int = 64,
                 out512: int = 32, act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.d2 = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1),
            LayerNorm2d(width), act(),
        )
        self.d4 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            LayerNorm2d(width * 2), act(),
        )
        self.r256 = nn.Sequential(
            ResBlock(width * 2, act),
            ResBlock(width * 2, act),
            nn.Conv2d(width * 2, out256, 3, padding=1),
            LayerNorm2d(out256), act(),
        )
        self.p512 = nn.Sequential(
            nn.Conv2d(width, out512, 3, padding=1),
            LayerNorm2d(out512), act(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.d2(x)          # 1/2
        b = self.d4(a)          # 1/4
        return self.r256(b), self.p512(a)


class MaskDecoder(nn.Module):
    """Coarse-to-fine, geometry-aware curve decoder.

    * region head   : the original SAM hypernetwork head (coarse mask @ 256)
    * skeleton/width: auxiliary geometry heads @ 256
    * refinement    : residual high-resolution decoder driven by the raw-image
                      DetailStem (512 features) -> refined mask @ 512
    The refinement is zero-initialised, so at init the final mask equals the
    coarse mask (never worse); it learns to add high-frequency detail.
    """

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
        curve_head_dim: int = 64,
        use_detail: bool = True,
        detail_dim: int = 64,
        detail512_dim: int = 32,
        detach_aux: bool = True,
        use_cascade: bool = True,
        refine_size: int = 512,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.use_adapter_skip = use_adapter_skip
        self.use_curve_heads = use_curve_heads
        self.use_detail = use_detail
        self.detach_aux = detach_aux
        self.use_cascade = use_cascade
        self.refine_size = refine_size

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
             for i in range(self.num_mask_tokens)]
        )
        self.output_hypernetworks_mlps_stage2 = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for i in range(self.num_mask_tokens)]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
        self.gating_mlp = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        if self.use_adapter_skip:
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

        if self.use_detail:
            self.detail_stem = DetailStem(3, width=32, out256=detail_dim,
                                          out512=detail512_dim, act=activation)
            self.detail_proj = nn.Sequential(
                nn.Conv2d(detail_dim, feat_dim, 1), LayerNorm2d(feat_dim), activation(),
            )
            nn.init.zeros_(self.detail_proj[0].weight)
            nn.init.zeros_(self.detail_proj[0].bias)

        if self.use_curve_heads:
            # true multi-branch: each geometry quantity gets its OWN decoder path
            self.branch_center = BranchDecoder(transformer_dim, 1, curve_head_dim, activation)
            self.branch_width = nn.Sequential(
                BranchDecoder(transformer_dim, 1, curve_head_dim, activation), nn.Softplus(),
            )
            # geometry reconstruction: [centerline, width] -> mask correction
            self.geo_fuse = self._make_conv_head(2, curve_head_dim, 1)
            self.fusion_head = self._make_conv_head(feat_dim + 4, curve_head_dim, 1)
            nn.init.zeros_(self.fusion_head[-1].weight)
            nn.init.zeros_(self.fusion_head[-1].bias)

        if self.use_cascade:
            # high-resolution residual refiner
            self.detail512_proj = nn.Sequential(
                nn.Conv2d(detail512_dim, feat_dim, 1), LayerNorm2d(feat_dim), activation(),
            )
            self.refiner = self._make_conv_head(feat_dim + 1, curve_head_dim, 1)
            nn.init.zeros_(self.refiner[-1].weight)
            nn.init.zeros_(self.refiner[-1].bias)

    @staticmethod
    def _make_curve_head(in_ch, hidden, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), LayerNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, out_ch, 1),
        )

    @staticmethod
    def _make_conv_head(in_ch, hidden, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), LayerNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, out_ch, 1),
        )

    def encode_detail(self, images: torch.Tensor):
        if not self.use_detail:
            return None, None
        return self.detail_stem(images)

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

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        if self.use_adapter_skip and adapter_features is not None:
            adapter_weights = []
            for adapter_feat in adapter_features:
                pooled_feat = adapter_feat.mean(dim=[1, 2])
                adapter_weights.append(self.adapter_attention(pooled_feat))
            adapter_weights = F.softmax(torch.stack(adapter_weights, dim=0), dim=0)
            weighted = torch.zeros_like(adapter_features[0])
            for i, (adapter_feat, weight) in enumerate(zip(adapter_features, adapter_weights)):
                weighted += adapter_feat * weight.unsqueeze(-1).unsqueeze(-1)
            processed = self.adapter_aggregator(weighted).permute(0, 3, 1, 2)
            if processed.shape[2:] != upscaled_embedding.shape[2:]:
                processed = F.interpolate(processed, size=upscaled_embedding.shape[2:],
                                          mode='bilinear', align_corners=False)
            upscaled_embedding = upscaled_embedding + processed

        detail256 = detail512 = None
        if self.use_detail and detail_features is not None:
            detail256, detail512 = detail_features
            d = self.detail_proj(detail256)
            if d.shape[2:] != upscaled_embedding.shape[2:]:
                d = F.interpolate(d, size=upscaled_embedding.shape[2:],
                                  mode='bilinear', align_corners=False)
            upscaled_embedding = upscaled_embedding + d

        # ---- region head (stage1 + stage2 hypernetworks) ----
        hyper1 = torch.stack([
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)], dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks_stage1 = (hyper1 @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        gating = F.softmax(masks_stage1.view(b, self.num_mask_tokens, -1).max(dim=-1).values, dim=1)
        sorted_indices = torch.argsort(gating, dim=1, descending=True)

        hyper2 = torch.stack([
            self.output_hypernetworks_mlps_stage2[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)], dim=1)
        masks_stage2 = (hyper2 @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        gather = sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        masks_ordered = torch.gather(masks_stage2, 1, gather)

        iou_pred = self.iou_prediction_head(iou_token_out)
        iou_ordered = torch.gather(iou_pred, 1, sorted_indices)

        out: Dict[str, torch.Tensor] = {"region": masks_ordered, "iou_pred": iou_ordered}
        out["skeleton"] = None
        out["width"] = None
        out["coarse"] = None
        out["masks"] = masks_ordered[:, 0:1]

        if not self.use_curve_heads:
            return out

        feat_aux = src.detach() if self.detach_aux else src
        center_logits = self.branch_center(feat_aux)
        width_pred = self.branch_width(feat_aux)
        geo = self.geo_fuse(torch.cat([center_logits, width_pred], dim=1))
        region_primary = masks_ordered[:, 0:1]
        coarse = region_primary + self.fusion_head(
            torch.cat([upscaled_embedding, region_primary, center_logits, width_pred, geo], dim=1)
        )
        out["skeleton"] = center_logits
        out["width"] = width_pred
        out["coarse"] = coarse
        out["masks"] = coarse

        if self.use_cascade and detail512 is not None:
            coarse_hi = F.interpolate(coarse, size=(self.refine_size, self.refine_size),
                                      mode='bilinear', align_corners=False)
            d512 = self.detail512_proj(detail512)
            if d512.shape[2:] != coarse_hi.shape[2:]:
                d512 = F.interpolate(d512, size=coarse_hi.shape[2:],
                                     mode='bilinear', align_corners=False)
            delta = self.refiner(torch.cat([d512, coarse_hi], dim=1))
            out["masks"] = coarse_hi + delta
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
