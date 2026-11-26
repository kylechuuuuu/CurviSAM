# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from typing import List, Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        use_adapter: bool = True,
        adapter_dim_ratio: float = 0.1,
        use_adaptive_scale: bool = False,
        adaptive_hidden: int = 32,
        adaptive_span: float = 1.0,
        adaptive_mode: str = "scale",
        adaptive_separate: bool = True,
        adaptive_scope: str = "both",
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
            use_adapter (bool): If True, use adapter modules in each block.
            adapter_dim_ratio (float): Ratio of adapter dim to embedding dim.
        """
        super().__init__()
        self.img_size = img_size
        self.use_adapter = use_adapter
        self.patch_size = patch_size
        self.depth = depth

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_adapter=use_adapter,
                adapter_dim_ratio=adapter_dim_ratio,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        
        # Initialize external adapter chain with improved fusion
        if use_adapter:
            # Create external adapters with improved design
            self.external_adapters = nn.ModuleList([
                AdapterModule(embed_dim, adapter_dim_ratio, act_layer) 
                for _ in range(depth)
            ])
            
            # Adapter fusion module with attention mechanism
            self.adapter_fusion = nn.Sequential(
                norm_layer(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                act_layer(),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )
            
            # Final layer norm for adapter chain
            self.adapter_norm = norm_layer(embed_dim)

            # Input-adaptive per-adapter modulation (TPAMI addition)
            self.use_adaptive_scale = use_adaptive_scale
            self.adaptive_mode = adaptive_mode
            self.adaptive_scope = adaptive_scope
            self.adaptive_scale = None
            self.adaptive_film = None
            if use_adaptive_scale:
                if adaptive_mode == "film":
                    self.adaptive_film = AdaptiveFiLM(
                        depth=depth,
                        embed_dim=embed_dim,
                        bottleneck_dim=int(embed_dim * adapter_dim_ratio),
                        hidden=adaptive_hidden,
                        span=adaptive_span,
                        separate_int_ext=adaptive_separate,
                    )
                else:
                    self.adaptive_scale = AdaptiveAdapterScale(
                        depth=depth,
                        embed_dim=embed_dim,
                        hidden=adaptive_hidden,
                        separate_int_ext=adaptive_separate,
                        span=adaptive_span,
                        mode=adaptive_mode,
                    )
        else:
            self.use_adaptive_scale = False
            self.adaptive_mode = adaptive_mode
            self.adaptive_scope = adaptive_scope
            self.adaptive_scale = None
            self.adaptive_film = None

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Initialize containers
        intermediate_features = []
        adapter_features = []  # 收集所有external adapter的输出
            
        # Process through blocks
        scale_fn = self.adaptive_scale if (self.use_adaptive_scale
                                           and self.adaptive_mode != "film") else None
        film_all = self.adaptive_film if (self.use_adaptive_scale
                                          and self.adaptive_mode == "film") else None
        for i, blk in enumerate(self.blocks):
            _dbg = os.environ.get("SAM_DEBUG_MEM", "0") == "1"
            if _dbg:
                print(f"[blk {i}] start alloc {torch.cuda.memory_allocated()/2**30:.2f} "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f}", flush=True)
            # Process through block (internal adapter modulated per image)
            if ((os.environ.get("SAM_CKPT_ALL", "0") == "1")
                    or (os.environ.get("SAM_CKPT_GLOBAL", "0") == "1"
                        and getattr(blk, "window_size", 0) == 0)) and torch.is_grad_enabled():
                # The four global-attention blocks store a (B*heads, 4096, 4096)
                # attention map for backward, ~1 GB each in fp32.  Recomputation
                # trades ~30% compute for several GB of memory; the result and
                # the gradients are identical.
                def _run(xx, blk=blk, i=i):
                    return blk(xx, scale_fn=scale_fn, layer_idx=i, film=film_all)
                x = torch.utils.checkpoint.checkpoint(_run, x, use_reentrant=False)
            else:
                x = blk(x, scale_fn=scale_fn, layer_idx=i, film=film_all)
            
            # Save intermediate features for skip connections
            # Save at specific layers for skip connections - beginning, 1/3, 2/3, and end of network
            if return_features and i in [0, self.depth//3, 2*self.depth//3, self.depth-1]:
                intermediate_features.append(x)
                
            # Apply external adapter and store output if adapters are enabled
            if self.use_adapter:
                # Apply external adapter to normalized input
                ext_input = self.adapter_norm(x)
                film_ext = film_all if (film_all is not None
                                        and self.adaptive_scope == "both") else None
                external_adapter_out = self.external_adapters[i](
                    ext_input, film=film_ext, layer_idx=i, ext=True, film_cond=x)

                # Input-adaptive scaling of this external adapter (old mode only)
                if scale_fn is not None:
                    external_adapter_out = scale_fn(ext_input, i, ext=True) * external_adapter_out

                # 收集adapter输出用于跳跃连接
                adapter_features.append(external_adapter_out)
                
                # Directly add the adapter output to the main branch
                # This simplifies the flow and reduces potential errors
                x = x + external_adapter_out

        # Apply neck to get final output
        output = self.neck(x.permute(0, 3, 1, 2))
        
        if return_features:
            if self.use_adapter:
                return output, intermediate_features, adapter_features
            else:
                return output, intermediate_features
        else:
            if self.use_adapter:
                return output, None, adapter_features
            else:
                return output


class AdapterModule(nn.Module):
    """Adapter module for efficient fine-tuning."""
    
    def __init__(self, dim: int, adapter_dim_ratio: float = 0.1, act_layer: Type[nn.Module] = nn.GELU):
        """
        Args:
            dim (int): Number of input channels.
            adapter_dim_ratio (float): Ratio of adapter dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()
        self.adapter_dim = int(dim * adapter_dim_ratio)
        
        self.down_proj = nn.Linear(dim, self.adapter_dim)
        self.act = act_layer()
        self.up_proj = nn.Linear(self.adapter_dim, dim)
        
        # Initialize weights to near zero to minimize initial impact
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x: torch.Tensor, film=None, layer_idx=None,
                ext: bool = False, film_cond=None) -> torch.Tensor:
        h = self.act(self.down_proj(x))
        if film is not None and layer_idx is not None:
            h = film(film_cond if film_cond is not None else x, h, layer_idx, ext)
        return self.up_proj(h)


class AdaptiveAdapterScale(nn.Module):
    """Input-conditioned scaling factor for every adapter (the TPAMI addition).

    For each Transformer layer ``l`` and each adapter branch
    ``b in {int, ext}``, a shared two-layer MLP reads the *spatially pooled*
    adapter input and predicts one scalar

        s_{l,b}(x) = 1 + span * tanh( MLP_b( mean_{h,w} z(x) ) + beta_{l,b} )

    so ``span=1`` gives ``s in (0, 2)`` with ``s=1`` meaning "keep the adapter
    exactly as trained", ``s->0`` suppress it and ``s->2`` amplify it.  The last
    linear layer and the per-layer bias are zero-initialised, hence the model
    starts *exactly* at the static dual-level adapter and only departs from it
    when the image evidence justifies it -- important with 18 training images.

    Pooling covers both spatial axes, so the scale is one scalar per image,
    per layer and per branch: no spatial map and no extra decoder.
    """

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        hidden: int = 32,
        separate_int_ext: bool = True,
        span: float = 1.0,
        mode: str = "scale",
    ) -> None:
        super().__init__()
        self.depth = depth
        self.span = span
        self.mode = mode  # "scale" -> 1 + span*tanh, "gate" -> sigmoid
        self.separate_int_ext = separate_int_ext

        def _head() -> nn.Sequential:
            h = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)
            return h

        self.ext_head = _head()
        self.ext_bias = nn.Parameter(torch.zeros(depth))
        if separate_int_ext:
            self.int_head = _head()
            self.int_bias = nn.Parameter(torch.zeros(depth))
        else:
            self.int_head = None
            self.int_bias = None
        self._log: List[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    def forward(self, features: torch.Tensor, layer_idx: int,
                ext: bool = True) -> torch.Tensor:
        """features: [B, H, W, C] -> scale broadcastable to [B, H, W, C]."""
        pooled = features.mean(dim=(1, 2))  # [B, C], both spatial axes
        if ext or self.int_head is None:
            logit = self.ext_head(pooled).squeeze(-1) + self.ext_bias[layer_idx]
        else:
            logit = self.int_head(pooled).squeeze(-1) + self.int_bias[layer_idx]
        if self.mode == "gate":
            s = torch.sigmoid(logit)
        else:
            s = 1.0 + self.span * torch.tanh(logit)
        self._log.append(s)
        return s.view(-1, 1, 1, 1)

    # ------------------------------------------------------------------ #
    def reset_log(self) -> None:
        self._log = []

    def penalty(self) -> torch.Tensor:
        """Budget term (s - target)^2 that keeps the scales near the identity."""
        if not self._log:
            return torch.zeros((), device=self.ext_bias.device)
        vals = torch.cat([s.reshape(-1) for s in self._log])
        target = 0.5 if self.mode == "gate" else 1.0
        return ((vals - target) ** 2).mean()

    def stats(self):
        """(int_mean, ext_mean, std) over the last forward, then clear the log."""
        if not self._log:
            return 0.0, 0.0, 0.0
        vals = torch.cat([s.detach().float().reshape(-1) for s in self._log])
        int_m = vals[0::2].mean().item() if vals.numel() > 1 else vals.mean().item()
        ext_m = vals[1::2].mean().item() if vals.numel() > 1 else vals.mean().item()
        std = vals.std().item() if vals.numel() > 1 else 0.0
        self._log = []
        return int_m, ext_m, std


class HardTopKGate(nn.Module):
    """Hard top-K adapter selection with a straight-through estimator.

    One ranking score per adapter position: internal adapters occupy
    ``0 .. depth-1`` and external adapters ``depth .. 2*depth-1``.  The forward
    pass keeps exactly ``k`` positions and switches the rest off; gradients
    reach every score through ``g = s + (mask - s).detach()``, so the ordering
    is learned end-to-end while the forward stays binary.

    A multiplicative soft gate is unidentifiable (it can be absorbed into the
    adapter's up-projection); a binary mask is not, which is why the selection
    is hard.
    """

    def __init__(self, depth: int, k: int, init_noise: float = 0.01):
        super().__init__()
        self.depth = depth
        self.k = int(k)
        self.scores = nn.Parameter(torch.randn(2 * depth) * init_noise)
        self._last_mask = None

    def mask(self) -> torch.Tensor:
        if self.k >= 2 * self.depth:
            return torch.ones_like(self.scores)
        idx = torch.topk(self.scores, self.k).indices
        m = torch.zeros_like(self.scores)
        m[idx] = 1.0
        return m

    def forward(self, features, layer_idx: int, ext: bool = True):
        s = torch.sigmoid(self.scores)
        m = self.mask()
        self._last_mask = m.detach()
        g = s + (m - s).detach()
        pos = self.depth + layer_idx if ext else layer_idx
        return g[pos]

    # ---- interface expected by train_sam.py --------------------------------
    def reset_log(self) -> None:
        pass

    def penalty(self) -> torch.Tensor:
        # The budget K replaces the sparsity penalty: no auxiliary term.
        return self.scores.new_zeros(())

    def stats(self):
        m = self._last_mask if self._last_mask is not None else self.mask().detach()
        d = self.depth
        return (float(m[:d].mean()), float(m[d:].mean()),
                float(m.std()))

    def selected(self):
        m = self.mask().detach().cpu()
        return [i for i in range(self.depth) if m[i] > 0.5], \
               [i - self.depth for i in range(self.depth, 2 * self.depth) if m[i] > 0.5]


class AdaptiveFiLM(nn.Module):
    """Input-conditional, per-channel modulation of the adapter bottleneck.

    Design notes
    ------------
    The scalar gate used before failed for three reasons: (i) the spatially
    pooled, LayerNorm-ed conditioning vector is almost image-invariant, so the
    predicted scale had a cross-image std of ~0.01 (measured); (ii) the gate
    multiplied the adapter *output*, which is exactly zero at step 0 because
    ``up_proj`` is zero-initialised, so the gate received no gradient; (iii) a
    scalar on the output is redundant with ``W_up``, so the optimiser only used
    it as a global shrink (ext -> 0.66) and destroyed the structural prior.

    This module instead applies a FiLM modulation inside the bottleneck:

        z      = LN([mean_{h,w} u, std_{h,w} u, max_{h,w} u])
        gamma  = span * tanh(W2 GELU(W1 z))            # [B, d]
        gamma <- gamma - rm[layer]                     # running mean over images
        h      <- h * (1 + gamma)                      # FiLM on the bottleneck

    Centering by a running per-channel mean is the key: any component that does
    not depend on the image is subtracted, so the module *cannot* act as a
    static gain.  If the head outputs a constant, ``gamma`` becomes zero and the
    model is exactly the static adapter; only image-dependent modulation
    survives.  ``W2`` is zero-initialised, so training starts from the base
    model, and the gradient is non-zero from step 0 because ``h != 0``.

    ``magnitude_loss`` is a small L2 budget on the modulation.
    """

    def __init__(self, depth: int, embed_dim: int, bottleneck_dim: int,
                 hidden: int = 32, span: float = 0.5,
                 separate_int_ext: bool = True, momentum: float = 0.9) -> None:
        super().__init__()
        self.depth = depth
        self.span = span
        self.separate_int_ext = separate_int_ext
        self.momentum = momentum
        in_dim = 3 * embed_dim

        def _head() -> nn.Sequential:
            h = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                              nn.Linear(hidden, bottleneck_dim))
            # small random init instead of exact zeros: the branch is alive from
            # step 0 (a zero-init head has dL/dW = 0 through the tanh) while the
            # modulation stays ~0.03, i.e. numerically negligible at start
            nn.init.normal_(h[-1].weight, std=1e-3)
            nn.init.zeros_(h[-1].bias)
            return h

        self.ext_head = _head()
        self.int_head = _head() if separate_int_ext else None
        self.register_buffer("rm_int", torch.zeros(depth, bottleneck_dim))
        self.register_buffer("rm_ext", torch.zeros(depth, bottleneck_dim))
        self._log = []   # [(layer, ext, per-sample mean gamma, mean square gamma)]

    @staticmethod
    def _descriptor(u: torch.Tensor) -> torch.Tensor:
        mu = u.mean(dim=(1, 2))
        sd = u.std(dim=(1, 2))
        mx = u.amax(dim=(1, 2))
        z = torch.cat([mu, sd, mx], dim=-1)
        return F.layer_norm(z, z.shape[-1:])

    def forward(self, features: torch.Tensor, hidden_state: torch.Tensor,
                layer_idx: int, ext: bool = True) -> torch.Tensor:
        head = self.ext_head if (ext or self.int_head is None) else self.int_head
        z = self._descriptor(features)
        gamma = self.span * torch.tanh(head(z))          # [B, d]
        rm = self.rm_ext if ext else self.rm_int
        if self.training:
            with torch.no_grad():
                rm[layer_idx].mul_(self.momentum).add_(
                    (1.0 - self.momentum) * gamma.detach().mean(dim=0))
        gamma = gamma - rm[layer_idx].unsqueeze(0)
        self._log.append((int(layer_idx), bool(ext),
                          gamma.detach(),
                          gamma.detach().pow(2).mean()))
        g = gamma
        while g.dim() < hidden_state.dim():
            g = g.unsqueeze(1)
        return hidden_state * (1.0 + g)

    def reset_log(self) -> None:
        self._log = []

    def magnitude_loss(self) -> torch.Tensor:
        if not self._log:
            return torch.zeros((), device=self.rm_int.device)
        return torch.stack([e[3] for e in self._log]).mean()

    def stats(self):
        """(int_std, ext_std): per-channel std of gamma across the last loop (images)."""
        ints, exts = [], []
        groups = {}
        for layer, ext, g, _ in self._log:
            groups.setdefault((layer, bool(ext)), []).append(g)
        for (layer, ext), vals in groups.items():
            if len(vals) > 1:
                v = torch.cat([x.reshape(1, -1) for x in vals], dim=0)  # [N, d]
                (exts if ext else ints).append(float(v.std(dim=0).mean()))
        return (sum(ints) / len(ints) if ints else 0.0,
                sum(exts) / len(exts) if exts else 0.0)


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        use_adapter: bool = False,
        adapter_dim_ratio: float = 0.1,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
            use_adapter (bool): If True, use adapter module after MLP in residual path.
            adapter_dim_ratio (float): Ratio of adapter dim to embedding dim.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        
        # Add adapter module in residual path after MLP
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter = AdapterModule(dim, adapter_dim_ratio, act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor, scale_fn=None,
                layer_idx: Optional[int] = None, film=None) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        
        # MLP block with adapter in residual path
        normed = self.norm2(x)
        mlp_output = self.mlp(normed)
        
        if self.use_adapter:
            # Add adapter output to the MLP path (FiLM / gate / scale if enabled)
            adapter_output = self.adapter(normed, film=film, layer_idx=layer_idx,
                                          ext=False, film_cond=x)
            if scale_fn is not None and layer_idx is not None:
                adapter_output = scale_fn(normed, layer_idx, ext=False) * adapter_output
            x = x + mlp_output + adapter_output
        else:
            x = x + mlp_output

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn_v = attn.view(B, q_h, q_w, k_h, k_w)
    if os.environ.get("SAM_CHUNK_RELPOS", "0") == "1" and q_h >= 32:
        # Chunked accumulation: the naive expression materialises two full
        # (B, heads, q_h*q_w, k_h*k_w) intermediates, which at 1024^2 input
        # costs ~2 GB in the four global-attention blocks.  Mathematically
        # identical, but the peak drops by roughly one full attention map.
        out = torch.empty_like(attn_v)
        step = 8
        for i in range(0, q_h, step):
            j = min(i + step, q_h)
            out[:, i:j] = (attn_v[:, i:j]
                           + rel_h[:, i:j, :, :, None]
                           + rel_w[:, i:j, :, None, :])
        attn = out.view(B, q_h * q_w, k_h * k_w)
    else:
        attn = (
            attn_v + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
