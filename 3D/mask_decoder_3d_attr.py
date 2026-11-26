"""3D attribute decoder: Conv3d centreline / radius / boundary-SDF heads on top
of the frozen 2D SAM slice embeddings.

Input : (B, 256, K, 64, 64)   stacked slice embeddings
Output: dict(masks, region, skeleton, width, sdf) at the attribute grid
        (default (K, 128, 128), i.e. 2x upsampled in-plane).

The design mirrors the 2D decoder: the mask is the region head plus a learned
residual whose inputs include explicit geometric evidence (soft distance to the
predicted centreline and an analytic tube render).  Both residual paths are
zero-initialised, so at step 0 ``masks == region`` and the attribute decoder can
never start worse than the plain Conv3d head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_distance_3d(center_logits: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Differentiable chessboard distance to the predicted centreline in 3D."""
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


class ConvBlock3D(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, k, padding=k // 2, bias=False),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MaskDecoder3DAttr(nn.Module):
    def __init__(self, in_ch=256, dims=(128, 64), attr_dim=32,
                 upsample=2, iters=6, tau=0.5):
        super().__init__()
        self.upsample = upsample
        self.iters = iters
        self.tau = tau

        self.stem = nn.Sequential(
            ConvBlock3D(in_ch, dims[0]),
            ConvBlock3D(dims[0], dims[1]),
        )
        if upsample > 1:
            self.up = nn.Sequential(
                nn.Conv3d(dims[1], dims[1], 3, padding=1),
                nn.GroupNorm(min(8, dims[1]), dims[1]),
                nn.GELU(),
            )
        else:
            self.up = None

        self.region = nn.Conv3d(dims[1], 1, 1)
        self.head_stem = ConvBlock3D(dims[1], attr_dim)
        self.branch_center = nn.Conv3d(attr_dim, 1, 1)
        self.branch_width = nn.Conv3d(attr_dim, 1, 1)
        self.branch_sdf = nn.Sequential(
            ConvBlock3D(attr_dim + 4, attr_dim),
            nn.Conv3d(attr_dim, 1, 1),
        )
        self.geo_fuse = nn.Conv3d(4, attr_dim, 1)
        self.fusion = nn.Sequential(
            ConvBlock3D(2 * attr_dim + 4, attr_dim),
            nn.Conv3d(attr_dim, 1, 1),
        )

        for m in (self.geo_fuse, self.fusion[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, emb):
        x = self.stem(emb)
        if self.up is not None:
            x = F.interpolate(x, scale_factor=(1, self.upsample, self.upsample),
                              mode="trilinear", align_corners=False)
            x = self.up(x)

        region = self.region(x)
        feat = self.head_stem(x)
        center = self.branch_center(feat)
        width = self.branch_width(feat)              # log1p(radius in voxels)
        radius = F.softplus(width)
        dist = soft_distance_3d(center, self.iters)
        render = torch.sigmoid((radius - dist) / self.tau)

        geo = self.geo_fuse(torch.cat([center, width, dist, render], 1))
        sdf = self.branch_sdf(torch.cat([feat, center, width, dist, render], 1))
        corr = self.fusion(torch.cat([feat, center, width, dist, render, geo], 1))

        return {"masks": region + corr, "region": region,
                "skeleton": center, "width": width, "sdf": sdf}
