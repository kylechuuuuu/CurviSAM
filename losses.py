"""Losses for curve-aware segmentation.

Components
----------
* region / fused mask : Dice + BCE (or Tversky) on the binary mask
* clDice              : differentiable skeleton (centerline) overlap -> topology
* skeleton head       : Dice + BCE against a sparse skeleton GT
* width head          : masked Smooth-L1 against a normalized distance transform
* fusion              : geometry-fused mask supervision
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# basic mask losses
# --------------------------------------------------------------------------- #
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits) if (logits.min() < 0 or logits.max() > 1) else logits
        prob = prob.reshape(prob.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        inter = (prob * target).sum(dim=1)
        union = prob.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        dice = self.dice(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice


class TverskyLoss(nn.Module):
    """Tversky index loss. beta > alpha favours recall (good for thin curves)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        prob = prob.reshape(prob.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        tp = (prob * target).sum(dim=1)
        fp = (prob * (1 - target)).sum(dim=1)
        fn = ((1 - prob) * target).sum(dim=1)
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky.mean()


class MaskLoss(nn.Module):
    """Dice+BCE or Tversky, selected by ``use_tversky``."""

    def __init__(
        self,
        use_tversky: bool = True,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ):
        super().__init__()
        self.use_tversky = use_tversky
        self.dice_bce = DiceBCELoss(bce_weight, dice_weight)
        self.tversky = TverskyLoss(tversky_alpha, tversky_beta)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_tversky:
            return self.tversky(logits, target)
        return self.dice_bce(logits, target)


# --------------------------------------------------------------------------- #
# clDice (topology-aware, differentiable soft-skeleton)
# --------------------------------------------------------------------------- #
class SoftSkeletonize(nn.Module):
    """Differentiable morphological soft-skeleton (Shit et al., clDice).

    Faithful implementation: each erosion layer contributes a residue and the
    skeleton is the *union* (accumulation) of all layers. Without accumulation
    the skeleton collapses to zero on thin structures and clDice loses all
    gradient.
    """

    def soft_erode(self, x: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))

    def soft_open(self, x: torch.Tensor) -> torch.Tensor:
        return self.soft_dilate(self.soft_erode(x))

    def soft_skel(self, x: torch.Tensor, iters: int) -> torch.Tensor:
        x1 = self.soft_open(x)
        skel = F.relu(x1 - self.soft_dilate(self.soft_erode(x1)))
        for _ in range(iters):
            x1 = self.soft_erode(x1)
            delta = F.relu(x1 - self.soft_dilate(self.soft_erode(x1)))
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, x: torch.Tensor, iters: int = 10) -> torch.Tensor:
        return self.soft_skel(x, iters)


class clDiceLoss(nn.Module):
    def __init__(self, iters: int = 10, smooth: float = 1e-6):
        super().__init__()
        self.iters = iters
        self.smooth = smooth
        self.skel = SoftSkeletonize()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        skel_pred = self.skel(prob, self.iters)
        skel_true = self.skel(target, self.iters)

        tprec = (skel_pred * target).sum(dim=(1, 2, 3)) / (
            skel_pred.sum(dim=(1, 2, 3)) + self.smooth
        )
        tsens = (skel_true * prob).sum(dim=(1, 2, 3)) / (
            skel_true.sum(dim=(1, 2, 3)) + self.smooth
        )
        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice.mean()


# --------------------------------------------------------------------------- #
# width / distance-transform regression
# --------------------------------------------------------------------------- #
class WidthLoss(nn.Module):
    """Masked Smooth-L1 between predicted width and normalized distance transform."""

    def __init__(self, band: int = 3):
        super().__init__()
        self.band = band

    def forward(
        self,
        width_pred: torch.Tensor,
        width_gt: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> torch.Tensor:
        valid = (mask_gt > 0.5).float()
        if self.band > 0:
            k = 2 * self.band + 1
            valid = F.max_pool2d(valid, k, 1, self.band)
        diff = F.smooth_l1_loss(width_pred * valid, width_gt * valid, reduction="sum")
        return diff / (valid.sum() + 1e-6)


# --------------------------------------------------------------------------- #
# combined objective
# --------------------------------------------------------------------------- #
class CurveSegLoss(nn.Module):
    """Total loss for the curve-aware decoder.

    ``targets`` must contain:
      * 'mask'     (B,1,H,W) at full resolution -> region / fused / clDice
      * 'mask_ds'  (B,1,h,w) at decoder res     -> width masking
      * 'skeleton' (B,1,h,w) at decoder res     -> skeleton head
      * 'width'    (B,1,h,w) at decoder res     -> width head

    region / fused are upsampled to the full mask resolution before supervision,
    matching the original training regime (thin structures are preserved).
    """

    def __init__(
        self,
        w_region: float = 1.0,
        w_fused: float = 1.0,
        w_coarse: float = 0.5,
        w_skel: float = 1.0,
        w_cldice: float = 1.0,
        w_width: float = 1.0,
        use_tversky: bool = True,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        skel_bce_weight: float = 0.3,
        skel_dice_weight: float = 0.7,
        cldice_iters: int = 10,
        cldice_size: int = 512,
        width_band: int = 3,
    ):
        super().__init__()
        self.w = dict(
            region=w_region, fused=w_fused, coarse=w_coarse, skel=w_skel,
            cldice=w_cldice, width=w_width
        )
        self.cldice_size = cldice_size
        self.mask_loss = MaskLoss(use_tversky, bce_weight, dice_weight,
                                  tversky_alpha, tversky_beta)
        self.skel_loss = DiceBCELoss(skel_bce_weight, skel_dice_weight)
        self.cldice = clDiceLoss(cldice_iters)
        self.width = WidthLoss(width_band)

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        mask = targets["mask"]              # full resolution (e.g. 1024)
        mask_ds = targets.get("mask_ds", mask)
        skel = targets["skeleton"]          # decoder resolution (e.g. 256)
        width = targets["width"]

        full_size = mask.shape[-2:]
        region_logits = F.interpolate(
            outputs["region"][:, :1], size=full_size, mode="bilinear", align_corners=False
        )
        fused_logits = F.interpolate(
            outputs["masks"], size=full_size, mode="bilinear", align_corners=False
        )

        l_region = self.mask_loss(region_logits, mask)
        l_fused = self.mask_loss(fused_logits, mask)
        if outputs.get("coarse") is not None:
            coarse_logits = F.interpolate(outputs["coarse"], size=full_size,
                                          mode="bilinear", align_corners=False)
            l_coarse = self.mask_loss(coarse_logits, mask)
        else:
            l_coarse = torch.zeros((), device=mask.device)

        # clDice on a smaller grid to keep activation memory bounded
        cl_fused, cl_mask = fused_logits, mask
        if self.cldice_size and mask.shape[-1] > self.cldice_size:
            cl_fused = F.interpolate(cl_fused, size=(self.cldice_size, self.cldice_size),
                                     mode="bilinear", align_corners=False)
            cl_mask = F.interpolate(cl_mask, size=(self.cldice_size, self.cldice_size),
                                    mode="bilinear", align_corners=False)
        l_cldice = self.cldice(cl_fused, cl_mask)

        if outputs.get("skeleton") is not None:
            l_skel = self.skel_loss(outputs["skeleton"], skel)
        else:
            l_skel = torch.zeros((), device=mask.device)

        if outputs.get("width") is not None:
            l_width = self.width(outputs["width"], width, mask_ds)
        else:
            l_width = torch.zeros((), device=mask.device)

        total = (
            self.w["region"] * l_region
            + self.w["fused"] * l_fused
            + self.w["coarse"] * l_coarse
            + self.w["skel"] * l_skel
            + self.w["cldice"] * l_cldice
            + self.w["width"] * l_width
        )
        return {
            "loss": total,
            "region": l_region.detach(),
            "fused": l_fused.detach(),
            "coarse": l_coarse.detach(),
            "skel": l_skel.detach(),
            "cldice": l_cldice.detach(),
            "width": l_width.detach(),
        }
