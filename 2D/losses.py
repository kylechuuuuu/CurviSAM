"""Losses for curve-attribute segmentation (v17).

Components
----------
* region / fused / coarse mask : Dice+BCE (or Tversky), supervised at full 1024 res
* clDice                       : differentiable soft-skeleton topology loss (512)
* centerline head              : Dice+BCE against a *soft tolerance-band* skeleton
* radius head                  : masked Smooth-L1 in log1p(px) space, skeleton band only
* boundary SDF head            : masked Smooth-L1 against the truncated signed
                                 distance (near the boundary only)

Why log-radius / band supervision
---------------------------------
The radius is an attribute *of the centerline*, so supervising it on the whole
vessel is redundant and dilutes the gradient. ``log1p`` also makes the loss
invariant to a per-image scale factor, which matters when the target domain's
resolution is unknown.
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
    def __init__(self, bce_weight: float = 0.7, dice_weight: float = 0.3,
                 smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + \
            self.dice_weight * self.dice(logits, target)


class TverskyLoss(nn.Module):
    """beta > alpha favours recall (good for thin curves)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits).reshape(logits.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        tp = (prob * target).sum(dim=1)
        fp = (prob * (1 - target)).sum(dim=1)
        fn = ((1 - prob) * target).sum(dim=1)
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class MaskLoss(nn.Module):
    """Dice+BCE, Tversky, and/or an extra precision-oriented Tversky term.

    ``use_tversky`` *replaces* Dice+BCE (legacy switch). ``w_prec > 0`` *adds* a
    Tversky term with ``prec_alpha > prec_beta`` on top, which penalises false
    positives more than false negatives - the targeted remedy for the systematic
    over-segmentation observed on unseen domains.
    """

    def __init__(self, use_tversky: bool = False, bce_weight: float = 0.7,
                 dice_weight: float = 0.3, tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7, w_prec: float = 0.0,
                 prec_alpha: float = 0.8, prec_beta: float = 0.2):
        super().__init__()
        self.use_tversky = use_tversky
        self.dice_bce = DiceBCELoss(bce_weight, dice_weight)
        self.tversky = TverskyLoss(tversky_alpha, tversky_beta)
        self.w_prec = w_prec
        if w_prec > 0:
            self.prec = TverskyLoss(prec_alpha, prec_beta)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_tversky:
            loss = self.tversky(logits, target)
        else:
            loss = self.dice_bce(logits, target)
        if self.w_prec > 0:
            loss = loss + self.w_prec * self.prec(logits, target)
        return loss


# --------------------------------------------------------------------------- #
# clDice (topology-aware, differentiable soft-skeleton)
# --------------------------------------------------------------------------- #
class SoftSkeletonize(nn.Module):
    """Differentiable morphological soft-skeleton (Shit et al., clDice).

    Each erosion layer contributes a residue and the skeleton is the union of all
    layers; without accumulation it collapses to zero on thin structures.
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
            skel_pred.sum(dim=(1, 2, 3)) + self.smooth)
        tsens = (skel_true * prob).sum(dim=(1, 2, 3)) / (
            skel_true.sum(dim=(1, 2, 3)) + self.smooth)
        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice.mean()


# --------------------------------------------------------------------------- #
# attribute losses
# --------------------------------------------------------------------------- #
class WidthLoss(nn.Module):
    """Masked Smooth-L1 on log1p(radius in px), supervised on the skeleton band."""

    def forward(self, width_pred: torch.Tensor, width_gt: torch.Tensor,
                band: torch.Tensor) -> torch.Tensor:
        w = (band > 0.5).to(width_pred.dtype)
        if float(w.sum()) <= 0:
            return torch.zeros((), device=width_pred.device, dtype=width_pred.dtype)
        diff = F.smooth_l1_loss(width_pred * w, width_gt * w, reduction="sum")
        return diff / (w.sum() + 1e-6)


class SDFLoss(nn.Module):
    """Masked Smooth-L1 on the truncated signed boundary distance (px)."""

    def __init__(self, trunc: float = 8.0):
        super().__init__()
        self.trunc = trunc

    def forward(self, sdf_pred: torch.Tensor, sdf_gt: torch.Tensor) -> torch.Tensor:
        valid = (sdf_gt.abs() < self.trunc).to(sdf_pred.dtype)
        if float(valid.sum()) <= 0:
            return torch.zeros((), device=sdf_pred.device, dtype=sdf_pred.dtype)
        diff = F.smooth_l1_loss(sdf_pred * valid, sdf_gt * valid, reduction="sum")
        return diff / (valid.sum() + 1e-6)


# --------------------------------------------------------------------------- #
# combined objective
# --------------------------------------------------------------------------- #
class CurveSegLoss(nn.Module):
    """Total loss for the v17 attribute decoder.

    ``targets`` keys
      * 'mask'  (B,1,1024,1024) binary -> region / fused / coarse / clDice
      * 'skel'  (B,1,512,512) soft tolerance-band centerline
      * 'width' (B,1,512,512) log1p(px radius)
      * 'sdf'   (B,1,512,512) truncated signed boundary distance (px)
      * 'band'  (B,1,512,512) radius supervision mask
    """

    def __init__(
        self,
        w_region: float = 1.0,
        w_fused: float = 1.0,
        w_coarse: float = 0.5,
        w_skel: float = 0.3,
        w_width: float = 0.3,
        w_sdf: float = 0.2,
        w_cldice: float = 1.0,
        use_tversky: bool = False,
        bce_weight: float = 0.7,
        dice_weight: float = 0.3,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        w_prec: float = 0.0,
        prec_alpha: float = 0.8,
        prec_beta: float = 0.2,
        cldice_iters: int = 10,
        cldice_size: int = 512,
        skel_bce_weight: float = 0.5,
        skel_dice_weight: float = 0.5,
        sdf_trunc: float = 8.0,
    ):
        super().__init__()
        self.w = dict(region=w_region, fused=w_fused, coarse=w_coarse,
                      skel=w_skel, width=w_width, sdf=w_sdf, cldice=w_cldice)
        self.cldice_size = cldice_size
        self.mask_loss = MaskLoss(use_tversky, bce_weight, dice_weight,
                                  tversky_alpha, tversky_beta,
                                  w_prec, prec_alpha, prec_beta)
        self.skel_loss = DiceBCELoss(skel_bce_weight, skel_dice_weight)
        self.cldice = clDiceLoss(cldice_iters)
        self.width = WidthLoss()
        self.sdf = SDFLoss(sdf_trunc)

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = targets["mask"]                      # 1024
        full_size = mask.shape[-2:]

        def _up(logits):
            if logits.shape[-2:] == full_size:
                return logits
            return F.interpolate(logits, size=full_size, mode="bilinear",
                                 align_corners=False)

        region_logits = _up(outputs["region"][:, :1])
        fused_logits = _up(outputs["masks"])
        l_region = self.mask_loss(region_logits, mask)
        l_fused = self.mask_loss(fused_logits, mask)

        if outputs.get("coarse") is not None:
            l_coarse = self.mask_loss(_up(outputs["coarse"]), mask)
        else:
            l_coarse = torch.zeros((), device=mask.device)

        # clDice on a smaller grid to keep activation memory bounded
        if self.cldice_size and mask.shape[-1] > self.cldice_size:
            f = float(mask.shape[-1] // self.cldice_size)
            cl_mask = F.max_pool2d(mask, int(f), int(f))
            cl_fused = _down_to(outputs["masks"], self.cldice_size)
        else:
            cl_mask = mask
            cl_fused = fused_logits
        l_cldice = self.cldice(cl_fused, cl_mask)

        if outputs.get("skeleton") is not None:
            l_skel = self.skel_loss(outputs["skeleton"], targets["skel"])
        else:
            l_skel = torch.zeros((), device=mask.device)

        if outputs.get("width") is not None:
            l_width = self.width(outputs["width"], targets["width"], targets["band"])
        else:
            l_width = torch.zeros((), device=mask.device)

        if outputs.get("sdf") is not None:
            l_sdf = self.sdf(outputs["sdf"], targets["sdf"])
        else:
            l_sdf = torch.zeros((), device=mask.device)

        total = (self.w["region"] * l_region
                 + self.w["fused"] * l_fused
                 + self.w["coarse"] * l_coarse
                 + self.w["skel"] * l_skel
                 + self.w["width"] * l_width
                 + self.w["sdf"] * l_sdf
                 + self.w["cldice"] * l_cldice)
        return {
            "loss": total,
            "region": l_region.detach(),
            "fused": l_fused.detach(),
            "coarse": l_coarse.detach(),
            "skel": l_skel.detach(),
            "width": l_width.detach(),
            "sdf": l_sdf.detach(),
            "cldice": l_cldice.detach(),
        }


def _down_to(logits: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
