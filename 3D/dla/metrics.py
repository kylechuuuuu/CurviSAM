"""Small metrics / prompt helpers (kept dependency-free)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    return (2.0 * inter + eps) / (float(pred.sum()) + float(gt.sum()) + eps)


def sample_point(seg: np.ndarray, rng: np.random.Generator, label: int = 1
                 ) -> Tuple[np.ndarray, int]:
    """One prompt click.  label=1 -> random foreground voxel, 0 -> random background.

    Foreground-biased for positives (60% inside the label, 40% on its surface)
    so that the click does not always land in the thickest part of the tree.
    """
    if label == 1:
        fg = np.argwhere(seg > 0)
        if rng.random() < 0.6 or len(fg) < 8:
            p = fg[rng.integers(len(fg))]
        else:
            # surface voxels: foreground with at least one background 6-neighbour
            from scipy import ndimage
            surf = (seg > 0) & (ndimage.binary_erosion(seg > 0, iterations=1) == 0)
            p = np.argwhere(surf)[rng.integers(int(surf.sum()))]
        return p.astype(np.float32), 1
    bg = np.argwhere(seg == 0)
    p = bg[rng.integers(len(bg))]
    return p.astype(np.float32), 0


def to_model_input(img: np.ndarray) -> torch.Tensor:
    """(D,H,W) float -> [1,1,D,H,W] tensor."""
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def points_tensor(points, labels, device) -> Tuple[torch.Tensor, torch.Tensor]:
    pc = torch.tensor(np.asarray(points, dtype=np.float32).reshape(1, -1, 3), device=device)
    pl = torch.tensor(np.asarray(labels, dtype=np.int64).reshape(1, -1), device=device)
    return pc, pl
