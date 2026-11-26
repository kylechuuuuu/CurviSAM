"""Pre-processing for 3D volumes -> fixed-size ROI tensors for SAM-Med3D.

Follows the official SAM-Med3D validation pipeline (`utils/infer_utils.py`)
without the torchio dependency:

    resample to isotropic spacing -> centre-crop/pad ROI around the label
    -> z-normalise on the foreground support of the image

Volumes are cached as .npz so training never touches NIfTI again.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk


def _resample(img: sitk.Image, lab: sitk.Image, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    size = img.GetSize()             # (x, y, z)
    sp = img.GetSpacing()            # (x, y, z)
    new_size = [int(round(s * p / spacing)) for s, p in zip(size, sp)]
    rs = sitk.ResampleImageFilter()
    rs.SetSize(new_size)
    rs.SetOutputSpacing((spacing, spacing, spacing))
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(0)
    rs.SetInterpolator(sitk.sitkLinear)
    img_rs = rs.Execute(img)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    lab_rs = rs.Execute(lab)
    return sitk.GetArrayFromImage(img_rs), sitk.GetArrayFromImage(lab_rs)


def _crop_pad(arr: np.ndarray, centre, size: int, fill: float = 0.0) -> np.ndarray:
    """Crop / pad a (D,H,W) array to (size,)*3 around `centre`."""
    out = np.full((size, size, size), fill, dtype=arr.dtype)
    src_slices, dst_slices = [], []
    for ax in range(3):
        n = arr.shape[ax]
        start = int(round(centre[ax] - size // 2))
        lo, hi = max(0, start), min(n, start + size)
        d_lo = lo - start
        src_slices.append(slice(lo, hi))
        dst_slices.append(slice(d_lo, d_lo + (hi - lo)))
    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def prepare_volume(img_path: str, gt_path: str, spacing: float = 1.5, size: int = 128,
                   min_fg: int = 50) -> Dict[str, np.ndarray]:
    img = sitk.ReadImage(img_path)
    lab = sitk.ReadImage(gt_path)
    if img.GetDirection() != lab.GetDirection():
        lab.SetDirection(img.GetDirection())
    if img.GetOrigin() != lab.GetOrigin():
        lab.SetOrigin(img.GetOrigin())

    arr, seg = _resample(img, lab, spacing)
    arr = arr.astype(np.float32)
    seg = (seg > 0).astype(np.uint8)
    if seg.sum() < min_fg:
        raise ValueError(f"empty/small label: {seg.sum()} voxels")

    fg = np.argwhere(seg > 0)
    centre = fg.mean(axis=0)  # (d, h, w)
    arr = _crop_pad(arr, centre, size)
    seg = _crop_pad(seg, centre, size, fill=0)

    support = arr[arr > 0]
    if support.size < 10:
        raise ValueError("image has no positive support")
    mean, std = float(support.mean()), float(support.std() + 1e-6)
    arr = (arr - mean) / std
    arr = np.clip(arr, -8, 8)

    return {
        "image": arr.astype(np.float16),
        "label": seg,
        "meta": np.array([json.dumps({
            "img": os.path.basename(img_path),
            "gt": os.path.basename(gt_path),
            "orig_size": list(img.GetSize()),
            "orig_spacing": [round(s, 4) for s in img.GetSpacing()],
            "target_spacing": spacing,
            "roi_centre_zyx": [round(float(c), 2) for c in centre],
            "mean": mean, "std": std,
            "fg_voxels_roi": int(seg.sum()),
        })]),
    }


def load_volume(path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    z = np.load(path, allow_pickle=False)
    return (z["image"].astype(np.float32), z["label"].astype(np.uint8),
            json.loads(str(z["meta"][0])))
