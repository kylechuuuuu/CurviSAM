"""Derive 3D attribute targets (centreline, radius, boundary SDF) from a binary
mask slab.

A slab is (K, H, W): K consecutive axial slices.  The attribute grid is the
in-plane downsampled mask (K, h, w); the skeleton is a 3D thinning of that small
volume, the radius is the in-plane distance transform sampled on the skeleton,
and the SDF is the signed distance to the boundary, truncated.
"""
import numpy as np
from scipy import ndimage

try:
    from skimage.morphology import skeletonize as _skel3d
except Exception:  # pragma: no cover
    _skel3d = None


def _skeleton(mask: np.ndarray) -> np.ndarray:
    if _skel3d is not None:
        return _skel3d(mask.astype(bool)).astype(np.float32)
    # fallback: distance-transform ridge (local maxima of the EDT)
    dt = ndimage.distance_transform_edt(mask > 0)
    mx = ndimage.maximum_filter(dt, size=3)
    ridge = (dt > 0) & (dt >= mx - 1e-6)
    return ridge.astype(np.float32)


def _soft_band(skel: np.ndarray) -> np.ndarray:
    """core 1.0, +/-1 voxel 0.5, +/-2 voxels 0.2 (3D)."""
    k3 = np.ones((3, 3, 3), np.uint8)
    k5 = np.ones((5, 5, 5), np.uint8)
    s = skel.astype(np.uint8)
    ring1 = ndimage.binary_dilation(s, k3).astype(np.float32) - s
    ring2 = ndimage.binary_dilation(s, k5).astype(np.float32) \
        - ndimage.binary_dilation(s, k3).astype(np.float32)
    return s.astype(np.float32) + 0.5 * ring1 + 0.2 * ring2


def derive_fields_3d(mask_slab: np.ndarray, sdf_trunc: float = 2.0):
    """mask_slab: (K, H, W) binary at the attribute grid.

    Returns dict with float32 arrays:
      skel  (K,H,W) soft tolerance band in [0,1]
      band  (K,H,W) hard skeleton band (for masking the radius loss)
      width (K,H,W) log1p(radius in voxels)
      sdf   (K,H,W) truncated signed boundary distance
    """
    m = (mask_slab > 0.5).astype(np.uint8)
    skel = _skeleton(m)
    band = ndimage.binary_dilation(skel > 0, np.ones((3, 3, 3), np.uint8))
    dt_in = ndimage.distance_transform_edt(m)
    dt_out = ndimage.distance_transform_edt(1 - m)
    sdf = np.clip(dt_in - dt_out, -sdf_trunc, sdf_trunc).astype(np.float32)
    radius = np.where(skel > 0, dt_in, 0.0).astype(np.float32)
    return {
        "skel": _soft_band(skel).astype(np.float32),
        "band": band.astype(np.float32),
        "width": np.log1p(radius).astype(np.float32),
        "sdf": sdf,
    }
