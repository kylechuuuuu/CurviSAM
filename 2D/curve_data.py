"""Dataset + GT construction for curve-attribute segmentation (v17).

Every geometric field is derived from the mask with **OpenCV only** (no scipy):

* ``skel``  : ``cv2.ximgproc.thinning`` (1 px) softened into a tolerance-band target
* ``width`` : ``cv2.distanceTransform`` inside the mask -> radius in *pixels of the
              512 grid*, stored as ``log1p(r)`` (no per-image max normalisation)
* ``sdf``   : truncated **signed** boundary distance (inside +, outside -), 512-grid px
* ``band``  : skeleton tolerance band; masks the radius supervision

Attribute grids live at 512 (``ATTR_SIZE``) to match the decoder's attribute heads,
while the binary mask stays at 1024 for the mask loss / clDice.

Augmentation transforms the image and the mask, then **re-derives** every geometric
field from the transformed mask.  v16 interpolated the geometry labels instead,
which broke 1 px skeletons at 0.4x scale and silently rescaled the radius target.
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 1024
ATTR_SIZE = 512
SDF_TRUNC = 8.0
BAND_RADIUS = 2
SUPPORTED = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

try:  # opencv-contrib is required for thinning
    _HAS_XIMGPROC = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning")
except Exception:  # pragma: no cover
    _HAS_XIMGPROC = False


# --------------------------------------------------------------------------- #
# geometry derivation (numpy / opencv)
# --------------------------------------------------------------------------- #
def skeletonize(binary: np.ndarray) -> np.ndarray:
    """binary: uint8 {0,1} -> uint8 {0,1} 1-px skeleton.

    Identical to the v16/v17 original: OpenCV contrib Zhang-Suen only, no
    fallback, so the generated GT matches the released runs exactly.
    """
    if not _HAS_XIMGPROC:
        raise RuntimeError(
            "cv2.ximgproc.thinning is unavailable; install opencv-contrib-python"
        )
    thin = cv2.ximgproc.thinning((binary * 255).astype(np.uint8))
    return (thin > 127).astype(np.uint8)


def _maxpool2(a: np.ndarray) -> np.ndarray:
    """(H,W) -> (H/2,W/2) max-pool (keeps 1 px structures, unlike resize)."""
    h, w = a.shape[-2:]
    return a.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))


def _radius_px(binary: np.ndarray) -> np.ndarray:
    """Inner distance transform: distance to the nearest background, in pixels."""
    return cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)


def _signed_distance(binary: np.ndarray, trunc: float) -> np.ndarray:
    """Truncated signed distance: > 0 inside, < 0 outside, in pixels."""
    inside = (binary > 0).astype(np.uint8)
    dt_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    dt_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    return np.clip(dt_in - dt_out, -trunc, trunc).astype(np.float32)


def _soft_skeleton(skel: np.ndarray) -> np.ndarray:
    """1 px core -> tolerance-band target (core 1.0, +/-1 px 0.5, +/-2 px 0.2)."""
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    ring1 = cv2.dilate(skel, k3) - skel
    ring2 = cv2.dilate(skel, k5) - cv2.dilate(skel, k3)
    return (skel.astype(np.float32)
            + 0.5 * ring1.astype(np.float32)
            + 0.2 * ring2.astype(np.float32))


def derive_fields(mask1024: np.ndarray, sdf_trunc: float = SDF_TRUNC,
                  band_radius: int = BAND_RADIUS) -> dict:
    """(H,W) uint8 {0,1} mask at 1024 -> all attribute targets at ``ATTR_SIZE``."""
    m = (mask1024 > 0).astype(np.uint8)
    skel1024 = skeletonize(m)
    m512 = _maxpool2(m)
    skel512 = _maxpool2(skel1024)

    radius = _radius_px(m512).astype(np.float32)
    sdf = _signed_distance(m512, sdf_trunc)
    k = 2 * band_radius + 1
    band = cv2.dilate(skel512, np.ones((k, k), np.uint8)).astype(np.float32)

    return {
        "skel": _soft_skeleton(skel512).astype(np.float16),
        "width": np.log1p(radius).astype(np.float16),
        "sdf": sdf.astype(np.float16),
        "band": band.astype(np.float16),
    }


def build_curve_gt(mask_path: str, cache_path: str, size: int = IMG_SIZE,
                   sdf_trunc: float = SDF_TRUNC) -> None:
    mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    m = (np.array(mask) > 0).astype(np.uint8)  # any non-zero is foreground
    fields = derive_fields(m, sdf_trunc)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, mask=m, **fields)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def _pick_dir(base: str, candidates) -> str:
    """Return the first existing sub-directory name (datasets differ: image/images)."""
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.isdir(p):
            return p
    return os.path.join(base, candidates[0])


class CurveSegDataset(Dataset):
    """Returns (image, mask, skel, width, sdf, band, name).

    * image : float [0,1], 1024
    * mask  : float {0,1}, 1024 (mask loss / clDice)
    * skel  : float (512) soft tolerance-band centerline target
    * width : float (512) log1p(radius in 512-grid px), supervised on ``band``
    * sdf   : float (512) truncated signed boundary distance in px
    * band  : float (512) 1 where radius supervision applies
    """

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        img_size: int = IMG_SIZE,
        cache: bool = True,
        cache_dir: Optional[str] = None,
        augment: bool = False,
        sdf_trunc: float = SDF_TRUNC,
    ) -> None:
        self.root_dir = root_dir
        self.img_size = img_size
        self.sdf_trunc = sdf_trunc
        self.augment = int(augment)  # 0 none | 1 scale+dihedral (v9) | 2 affine
        if split:
            base = os.path.join(root_dir, split)
        else:
            base = root_dir
        self.image_dir = _pick_dir(base, ("images", "image"))
        self.mask_dir = _pick_dir(base, ("masks", "mask", "annotation_mask"))
        if cache_dir is None:
            tag = root_dir.strip("./").replace("/", "_").replace("\\", "_")
            cache_dir = os.path.join("cache", tag, split or "all")
        self.cache_dir = cache_dir
        self.cache = cache

        self.files = []
        for name in os.listdir(self.image_dir):
            if name.lower().endswith(SUPPORTED):
                base = os.path.splitext(name)[0]
                if any(os.path.exists(os.path.join(self.mask_dir, base + e)) for e in SUPPORTED):
                    self.files.append(name)
        self.files.sort()
        if not self.files:
            raise ValueError(f"No valid image-mask pairs found in {self.image_dir}")

        self.img_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        )

    def _mask_path(self, name: str) -> str:
        base = os.path.splitext(name)[0]
        for e in SUPPORTED:
            p = os.path.join(self.mask_dir, base + e)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(name)

    def _get_gt(self, name: str) -> dict:
        base = os.path.splitext(name)[0]
        cache_path = os.path.join(self.cache_dir, base + ".npz")
        if not (self.cache and os.path.exists(cache_path)):
            build_curve_gt(self._mask_path(name), cache_path, self.img_size, self.sdf_trunc)
        d = np.load(cache_path)
        return d

    def __len__(self) -> int:
        return len(self.files)

    # ------------------------------------------------------------------ #
    def _apply_augment(self, image: torch.Tensor, mask: torch.Tensor):
        S = self.img_size
        if self.augment == 2:
            # affine: rotation (any angle) + scale + translation
            import math
            ang = float(torch.empty(1).uniform_(-180.0, 180.0)) * math.pi / 180.0
            sc = float(torch.empty(1).uniform_(0.4, 1.6))
            tx = float(torch.empty(1).uniform_(-0.12, 0.12))
            ty = float(torch.empty(1).uniform_(-0.12, 0.12))
            cos, sin = math.cos(ang) * sc, math.sin(ang) * sc
            theta = torch.tensor([[[cos, -sin, tx], [sin, cos, ty]]], dtype=torch.float32)

            def _aff(t, mode):
                grid = F.affine_grid(theta, (1, t.shape[0], S, S), align_corners=False)
                return F.grid_sample(t.unsqueeze(0), grid, mode=mode,
                                     padding_mode="zeros", align_corners=False)[0]

            image = _aff(image, "bilinear")
            mask = _aff(mask, "nearest")
        else:
            # v9-style: random scale + crop/pad + dihedral
            if torch.rand(1).item() < 0.7:
                scale = float(torch.empty(1).uniform_(0.4, 1.6).item())
                ns = max(64, int(round(S * scale)))

                def _rs(t, mode):
                    kw = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
                    return F.interpolate(t.unsqueeze(0), size=(ns, ns), mode=mode, **kw)[0]

                image = _rs(image, "bilinear")
                mask = _rs(mask, "nearest")
                if ns >= S:
                    x0 = int(torch.randint(0, ns - S + 1, (1,)).item())
                    y0 = int(torch.randint(0, ns - S + 1, (1,)).item())
                    image = image[:, y0:y0 + S, x0:x0 + S]
                    mask = mask[:, y0:y0 + S, x0:x0 + S]
                else:
                    pad = S - ns
                    x0 = int(torch.randint(0, pad + 1, (1,)).item())
                    y0 = int(torch.randint(0, pad + 1, (1,)).item())

                    def _place(t):
                        c = torch.zeros((t.shape[0], S, S), dtype=t.dtype, device=t.device)
                        c[:, y0:y0 + ns, x0:x0 + ns] = t
                        return c

                    image, mask = _place(image), _place(mask)
            k = torch.randint(0, 4, (1,)).item()
            if k:
                image = torch.rot90(image, k, [1, 2])
                mask = torch.rot90(mask, k, [1, 2])

        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [2])
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, [1])
            mask = torch.flip(mask, [1])

        # photometric jitter (brightness / contrast / gamma)
        if torch.rand(1).item() < 0.5:
            image = torch.clamp(image * (1.0 + 0.3 * float(torch.randn(1)))
                                + 0.1 * float(torch.randn(1)), 0, 1)
        if torch.rand(1).item() < 0.3:
            image = torch.clamp(image, 1e-4, 1.0) ** (1.0 + 0.5 * float(torch.randn(1)))
        return image, (mask > 0.5).float()

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int):
        name = self.files[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        image = self.img_transform(image)

        gt = self._get_gt(name)
        if self.augment:
            mask_t = torch.from_numpy(gt["mask"].astype(np.float32)).unsqueeze(0)
            image, mask_t = self._apply_augment(image, mask_t)
            # re-derive geometry from the *transformed* mask
            fields = derive_fields(mask_t[0].numpy().astype(np.uint8), self.sdf_trunc)
        else:
            mask_t = torch.from_numpy(gt["mask"].astype(np.float32)).unsqueeze(0)
            fields = {k: gt[k] for k in ("skel", "width", "sdf", "band")}

        skel = torch.from_numpy(fields["skel"].astype(np.float32)).unsqueeze(0)
        width = torch.from_numpy(fields["width"].astype(np.float32)).unsqueeze(0)
        sdf = torch.from_numpy(fields["sdf"].astype(np.float32)).unsqueeze(0)
        band = torch.from_numpy(fields["band"].astype(np.float32)).unsqueeze(0)
        return image, mask_t, skel, width, sdf, band, name


def prepare(root_dir: str, split: Optional[str] = None, img_size: int = IMG_SIZE,
            cache_dir: Optional[str] = None) -> int:
    """Pre-compute and cache all attribute fields for a whole split."""
    ds = CurveSegDataset(root_dir, split, img_size, cache=True, cache_dir=cache_dir)
    for name in ds.files:
        ds._get_gt(name)
    return len(ds.files)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute curve attribute GT (v17)")
    parser.add_argument("--data_root", type=str, default="./datasets/eee")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    args = parser.parse_args()
    if args.split:
        n = prepare(args.data_root, args.split, args.img_size)
        print(f"Prepared {n} samples in {args.data_root}/{args.split}")
    else:
        for split in ("train", "val"):
            if os.path.isdir(os.path.join(args.data_root, split)):
                n = prepare(args.data_root, split, args.img_size)
                print(f"Prepared {n} samples in {args.data_root}/{split}")
