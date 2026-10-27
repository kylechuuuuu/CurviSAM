"""Dataset + GT construction for curve segmentation.

For every image/mask pair we additionally build:
  * skeleton : 1-px centerline via cv2.ximgproc.thinning
  * width    : normalized Euclidean distance transform (scipy), i.e. local radius

Both are cached to ``<split>/../_curve_gt/<base>.npz`` so they are computed only
once (the masks are resized to the model input size first).
"""

import os
import math
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 1024
SUPPORTED = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """binary: uint8 {0,1} -> uint8 {0,1} 1-px skeleton."""
    thin = cv2.ximgproc.thinning((binary * 255).astype(np.uint8))
    return (thin > 127).astype(np.uint8)


def width_map(binary: np.ndarray) -> np.ndarray:
    """binary: uint8 {0,1} -> float32 normalized distance transform."""
    dt = distance_transform_edt(binary > 0).astype(np.float32)
    m = float(dt.max())
    if m > 0:
        dt = dt / m
    return dt


def build_curve_gt(mask_path: str, cache_path: str, size: int = IMG_SIZE) -> None:
    mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    m = np.array(mask)
    binary = (m > 0).astype(np.uint8)  # any non-zero is foreground (DSCA 127/255)
    skel = skeletonize(binary)
    width = width_map(binary)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, mask=binary, skeleton=skel, width=width.astype(np.float16))


class CurveSegDataset(Dataset):
    """Returns (image, mask, skeleton, width) all at ``img_size``.

    Images -> float tensor [0,1]; mask/skeleton -> float {0,1}; width -> float [0,1].
    """

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        img_size: int = IMG_SIZE,
        cache: bool = True,
        cache_dir: Optional[str] = None,
        augment: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = int(augment)  # 0 none | 1 scale+dihedral (v9) | 2 affine
        if split:
            self.image_dir = os.path.join(root_dir, split, "images")
            self.mask_dir = os.path.join(root_dir, split, "masks")
        else:
            self.image_dir = os.path.join(root_dir, "images")
            self.mask_dir = os.path.join(root_dir, "masks")
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

    def _get_gt(self, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base = os.path.splitext(name)[0]
        cache_path = os.path.join(self.cache_dir, base + ".npz")
        if not (self.cache and os.path.exists(cache_path)):
            build_curve_gt(self._mask_path(name), cache_path, self.img_size)
        d = np.load(cache_path)
        return d["mask"], d["skeleton"], d["width"]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        image = self.img_transform(image)

        mask, skel, width = self._get_gt(name)
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        skel = torch.from_numpy(skel.astype(np.float32)).unsqueeze(0)
        width = torch.from_numpy(width.astype(np.float32)).unsqueeze(0)

        if self.augment:
            S = self.img_size
            if self.augment == 2:
                # ---- affine: rotation (any angle) + scale + translation ----
                ang = float(torch.empty(1).uniform_(-180.0, 180.0)) * math.pi / 180.0
                sc = float(torch.empty(1).uniform_(0.4, 1.6))
                tx = float(torch.empty(1).uniform_(-0.12, 0.12))
                ty = float(torch.empty(1).uniform_(-0.12, 0.12))
                cos, sin = math.cos(ang) * sc, math.sin(ang) * sc
                theta = torch.tensor([[[cos, -sin, tx], [sin, cos, ty]]], dtype=torch.float32)

                def _aff(t, mode):
                    grid = F.affine_grid(theta, (1, t.shape[0], S, S), align_corners=False)
                    return F.grid_sample(t.unsqueeze(0), grid, mode=mode,
                                         padding_mode='zeros', align_corners=False)[0]

                image = _aff(image, 'bilinear')
                mask = _aff(mask, 'nearest')
                skel = _aff(skel, 'nearest')
                width = _aff(width, 'bilinear')
            else:
                # ---- v9: random scale + dihedral ----
                if torch.rand(1).item() < 0.7:
                    scale = float(torch.empty(1).uniform_(0.4, 1.6).item())
                    ns = max(64, int(round(S * scale)))
                    def _rs(t, mode):
                        kw = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
                        return F.interpolate(t.unsqueeze(0), size=(ns, ns), mode=mode, **kw)[0]
                    image = _rs(image, 'bilinear'); mask = _rs(mask, 'nearest')
                    skel = _rs(skel, 'nearest'); width = _rs(width, 'bilinear')
                    if ns >= S:
                        x0 = int(torch.randint(0, ns - S + 1, (1,)).item())
                        y0 = int(torch.randint(0, ns - S + 1, (1,)).item())
                        image = image[:, y0:y0 + S, x0:x0 + S]
                        mask = mask[:, y0:y0 + S, x0:x0 + S]
                        skel = skel[:, y0:y0 + S, x0:x0 + S]
                        width = width[:, y0:y0 + S, x0:x0 + S]
                    else:
                        pad = S - ns
                        x0 = int(torch.randint(0, pad + 1, (1,)).item())
                        y0 = int(torch.randint(0, pad + 1, (1,)).item())
                        def _place(t):
                            c = torch.zeros((t.shape[0], S, S), dtype=t.dtype, device=t.device)
                            c[:, y0:y0 + ns, x0:x0 + ns] = t
                            return c
                        image, mask, skel, width = _place(image), _place(mask), _place(skel), _place(width)
                k = torch.randint(0, 4, (1,)).item()
                if k:
                    image = torch.rot90(image, k, [1, 2]); mask = torch.rot90(mask, k, [1, 2])
                    skel = torch.rot90(skel, k, [1, 2]); width = torch.rot90(width, k, [1, 2])

            # ---- flips ----
            if torch.rand(1).item() < 0.5:
                image = torch.flip(image, [2]); mask = torch.flip(mask, [2])
                skel = torch.flip(skel, [2]); width = torch.flip(width, [2])
            if torch.rand(1).item() < 0.5:
                image = torch.flip(image, [1]); mask = torch.flip(mask, [1])
                skel = torch.flip(skel, [1]); width = torch.flip(width, [1])

            # ---- photometric jitter (brightness / contrast / gamma) ----
            if torch.rand(1).item() < 0.5:
                image = torch.clamp(image * (1.0 + 0.3 * float(torch.randn(1))) + 0.1 * float(torch.randn(1)), 0, 1)
            if torch.rand(1).item() < 0.3:
                image = torch.clamp(image, 1e-4, 1.0) ** (1.0 + 0.5 * float(torch.randn(1)))

        return image, mask, skel, width, name


def prepare(root_dir: str, split: Optional[str] = None, img_size: int = IMG_SIZE,
            cache_dir: Optional[str] = None) -> int:
    """Pre-compute and cache skeleton/width for a whole split."""
    ds = CurveSegDataset(root_dir, split, img_size, cache=True, cache_dir=cache_dir)
    for name in ds.files:
        ds._get_gt(name)
    return len(ds.files)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute skeleton/width GT")
    parser.add_argument("--data_root", type=str, default="./datasets/DSCA")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    args = parser.parse_args()
    n = prepare(args.data_root, args.split or None, args.img_size)
    print(f"Prepared curve GT for {n} samples in {args.data_root}")
