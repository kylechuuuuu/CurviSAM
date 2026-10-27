"""Multi-decoder ensemble evaluation (no TTA).

Averages the per-model probabilities on the *fused* and/or *region* output.
Each model's logits are interpolated to the original image size and sigmoided
first (same protocol as test_dice), then averaged and thresholded at 0.5.

Usage:
  python ensemble_eval.py --weights a.pth b.pth c.pth --keys masks region \
      --datasets datasets/DSCA datasets/ROAD
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from segment_anything import build_sam_vit_l

SUPPORTED = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class DS(Dataset):
    def __init__(self, root):
        self.i = os.path.join(root, "images")
        self.m = os.path.join(root, "masks")
        self.f = []
        for name in os.listdir(self.i):
            if name.lower().endswith(SUPPORTED):
                b = os.path.splitext(name)[0]
                if any(os.path.exists(os.path.join(self.m, b + e)) for e in SUPPORTED):
                    self.f.append(name)
        self.f.sort()

    def __len__(self):
        return len(self.f)

    def __getitem__(self, k):
        name = self.f[k]
        b = os.path.splitext(name)[0]
        img = Image.open(os.path.join(self.i, name)).convert("RGB")
        m = np.array(Image.open(os.path.join(self.m, b + ".png")).convert("L"))
        x = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])(img)
        return x, torch.from_numpy((m > 0).astype(np.float32))[None]


def dice(pred_bin, g):
    p = pred_bin.reshape(-1).astype(np.float64)
    t = g.reshape(-1).astype(np.float64)
    return (2 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)


def load(weights, device):
    sam = build_sam_vit_l(checkpoint=None, use_adapter=True, adapter_dim_ratio=0.1,
                          use_detail=False).to(device)
    sam.load_state_dict(torch.load(weights, map_location=device)["model_state_dict"], strict=False)
    sam.eval()
    return sam


def evaluate(weights_list, keys, dataset_roots, device):
    models = [load(w, device) for w in weights_list]
    print(f"ensembling {len(models)} models: {[os.path.basename(w) for w in weights_list]}")
    sp = torch.zeros((1, 0, 256), device=device)
    de = torch.zeros((1, 256, 64, 64), device=device)

    def logits(sam, x, k):
        ie, ef, af = sam.image_encoder(x, return_features=True)
        o = sam.mask_decoder(image_embeddings=ie, image_pe=sam.prompt_encoder.get_dense_pe(),
                             sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                             multimask_output=False, adapter_features=af)
        return o[k]

    for root in dataset_roots:
        ds = DS(root)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
        res = {f"{k}_ens": [] for k in keys}
        singles = {f"{k}_{i}": [] for k in keys for i in range(len(models))}
        with torch.no_grad():
            for x, gt in dl:
                x = x.to(device)
                h, w = gt.shape[-2:]
                g = gt[0, 0].numpy().astype(np.uint8)
                for k in keys:
                    probs = []
                    for i, sam in enumerate(models):
                        lo = F.interpolate(logits(sam, x, k), size=(h, w),
                                           mode="bilinear", align_corners=False)
                        p = torch.sigmoid(lo)
                        probs.append(p)
                        singles[f"{k}_{i}"].append(dice((p[0, 0] > 0.5).cpu().numpy(), g))
                    ens = torch.stack(probs).mean(0)
                    res[f"{k}_ens"].append(dice((ens[0, 0] > 0.5).cpu().numpy(), g))
        print(f"\n== {root} (n={len(ds)}) ==")
        for k in keys:
            print(f"   [{k}] singles=" +
                  " ".join(f"{np.mean(singles[f'{k}_{i}']):.4f}" for i in range(len(models))) +
                  f"  ENSEMBLE={np.mean(res[f'{k}_ens']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, nargs="+", required=True)
    parser.add_argument("--keys", type=str, nargs="+", default=["masks", "region"])
    parser.add_argument("--datasets", type=str, nargs="+", default=["datasets/DSCA", "datasets/ROAD"])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args.weights, args.keys, args.datasets, device)
