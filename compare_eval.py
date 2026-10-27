"""Compare checkpoints on curve datasets, optionally with dihedral TTA.

For each dataset and each output key ('masks' = fused, 'region') it reports:
  dice, clDice (topology, soft-skeleton), skeleton recall (tol px).

Usage:
  python compare_eval.py --weights best_model_curve_v4.pth --tta \
      --datasets datasets/DSCA datasets/ROAD
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from segment_anything import build_sam_vit_l
from losses import SoftSkeletonize

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
        gt = torch.from_numpy((m > 0).astype(np.float32)).unsqueeze(0)
        return x, gt


def dice_np(pred, true):
    p = pred.reshape(-1).astype(np.float64)
    t = true.reshape(-1).astype(np.float64)
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    return (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)


def cl_dice_prob(prob, target, skel, iters=10):
    sp = skel(prob, iters)
    st = skel(target, iters)
    tprec = (sp * target).sum() / (sp.sum() + 1e-6)
    tsens = (st * prob).sum() / (st.sum() + 1e-6)
    return float((2 * tprec * tsens) / (tprec + tsens + 1e-6))


def evaluate(weights, dataset_roots, device, skel_tol=2, tta=False):
    sam = build_sam_vit_l(checkpoint=None, use_adapter=True, adapter_dim_ratio=0.1,
                          use_detail=True).to(device)
    ck = torch.load(weights, map_location=device)
    missing, unexpected = sam.load_state_dict(ck["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"  (strict=False: {len(missing)} missing, {len(unexpected)} unexpected)")
    sam.eval()
    print(f"weights: {weights} (ckpt f1={ck.get('f1_score', float('nan')):.4f}) tta={tta}")

    sp_e = torch.zeros((1, 0, 256), device=device)
    de = torch.zeros((1, 256, 64, 64), device=device)
    skel = SoftSkeletonize()

    def forward(x):
        detail = sam.mask_decoder.encode_detail(x)
        ie, ef, af = sam.image_encoder(x, return_features=True)
        return sam.mask_decoder(image_embeddings=ie, image_pe=sam.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sp_e, dense_prompt_embeddings=de,
                                multimask_output=False, adapter_features=af,
                                detail_features=detail)

    def probs(x, keys):
        out = forward(x)
        acc = {k: torch.sigmoid(out[k]) for k in keys}
        if not tta:
            return acc
        acc = {k: torch.zeros_like(acc[k]) for k in keys}
        n = 0
        for r in range(4):
            xk = torch.rot90(x, r, [2, 3])
            for flip in (False, True):
                xf = torch.flip(xk, [3]) if flip else xk
                o = forward(xf)
                for k in keys:
                    p = torch.sigmoid(o[k])
                    if flip:
                        p = torch.flip(p, [3])
                    p = torch.rot90(p, -r, [2, 3])
                    acc[k] = acc[k] + p
                n += 1
        return {k: acc[k] / n for k in keys}

    for root in dataset_roots:
        ds = DS(root)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
        keys = ["masks", "region"]
        res = {f"{k}_{m}": [] for k in keys for m in ["dice", "cldice", "skelrec"]}
        with torch.no_grad():
            for x, gt in dl:
                x = x.to(device)
                gt = gt.to(device)
                gt_bin = (gt > 0.5).float()
                h, w = gt.shape[-2:]
                g = gt_bin[0, 0].cpu().numpy().astype(np.uint8)
                gs = (cv2.ximgproc.thinning((g * 255).astype(np.uint8)) > 127).astype(np.uint8)
                gm = F.interpolate(gt_bin, size=(512, 512), mode="bilinear", align_corners=False)
                ps = probs(x, keys)
                for key in keys:
                    p = F.interpolate(ps[key], size=(h, w), mode="bilinear", align_corners=False)
                    pb = (p[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
                    res[f"{key}_dice"].append(dice_np(pb, g))
                    p512 = F.interpolate(ps[key], size=(512, 512), mode="bilinear", align_corners=False)
                    res[f"{key}_cldice"].append(cl_dice_prob(p512, gm, skel))
                    if gs.sum() > 0:
                        kk = 2 * skel_tol + 1
                        pd = cv2.dilate(pb, np.ones((kk, kk), np.uint8))
                        res[f"{key}_skelrec"].append(float((gs * pd).sum()) / float(gs.sum()))
                    else:
                        res[f"{key}_skelrec"].append(float("nan"))
        print(f"\n== {root} (n={len(ds)}) ==")
        for key in keys:
            print(f"   [{key:6s}] dice={np.nanmean(res[key + '_dice']):.4f}  "
                  f"clDice={np.nanmean(res[key + '_cldice']):.4f}  "
                  f"skelrecall={np.nanmean(res[key + '_skelrec']):.4f} (tol={skel_tol}px)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=["datasets/DSCA", "datasets/ROAD"])
    parser.add_argument("--skel_tol", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args.weights, args.datasets, device, args.skel_tol, args.tta)
