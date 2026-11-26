"""Deployment-time adaptive rank for the native-3D model (same recipe as 2D).

The contribution spectrum is computed on the k training volumes through the
adapter activations, and the SVD output-subspace projection is applied to every
adapter's up-projection.  No layer deletion, no retraining.

Usage:
  python scripts/rank_calib_3d.py --run-dir runs/r08_attr --k 10 \
      --data-dir data/tof_mra_08mm --budgets 0.6,0.4
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

from dla.convdec_attr import build_sam3d_convdec_attr  # noqa: E402
from dla.data import load_volume  # noqa: E402
from scripts.train_fewshot import evaluate, split_cases  # noqa: E402


def adapter_mods(model):
    enc = model.image_encoder
    mods = [("int", i, a) for i, a in enumerate(enc.internal_adapters)]
    mods += [("ext", i, a) for i, a in enumerate(enc.external_adapters)]
    return mods


def calibrate(model, images, device):
    sums, cnt, hooks = {}, {}, []
    for kind, i, ad in adapter_mods(model):
        key = f"{kind}{i}"

        def h(mod, inp, out, key=key):
            v = out.detach().float()
            s = v.pow(2).sum(dim=(0, 1, 2, 3))       # [hidden]
            sums[key] = sums.get(key, torch.zeros_like(s)) + s
            cnt[key] = cnt.get(key, 0) + v.numel() // v.shape[-1]

        hooks.append(ad.act.register_forward_hook(h))
    with torch.no_grad():
        for im in images:
            model.image_encoder(im.to(device))
    for h in hooks:
        h.remove()
    return {k: (sums[k] / max(cnt[k], 1)).cpu().numpy() for k in sums}


def apply_rank(model, stats, budget):
    ks = []
    for kind, i, ad in adapter_mods(model):
        key = f"{kind}{i}"
        W = ad.up.weight.data.float()                # [dim, hidden]
        w2 = (W ** 2).sum(dim=0).clamp_min(1e-12)
        c = torch.tensor(stats[key], device=W.device, dtype=torch.float32)
        h2 = c / w2
        A = W * torch.sqrt(h2).unsqueeze(0)
        U, S, _ = torch.linalg.svd(A, full_matrices=False)
        k = max(1, int(round(budget * U.shape[1])))
        P = U[:, :k] @ U[:, :k].t()
        ad.up.weight.data = (P @ W).to(ad.up.weight.dtype)
        ks.append(k)
    return ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/r08_attr")
    ap.add_argument("--data-dir", default="data/tof_mra_08mm")
    ap.add_argument("--ckpt", default="../ckpt/sam_med3d_turbo.pth")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--budgets", default="1.0,0.6,0.4")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    with open(os.path.join(args.data_dir, "index.json")) as f:
        index = json.load(f)["cases"]
    train_cases, test_cases = split_cases(index, args.k, args.n_test, args.split_seed)

    model = build_sam3d_convdec_attr(checkpoint=args.ckpt, adapter_ratio=0.25,
                                     ext_levels=(0, 3, 6, 9, 11)).to(device)
    ck = torch.load(os.path.join(args.run_dir, "best.pth"), map_location="cpu")
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    print(f"loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    model.set_dla(True)

    images = []
    for c in train_cases:
        img, seg, _ = load_volume(os.path.join(args.data_dir, c["file"]))
        images.append(torch.from_numpy(img)[None, None])
    stats = calibrate(model, images, device)
    print(f"calibrated on {len(images)} volumes, "
          f"{len(stats)} adapters, hidden={len(next(iter(stats.values())))}", flush=True)

    for b in [float(v) for v in args.budgets.split(",")]:
        if b < 1.0:
            ks = apply_rank(model, stats, b)
            print(f"budget={b:.2f} mean_k={np.mean(ks):.1f}/{len(next(iter(stats.values())))}",
                  flush=True)
        res = evaluate(model, test_cases, args.data_dir, device)
        print(f"RESULT budget={b:.2f} test_dice={res['dice_mean']:.4f} "
              f"median={res['dice_median']:.4f} n={res['n_test']}", flush=True)


if __name__ == "__main__":
    main()
