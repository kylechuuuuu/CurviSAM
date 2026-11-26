"""Rank allocation v2: two-factor allocation and subspace projection.

Motivation from rank_spectrum.py:
  * total_l = sum_j c_{l,j} spans two orders of magnitude across adapters
  * pr_l (effective rank) is *uncorrelated* with total_l (Spearman -0.05)
  * up_proj columns have mean |cos| ~0.25, so keeping coordinate axes is lossy

So the allocation should use both magnitude and spectral shape, and the
pruning should happen in the optimal subspace rather than on axes.

Allocators (all keep the total channel budget K = budget * 48 * d):
  uniform     k_l = K / 48
  importance  k_l ~ sqrt(total_l)                      (previous default)
  global      one global threshold on c_{l,j}: keep the K largest entries
              -> adapts to magnitude and shape simultaneously
  waterfill   k_l = min(d, lambda * total_l), sum = K  (uses the full budget;
              importance+clip loses 19% of the budget at budget=1.0)

Pruners:
  mask  zero the low-contribution bottleneck channels (axis aligned)
  svd   keep the top-k left singular subspace of W_up diag(sqrt(E[h^2]));
        this is the optimal rank-k approximation of the adapter's output map
        and costs nothing at inference (folded into W_up).
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from rank_calib import adapters, apply_masks, build, calibrate, evaluate  # noqa: E402


def alloc_uniform(n, d, budget):
    k = int(round(budget * d))
    return np.full(n, min(max(k, 1), d), dtype=int)


def alloc_prop(w, d, budget, power):
    K = budget * d * len(w)
    ww = w ** power
    return np.clip(np.round(ww / ww.sum() * K), 1, d).astype(int)


def alloc_global(cs, d, budget):
    K = int(round(budget * d * len(cs)))
    flat = np.concatenate(cs)
    order = np.argsort(-flat)
    lam = flat[order[min(K, len(flat)) - 1]]
    ks = np.array([int((c >= lam).sum()) for c in cs], dtype=int)
    # make sure the budget is met exactly after ties
    return np.clip(ks, 1, d)


def alloc_waterfill(w, d, budget):
    K = budget * d * len(w)
    lo, hi = 1e-12, 1e9
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        s = np.minimum(d, lam * w).sum()
        hi, lo = (lam, lo) if s > K else (hi, lam)
    k = np.floor(0.5 * (lo + hi) * w).astype(int)
    return np.clip(k, 1, d)


def masks_from_ks(cs, keys, ks):
    masks = {}
    for c, key, k in zip(cs, keys, ks):
        order = np.argsort(-c)
        m = np.zeros_like(c)
        m[order[:int(k)]] = 1.0
        masks[key] = m
    return masks


def apply_svd(m, scores, keys, cs, ks):
    """Fold a rank-k output projection into every adapter's up_proj."""
    with torch.no_grad():
        for (i, ext, ad), key, c, k in zip(adapters(m), keys, cs, ks):
            W = ad.up_proj.weight.data.float()
            w2 = (W ** 2).sum(dim=0).clamp_min(1e-12)
            h2 = torch.tensor(c, device=W.device, dtype=torch.float32) / w2
            A = W * torch.sqrt(h2).unsqueeze(0)          # (out, d)
            U, S, _ = torch.linalg.svd(A, full_matrices=False)
            kk = int(min(k, U.shape[1]))
            P = U[:, :kk] @ U[:, :kk].t()
            ad.up_proj.weight.data = (P @ W).to(ad.up_proj.weight.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/dev/shm/tpami_orig/orig_base_v16_best.pth")
    ap.add_argument("--adapter_dim_ratio", type=float, default=0.1)
    ap.add_argument("--use_sdf", type=int, default=0)
    ap.add_argument("--tasks", required=True,
                    help="comma separated  pruner:allocator:budget:dataset")
    args = ap.parse_args()

    m = build(args.ckpt, use_sdf=bool(args.use_sdf),
          adapter_dim_ratio=args.adapter_dim_ratio)
    scores, _ = calibrate(m, "contrib")
    keys = [(i, ext) for i, ext, _ in adapters(m)]
    d = len(scores[keys[0]])
    cs = [np.maximum(scores[k].astype(np.float64), 0.0) for k in keys]
    w = np.array([c.sum() for c in cs])
    w = w / w.mean()
    orig = [ad.up_proj.weight.data.clone() for _, _, ad in adapters(m)]

    for task in args.tasks.split(","):
        pruner, alloc, budget, ds = task.split(":")
        budget = float(budget)
        if alloc == "uniform":
            ks = alloc_uniform(len(keys), d, budget)
        elif alloc == "importance":
            ks = alloc_prop(w, d, budget, 0.5)
        elif alloc == "linear":
            ks = alloc_prop(w, d, budget, 1.0)
        elif alloc == "global":
            ks = alloc_global(cs, d, budget)
        elif alloc == "waterfill":
            ks = alloc_waterfill(w, d, budget)
        else:
            raise ValueError(alloc)

        for (_, _, ad), o in zip(adapters(m), orig):
            ad.up_proj.weight.data = o.clone()

        hooks = []
        if pruner == "mask":
            hooks = apply_masks(m, masks_from_ks(cs, keys, ks))
        elif pruner == "svd":
            apply_svd(m, scores, keys, cs, ks)
        else:
            raise ValueError(pruner)

        dice = evaluate(m, f"datasets/{ds}")
        for h in hooks:
            h.remove()
        print(f"RESULT pruner={pruner} alloc={alloc} budget={budget:.2f} "
              f"k_total={int(ks.sum())} r_actual={ks.mean()/d:.3f} {ds} dice={dice:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
