"""Train a 3D decoder on cached slice embeddings.

Two decoders share the same frozen encoder, the same cached embeddings and the
same slabs, so the comparison is internal:

  --decoder plain   MaskDecoder3D (Conv3d stack, mask only)
  --decoder attr    MaskDecoder3DAttr (Conv3d + centreline / radius / SDF)

The encoder is the frozen 2D SAM ViT-L with dual-level adapters; embeddings are
cached once per slice.
"""
import argparse
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import build_sam_vit_l
from segment_anything.modeling.mask_decoder_3d import MaskDecoder3D
from mask_decoder_3d_attr import MaskDecoder3DAttr
from curve_data_3d import derive_fields_3d


def read_mhd(path):
    txt = open(path).read()
    x, y, z = [int(v) for v in re.search(r"DimSize = (.+)", txt).group(1).split()]
    dtype = {"MET_SHORT": "<i2", "MET_UCHAR": "u1"}[
        re.search(r"ElementType = (\S+)", txt).group(1)]
    raw = os.path.join(os.path.dirname(path),
                       re.search(r"ElementDataFile = (\S+)", txt).group(1))
    return np.fromfile(raw, dtype=dtype).reshape(z, y, x)


def window2d(a):
    lo, hi = np.percentile(a, [0.5, 99.5])
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return ((np.clip(a, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def dice_np(p, g):
    return 2.0 * np.logical_and(p, g).sum() / max(p.sum() + g.sum(), 1)


def dice_loss(p, g, smooth=1e-6):
    inter = (p * g).sum()
    return 1.0 - (2.0 * inter + smooth) / (p.sum() + g.sum() + smooth)


def build_encoder(ckpt, device):
    sam = build_sam_vit_l(checkpoint=None, use_adapter=True, adapter_dim_ratio=0.1)
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    info = sam.load_state_dict(sd, strict=False)
    miss = [k for k in info.missing_keys if k.startswith("image_encoder")]
    print(f"encoder: missing={len(miss)} unexpected={len(info.unexpected_keys)}", flush=True)
    if miss[:3]:
        print("  missing sample:", miss[:3], flush=True)
    for p in sam.parameters():
        p.requires_grad = False
    return sam.to(device).eval()


@torch.no_grad()
def embed_slice(sam, sl, device):
    x = torch.from_numpy(sl)[None, None].repeat(1, 3, 1, 1).float() / 255.0
    x = F.interpolate(x, size=(1024, 1024), mode="bilinear",
                      align_corners=False).to(device)
    out = sam.image_encoder(x, return_features=True)
    emb = out[0] if isinstance(out, tuple) else out
    return emb[0].half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", choices=["plain", "attr"], default="attr")
    ap.add_argument("--data_root", default="datasets/vessel12/vessel12")
    ap.add_argument("--case", default="VESSEL12_23")
    ap.add_argument("--ckpt", default="work/orig_base_v16_best.pth")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--slabs", type=int, default=40)
    ap.add_argument("--attr_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="work/dec3d_attr.pth")
    ap.add_argument("--rank_budget", type=float, default=0.0,
                    help="adaptive rank: SVD output-subspace budget (0 = off, 0.40 recommended)")
    ap.add_argument("--calib_root", default="datasets/eee_v16/train/images")
    ap.add_argument("--holdout", type=int, default=10,
                    help="number of slabs held out for evaluation (0 = evaluate on all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    sam = build_encoder(args.ckpt, device)

    if args.rank_budget > 0:
        import rank_calib
        import rank_v2
        scores, n_cal = rank_calib.calibrate(sam, "contrib")
        keys = [(i, ext) for i, ext, _ in rank_calib.adapters(sam)]
        cs = [np.maximum(scores[k].astype(np.float64), 0.0) for k in keys]
        w = np.array([c.sum() for c in cs])
        w = w / w.mean()
        d = len(cs[0])
        ks = rank_v2.alloc_prop(w, d, args.rank_budget, 0.5)
        rank_v2.apply_svd(sam, scores, keys, cs, ks)
        print(f"adaptive rank: calibrated on {n_cal} images, budget={args.rank_budget} "
              f"mean_r={ks.mean() / d:.3f} (no layer deletion)", flush=True)

    vol = read_mhd(os.path.join(args.data_root, "Scans", args.case + ".mhd"))
    gt = read_mhd(os.path.join(args.data_root, "Lungmasks", args.case + ".mhd")) > 0
    z = vol.shape[0]
    K = args.K
    n_slabs = 6 if args.smoke else args.slabs
    epochs = 2 if args.smoke else args.epochs

    centers = np.linspace(K // 2, z - 1 - K // 2, n_slabs).round().astype(int)
    need = sorted({int(i) for c in centers for i in range(c - K // 2, c + K // 2)})
    print(f"{len(need)} slices to embed ({args.case}, z={z})", flush=True)
    cache = {}
    for n, i in enumerate(need):
        cache[i] = embed_slice(sam, window2d(vol[i]), device)
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(need)}", flush=True)

    slabs = []
    for c in centers:
        idx = [min(max(j, 0), z - 1) for j in range(c - K // 2, c - K // 2 + K)]
        emb = np.stack([cache[j] for j in idx]).astype(np.float32).transpose(1, 0, 2, 3)
        msk = gt[idx].astype(np.float32)
        attr = None
        if args.decoder == "attr":
            small = F.interpolate(torch.from_numpy(msk)[None, None], size=(K, args.attr_size, args.attr_size),
                                  mode="trilinear", align_corners=False)[0, 0].numpy()
            attr = derive_fields_3d(small)
        slabs.append((emb, msk, attr))
    del cache

    if args.decoder == "plain":
        dec = MaskDecoder3D().to(device)
    else:
        dec = MaskDecoder3DAttr(upsample=args.attr_size // 64).to(device)
    print(f"decoder={args.decoder} trainable={sum(p.numel() for p in dec.parameters())/1e3:.0f}K",
          flush=True)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(slabs))
    if args.holdout > 0:
        test_ids = sorted(perm[:args.holdout].tolist())
        test_set = set(test_ids)
        train_slabs = [s for i, s in enumerate(slabs) if i not in test_set]
        test_slabs = [slabs[i] for i in test_ids]
        print(f"split: {len(train_slabs)} train / {len(test_slabs)} test slabs", flush=True)
    else:
        train_slabs, test_slabs = slabs, slabs

    def logits_of(out, K):
        m = out["masks"] if isinstance(out, dict) else out
        return F.interpolate(m, size=(K, 512, 512), mode="trilinear",
                             align_corners=False)

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    bce = nn.BCEWithLogitsLoss()
    best = 0.0

    for ep in range(epochs):
        dec.train()
        order = np.random.permutation(len(train_slabs))
        tot = 0.0
        for si in order:
            emb, msk, attr = train_slabs[si]
            e = torch.from_numpy(emb)[None].to(device)
            m = torch.from_numpy(msk)[None, None].to(device)
            out = dec(e)
            logits = logits_of(out, K)
            loss = 0.7 * bce(logits, m) + 0.3 * dice_loss(torch.sigmoid(logits), m)
            if attr is not None:
                a = {k: torch.from_numpy(v)[None, None].to(device) for k, v in attr.items()}
                c = out["skeleton"]
                loss = loss + 0.3 * (bce(c, a["skel"]) + dice_loss(torch.sigmoid(c), a["skel"]))
                band = a["band"]
                wl = F.smooth_l1_loss(out["width"] * band, a["width"] * band,
                                      reduction="sum") / band.sum().clamp_min(1.0)
                loss = loss + 0.3 * wl
                valid = (a["sdf"].abs() < 2.0).float()
                sl = F.smooth_l1_loss(out["sdf"] * valid, a["sdf"] * valid,
                                      reduction="sum") / valid.sum().clamp_min(1.0)
                loss = loss + 0.2 * sl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()

        dec.eval()
        ds = []
        with torch.no_grad():
            for emb, msk, _ in test_slabs:
                e = torch.from_numpy(emb)[None].to(device)
                out = dec(e)
                logits = logits_of(out, K)
                p = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0, 0]
                ds.append(dice_np(p, msk))
        mean_d = float(np.mean(ds))
        if mean_d > best:
            best = mean_d
            torch.save({"decoder": args.decoder, "state_dict": dec.state_dict(),
                        "K": K, "epoch": ep + 1, "dice": mean_d}, args.out)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1}/{epochs} loss={tot/len(slabs):.4f} slabDice={mean_d:.4f} best={best:.4f}",
                  flush=True)

    print(f"done: best slab Dice = {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
