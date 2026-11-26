"""Few-shot fine-tuning of SAM-Med3D with / without the dual-level adapter (DLA).

SACM prompt setting: `prompt_encoder(points=None, boxes=None, masks=None)` -- no
prompt at all, task = image -> mask (same as train_sam.py in kylechuuuuu/SACM).

Protocol used for the k=10 runs:
  * `k` volumes for training, a *disjoint* `--n-test` volumes for testing
  * `--epochs` epochs (one epoch = one pass over the k training volumes)
  * after every epoch the model is scored **on the training volumes** and the
    weights with the best train Dice are kept (`--select-best 1`)
  * only that single checkpoint is written (`best.pth`), then tested on the
    held-out volumes

variants:
    zero  -- no training, pretrained SAM-Med3D
    dec   -- train mask decoder only (encoder frozen, adapters disabled)
    dla   -- SACM setting: encoder frozen + both adapter levels + decoder
    full  -- full fine-tuning (all 105.8 M parameters)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dla.data import load_volume                                    # noqa: E402
from dla.metrics import dice                                        # noqa: E402
from dla.model import build_sam3d_dla, trainable_parameters         # noqa: E402
from dla.convdec import build_sam3d_convdec, trainable_conv          # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-test", type=int, default=10, help="0 = all held-out volumes")
    p.add_argument("--variant", choices=["zero", "dec", "dla", "full", "frozen", "adapt"], default="dla")
    p.add_argument("--init-from", default="", help="load a previous best.pth before training")
    p.add_argument("--decoder", choices=["transformer", "conv3d", "conv3d_attr"], default="transformer")
    p.add_argument("--conv-stem", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--iters", type=int, default=-1, help="override total steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--adapter-lr", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--adapter-ratio", type=float, default=0.25)
    p.add_argument("--levels", default="0,3,6,9,11")
    p.add_argument("--pos-weight", type=float, default=5.0)
    p.add_argument("--shift-aug", type=int, default=0)
    p.add_argument("--select-best", type=int, default=1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save-ckpt", type=int, default=1)
    return p.parse_args()


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def split_cases(index: List[dict], k: int, n_test: int, seed: int):
    idx = list(range(len(index)))
    random.Random(seed).shuffle(idx)
    train = [index[i] for i in idx[:k]]
    rest = idx[k:]
    if n_test > 0:
        rest = rest[:n_test]
    return train, [index[i] for i in sorted(rest)]


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    """Unified forward: transformer mask decoder or 3D conv decoder."""
    if hasattr(model, "decoder"):                     # Sam3D_ConvDec
        out = model(x)
        return out["masks"] if isinstance(out, dict) else out
    emb, feats = model.image_encoder(x)
    sparse, dense = model.prompt_encoder(points=None, boxes=None, masks=None)
    low, _ = model.mask_decoder(
        image_embeddings=emb, image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False, adapter_features=feats)
    return low


@torch.no_grad()
def evaluate(model, cases: List[dict], data_dir: str, device) -> Dict:
    """No-prompt inference (SACM setting): image -> mask."""
    model.eval()
    per_case, all_dice = [], []
    for case in cases:
        img, seg, _ = load_volume(os.path.join(data_dir, case["file"]))
        x = torch.from_numpy(img)[None, None].to(device)
        low = forward_logits(model, x)
        prob = torch.sigmoid(F.interpolate(low, size=seg.shape, mode="trilinear",
                                           align_corners=False))
        pred = (prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        d = dice(pred, seg)
        per_case.append({"case": case["case"], "dice": float(d)})
        all_dice.append(d)
    return {"dice_mean": float(np.mean(all_dice)), "dice_std": float(np.std(all_dice)),
            "dice_median": float(np.median(all_dice)), "n_test": len(all_dice),
            "per_case": per_case}


def augment(img: torch.Tensor, seg: torch.Tensor, rng: random.Random):
    for ax in (2, 3, 4):
        if rng.random() < 0.5:
            img = torch.flip(img, dims=(ax,))
            seg = torch.flip(seg, dims=(ax,))
    k = rng.randrange(4)
    if k:
        img = torch.rot90(img, k, dims=(3, 4))
        seg = torch.rot90(seg, k, dims=(3, 4))
    return img, seg


def shift(img: torch.Tensor, seg: torch.Tensor, max_s: int, rng: random.Random):
    if max_s <= 0:
        return img, seg
    d, h, w = img.shape[-3:]
    off = [rng.randint(-max_s, max_s) for _ in range(3)]
    img, seg = F.pad(img, (max_s,) * 6), F.pad(seg, (max_s,) * 6)
    s = [max_s + o for o in off]
    return (img[..., s[0]:s[0] + d, s[1]:s[1] + h, s[2]:s[2] + w],
            seg[..., s[0]:s[0] + d, s[1]:s[1] + h, s[2]:s[2] + w])


def soft_dice(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return 1 - (2 * (prob * target).sum()) / (prob.sum() + target.sum() + eps)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.data_dir, "index.json")) as f:
        index = json.load(f)["cases"]
    train_cases, test_cases = split_cases(index, args.k, args.n_test, args.split_seed)

    run_name = args.run_name or f"{os.path.basename(args.data_dir)}_k{args.k}_{args.variant}"
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    levels = tuple(int(v) for v in args.levels.split(","))
    if args.decoder == "conv3d":
        model = build_sam3d_convdec(checkpoint=args.ckpt, adapter_ratio=args.adapter_ratio,
                                    ext_levels=levels,
                                    use_stem=bool(args.conv_stem)).to(device)
    elif args.decoder == "conv3d_attr":
        from dla.convdec_attr import build_sam3d_convdec_attr
        model = build_sam3d_convdec_attr(checkpoint=args.ckpt, adapter_ratio=args.adapter_ratio,
                                         ext_levels=levels,
                                         use_stem=bool(args.conv_stem)).to(device)
    else:
        model = build_sam3d_dla(checkpoint=args.ckpt, adapter_ratio=args.adapter_ratio,
                                ext_levels=levels).to(device)

    result: Dict[str, object] = {
        "run": run_name, "config": vars(args).copy(), "prompt": "none (SACM setting)",
        "train_cases": [c["case"] for c in train_cases],
        "test_cases": [c["case"] for c in test_cases],
        "n_train": len(train_cases), "n_test": len(test_cases),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }

    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[init] loaded {len(sd)} tensors from {args.init_from} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if args.variant == "frozen":
            args.epochs = 0

    if args.variant in ("zero", "frozen"):
        t0 = time.time()
        result.update(evaluate(model, test_cases, args.data_dir, device),
                      train_time_sec=0.0, eval_time_sec=round(time.time() - t0, 1),
                      trainable_params=0)
    else:
        use_adapters = args.variant in ("dla", "full", "adapt")
        train_dec = args.variant != "adapt"
        if args.decoder in ("conv3d", "conv3d_attr"):
            params, n_train, n_total = trainable_conv(
                model, adapters=use_adapters, full=(args.variant == "full"))
            if not train_dec:
                for n_, p_ in model.named_parameters():
                    if n_.startswith("decoder."):
                        p_.requires_grad = False
                params = [p_ for p_ in model.parameters() if p_.requires_grad]
                n_train = sum(p_.numel() for p_ in params)
        else:
            params, n_train, n_total = trainable_parameters(
                model, adapters=use_adapters, decoder=train_dec,
                full=(args.variant == "full"))
        if not use_adapters:
            model.set_dla(False)
        print(f"[train] variant={args.variant} trainable={n_train/1e6:.2f}M / {n_total/1e6:.2f}M "
              f"| k={len(train_cases)} test={len(test_cases)} epochs={args.epochs}")

        if use_adapters and args.adapter_lr > 0:
            is_dec = ((lambda n: n.startswith("decoder.")) if args.decoder in ("conv3d", "conv3d_attr")
                      else (lambda n: "mask_decoder" in n))
            ad = [p for n, p in model.named_parameters() if p.requires_grad and not is_dec(n)]
            dec = [p for n, p in model.named_parameters() if p.requires_grad and is_dec(n)]
            opt = torch.optim.AdamW([{"params": dec, "lr": args.lr},
                                     {"params": ad, "lr": args.adapter_lr}],
                                    weight_decay=args.weight_decay)
            print(f"[train] decoder_lr={args.lr} adapter_lr={args.adapter_lr}")
        else:
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        train_data = []
        for c in train_cases:
            img, seg, _ = load_volume(os.path.join(args.data_dir, c["file"]))
            train_data.append((torch.from_numpy(img)[None, None],
                               torch.from_numpy(seg.astype(np.float32))[None, None]))

        steps_per_epoch = max(1, len(train_data))
        total = args.iters if args.iters > 0 else args.epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)

        pos_weight = torch.tensor(args.pos_weight, device=device)
        rng = random.Random(args.seed)
        best = {"dice": -1.0, "epoch": 0, "state": None}
        curve: List[dict] = []
        it = 0
        t0 = time.time()
        model.train()
        for ep in range(1, args.epochs + 1):
            order = list(range(len(train_data)))
            rng.shuffle(order)
            for idx in order:
                it += 1
                img, seg = train_data[idx]
                img, seg = shift(img.clone(), seg.clone(), args.shift_aug, rng)
                img, seg = augment(img, seg, rng)
                img, seg = img.to(device), seg.to(device)

                out = model(img) if args.decoder == "conv3d_attr" else forward_logits(model, img)
                low = out["masks"] if isinstance(out, dict) else out

                target = F.interpolate(seg, size=low.shape[2:], mode="trilinear",
                                       align_corners=False)
                bce = F.binary_cross_entropy_with_logits(low, target, pos_weight=pos_weight)
                loss = bce + soft_dice(torch.sigmoid(low), target)

                if isinstance(out, dict):
                    from dla.curve_data_3d import derive_fields_3d
                    f = derive_fields_3d(seg[0, 0].detach().cpu().numpy(), sdf_trunc=2.0)
                    at = {k: torch.from_numpy(v)[None, None].to(device) for k, v in f.items()}
                    c = out["skeleton"]
                    loss = loss + 0.3 * (F.binary_cross_entropy_with_logits(c, at["skel"])
                                         + soft_dice(torch.sigmoid(c), at["skel"]))
                    band = at["band"]
                    loss = loss + 0.3 * (F.smooth_l1_loss(out["width"] * band,
                                                          at["width"] * band,
                                                          reduction="sum")
                                         / band.sum().clamp_min(1.0))
                    valid = (at["sdf"].abs() < 2.0).float()
                    loss = loss + 0.2 * (F.smooth_l1_loss(out["sdf"] * valid,
                                                          at["sdf"] * valid,
                                                          reduction="sum")
                                         / valid.sum().clamp_min(1.0))

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()

            tr = evaluate(model, train_cases, args.data_dir, device)
            curve.append({"epoch": ep, "step": it, "loss": float(loss.item()),
                          "train_dice": tr["dice_mean"]})
            if args.select_best and tr["dice_mean"] > best["dice"]:
                best = {"dice": tr["dice_mean"], "epoch": ep,
                        "state": copy.deepcopy({k: v.detach().cpu()
                                                for k, v in model.state_dict().items()})}
            print(f"  ep {ep:2d} (it {it:4d}) loss {loss.item():.4f} "
                  f"train_dice {tr['dice_mean']:.4f} | best {best['dice']:.4f} @ep{best['epoch']} "
                  f"| {it/(time.time()-t0):.1f} it/s", flush=True)
            model.train()

        train_time = time.time() - t0
        if args.select_best and best["state"] is not None:
            model.load_state_dict(best["state"], strict=True)
            print(f"[select] restored best train-dice weights (ep {best['epoch']}, "
                  f"train_dice {best['dice']:.4f})")

        t1 = time.time()
        result.update(evaluate(model, test_cases, args.data_dir, device),
                      train_time_sec=round(train_time, 1),
                      eval_time_sec=round(time.time() - t1, 1),
                      trainable_params=n_train, curve=curve,
                      selected_epoch=best["epoch"], best_train_dice=best["dice"])

        if args.save_ckpt and best["state"] is not None:
            keep = {n for n, p in model.named_parameters() if p.requires_grad}
            best["state"] = {k: v for k, v in best["state"].items() if k in keep}
            torch.save({"state_dict": best["state"], "config": vars(args),
                        "best_train_dice": best["dice"], "best_epoch": best["epoch"]},
                       os.path.join(out_dir, "best.pth"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n== {run_name} ==\n  variant={args.variant} k={len(train_cases)} "
          f"n_test={result['n_test']} | best_train_dice={result.get('best_train_dice', 0):.4f} "
          f"(ep {result.get('selected_epoch', '-')}) -> TEST dice={result['dice_mean']:.4f} "
          f"(median {result['dice_median']:.4f})")


if __name__ == "__main__":
    main()

