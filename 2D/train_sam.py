"""Training entry point for tpami_sacm.

tpami_sacm = pre-trained SAM ViT-L with three requested changes:

1. Dual-level adapters: Adapter-I inside every Transformer block (MLP residual
   path) and Adapter-E at every block boundary (24 + 24 adapters).
2. Input-adaptive per-adapter scaling (new): a shared MLP reads the spatially
   pooled adapter input and predicts s = 1 + tanh(.) in (0, 2) for each
   (image, layer, branch).  Zero-initialised, so training starts exactly at the
   static dual-level adapter; a small (s-1)^2 budget term keeps it honest.
3. Two attribute decoders besides SAM's region decoder: a centerline branch and
   a log-radius (width) branch at 512, fused into a geometric residual that
   corrects the region mask (--use_sdf 0 keeps exactly these two).

Geometry targets follow curve_data.py: 1-px skeleton softened into a tolerance
band (1.0 / 0.5 / 0.2), log1p radius in 512-grid pixels supervised on the
skeleton band, and a truncated signed boundary distance.  Mask losses are
computed at 1024, clDice at 512.
"""

import argparse
import collections
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from segment_anything import build_sam_vit_l
from curve_data import CurveSegDataset
from losses import CurveSegLoss

try:
    from tqdm import tqdm
except ImportError:  # minimal fallback so the script runs without tqdm
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            pass


def setup_logging(args):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info("Training Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    return log_file


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_f1(pred_logits, true_masks, threshold=0.5):
    """Pooled (micro) binary F1, computed without scikit-learn."""
    pred = (torch.sigmoid(pred_logits) > threshold)
    true = (true_masks > threshold)
    tp = (pred & true).sum().item()
    fp = (pred & ~true).sum().item()
    fn = (~pred & true).sum().item()
    return (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)


def encode_and_decode(sam, images):
    """Prompt-free forward: empty sparse prompt + zero dense prompt."""
    x = images  # raw [0,1] tensors (SAM normalisation hurt in this pipeline)
    detail = sam.mask_decoder.encode_detail(x)
    encoder_output = sam.image_encoder(x, return_features=True)
    if len(encoder_output) == 3:
        image_embeddings, encoder_features, adapter_features = encoder_output
    else:
        image_embeddings, encoder_features = encoder_output
        adapter_features = None

    b = x.size(0)
    sparse_embeddings = torch.zeros((b, 0, 256), dtype=torch.float, device=x.device)
    dense_embeddings = torch.zeros((b, 256, 64, 64), dtype=torch.float, device=x.device)
    return sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        adapter_features=adapter_features,
        detail_features=detail,
    )


def build_targets(mask, skel, width, sdf, band):
    return {"mask": mask, "skel": skel, "width": width, "sdf": sdf, "band": band}


def train(args):
    setup_logging(args)
    set_seed(args.seed)
    logging.info(f"Random seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    sam = build_sam_vit_l(
        checkpoint=args.checkpoint,
        use_adapter=True,
        adapter_dim_ratio=args.adapter_dim_ratio,
        use_detail=bool(args.use_detail),
        use_curve_heads=bool(args.use_curve_heads),
        use_sdf=bool(args.use_sdf),
        use_ibn=bool(args.use_ibn),
        max_radius=args.max_radius,
        detach_aux=bool(args.detach_aux),
        learn_tau=bool(args.learn_tau),
        use_adaptive_scale=bool(args.use_adaptive_scale),
        adaptive_hidden=args.adaptive_hidden,
        adaptive_span=args.adaptive_span,
        adaptive_mode=args.adaptive_mode,
        adaptive_separate=bool(args.adaptive_separate),
        adaptive_scope=args.adaptive_scope,
    )
    sam.to(device)

    topk_gate = None
    if args.use_topk_selector:
        from segment_anything.modeling.image_encoder import HardTopKGate
        topk_gate = HardTopKGate(
            depth=sam.image_encoder.depth, k=args.topk,
            init_noise=args.topk_init_noise).to(device)
        sam.image_encoder.adaptive_scale = topk_gate
        sam.image_encoder.use_adaptive_scale = True
        sam.image_encoder.adaptive_mode = "scale"
        logging.info(f"Hard top-K selector: K={args.topk} / {2 * sam.image_encoder.depth}, "
                     f"lr={args.topk_lr}")

    total_params, trainable_params = count_parameters(sam)
    logging.info(f"Total parameters: {total_params:,}")

    for name, param in sam.image_encoder.named_parameters():
        if "adapter" not in name and "adaptive_scale" not in name:
            param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    total_params, trainable_params = count_parameters(sam)
    logging.info(f"Trainable parameters: {trainable_params:,} "
                 f"({trainable_params / total_params * 100:.2f}%)")

    train_dataset = CurveSegDataset(args.data_root, "train", augment=args.augment,
                                   sdf_trunc=args.sdf_trunc)
    val_dataset = CurveSegDataset(args.data_root, "val", sdf_trunc=args.sdf_trunc)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, worker_init_fn=_worker_init,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, worker_init_fn=_worker_init)
    logging.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    criterion = CurveSegLoss(
        w_region=args.w_region, w_fused=args.w_fused, w_coarse=args.w_coarse,
        w_skel=args.w_skel, w_width=args.w_width, w_sdf=args.w_sdf,
        w_cldice=args.w_cldice, use_tversky=bool(args.use_tversky),
        bce_weight=args.bce_weight, dice_weight=args.dice_weight,
        tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta,
        w_prec=args.w_prec, prec_alpha=args.prec_alpha, prec_beta=args.prec_beta,
        cldice_iters=args.cldice_iters, sdf_trunc=args.sdf_trunc,
    )
    logging.info(
        "Loss weights: region=%.2f fused=%.2f coarse=%.2f skel=%.2f width=%.2f sdf=%.2f cldice=%.2f",
        args.w_region, args.w_fused, args.w_coarse, args.w_skel, args.w_width,
        args.w_sdf, args.w_cldice)

    adapter_params, decoder_params = [], []
    gate_params = list(topk_gate.parameters()) if topk_gate is not None else []
    gate_ids = {id(p) for p in gate_params}
    for name, param in sam.image_encoder.named_parameters():
        if ("adapter" in name or "adaptive_scale" in name) and param.requires_grad:
            if id(param) not in gate_ids:
                adapter_params.append(param)
    for param in sam.mask_decoder.parameters():
        if param.requires_grad:
            decoder_params.append(param)

    groups = [
        {"params": adapter_params, "lr": args.adapter_lr, "weight_decay": args.weight_decay},
        {"params": decoder_params, "lr": args.decoder_lr, "weight_decay": args.weight_decay},
    ]
    if gate_params:
        groups.append({"params": gate_params, "lr": args.topk_lr, "weight_decay": 0.0})
    optimizer = optim.AdamW(groups)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        scheduler = None

    comp_keys = ["region", "fused", "coarse", "skel", "width", "sdf", "cldice"]
    film_mod = sam.image_encoder.adaptive_film if (
        getattr(sam.image_encoder, "use_adaptive_scale", False)
        and getattr(sam.image_encoder, "adaptive_mode", "") == "film") else None
    scale_mod = sam.image_encoder.adaptive_scale if (
        getattr(sam.image_encoder, "use_adaptive_scale", False)
        and getattr(sam.image_encoder, "adaptive_mode", "") != "film") else None
    best_f1 = 0.0
    swa_queue = collections.deque(maxlen=args.swa_last) if args.swa_last > 0 else None
    for epoch in range(args.epochs):
        sam.train()
        train_loss = 0.0
        comp_sums = {k: 0.0 for k in comp_keys}

        pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for images, mask, skel, width, sdf, band, _ in pbar:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in build_targets(mask, skel, width, sdf, band).items()}

            optimizer.zero_grad()
            out = encode_and_decode(sam, images)
            losses = criterion(out, targets)
            total_loss = losses["loss"]
            if film_mod is not None and args.adaptive_reg > 0:
                total_loss = total_loss + args.adaptive_reg * film_mod.magnitude_loss()
            elif scale_mod is not None and args.adaptive_reg > 0:
                total_loss = total_loss + args.adaptive_reg * scale_mod.penalty()
            total_loss.backward()
            if film_mod is not None:
                film_mod.reset_log()
            if scale_mod is not None:
                scale_mod.reset_log()

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in sam.parameters() if p.requires_grad], args.clip_grad_norm)
            optimizer.step()

            train_loss += losses["loss"].item()
            for k in comp_keys:
                comp_sums[k] += float(losses[k])
            pbar.set_postfix({"loss": f"{losses['loss'].item():.4f}"})

        n = len(train_loader)
        avg_train_loss = train_loss / n
        logging.info(
            f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} | "
            + " ".join(f"{k}={comp_sums[k] / n:.4f}" for k in comp_keys))

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            sam.eval()
            val_loss = 0.0
            tp = fp = fn = 0.0
            with torch.no_grad():
                vbar = tqdm(val_loader, total=len(val_loader),
                            desc=f"Epoch {epoch + 1}/{args.epochs} [Val]")
                for images, mask, skel, width, sdf, band, _ in vbar:
                    images = images.to(device)
                    targets = {k: v.to(device)
                               for k, v in build_targets(mask, skel, width, sdf, band).items()}
                    out = encode_and_decode(sam, images)
                    losses = criterion(out, targets)
                    val_loss += losses["loss"].item()

                    fused = out["masks"]
                    tgt_mask = targets["mask"]
                    if fused.shape[-2:] != tgt_mask.shape[-2:]:
                        fused = F.interpolate(fused, size=tgt_mask.shape[-2:], mode="bilinear",
                                              align_corners=False)
                    pred = torch.sigmoid(fused) > 0.5
                    true = tgt_mask > 0.5
                    tp += (pred & true).sum().item()
                    fp += (pred & ~true).sum().item()
                    fn += (~pred & true).sum().item()

            f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
            avg_val_loss = val_loss / len(val_loader)
            logging.info(f"Epoch {epoch + 1}/{args.epochs} - Val Loss: {avg_val_loss:.4f} "
                         f"| F1 (fused): {f1:.4f}")
            if film_mod is not None:
                int_std, ext_std = film_mod.stats()
                logging.info(f"  adaptive FiLM: std_across_images int={int_std:.4f} "
                             f"ext={ext_std:.4f}  (collapsed if <0.02)")
            if scale_mod is not None:
                int_m, ext_m, std = scale_mod.stats()
                logging.info(f"  adaptive scale: int={int_m:.4f} ext={ext_m:.4f} std={std:.4f}")
            if topk_gate is not None:
                sel_int, sel_ext = topk_gate.selected()
                logging.info(f"  top-K selected int={sel_int} ext={sel_ext}")

            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": sam.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "f1_score": f1,
                }, args.output_weights)
                logging.info(f"New best model saved ({args.output_weights}) F1={best_f1:.4f}")

            if swa_queue is not None:
                # keep the last K validation snapshots for weight averaging
                swa_queue.append({k: v.detach().cpu().clone()
                                  for k, v in sam.state_dict().items()})

        if scheduler is not None:
            scheduler.step()

    # always keep the final weights for weight averaging / diagnostics
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": sam.state_dict(),
        "best_f1": best_f1,
    }, args.last_weights)
    logging.info(f"Last-epoch weights saved to {args.last_weights} (best F1={best_f1:.4f})")

    if swa_queue is not None and len(swa_queue) > 0:
        keys = list(swa_queue[0].keys())
        avg = {k: (sum(s[k].float() for s in swa_queue) / len(swa_queue)).to(torch.float32)
               for k in keys}
        torch.save({"model_state_dict": avg, "swa_n": len(swa_queue),
                    "note": f"average of last {len(swa_queue)} validation snapshots"},
                   args.swa_weights)
        logging.info(f"SWA weights ({len(swa_queue)} snapshots) saved to {args.swa_weights}")

    if getattr(sam.mask_decoder, "learn_tau", False):
        logging.info(f"learned render tau = {float(sam.mask_decoder.tau_value):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM with v17 attribute decoder")
    parser.add_argument("--data_root", type=str, default="./datasets/eee")
    parser.add_argument("--checkpoint", type=str, default="./sam_pth/sam_vit_l_0b3195.pth")
    parser.add_argument("--output_weights", type=str, default="best_model.pth")
    parser.add_argument("--last_weights", type=str, default="last_model.pth")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter_dim_ratio", type=float, default=0.1)
    parser.add_argument("--adapter_lr", type=float, default=3e-4)
    parser.add_argument("--decoder_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--augment", type=int, default=1,
                        help="0 none | 1 scale+dihedral | 2 affine")
    parser.add_argument("--use_detail", type=int, default=1)
    parser.add_argument("--use_curve_heads", type=int, default=1)
    parser.add_argument("--use_sdf", type=int, default=0,
                        help="1 = add a third (boundary SDF) attribute head; "
                             "0 = the two attribute heads (centerline + radius)")
    parser.add_argument("--use_ibn", type=int, default=1)
    parser.add_argument("--detach_aux", type=int, default=1)
    parser.add_argument("--learn_tau", type=int, default=0,
                        help="1 = learn the render temperature (boundary softness)")
    # ---- TPAMI addition: input-adaptive per-adapter scaling ----
    parser.add_argument("--use_adaptive_scale", type=int, default=1,                        help="1 = predict a per-image scale for every adapter")
    parser.add_argument("--adaptive_hidden", type=int, default=32)
    parser.add_argument("--adaptive_span", type=float, default=0.5,
                        help="FiLM bound: gain in (1-span, 1+span) per channel")
    parser.add_argument("--adaptive_mode", type=str, default="film",
                        choices=["film", "scale", "gate"],
                        help="film: per-channel FiLM on the bottleneck (new); "
                             "scale: 1+span*tanh scalar on the output (old); "
                             "gate: sigmoid scalar (old)")
    parser.add_argument("--adaptive_separate", type=int, default=1,
                        help="1 = separate predictors for internal/external adapters")
    parser.add_argument("--adaptive_scope", type=str, default="int",
                        choices=["int", "both"],
                        help="int: modulate only Adapter-I (external stays 1.0)")
    parser.add_argument("--adaptive_reg", type=float, default=0.01,
                        help="L2 budget on the FiLM modulation (film mode); "
                             "budget weight on (s-1)^2 in scale/gate mode")
    parser.add_argument("--adaptive_var", type=float, default=0.1,
                        help="weight of the anti-collapse variance term")
    parser.add_argument("--adaptive_target_std", type=float, default=0.05,
                        help="required std of mean(gamma) across images")
    parser.add_argument("--swa_last", type=int, default=0,
                        help=">0 = average the last K validation snapshots into --swa_weights")
    parser.add_argument("--swa_weights", type=str, default="swa_model.pth")
    parser.add_argument("--use_topk_selector", type=int, default=0,
                        help="1 = hard top-K adapter selection with STE")
    parser.add_argument("--topk", type=int, default=16,
                        help="number of active adapter positions out of 2*depth")
    parser.add_argument("--topk_lr", type=float, default=0.05,
                        help="learning rate for the ranking scores")
    parser.add_argument("--topk_init_noise", type=float, default=0.01,
                        help="noise on the initial ranking scores (breaks ties)")
    parser.add_argument("--max_radius", type=float, default=16.0,
                        help="radius cap (px @512) and soft-distance iteration count")
    parser.add_argument("--sdf_trunc", type=float, default=8.0)

    parser.add_argument("--w_region", type=float, default=1.0)
    parser.add_argument("--w_fused", type=float, default=1.0)
    parser.add_argument("--w_coarse", type=float, default=0.5)
    parser.add_argument("--w_skel", type=float, default=0.3)
    parser.add_argument("--w_width", type=float, default=0.3)
    parser.add_argument("--w_sdf", type=float, default=0.2)
    parser.add_argument("--w_cldice", type=float, default=1.0)
    parser.add_argument("--w_prec", type=float, default=0.0,
                        help=">0 adds a precision-oriented Tversky term (penalises FP)")
    parser.add_argument("--prec_alpha", type=float, default=0.8,
                        help="FP weight of the precision term (alpha > beta)")
    parser.add_argument("--prec_beta", type=float, default=0.2, help="FN weight of the precision term")
    parser.add_argument("--use_tversky", type=int, default=0)
    parser.add_argument("--bce_weight", type=float, default=0.7)
    parser.add_argument("--dice_weight", type=float, default=0.3)
    parser.add_argument("--tversky_alpha", type=float, default=0.3)
    parser.add_argument("--tversky_beta", type=float, default=0.7)
    parser.add_argument("--cldice_iters", type=int, default=10)

    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "none"])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_step_size", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.1)

    train(parser.parse_args())
