"""Evaluation for tpami_sacm.

Protocol (identical to the CurviSAM/v17 protocol so numbers stay comparable):
single forward pass, logits bilinearly resized to the original image size,
``sigmoid > 0.5``, no TTA, no post-processing.

Reported: Dice, IoU, clDice (centerline) and HD95 (in native-resolution pixels).
The centerline is obtained with ``cv2.ximgproc.thinning`` when available and
``skimage.morphology.skeletonize`` otherwise.
"""

import argparse
import logging
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from segment_anything import build_sam_vit_l

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class SegmentationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = None
        self.mask_dir = None
        for c in ("images", "image"):
            p = os.path.join(root_dir, c)
            if os.path.isdir(p):
                self.image_dir = p
                break
        for c in ("masks", "mask", "annotation_mask"):
            p = os.path.join(root_dir, c)
            if os.path.isdir(p):
                self.mask_dir = p
                break
        if self.image_dir is None or self.mask_dir is None:
            raise ValueError(f"missing images/masks under {root_dir}")
        self.image_files = []
        supported = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        for filename in sorted(os.listdir(self.image_dir)):
            if filename.lower().endswith(supported):
                base = os.path.splitext(filename)[0]
                if any(os.path.exists(os.path.join(self.mask_dir, base + e))
                       for e in supported):
                    self.image_files.append(filename)
        if not self.image_files:
            raise ValueError(f"No valid image-mask pairs found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base = os.path.splitext(img_name)[0]
        supported = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        mask_path = None
        for e in supported:
            p = os.path.join(self.mask_dir, base + e)
            if os.path.exists(p):
                mask_path = p
                break
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        original_size = image.size
        model_input = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])(image)
        m = np.array(mask)
        original_mask = torch.from_numpy((m > 0).astype(np.float32)).unsqueeze(0)
        mask_255 = torch.from_numpy((m > 127).astype(np.float32)).unsqueeze(0)
        return model_input, original_mask, mask_255, original_size, img_name


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def dice_iou(pred_bin, true_bin):
    pred = pred_bin.astype(np.float64).reshape(-1)
    true = true_bin.astype(np.float64).reshape(-1)
    tp = (pred * true).sum()
    fp = (pred * (1 - true)).sum()
    fn = ((1 - pred) * true).sum()
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    return dice, iou


def _thin(binary_u8):
    if binary_u8.sum() == 0:
        return binary_u8
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        thin = cv2.ximgproc.thinning((binary_u8 * 255).astype(np.uint8))
        return (thin > 127).astype(np.uint8)
    try:
        from skimage.morphology import skeletonize as sk_skel
        return sk_skel(binary_u8 > 0).astype(np.uint8)
    except Exception:  # pragma: no cover
        return binary_u8


def cldice_metric(pred_bin, true_bin):
    sp = _thin(pred_bin.astype(np.uint8))
    sg = _thin(true_bin.astype(np.uint8))
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    tprec = (sp * true_bin).sum() / (sp.sum() + 1e-6)
    tsens = (sg * pred_bin).sum() / (sg.sum() + 1e-6)
    return float(2 * tprec * tsens / (tprec + tsens + 1e-6))


def hd95_metric(pred_bin, true_bin):
    """95th percentile symmetric surface distance, in pixels."""
    if pred_bin.sum() == 0 or true_bin.sum() == 0:
        return float("nan")
    sp = pred_bin.astype(bool) ^ binary_erosion(pred_bin.astype(bool))
    st = true_bin.astype(bool) ^ binary_erosion(true_bin.astype(bool))
    dt_t = distance_transform_edt(~st)
    dt_p = distance_transform_edt(~sp)
    d_pt = dt_t[sp]
    d_tp = dt_p[st]
    return float(np.percentile(np.concatenate([d_pt, d_tp]), 95))


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(args):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/dice_{args.tag}_{timestamp}.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    sam = build_sam_vit_l(checkpoint=None, use_adapter=True,
                          adapter_dim_ratio=args.adapter_dim_ratio,
                          use_detail=bool(args.use_detail),
                          use_curve_heads=bool(args.use_curve_heads),
                          use_sdf=bool(args.use_sdf),
                          use_ibn=bool(args.use_ibn),
                          max_radius=args.max_radius,
                          use_adaptive_scale=bool(args.use_adaptive_scale),
                          adaptive_hidden=args.adaptive_hidden,
                          adaptive_span=args.adaptive_span,
                          adaptive_mode=args.adaptive_mode,
                          adaptive_separate=bool(args.adaptive_separate),
                          adaptive_scope=args.adaptive_scope)
    sam.to(device)
    ckpt = torch.load(args.trained_weights, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing = sam.load_state_dict(state, strict=False)
    logging.info(f"Loaded {args.trained_weights}")
    logging.info(f"missing keys: {len(missing.missing_keys)} | "
                 f"unexpected: {len(missing.unexpected_keys)}")
    if missing.missing_keys:
        logging.info(f"  e.g. missing: {missing.missing_keys[:5]}")

    ds = SegmentationDataset(args.data_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info(f"Dataset {args.data_root}: {len(ds)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    sam.eval()
    sparse_embeddings = torch.zeros((1, 0, 256), dtype=torch.float, device=device)
    dense_embeddings = torch.zeros((1, 256, 64, 64), dtype=torch.float, device=device)

    dices, ious, cldices, hds, dices255 = [], [], [], [], []
    g_tp = g_fp = g_fn = 0.0
    with torch.no_grad():
        for model_input, original_mask, mask_255, original_size, img_name in tqdm(
                loader, desc="Dice eval"):
            model_input = model_input.to(device)
            detail = sam.mask_decoder.encode_detail(model_input)
            encoder_output = sam.image_encoder(model_input, return_features=True)
            if len(encoder_output) == 3:
                image_embeddings, _, adapter_features = encoder_output
            else:
                image_embeddings, _ = encoder_output
                adapter_features = None
            decoder_out = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                adapter_features=adapter_features,
                detail_features=detail,
            )
            logits = F.interpolate(decoder_out["masks"], size=original_size[::-1],
                                   mode="bilinear", align_corners=False)
            if args.match_model_res:
                # evaluate on the 1024 grid the model was trained on
                logits = F.interpolate(decoder_out["masks"], size=(1024, 1024),
                                       mode="bilinear", align_corners=False)
                original_mask = F.interpolate(original_mask, size=(1024, 1024), mode="nearest")
                mask_255 = F.interpolate(mask_255, size=(1024, 1024), mode="nearest")
            pred_bin = (torch.sigmoid(logits)[0, 0] > args.threshold).cpu().numpy().astype(np.uint8)
            true_bin = (original_mask[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
            true255 = (mask_255[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

            d, i = dice_iou(pred_bin, true_bin)
            d255, _ = dice_iou(pred_bin, true255)
            cl = cldice_metric(pred_bin, true_bin)
            hd = hd95_metric(pred_bin, true_bin)
            dices.append(d)
            ious.append(i)
            dices255.append(d255)
            cldices.append(cl)
            hds.append(hd)

            p = pred_bin.reshape(-1).astype(np.float64)
            t = true_bin.reshape(-1).astype(np.float64)
            g_tp += (p * t).sum()
            g_fp += (p * (1 - t)).sum()
            g_fn += ((1 - p) * t).sum()
            logging.info(f"{img_name[0]}: dice={d:.4f} iou={i:.4f} "
                         f"clDice={cl:.4f} hd95={hd:.2f}")

    pooled = (2 * g_tp + 1e-6) / (2 * g_tp + g_fp + g_fn + 1e-6)
    summary = [
        f"Dataset: {args.data_root}",
        f"Weights: {args.trained_weights}",
        f"Samples: {len(ds)}",
        f"Mean Dice: {np.mean(dices):.4f}",
        f"Std Dice : {np.std(dices):.4f}",
        f"Pooled Dice (micro): {pooled:.4f}",
        f"Mean IoU: {np.mean(ious):.4f}",
        f"Mean clDice (centerline): {np.mean(cldices):.4f}",
        f"Mean HD95 (px): {np.nanmean(hds):.4f}",
        f"Mean Dice (GT = 255 only): {np.mean(dices255):.4f}",
    ]
    for line in summary:
        logging.info(line)
    with open(os.path.join(args.output_dir, "dice_summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tpami_sacm")
    parser.add_argument("--data_root", type=str, default="./datasets/DSCA")
    parser.add_argument("--trained_weights", type=str, default="best_model.pth")
    parser.add_argument("--output_dir", type=str, default="./results/DSCA")
    parser.add_argument("--tag", type=str, default="tpami")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--match_model_res", type=int, default=0,
                        help="1 = also evaluate on the 1024 grid the model is trained on")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adapter_dim_ratio", type=float, default=0.1)
    parser.add_argument("--use_detail", type=int, default=1)
    parser.add_argument("--use_curve_heads", type=int, default=1)
    parser.add_argument("--use_sdf", type=int, default=0)
    parser.add_argument("--use_ibn", type=int, default=1)
    parser.add_argument("--max_radius", type=float, default=16.0)
    parser.add_argument("--use_adaptive_scale", type=int, default=1)
    parser.add_argument("--adaptive_hidden", type=int, default=32)
    parser.add_argument("--adaptive_span", type=float, default=0.5)
    parser.add_argument("--adaptive_mode", type=str, default="film")
    parser.add_argument("--adaptive_separate", type=int, default=1)
    parser.add_argument("--adaptive_scope", type=str, default="int")
    evaluate(parser.parse_args())
