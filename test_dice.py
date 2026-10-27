import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from segment_anything import build_sam_vit_l
import logging
from datetime import datetime
import argparse
from tqdm import tqdm


def setup_logging(tag):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/dice_{tag}_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return log_file


class SegmentationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_files = []
        supported = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith(supported):
                base = os.path.splitext(filename)[0]
                if any(os.path.exists(os.path.join(self.mask_dir, base + e)) for e in supported):
                    self.image_files.append(filename)
        if not self.image_files:
            raise ValueError(f"No valid image-mask pairs found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        base = os.path.splitext(img_name)[0]
        supported = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        mask_path = None
        for e in supported:
            p = os.path.join(self.mask_dir, base + e)
            if os.path.exists(p):
                mask_path = p
                break
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        original_size = image.size
        model_input = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])(image)
        # IMPORTANT: binarize ground truth as "any non-zero pixel = foreground".
        # DSCA masks use 127 (fine vessels) and 255 (main trunk); a plain
        # ToTensor() gives 127/255=0.498 which a >0.5 threshold would drop.
        m = np.array(mask)
        original_mask = torch.from_numpy((m > 0).astype(np.float32)).unsqueeze(0)
        # keep the naive 255-only encoding too, for side-by-side reporting
        mask_255 = torch.from_numpy((m > 127).astype(np.float32)).unsqueeze(0)
        return model_input, original_mask, mask_255, original_size, img_name


def dice_iou(pred_bin, true_bin):
    pred = pred_bin.astype(np.float64).reshape(-1)
    true = true_bin.astype(np.float64).reshape(-1)
    tp = (pred * true).sum()
    fp = (pred * (1 - true)).sum()
    fn = ((1 - pred) * true).sum()
    denom = 2 * tp + fp + fn
    dice = (2 * tp + 1e-6) / (denom + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    return dice, iou


def evaluate(args):
    log_file = setup_logging(args.tag)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"Config: data_root={args.data_root}, weights={args.trained_weights}, "
                 f"adapter_dim_ratio={args.adapter_dim_ratio}")

    sam = build_sam_vit_l(checkpoint=None, use_adapter=True,
                          adapter_dim_ratio=args.adapter_dim_ratio,
                          use_detail=bool(args.use_detail))
    sam.to(device)
    ckpt = torch.load(args.trained_weights, map_location=device)
    sam.load_state_dict(ckpt['model_state_dict'], strict=False)
    logging.info(f"Loaded weights from {args.trained_weights} (ckpt f1={ckpt.get('f1_score', float('nan')):.4f})")

    ds = SegmentationDataset(args.data_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    logging.info(f"Dataset {args.data_root}: {len(ds)} samples")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sam.eval()
    sparse_embeddings = torch.zeros((1, 0, 256), dtype=torch.float, device=device)
    dense_embeddings = torch.zeros((1, 256, 64, 64), dtype=torch.float, device=device)

    dices, ious = [], []
    dices255, ious255 = [], []
    dices_raw, ious_raw = [], []
    g_tp = g_fp = g_fn = 0.0
    with torch.no_grad():
        for model_input, original_mask, mask_255, original_size, img_name in tqdm(loader, desc="Dice eval"):
            model_input = model_input.to(device)
            original_mask = original_mask.to(device)
            mask_255 = mask_255.to(device)
            encoder_output = sam.image_encoder(model_input, return_features=True)
            if len(encoder_output) == 3:
                image_embeddings, encoder_features, adapter_features = encoder_output
            else:
                image_embeddings, encoder_features = encoder_output
                adapter_features = None
            detail = sam.mask_decoder.encode_detail(model_input)
            decoder_out = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                adapter_features=adapter_features,
                detail_features=detail,
            )
            low_res_masks = decoder_out['masks']
            logits = F.interpolate(low_res_masks, size=original_size[::-1],
                                   mode='bilinear', align_corners=False)
            # proper Dice: sigmoid then 0.5 threshold
            pred_bin = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
            # raw-logit threshold (matches test_sam.py F1 behaviour)
            pred_raw = (logits[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
            true_bin = (original_mask[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
            true255 = (mask_255[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

            d, i = dice_iou(pred_bin, true_bin)
            dr, ir = dice_iou(pred_raw, true_bin)
            d255, i255 = dice_iou(pred_bin, true255)
            dices.append(d); ious.append(i)
            dices_raw.append(dr); ious_raw.append(ir)
            dices255.append(d255); ious255.append(i255)

            p = pred_bin.reshape(-1).astype(np.float64)
            t = true_bin.reshape(-1).astype(np.float64)
            g_tp += (p * t).sum()
            g_fp += (p * (1 - t)).sum()
            g_fn += ((1 - p) * t).sum()

            logging.info(f"{img_name[0]}: dice={d:.4f} iou={i:.4f} | dice255={d255:.4f} | raw_dice={dr:.4f}")

    pooled_dice = (2 * g_tp + 1e-6) / (2 * g_tp + g_fp + g_fn + 1e-6)
    summary = [
        f"Dataset: {args.data_root}",
        f"Weights: {args.trained_weights}",
        f"Samples: {len(ds)}",
        f"--- Dice (GT = any non-zero 127/255, sigmoid>0.5) [CORRECT] ---",
        f"Mean Dice: {np.mean(dices):.4f}",
        f"Std Dice : {np.std(dices):.4f}",
        f"Pooled Dice (micro): {pooled_dice:.4f}",
        f"Mean IoU: {np.mean(ious):.4f}",
        f"--- Dice (GT = 255 only, naive ToTensor encoding) [BUGGY] ---",
        f"Mean Dice: {np.mean(dices255):.4f}",
        f"Std Dice : {np.std(dices255):.4f}",
        f"Mean IoU: {np.mean(ious255):.4f}",
        f"--- Dice on raw logit>0.5 (matches test_sam.py F1) vs any-nonzero GT ---",
        f"Mean Dice: {np.mean(dices_raw):.4f}",
        f"Std Dice : {np.std(dices_raw):.4f}",
        f"Mean IoU: {np.mean(ious_raw):.4f}",
    ]
    for line in summary:
        logging.info(line)
    with open(os.path.join(args.output_dir, 'dice_summary.txt'), 'w') as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Dice of CurviSAM model')
    parser.add_argument('--data_root', type=str, default='./datasets/ROAD')
    parser.add_argument('--trained_weights', type=str, default='best_model.pth')
    parser.add_argument('--output_dir', type=str, default='./results/ROAD_dice')
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.1)
    parser.add_argument('--use_detail', type=int, default=0, help='1 = build with DetailStem')
    parser.add_argument('--tag', type=str, default='eval')
    args = parser.parse_args()
    evaluate(args)
