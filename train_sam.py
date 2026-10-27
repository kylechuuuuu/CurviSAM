import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from segment_anything import build_sam_vit_l
import argparse
import logging
from datetime import datetime
from sklearn.metrics import f1_score
from tqdm import tqdm

from curve_data import CurveSegDataset
from losses import CurveSegLoss


def setup_logging(args):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("Training Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")


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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calculate_f1_score(pred_masks, true_masks, threshold=0.5):
    pred_masks = pred_masks.detach().cpu().numpy()
    true_masks = true_masks.detach().cpu().numpy()
    pred_masks = (pred_masks > threshold).astype(np.uint8)
    true_masks = (true_masks > threshold).astype(np.uint8)
    pred_masks = pred_masks.reshape(-1)
    true_masks = true_masks.reshape(-1)
    return f1_score(true_masks, pred_masks)


def make_targets(mask, skeleton, width, target_size=256):
    """Build targets for the combined loss.

    * 'mask' stays at full resolution (thin structures preserved for region/fused/clDice)
    * 'mask_ds' / 'skeleton' / 'width' are max-pooled to the decoder resolution so
      skeleton/width heads are supervised on a grid aligned with their output.
    """
    factor = mask.shape[-1] // target_size
    if factor < 1:
        factor = 1
    if factor > 1:
        mask_ds = F.max_pool2d(mask, factor, factor)
        skeleton = F.max_pool2d(skeleton, factor, factor)
        width = F.max_pool2d(width, factor, factor)
    else:
        mask_ds = mask
    return {
        "mask": mask,
        "mask_ds": mask_ds,
        "skeleton": skeleton,
        "width": width,
    }


def encode_and_decode(sam, images):
    # NOTE: feed raw [0,1] tensors (original behaviour). SAM-style normalization
    # was tested and hurt in this pipeline.
    x = images
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
    out = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        adapter_features=adapter_features,
        detail_features=detail,
    )
    return out


def train(args):
    setup_logging(args)
    set_seed(args.seed)
    logging.info(f"Random seed: {args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Build SAM model with adapters + curve-aware decoder enabled
    sam = build_sam_vit_l(checkpoint=args.checkpoint, use_adapter=True,
                          adapter_dim_ratio=args.adapter_dim_ratio,
                          use_detail=bool(args.use_detail),
                          use_curve_heads=bool(args.use_curve_heads))
    sam.to(device)

    total_params, trainable_params = count_parameters(sam)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Trainable parameters percentage: {trainable_params/total_params*100:.2f}%")

    # Freeze image encoder parameters except for adapters
    for name, param in sam.image_encoder.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False

    # Freeze prompt encoder parameters
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False

    # Enable mask decoder parameters (region + skeleton + width + fusion)
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    total_params, trainable_params = count_parameters(sam)
    logging.info(f"After freezing - Total parameters: {total_params:,}")
    logging.info(f"After freezing - Trainable parameters: {trainable_params:,}")
    logging.info(f"After freezing - Trainable parameters percentage: {trainable_params/total_params*100:.2f}%")

    train_dataset = CurveSegDataset(args.data_root, 'train', augment=args.augment)
    val_dataset = CurveSegDataset(args.data_root, 'val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, worker_init_fn=_worker_init,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, worker_init_fn=_worker_init)

    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

    criterion = CurveSegLoss(
        w_region=args.w_region,
        w_fused=args.w_fused,
        w_coarse=args.w_coarse,
        w_skel=args.w_skel,
        w_cldice=args.w_cldice,
        w_width=args.w_width,
        use_tversky=args.use_tversky,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        cldice_iters=args.cldice_iters,
    )
    logging.info(
        "Loss weights: region=%.2f fused=%.2f skel=%.2f cldice=%.2f width=%.2f | "
        "tversky=%s (a=%.2f b=%.2f)",
        args.w_region, args.w_fused, args.w_skel, args.w_cldice, args.w_width,
        args.use_tversky, args.tversky_alpha, args.tversky_beta,
    )

    # Create parameter groups for optimizer - adapters and mask decoder
    adapter_params = []
    mask_decoder_params = []

    for name, param in sam.image_encoder.named_parameters():
        if 'adapter' in name and param.requires_grad:
            adapter_params.append(param)

    for param in sam.mask_decoder.parameters():
        if param.requires_grad:
            mask_decoder_params.append(param)

    optimizer = optim.AdamW([
        {'params': adapter_params, 'lr': args.adapter_lr, 'weight_decay': args.weight_decay},
        {'params': mask_decoder_params, 'lr': args.decoder_lr, 'weight_decay': args.weight_decay}
    ])

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.scheduler == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_gamma, patience=args.patience, verbose=True)
    else:
        scheduler = None

    logging.info(f"Adapter parameters: {len(adapter_params)}, learning rate: {args.adapter_lr}")
    logging.info(f"Mask decoder parameters: {len(mask_decoder_params)}, learning rate: {args.decoder_lr}")
    logging.info(f"Scheduler: {args.scheduler}")

    best_f1_score = 0.0
    for epoch in range(args.epochs):
        sam.train()
        train_loss = 0.0
        comp_sums = {k: 0.0 for k in ['region', 'fused', 'coarse', 'skel', 'cldice', 'width']}

        train_pbar = tqdm(train_loader, total=len(train_loader),
                          desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, masks, skeletons, widths, _ in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            skeletons = skeletons.to(device)
            widths = widths.to(device)

            optimizer.zero_grad()

            out = encode_and_decode(sam, images)
            tgt = make_targets(masks, skeletons, widths, args.target_size)
            losses = criterion(out, tgt)
            loss = losses['loss']

            loss.backward()

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in sam.parameters() if p.requires_grad],
                    args.clip_grad_norm)

            optimizer.step()

            train_loss += loss.item()
            for k in comp_sums:
                comp_sums[k] += float(losses[k])
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        n = len(train_loader)
        avg_train_loss = train_loss / n
        logging.info(f'Epoch {epoch+1}/{args.epochs} - Average Train Loss: {avg_train_loss:.4f} '
                     f'| ' + ' '.join(f'{k}={comp_sums[k]/n:.4f}' for k in comp_sums))

        if (epoch + 1) % args.val_interval == 0:
            sam.eval()
            val_loss = 0.0
            all_pred_masks = []
            all_true_masks = []

            val_pbar = tqdm(val_loader, total=len(val_loader),
                            desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            with torch.no_grad():
                for images, masks, skeletons, widths, _ in val_pbar:
                    images = images.to(device)
                    masks = masks.to(device)
                    skeletons = skeletons.to(device)
                    widths = widths.to(device)

                    out = encode_and_decode(sam, images)
                    tgt = make_targets(masks, skeletons, widths, args.target_size)
                    losses = criterion(out, tgt)
                    val_loss += losses['loss'].item()

                    # F1 evaluated on the fused mask at full resolution
                    fused = out['masks']
                    fused_full = F.interpolate(fused, size=(1024, 1024),
                                               mode='bilinear', align_corners=False)
                    all_pred_masks.append(fused_full)
                    all_true_masks.append(masks)

            all_pred_masks = torch.cat(all_pred_masks, dim=0)
            all_true_masks = torch.cat(all_true_masks, dim=0)
            f1 = calculate_f1_score(all_pred_masks, all_true_masks)

            avg_val_loss = val_loss / len(val_loader)
            logging.info(f'Epoch {epoch+1}/{args.epochs} - Average Val Loss: {avg_val_loss:.4f}')
            logging.info(f'Epoch {epoch+1}/{args.epochs} - F1 Score (fused): {f1:.4f}')

            if args.scheduler == 'reduce':
                scheduler.step(avg_val_loss)

            if f1 > best_f1_score:
                best_f1_score = f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': sam.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'f1_score': f1,
                }, args.output_weights)
                logging.info(f'New best model saved with F1 score: {best_f1_score:.4f}')

        if scheduler is not None and args.scheduler != 'reduce':
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Current learning rate: {current_lr:.7f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SAM with curve-aware decoder')
    parser.add_argument('--data_root', type=str, default='./datasets/eee', help='Path to dataset root directory')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam_vit_l_0b3195.pth', help='Path to SAM ViT-L checkpoint')
    parser.add_argument('--output_weights', type=str, default='best_model.pth', help='Where to save best weights')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (fixed for controlled comparison)')
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.1, help='Ratio of adapter dimension to model dimension')
    parser.add_argument('--adapter_lr', type=float, default=3e-4, help='Learning rate for adapter modules')
    parser.add_argument('--decoder_lr', type=float, default=3e-4, help='Learning rate for mask decoder')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--val_interval', type=int, default=10, help='Validation interval (epochs)')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--target_size', type=int, default=256, help='Resolution at which losses are computed')
    parser.add_argument('--augment', type=int, default=0, help='1 = dihedral augmentation on the train split')
    parser.add_argument('--use_detail', type=int, default=1, help='1 = high-frequency DetailStem branch')
    parser.add_argument('--use_curve_heads', type=int, default=1, help='1 = skeleton/width heads + fusion')

    # Loss function weights
    parser.add_argument('--w_region', type=float, default=1.0, help='Weight for region head loss')
    parser.add_argument('--w_fused', type=float, default=1.0, help='Weight for fused mask loss')
    parser.add_argument('--w_coarse', type=float, default=0.5, help='Weight for stage-1 coarse mask loss')
    parser.add_argument('--w_skel', type=float, default=1.0, help='Weight for skeleton head loss')
    parser.add_argument('--w_cldice', type=float, default=1.0, help='Weight for clDice (topology) loss')
    parser.add_argument('--w_width', type=float, default=1.0, help='Weight for width (distance) loss')
    parser.add_argument('--use_tversky', type=int, default=1, help='1 = Tversky, 0 = Dice+BCE for mask losses')
    parser.add_argument('--bce_weight', type=float, default=0.5, help='Weight for BCE loss in Dice+BCE')
    parser.add_argument('--dice_weight', type=float, default=0.5, help='Weight for Dice loss in Dice+BCE')
    parser.add_argument('--tversky_alpha', type=float, default=0.3, help='Tversky FP weight')
    parser.add_argument('--tversky_beta', type=float, default=0.7, help='Tversky FN weight (higher -> recall)')
    parser.add_argument('--cldice_iters', type=int, default=10, help='Soft-skeleton iterations for clDice')

    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'reduce', 'none'], help='LR scheduler type')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--lr_step_size', type=int, default=10, help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for StepLR and ReduceLROnPlateau schedulers')
    parser.add_argument('--patience', type=int, default=3, help='Patience for ReduceLROnPlateau scheduler')

    args = parser.parse_args()
    train(args)
