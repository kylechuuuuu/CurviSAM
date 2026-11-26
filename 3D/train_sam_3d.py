"""Train the SACM 3D mask decoder on ONE case (VESSEL12_23).

Pipeline:
  1. Build the SAM ViT-L encoder with adapters, load the v3-trained weights
     (best_model.pth = 15-slice model), freeze the WHOLE encoder.
  2. Cache embeddings for N slab samples (K=8 consecutive axial slices of
     case 23, uniformly spaced slab centers) under no_grad.
  3. Train MaskDecoder3D from scratch on the cached embeddings with
     DiceBCE loss at full 512x512 resolution.

Usage:
  CUDA_VISIBLE_DEVICES=7 python train_sam_3d.py
"""
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from segment_anything import build_sam_vit_l
from segment_anything.modeling.mask_decoder_3d import MaskDecoder3D

CASE = 'VESSEL12_23'
K = 8
N_SLABS = 40
EPOCHS = 80
LR = 2e-3
CKPT_DIR = '/media/72TB_disk/personal/zk/SACM-3D'
ROOT = '/media/72TB_disk/personal/zk/vessel12_extract/vessel12'


def read_mhd(path):
    txt = open(path).read()
    x, y, z = [int(v) for v in re.search(r'DimSize = (.+)', txt).group(1).split()]
    dtype = {'MET_SHORT': '<i2', 'MET_UCHAR': 'u1'}[
        re.search(r'ElementType = (\S+)', txt).group(1)]
    raw = os.path.join(os.path.dirname(path),
                       re.search(r'ElementDataFile = (\S+)', txt).group(1))
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


def build_encoder(device):
    sam = build_sam_vit_l(checkpoint=os.path.join(CKPT_DIR, 'sam_vit_l_0b3195.pth'),
                          use_adapter=True, adapter_dim_ratio=0.1)
    v3 = torch.load(os.path.join(CKPT_DIR, 'best_model.pth'), map_location='cpu')
    sam.load_state_dict(v3['model_state_dict'])
    for p in sam.parameters():
        p.requires_grad = False
    sam.to(device).eval()
    print(f"encoder frozen (v3 train F1={v3['f1_score']:.4f})", flush=True)
    return sam


@torch.no_grad()
def embed_slice(sam, sl, device):
    x = torch.from_numpy(sl)[None, None].repeat(1, 3, 1, 1).float() / 255.0
    x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False).to(device)
    out = sam.image_encoder(x, return_features=True)
    emb = out[0] if isinstance(out, tuple) else out
    return emb[0].half().cpu().numpy()          # (256, 64, 64)


def main():
    device = torch.device('cuda')
    sam = build_encoder(device)

    vol = read_mhd(os.path.join(ROOT, 'Scans', CASE + '.mhd'))
    gt = read_mhd(os.path.join(ROOT, 'Lungmasks', CASE + '.mhd')) > 0
    z = vol.shape[0]

    centers = np.linspace(K // 2, z - 1 - K // 2, N_SLABS).round().astype(int)
    print(f'{N_SLABS} slabs of K={K} slices from {CASE} (z={z})', flush=True)

    need = sorted({int(i) for c in centers for i in range(c - K // 2, c + K // 2)})
    print(f'caching embeddings for {len(need)} slices...', flush=True)
    emb_cache = {}
    for n, i in enumerate(need):
        emb_cache[i] = embed_slice(sam, window2d(vol[i]), device)
        if (n + 1) % 100 == 0:
            print(f'  {n+1}/{len(need)}', flush=True)

    slabs = []
    for c in centers:
        idx = [min(max(j, 0), z - 1) for j in range(c - K // 2, c - K // 2 + K)]
        slabs.append((np.array(idx, dtype=np.int64),
                      np.stack([emb_cache[j] for j in idx]).astype(np.float32).transpose(1, 0, 2, 3),
                      gt[idx].astype(np.float32)))
    del emb_cache

    decoder = MaskDecoder3D().to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f'decoder3d trainable params: {n_params/1e3:.0f}K', flush=True)

    opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    bce = nn.BCEWithLogitsLoss()
    best_dice = 0.0

    for ep in range(EPOCHS):
        decoder.train()
        order = np.random.permutation(len(slabs))
        tl = []
        for si in order:
            idx, emb, msk = slabs[si]
            e = torch.from_numpy(emb)[None].to(device)          # (1,256,K,64,64)
            m = torch.from_numpy(msk)[None, None].to(device)    # (1,1,K,512,512)
            logits = decoder(e)
            logits = F.interpolate(logits[0], size=(512, 512), mode='bilinear',
                                   align_corners=False)[None]
            opt.zero_grad()
            loss = 0.7 * bce(logits, m) + 0.3 * dice_loss(torch.sigmoid(logits), m)
            loss.backward()
            opt.step()
            tl.append(loss.item())
        sched.step()

        decoder.eval()
        ds = []
        with torch.no_grad():
            for idx, emb, msk in slabs:
                e = torch.from_numpy(emb)[None].to(device)
                logits = decoder(e)
                logits = F.interpolate(logits[0], size=(512, 512), mode='bilinear',
                                       align_corners=False)
                p = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0]
                ds.append(dice_np(p, msk))
        mean_d = float(np.mean(ds))
        if mean_d > best_dice:
            best_dice = mean_d
            torch.save({'decoder3d_state_dict': decoder.state_dict(),
                        'K': K, 'epoch': ep + 1, 'f1': mean_d},
                       os.path.join(CKPT_DIR, 'best_model_3d.pth'))
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'epoch {ep+1}/{EPOCHS}  slab Dice={mean_d:.4f}  best={best_dice:.4f}',
                  flush=True)

    print(f'done, best slab Dice = {best_dice:.4f}')


if __name__ == '__main__':
    main()
