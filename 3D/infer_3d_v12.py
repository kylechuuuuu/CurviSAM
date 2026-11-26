"""Slice-wise 2D->3D inference on the full VESSEL12 volumes with SACM weights.

Each axial slice (512x512) is windowed and passed through the zero-prompt SAM
forward (same as test_sam.py); sigmoid logits are stacked into a 3D probability
volume, then evaluated against the dense Lungmasks GT (3D Dice / HD95 / CC),
with a per-slice Dice curve and overlay previews saved.
"""
import os
import re
import glob
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

from segment_anything import build_sam_vit_l


def read_mhd(path):
    txt = open(path).read()
    dims = [int(v) for v in re.search(r'DimSize = (.+)', txt).group(1).split()]
    spacing = [float(v) for v in re.search(r'ElementSpacing = (.+)', txt).group(1).split()]
    offset = [float(v) for v in re.search(r'Offset = (.+)', txt).group(1).split()]
    etype = re.search(r'ElementType = (\S+)', txt).group(1)
    dtype = {'MET_SHORT': '<i2', 'MET_UCHAR': 'u1', 'MET_CHAR': 'i1', 'MET_LONG': '<i4'}[etype]
    raw = os.path.join(os.path.dirname(path),
                       re.search(r'ElementDataFile = (\S+)', txt).group(1))
    x, y, z = dims
    vol = np.fromfile(raw, dtype=dtype).reshape(z, y, x)
    affine = np.diag(spacing + [1.0])
    affine[:3, 3] = offset
    return vol, spacing, affine


def window2d(arr):
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def dice(p, g):
    return 2.0 * np.logical_and(p, g).sum() / max(p.sum() + g.sum(), 1)


def hd95(pred, gt, sampling):
    st = ndimage.generate_binary_structure(3, 1)
    ps = pred ^ ndimage.binary_erosion(pred, structure=st)
    gs = gt ^ ndimage.binary_erosion(gt, structure=st)
    if ps.sum() == 0 or gs.sum() == 0:
        return np.nan
    d_gt = ndimage.distance_transform_edt(~gs, sampling=sampling)
    d_pr = ndimage.distance_transform_edt(~ps, sampling=sampling)
    return float(max(np.percentile(d_gt[ps], 95), np.percentile(d_pr[gs], 95)))


@torch.no_grad()
def infer_case(sam, vol, device):
    z, y, x = vol.shape
    probs = np.zeros((z, y, x), dtype=np.float16)
    sparse = torch.zeros((1, 0, 256), dtype=torch.float, device=device)
    dense = torch.zeros((1, 256, 64, 64), dtype=torch.float, device=device)
    for k in tqdm(range(z), desc='slices', leave=False):
        sl = window2d(vol[k])
        t = torch.from_numpy(sl)[None, None].repeat(1, 3, 1, 1).float() / 255.0
        t = F.interpolate(t, size=(1024, 1024), mode='bilinear', align_corners=False).to(device)
        out = sam.image_encoder(t, return_features=True)
        emb, feats, adapter = out if len(out) == 3 else (*out, None)
        logits, _ = sam.mask_decoder(
            image_embeddings=emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            adapter_features=adapter,
        )
        logit = F.interpolate(logits, size=(y, x), mode='bilinear',
                              align_corners=False)[0, 0]
        probs[k] = torch.sigmoid(logit).half().cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trained_weights', default='best_model.pth')
    ap.add_argument('--adapter_dim_ratio', type=float, default=0.1)
    ap.add_argument('--output_dir', default='/media/72TB_disk/personal/zk/vessel12_3d_results')
    args = ap.parse_args()

    device = torch.device('cuda')
    sam = build_sam_vit_l(checkpoint=None, use_adapter=True,
                          adapter_dim_ratio=args.adapter_dim_ratio)
    ckpt = torch.load(args.trained_weights, map_location=device)
    sam.load_state_dict(ckpt['model_state_dict'])
    sam.to(device).eval()
    print(f"model loaded, train-set F1={ckpt['f1_score']:.4f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    scans = sorted(glob.glob('/media/72TB_disk/personal/zk/vessel12_extract/vessel12/Scans/*.mhd'))
    rows = []
    for ci, sp in enumerate(scans):
        case = os.path.basename(sp).replace('.mhd', '')
        vol, spacing, affine = read_mhd(sp)
        gt, _, _ = read_mhd(os.path.join(
            '/media/72TB_disk/personal/zk/vessel12_extract/vessel12/Lungmasks', case + '.mhd'))
        gt = gt > 0
        z, y, x = vol.shape

        probs = infer_case(sam, vol, device)
        prob = probs.astype(np.float32)

        res = {'case': case}
        for t in (0.3, 0.5, 0.7):
            res[f'dice_t{int(t*10):02d}'] = dice(prob > t, gt)
        pred = prob > 0.5
        res['sens'] = np.logical_and(pred, gt).sum() / max(gt.sum(), 1)
        res['prec'] = np.logical_and(pred, gt).sum() / max(pred.sum(), 1)
        res['cc_pred'] = int(ndimage.label(pred, structure=np.ones((3, 3, 3)))[1])
        res['cc_gt'] = int(ndimage.label(gt, structure=np.ones((3, 3, 3)))[1])
        sl = [dice(pred[k], gt[k]) for k in range(z)]
        res['slice_dice_mean'] = float(np.mean(sl))
        res['hd95'] = hd95(pred, gt, spacing)
        nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine),
                 os.path.join(args.output_dir, f'{case}_pred.nii.gz'))

        rows.append(res)
        print(f"[{ci+1}/3] {case}  dice_t05={res['dice_t05']:.4f} "
              f"slice_dice={res['slice_dice_mean']:.4f} cc={res['cc_pred']} vs {res['cc_gt']}",
              flush=True)

        # per-slice dice curve
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(sl, lw=1.2)
        ax.set_xlabel('axial slice z'); ax.set_ylabel('2D Dice')
        ax.set_title(f'{case}: per-slice Dice (mean {res["slice_dice_mean"]:.3f})')
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(args.output_dir, f'{case}_slice_dice_curve.png'),
                    dpi=110, bbox_inches='tight')
        plt.close(fig)

        # overlay preview: 3 representative slices
        picks = [int(z*0.2), int(z*0.5), int(z*0.85)]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for c, k in enumerate(picks):
            img = window2d(vol[k])
            axes[c].imshow(img, cmap='gray')
            axes[c].contour(gt[k], colors='red', linewidths=0.6)
            axes[c].contour(pred[k], colors='lime', linewidths=0.6)
            axes[c].set_title(f'z={k} dice={sl[k]:.3f}')
            axes[c].axis('off')
        fig.suptitle(f'{case}: green=pred red=GT (axial slices)')
        fig.savefig(os.path.join(args.output_dir, f'{case}_preview.png'),
                    dpi=110, bbox_inches='tight')
        plt.close(fig)

    keys = [k for k in rows[0] if k != 'case']
    print('\n=== VESSEL12 全体数据 2D->3D 汇总 (3 例) ===')
    for k in keys:
        vals = np.array([r[k] for r in rows])
        print(f'{k:10s}: mean={np.nanmean(vals):.4f}  std={np.nanstd(vals):.4f}')
    with open(os.path.join(args.output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write('case,' + ','.join(keys) + '\n')
        for r in rows:
            f.write(r['case'] + ',' + ','.join(
                f'{r[k]:.4f}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')


if __name__ == '__main__':
    main()
