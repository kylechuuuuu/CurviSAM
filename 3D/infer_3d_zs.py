"""Zero-shot 2D-slice inference of eee-trained SACM weights on TOF-MRA volumes.

For each test case: window + normalize each axial slice, run the frozen-prompt
SAM forward (same as test_sam.py), stack sigmoid logits into a 3D probability
volume, threshold, conservatively clean, save nii.gz, and report 3D metrics.
"""
import os
import glob
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from segment_anything import build_sam_vit_l

SPACING = (0.264, 0.264, 0.8)  # (x, y, z) mm, from the data header


def window_slice(arr):
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def load_model(ckpt_path, adapter_dim_ratio, device):
    sam = build_sam_vit_l(checkpoint=None, use_adapter=True,
                          adapter_dim_ratio=adapter_dim_ratio)
    ckpt = torch.load(ckpt_path, map_location=device)
    sam.load_state_dict(ckpt['model_state_dict'])
    sam.to(device).eval()
    return sam, float(ckpt.get('f1_score', float('nan')))


@torch.no_grad()
def infer_case(sam, raw_path, device):
    img_nii = nib.load(raw_path)
    vol = np.asarray(img_nii.dataobj, dtype=np.float32)  # (X, Y, Z)
    sx, sy, sz = vol.shape
    probs = np.zeros((sx, sy, sz), dtype=np.float16)

    sparse = torch.zeros((1, 0, 256), dtype=torch.float, device=device)
    dense = torch.zeros((1, 256, 64, 64), dtype=torch.float, device=device)

    for z in range(sz):
        sl = window_slice(vol[:, :, z])
        x = torch.from_numpy(sl)[None, None].repeat(1, 3, 1, 1).float() / 255.0
        x = x.to(device)
        out = sam.image_encoder(x, return_features=True)
        if len(out) == 3:
            emb, feats, adapter = out
        else:
            emb, feats = out
            adapter = None
        logits, _ = sam.mask_decoder(
            image_embeddings=emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            adapter_features=adapter,
        )
        logit = torch.nn.functional.interpolate(
            logits, size=(sx, sy), mode='bilinear', align_corners=False
        )[0, 0]
        probs[:, :, z] = torch.sigmoid(logit).half().cpu().numpy()
    return img_nii, probs


def dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 2.0 * inter / max(denom, 1)


def hd95(pred, gt, sampling):
    """95th percentile symmetric surface distance in mm."""
    struct = ndimage.generate_binary_structure(3, 1)
    ps = pred ^ ndimage.binary_erosion(pred, structure=struct)
    gs = gt ^ ndimage.binary_erosion(gt, structure=struct)
    if ps.sum() == 0 or gs.sum() == 0:
        return np.nan
    d_gt = ndimage.distance_transform_edt(~gs, sampling=sampling)
    d_pr = ndimage.distance_transform_edt(~ps, sampling=sampling)
    return float(max(np.percentile(d_pr[ps], 95), np.percentile(d_gt[gs], 95)))


def cldice(pred, gt, iters=5):
    """clDice on an isotropically-resampled grid."""
    from skimage.morphology import skeletonize
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    sp = skeletonize(pred)
    sg = skeletonize(gt)
    tprec = np.logical_and(sp, ndimage.binary_dilation(gt, iterations=iters)).sum() / max(sp.sum(), 1)
    tsens = np.logical_and(sg, ndimage.binary_dilation(pred, iterations=iters)).sum() / max(sg.sum(), 1)
    return 2 * tprec * tsens / max(tprec + tsens, 1e-6)


def resample_iso(mask, spacing, iso=0.5):
    zoom = tuple(s / i for s, i in zip(spacing, (iso,) * 3))
    return ndimage.zoom(mask.astype(np.uint8), zoom, order=0) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_glob', default='/media/72TB_disk/personal/zk/TOF-MRA/part4/raw/*.nii.gz')
    ap.add_argument('--trained_weights', default='best_model.pth')
    ap.add_argument('--output_dir', default='mra_zs_results')
    ap.add_argument('--adapter_dim_ratio', type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device('cuda')
    sam, train_f1 = load_model(args.trained_weights, args.adapter_dim_ratio, device)
    print(f'model loaded (train F1 {train_f1:.4f})')

    os_dir = args.output_dir
    import os
    os.makedirs(os_dir, exist_ok=True)
    rows = []
    cases = sorted(glob.glob(args.raw_glob))
    for ci, raw_path in enumerate(cases):
        name = os.path.basename(raw_path).replace('.nii.gz', '')
        gt_path = raw_path.replace('/raw/', '/gt/').replace('MRA.nii.gz', 'MRA_GT.nii.gz')
        gt_nii = nib.load(gt_path)
        gt = np.asarray(gt_nii.dataobj) > 0

        img_nii, probs = infer_case(sam, raw_path, device)
        prob16 = probs.astype(np.float32)

        # threshold sweep + cleaning (conservative: keep CC > 8 voxels)
        res = {'case': name}
        for t in (0.3, 0.5, 0.7):
            res[f'dice_t{int(t*10):02d}'] = dice(prob16 > t, gt)
        pred = prob16 > 0.5
        lab, n_pred = ndimage.label(pred, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(lab.ravel())
        keep = np.zeros(n_pred + 1, dtype=bool)
        keep[1:] = sizes[1:] > 8
        cleaned = keep[lab]
        lab2, n_clean = ndimage.label(cleaned, structure=np.ones((3, 3, 3)))

        res['dice_cc8'] = dice(cleaned, gt)
        res['sens'] = np.logical_and(pred, gt).sum() / max(gt.sum(), 1)
        res['prec'] = np.logical_and(pred, gt).sum() / max(pred.sum(), 1)
        res['cc_pred_raw'] = int(n_pred)
        res['cc_pred_clean'] = int(n_clean)
        res['cc_gt'] = int(ndimage.label(gt, structure=np.ones((3, 3, 3)))[1])
        # per-slice 2D dice mean
        sl = [dice(pred[:, :, z], gt[:, :, z]) for z in range(pred.shape[2])]
        res['slice_dice_mean'] = float(np.mean(sl))
        # clDice & HD95
        pred_iso, gt_iso = resample_iso(cleaned, SPACING), resample_iso(gt, SPACING)
        res['cldice'] = cldice(pred_iso, gt_iso)
        res['hd95'] = hd95(pred, gt, sampling=SPACING)

        out_nii = nib.Nifti1Image(cleaned.astype(np.uint8), gt_nii.affine)
        nib.save(out_nii, os.path.join(os_dir, f'{name}_pred.nii.gz'))
        rows.append(res)
        print(f"[{ci+1}/{len(cases)}] {name}  dice_t05={res['dice_t05']:.4f} "
              f"slice_dice={res['slice_dice_mean']:.4f} cc_pred={n_pred} cc_gt={res['cc_gt']}", flush=True)

        # preview overlays on the first case
        if ci == 0:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            for c, z in enumerate((20, 46, 70)):
                img = window_slice(np.asarray(img_nii.dataobj, dtype=np.float32)[:, :, z])
                axes[0, c].imshow(img, cmap='gray'); axes[0, c].set_title(f'slice z={z}')
                axes[0, c].contour(gt[:, :, z], colors='red', linewidths=0.5)
                axes[1, c].imshow(img, cmap='gray')
                axes[1, c].contour(pred[:, :, z], colors='lime', linewidths=0.5)
                axes[1, c].contour(gt[:, :, z], colors='red', linewidths=0.5)
                for ax in axes.flat:
                    ax.axis('off')
            fig.suptitle('top: GT (red) | bottom: pred (green) vs GT (red)')
            fig.savefig(os.path.join(os_dir, 'preview_overlays.png'), dpi=110, bbox_inches='tight')
            plt.close(fig)

    # summary
    keys = [k for k in rows[0] if k != 'case']
    print('\n=== 零样本切片推理汇总 (9 例) ===')
    for k in keys:
        vals = np.array([r[k] for r in rows])
        print(f'{k:18s}: mean={np.nanmean(vals):.4f}  std={np.nanstd(vals):.4f}')
    with open(os.path.join(os_dir, 'metrics_summary.txt'), 'w') as f:
        f.write('case,' + ','.join(keys) + '\n')
        for r in rows:
            f.write(r['case'] + ',' + ','.join(f'{r[k]:.4f}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')


if __name__ == '__main__':
    main()
