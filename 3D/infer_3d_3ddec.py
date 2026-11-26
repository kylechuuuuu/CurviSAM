"""Full-volume sliding-slab inference with the SACM 3D mask decoder.

For each case: encode every axial slice once (frozen v3-adapted encoder),
then decode each slice with MaskDecoder3D using a K=8 window of embeddings
(edges clamped), stack sigmoid probs, threshold, evaluate vs Lungmasks,
save pred nii.gz + per-slice curve + 3-plane/3D check figures.

Usage:
  CUDA_VISIBLE_DEVICES=7 python infer_3d_3ddec.py \
      --cases VESSEL12_21 VESSEL12_22 VESSEL12_23
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
from segment_anything.modeling.mask_decoder_3d import MaskDecoder3D
from train_sam_3d import read_mhd, window2d, dice_np

ROOT = '/media/72TB_disk/personal/zk/vessel12_extract/vessel12'
CKPT_DIR = '/media/72TB_disk/personal/zk/SACM-3D'


def hd95(pred, gt, sampling):
    st = ndimage.generate_binary_structure(3, 1)
    ps = pred ^ ndimage.binary_erosion(pred, structure=st)
    gs = gt ^ ndimage.binary_erosion(gt, structure=st)
    if ps.sum() == 0 or gs.sum() == 0:
        return np.nan
    d_gt = ndimage.distance_transform_edt(~gs, sampling=sampling)
    d_pr = ndimage.distance_transform_edt(~ps, sampling=sampling)
    return float(max(np.percentile(d_gt[ps], 95), np.percentile(d_pr[gs], 95)))


def dice(p, g):
    return 2.0 * np.logical_and(p, g).sum() / max(p.sum() + g.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='+',
                    default=['VESSEL12_21', 'VESSEL12_22', 'VESSEL12_23'])
    ap.add_argument('--output_dir', default='/media/72TB_disk/personal/zk/vessel12_3d_results_3ddec')
    args = ap.parse_args()

    device = torch.device('cuda')
    sam = build_sam_vit_l(checkpoint=os.path.join(CKPT_DIR, 'sam_vit_l_0b3195.pth'),
                          use_adapter=True, adapter_dim_ratio=0.1)
    v3 = torch.load(os.path.join(CKPT_DIR, 'best_model.pth'), map_location='cpu')
    sam.load_state_dict(v3['model_state_dict'])
    for p in sam.parameters():
        p.requires_grad = False
    sam.to(device).eval()

    d3 = MaskDecoder3D().to(device)
    d3.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'best_model_3d.pth'),
                                  map_location=device)['decoder3d_state_dict'])
    K = torch.load(os.path.join(CKPT_DIR, 'best_model_3d.pth'),
                   map_location='cpu')['K']
    d3.eval()
    print(f'loaded 3d decoder (K={K}, slab F1='
          f'{torch.load(os.path.join(CKPT_DIR, "best_model_3d.pth"), map_location="cpu")["f1"]:.4f})',
          flush=True)
    half = K // 2

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for case in args.cases:
        vol, spacing, _ = None, None, None
        txt = open(os.path.join(ROOT, 'Scans', case + '.mhd')).read()
        x, y, z = [int(v) for v in re.search(r'DimSize = (.+)', txt).group(1).split()]
        spacing = [float(v) for v in re.search(r'ElementSpacing = (.+)', txt).group(1).split()]
        off = [float(v) for v in re.search(r'Offset = (.+)', txt).group(1).split()]
        raw = os.path.join(ROOT, 'Scans',
                           re.search(r'ElementDataFile = (\S+)', txt).group(1))
        vol = np.fromfile(raw, dtype='<i2').reshape(z, y, x)
        gt = read_mhd(os.path.join(ROOT, 'Lungmasks', case + '.mhd')) > 0

        # 1) encode all slices once (fp16 cache)
        embs = np.zeros((z, 256, 64, 64), dtype=np.float16)
        for k in tqdm(range(z), desc=f'{case} encode', leave=False):
            sl = window2d(vol[k])
            xx = torch.from_numpy(sl)[None, None].repeat(1, 3, 1, 1).float() / 255.0
            xx = F.interpolate(xx, size=(1024, 1024), mode='bilinear',
                               align_corners=False).to(device)
            with torch.no_grad():
                out = sam.image_encoder(xx, return_features=True)
                emb = out[0] if isinstance(out, tuple) else out
            embs[k] = emb[0].half().cpu().numpy()

        # 2) sliding decode (stride 1, edges clamped)
        probs = np.zeros((z, y, x), dtype=np.float16)
        starts = [min(max(c - half, 0), z - K) for c in range(z)]
        for c in tqdm(range(z), desc=f'{case} decode', leave=False):
            s = starts[c]
            e = torch.from_numpy(embs[s:s+K].astype(np.float32).transpose(1, 0, 2, 3))[None].to(device)
            with torch.no_grad():
                logits = d3(e)                       # (1,1,K,64,64)
            j = c - s
            lg = F.interpolate(logits[0, :, j][None], size=(y, x), mode='bilinear',
                               align_corners=False)
            probs[c] = torch.sigmoid(lg)[0, 0].half().cpu().numpy()

        prob = probs.astype(np.float32)
        pred = prob > 0.5
        lab, n_cc = ndimage.label(pred, structure=np.ones((3, 3, 3)))
        cc_gt = ndimage.label(gt, structure=np.ones((3, 3, 3)))[1]
        sl = [dice(pred[k], gt[k]) for k in range(z)]
        aff = np.diag(spacing + [1.0]); aff[:3, 3] = off
        nib.save(nib.Nifti1Image(pred.astype(np.uint8), aff),
                 os.path.join(args.output_dir, f'{case}_pred.nii.gz'))

        res = {'case': case, 'dice': dice(pred, gt),
               'sens': np.logical_and(pred, gt).sum() / max(gt.sum(), 1),
               'prec': np.logical_and(pred, gt).sum() / max(pred.sum(), 1),
               'cc_pred': int(n_cc), 'cc_gt': int(cc_gt),
               'slice_dice_mean': float(np.mean(sl)),
               'hd95': hd95(pred, gt, spacing)}
        rows.append(res)
        print(f"{case}: dice={res['dice']:.4f} slice_dice={res['slice_dice_mean']:.4f} "
              f"cc={n_cc} (gt {cc_gt}) hd95={res['hd95']:.1f}mm", flush=True)

        # figures: slice curve + 3-plane overlay + 3D surface
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage.measure import marching_cubes

        fig = plt.figure(figsize=(15, 9))
        for c, (o, k) in enumerate([('axial', int(z*0.5)), ('coronal', int(y*0.45)),
                                    ('sagittal', int(x*0.55))]):
            if o == 'axial':   a, m, p = vol[k], gt[k], pred[k]
            elif o == 'coronal': a, m, p = vol[:, k, :], gt[:, k, :], pred[:, k, :]
            else:               a, m, p = vol[:, :, k], gt[:, :, k], pred[:, :, k]
            ax = fig.add_subplot(2, 3, c + 1)
            ax.imshow(window2d(a), cmap='gray')
            ax.contour(m, colors='red', linewidths=0.6)
            ax.contour(p, colors='lime', linewidths=0.6)
            ax.set_title(f'{o} dice={dice(p, m):.3f}', fontsize=10)
            ax.axis('off')
        ax3d = fig.add_subplot(2, 1, 2, projection='3d')
        step = 6
        for msk, col, al in [(gt, 'red', 0.35), (pred, 'lime', 0.35)]:
            ds = msk[::step, ::step, ::step]
            ds = ndimage.binary_closing(ds, iterations=1)
            if ds.sum() > 10:
                verts, faces, _, _ = marching_cubes(ds.astype(np.float32), 0.5)
                sx, sy, sz_ = [s * step for s in spacing]
                mesh = np.stack([verts[:, 2] * sx, verts[:, 1] * sy, verts[:, 0] * sz_], axis=1)
                ax3d.add_collection3d(Poly3DCollection(mesh[faces], alpha=0.35,
                                                       facecolor=col, edgecolor='none'))
        ax3d.set_title(f'{case} 3D: red=GT green=pred (Dice {res["dice"]:.4f})')
        ax3d.set_axis_off()
        ax3d.view_init(15, -70)
        ex = np.array([x, y, z]) * np.array(spacing)
        ax3d.set_box_aspect(tuple(ex))
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'{case}_3d_check.png'),
                    dpi=110, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(sl, lw=1.2); ax.grid(alpha=0.3)
        ax.set_title(f'{case}: per-slice Dice (mean {res["slice_dice_mean"]:.3f})')
        fig.savefig(os.path.join(args.output_dir, f'{case}_slice_dice_curve.png'),
                    dpi=110, bbox_inches='tight')
        plt.close(fig)

    keys = ['dice', 'sens', 'prec', 'cc_pred', 'cc_gt', 'slice_dice_mean', 'hd95']
    print('\n=== 3D decoder 整卷评估 ===')
    for k in keys:
        vals = np.array([r[k] for r in rows])
        print(f'{k:16s}: mean={np.nanmean(vals):.4f}  std={np.nanstd(vals):.4f}')
    with open(os.path.join(args.output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write('case,' + ','.join(keys) + '\n')
        for r in rows:
            f.write(r['case'] + ',' + ','.join(f'{r[k]:.4f}' for k in keys) + '\n')


if __name__ == '__main__':
    main()
