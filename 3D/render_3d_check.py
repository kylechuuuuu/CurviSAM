"""Independent audit + 3D visualization for the VESSEL12 slice-inference results.

1) Re-compute 3D Dice from scratch (pred nii.gz vs Lungmasks GT) for all cases.
2) Render per-case figures: 3-plane overlays (GT red vs pred green) on top,
   marching-cubes 3D surface render of GT vs pred below.
"""
import os
import re
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from scipy import ndimage

ROOT = '/media/72TB_disk/personal/zk/vessel12_extract/vessel12'
PRED_DIR = '/media/72TB_disk/personal/zk/vessel12_3d_results_v3'
OUT = PRED_DIR


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
    return ((np.clip(a, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def dice(p, g):
    return 2.0 * np.logical_and(p, g).sum() / max(p.sum() + g.sum(), 1)


def render3d(ax, mask, color, alpha, spacing, step):
    ds = mask[::step, ::step, ::step]
    if ds.sum() < 10:
        return
    ds = ndimage.binary_closing(ds, iterations=1)
    if ds.sum() == 0:
        return
    verts, faces, _, _ = marching_cubes(ds.astype(np.float32), 0.5)
    # verts are (z, y, x) in voxel units -> physical mm per axis
    sx, sy, sz = spacing[0]*step, spacing[1]*step, spacing[2]*step
    mesh = np.stack([verts[:, 2]*sx, verts[:, 1]*sy, verts[:, 0]*sz], axis=1)
    pc = Poly3DCollection(mesh[faces], alpha=alpha, facecolor=color,
                          edgecolor='none')
    ax.add_collection3d(pc)
    return mesh


def main():
    print('=== 泄漏审计:训练集文件列表 ===')
    print('\n'.join(sorted(os.listdir(os.path.join(
        '/media/72TB_disk/personal/zk/vessel12_2d/train/images')))))

    for pred_path in sorted(glob.glob(os.path.join(OUT, '*_pred.nii.gz'))):
        case = os.path.basename(pred_path).replace('_pred.nii.gz', '')
        pred = np.asarray(nib.load(pred_path).dataobj) > 0
        gt, spacing, _ = None, None, None
        txt = open(os.path.join(ROOT, 'Lungmasks', case + '.mhd')).read()
        x, y, z = [int(v) for v in re.search(r'DimSize = (.+)', txt).group(1).split()]
        spacing = [float(v) for v in re.search(r'ElementSpacing = (.+)', txt).group(1).split()]
        gt = np.fromfile(os.path.join(ROOT, 'Lungmasks',
                         re.search(r'ElementDataFile = (\S+)', txt).group(1)),
                         dtype='u1').reshape(z, y, x) > 0
        vol, _, _ = None, None, None
        t2 = open(os.path.join(ROOT, 'Scans', case + '.mhd')).read()
        raw = os.path.join(ROOT, 'Scans',
                           re.search(r'ElementDataFile = (\S+)', t2).group(1))
        vol = np.fromfile(raw, dtype='<i2').reshape(z, y, x)

        d = dice(pred, gt)
        print(f'\n{case}: 独立重算 3D Dice = {d:.4f} '
              f'(pred {pred.sum()} vox, gt {gt.sum()} vox)')

        # ---- figure: 3-plane overlays + 3D render ----
        fig = plt.figure(figsize=(15, 9))
        picks = {'axial': int(z*0.5), 'coronal': int(y*0.45), 'sagittal': int(x*0.55)}
        for c, (o, k) in enumerate(picks.items()):
            if o == 'axial':
                a, m, p = vol[k], gt[k], pred[k]
            elif o == 'coronal':
                a, m, p = vol[:, k, :], gt[:, k, :], pred[:, k, :]
            else:
                a, m, p = vol[:, :, k], gt[:, :, k], pred[:, :, k]
            ax = fig.add_subplot(2, 3, c+1)
            ax.imshow(window2d(a), cmap='gray')
            ax.contour(m, colors='red', linewidths=0.6)
            ax.contour(p, colors='lime', linewidths=0.6)
            ax.set_title(f'{o} ({"xyzk"[c]}={k}), dice={dice(p, m):.3f}', fontsize=10)
            ax.axis('off')

        ax3d = fig.add_subplot(2, 1, 2, projection='3d')
        render3d(ax3d, gt, 'red', 0.35, spacing, step=6)
        render3d(ax3d, pred, 'lime', 0.35, spacing, step=6)
        ax3d.set_title(f'{case} 3D surface: red=GT  green=pred   (Dice {d:.4f})')
        ax3d.set_axis_off()
        ax3d.view_init(elev=15, azim=-70)
        sx, sy, sz = np.array([x, y, z]) * np.array(spacing)
        ax3d.set_box_aspect((sx, sy, sz))
        fig.tight_layout()
        out_png = os.path.join(OUT, f'{case}_3d_check.png')
        fig.savefig(out_png, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {out_png}')


if __name__ == '__main__':
    main()
