# 3D

The 2D design applied slice by slice to 3D volumes: a frozen 2D SAM ViT-L
encodes each slice, the K consecutive slice embeddings are stacked, and a 3D
decoder predicts K consistent masks.

| file | role |
|---|---|
| `train_3d.py` | train a 3D decoder on cached slice embeddings (`--decoder plain|attr`) |
| `mask_decoder_3d_attr.py` | 3D attribute decoder: Conv3d centreline / radius / boundary-SDF heads |
| `curve_data_3d.py` | derive 3D attribute targets (skeleton, radius, SDF) from a mask slab |
| `segment_anything/modeling/mask_decoder_3d.py` | plain Conv3d mask head (baseline) |
| `train_sam_3d.py` | original single-case 3D training script |
| `infer_3d_3ddec.py` | inference with the 3D decoder |
| `infer_3d_v12.py`, `infer_3d_zs.py` | earlier 2.5D / zero-shot variants |
| `sacm3d_architecture.png` | architecture figure |

## Attribute decoder

`MaskDecoder3DAttr` mirrors the 2D decoder with `Conv3d`:

```
embeddings (B,256,K,64,64)
  -> 3D stem (Conv3d 3x3x3, GroupNorm, GELU)
  -> optional trilinear upsample to (K,128,128)
  -> region head (1x1x1)
  -> attribute stem -> centreline / log-radius / SDF heads
  -> soft 3D distance to the centreline -> tube render
  -> geo_fuse (zero-init) -> fusion (zero-init) -> masks = region + residual
```

## Adaptive rank (same as 2D)

The 18-source-image contribution spectrum is computed on the encoder adapters and
the SVD output-subspace projection is applied **before** the embeddings are
cached, with no layer deletion:

```bash
python train_3d.py --decoder attr --rank_budget 0.40 \
  --data_root datasets/vessel12 --ckpt work/orig_base_v16_best.pth
```

## Usage

```bash
python train_3d.py --decoder attr --holdout 10 \
  --data_root datasets/vessel12 --ckpt work/orig_base_v16_best.pth \
  --out work/dec3d_attr.pth

python infer_3d_3ddec.py --cases <case> --output_dir <dir>
```

## Result

VESSEL12_23, z-disjoint split (35 train / 11 test slabs, no shared slices):

| decoder | test slab Dice |
|---|---:|
| plain Conv3d (baseline) | 0.6873 |
| **attribute decoder (Conv3d)** | **0.6923** |
| attribute + SVD rank 0.40 | 0.6885 |

Single case, 11 test slabs: ±0.5 Dice is within noise. The attribute decoder is
slightly better; the adaptive rank is close to neutral in 3D (it is lossless in
2D, where the calibration and target domains are closer).
