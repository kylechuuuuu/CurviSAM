# 3D — native 3D

Frozen SAM-Med3D 3D ViT + 3D dual-level adapters + a Conv3d attribute decoder.
Input is a native 3D volume (128^3); no per-slice 2D encoder.

## Structure

| file | role |
|---|---|
| `dla/model.py` | SAM-Med3D ViT wrapper + 3D dual-level adapters (internal / external) |
| `dla/convdec.py` | Conv3d decoder with a fine-scale ConvStem (8^3 -> 128^3) |
| `dla/convdec_attr.py` | 3D attribute decoder: Conv3d centreline / radius / boundary-SDF heads |
| `dla/curve_data_3d.py` | derive 3D attribute targets (skeleton, radius, SDF) from a mask volume |
| `scripts/train_fewshot.py` | few-shot training (`--decoder transformer\|conv3d\|conv3d_attr`) |
| `scripts/rank_calib_3d.py` | deployment-time adaptive rank (SVD output subspace, no layer deletion) |
| `scripts/prepare_data.py` | NIfTI -> 128^3 npz |

## Setup

```bash
git clone --depth 1 https://github.com/uni-medical/SAM-Med3D third_party/SAM-Med3D
pip install -r requirements.txt
```

## Results

TOF-MRA 0.8 mm, k=10 labelled volumes, 40 epochs, no prompt:

| decoder | train Dice | test Dice |
|---|---:|---:|
| `dec` (Conv3d + ConvStem) | 0.7402 | 0.7158 |
| `dla` (Conv3d + adapters) | 0.7420 | 0.7171 |
| **`dla + attribute`** | **0.8021** | **0.7858** |
| `dla + attribute + SVD rank 0.40` | — | 0.7858 |

- The attribute decoder is the 3D gain (**+6.9 Dice**); the adapters contribute +0.1.
- Adaptive rank is lossless at the decision level: pruning to 40% of the adapter
  output rank flips 4.8e-7 of the voxels (adapter parameters -25%).
- Use 0.8 mm data. At 1.5 mm every variant underfits (Dice <= 0.13).

## Usage

```bash
python scripts/prepare_data.py --raw-dir data/TOF-MRA --out-dir data/tof_mra_08mm --spacing 0.8

python scripts/train_fewshot.py \
  --data-dir data/tof_mra_08mm --ckpt ckpt/sam_med3d_turbo.pth \
  --k 10 --n-test 10 --decoder conv3d_attr --conv-stem 1 --variant dla \
  --epochs 40 --lr 1e-3 --adapter-lr 3e-4 --run-name tof08_attr

python scripts/rank_calib_3d.py --run-dir runs/tof08_attr \
  --data-dir data/tof_mra_08mm --budgets 1.0,0.6,0.4
```
