# CurviSAM

Prompt-free curvilinear-structure segmentation with a frozen SAM, in 2D and 3D.

## 1. Final 2D architecture

| Component | Choice |
|---|---|
| Backbone | frozen SAM ViT-L |
| Dual-level adapter | 24 internal (MLP residual) + 24 external (block boundary), r=0.1, d=102 |
| up_proj init | zero |
| Attribute decoder | centerline + radius + boundary SDF at 512, explicit tube rendering, zero-init residuals |
| Precision term | Tversky, `w_prec=0.5, α=0.8, β=0.2` |
| SWA | average of the last 5 validation snapshots |
| Adaptive rank (deployment) | 18-image contribution spectrum → SVD output-subspace rank r≈0.40, **no layer deletion** |
| Training-time adaptive scaling | not used |
| Layer deletion | not used |

## 2. Final 2D results (1 seed, 18 training images)

| Model | DSCA | ROAD | Adapter params |
|---|---:|---:|---:|
| v16 baseline | 0.7213 | 0.5056 | 10.03 M |
| **CurviSAM (final)** | **0.7295** | **0.5398** | 10.03 M |
| + SVD rank 0.40 | 0.7284 | 0.5382 | 7.23 M (−28%) |

DSCA = Base-Unseen (coronary DSA), ROAD = Novel-Unseen (retinal arteries).
Dice at threshold 0.5, no test-time adaptation. Total trainable parameters: 17.28 M (5.31% of the backbone).

## 3. Repository

```
2D/    frozen ViT-L + dual-level adapters + attribute decoder + rank calibration
3D/    native 3D: SAM-Med3D ViT + 3D dual-level adapters + Conv3d attribute decoder
```

## 4. Quick start (2D)

```bash
cd 2D

# fine-tune on 18 images (Tversky + SDF + SWA)
python train_sam.py \
  --data_root ../datasets/eee --checkpoint ../sam_vit_l_0b3195.pth \
  --adapter_dim_ratio 0.1 --use_sdf 1 \
  --use_adaptive_scale 0 --use_topk_selector 0 \
  --w_prec 0.5 --prec_alpha 0.8 --prec_beta 0.2 \
  --epochs 100 --val_interval 10 --swa_last 5 \
  --output_weights work/best.pth --swa_weights work/swa.pth

# evaluate
python eval_dice.py --data_root ../datasets/DSCA \
  --trained_weights work/swa.pth --output_dir results/DSCA --use_sdf 1
```

### Adaptive rank at deployment

`rank_calib.py` computes the per-adapter contribution spectrum from the 18
training images; `rank_v2.py` applies it. SVD rank reduction is lossless and is
the recommended setting; `mask` is kept for comparison only.

```bash
python rank_v2.py --ckpt work/swa.pth --use_sdf 1 --adapter_dim_ratio 0.1 \
  --tasks "svd:importance:0.40:DSCA,svd:importance:0.40:ROAD"
```

## 5. 3D

`3D/` is native 3D: the frozen SAM-Med3D 3D ViT with 3D dual-level adapters and a
Conv3d attribute decoder. Clone SAM-Med3D into `3D/third_party/` first.

```bash
cd 3D
python scripts/prepare_data.py --raw-dir data/TOF-MRA --out-dir data/tof_mra_08mm --spacing 0.8
python scripts/train_fewshot.py --data-dir data/tof_mra_08mm \
  --ckpt ckpt/sam_med3d_turbo.pth --k 10 --n-test 10 \
  --decoder conv3d_attr --conv-stem 1 --variant dla \
  --epochs 40 --lr 1e-3 --adapter-lr 3e-4 --run-name tof08_attr
python scripts/rank_calib_3d.py --run-dir runs/tof08_attr --budgets 1.0,0.6,0.4
```

## 6. Data layout

```
datasets/
  eee/      train/{images,masks}  val/{images,masks}    # 18 annotated images
  DSCA/     images/  masks/                             # Base-Unseen test split
  ROAD/     images/  masks/                             # Novel-Unseen test split
```

## Reference

Built on [Segment Anything](https://github.com/facebookresearch/segment-anything)
and [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D).
