# CurviSAM

Prompt-free curvilinear-structure segmentation with a frozen SAM, in 2D and 3D.

## 1. Final 2D architecture

| Component | Choice | Evidence |
|---|---|---|
| Backbone | frozen SAM ViT-L | unchanged throughout |
| Dual-level adapter | 24 internal (MLP residual) + 24 external (block boundary), r=0.1, d=102 | removing Adapter-I −8.96, removing Adapter-E −10.09 |
| up_proj init | zero | training starts exactly at the pretrained SAM |
| Attribute decoder | centerline + radius + boundary SDF at 512, explicit tube rendering, zero-init residuals | no SDF 0.7173/0.5106; with SDF 0.7295/0.5398 (ROAD +2.9) |
| Precision term | Tversky, `w_prec=0.5, α=0.8, β=0.2` | without 0.6340–0.6549; with 0.7105–0.7295 (DSCA +6–8) |
| SWA | average of the last 5 validation snapshots | best 0.7105 → SWA 0.7295 (+1.9) |
| Adaptive rank (deployment) | 18-image contribution spectrum → SVD output-subspace rank r≈0.40, **no layer deletion** | 0.7295/0.5398 → 0.7284/0.5382 (−0.11/−0.16, adapter params −28%) |
| Training-time adaptive scaling | not used | collapses (cross-image std 0.012), DSCA −9.0 |
| Layer deletion | not used | −1.0 to −2.4 on the final model, unstable across models |

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
3D/    the same design applied slice-wise, with a 3D mask decoder
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

`3D/` applies the same frozen-encoder + dual-level adapter design slice by slice,
with `segment_anything/modeling/mask_decoder_3d.py` replacing the 2D decoder.

```bash
cd 3D
python train_sam_3d.py          # training (paths configured at the top of the file)
python infer_3d_3ddec.py --cases <case> --output_dir <dir>
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
