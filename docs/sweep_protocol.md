# Sweep protocol (2026-10-27)

## Fixed training protocol

- Data: `./datasets/eee`, all **18** training images.
- Model: SAM ViT-L + adapter, `use_detail=0`, `use_curve_heads=1`.
- Epochs: **60**, cosine schedule, `min_lr=1e-6`.
- Weights: **last epoch** (`--save_last 1`), not best-on-train.
- Batch size 1, seed 42.
- Augmentation: `--augment 1` (scale 0.4–1.6, dihedral, photometric).
- LR: adapter 3e-4, decoder 3e-4, weight decay 0.01.
- Mask loss: Dice+BCE (`bce=0.7`, `dice=0.3`), `w_region=1`, `w_fused=1`.
- Base aux weights: `w_cldice=1.0`, `w_skel=0.3`, `w_coarse=0.5`, `w_width=0.3`.

> Note: `datasets/eee/train` and `datasets/eee/val` contain the same 18 files
> (identical md5). The original `best_model` is therefore selected on training
> data. This sweep uses the last epoch to make the comparison fair.

## Sweep A — loss weights (one-at-a-time, step 0.1)

- `w_cldice` ∈ {0.1, ..., 0.9} + baseline 1.0
- `w_skel` ∈ {0.1, ..., 0.9} (baseline 0.3)
- `w_coarse` ∈ {0.1, ..., 0.9} (baseline 0.5)
- `w_width` fixed at 0.3 (its effective contribution is ~0.16% of the total loss).

## Sweep B — adapter capacity

- `adapter_dim_ratio` ∈ {0.1, 0.2, ..., 0.9}
- Loss weights fixed at baseline.

## Evaluation

- `test_dice.py`, no TTA, `sigmoid(logits) > 0.5`.
- DSCA: GT binarized as `mask > 0` (127/255 -> 1), handled in `test_dice.py`.
- ROAD: same binarization.
- Report mean Dice, std, pooled Dice, mean IoU.

## Files

- `sweep_runner.py` — runs all jobs sequentially, resume-safe.
- `sweep_results.csv` / `sweep_results.json` — raw results.
- `sweep_analyze.py` — tables + `sweep_curves.png` + `sweep_summary.md`.
- `sweep_weights/` — checkpoints (one per job; ~1.4–2.3 GB each).
- `logs/sweep/` — per-job train/eval logs.
