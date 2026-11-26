#!/usr/bin/env bash
# conv3d decoder grid on TOF-MRA 0.8mm: dla / dec / dla-no-stem, 3 seeds.
# Usage: bash run_conv3d.sh <gpu> <seed>
set -u
cd "$(dirname "$0")"
GPU=${1:-0}; S=${2:-0}
P=${PY:-python}; CKPT=${CKPT:-ckpt/sam_med3d_turbo.pth}
DATA=data/tof_mra_08mm

one () {  # one <tag> <variant> <stem>
  echo "=== $1 (seed $S) $(date +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES=$GPU $P scripts/train_fewshot.py \
      --data-dir "$DATA" --ckpt "$CKPT" --k 10 --n-test 10 --decoder conv3d \
      --conv-stem "$3" --variant "$2" --epochs 40 --lr 1e-3 --adapter-lr 3e-4 \
      --split-seed "$S" --seed "$S" --gpu 0 --save-ckpt 1 \
      --run-name "$1" 2>&1 | tail -4
}

one "tof08conv_dla_s$S"    dla 1
one "tof08conv_dec_s$S"    dec 1
one "tof08conv_nostem_s$S" dla 0
echo "seed $S done"
