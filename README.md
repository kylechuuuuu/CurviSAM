# CurviSAM (v16)

**CurviSAM: Curvilinear Structure Segmentation with Geometry-Aware SAM.**

**Curve-prior SAM adapter for binary segmentation.** 中心线（centerline）与宽度（width）
各走独立解码分支，再通过几何重建 `mask = region + f(centerline, width)` 融合，
把"曲线 = 中心线 ⊕ 宽度"作为结构先验。

> 本仓库为 **纯代码版本**，不含任何模型权重（`.pth`）、数据、日志和实验结果。

## 1. 方法要点

- **主干**：SAM ViT-L + adapter（`adapter_dim_ratio=0.1`），只训练 adapter 与解码器。
- **region 头**：保留 SAM hypernetwork 的 coarse mask 分支。
- **branch_center / branch_width**：两个独立的 `BranchDecoder`
  （ConvTranspose 256→64→32 + conv stack），从 ViT `src` 特征直接上采样到 256，
  而不是在共享 32 通道上挂浅头。
- **geo_fuse**：`[centerline, width] → mask correction`，显式几何重建。
- **fusion_head**：`mask = region + conv([feat, region, center, width, geo])`，
  残差零初始化。
- **辅助头**读取 **detach** 后的特征，避免骨架/宽度损失污染共享 trunk。

训练损失（`losses.py`）：

- region / fused：Dice + BCE（0.7 / 0.3），1024 全分辨率监督；
- 骨架头：Dice + BCE；宽度头：前景带内 masked Smooth-L1；
- clDice：软骨架拓扑损失（512 分辨率）。

## 2. 目录结构

```
CurviSAM/
├── train_sam.py            # 训练入口（v16 配置）
├── test_dice.py            # 单数据集 Dice 评估
├── test_sam.py             # 推理并保存预测图
├── compare_eval.py         # 多数据集对比评估（可选 TTA）
├── ensemble_eval.py        # 多权重集成评估
├── curve_data.py           # 数据集 + 骨架/宽度缓存
├── losses.py               # Dice/BCE/Tversky/clDice/宽度损失
├── segment_anything/       # 修改版 SAM（含 BranchDecoder / 曲线头）
├── tools/                  # 超参 sweep、诊断脚本（可选）
├── docs/sweep_protocol.md  # 阈值 sweep 评估协议
└── checkpoints/            # 放 SAM ViT-L 主干（需自行下载，不入库）
```

## 3. 环境

```bash
pip install -r requirements.txt
```

需要 CUDA（单卡 24 GB 可跑 1024 分辨率、batch size 1）。

## 4. 预训练权重

代码不含权重，需要准备两个文件：

| 用途 | 文件 | 说明 |
|---|---|---|
| SAM ViT-L 主干（仅训练需要） | `checkpoints/sam_vit_l_0b3195.pth` | 官方 SAM ViT-L，约 1.2 GB，[下载](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) |
| 训练好的 CurviSAM 权重（推理需要） | 任意路径，如 `checkpoints/curvisam.pth` | 约 1.4 GB，需自行训练或向作者索取 |

评估时 `build_sam_vit_l(checkpoint=None, ...)` 后直接 `load_state_dict`，
**不需要**再加载 SAM 主干。

## 5. 数据组织

```
datasets/
  <name>/
    images/*.png
    masks/*.png      # 0/255 二值
```

`curve_data.py` 会自动从 mask 生成骨架（`cv2.ximgproc.thinning`）与宽度（`scipy` EDT），
缓存到 `cache/`，首次运行稍慢，之后直接读缓存。

## 6. 训练

```bash
python3 train_sam.py \
  --data_root ./datasets/eee \
  --checkpoint ./checkpoints/sam_vit_l_0b3195.pth \
  --epochs 100 --val_interval 20 --seed 42 \
  --use_tversky 0 --bce_weight 0.7 --dice_weight 0.3 \
  --w_skel 0.3 --w_width 0.3 --w_coarse 0.5 \
  --use_detail 0 --augment 1 --use_curve_heads 1 \
  --output_weights ./checkpoints/best_model.pth
```

60 epoch 版本（更省时，ROAD 略高）把 `--epochs 100 --val_interval 20`
换成 `--epochs 60 --val_interval 10` 即可。

**经验**：必须配轻增广（`--augment 1`），否则独立分支会在少样本源域上过拟合，
迁移到难域反而变差。

## 7. 推理 / 评估

```bash
# 单数据集 Dice（与 baseline 同口径：sigmoid>0.5，无 TTA）
python3 test_dice.py --data_root ./datasets/DSCA \
  --trained_weights ./checkpoints/best_model.pth \
  --output_dir ./results/DSCA

# 保存预测图
python3 test_sam.py --data_root ./datasets/DSCA \
  --trained_weights ./checkpoints/best_model.pth \
  --output_dir ./results/DSCA_pred

# 多数据集对比
python3 compare_eval.py --weights ./checkpoints/best_model.pth \
  --datasets datasets/DSCA datasets/ROAD
```

## 8. 参考结果（test Dice，no-TTA，seed 42）

| 模型 | epoch | DSCA | ROAD |
|---|---:|---:|---:|
| baseline（原始 SACM，未知 seed） | 60 | 0.6730 | 0.4280 |
| baseline 架构 @seed42 | 60 | 0.6463 | 0.3674 |
| v9（旧解码器 + 强增广） | 100 | 0.7033 | 0.5313 |
| v15（本解码器，最省时） | 60 | 0.6479 | **0.5459** |
| **v16（本解码器，均衡最优）** | 100 | **0.7129** | 0.5232 |
| v15 + v16 集成 | — | 0.6965 | 0.5411 |

在 8 个 TMM 数据集上的全量评估（mean Dice，623 张图）为 **0.5854**，
其中 DSCA 0.7129、FIVES 0.8327、XCAD 0.7888。

## 9. 说明

- 本仓库代码含 `BranchDecoder`，**与 v9（`sacm_curve_improved`）的权重不兼容**，请勿混用。
- `segment_anything/` 修改自 [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
  （Apache-2.0），文件头版权声明保留，见 `NOTICE.md`。
- 训练/评估脚本会在当前目录写 `logs/`、`cache/`、`results/`，已在 `.gitignore` 中忽略。
