# checkpoints

权重不随仓库分发，请把文件放在这里（或任意路径，用命令行参数指定）。

| 文件 | 用途 | 大小 | 获取方式 |
|---|---|---|---|
| `sam_vit_l_0b3195.pth` | SAM ViT-L 主干，**仅训练**需要 | ~1.2 GB | 官方下载：<https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth> |
| `best_model.pth` | 训练好的 CurviSAM，**推理**需要 | ~1.4 GB | 自行训练（见 README 第 6 节）或向作者索取 |

评估/推理时无需加载 SAM 主干：`test_dice.py` / `test_sam.py` 用
`build_sam_vit_l(checkpoint=None, ...)` 构建结构后直接加载 CurviSAM 权重。
