# NOTICE

## segment_anything/

`segment_anything/` 修改自 Meta 的
[Segment Anything](https://github.com/facebookresearch/segment-anything)：

```
Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the Apache License, Version 2.0.
```

本仓库对其中的 `build_sam.py`、`modeling/mask_decoder.py` 等文件做了修改
（adapter、BranchDecoder、centerline/width 分支、几何融合），原始版权与
Apache-2.0 声明保留在各文件头部。

## SAM ViT-L 预训练权重

SAM ViT-L 主干权重（`sam_vit_l_0b3195.pth`）由 Meta 以 Apache-2.0 发布，
不包含在本仓库中，需自行下载。

## 其余代码

`train_sam.py`、`test_dice.py`、`test_sam.py`、`compare_eval.py`、
`ensemble_eval.py`、`curve_data.py`、`losses.py` 等为本项目代码。
