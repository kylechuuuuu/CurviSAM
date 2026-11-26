"""Non-training per-adapter rank calibration from the 18 fine-tuning images.

For every adapter l and bottleneck channel j:

    c_{l,j} = E_{18 images, tokens}[ h_{l,j}^2 ] * || u_{l,j} ||^2

i.e. the second moment of the GELU bottleneck times the squared norm of the
corresponding up-projection column, which is exactly how much that channel can
contribute to the residual stream.  Sorting c descending and keeping the
smallest set that covers a fraction tau of the total energy gives a per-adapter
rank k_l (i.e. r_l = k_l / d).  The mask is applied at inference only: no
gradient, no retraining, the checkpoint is untouched.

tau = 1.0 is the identity mask and must reproduce the fixed-r=0.1 baseline.
"""

import glob
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, ".")
from segment_anything import build_sam_vit_l  # noqa: E402
from eval_dice import SegmentationDataset, dice_iou  # noqa: E402

CKPT = "work/orig_base_v16_best.pth"
TRAIN = "datasets/eee_v16/train/images"
OUT = "rank_calib.json"


def build(ckpt=CKPT, use_sdf=False, adapter_dim_ratio=0.1):
    m = build_sam_vit_l(checkpoint=None, use_adapter=True,
                        adapter_dim_ratio=adapter_dim_ratio,
                        use_detail=True, use_curve_heads=True, use_sdf=use_sdf,
                        use_adaptive_scale=False)
    ck = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(ck["model_state_dict"], strict=False)
    return m.cuda().eval()


def adapters(m):
    out = []
    for i, blk in enumerate(m.image_encoder.blocks):
        out.append((i, False, blk.adapter))
    for i, ext in enumerate(m.image_encoder.external_adapters):
        out.append((i, True, ext))
    return out


def calibrate(m, score="contrib"):
    sums, cnt = {}, {}
    hooks = []

    def make(key):
        def h(mod, inp, out):
            v = out.detach().float()
            s = v.pow(2).sum(dim=(0, 1, 2))
            sums[key] = sums.get(key, torch.zeros_like(s)) + s
            cnt[key] = cnt.get(key, 0) + v.numel() // v.shape[-1]
        return h

    for i, ext, ad in adapters(m):
        hooks.append(ad.act.register_forward_hook(make((i, ext))))

    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
    files = sorted(glob.glob(TRAIN + "/*"))
    with torch.no_grad():
        for f in files:
            x = tf(Image.open(f).convert("RGB")).unsqueeze(0).cuda()
            m.image_encoder(x, return_features=True)
    for h in hooks:
        h.remove()

    scores = {}
    for i, ext, ad in adapters(m):
        e = (sums[(i, ext)] / max(cnt[(i, ext)], 1)).float()
        if score == "contrib":
            w = ad.up_proj.weight.detach().float().pow(2).sum(dim=0)
            e = e * w
        scores[(i, ext)] = e.cpu().numpy()
    return scores, len(files)


def choose_masks(scores, tau):
    masks, ks = {}, {}
    for k, e in scores.items():
        tot = float(e.sum())
        if tot <= 0:
            masks[k] = np.ones_like(e, dtype=np.float32)
            ks[k] = len(e)
            continue
        order = np.argsort(-e)
        cum = np.cumsum(e[order]) / tot
        n = int(np.searchsorted(cum, tau) + 1)
        mask = np.zeros_like(e, dtype=np.float32)
        mask[order[:n]] = 1.0
        masks[k] = mask
        ks[k] = n
    return masks, ks


def apply_masks(m, masks):
    hooks = []
    for i, ext, ad in adapters(m):
        mask = torch.tensor(masks[(i, ext)], device="cuda",
                            dtype=torch.float32).view(1, 1, 1, -1)

        def h(mod, inp, out, mask=mask):
            return out * mask

        hooks.append(ad.act.register_forward_hook(h))
    return hooks


@torch.no_grad()
def evaluate(m, root):
    ds = SegmentationDataset(root)
    dices = []
    for idx in range(len(ds)):
        x, gt, _m255, size, _name = ds[idx]
        x = x.unsqueeze(0).cuda()
        detail = m.mask_decoder.encode_detail(x)
        emb, _, ada = m.image_encoder(x, return_features=True)
        se = torch.zeros((1, 0, 256), device="cuda")
        de = torch.zeros((1, 256, 64, 64), device="cuda")
        out = m.mask_decoder(emb, m.prompt_encoder.get_dense_pe(), se, de,
                             multimask_output=False, adapter_features=ada,
                             detail_features=detail)["masks"]
        up = F.interpolate(out, size=size[::-1], mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(up)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        gt_np = (gt[0].numpy() > 0.5).astype(np.uint8)
        d, _ = dice_iou(pred, gt_np)
        dices.append(d)
    return float(np.mean(dices))


def main():
    m = build()
    results = {}
    for score in ["contrib", "act"]:
        scores, n_img = calibrate(m, score)
        taus = [1.0, 0.99, 0.95, 0.90] if score == "contrib" else [1.0, 0.95]
        for tau in taus:
            masks, ks = choose_masks(scores, tau)
            r_mean = float(np.mean([v / len(masks[k]) for k, v in ks.items()]))
            hooks = apply_masks(m, masks)
            d_dsca = evaluate(m, "datasets/DSCA")
            d_road = evaluate(m, "datasets/ROAD")
            for h in hooks:
                h.remove()
            key = f"{score}_tau{tau:.2f}"
            results[key] = {"r_mean": r_mean, "dsca": d_dsca, "road": d_road,
                            "ks": {f"{k[0]}_{'ext' if k[1] else 'int'}": int(v)
                                   for k, v in ks.items()}}
            print(f"{key:18s} mean_r={r_mean:.3f}  DSCA={d_dsca:.4f}  ROAD={d_road:.4f}", flush=True)
            if tau in (1.0, 0.95):
                ints = [v for k, v in ks.items() if not k[1]]
                exts = [v for k, v in ks.items() if k[1]]
                print(f"   int k: {' '.join(str(v) for v in ints)}", flush=True)
                print(f"   ext k: {' '.join(str(v) for v in exts)}", flush=True)

    json.dump(results, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
