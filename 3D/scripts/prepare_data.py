"""Cache NIfTI volumes into fixed-size .npz ROIs (see dla/data.py).

    python scripts/prepare_data.py --raw-dir data/TOF-MRA --out-dir data/tof_mra_15mm

Expected layout (the TOF-MRA / IXI style used here):

    <raw-dir>/part*/raw/<case>.nii.gz
    <raw-dir>/part*/gt/<case>_GT.nii.gz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dla.data import prepare_volume  # noqa: E402


def collect_pairs(raw_dir: str):
    pairs = []
    gt_files = glob(os.path.join(raw_dir, "*", "gt", "*.nii.gz"))
    gt_by_case = {}
    for g in gt_files:
        base = os.path.basename(g).replace("_GT.nii.gz", "")
        gt_by_case[base] = g
    for img in sorted(glob(os.path.join(raw_dir, "*", "raw", "*.nii.gz"))):
        case = os.path.basename(img).replace(".nii.gz", "")
        gt = gt_by_case.get(case)
        if gt is None:
            print(f"[skip] no GT for {case}")
            continue
        pairs.append((case, img, gt))
    return pairs


def _worker(job):
    case, img, gt, out_dir, spacing, size = job
    t0 = time.time()
    try:
        vol = prepare_volume(img, gt, spacing=spacing, size=size)
    except Exception as e:  # noqa: BLE001
        return {"case": case, "ok": False, "err": repr(e)}
    out = os.path.join(out_dir, f"{case}.npz")
    np.savez_compressed(out, **vol)
    meta = json.loads(str(vol["meta"][0]))
    return {"case": case, "ok": True, "file": f"{case}.npz",
            "fg_voxels_roi": meta["fg_voxels_roi"],
            "orig_size": meta["orig_size"], "orig_spacing": meta["orig_spacing"],
            "sec": round(time.time() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--spacing", type=float, default=1.5)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    index_path = os.path.join(args.out_dir, "index.json")
    pairs = collect_pairs(args.raw_dir)
    print(f"[data] {len(pairs)} image/label pairs")

    jobs = [(c, i, g, args.out_dir, args.spacing, args.size) for c, i, g in pairs]
    if not args.force and os.path.exists(index_path):
        with open(index_path) as f:
            done = {c["case"] for c in json.load(f)["cases"]}
        jobs = [j for j in jobs if j[0] not in done]
        print(f"[data] resuming, {len(jobs)} to do")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_worker, jobs):
            print(("  ok  " if r["ok"] else "  FAIL") + f" {r['case']} "
                  + (f"{r.get('fg_voxels_roi')} fg, {r.get('sec')}s" if r["ok"] else r["err"]))
            results.append(r)

    cases = [r for r in results if r["ok"]]
    if os.path.exists(index_path) and not args.force:
        with open(index_path) as f:
            old = {c["case"]: c for c in json.load(f)["cases"]}
        old.update({c["case"]: c for c in cases})
        cases = sorted(old.values(), key=lambda c: c["case"])

    with open(index_path, "w") as f:
        json.dump({"raw_dir": os.path.abspath(args.raw_dir), "spacing": args.spacing,
                   "size": args.size, "cases": cases}, f, indent=2)
    print(f"[data] {len(cases)} cases -> {index_path}")


if __name__ == "__main__":
    main()
