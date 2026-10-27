#!/usr/bin/env python3
"""Sequential hyper-parameter sweep runner for CurviSAM v16.

Two sweeps:
  * loss weights: w_cldice / w_skel / w_coarse, one-at-a-time, 0.1..0.9 (step 0.1)
  * adapter capacity: adapter_dim_ratio 0.1..0.9 (step 0.1)

Protocol (fixed):
  * train on all 18 images of datasets/eee (train == val in this repo)
  * 60 epochs, last-epoch weights (NOT best-on-train)
  * evaluate DSCA and ROAD with test_dice.py
  * DSCA GT is binarized as (mask > 0) inside test_dice.py (127/255 -> 1)

Resume-safe: completed jobs are read from sweep_results.csv and skipped.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

BASE = dict(
    w_cldice=1.0,
    w_skel=0.3,
    w_coarse=0.5,
    w_width=0.3,
    adapter_ratio=0.1,
)

EPOCHS = 60
WEIGHTS_DIR = os.path.join(ROOT, "sweep_weights")
RESULTS_DIR = os.path.join(ROOT, "sweep_results")
LOGS_DIR = os.path.join(ROOT, "logs", "sweep")
CSV_PATH = os.path.join(ROOT, "sweep_results.csv")
JSON_PATH = os.path.join(ROOT, "sweep_results.json")

FIELDS = [
    "job", "phase", "status",
    "w_cldice", "w_skel", "w_coarse", "w_width", "adapter_ratio",
    "dsca_dice", "dsca_std", "dsca_pooled", "dsca_iou",
    "road_dice", "road_std", "road_pooled", "road_iou",
    "train_sec", "eval_sec", "weight_path", "timestamp", "error",
]


def job_name(kind, value=None):
    if kind == "baseline":
        return "baseline"
    return f"{kind}_{value:.1f}"


def build_jobs(phases):
    """Return a de-duplicated, ordered list of job dicts."""
    jobs = []

    def add(name, cfg, phase, is_base=False):
        for j in jobs:
            if all(abs(j["cfg"][k] - cfg[k]) < 1e-9 for k in BASE):
                return
        jobs.append(dict(name=name, cfg=cfg, phase=phase, is_base=is_base))

    if "loss" in phases or "all" in phases:
        add("baseline", dict(BASE), "baseline", True)
        for v in [round(0.1 * i, 1) for i in range(1, 10)]:
            cfg = dict(BASE); cfg["w_cldice"] = v
            add(job_name("cldice", v), cfg, "loss_cldice")
        for v in [round(0.1 * i, 1) for i in range(1, 10)]:
            cfg = dict(BASE); cfg["w_skel"] = v
            add(job_name("skel", v), cfg, "loss_skel")
        for v in [round(0.1 * i, 1) for i in range(1, 10)]:
            cfg = dict(BASE); cfg["w_coarse"] = v
            add(job_name("coarse", v), cfg, "loss_coarse")

    if "adapter" in phases or "all" in phases:
        for v in [round(0.1 * i, 1) for i in range(1, 10)]:
            cfg = dict(BASE); cfg["adapter_ratio"] = v
            add(job_name("adapter", v), cfg, "adapter")

    return jobs


def load_done():
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok" and row.get("dsca_dice") and row.get("road_dice"):
                    done.add(row["job"])
    return done


def append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    # also keep a JSON array for convenient plotting
    rows = []
    if os.path.exists(JSON_PATH):
        try:
            rows = json.load(open(JSON_PATH))
        except Exception:
            rows = []
    rows.append(row)
    with open(JSON_PATH, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def parse_summary(path):
    text = open(path, errors="ignore").read()
    m = re.search(
        r"\[CORRECT\].*?Mean Dice:\s*([0-9.]+).*?Std Dice\s*:\s*([0-9.]+).*?"
        r"Pooled Dice \(micro\):\s*([0-9.]+).*?Mean IoU:\s*([0-9.]+)",
        text, re.S,
    )
    if not m:
        # fallback: first Mean Dice after [CORRECT]
        m2 = re.search(r"\[CORRECT\].*?Mean Dice:\s*([0-9.]+)", text, re.S)
        if not m2:
            raise ValueError(f"cannot parse {path}")
        return dict(dice=float(m2.group(1)), std=None, pooled=None, iou=None)
    return dict(dice=float(m.group(1)), std=float(m.group(2)),
                pooled=float(m.group(3)), iou=float(m.group(4)))


def run(cmd, log_path):
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, time.time() - t0


def train_cmd(cfg, weight_path, seed=42):
    return [
        PY, "train_sweep.py",
        "--data_root", "./datasets/eee",
        "--checkpoint", "./checkpoints/sam_vit_l_0b3195.pth",
        "--output_weights", weight_path,
        "--batch_size", "1", "--seed", str(seed),
        "--adapter_dim_ratio", f"{cfg['adapter_ratio']:.4f}",
        "--adapter_lr", "3e-4", "--decoder_lr", "3e-4",
        "--weight_decay", "0.01",
        "--epochs", str(EPOCHS), "--val_interval", "1000",
        "--save_last", "1", "--clip_grad_norm", "1.0",
        "--target_size", "256", "--augment", "1",
        "--use_detail", "0", "--use_curve_heads", "1",
        "--w_region", "1.0", "--w_fused", "1.0",
        "--w_coarse", f"{cfg['w_coarse']:.4f}",
        "--w_skel", f"{cfg['w_skel']:.4f}",
        "--w_cldice", f"{cfg['w_cldice']:.4f}",
        "--w_width", f"{cfg['w_width']:.4f}",
        "--use_tversky", "0",
        "--bce_weight", "0.7", "--dice_weight", "0.3",
        "--tversky_alpha", "0.3", "--tversky_beta", "0.7",
        "--cldice_iters", "10",
        "--scheduler", "cosine", "--min_lr", "1e-6",
    ]


def eval_cmd(dataset, weight_path, out_dir, tag, adapter_ratio):
    return [
        PY, "test_dice.py",
        "--data_root", f"./datasets/{dataset}",
        "--trained_weights", weight_path,
        "--output_dir", out_dir,
        "--adapter_dim_ratio", f"{adapter_ratio:.4f}",
        "--use_detail", "0",
        "--tag", tag,
    ]


def evaluate_job(job, weight_path):
    cfg = job["cfg"]
    name = job["name"]
    out = {}
    eval_sec = 0.0
    for ds in ["DSCA", "ROAD"]:
        out_dir = os.path.join(RESULTS_DIR, name, ds)
        log_path = os.path.join(LOGS_DIR, f"{name}.{ds}.eval.log")
        rc, sec = run(eval_cmd(ds, weight_path, out_dir, f"{name}_{ds}", cfg["adapter_ratio"]), log_path)
        eval_sec += sec
        if rc != 0:
            raise RuntimeError(f"{ds} eval failed (rc={rc}); see {log_path}")
        summ = os.path.join(out_dir, "dice_summary.txt")
        if not os.path.exists(summ):
            raise RuntimeError(f"missing {summ}")
        m = parse_summary(summ)
        key = ds.lower()
        out[f"{key}_dice"] = m["dice"]
        out[f"{key}_std"] = m["std"]
        out[f"{key}_pooled"] = m["pooled"]
        out[f"{key}_iou"] = m["iou"]
    return out, eval_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["loss", "adapter", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    jobs = build_jobs({args.phase})
    done = load_done()
    todo = [j for j in jobs if j["name"] not in done]

    print(f"[{datetime.now():%F %T}] phase={args.phase} jobs={len(jobs)} done={len(done)} todo={len(todo)}", flush=True)
    for j in jobs:
        mark = "done" if j["name"] in done else "TODO"
        print(f"  {mark:4s} {j['name']:20s} cldice={j['cfg']['w_cldice']:.1f} "
              f"skel={j['cfg']['w_skel']:.1f} coarse={j['cfg']['w_coarse']:.1f} "
              f"ratio={j['cfg']['adapter_ratio']:.1f}", flush=True)
    if args.dry_run:
        return

    for idx, job in enumerate(todo, 1):
        name = job["name"]
        cfg = job["cfg"]
        weight_path = os.path.join(WEIGHTS_DIR, f"{name}.pth")
        print(f"\n[{datetime.now():%F %T}] ({idx}/{len(todo)}) START {name} "
              f"cldice={cfg['w_cldice']} skel={cfg['w_skel']} coarse={cfg['w_coarse']} "
              f"ratio={cfg['adapter_ratio']}", flush=True)

        row = dict(job=name, phase=job["phase"], status="running", **cfg,
                   weight_path=weight_path, timestamp=datetime.now().isoformat(timespec="seconds"))

        try:
            if os.path.exists(weight_path) and os.path.getsize(weight_path) > 0:
                print(f"  reusing existing weights {weight_path}", flush=True)
                train_sec = 0.0
            else:
                tlog = os.path.join(LOGS_DIR, f"{name}.train.log")
                rc, train_sec = run(train_cmd(cfg, weight_path), tlog)
                if rc != 0 or not os.path.exists(weight_path):
                    raise RuntimeError(f"train failed (rc={rc}); see {tlog}")

            eval_out, eval_sec = evaluate_job(job, weight_path)
            row.update(eval_out)
            row.update(status="ok", train_sec=round(train_sec, 1),
                       eval_sec=round(eval_sec, 1), error="")
            print(f"  OK {name}: DSCA={row['dsca_dice']:.4f} ROAD={row['road_dice']:.4f} "
                  f"(train {train_sec/60:.1f} min, eval {eval_sec/60:.1f} min)", flush=True)
        except Exception as e:
            row.update(status="failed", train_sec="", eval_sec="", error=str(e))
            print(f"  FAIL {name}: {e}", flush=True)

        append_row(row)

    print(f"\n[{datetime.now():%F %T}] finished phase={args.phase}. Results: {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
