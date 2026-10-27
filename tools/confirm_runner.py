#!/usr/bin/env python3
"""3-seed confirmation for the best configurations found by sweep_runner.py.

Runs (60 epochs, last-epoch weights, same protocol as the sweep):
  baseline  x seeds 43,44      (seed 42 already in sweep_results.csv)
  comboA    x seeds 42,43,44   cldice=.4 skel=.4 coarse=.2 ratio=.2
  comboB    x seeds 42,43,44   cldice=.4 skel=.5 coarse=.1 ratio=.3
"""

import csv
import json
import os
import time
from datetime import datetime

from sweep_runner import (
    BASE, EPOCHS, RESULTS_DIR, ROOT, WEIGHTS_DIR,
    run, train_cmd, eval_cmd, parse_summary,
)

CSV_PATH = os.path.join(ROOT, "confirm_results.csv")
JSON_PATH = os.path.join(ROOT, "confirm_results.json")
LOGS_DIR = os.path.join(ROOT, "logs", "confirm")
CONFIRM_WEIGHTS = os.path.join(ROOT, "confirm_weights")

FIELDS = [
    "job", "config", "seed", "status",
    "w_cldice", "w_skel", "w_coarse", "w_width", "adapter_ratio",
    "dsca_dice", "dsca_std", "dsca_pooled", "dsca_iou",
    "road_dice", "road_std", "road_pooled", "road_iou",
    "train_sec", "eval_sec", "weight_path", "timestamp", "error",
]

COMBOS = {
    "baseline": dict(BASE),
    "comboA": dict(BASE, w_cldice=0.4, w_skel=0.4, w_coarse=0.2, adapter_ratio=0.2),
    "comboB": dict(BASE, w_cldice=0.4, w_skel=0.5, w_coarse=0.1, adapter_ratio=0.3),
}


def load_done():
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok" and r.get("dsca_dice") and r.get("road_dice"):
                    done.add(r["job"])
    return done


def append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    rows = []
    if os.path.exists(JSON_PATH):
        try:
            rows = json.load(open(JSON_PATH))
        except Exception:
            rows = []
    rows.append(row)
    json.dump(rows, open(JSON_PATH, "w"), indent=2, ensure_ascii=False)


def evaluate(job_name, cfg, weight_path):
    out = {}
    eval_sec = 0.0
    for ds in ["DSCA", "ROAD"]:
        out_dir = os.path.join(RESULTS_DIR, job_name, ds)
        log_path = os.path.join(LOGS_DIR, f"{job_name}.{ds}.eval.log")
        rc, sec = run(eval_cmd(ds, weight_path, out_dir, f"{job_name}_{ds}",
                               cfg["adapter_ratio"]), log_path)
        eval_sec += sec
        if rc != 0:
            raise RuntimeError(f"{ds} eval failed rc={rc}; {log_path}")
        m = parse_summary(os.path.join(out_dir, "dice_summary.txt"))
        k = ds.lower()
        out[f"{k}_dice"] = m["dice"]
        out[f"{k}_std"] = m["std"]
        out[f"{k}_pooled"] = m["pooled"]
        out[f"{k}_iou"] = m["iou"]
    return out, eval_sec


def main():
    os.makedirs(CONFIRM_WEIGHTS, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    jobs = []
    for seed in (43, 44):
        jobs.append(("baseline", 42 if False else seed))
    for combo in ("comboA", "comboB"):
        for seed in (42, 43, 44):
            jobs.append((combo, seed))

    done = load_done()
    todo = [(c, s) for c, s in jobs if f"{c}_s{s}" not in done]
    print(f"[{datetime.now():%F %T}] confirm jobs={len(jobs)} done={len(done)} todo={len(todo)}", flush=True)
    for c, s in jobs:
        print(f"  {'done' if f'{c}_s{s}' in done else 'TODO'} {c}_s{s}", flush=True)

    for i, (combo, seed) in enumerate(todo, 1):
        cfg = COMBOS[combo]
        name = f"{combo}_s{seed}"
        weight_path = os.path.join(CONFIRM_WEIGHTS, f"{name}.pth")
        print(f"\n[{datetime.now():%F %T}] ({i}/{len(todo)}) START {name} "
              f"cldice={cfg['w_cldice']} skel={cfg['w_skel']} coarse={cfg['w_coarse']} "
              f"ratio={cfg['adapter_ratio']}", flush=True)

        row = dict(job=name, config=combo, seed=seed, status="running", **cfg,
                   weight_path=weight_path,
                   timestamp=datetime.now().isoformat(timespec="seconds"))
        try:
            if os.path.exists(weight_path) and os.path.getsize(weight_path) > 0:
                train_sec = 0.0
                print("  reuse existing weights", flush=True)
            else:
                tlog = os.path.join(LOGS_DIR, f"{name}.train.log")
                rc, train_sec = run(train_cmd(cfg, weight_path, seed=seed), tlog)
                if rc != 0 or not os.path.exists(weight_path):
                    raise RuntimeError(f"train failed rc={rc}; {tlog}")
            out, eval_sec = evaluate(name, cfg, weight_path)
            row.update(out, status="ok", train_sec=round(train_sec, 1),
                       eval_sec=round(eval_sec, 1), error="")
            print(f"  OK {name}: DSCA={row['dsca_dice']:.4f} ROAD={row['road_dice']:.4f}", flush=True)
        except Exception as e:
            row.update(status="failed", error=str(e))
            print(f"  FAIL {name}: {e}", flush=True)
        append_row(row)

    print(f"\n[{datetime.now():%F %T}] finished. {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
