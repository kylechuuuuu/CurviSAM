#!/usr/bin/env python3
"""Summarize the 3-seed confirmation runs."""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(ROOT, "sweep_results.csv")
CONF = os.path.join(ROOT, "confirm_results.csv")
OUT_MD = os.path.join(ROOT, "confirm_summary.md")
OUT_PNG = os.path.join(ROOT, "confirm_curves.png")


def main():
    rows = []
    if os.path.exists(SWEEP):
        s = pd.read_csv(SWEEP)
        b = s[(s["job"] == "baseline") & (s["status"] == "ok")]
        if not b.empty:
            r = b.iloc[0]
            rows.append(dict(config="baseline", seed=42,
                             dsca_dice=r["dsca_dice"], road_dice=r["road_dice"]))
    if os.path.exists(CONF):
        c = pd.read_csv(CONF)
        c = c[c["status"] == "ok"]
        for _, r in c.iterrows():
            rows.append(dict(config=r["config"], seed=int(r["seed"]),
                             dsca_dice=r["dsca_dice"], road_dice=r["road_dice"]))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no confirmation results yet")

    summary = df.groupby("config").agg(
        n=("seed", "count"),
        dsca_mean=("dsca_dice", "mean"), dsca_std=("dsca_dice", "std"),
        road_mean=("road_dice", "mean"), road_std=("road_dice", "std"),
    ).reset_index()
    summary["mean_dice"] = (summary["dsca_mean"] + summary["road_mean"]) / 2
    order = ["baseline", "comboA", "comboB"]
    summary["config"] = pd.Categorical(summary["config"], categories=order, ordered=True)
    summary = summary.sort_values("config")

    md = ["# 3-seed confirmation", "",
          "Protocol: 60 epochs, last-epoch weights, 18 train images, no TTA.", "",
          "| config | n | DSCA (mean±std) | ROAD (mean±std) | mean |",
          "|---|---:|---|---|---:|"]
    for _, r in summary.iterrows():
        md.append(f"| {r['config']} | {int(r['n'])} | "
                  f"{r['dsca_mean']:.4f} ± {r['dsca_std']:.4f} | "
                  f"{r['road_mean']:.4f} ± {r['road_std']:.4f} | "
                  f"{r['mean_dice']:.4f} |")
    md += ["", "comboA: cldice=0.4, skel=0.4, coarse=0.2, adapter=0.2",
           "comboB: cldice=0.4, skel=0.5, coarse=0.1, adapter=0.3", ""]
    open(OUT_MD, "w").write("\n".join(md))
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(summary))
    w = 0.36
    ax.bar([i - w/2 for i in x], summary["dsca_mean"], w,
           yerr=summary["dsca_std"], capsize=4, label="DSCA")
    ax.bar([i + w/2 for i in x], summary["road_mean"], w,
           yerr=summary["road_std"], capsize=4, label="ROAD")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["config"])
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print(f"saved {OUT_MD} and {OUT_PNG}")


if __name__ == "__main__":
    main()
