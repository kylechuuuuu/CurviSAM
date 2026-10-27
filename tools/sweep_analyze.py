#!/usr/bin/env python3
"""Analyze sweep_results.csv produced by sweep_runner.py."""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "sweep_results.csv")
OUT_PNG = os.path.join(ROOT, "sweep_curves.png")
OUT_MD = os.path.join(ROOT, "sweep_summary.md")


def load():
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV}")
    df = pd.read_csv(CSV)
    for c in ["w_cldice", "w_skel", "w_coarse", "w_width", "adapter_ratio",
              "dsca_dice", "road_dice", "dsca_iou", "road_iou"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["status"] == "ok"].copy()
    df["mean_dice"] = (df["dsca_dice"] + df["road_dice"]) / 2.0
    return df


def table(df, phase_col, phase_name, fixed, base_x):
    sub = df[df["phase"] == phase_name].copy()
    if sub.empty:
        return f"\n### {phase_col}\n(no results yet)\n"
    sub = sub.sort_values(phase_col)
    base = df[df["job"] == "baseline"]
    lines = [f"\n### {phase_col}  (fixed: {fixed})", "",
             "| value | DSCA | ROAD | mean | note |", "|---:|---:|---:|---:|---|"]
    if not base.empty:
        b = base.iloc[0]
        lines.append(f"| {base_x:.1f} | {b['dsca_dice']:.4f} | {b['road_dice']:.4f} | {b['mean_dice']:.4f} | baseline* |")
    for _, r in sub.iterrows():
        lines.append(f"| {r[phase_col]:.1f} | {r['dsca_dice']:.4f} | {r['road_dice']:.4f} | {r['mean_dice']:.4f} | |")
    # best excluding baseline for a fair "tuned" recommendation
    best_d = sub.loc[sub["dsca_dice"].idxmax()]
    best_r = sub.loc[sub["road_dice"].idxmax()]
    best_m = sub.loc[sub["mean_dice"].idxmax()]
    lines += ["",
              f"- best DSCA (excluding baseline): `{phase_col}={best_d[phase_col]:.1f}` -> {best_d['dsca_dice']:.4f}",
              f"- best ROAD (excluding baseline): `{phase_col}={best_r[phase_col]:.1f}` -> {best_r['road_dice']:.4f}",
              f"- best mean (excluding baseline): `{phase_col}={best_m[phase_col]:.1f}` -> {best_m['mean_dice']:.4f}",
              f"- * baseline is the same run in all four sweeps: cldice=1.0, skel=0.3, coarse=0.5, ratio=0.1.",
              ""]
    return "\n".join(lines)


def main():
    df = load()
    if df.empty:
        raise SystemExit("no successful rows yet")
    base = df[df["job"] == "baseline"]
    base_txt = ""
    if not base.empty:
        b = base.iloc[0]
        base_txt = (f"baseline: DSCA={b['dsca_dice']:.4f}, ROAD={b['road_dice']:.4f}, "
                    f"mean={b['mean_dice']:.4f}  "
                    f"(cldice={b['w_cldice']}, skel={b['w_skel']}, coarse={b['w_coarse']}, "
                    f"ratio={b['adapter_ratio']})")

    md = ["# CurviSAM v16 hyper-parameter sweep", "",
          "Protocol: 18 train images, 60 epochs, last-epoch weights, no TTA, sigmoid>0.5.",
          "DSCA GT = any non-zero mask pixel (127/255 -> 1).", "",
          base_txt, ""]

    md.append(table(df, "w_cldice", "loss_cldice", "skel=0.3, coarse=0.5, ratio=0.1", 1.0))
    md.append(table(df, "w_skel", "loss_skel", "cldice=1.0, coarse=0.5, ratio=0.1", 0.3))
    md.append(table(df, "w_coarse", "loss_coarse", "cldice=1.0, skel=0.3, ratio=0.1", 0.5))
    md.append(table(df, "adapter_ratio", "adapter", "cldice=1.0, skel=0.3, coarse=0.5", 0.1))

    md += ["", "## Overall best", ""]
    for key, label in [("dsca_dice", "DSCA"), ("road_dice", "ROAD"), ("mean_dice", "mean")]:
        r = df.loc[df[key].idxmax()]
        md.append(f"- best {label}: `{r['job']}` -> DSCA={r['dsca_dice']:.4f}, "
                  f"ROAD={r['road_dice']:.4f}")
    md.append("")
    open(OUT_MD, "w").write("\n".join(md))
    print("\n".join(md))

    # plots
    specs = [
        ("loss_cldice", "w_cldice", "w_cldice"),
        ("loss_skel", "w_skel", "w_skel"),
        ("loss_coarse", "w_coarse", "w_coarse"),
        ("adapter", "adapter_ratio", "adapter_dim_ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (phase, col, label) in zip(axes.ravel(), specs):
        sub = df[df["phase"] == phase].sort_values(col)
        if sub.empty:
            ax.set_title(f"{label} (no data)")
            continue
        ax.plot(sub[col], sub["dsca_dice"], "o-", label="DSCA")
        ax.plot(sub[col], sub["road_dice"], "s-", label="ROAD")
        b = df[df["job"] == "baseline"]
        if not b.empty:
            ax.axhline(b.iloc[0]["dsca_dice"], color="C0", ls=":", lw=1.2, alpha=0.8)
            ax.axhline(b.iloc[0]["road_dice"], color="C1", ls=":", lw=1.2, alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Dice")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print(f"\nsaved {OUT_PNG} and {OUT_MD}")


if __name__ == "__main__":
    main()
