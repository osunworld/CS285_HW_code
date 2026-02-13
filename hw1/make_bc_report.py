#!/usr/bin/env python3
"""
Simple Q2 plotting script for CS285 HW1.

What this script does
1) Read `q2_table.md`
2) Parse columns:
   - expert_data_fraction
   - Final Eval_AverageReturn Mean
   - Final Eval_AverageReturn Std
3) Save plot as `q2_plot.png`

Usage:
  cd /Users/Sun/Documents/cs285_hw/hw1
  ../cs285/bin/python make_bc_report.py \
    --table q2_table.md \
    --out q2_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

# Headless backend so script works in terminal without GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_q2_table(md_path: Path) -> List[Tuple[float, str, float, float]]:
    """
    Parse q2_table markdown rows.

    Returns list of tuples:
      (fraction_numeric, fraction_label, mean_return, std_return)
    """
    rows: List[Tuple[float, str, float, float]] = []

    for raw in md_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        if line.startswith("| ---"):
            continue
        if "expert_data_fraction" in line:
            continue

        parts = [p.strip() for p in line.strip("|").split("|")]
        # Expected:
        # [fraction, num_seeds, mean, std, runs]
        if len(parts) < 4:
            continue

        frac_label = parts[0]             # e.g. 0p25
        mean_str = parts[2]               # e.g. 3436.13
        std_str = parts[3]                # e.g. 0.00

        frac_value = float(frac_label.replace("p", "."))
        mean_value = float(mean_str)
        std_value = float(std_str)

        rows.append((frac_value, frac_label, mean_value, std_value))

    rows.sort(key=lambda x: x[0])
    return rows


def draw_q2_plot(rows: List[Tuple[float, str, float, float]], out_path: Path) -> None:
    if not rows:
        raise ValueError("No rows parsed from q2_table.md")

    xs = [r[0] for r in rows]
    xlabels = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    yerrs = [r[3] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.errorbar(xs, ys, yerr=yerrs, marker="o", linewidth=2, capsize=4)

    # Show fraction labels above points for readability.
    for x, y, label in zip(xs, ys, xlabels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)

    plt.xticks(xs, xlabels)
    plt.xlabel("expert_data_fraction")
    plt.ylabel("Eval_AverageReturn (mean)")
    plt.title("BC Performance vs Expert Data Fraction")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make q2 plot from q2_table.md")
    parser.add_argument("--table", default="q2_table.md", help="Path to q2 markdown table")
    parser.add_argument("--out", default="q2_plot.png", help="Output plot path")
    args = parser.parse_args()

    table_path = Path(args.table)
    out_path = Path(args.out)

    rows = parse_q2_table(table_path)
    draw_q2_plot(rows, out_path)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
