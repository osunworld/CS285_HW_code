# hw5/plot_finetune_online_start.py
from __future__ import annotations
import argparse
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def moving_avg(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(y) == 0:
        return y
    w = min(w, len(y))
    k = np.ones(w) / w
    ypad = np.pad(y, (w - 1, 0), mode="edge")
    return np.convolve(ypad, k, mode="valid")


def find_latest_run(data_dir: Path, run_match: str) -> Path:
    runs = [p for p in data_dir.iterdir() if p.is_dir() and run_match in p.name]
    if not runs:
        raise FileNotFoundError(f"No run found in {data_dir} matching '{run_match}'")
    runs.sort(key=lambda p: p.stat().st_mtime)
    return runs[-1]


def load_eval_return(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "eval_return" not in tags:
        raise ValueError(f"'eval_return' not found. available tags: {tags}")
    ev = ea.Scalars("eval_return")
    x = np.array([e.step for e in ev], dtype=np.float64)
    y = np.array([e.value for e in ev], dtype=np.float64)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="e.g. hw5/experiments/finetuning/pointmass_hard_cql_finetune.yaml")
    parser.add_argument("--data_dir", type=Path, default=Path("hw5/data"))
    parser.add_argument("--run_dir", type=Path, default=None,
                        help="If given, use this run dir directly")
    parser.add_argument("--run_match", type=str, default="hw5_finetune_",
                        help="Used only when --run_dir is not given")
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("hw5_finetune_eval_with_online_start.png"))
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    offline_steps = int(cfg["offline_steps"])

    run_dir = args.run_dir if args.run_dir is not None else find_latest_run(args.data_dir, args.run_match)

    x, y = load_eval_return(run_dir)
    ys = moving_avg(y, args.smooth)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, alpha=0.25, linewidth=1.2, label="eval_return (raw)")
    plt.plot(x, ys, linewidth=2.2, label=f"eval_return (smooth={args.smooth})")
    plt.axvline(offline_steps, color="red", linestyle="--", linewidth=2,
                label=f"online starts @ {offline_steps}")
    plt.title("Online Fine-Tuning: Eval Return")
    plt.xlabel("training step")
    plt.ylabel("eval_return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=170)
    print(f"[saved] {args.out}")
    print(f"[run] {run_dir}")
    print(f"[offline_steps] {offline_steps}")


if __name__ == "__main__":
    main()
