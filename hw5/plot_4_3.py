from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


EVAL_TAG_CANDIDATES = ["eval_return", "Eval_AverageReturn"]


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) == 0:
        return y
    window = min(window, len(y))
    kernel = np.ones(window) / window
    y_pad = np.pad(y, (window - 1, 0), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def load_scalar(run_dir: Path, tag_candidates: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    tag = next((t for t in tag_candidates if t in tags), None)
    if tag is None:
        raise ValueError(f"No eval tag in {run_dir}. available={tags}")
    events = ea.Scalars(tag)
    x = np.array([e.step for e in events], dtype=np.float64)
    y = np.array([e.value for e in events], dtype=np.float64)
    return x, y, tag


def resolve_run(logdir: Path, key_or_path: str) -> Path:
    p = Path(key_or_path)
    if p.exists():
        return p
    cands = sorted(
        [d for d in logdir.iterdir() if d.is_dir() and key_or_path in d.name],
        key=lambda d: d.stat().st_mtime,
    )
    if not cands:
        raise FileNotFoundError(f"No run matched '{key_or_path}' in {logdir}")
    return cands[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, default=Path("data"))
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="format: SIZE=RUN_PATH_OR_KEYWORD (repeat 4 times)",
    )
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--title", type=str, default="CQL Data Ablation")
    parser.add_argument("--out", type=Path, default=Path("hw5_cql_data_ablation.png"))
    args = parser.parse_args()

    # parse run specs
    size_to_run = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"Invalid --run '{spec}'. expected SIZE=...")
        size_str, key = spec.split("=", 1)
        size = int(size_str)
        run_dir = resolve_run(args.logdir, key)
        size_to_run[size] = run_dir

    sizes = sorted(size_to_run.keys())
    curves = {}
    best_scores = []
    last_scores = []

    for s in sizes:
        run_dir = size_to_run[s]
        x, y, tag = load_scalar(run_dir, EVAL_TAG_CANDIDATES)
        ys = moving_average(y, args.smooth)
        curves[s] = (x, y, ys, run_dir.name, tag)
        best_scores.append(float(np.max(y)))
        last_scores.append(float(y[-1]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    # (1) eval curve
    for s in sizes:
        x, y, ys, run_name, _ = curves[s]
        axes[0].plot(x, y, alpha=0.20, linewidth=1.2)
        axes[0].plot(x, ys, linewidth=2.2, label=f"dataset={s}")
    axes[0].set_title(f"{args.title} - Evaluation Curves")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("eval_return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # (2) performance bar (best / last)
    x_idx = np.arange(len(sizes))
    w = 0.35
    axes[1].bar(x_idx - w / 2, best_scores, width=w, label="best eval_return")
    axes[1].bar(x_idx + w / 2, last_scores, width=w, label="last eval_return")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels([str(s) for s in sizes])
    axes[1].set_title(f"{args.title} - Final Performance by Dataset Size")
    axes[1].set_xlabel("dataset size (RND total_steps)")
    axes[1].set_ylabel("eval_return")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180)
    print(f"[saved] {args.out}")
    for s in sizes:
        print(f"[run] {s}: {size_to_run[s]}")


if __name__ == "__main__":
    main()
