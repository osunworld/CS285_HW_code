#!/usr/bin/env python3
"""
Compare Humanoid learning curves between HW2 and HW3 runs.

x-axis: environment steps
y-axis: evaluation return

Usage examples:
  python hw3/plot_humanoid_hw2_vs_hw3.py
  python hw3/plot_humanoid_hw2_vs_hw3.py --smooth 5
  python hw3/plot_humanoid_hw2_vs_hw3.py --hw3_data /path/to/hw3/data --hw2_data /path/to/hw2/data
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


HW2_Y_TAG_CANDIDATES = ("Eval_AverageReturn", "eval_return")
HW2_X_TAG_CANDIDATES = ("Train_EnvstepsSoFar", "train_envsteps", "env_steps")
HW3_Y_TAG_CANDIDATES = ("eval_return", "Eval_AverageReturn")
HW3_X_TAG_CANDIDATES = ("Train_EnvstepsSoFar", "train_envsteps", "env_steps")


def _ellipsize(text: str, max_len: int) -> str:
    if max_len <= 3 or len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size == 0:
        return y
    window = min(window, y.size)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    y_pad = np.pad(y, (window - 1, 0), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def _pick_first_existing(candidates: Sequence[str], existing: Iterable[str]) -> Optional[str]:
    existing_set = set(existing)
    for tag in candidates:
        if tag in existing_set:
            return tag
    return None


def _find_run_dirs(data_dir: Path, include_substring: str) -> List[Path]:
    if not data_dir.exists():
        return []
    runs = []
    for p in data_dir.iterdir():
        if not p.is_dir():
            continue
        if include_substring.lower() in p.name.lower():
            runs.append(p)
    runs.sort(key=lambda p: p.stat().st_mtime)
    return runs


def _find_latest_run(data_dir: Path, include_substring: str) -> Optional[Path]:
    runs = _find_run_dirs(data_dir, include_substring)
    return runs[-1] if runs else None


def _load_xy_from_run(
    run_dir: Path,
    y_tag_candidates: Sequence[str],
    x_tag_candidates: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, str, Optional[str]]:
    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])

    y_tag = _pick_first_existing(y_tag_candidates, scalar_tags)
    if y_tag is None:
        raise ValueError(
            f"No eval-return tag found in {run_dir}. "
            f"Candidates: {list(y_tag_candidates)}, available: {scalar_tags}"
        )

    y_events = ea.Scalars(y_tag)
    y_step_to_val = {e.step: float(e.value) for e in y_events}

    x_tag = _pick_first_existing(x_tag_candidates, scalar_tags)
    if x_tag is None:
        # Fallback: use tensorboard step as x-axis
        steps = np.array([e.step for e in y_events], dtype=np.float64)
        ys = np.array([float(e.value) for e in y_events], dtype=np.float64)
        return steps, ys, y_tag, None

    x_events = ea.Scalars(x_tag)
    x_step_to_val = {e.step: float(e.value) for e in x_events}

    common_steps = [s for s in sorted(y_step_to_val.keys()) if s in x_step_to_val]
    if not common_steps:
        # If alignment by step fails, fallback to y steps.
        steps = np.array([e.step for e in y_events], dtype=np.float64)
        ys = np.array([float(e.value) for e in y_events], dtype=np.float64)
        return steps, ys, y_tag, None

    xs = np.array([x_step_to_val[s] for s in common_steps], dtype=np.float64)
    ys = np.array([y_step_to_val[s] for s in common_steps], dtype=np.float64)
    return xs, ys, y_tag, x_tag


def _clip_to_common_step_range(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    두 곡선을 공통 스텝 범위 [max(min), min(max)] 로 자른다.
    한쪽 로그가 아직 덜 학습된 경우, 긴 쪽 tail을 자동 제거해서 비교를 공정하게 만든다.
    """
    if x1.size == 0 or x2.size == 0:
        return x1, y1, x2, y2, np.nan, np.nan

    lo = max(float(np.min(x1)), float(np.min(x2)))
    hi = min(float(np.max(x1)), float(np.max(x2)))

    m1 = (x1 >= lo) & (x1 <= hi)
    m2 = (x2 >= lo) & (x2 <= hi)
    return x1[m1], y1[m1], x2[m2], y2[m2], lo, hi


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_hw2_data = repo_root / "hw2" / "data"
    default_hw3_data = repo_root / "hw3" / "data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--hw2_data", type=Path, default=default_hw2_data)
    parser.add_argument("--hw3_data", type=Path, default=default_hw3_data)
    parser.add_argument("--hw2_match", type=str, default="Humanoid-v4")
    parser.add_argument("--hw3_match", type=str, default="Humanoid")
    parser.add_argument("--out", type=Path, default=repo_root / "hw3" / "hw3_humanoid_hw2_vs_hw3.png")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window")
    parser.add_argument(
        "--max_label_len",
        type=int,
        default=36,
        help="Legend label max length (truncate with ...)",
    )
    args = parser.parse_args()

    hw2_run = _find_latest_run(args.hw2_data, args.hw2_match)
    hw3_run = _find_latest_run(args.hw3_data, args.hw3_match)

    if hw2_run is None:
        raise FileNotFoundError(f"No HW2 run found in {args.hw2_data} matching '{args.hw2_match}'")
    if hw3_run is None:
        raise FileNotFoundError(f"No HW3 run found in {args.hw3_data} matching '{args.hw3_match}'")

    x2, y2, y2_tag, x2_tag = _load_xy_from_run(hw2_run, HW2_Y_TAG_CANDIDATES, HW2_X_TAG_CANDIDATES)
    x3, y3, y3_tag, x3_tag = _load_xy_from_run(hw3_run, HW3_Y_TAG_CANDIDATES, HW3_X_TAG_CANDIDATES)

    # 공통 step 구간으로 자르기 (덜 학습된 로그 길이에 맞춰 비교)
    x2, y2, x3, y3, lo, hi = _clip_to_common_step_range(x2, y2, x3, y3)
    if x2.size == 0 or x3.size == 0:
        raise ValueError(
            "After clipping to common step range, one of the curves is empty. "
            f"hw2_points={x2.size}, hw3_points={x3.size}, range=({lo}, {hi})"
        )

    y2s = _moving_average(y2, args.smooth)
    y3s = _moving_average(y3, args.smooth)

    hw2_label = "HW2: " + _ellipsize(hw2_run.name, args.max_label_len)
    hw3_label = "HW3 SAC: " + _ellipsize(hw3_run.name, args.max_label_len)

    plt.figure(figsize=(10, 6))
    plt.plot(x2, y2, color="#1f77b4", alpha=0.20, linewidth=1.2)
    plt.plot(x3, y3, color="#d62728", alpha=0.20, linewidth=1.2)
    plt.plot(x2, y2s, color="#1f77b4", linewidth=2.5, label=hw2_label)
    plt.plot(x3, y3s, color="#d62728", linewidth=2.5, label=hw3_label)

    plt.title("Humanoid-v4: HW2 vs HW3 SAC (Eval Return)")
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=160)
    print(f"[saved] {args.out}")
    print(f"[hw2] run={hw2_run} y_tag={y2_tag} x_tag={x2_tag or 'tb_step'}")
    print(f"[hw3] run={hw3_run} y_tag={y3_tag} x_tag={x3_tag or 'tb_step'}")
    print(f"[clip] common_step_range=[{lo:.0f}, {hi:.0f}] hw2_points={x2.size} hw3_points={x3.size}")


if __name__ == "__main__":
    main()
