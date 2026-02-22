import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

import matplotlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _extract_exp_name(run_dirname: str, env_name: str) -> Optional[str]:
    prefix = "q2_pg_"
    token = f"_{env_name}_"
    if not run_dirname.startswith(prefix) or token not in run_dirname:
        return None
    return run_dirname[len(prefix) : run_dirname.index(token)]


def latest_runs_by_exp(
    data_dir: str,
    env_name: str,
    match: Optional[str],
    experiments: Optional[List[str]],
) -> Dict[str, str]:
    pattern = os.path.join(data_dir, f"q2_pg_*_{env_name}_*")
    run_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    by_exp: Dict[str, List[str]] = {}
    for run_dir in run_dirs:
        exp = _extract_exp_name(os.path.basename(run_dir), env_name)
        if exp is None:
            continue
        if match and match not in exp:
            continue
        if experiments is not None and exp not in experiments:
            continue
        by_exp.setdefault(exp, []).append(run_dir)

    latest: Dict[str, str] = {}
    for exp, dirs in by_exp.items():
        latest[exp] = max(dirs, key=os.path.getmtime)
    return latest


def _get_scalars(
    ea: EventAccumulator, candidate_tags: List[str]
) -> Optional[List]:
    available = set(ea.Tags().get("scalars", []))
    for tag in candidate_tags:
        if tag in available:
            return ea.Scalars(tag)
    return None


def load_xy_from_run(
    run_dir: str, y_tag_candidates: List[str]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        return None
    event_file = max(event_files, key=os.path.getmtime)
    ea = EventAccumulator(event_file)
    ea.Reload()

    x_scalars = _get_scalars(ea, ["Train_EnvstepsSoFar"])
    y_scalars = _get_scalars(ea, y_tag_candidates)
    if y_scalars is None:
        return None

    if x_scalars is None:
        x = np.array([e.step for e in y_scalars], dtype=np.float64)
        y = np.array([e.value for e in y_scalars], dtype=np.float64)
        return x, y

    step_to_env = {e.step: e.value for e in x_scalars}
    pts = [(step_to_env[e.step], e.value) for e in y_scalars if e.step in step_to_env]
    if not pts:
        return None
    pts.sort(key=lambda z: z[0])
    x = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)
    return x, y


def maybe_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def plot_metric(
    runs: Dict[str, str],
    y_tag_candidates: List[str],
    title: str,
    ylabel: str,
    out_png: str,
    smooth_window: int,
) -> None:
    plt.figure(figsize=(9, 5))
    plotted = 0
    for exp in sorted(runs.keys()):
        xy = load_xy_from_run(runs[exp], y_tag_candidates)
        if xy is None:
            print(f"[skip] {exp}: metric not found ({y_tag_candidates})")
            continue
        x, y = xy
        y_plot = maybe_smooth(y, smooth_window)
        plt.plot(x, y_plot, linewidth=2, label=exp)
        plotted += 1

    if plotted == 0:
        print(f"[warn] No runs had metric {y_tag_candidates}.")
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Number of environment steps")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    print(f"[saved] {out_png}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v4")
    parser.add_argument(
        "--match",
        type=str,
        default="baseline",
        help="Only include experiments whose exp_name contains this substring. "
        "Use empty string to include all.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="",
        help="Comma-separated exact exp_name list. If set, overrides --match filter.",
    )
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--out_baseline", type=str, default="hw2_nnb_na_loss.png")
    parser.add_argument("--out_return", type=str, default="hw2_nnb_na_eval_return.png")
    args = parser.parse_args()

    experiments = None
    if args.experiments.strip():
        experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]

    match = args.match.strip() if experiments is None else None
    if match == "":
        match = None

    runs = latest_runs_by_exp(
        data_dir=args.data_dir,
        env_name=args.env_name,
        match=match,
        experiments=experiments,
    )
    if not runs:
        raise FileNotFoundError(
            "No matching runs found. "
            "Check --data_dir, --env_name, --match, or --experiments."
        )

    print("Runs used:")
    for exp in sorted(runs.keys()):
        print(f"  {exp}: {runs[exp]}")

    plot_metric(
        runs=runs,
        y_tag_candidates=["Baseline_Loss", "Baseline Loss"],
        title=f"NN Baseline Loss ({args.env_name})",
        ylabel="Baseline loss",
        out_png=args.out_baseline,
        smooth_window=args.smooth_window,
    )
    plot_metric(
        runs=runs,
        y_tag_candidates=["Eval_AverageReturn"],
        title=f"NN Baseline Eval Return ({args.env_name})",
        ylabel="Average return",
        out_png=args.out_return,
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()
