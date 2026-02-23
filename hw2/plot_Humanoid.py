import argparse
import glob
import os
from typing import List, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

import matplotlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"]:
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue


def _find_run_dirs(data_dir: str, exp_name: str, env_name: str) -> List[str]:
    pattern = os.path.join(data_dir, f"q2_pg_{exp_name}_{env_name}_*")
    run_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    run_dirs.sort(key=os.path.getmtime)
    return run_dirs


def _load_curve(run_dir: str, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event file in {run_dir}")
    event_file = max(event_files, key=os.path.getmtime)

    ea = EventAccumulator(event_file)
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    if metric not in tags:
        raise KeyError(f"{metric} not found in {run_dir}")
    if "Train_EnvstepsSoFar" not in tags:
        raise KeyError(f"Train_EnvstepsSoFar not found in {run_dir}")

    envsteps = {e.step: e.value for e in ea.Scalars("Train_EnvstepsSoFar")}
    metric_points = ea.Scalars(metric)
    pts = [(envsteps[e.step], e.value) for e in metric_points if e.step in envsteps]
    pts.sort(key=lambda z: z[0])

    x = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)
    return x, y


def _maybe_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def _summarize(name: str, x: np.ndarray, y: np.ndarray) -> str:
    lastk = y[-10:] if len(y) >= 10 else y
    return (
        f"{name}: final={y[-1]:.2f}, best={np.max(y):.2f}, "
        f"last10_mean={np.mean(lastk):.2f}, final_envsteps={x[-1]:.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--env_name", type=str, default="Humanoid-v4")
    parser.add_argument("--exp_name", type=str, default="humanoid")
    parser.add_argument(
        "--metric",
        type=str,
        default="Eval_AverageReturn",
        choices=["Eval_AverageReturn", "Train_AverageReturn"],
    )
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument(
        "--use_all_runs",
        action="store_true",
        help="If set, overlay all runs for the same exp_name. Otherwise, only latest run is used.",
    )
    parser.add_argument(
        "--show_raw",
        action="store_true",
        help="Show raw (unsmoothed) curve as faint line.",
    )
    parser.add_argument("--out", type=str, default="hw2_humanoid_learning_curve.png")
    args = parser.parse_args()

    run_dirs = _find_run_dirs(args.data_dir, args.exp_name, args.env_name)
    if not run_dirs:
        raise FileNotFoundError(
            "No matching runs found. "
            "Check --data_dir, --exp_name, and --env_name."
        )

    selected_dirs = run_dirs if args.use_all_runs else [run_dirs[-1]]

    plt.figure(figsize=(10, 6))
    summaries = []
    for run_dir in selected_dirs:
        x, y = _load_curve(run_dir, args.metric)
        y_s = _maybe_smooth(y, args.smooth_window)
        label = os.path.basename(run_dir).replace(f"q2_pg_{args.exp_name}_{args.env_name}_", "")

        if args.show_raw:
            plt.plot(x, y, linewidth=1.0, alpha=0.25)
        plt.plot(x, y_s, linewidth=2.5, label=label)
        summaries.append(_summarize(os.path.basename(run_dir), x, y))

    plt.title(f"{args.env_name} Learning Curve ({args.exp_name})")
    plt.xlabel("Number of environment steps")
    plt.ylabel(args.metric.replace("_", " "))
    plt.grid(alpha=0.25)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()

    print(f"[saved] {args.out}")
    print("Runs used:")
    for d in selected_dirs:
        print(f"  {d}")
    print("Summary:")
    for s in summaries:
        print(f"  {s}")


if __name__ == "__main__":
    main()
