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

# Choose any available readable style.
for _style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"]:
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue


def _extract_exp_name(run_dirname: str, env_name: str) -> Optional[str]:
    prefix = "q2_pg_"
    token = f"_{env_name}_"
    if not run_dirname.startswith(prefix) or token not in run_dirname:
        return None
    return run_dirname[len(prefix) : run_dirname.index(token)]


def _latest_runs_by_exp(data_dir: str, env_name: str) -> Dict[str, str]:
    pattern = os.path.join(data_dir, f"q2_pg_*_{env_name}_*")
    run_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    grouped: Dict[str, List[str]] = {}
    for run_dir in run_dirs:
        exp = _extract_exp_name(os.path.basename(run_dir), env_name)
        if exp is None:
            continue
        grouped.setdefault(exp, []).append(run_dir)

    latest: Dict[str, str] = {}
    for exp, dirs in grouped.items():
        latest[exp] = max(dirs, key=os.path.getmtime)
    return latest


def _load_curve(run_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event file in {run_dir}")
    event_file = max(event_files, key=os.path.getmtime)

    ea = EventAccumulator(event_file)
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    if "Eval_AverageReturn" not in tags:
        raise KeyError(f"Eval_AverageReturn not found in {run_dir}")
    if "Train_EnvstepsSoFar" not in tags:
        raise KeyError(f"Train_EnvstepsSoFar not found in {run_dir}")

    envsteps = {e.step: e.value for e in ea.Scalars("Train_EnvstepsSoFar")}
    eval_ret = ea.Scalars("Eval_AverageReturn")
    pts = [(envsteps[e.step], e.value) for e in eval_ret if e.step in envsteps]
    pts.sort(key=lambda z: z[0])

    x = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)
    return x, y


def _maybe_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def _parse_csv(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def _resolve_run_dirs(
    latest_by_exp: Dict[str, str], exp_names: List[str]
) -> Tuple[List[str], List[str]]:
    run_dirs: List[str] = []
    missing: List[str] = []
    for exp in exp_names:
        if exp in latest_by_exp:
            run_dirs.append(latest_by_exp[exp])
        else:
            missing.append(exp)
    return run_dirs, missing


def _aggregate_curves(
    curves: List[Tuple[np.ndarray, np.ndarray]], n_grid: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(curves) == 1:
        x, y = curves[0]
        return x, y, np.zeros_like(y)

    # Interpolate on common overlapping x-range to compute mean/std across seeds.
    left = max(float(np.min(x)) for x, _ in curves)
    right = min(float(np.max(x)) for x, _ in curves)
    if right <= left:
        # Fallback: use first curve only if ranges do not overlap.
        x, y = curves[0]
        return x, y, np.zeros_like(y)

    x_grid = np.linspace(left, right, n_grid)
    ys = np.stack([np.interp(x_grid, x, y) for x, y in curves], axis=0)
    mean = ys.mean(axis=0)
    std = ys.std(axis=0)
    return x_grid, mean, std


def _summarize(name: str, x: np.ndarray, y: np.ndarray) -> str:
    lastk = y[-10:] if len(y) >= 10 else y
    return (
        f"{name}: final={y[-1]:.2f}, best={np.max(y):.2f}, "
        f"last10_mean={np.mean(lastk):.2f}, final_envsteps={x[-1]:.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--env_name", type=str, default="InvertedPendulum-v4")

    parser.add_argument(
        "--baseline_exps",
        type=str,
        default=(
            "pendulum_default_s1,pendulum_default_s2,pendulum_default_s3,"
            "pendulum_default_s4,pendulum_default_s5"
        ),
        help="Comma-separated exp_name list for baseline/default condition.",
    )
    parser.add_argument(
        "--tuned_exps",
        type=str,
        default="pendulum_tune_s1",
        help="Comma-separated exp_name list for tuned condition.",
    )

    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument(
        "--show_individual",
        action="store_true",
        help="Show each seed/run as faint line behind mean curve.",
    )
    parser.add_argument(
        "--show_std_band",
        action="store_true",
        help="Show mean ± std shaded band (off by default).",
    )
    parser.add_argument("--n_grid", type=int, default=300)
    parser.add_argument("--out", type=str, default="hw2_pendulum_hyper_compare.png")
    args = parser.parse_args()

    baseline_exps = _parse_csv(args.baseline_exps)
    tuned_exps = _parse_csv(args.tuned_exps)

    latest_by_exp = _latest_runs_by_exp(args.data_dir, args.env_name)

    baseline_dirs, baseline_missing = _resolve_run_dirs(latest_by_exp, baseline_exps)
    tuned_dirs, tuned_missing = _resolve_run_dirs(latest_by_exp, tuned_exps)

    if baseline_missing:
        print(f"[warn] missing baseline exps: {baseline_missing}")
    if tuned_missing:
        print(f"[warn] missing tuned exps: {tuned_missing}")

    if not baseline_dirs or not tuned_dirs:
        raise FileNotFoundError(
            "Need at least one run for both baseline and tuned groups."
        )

    baseline_curves = [_load_curve(d) for d in baseline_dirs]
    tuned_curves = [_load_curve(d) for d in tuned_dirs]

    # Smooth first, then aggregate.
    baseline_curves = [(x, _maybe_smooth(y, args.smooth_window)) for x, y in baseline_curves]
    tuned_curves = [(x, _maybe_smooth(y, args.smooth_window)) for x, y in tuned_curves]

    bx, by, bstd = _aggregate_curves(baseline_curves, args.n_grid)
    tx, ty, tstd = _aggregate_curves(tuned_curves, args.n_grid)

    plt.figure(figsize=(10, 6))

    if args.show_individual:
        for x, y in baseline_curves:
            plt.plot(x, y, color="tab:blue", alpha=0.2, linewidth=1.0)
        for x, y in tuned_curves:
            plt.plot(x, y, color="tab:orange", alpha=0.2, linewidth=1.0)

    plt.plot(bx, by, color="tab:blue", linewidth=2.8, label=f"Default (n={len(baseline_curves)})")
    if args.show_std_band and len(baseline_curves) > 1:
        plt.fill_between(bx, by - bstd, by + bstd, color="tab:blue", alpha=0.18)

    plt.plot(tx, ty, color="tab:orange", linewidth=2.8, label=f"Tuned (n={len(tuned_curves)})")
    if args.show_std_band and len(tuned_curves) > 1:
        plt.fill_between(tx, ty - tstd, ty + tstd, color="tab:orange", alpha=0.18)

    plt.title(f"Hyperparameter Comparison ({args.env_name})")
    plt.xlabel("Number of environment steps")
    plt.ylabel("Eval average return")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()

    print(f"[saved] {args.out}")
    print(_summarize("Default", bx, by))
    print(_summarize("Tuned", tx, ty))
    print("Baseline runs used:")
    for d in baseline_dirs:
        print(f"  {d}")
    print("Tuned runs used:")
    for d in tuned_dirs:
        print(f"  {d}")


if __name__ == "__main__":
    main()
