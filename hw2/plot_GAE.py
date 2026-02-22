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

# Use a readable style if available, but fall back gracefully.
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


def _lambda_candidates(exp_prefix: str, lam: str) -> List[str]:
    lam = lam.strip()
    if not lam:
        return []
    forms = [lam, lam.replace(".", "p"), lam.replace(".", "_"), lam.replace(".", "")]
    candidates: List[str] = []
    for f in forms:
        name = f"{exp_prefix}{f}"
        if name not in candidates:
            candidates.append(name)
    return candidates


def _match_runs_for_lambdas(
    latest_by_exp: Dict[str, str], exp_prefix: str, lambdas: List[str]
) -> Dict[str, str]:
    matched: Dict[str, str] = {}
    for lam in lambdas:
        candidates = _lambda_candidates(exp_prefix, lam)
        picked = None
        for cand in candidates:
            if cand in latest_by_exp:
                picked = cand
                break
        if picked is None:
            # Fallback: relaxed search by substring.
            token = f"{exp_prefix}{lam}"
            relaxed = [e for e in latest_by_exp.keys() if token in e]
            if relaxed:
                picked = sorted(relaxed)[0]
        if picked is not None:
            matched[lam] = latest_by_exp[picked]
    return matched


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


def _lambda_sort_key(lam: str):
    try:
        return float(lam)
    except ValueError:
        return lam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--env_name", type=str, default="LunarLander-v2")
    parser.add_argument("--exp_prefix", type=str, default="lunar_lander_lambda")
    parser.add_argument("--lambdas", type=str, default="0,0.95,0.98,0.99,1")
    parser.add_argument("--smooth_window", type=int, default=15)
    parser.add_argument(
        "--show_raw",
        action="store_true",
        help="Overlay raw noisy curves as faint lines.",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=2.0,
        help="Clip y-axis to [p, 100-p] percentile of smoothed values. 0 disables.",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=2.5,
    )
    parser.add_argument("--out", type=str, default="hw2_lunarlander_gae_curve.png")
    args = parser.parse_args()

    lambdas = [x.strip() for x in args.lambdas.split(",") if x.strip()]
    latest_by_exp = _latest_runs_by_exp(args.data_dir, args.env_name)
    matched = _match_runs_for_lambdas(latest_by_exp, args.exp_prefix, lambdas)

    if not matched:
        raise FileNotFoundError(
            "No lambda runs matched. "
            "Check --env_name, --exp_prefix, --lambdas, and data directory."
        )

    missing = [lam for lam in lambdas if lam not in matched]
    if missing:
        print(f"[warn] Missing lambdas (no run found): {missing}")

    plt.figure(figsize=(10, 6))
    summary = []
    all_smoothed_values: List[np.ndarray] = []
    for lam in sorted(matched.keys(), key=_lambda_sort_key):
        run_dir = matched[lam]
        x, y = _load_curve(run_dir)
        y_plot = _maybe_smooth(y, args.smooth_window)
        if args.show_raw:
            plt.plot(x, y, linewidth=1.0, alpha=0.25)
        plt.plot(x, y_plot, linewidth=args.linewidth, label=f"lambda={lam}")
        all_smoothed_values.append(y_plot)
        summary.append(
            {
                "lambda": lam,
                "final": float(y[-1]),
                "best": float(np.max(y)),
                "last10_mean": float(np.mean(y[-10:] if len(y) >= 10 else y)),
                "run_dir": run_dir,
            }
        )

    if args.clip_percentile > 0 and all_smoothed_values:
        yy = np.concatenate(all_smoothed_values)
        lo = np.percentile(yy, args.clip_percentile)
        hi = np.percentile(yy, 100 - args.clip_percentile)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            pad = 0.05 * (hi - lo)
            plt.ylim(lo - pad, hi + pad)

    plt.title(f"LunarLander GAE Sweep ({args.env_name})")
    plt.xlabel("Number of environment steps")
    plt.ylabel("Eval average return")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()

    print(f"[saved] {args.out}")
    print("Runs used:")
    for row in summary:
        print(
            f"  lambda={row['lambda']}: final={row['final']:.2f}, "
            f"best={row['best']:.2f}, last10_mean={row['last10_mean']:.2f}"
        )
        print(f"    {row['run_dir']}")


if __name__ == "__main__":
    main()
