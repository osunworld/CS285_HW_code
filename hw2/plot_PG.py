import os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SMALL = ["cartpole", "cartpole_rtg", "cartpole_na", "cartpole_rtg_na"]
LARGE = ["cartpole_lb", "cartpole_lb_rtg", "cartpole_lb_na", "cartpole_lb_rtg_na"]

def latest_run_dir(exp_name):
    cands = glob.glob(f"data/q2_pg_{exp_name}_CartPole-v0_*")
    if not cands:
        raise FileNotFoundError(f"No run for {exp_name}")
    return max(cands, key=os.path.getmtime)

def load_curve(run_dir):
    event_file = max(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")), key=os.path.getmtime)
    ea = EventAccumulator(event_file)
    ea.Reload()

    envsteps = {e.step: e.value for e in ea.Scalars("Train_EnvstepsSoFar")}
    eval_ret = ea.Scalars("Eval_AverageReturn")

    pts = [(envsteps[e.step], e.value) for e in eval_ret if e.step in envsteps]
    pts.sort(key=lambda x: x[0])
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    return x, y

def plot_group(exp_names, title, out_png):
    plt.figure(figsize=(8,5))
    summary = []
    for exp in exp_names:
        run_dir = latest_run_dir(exp)
        x, y = load_curve(run_dir)
        plt.plot(x, y, label=exp, linewidth=2)
        final = y[-1]
        best = y.max()
        auc = np.trapz(y, x) / (x[-1] - x[0] + 1e-8)
        summary.append((exp, final, best, auc))

    plt.xlabel("Number of environment steps")
    plt.ylabel("Average return")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    print(f"Saved: {out_png}")
    print("\nSummary (exp, final, best, auc-like):")
    for row in summary:
        print(row)

if __name__ == "__main__":
    plot_group(SMALL, "CartPole PG (Small batch)", "hw2_small_batch.png")
    plot_group(LARGE, "CartPole PG (Large batch)", "hw2_large_batch.png")
