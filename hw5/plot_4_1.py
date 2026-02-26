
import glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def read_scalar(run_dir, tag):
    ea = event_accumulator.EventAccumulator(run_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    s = ea.Scalars(tag)
    x = np.array([e.step for e in s], dtype=float)
    y = np.array([e.value for e in s], dtype=float)
    return x, y

# CQL runs (alpha sweep)
cql_runs = sorted(glob.glob("data/hw5_offline_*PointmassMedium-v0_cql*"))
# optional DQN baseline
dqn_runs = sorted(glob.glob("data/hw5_offline_*pointmass_medium_dqn*"))

alpha_to_run = {}
for r in cql_runs:
    m = re.search(r"_cql([0-9.]+)_", os.path.basename(r))
    if m:
        alpha = float(m.group(1))
        alpha_to_run[alpha] = r  # latest by sorted order

alphas = sorted(alpha_to_run.keys())

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for a in alphas:
    r = alpha_to_run[a]
    x1, y1 = read_scalar(r, "eval_return")
    x2, y2 = read_scalar(r, "q_values")
    if x1 is not None:
        axes[0].plot(x1, y1, label=f"alpha={a}")
    if x2 is not None:
        axes[1].plot(x2, y2, label=f"alpha={a}")

# DQN baseline (alpha=0 equivalent) if exists
if dqn_runs:
    r = dqn_runs[-1]
    x1, y1 = read_scalar(r, "eval_return")
    x2, y2 = read_scalar(r, "q_values")
    if x1 is not None:
        axes[0].plot(x1, y1, "--", color="black", label="DQN baseline")
    if x2 is not None:
        axes[1].plot(x2, y2, "--", color="black", label="DQN baseline")

axes[0].set_title("PointmassMedium: CQL alpha sweep")
axes[0].set_ylabel("eval_return")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].set_xlabel("training step")
axes[1].set_ylabel("q_values")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig("hw5_cql_alpha_medium.png", dpi=180)
print("saved: hw5_cql_alpha_medium.png")
