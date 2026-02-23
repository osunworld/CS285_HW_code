import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

DATA_DIR = Path("data")
TAG = "eval_return"

runs = sorted(DATA_DIR.glob("hw3_dqn_*LunarLander-v2*"), key=lambda p: p.stat().st_mtime)
basic = [r for r in runs if "_doubleq" not in r.name][-3:]   # 최근 3개
doubleq = [r for r in runs if "_doubleq" in r.name][-3:]       # 최근 3개

if len(basic) < 1 or len(doubleq) < 1:
    raise RuntimeError("LunarLander basic/doubleq 로그를 찾지 못했습니다.")

def load_scalar(run_dir, tag):
    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None, None
    ev = ea.Scalars(tag)
    x = np.array([e.step for e in ev], dtype=np.int64)
    y = np.array([e.value for e in ev], dtype=np.float32)
    return x, y

def plot_group(group, color, label):
    curves = []
    for r in group:
        x, y = load_scalar(r, TAG)
        if x is None:
            continue
        plt.plot(x, y, color=color, alpha=0.25, linewidth=1.0)
        curves.append((x, y))

    if not curves:
        return

    m = min(len(y) for _, y in curves)
    x0 = curves[0][0][:m]
    ys = np.vstack([y[:m] for _, y in curves])
    mean = ys.mean(axis=0)
    std = ys.std(axis=0)

    plt.plot(x0, mean, color=color, linewidth=2.5, label=label)
    plt.fill_between(x0, mean - std, mean + std, color=color, alpha=0.15)

plt.figure(figsize=(10, 6))
plot_group(basic, "blue", "Basic DQN (3 seeds)")
plot_group(doubleq, "red", "Double DQN (3 seeds)")

plt.xlabel("Environment steps")
plt.ylabel("Eval return")
plt.title("LunarLander-v2: Basic DQN vs Double DQN")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("hw3_lunar_basic_vs_doubleq.png", dpi=180)
print("saved: hw3_lunar_basic_vs_doubleq.png")
