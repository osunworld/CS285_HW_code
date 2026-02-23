import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

DATA_DIR = Path("data")
TAG = "eval_return"

# (run 이름 키, 범례 라벨)
EXP_KEYS = [
    ("lunar_hp_lr1e3", "lr=1e-3"),
    ("lunar_hp_lr3e4", "lr=3e-4"),
    ("lunar_hp_lr3e3", "lr=3e-3"),
    ("lunar_hp_lr1e3_epsfix", "lr=1e-3 + eval_eps_fix"),
]

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LINESTYLES = ["-", "--", "-.", ":"]


def load_eval_curve(run_dir: Path):
    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if TAG not in ea.Tags()["scalars"]:
        return None, None
    s = ea.Scalars(TAG)
    x = np.array([e.step for e in s], dtype=np.int64)
    y = np.array([e.value for e in s], dtype=np.float32)
    return x, y


plt.figure(figsize=(11, 6))

for i, ((key, label), color, ls) in enumerate(zip(EXP_KEYS, COLORS, LINESTYLES)):
    cands = sorted(DATA_DIR.glob(f"hw3_dqn_*{key}*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        print(f"[warn] run not found: {key}")
        continue

    run = cands[-1]
    x, y = load_eval_curve(run)
    if x is None:
        print(f"[warn] eval_return not found: {run.name}")
        continue

    # 스무딩 없이 raw eval_return 그대로 그림
    # 겹칠 때도 색이 보이도록:
    # - alpha < 1
    # - line style 다르게
    # - marker 위치를 곡선마다 오프셋
    plt.plot(
        x, y,
        color=color,
        linestyle=ls,
        linewidth=2.8,
        alpha=0.78,
        marker="o",
        markevery=(i, 5),   # 곡선마다 다른 위치에 마커
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label=label,
    )

    print(f"[ok] {label:22s} -> {run.name} (last={y[-1]:.2f})")

plt.title("LunarLander Hyperparameter Comparison")
plt.xlabel("Environment steps")
plt.ylabel("Eval return")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("hw3_lunar_hyper_eval_return.png", dpi=180)
print("saved: hw3_lunar_hyper_eval_return.png")
