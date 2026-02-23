import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def moving_average(y, w):
    if len(y) < w:
        return np.array(y)
    return np.convolve(y, np.ones(w) / w, mode="valid")


# latest MsPacman run
runs = sorted(glob.glob("data/hw3_dqn_*MsPacmanNoFrameskip-v0*"), key=os.path.getmtime)
if not runs:
    raise RuntimeError("No MsPacman run found in data/")
run = runs[-1]
print("Using run:", run)

ea = event_accumulator.EventAccumulator(run)
ea.Reload()

train_tag = "train_return"
eval_tag = "eval_return"

if train_tag not in ea.Tags()["scalars"] or eval_tag not in ea.Tags()["scalars"]:
    raise RuntimeError("train_return or eval_return tag missing in event file.")

train = ea.Scalars(train_tag)
ev = ea.Scalars(eval_tag)

x_train = np.array([e.step for e in train], dtype=np.int64)
y_train = np.array([e.value for e in train], dtype=np.float32)

x_eval = np.array([e.step for e in ev], dtype=np.int64)
y_eval = np.array([e.value for e in ev], dtype=np.float32)

# smoothing windows
w_train = 50
w_eval = 5

y_train_s = moving_average(y_train, w_train)
x_train_s = x_train[w_train - 1:] if len(y_train) >= w_train else x_train

y_eval_s = moving_average(y_eval, w_eval)
x_eval_s = x_eval[w_eval - 1:] if len(y_eval) >= w_eval else x_eval

plt.figure(figsize=(11, 6))

# raw (faint)
plt.plot(x_train, y_train, color="tab:orange", alpha=0.15, linewidth=1, label="train_return (raw)")
plt.plot(x_eval, y_eval, color="tab:blue", alpha=0.20, linewidth=1, label="eval_return (raw)")

# smoothed (main)
plt.plot(x_train_s, y_train_s, color="tab:orange", linewidth=2.5, label=f"train_return (MA{w_train})")
plt.plot(x_eval_s, y_eval_s, color="tab:blue", linewidth=2.5, label=f"eval_return (MA{w_eval})")

plt.title("MsPacman: Train vs Eval Return ")
plt.xlabel("Environment steps")
plt.ylabel("Return")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("hw3_mspacman_train_vs_eval.png", dpi=180)
print("Saved: hw3_mspacman_train_vs_eval.png")
