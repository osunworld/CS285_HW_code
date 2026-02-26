
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def read_eval(run_dir):
    ea = event_accumulator.EventAccumulator(run_dir); ea.Reload()
    s = ea.Scalars("eval_return")
    x = np.array([e.step for e in s]); y = np.array([e.value for e in s])
    return x, y

awac = sorted(glob.glob("data/hw5_offline_*awac*"))[-1]
iql  = sorted(glob.glob("data/hw5_offline_*iql*"))[-1]

xa, ya = read_eval(awac)
xi, yi = read_eval(iql)

plt.figure(figsize=(8,5))
plt.plot(xa, ya, label="AWAC")
plt.plot(xi, yi, label="IQL")
plt.xlabel("training step")
plt.ylabel("eval_return")
plt.title("AWAC vs IQL (offline)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("hw5_awac_vs_iql_eval.png", dpi=180)
print("saved: hw5_awac_vs_iql_eval.png")