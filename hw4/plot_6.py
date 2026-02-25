

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

runs = {
    "Model-free SAC (L=0)": "data/cheetah-cs285-v0_cheetah_mbpo_l2_h250_mpcrandom_horizon10_actionseq1000_24-02-2026_23-10-51",
    "Dyna-style (L=1)": "data/cheetah-cs285-v0_cheetah_mbpo_l2_h250_mpcrandom_horizon10_actionseq1000_24-02-2026_23-11-35",
    "Full MBPO (L=10)": "data/cheetah-cs285-v0_cheetah_mbpo_l2_h250_mpcrandom_horizon10_actionseq1000_24-02-2026_23-11-52",
}



for label, r in runs.items():
    ea = event_accumulator.EventAccumulator(r); ea.Reload()
    if "eval_return" not in ea.Tags().get("scalars", []):
        print(f"[skip] no eval_return: {r}")
        continue
    s = ea.Scalars("eval_return")
    x = [e.step for e in s]
    y = [e.value for e in s]
    plt.plot(x, y, linewidth=2, label=label)

plt.title("HalfCheetah MBPO Comparison")
plt.xlabel("Iteration")
plt.ylabel("Eval return")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("hw4_mbpo_compare.png", dpi=170)
print("saved: hw4_mbpo_compare.png")