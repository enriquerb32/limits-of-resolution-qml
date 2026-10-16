"""Figures built ONLY from numbers reported in summary_for_Pawel.md (Exp 7 and Exp 8)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("paper/figs", exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig: observed gaps vs minimum detectable effect (train-only rerun, 15-seed means)
pairs = ["RF $-$ XGBoost", "RF $-$ LSTM", "XGBoost $-$ LSTM"]
gap = [0.021, 0.015, -0.007]
mde_iid = [0.036, 0.038, 0.052]
mde_blk = [0.039, 0.027, 0.051]
fig, ax = plt.subplots(figsize=(6.2, 2.7))
y = np.arange(len(pairs))[::-1]
for yi, g, a, b in zip(y, gap, mde_iid, mde_blk):
    ax.barh(yi + 0.14, 2 * a, left=-a, height=0.22, color="#c9d6e8", edgecolor="none", label="$\\pm$MDE, DeLong (iid)" if yi == y[0] else None)
    ax.barh(yi - 0.14, 2 * b, left=-b, height=0.22, color="#f1d3b3", edgecolor="none", label="$\\pm$MDE, year-block bootstrap" if yi == y[0] else None)
    ax.plot([g], [yi], "o", color="#1f2a44", ms=6, zorder=5, label="observed gap (15-seed means)" if yi == y[0] else None)
ax.axvline(0, color="0.3", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(pairs)
ax.set_xlabel("AUC difference")
ax.set_xlim(-0.06, 0.06)
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, handlelength=1.2, columnspacing=1.0)
fig.tight_layout()
fig.savefig("paper/figs/fig_noise_budget.pdf"); plt.close(fig)

# ---- Fig: positive control detection curves
d = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
same_iid = [np.nan, 8, 20, 52, 84, 97]
same_blk = [np.nan, 15, 30, 62, 80, 92]
diff_iid = [2, 7, 17, 47, 75, 93]
diff_blk = [5, 12, 27, 57, 76, 92]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(d, diff_iid, "-o", color="#1f4e9c", ms=4, label="different seeds, DeLong (iid)")
ax.plot(d, diff_blk, "-s", color="#c4601a", ms=4, label="different seeds, year-block bootstrap")
ax.plot(d, same_iid, "--o", color="#1f4e9c", ms=3, alpha=0.5, label="same seed, DeLong (iid)")
ax.plot(d, same_blk, "--s", color="#c4601a", ms=3, alpha=0.5, label="same seed, year-block bootstrap")
ax.axhline(80, color="0.4", lw=0.8, ls=":")
ax.text(0.0005, 82, "80 % detection", fontsize=7.5, color="0.3")
ax.axhline(5, color="0.6", lw=0.6, ls=":")
ax.set_xlabel("True AUC loss $\\delta$ imposed on Random Forest")
ax.set_ylabel("Runs in which the loss is detected (%)")
ax.set_ylim(0, 100); ax.set_xlim(-0.002, 0.052)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig("paper/figs/fig_positive_control.pdf"); plt.close(fig)
print("ok")
