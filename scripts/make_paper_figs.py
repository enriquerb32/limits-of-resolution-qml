"""
scripts/make_paper_figs.py - Generate all paper figures from saved results JSONs.

Produces:
  paper/figs/fig_paired_ci.pdf (.png)      <- results/json/results_exp10_paired_ci.json
  paper/figs/fig_sim.pdf (.png)            <- results/json/sim{B_rerun,C_diff,D_unequal,E_strong}.json
  paper/figs/fig_noise_budget.pdf          <- results/local_exp7_noise_budget.json
  paper/figs/fig_positive_control.pdf      <- results/local_exp8_positive_control.json
"""
import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_JSON = os.path.join(ROOT, "results", "json")
RES_LOCAL = os.path.join(ROOT, "results")
OUT_FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(OUT_FIGS, exist_ok=True)

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#6b6b6b"

def plot_paired_ci():
    exp10_path = os.path.join(RES_JSON, "results_exp10_paired_ci.json")
    if not os.path.exists(exp10_path):
        print(f"[skip] {exp10_path} not found")
        return
    data = json.load(open(exp10_path, encoding="utf-8"))["pairs"]
    
    pairs = [
        ("RF $-$ XGBoost", "RF-XGB"),
        ("RF $-$ LSTM", "RF-LSTM"),
        ("XGBoost $-$ LSTM", "XGB-LSTM"),
    ]
    
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    off = [0.22, 0, -0.22]
    lab = ["DeLong (iid)", "Year-block bootstrap", "Two-way bootstrap (years, seeds)"]
    mk = ["o", "s", "D"]
    col = [BLUE, ORANGE, GREY]
    
    for i, (label, key) in enumerate(pairs):
        row = data[key]
        d = row["delta"]
        se = row["se_delong"]
        ci_dl = (d - 1.96 * se, d + 1.96 * se)
        ci_blk = tuple(row["ci_block"])
        ci_two = tuple(row["ci_two_way"])
        cis = [ci_dl, ci_blk, ci_two]
        
        y0 = len(pairs) - 1 - i
        for k in range(3):
            ax.plot(cis[k], [y0 + off[k]] * 2, color=col[k], lw=1.6, solid_capstyle="butt")
            ax.plot([d], [y0 + off[k]], marker=mk[k], color=col[k], ms=5,
                    mec="white", mew=0.8, label=lab[k] if i == 0 else None, ls="none")
            
    ax.axvline(0, color="#333", lw=0.8, ls="--")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p[0] for p in pairs][::-1])
    ax.set_xlabel("Difference of 15-seed mean AUC (95% interval)")
    ax.set_xlim(-0.045, 0.05)
    ax.grid(axis="x", color="#e3e3e3", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, fontsize=7.5, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIGS, "fig_paired_ci.pdf"))
    fig.savefig(os.path.join(OUT_FIGS, "fig_paired_ci.png"), dpi=130)
    plt.close(fig)
    print("generated fig_paired_ci.pdf and fig_paired_ci.png")

def plot_sim():
    sc = [
        ("B", "simB_rerun.json"),
        ("C", "simC_diff.json"),
        ("D", "simD_unequal.json"),
        ("E", "simE_strong.json")
    ]
    names = ["B\nshared", "C\n+ scorer-\nspecific", "D\n+ unequal\nyears", "E\nall $\\tau$=0.6"]
    t1, ra = [], []
    for k, fn in sc:
        fp = os.path.join(RES_JSON, fn)
        if not os.path.exists(fp):
            print(f"[skip] {fp} not found")
            return
        r = json.load(open(fp, encoding="utf-8"))["rows"][0]
        t1.append((r["rejection_delong"], r["rejection_block"]))
        ra.append((r["sd_delta_hat"] / r["mean_se_delong"], r["sd_delta_hat"] / r["mean_se_block"]))
        
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
    x = np.arange(4)
    w = 0.36
    
    def bars(ax, vals, ylabel, ref=None, mc=None, ylim=None):
        ax.bar(x - w / 2, [v[0] for v in vals], w, color=BLUE, label="DeLong", zorder=3)
        ax.bar(x + w / 2, [v[1] for v in vals], w, color=ORANGE, label="Year-block bootstrap",
               zorder=3, hatch="///", edgecolor="white", lw=0)
        if ref is not None:
            ax.axhline(ref, color="#333", lw=0.9, ls="--", zorder=4)
        if mc:
            ax.axhspan(ref - mc, ref + mc, color="#999", alpha=0.18, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#e3e3e3", lw=0.6)
        ax.set_axisbelow(True)
        
    mc = 1.96 * math.sqrt(0.05 * 0.95 / 500)
    bars(axs[0], t1, "Type-I error (nominal 0.05)", ref=0.05, mc=mc, ylim=(0, 0.25))
    bars(axs[1], ra, "SD of estimate / mean SE", ref=1.0, ylim=(0.8, 1.7))
    axs[0].set_title("(a) Rejection of an exactly null difference", fontsize=8.5)
    axs[1].set_title("(b) Calibration of the standard error", fontsize=8.5)
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIGS, "fig_sim.pdf"))
    fig.savefig(os.path.join(OUT_FIGS, "fig_sim.png"), dpi=130)
    plt.close(fig)
    print("generated fig_sim.pdf and fig_sim.png")

def plot_noise_budget():
    fp = os.path.join(RES_LOCAL, "local_exp7_noise_budget.json")
    if not os.path.exists(fp):
        print(f"[skip] {fp} not found")
        return
    d = json.load(open(fp, encoding="utf-8"))["train_only"]["pairs"]
    pairs = ["RF $-$ XGBoost", "RF $-$ LSTM", "XGBoost $-$ LSTM"]
    keys = ["RF-XGB", "RF-LSTM", "XGB-LSTM"]
    gap = [d[k]["gap_seed_mean"] for k in keys]
    mde_iid = [d[k]["mde_fixed_seed_iid"] for k in keys]
    mde_blk = [d[k]["mde_fixed_seed_block"] for k in keys]
    
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    y = np.arange(len(pairs))[::-1]
    for yi, g, a, b in zip(y, gap, mde_iid, mde_blk):
        ax.barh(yi + 0.14, 2 * a, left=-a, height=0.22, color="#c9d6e8", edgecolor="none",
                label="$\\pm$MDE, DeLong (iid)" if yi == y[0] else None)
        ax.barh(yi - 0.14, 2 * b, left=-b, height=0.22, color="#f1d3b3", edgecolor="none",
                label="$\\pm$MDE, year-block bootstrap" if yi == y[0] else None)
        ax.plot([g], [yi], "o", color="#1f2a44", ms=6, zorder=5,
                label="observed gap (15-seed means)" if yi == y[0] else None)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(pairs)
    ax.set_xlabel("AUC difference")
    ax.set_xlim(-0.06, 0.06)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.28),
              ncol=3, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIGS, "fig_noise_budget.pdf"))
    plt.close(fig)
    print("generated fig_noise_budget.pdf")

def plot_positive_control():
    fp = os.path.join(RES_LOCAL, "local_exp8_positive_control.json")
    if not os.path.exists(fp):
        print(f"[skip] {fp} not found")
        return
    d = json.load(open(fp, encoding="utf-8"))["by_delta"]
    deltas = sorted([float(k) for k in d.keys()])
    same_iid = [np.nan if k == 0.0 else d[f"{k:.2f}"]["paired"]["detect_iid"] * 100 for k in deltas]
    same_blk = [np.nan if k == 0.0 else d[f"{k:.2f}"]["paired"]["detect_block"] * 100 for k in deltas]
    diff_iid = [d[f"{k:.2f}"]["cross"]["detect_iid"] * 100 for k in deltas]
    diff_blk = [d[f"{k:.2f}"]["cross"]["detect_block"] * 100 for k in deltas]
    
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    d_arr = np.array(deltas)
    ax.plot(d_arr, diff_iid, "-o", color="#1f4e9c", ms=4, label="different seeds, DeLong (iid)")
    ax.plot(d_arr, diff_blk, "-s", color="#c4601a", ms=4, label="different seeds, year-block bootstrap")
    ax.plot(d_arr, same_iid, "--o", color="#1f4e9c", ms=3, alpha=0.5, label="same seed, DeLong (iid)")
    ax.plot(d_arr, same_blk, "--s", color="#c4601a", ms=3, alpha=0.5, label="same seed, year-block bootstrap")
    ax.axhline(80, color="0.4", lw=0.8, ls=":")
    ax.text(0.0005, 82, "80 % detection", fontsize=7.5, color="0.3")
    ax.axhline(5, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("Imposed AUC reduction $\\delta$ on Random Forest")
    ax.set_ylabel("Runs in which the loss is detected (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.002, 0.052)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIGS, "fig_positive_control.pdf"))
    plt.close(fig)
    print("generated fig_positive_control.pdf")

if __name__ == "__main__":
    plot_paired_ci()
    plot_sim()
    plot_noise_budget()
    plot_positive_control()
    print("all figures successfully generated from results/")
