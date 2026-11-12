"""Step 10. 
Tables and figures: score-vs-RMSD for Vina and the three rescoring methods, and the rank of the best native-like pose per seed (native-like = heavy-atom RMSD <= 2.0 A to the crystal pose).

Reads the reference results in data/results/ by default. 
To analyze regenerated files in work/, pass --work 
"""
import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from redock.common import REF, WORK, work

ap = argparse.ArgumentParser(); ap.add_argument("--work", action="store_true", help="use results regenerated in work/ instead of data/results/")
a = ap.parse_args()
R = WORK if a.work else REF
X = np.load(R / "poses_xyz.npy"); meta = json.load(open(R / "poses_meta.json"))
gb = json.load(open(R / "openmm_mmgbsa_results.json"))
am = json.load(open((R / "amber" / "amber_results.json") if not a.work else (WORK / "amber" / "results_all.json")))
n = 45
rm = np.array([m["rmsd"] for m in meta[:n]]); seed = np.array([m["seed"] for m in meta[:n]])
sc = {"AutoDock Vina score": np.array([m["vina"] for m in meta[:n]]),
      "OpenMM MM-GBSA": np.array([g["dE_gbsa"] for g in gb[:n]]),
      "Amber MM-GBSA": np.array([x["amber"]["gb"] for x in am[:n]]),
      "Amber MM-PBSA": np.array([x["amber"]["pb"] for x in am[:n]])}
assert all(x["index"] == i for i, x in enumerate(am)) and [g["seed"] for g in gb[:n]] == list(seed)
cryst = {"OpenMM MM-GBSA": gb[n]["dE_gbsa"], "Amber MM-GBSA": am[n]["amber"]["gb"], "Amber MM-PBSA": am[n]["amber"]["pb"]}
BG, PANEL, LINE, FG, MUTED = "#14181c", "#1b2026", "#3a434d", "#e6e9ec", "#a3adb8"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": PANEL,
                     "savefig.facecolor": BG, "axes.edgecolor": LINE, "axes.labelcolor": FG, "text.color": FG, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": FG, "grid.color": LINE, "legend.labelcolor": FG})
BL, GR, RD = "#6aa5e6", "#6b7783", "#ee8470"
MC = ["#9aa7b4", "#5cc2b8", "#6aa5e6", "#b49be0"]
# Figure A: score vs RMSD, four methods
fig, axs = plt.subplots(1, 4, figsize=(15, 3.7))
for ax, (k, y) in zip(axs, sc.items()):
    nat = rm <= 2.0
    ax.scatter(rm[~nat], y[~nat], c=GR, s=26, label="not native-like", alpha=0.9); ax.scatter(rm[nat], y[nat], c=BL, s=34, label="native-like (<2 Å)")
    top = [i for i in range(n) if meta[i]["rank"] == 1]
    ax.scatter(rm[top], y[top], facecolors="none", edgecolors=RD, s=95, lw=1.6, label="Vina rank 1")
    if k in cryst:
        ax.scatter([0], [cryst[k]], marker="*", c="#d6b658", s=190, edgecolors=BG, label="crystal pose", zorder=5)
    ax.set_xlabel("RMSD to crystal pose (Å)"); ax.set_title(k, fontsize=10)
axs[0].set_ylabel("score (kcal/mol, lower = better)")
h, l = axs[3].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=FG)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(work("figures", "fig_score_vs_rmsd_4way.png"), dpi=200); plt.close(fig)
# Figure B: rank of the best native-like pose per seed
fig, ax = plt.subplots(figsize=(7, 3.6)); seeds = [1, 2, 3, 4, 5]; mk = ["o", "s", "^", "D"]; table = {}
for j, (k, y) in enumerate(sc.items()):
    r = []
    for s_ in seeds:
        idx = np.where(seed == s_)[0]; order = idx[np.argsort(y[idx])]
        r.append(next(p + 1 for p, i in enumerate(order) if rm[i] <= 2.0))
    table[k] = r; print(k, r)
    ax.plot(np.arange(5) + (j - 1.5) * 0.12, r, mk[j], ms=8, color=MC[j], label=k, ls="")
ax.set_xticks(range(5)); ax.set_xticklabels([f"seed {s_}" for s_ in seeds]); ax.set_ylabel("rank of best native-like pose\n(1 = scored best of 9)")
ax.set_ylim(9.5, 0.5); ax.set_yticks(range(1, 10)); ax.grid(axis="y", color=LINE, lw=0.6)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right"); fig.tight_layout(); fig.savefig(work("figures", "fig_native_rank.png"), dpi=200); plt.close(fig)
work("rank_table.json").write_text(json.dumps(table, indent=1))
print("ok")
