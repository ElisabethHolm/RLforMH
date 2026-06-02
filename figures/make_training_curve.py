"""Generate BCQ training curve for poster. Run from repo root."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
BCQ_LOG = PROJECT_ROOT / "models" / "bcq_training_log.csv"
CQL_TD_CSV = PROJECT_ROOT / "d3rlpy_logs" / "DiscreteCQL_20260516111543" / "td_loss.csv"
CQL_CONS_CSV = PROJECT_ROOT / "d3rlpy_logs" / "DiscreteCQL_20260516111543" / "conservative_loss.csv"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.alpha": 0.25,
    "font.family": "sans-serif",
    "font.size": 11,
})

bcq = pd.read_csv(BCQ_LOG)

# CQL log has columns: epoch, step, value
cql_td = pd.read_csv(CQL_TD_CSV, header=None, names=["epoch", "step", "value"])
cql_cons = pd.read_csv(CQL_CONS_CSV, header=None, names=["epoch", "step", "value"])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

# ---- Left: BCQ training losses ----
ax = axes[0]
ax.plot(bcq["step"], bcq["total_loss"], color="#2171b5", linewidth=1.5,
        label="BCQ total loss", alpha=0.9, zorder=3)
ax.plot(bcq["step"], bcq["bc_loss"], color="#6baed6", linewidth=1.2,
        linestyle="--", label="BC loss (cloning)", alpha=0.85, zorder=3)
ax.plot(bcq["step"], bcq["td_loss"], color="#9ecae1", linewidth=1.0,
        linestyle=":", label="TD loss (Q-function)", alpha=0.85, zorder=3)

ax.set_xlabel("Gradient step", labelpad=8)
ax.set_ylabel("Loss", labelpad=8)
ax.set_title("BCQ Training Loss", fontweight="semibold", pad=10)
ax.legend(frameon=True, fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.2)

# ---- Right: CQL training ----
ax2 = axes[1]
ax2.plot(cql_td["step"], cql_td["value"], color="#6baed6", linewidth=2.0,
         marker="o", markersize=7, label="TD loss", zorder=3)
ax2.plot(cql_cons["step"], cql_cons["value"], color="#c6dbef", linewidth=2.0,
         marker="s", markersize=7, label="Conservative loss", zorder=3)
ax2.set_xlabel("Training step", labelpad=8)
ax2.set_ylabel("Loss", labelpad=8)
ax2.set_title("CQL Training Loss (final checkpoint)", fontweight="semibold", pad=10)
ax2.legend(frameon=True, fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.2)
ax2.set_xticks(cql_td["step"].unique())
ax2.text(0.5, 0.5, "Single logged checkpoint\n(d3rlpy logs one point\nper 10k-step run)",
         transform=ax2.transAxes, ha="center", va="center",
         fontsize=9, color="#888888", style="italic")

fig.suptitle("Algorithm Training Curves (StudentLife Offline Dataset)",
             fontsize=13, fontweight="semibold", y=1.01)
fig.tight_layout()

out = FIGURES_DIR / "training_curves.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out}")
