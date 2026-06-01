"""
Visualize DQN / Double DQN and BCQ results against baselines on the test split.

Reads saved evaluation artifacts (no model loading by default):
  - models/dqn_saved_model_ope_metrics.csv
  - models/extended_comparison_results.csv

Run from repo root:
    python algorithms/visualize_policy_comparison.py
    python algorithms/visualize_policy_comparison.py --all-policies
    python algorithms/visualize_policy_comparison.py --split test

If DQN rows are missing for reward_dense, run first:
    python algorithms/evaluate_dqn_models.py --best-per-algo --retrain-missing \\
        --reward-variants reward_dense --splits test
    python algorithms/extended_policy_comparison.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:  # pragma: no cover
    HAS_SEABORN = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

DQN_METRICS_CSV = MODEL_DIR / "dqn_saved_model_ope_metrics.csv"
EXTENDED_CSV = MODEL_DIR / "extended_comparison_results.csv"
SUBGROUP_CSV = MODEL_DIR / "dqn_subgroup_policy_analysis.csv"
OUT_TABLE_CSV = MODEL_DIR / "policy_comparison_plot_table.csv"

ACTION_COLS = [
    "action_none",
    "action_increase_activity",
    "action_decrease_activity",
    "action_increase_sleep",
    "action_decrease_sleep",
    "action_increase_social",
    "action_decrease_social",
]
ACTION_LABELS = [
    "none",
    "↑ activity",
    "↓ activity",
    "↑ sleep",
    "↓ sleep",
    "↑ social",
    "↓ social",
]

FOCUS_POLICIES = [
    "behavior_cloning_logistic",
    "contextual_bandit",
    "bcq",
    "dqn",
    "double_dqn",
]

POLICY_ORDER = [
    "random_uniform",
    "majority_action",
    "action_frequency",
    "rule_based",
    "behavior_cloning_logistic",
    "contextual_bandit",
    "bcq",
    "cql",
    "dqn",
    "double_dqn",
]

POLICY_LABELS = {
    "random_uniform": "Random",
    "majority_action": "Majority",
    "action_frequency": "Freq. sampling",
    "rule_based": "Rule-based",
    "behavior_cloning_logistic": "Beh. cloning",
    "contextual_bandit": "Ctx. bandit",
    "bcq": "BCQ",
    "cql": "CQL",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
}

POLICY_TYPE = {
    "random_uniform": "Baseline",
    "majority_action": "Baseline",
    "action_frequency": "Baseline",
    "rule_based": "Baseline",
    "behavior_cloning_logistic": "Baseline",
    "contextual_bandit": "Bandit",
    "bcq": "Offline RL",
    "cql": "Offline RL",
    "dqn": "Offline RL",
    "double_dqn": "Offline RL",
}

# Distinct colors per policy (poster-friendly palette).
POLICY_COLORS = {
    "random_uniform": "#bdbdbd",
    "majority_action": "#9e9e9e",
    "action_frequency": "#757575",
    "rule_based": "#616161",
    "behavior_cloning_logistic": "#525252",
    "contextual_bandit": "#fd8d3c",
    "bcq": "#2171b5",
    "cql": "#6baed6",
    "dqn": "#41ab5d",
    "double_dqn": "#238b45",
}

TYPE_COLORS = {
    "Baseline": "#9e9e9e",
    "Bandit": "#fd8d3c",
    "Offline RL": "#2171b5",
}

OPE_METRICS = [
    ("pdis", "PDIS"),
    ("weighted_pdis", "Weighted PDIS"),
    ("dr", "Doubly robust"),
]
OPE_COLORS = ["#4c72b0", "#55a868", "#c44e52"]


def apply_plot_style() -> None:
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "font.family": "sans-serif",
        }
    )


def _display_name(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy.replace("_", " ").title())


def _policy_type(policy: str) -> str:
    return POLICY_TYPE.get(policy, "Baseline")


def _policy_color(policy: str) -> str:
    return POLICY_COLORS.get(policy, "#666666")


def _sort_policies(df: pd.DataFrame) -> pd.DataFrame:
    order = {name: idx for idx, name in enumerate(POLICY_ORDER)}
    df = df.copy()
    df["_order"] = df["policy"].map(lambda p: order.get(p, 999))
    return df.sort_values(["_order", "policy"]).drop(columns="_order")


def _filter_policies(df: pd.DataFrame, focus: bool) -> pd.DataFrame:
    if not focus:
        return df
    keep = [p for p in FOCUS_POLICIES if p in set(df["policy"])]
    return df[df["policy"].isin(keep)].copy()


def load_dqn_rows(
    path: Path,
    split: str,
    reward_variant: str | None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python algorithms/evaluate_dqn_models.py "
            f"--best-per-algo --retrain-missing"
        )

    df = pd.read_csv(path)
    split_name = {"val": "val", "validation": "val"}.get(split, split)
    mask = df["split"] == split_name
    if reward_variant:
        mask &= df["reward_variant"] == reward_variant
    rows = df.loc[mask].copy()
    if rows.empty:
        raise ValueError(
            f"No DQN rows for split={split_name}, reward_variant={reward_variant}"
        )

    rows["policy"] = rows["algo"]
    rows["match_rate"] = rows["action_match"]
    rows["n_matched"] = np.nan
    rows["mean_behavior_support"] = np.nan
    rows["pct_low_support"] = np.nan
    rows["source"] = "dqn_ope"
    base_cols = [
        "policy",
        "reward_variant",
        "split",
        "pdis",
        "weighted_pdis",
        "dr",
        "match_rate",
        "mood_improvement",
        "n_matched",
        "mean_behavior_support",
        "pct_low_support",
        "effective_sample_size",
        "source",
    ]
    extra = [c for c in ACTION_COLS if c in rows.columns]
    return rows[base_cols + extra]


def load_dqn_action_rates(
    path: Path,
    split: str,
    reward_variant: str,
    policy: str,
) -> dict[str, float]:
    if not path.exists():
        return {}

    split_name = {"val": "val", "validation": "val"}.get(split, split)
    df = pd.read_csv(path)
    mask = (
        (df["reward_variant"] == reward_variant)
        & (df["split"] == split_name)
        & (df["algo"] == policy)
        & (df["subgroup"].isin(["mood_observed", "mood_missing"]))
    )
    sub = df.loc[mask]
    if sub.empty:
        return {}

    rate_cols = [c for c in sub.columns if c.startswith("policy_action_rate_")]
    total = sub["num_rows"].sum()
    rates = {}
    for col in rate_cols:
        action = col.replace("policy_action_rate_", "")
        rates[f"action_{action}"] = float(
            (sub[col] * sub["num_rows"]).sum() / total
        )
    return rates


def load_extended_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python algorithms/extended_policy_comparison.py"
        )

    df = pd.read_csv(path)
    df["reward_variant"] = "reward_dense"
    df["split"] = "test"
    df["source"] = "extended_comparison"
    for col in ACTION_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def build_comparison_table(
    split: str = "test",
    reward_variant: str = "reward_dense",
    focus: bool = True,
) -> pd.DataFrame:
    dqn = load_dqn_rows(DQN_METRICS_CSV, split=split, reward_variant=reward_variant)
    extended = load_extended_rows(EXTENDED_CSV)

    keep_cols = [
        "policy",
        "reward_variant",
        "split",
        "pdis",
        "weighted_pdis",
        "dr",
        "match_rate",
        "mood_improvement",
        "n_matched",
        "mean_behavior_support",
        "pct_low_support",
        "effective_sample_size",
        "source",
        *ACTION_COLS,
    ]
    extended = extended.reindex(columns=keep_cols)

    merged = pd.concat([extended, dqn], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=["policy"], keep="last")

    for policy in dqn["policy"].unique():
        if pd.notna(merged.loc[merged["policy"] == policy, ACTION_COLS[0]]).any():
            continue
        rates = load_dqn_action_rates(
            SUBGROUP_CSV, split, reward_variant, policy
        )
        if not rates:
            continue
        idx = merged["policy"] == policy
        for col, val in rates.items():
            merged.loc[idx, col] = val

    merged = _sort_policies(merged)
    merged["display_name"] = merged["policy"].map(_display_name)
    merged["policy_type"] = merged["policy"].map(_policy_type)
    merged["color"] = merged["policy"].map(_policy_color)
    return _filter_policies(merged, focus=focus)


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")
    return path


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_bars(
    ax,
    bars,
    fmt: str = "{:.3f}",
    *,
    inside_min: float = 0.045,
) -> None:
    """Place value labels on tall bars; above bar otherwise."""
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo if y_hi > y_lo else 1.0
    for bar in bars:
        height = bar.get_height()
        if height is None or not np.isfinite(height):
            continue
        label = fmt.format(height)
        if height >= inside_min:
            y_text = height * 0.55
            va = "center"
            color = "white"
            weight = "bold"
            size = 7.5
        elif height <= -0.001:
            y_text = height - 0.008 * span
            va = "top"
            color = "#333333"
            weight = "normal"
            size = 7.5
        else:
            y_text = height + 0.012 * span
            va = "bottom"
            color = "#333333"
            weight = "normal"
            size = 7.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            label,
            ha="center",
            va=va,
            fontsize=size,
            color=color,
            fontweight=weight,
            zorder=4,
        )


def plot_ope_metrics(df: pd.DataFrame, split: str, suffix: str) -> None:
    """
    Grouped bar chart comparing offline value estimators across policies.
    """
    policies = df["display_name"].tolist()
    n = len(policies)
    x = np.arange(n)
    width = 0.17

    split_label = "Test" if split == "test" else split.title()

    fig, ax = plt.subplots(figsize=(max(10, n * 1.15), 5.8))

    all_values = []
    bar_groups = []
    for idx, ((col, label), color) in enumerate(zip(OPE_METRICS, OPE_COLORS)):
        raw = df[col].to_numpy(dtype=float)
        values = np.where(np.isfinite(raw), raw, np.nan)
        all_values.extend(v for v in values if np.isfinite(v))
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        bar_groups.append(bars)

    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)

    if all_values:
        y_min = min(0.0, min(all_values))
        y_max = max(all_values)
        pad = max(0.012, (y_max - y_min) * 0.22)
        ax.set_ylim(y_min - pad * 0.35, y_max + pad)

    for bars in bar_groups:
        _annotate_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right", fontsize=10)
    ax.set_xlabel("Policy method", labelpad=10, fontsize=11)
    ax.set_ylabel(
        "Offline value estimate\n(dense wellness reward)",
        labelpad=10,
        fontsize=11,
    )
    ax.set_title(
        f"Hold-out {split_label}: Offline Policy Value (Dense Reward)",
        pad=12,
        fontsize=12,
        fontweight="semibold",
        loc="left",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.4,
        columnspacing=1.8,
    )
    _style_axes(ax)
    fig.subplots_adjust(bottom=0.34, left=0.11, right=0.98, top=0.90)

    _save(fig, f"ope_metrics_{split}{suffix}.png")


def plot_ope_vs_action_match(df: pd.DataFrame, split: str, suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))

    for _, row in df.iterrows():
        x = row["match_rate"] * 100
        y = row["weighted_pdis"]
        if pd.isna(x) or pd.isna(y):
            continue
        color = row["color"]
        ax.scatter(
            x,
            y,
            s=160,
            c=color,
            edgecolors="#333333",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            row["display_name"],
            (x, y),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
            color="#222222",
        )

    ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.set_xlabel("Action match with logged behavior (%)")
    ax.set_ylabel("Weighted PDIS")
    ax.set_title(
        f"Support overlap vs. estimated value ({split})",
        pad=12,
        fontweight="semibold",
    )
    _style_axes(ax)

    _save(fig, f"ope_vs_action_match_{split}{suffix}.png")


def plot_mood_improvement(df: pd.DataFrame, split: str, suffix: str) -> None:
    plot_df = df[df["mood_improvement"].notna() | df["n_matched"].notna()].copy()
    if plot_df.empty:
        print("Skipping mood figure: no mood-improvement values available.")
        return

    plot_df["mood_plot"] = plot_df["mood_improvement"].fillna(0.0)
    colors = plot_df["color"].tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.0), 5))
    bars = ax.bar(
        plot_df["display_name"],
        plot_df["mood_plot"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
        width=0.65,
        zorder=3,
    )
    ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.65)
    ax.set_ylabel("Mean Δ mood (matched actions)")
    ax.set_title(
        f"In-support mood improvement ({split}) — sparse mood signal",
        pad=12,
        fontweight="semibold",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        n = row.get("n_matched")
        label = f"n={int(n)}" if pd.notna(n) else "n=?"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.04 if bar.get_height() >= 0 else -0.08),
            label,
            ha="center",
            va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=8,
            color="#555555",
        )

    ax.text(
        0.01,
        0.02,
        "Only timesteps with observed mood and matching logged action.",
        transform=ax.transAxes,
        fontsize=8,
        color="#666666",
    )
    _style_axes(ax)

    _save(fig, f"mood_improvement_{split}{suffix}.png")


def plot_action_distribution(df: pd.DataFrame, split: str, suffix: str) -> None:
    action_df = df.dropna(subset=ACTION_COLS[:1], how="all").copy()
    if action_df.empty:
        print("Skipping action heatmap: no action distribution columns found.")
        return

    matrix = action_df[ACTION_COLS].to_numpy(dtype=float) * 100.0
    ylabels = action_df["display_name"].tolist()

    fig, ax = plt.subplots(figsize=(10.5, max(3.8, len(ylabels) * 0.72)))
    if HAS_SEABORN:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            xticklabels=ACTION_LABELS,
            yticklabels=ylabels,
            cmap="YlGnBu",
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "% of days", "shrink": 0.85},
            annot_kws={"size": 9},
        )
    else:
        im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(ACTION_LABELS)))
        ax.set_xticklabels(ACTION_LABELS, rotation=25, ha="right")
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        fig.colorbar(im, ax=ax, label="% of days")

    ax.set_title(
        f"Recommended action mix ({split})",
        pad=12,
        fontweight="semibold",
    )
    ax.set_xlabel("Action")
    ax.set_ylabel("Policy")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    _save(fig, f"action_distribution_{split}{suffix}.png")


def plot_ess_diagnostics(df: pd.DataFrame, split: str, suffix: str) -> None:
    ess_df = df.dropna(subset=["effective_sample_size"]).copy()
    if ess_df.empty:
        print("Skipping ESS figure: no effective_sample_size values available.")
        return

    fig, ax = plt.subplots(figsize=(max(7, len(ess_df) * 0.9), 4.5))
    bars = ax.bar(
        ess_df["display_name"],
        ess_df["effective_sample_size"],
        color=ess_df["color"],
        edgecolor="#333333",
        linewidth=0.5,
        width=0.6,
    )
    ax.set_ylabel("Effective sample size (episodes)")
    ax.set_title(
        f"Importance-weight diagnostics ({split})",
        pad=12,
        fontweight="semibold",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    _annotate_bars(ax, bars, fmt="{:.1f}")
    _style_axes(ax)

    _save(fig, f"ope_ess_{split}{suffix}.png")


def plot_dqn_comparison(df: pd.DataFrame, split: str, suffix: str) -> None:
    """Side-by-side DQN vs Double DQN on key metrics (when both present)."""
    rl = df[df["policy"].isin(["dqn", "double_dqn"])].copy()
    if len(rl) < 2:
        return

    metrics = ["pdis", "weighted_pdis", "dr", "match_rate"]
    labels = ["PDIS", "WPDIS", "DR", "Action match"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (_, row) in enumerate(rl.iterrows()):
        values = [row[m] if m != "match_rate" else row[m] for m in metrics]
        values = [
            v * 100 if m == "match_rate" and pd.notna(v) else v
            for m, v in zip(metrics, values)
        ]
        offset = (idx - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=row["display_name"],
            color=row["color"],
            edgecolor="#333333",
            linewidth=0.5,
        )
        _annotate_bars(ax, bars, fmt="{:.2f}" if idx == 1 else "{:.3f}")

    ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(
        f"DQN vs Double DQN ({split})",
        pad=12,
        fontweight="semibold",
    )
    ax.legend(frameon=True)
    _style_axes(ax)

    _save(fig, f"dqn_vs_double_dqn_{split}{suffix}.png")


def print_interpretation(df: pd.DataFrame, split: str) -> None:
    best_wpdis = df.loc[df["weighted_pdis"].idxmax()]
    best_dr = df.dropna(subset=["dr"])
    best_dr_row = best_dr.loc[best_dr["dr"].idxmax()] if not best_dr.empty else None

    print(f"\n=== Interpretation ({split} split) ===")
    print(
        f"Best weighted PDIS: {best_wpdis['display_name']} "
        f"({best_wpdis['weighted_pdis']:.4f})"
    )
    if best_dr_row is not None:
        print(
            f"Best DR: {best_dr_row['display_name']} "
            f"({best_dr_row['dr']:.4f})"
        )

    dqn_row = df[df["policy"] == "dqn"]
    ddqn_row = df[df["policy"] == "double_dqn"]
    if not dqn_row.empty and not ddqn_row.empty:
        d, dd = dqn_row.iloc[0], ddqn_row.iloc[0]
        print(
            f"\nDQN vs Double DQN: WPDIS {d['weighted_pdis']:.4f} vs "
            f"{dd['weighted_pdis']:.4f}; match "
            f"{d['match_rate']*100:.1f}% vs {dd['match_rate']*100:.1f}%."
        )

    bcq_row = df[df["policy"] == "bcq"]
    if not ddqn_row.empty and not bcq_row.empty:
        dd, b = ddqn_row.iloc[0], bcq_row.iloc[0]
        print(
            f"Double DQN vs BCQ: WPDIS {dd['weighted_pdis']:.4f} vs "
            f"{b['weighted_pdis']:.4f}."
        )

    print(
        "\nCaveats: inferred actions, sparse mood (~7% rows), low ESS on test. "
        "Use for exploratory ranking, not causal claims."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot DQN/Double DQN and BCQ results against baselines."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split for DQN metrics (extended comparison is test-only).",
    )
    parser.add_argument(
        "--reward-variant",
        default="reward_dense",
        help="Reward variant for DQN rows (default: reward_dense).",
    )
    parser.add_argument(
        "--all-policies",
        action="store_true",
        help="Include all baselines (random, majority, CQL, etc.). Default is a focused set.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open figures interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    suffix = "" if args.all_policies else "_focus"

    try:
        df = build_comparison_table(
            split=args.split,
            reward_variant=args.reward_variant,
            focus=not args.all_policies,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if "dqn" not in set(df["policy"]) and args.reward_variant == "reward_dense":
        print(
            "Warning: DQN not in metrics table. Run:\n"
            "  python algorithms/evaluate_dqn_models.py --best-per-algo "
            "--retrain-missing --reward-variants reward_dense --splits test",
            file=sys.stderr,
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TABLE_CSV, index=False)
    print(f"Saved merged table → {OUT_TABLE_CSV.relative_to(PROJECT_ROOT)}")

    print(f"\nGenerating figures for split={args.split} (focus={not args.all_policies}) ...")
    plot_ope_metrics(df, args.split, suffix)
    plot_ope_vs_action_match(df, args.split, suffix)
    plot_mood_improvement(df, args.split, suffix)
    plot_action_distribution(df, args.split, suffix)
    plot_ess_diagnostics(df, args.split, suffix)
    plot_dqn_comparison(df, args.split, suffix)
    print_interpretation(df, args.split)

    if args.show:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
