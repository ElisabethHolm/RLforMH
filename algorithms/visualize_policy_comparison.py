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

from _env_check import require_numpy1_for_matplotlib

require_numpy1_for_matplotlib()

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
BANDIT_CSV = MODEL_DIR / "contextual_bandit_metrics.csv"
SUBGROUP_CSV = MODEL_DIR / "dqn_subgroup_policy_analysis.csv"
OUT_TABLE_CSV = MODEL_DIR / "policy_comparison_plot_table.csv"
POSTER_TABLE_TEX = MODEL_DIR / "poster_results_table_rows.tex"

# Row order for poster.tex results block (must match extended + DQN exports).
POSTER_POLICY_ORDER = [
    "random_uniform",
    "majority_action",
    "action_frequency",
    "behavior_cloning_logistic",
    "rule_based",
    "contextual_bandit",
    "dqn",
    "double_dqn",
    "bcq",
    "cql",
]

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
    "cql",
    "dqn",
    "double_dqn",
]

# DR can explode when Q-values are poor (e.g. CQL); clip bar height only, label true value.
DR_PLOT_CLIP = (-0.5, 0.65)
DR_DISPLAY_CLIP = DR_PLOT_CLIP
OPE_IS_YLIM = (-0.06, 0.10)
SCATTER_WPDIS_MAX_WHISKER = 0.04
MOOD_CI_MAX_SPAN = 0.5
MOOD_MIN_N_FOR_CI = 3

# Label offsets (points) for scatter — avoid whisker overlap.
SCATTER_LABEL_OFFSETS = {
    "behavior_cloning_logistic": (-42, -12),
    "contextual_bandit": (10, 14),
    "bcq": (-8, -18),
    "cql": (10, -14),
    "dqn": (10, 10),
    "double_dqn": (10, -16),
}

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

CI_SUFFIXES = ("_se", "_ci_low", "_ci_high")
CI_METRICS = ("pdis", "weighted_pdis", "dr", "match_rate", "mood_improvement")

# Runtime display options (set in main from CLI).
_POSTER_STYLE = True
_FULL_CI = False


def configure_plot_display(*, poster_style: bool, full_ci: bool) -> None:
    global _POSTER_STYLE, _FULL_CI
    _POSTER_STYLE = poster_style
    _FULL_CI = full_ci


def _uncertainty_column_names() -> list[str]:
    return [f"{metric}{suffix}" for metric in CI_METRICS for suffix in CI_SUFFIXES]


def _display_ci_whiskers(
    value: float,
    lo: float | None,
    hi: float | None,
    *,
    display_value: float | None = None,
    y_clip: tuple[float, float] | None = None,
    max_span: float | None = None,
    se: float | None = None,
) -> tuple[float, float]:
    """Asymmetric whisker lengths for matplotlib (lower, upper)."""
    if not np.isfinite(value):
        return 0.0, 0.0

    dv = float(display_value) if display_value is not None else float(value)
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        disp_lo, disp_hi = float(lo), float(hi)
        if y_clip is not None:
            ymin, ymax = y_clip
            disp_lo = max(disp_lo, ymin)
            disp_hi = min(disp_hi, ymax)
        lower = max(0.0, dv - disp_lo)
        upper = max(0.0, disp_hi - dv)
    elif se is not None and np.isfinite(se):
        lower = upper = float(se)
    else:
        return 0.0, 0.0

    if max_span is not None and max_span > 0 and not _FULL_CI:
        half = max_span / 2.0
        lower = min(lower, half)
        upper = min(upper, half)

    return lower, upper


def _whiskers_from_row(
    row: pd.Series,
    metric: str,
    *,
    display_value: float | None = None,
    y_clip: tuple[float, float] | None = None,
    max_span: float | None = None,
    scale: float = 1.0,
) -> tuple[float, float]:
    """Whisker lengths for one metric on one policy row."""
    value = row.get(metric)
    if not np.isfinite(value):
        return 0.0, 0.0
    lo = row.get(f"{metric}_ci_low")
    hi = row.get(f"{metric}_ci_high")
    se = row.get(f"{metric}_se")
    value = float(value) * scale
    dv = (
        float(display_value) * scale
        if display_value is not None and np.isfinite(display_value)
        else None
    )
    if pd.notna(lo):
        lo = float(lo) * scale
    if pd.notna(hi):
        hi = float(hi) * scale
    if pd.notna(se):
        se = float(se) * scale
    return _display_ci_whiskers(
        float(value),
        lo if pd.notna(lo) else None,
        hi if pd.notna(hi) else None,
        display_value=dv,
        y_clip=y_clip,
        max_span=max_span,
        se=se if pd.notna(se) else None,
    )


def _yerr_from_dataframe(
    df: pd.DataFrame,
    metric: str,
    *,
    values: np.ndarray | None = None,
    display_values: np.ndarray | None = None,
    y_clip: tuple[float, float] | None = None,
    max_span: float | None = None,
    scale: float = 1.0,
) -> np.ndarray | None:
    """Asymmetric 95% CI bars for matplotlib (shape 2 x n)."""
    low_col = f"{metric}_ci_low"
    high_col = f"{metric}_ci_high"
    if low_col not in df.columns and f"{metric}_se" not in df.columns:
        return None

    lower, upper = [], []
    has_error = False
    for i, (_, row) in enumerate(df.iterrows()):
        v = values[i] if values is not None else row.get(metric)
        dv = (
            display_values[i]
            if display_values is not None and i < len(display_values)
            else None
        )
        lo, hi = _whiskers_from_row(
            row,
            metric,
            display_value=dv,
            y_clip=y_clip,
            max_span=max_span,
            scale=scale,
        )
        lower.append(lo)
        upper.append(hi)
        has_error = has_error or (lo > 0 or hi > 0)
    return np.array([lower, upper]) if has_error else None


def _errorbar_kwargs() -> dict:
    """Keyword args for ax.errorbar (not ax.bar)."""
    return {"capsize": 3, "elinewidth": 1.0, "ecolor": "#333333", "capthick": 1.0}


def _bar_yerr_params(yerr: np.ndarray | None) -> dict:
    """capsize + error_kw for ax.bar when yerr is set."""
    if yerr is None:
        return {}
    return {
        "yerr": yerr,
        "capsize": 3,
        "error_kw": {"elinewidth": 1.0, "ecolor": "#333333", "capthick": 1.0},
    }


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
    for suffix in CI_SUFFIXES:
        am_key = f"action_match{suffix}"
        mr_key = f"match_rate{suffix}"
        if am_key in rows.columns:
            rows[mr_key] = rows[am_key]
        elif mr_key in rows.columns:
            rows[am_key] = rows[mr_key]
    if "mood_n_matched" in rows.columns:
        rows["n_matched"] = rows["mood_n_matched"]
    else:
        rows["n_matched"] = np.nan
    rows["dr_estimator"] = "sequential"
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
        "dr_estimator",
        "mean_behavior_support",
        "pct_low_support",
        "effective_sample_size",
        "source",
        *_uncertainty_column_names(),
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


def _enrich_bandit_metrics(df: pd.DataFrame, reward_variant: str, split: str) -> pd.DataFrame:
    """Attach one-step bandit DR when sequential DR is unavailable."""
    if not BANDIT_CSV.exists() or "contextual_bandit" not in set(df["policy"]):
        return df

    bandit_df = pd.read_csv(BANDIT_CSV)
    split_name = {"val": "validation"}.get(split, split)
    mask = (bandit_df["reward_variant"] == reward_variant) & (
        bandit_df["split"] == split_name
    )
    if not mask.any():
        return df

    row = bandit_df.loc[mask].iloc[0]
    idx = df["policy"] == "contextual_bandit"
    if not idx.any():
        return df

    df.loc[idx, "dr"] = float(row["dr_reward"])
    df.loc[idx, "dr_estimator"] = "bandit_1step"
    mood_n = int(row["matched_mood_n"]) if "matched_mood_n" in row.index else 0
    df.loc[idx, "n_matched"] = mood_n
    if mood_n > 0 and pd.notna(row.get("matched_mood_improvement")):
        df.loc[idx, "mood_improvement"] = float(row["matched_mood_improvement"])
    return df


def _mood_count(row: pd.Series) -> int | None:
    n = row.get("n_matched")
    if pd.isna(n):
        return None
    return int(n)


def fmt_latex_signed(value: float, decimals: int = 4) -> str:
    return f"${value:+.{decimals}f}$"


def fmt_latex_signed_ci(row: pd.Series, metric: str, decimals: int = 4) -> str:
    value = row.get(metric)
    if pd.isna(value):
        return "---"
    lo, hi = row.get(f"{metric}_ci_low"), row.get(f"{metric}_ci_high")
    if pd.notna(lo) and pd.notna(hi):
        err = max(abs(float(value) - float(lo)), abs(float(hi) - float(value)))
        return f"${float(value):+.{decimals}f} \\pm {err:.{decimals}f}$"
    return fmt_latex_signed(float(value), decimals=decimals)


def fmt_latex_dr(row: pd.Series) -> str:
    dr = row.get("dr")
    if pd.isna(dr):
        return "---"
    est = row.get("dr_estimator", "")
    tag = r"^\dagger" if est == "bandit_1step" else ""
    return f"${dr:+.4f}{tag}$"


def fmt_latex_mood(row: pd.Series) -> str:
    n = _mood_count(row)
    mood = row.get("mood_improvement")
    if n == 0:
        return r"$\mathrm{NA}\,(0)$"
    if pd.notna(mood) and n is not None and n > 0:
        lo, hi = row.get("mood_improvement_ci_low"), row.get("mood_improvement_ci_high")
        if pd.notna(lo) and pd.notna(hi):
            err = max(abs(float(mood) - float(lo)), abs(float(hi) - float(mood)))
            return f"${float(mood):+.2f} \\pm {err:.2f}\,({n})$"
        return f"${mood:+.2f}\,({n})$"
    if pd.notna(mood):
        return fmt_latex_signed(mood, decimals=2)
    return "---"


def export_poster_table_tex(df: pd.DataFrame, path: Path = POSTER_TABLE_TEX) -> Path:
    """Write LaTeX table rows for poster.tex from the merged metrics table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    alt = False
    for policy in POSTER_POLICY_ORDER:
        sub = df[df["policy"] == policy]
        if sub.empty:
            continue
        row = sub.iloc[0]
        name = row["display_name"]
        if policy in ("dqn", "double_dqn", "bcq", "cql"):
            name = f"\\textbf{{{name}}}"
        pdis = fmt_latex_signed_ci(row, "pdis")
        wpdis = fmt_latex_signed_ci(row, "weighted_pdis")
        dr = fmt_latex_dr(row)
        mood = fmt_latex_mood(row)
        match_val = float(row["match_rate"]) * 100
        if pd.notna(row.get("match_rate_ci_low")) and pd.notna(row.get("match_rate_ci_high")):
            lo = float(row["match_rate_ci_low"]) * 100
            hi = float(row["match_rate_ci_high"]) * 100
            err = max(abs(match_val - lo), abs(hi - match_val))
            match = f"{match_val:.1f} $\\pm$ {err:.1f}"
        else:
            match = f"{match_val:.1f}"
        prefix = "\\rowcolor{lightgray}\n" if alt else ""
        alt = not alt
        lines.append(
            f"{prefix}{name:<20} & {pdis} & {wpdis} & {dr} & {mood} & {match} \\\\"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved LaTeX table rows → {path.relative_to(PROJECT_ROOT)}")
    return path


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
        "dr_estimator",
        "mean_behavior_support",
        "pct_low_support",
        "effective_sample_size",
        "source",
        *ACTION_COLS,
        *_uncertainty_column_names(),
    ]
    extended = extended.reindex(columns=keep_cols)
    if "dr_estimator" not in extended.columns:
        extended["dr_estimator"] = np.where(
            extended["dr"].notna(), "sequential", "none"
        )

    merged = pd.concat([extended, dqn], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=["policy"], keep="last")
    merged = _enrich_bandit_metrics(merged, reward_variant=reward_variant, split=split)

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
    label_values: np.ndarray | None = None,
) -> None:
    """Place value labels on tall bars; above bar otherwise."""
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo if y_hi > y_lo else 1.0
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height is None or not np.isfinite(height):
            continue
        display_val = (
            label_values[i]
            if label_values is not None and i < len(label_values) and np.isfinite(label_values[i])
            else height
        )
        label = fmt.format(display_val)
        if label_values is not None and np.isfinite(label_values[i]) and label_values[i] != height:
            label = f"{label}†"
        height_for_layout = height
        if height_for_layout >= inside_min:
            y_text = height_for_layout * 0.55
            va = "center"
            color = "white"
            weight = "bold"
            size = 7.5
        elif height_for_layout <= -0.001:
            y_text = height_for_layout - 0.008 * span
            va = "top"
            color = "#333333"
            weight = "normal"
            size = 7.5
        else:
            y_text = height_for_layout + 0.012 * span
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


def _plot_metric_bars_on_ax(
    ax,
    df: pd.DataFrame,
    x: np.ndarray,
    metrics: list[tuple[str, str, str]],
    *,
    width: float = 0.36,
    y_clip: tuple[float, float] | None = None,
    annotate: bool = False,
    dr_clip_bounds: tuple[float, float] | None = None,
) -> bool:
    """Draw grouped bars for metrics list [(col, legend, color), ...]. Returns dr_clipped."""
    dr_clipped = False
    n_metrics = len(metrics)
    for idx, (col, label, color) in enumerate(metrics):
        raw = df[col].to_numpy(dtype=float)
        if col == "dr" and dr_clip_bounds is not None:
            plot_vals = np.array(
                [
                    np.clip(v, *dr_clip_bounds) if np.isfinite(v) else np.nan
                    for v in raw
                ],
                dtype=float,
            )
            dr_clipped |= any(
                np.isfinite(v) and (v < dr_clip_bounds[0] or v > dr_clip_bounds[1])
                for v in raw
            )
            clip_for_err = dr_clip_bounds
            disp = plot_vals
        else:
            plot_vals = np.where(np.isfinite(raw), raw, np.nan)
            clip_for_err = y_clip
            disp = plot_vals

        offset = (idx - (n_metrics - 1) / 2.0) * width
        yerr = _yerr_from_dataframe(
            df,
            col,
            values=raw,
            display_values=disp,
            y_clip=clip_for_err,
        )
        bars = ax.bar(
            x + offset,
            plot_vals,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
            **_bar_yerr_params(yerr),
        )
        if annotate:
            _annotate_bars(ax, bars, label_values=raw if col == "dr" else plot_vals)
    return dr_clipped


def plot_ope_metrics(df: pd.DataFrame, split: str, suffix: str) -> None:
    """Grouped bar chart comparing offline value estimators across policies."""
    if _POSTER_STYLE:
        _plot_ope_metrics_poster(df, split, suffix)
    else:
        _plot_ope_metrics_combined(df, split, suffix)


def _plot_ope_metrics_poster(df: pd.DataFrame, split: str, suffix: str) -> None:
    policies = df["display_name"].tolist()
    n = len(policies)
    x = np.arange(n)
    split_label = "Test" if split == "test" else split.title()

    fig, (ax_is, ax_dr) = plt.subplots(
        2,
        1,
        figsize=(max(10, n * 1.2), 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.08},
    )

    is_metrics = [
        ("pdis", "PDIS", OPE_COLORS[0]),
        ("weighted_pdis", "Weighted PDIS", OPE_COLORS[1]),
    ]
    _plot_metric_bars_on_ax(
        ax_is,
        df,
        x,
        is_metrics,
        width=0.36,
        y_clip=OPE_IS_YLIM,
        annotate=False,
    )
    ax_is.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)
    ax_is.set_ylim(OPE_IS_YLIM)
    ax_is.set_ylabel("PDIS / WPDIS", fontsize=11, labelpad=8)
    ax_is.legend(loc="upper right", frameon=False, fontsize=9)
    _style_axes(ax_is)

    dr_clipped = _plot_metric_bars_on_ax(
        ax_dr,
        df,
        x,
        [("dr", "Doubly robust", OPE_COLORS[2])],
        width=0.42,
        dr_clip_bounds=DR_DISPLAY_CLIP,
        annotate=True,
    )
    ax_dr.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)
    pad = 0.05
    ax_dr.set_ylim(DR_DISPLAY_CLIP[0] - pad, DR_DISPLAY_CLIP[1] + pad)
    ax_dr.set_ylabel("Doubly robust", fontsize=11, labelpad=8)
    _style_axes(ax_dr)

    ax_dr.set_xticks(x)
    ax_dr.set_xticklabels(policies, rotation=22, ha="right", fontsize=10)
    ax_dr.set_xlabel("Policy method", labelpad=10, fontsize=11)

    footnotes = [
        "Error bars: 95% episode-bootstrap CI. DR panel clipped to "
        f"y in [{DR_DISPLAY_CLIP[0]:.1f}, {DR_DISPLAY_CLIP[1]:.2f}]; full CIs in results table."
    ]
    if dr_clipped:
        footnotes.insert(0, "† DR bar clipped; label shows true value.")
    fig.text(
        0.11,
        0.01,
        " ".join(footnotes),
        fontsize=8,
        color="#666666",
    )

    fig.suptitle(
        f"Hold-out {split_label}: Offline Policy Value (Dense Reward)",
        fontsize=12,
        fontweight="semibold",
        x=0.11,
        ha="left",
        y=0.98,
    )
    fig.subplots_adjust(bottom=0.22, left=0.11, right=0.98, top=0.92)

    _save(fig, f"ope_metrics_{split}{suffix}.png")


def _plot_ope_metrics_combined(df: pd.DataFrame, split: str, suffix: str) -> None:
    """Single-panel layout for appendix / --no-poster-style runs."""
    policies = df["display_name"].tolist()
    n = len(policies)
    x = np.arange(n)
    width = 0.17
    split_label = "Test" if split == "test" else split.title()

    fig, ax = plt.subplots(figsize=(max(10, n * 1.15), 5.8))
    dr_clipped = _plot_metric_bars_on_ax(
        ax,
        df,
        x,
        list(zip(
            [m[0] for m in OPE_METRICS],
            [m[1] for m in OPE_METRICS],
            OPE_COLORS,
        )),
        width=width,
        dr_clip_bounds=DR_DISPLAY_CLIP,
        annotate=not _POSTER_STYLE,
    )
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)

    if _FULL_CI:
        y_min, y_max = -0.5, 0.7
        for col, _ in OPE_METRICS:
            yerr = _yerr_from_dataframe(df, col)
            if yerr is not None:
                vals = df[col].to_numpy(dtype=float)
                y_max = max(y_max, float(np.nanmax(vals + yerr[1])))
                y_min = min(y_min, float(np.nanmin(vals - yerr[0])))
        pad = max(0.012, (y_max - y_min) * 0.15)
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        ax.set_ylim(-0.08, 0.72)

    footnotes = []
    if dr_clipped:
        footnotes.append("† DR clipped for display.")
    if footnotes:
        ax.text(0.01, 0.01, " ".join(footnotes), transform=ax.transAxes, fontsize=8, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right", fontsize=10)
    ax.set_xlabel("Policy method", labelpad=10, fontsize=11)
    ax.set_ylabel("Offline value estimate\n(dense wellness reward)", labelpad=10, fontsize=11)
    ax.set_title(
        f"Hold-out {split_label}: Offline Policy Value (Dense Reward)",
        pad=12,
        fontsize=12,
        fontweight="semibold",
        loc="left",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3, frameon=False, fontsize=10)
    _style_axes(ax)
    fig.subplots_adjust(bottom=0.34, left=0.11, right=0.98, top=0.90)
    _save(fig, f"ope_metrics_{split}{suffix}.png")


def plot_ope_vs_action_match(df: pd.DataFrame, split: str, suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    wpdis_cap = None if _FULL_CI else SCATTER_WPDIS_MAX_WHISKER

    for _, row in df.iterrows():
        x = row["match_rate"] * 100
        y = row["weighted_pdis"]
        if pd.isna(x) or pd.isna(y):
            continue
        color = row["color"]
        policy = row.get("policy", "")
        x_lo, x_hi = _whiskers_from_row(row, "match_rate", scale=100.0)
        y_lo, y_hi = _whiskers_from_row(
            row,
            "weighted_pdis",
            max_span=wpdis_cap,
        )
        xerr = np.array([[x_lo], [x_hi]]) if (x_lo > 0 or x_hi > 0) else None
        yerr = np.array([[y_lo], [y_hi]]) if (y_lo > 0 or y_hi > 0) else None

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=9,
            mfc=color,
            mec="#333333",
            mew=0.6,
            zorder=3,
            **_errorbar_kwargs(),
        )
        offset = SCATTER_LABEL_OFFSETS.get(policy, (10, 6))
        ax.annotate(
            row["display_name"],
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=9,
            color="#222222",
        )

    ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.set_xlim(-5, 102)
    if not _FULL_CI:
        y_vals = df["weighted_pdis"].dropna()
        if not y_vals.empty:
            y_mid = float(y_vals.median())
            ax.set_ylim(y_mid - 0.06, y_mid + 0.08)
    ax.set_xlabel("Action match with logged behavior (%)")
    ax.set_ylabel("Weighted PDIS")
    ax.set_title(
        f"Support overlap vs. estimated value ({split})",
        pad=12,
        fontweight="semibold",
    )
    if _POSTER_STYLE and not _FULL_CI:
        ax.text(
            0.01,
            0.02,
            "Vertical whiskers capped for display; full WPDIS CIs in table.",
            transform=ax.transAxes,
            fontsize=8,
            color="#666666",
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
    mood_yerr_rows = []
    for _, row in plot_df.iterrows():
        n = row.get("n_matched")
        show_ci = pd.notna(n) and int(n) >= MOOD_MIN_N_FOR_CI
        if show_ci and not _FULL_CI:
            lo, hi = _whiskers_from_row(
                row,
                "mood_improvement",
                max_span=MOOD_CI_MAX_SPAN,
            )
        elif show_ci:
            lo, hi = _whiskers_from_row(row, "mood_improvement")
        else:
            lo, hi = 0.0, 0.0
        mood_yerr_rows.append((lo, hi))
    if any(lo > 0 or hi > 0 for lo, hi in mood_yerr_rows):
        mood_yerr = np.array(
            [[r[0] for r in mood_yerr_rows], [r[1] for r in mood_yerr_rows]]
        )
    else:
        mood_yerr = None

    bars = ax.bar(
        plot_df["display_name"],
        plot_df["mood_plot"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
        width=0.65,
        zorder=3,
        **_bar_yerr_params(mood_yerr),
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

    note = "Only timesteps with observed mood and matching logged action."
    if mood_yerr is not None:
        note += (
            f" Error bars: 95% CI when n≥{MOOD_MIN_N_FOR_CI} matched steps."
        )
    ax.text(
        0.01,
        0.02,
        note,
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
        offset = (idx - 0.5) * width
        metric_vals = np.array(
            [row[m] if m != "match_rate" else row[m] for m in metrics],
            dtype=float,
        )
        plot_vals = np.array(
            [
                v * 100 if m == "match_rate" and pd.notna(v) else v
                for m, v in zip(metrics, metric_vals)
            ],
            dtype=float,
        )
        err_low, err_high = [], []
        for m, pv in zip(metrics, plot_vals):
            raw_v = row[m]
            if m == "dr" and pd.notna(raw_v):
                disp = (
                    np.clip(float(raw_v), *DR_DISPLAY_CLIP)
                    if np.isfinite(raw_v)
                    else raw_v
                )
                y_clip = DR_DISPLAY_CLIP
                max_span = None
            elif m == "match_rate":
                disp = pv
                y_clip = None
                max_span = None
            else:
                disp = pv
                y_clip = OPE_IS_YLIM if m in ("pdis", "weighted_pdis") else None
                max_span = None
            lo, hi = _whiskers_from_row(
                row,
                m,
                display_value=disp if m == "dr" else None,
                y_clip=y_clip,
                max_span=max_span,
                scale=100.0 if m == "match_rate" else 1.0,
            )
            err_low.append(lo)
            err_high.append(hi)
        yerr = (
            np.array([err_low, err_high])
            if any(err_low) or any(err_high)
            else None
        )

        bars = ax.bar(
            x + offset,
            plot_vals,
            width,
            label=row["display_name"],
            color=row["color"],
            edgecolor="#333333",
            linewidth=0.5,
            **_bar_yerr_params(yerr),
        )
        if not _POSTER_STYLE:
            _annotate_bars(ax, bars, fmt="{:.2f}" if idx == 1 else "{:.3f}")

    ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if _POSTER_STYLE and not _FULL_CI:
        ax.set_ylim(-0.08, 0.75)
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
    parser.add_argument(
        "--export-latex",
        action="store_true",
        help="Write models/poster_results_table_rows.tex for poster.tex.",
    )
    parser.add_argument(
        "--poster-style",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Split OPE panels, cap display whiskers, reduce label clutter (default: on for _focus).",
    )
    parser.add_argument(
        "--full-ci",
        action="store_true",
        help="Show full bootstrap CI whiskers without display caps (debug/appendix).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    suffix = "" if args.all_policies else "_focus"
    poster_style = (
        args.poster_style if args.poster_style is not None else suffix == "_focus"
    )
    configure_plot_display(poster_style=poster_style, full_ci=args.full_ci)

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

    if args.export_latex:
        full_df = build_comparison_table(
            split=args.split,
            reward_variant=args.reward_variant,
            focus=False,
        )
        export_poster_table_tex(full_df)

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
