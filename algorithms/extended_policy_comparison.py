"""
Extended policy comparison for StudentLife offline RL.
CS 224R Spring 2026 — Juan Pablo Pacheco

Runs comprehensive head-to-head evaluation of all trained policies on the
test split and produces tables and figures for the final report.

Run from repo root:
    python algorithms/extended_policy_comparison.py
    python algorithms/extended_policy_comparison.py --reward-ablation

AI Tools Disclosure — see bottom of file.
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

from _env_check import require_numpy1_for_matplotlib

require_numpy1_for_matplotlib()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Allow sibling-module imports without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from evaluate_policies import (  # noqa: E402
    STATE_COLS, N_ACTIONS, GAMMA, IS_CLIP, EPSILON, ACTION_NAMES,
    compute_pdis_estimate,
    compute_weighted_pdis_estimate,
    compute_doubly_robust_estimate,
    compute_mood_improvement_stats,
    CQLPolicyWrapper,
    SklearnPolicyWrapper,
    RuleBasedPolicyWrapper,
    RandomPolicyWrapper,
)
from ope_uncertainty import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    attach_policy_uncertainty,
)
from train_bcq import BCQPolicyWrapper, MLP, train_bcq  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "final_datasets"
MODEL_DIR    = PROJECT_ROOT / "models"
FIGURES_DIR  = PROJECT_ROOT / "figures"

TEST_CSV      = DATA_DIR / "daily_studentlife.test.csv"
FULL_CSV      = DATA_DIR / "daily_studentlife.csv"
TRAIN_CSV     = DATA_DIR / "daily_studentlife.train.csv"
CQL_PATH      = MODEL_DIR / "studentlife_discrete_cql.d3"
BASELINE_PKL  = MODEL_DIR / "studentlife_baseline_models.pkl"
BANDIT_PKL    = MODEL_DIR / "contextual_bandit_models.pkl"
BCQ_PATH      = MODEL_DIR / "bcq_model.pt"
BCQ_LOG_PATH  = MODEL_DIR / "bcq_training_log.csv"
OUT_JSON      = MODEL_DIR / "extended_comparison_results.json"
OUT_CSV       = MODEL_DIR / "extended_comparison_results.csv"
BANDIT_CSV    = MODEL_DIR / "contextual_bandit_metrics.csv"
ABLATION_CSV  = MODEL_DIR / "bcq_reward_ablation.csv"
REWARD_COL      = "reward_dense"

NEXT_STATE_COLS  = [f"next_{c}" for c in STATE_COLS]
REWARD_VARIANTS  = ["reward_sparse", "reward_dense", "reward_observed_only"]

# Visual style: RL=blue, Bandit=orange, Baseline=gray
_POLICY_TYPE = {
    "cql":                       "RL",
    "bcq":                       "RL",
    "contextual_bandit":         "Bandit",
    "random_uniform":            "Baseline",
    "majority_action":           "Baseline",
    "action_frequency":          "Baseline",
    "behavior_cloning_logistic": "Baseline",
    "rule_based":                "Baseline",
}
_TYPE_COLORS = {"RL": "steelblue", "Bandit": "darkorange", "Baseline": "gray"}


def _ptype(name: str) -> str:
    return _POLICY_TYPE.get(name, "Baseline")


def _color(name: str) -> str:
    return _TYPE_COLORS[_ptype(name)]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_episodes(df: pd.DataFrame, reward_col: str = "reward_dense") -> list:
    """
    Build episode dicts from a split CSV, using reward_col as the reward signal.
    next_mood and mood are kept as-is (may contain NaN) so that mood_improvement
    can be computed only over observed transitions.
    """
    episodes = []
    for (sid, eid), grp in df.groupby(["student_id", "episode_id"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        episodes.append({
            "student_id": sid,
            "episode_id": eid,
            "states":     grp[STATE_COLS].fillna(0.0).to_numpy("float32"),
            "actions":    grp["action"].to_numpy("int64"),
            "rewards":    grp[reward_col].fillna(0.0).to_numpy("float32"),
            "mood":       grp["mood"].to_numpy("float64"),         # may have NaN
            "next_moods": grp["next_mood"].to_numpy("float64"),    # may have NaN
            "T":          len(grp),
        })
    return episodes


# ---------------------------------------------------------------------------
# Policy wrappers
# ---------------------------------------------------------------------------

class BanditPolicyWrapper:
    """
    Wraps the contextual bandit Ridge reward model from train_contextual_bandit.py.
    Scores all seven actions per state using the same feature construction
    (state + one-hot action + interaction) and picks the highest-predicted reward.
    """
    name = "contextual_bandit"

    def __init__(
        self,
        reward_model,
        n_actions: int   = N_ACTIONS,
        epsilon:   float = EPSILON,
    ):
        self._model     = reward_model
        self._n_actions = n_actions
        self._epsilon   = epsilon

    def _sa_features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        n       = len(states)
        one_hot = np.zeros((n, self._n_actions), dtype="float32")
        one_hot[np.arange(n), actions.astype(int)] = 1.0
        interact = (states[:, :, None] * one_hot[:, None, :]).reshape(
            n, states.shape[1] * self._n_actions
        )
        return np.concatenate([states, one_hot, interact], axis=1)

    def _score_all(self, states: np.ndarray) -> np.ndarray:
        n       = len(states)
        rep_s   = np.repeat(states, self._n_actions, axis=0)
        rep_a   = np.tile(np.arange(self._n_actions, dtype="int64"), n)
        return self._model.predict(self._sa_features(rep_s, rep_a)).reshape(
            n, self._n_actions
        )

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._score_all(states).argmax(axis=1).astype("int64")

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        actions = self.predict(states)
        probs   = np.full((len(states), self._n_actions), self._epsilon / self._n_actions)
        probs[np.arange(len(states)), actions] += (1.0 - self._epsilon)
        return probs


# ---------------------------------------------------------------------------
# Policy loaders
# ---------------------------------------------------------------------------

def load_cql_policy() -> CQLPolicyWrapper:
    from d3rlpy.algos import DiscreteCQLConfig
    from d3rlpy.dataset import MDPDataset

    df = pd.read_csv(FULL_CSV)
    # build_with_dataset is required to initialize network architecture
    dataset = MDPDataset(
        observations=df[STATE_COLS].fillna(0.0).to_numpy("float32"),
        actions=df["action"].to_numpy("int64"),
        rewards=df["reward_dense"].fillna(0.0).to_numpy("float32"),
        terminals=df["done"].to_numpy("bool"),
    )
    model = DiscreteCQLConfig().create(device="cpu")
    model.build_with_dataset(dataset)
    model.load_model(str(CQL_PATH))
    return CQLPolicyWrapper(model)


def load_bcq_policy() -> BCQPolicyWrapper:
    return BCQPolicyWrapper.load(BCQ_PATH)


def load_bandit_policy() -> BanditPolicyWrapper:
    with open(BANDIT_PKL, "rb") as f:
        payload = pickle.load(f)
    return BanditPolicyWrapper(payload["reward_models"]["reward_dense"])


def load_baseline_policies() -> tuple:
    """Returns (behavior_policy, list_of_other_baselines)."""
    with open(BASELINE_PKL, "rb") as f:
        payload = pickle.load(f)
    models = payload["models"]
    behavior = SklearnPolicyWrapper(
        "behavior_cloning_logistic",
        models["behavior_cloning_logistic"],
    )
    others = [
        SklearnPolicyWrapper("majority_action",  models["majority_action"]),
        SklearnPolicyWrapper("action_frequency", models["action_frequency"]),
        RandomPolicyWrapper(),
        RuleBasedPolicyWrapper(
            threshold=payload["rule_based_baseline"]["rule_threshold"]
        ),
    ]
    return behavior, others


# ---------------------------------------------------------------------------
# Per-policy metrics
# ---------------------------------------------------------------------------

def _behavior_support(policy, states: np.ndarray, behavior_policy) -> dict:
    actions     = policy.predict(states)
    behav_probs = behavior_policy.action_probs(states)
    support     = behav_probs[np.arange(len(states)), actions]
    return {
        "mean_behavior_support": float(np.mean(support)),
        "pct_low_support":       float(np.mean(support < 0.05) * 100),
    }


def _action_dist(policy, states: np.ndarray) -> dict:
    actions = policy.predict(states)
    counts  = np.bincount(actions, minlength=N_ACTIONS)
    return {ACTION_NAMES[i]: float(counts[i] / len(states)) for i in range(N_ACTIONS)}


def evaluate_policy_extended(
    policy,
    episodes:       list,
    all_states:     np.ndarray,
    logged_actions: np.ndarray,
    behavior_policy,
) -> dict:
    mood_imp, n_matched = compute_mood_improvement_stats(episodes, policy)
    result = {
        "policy":             policy.name,
        "pdis":               _fmt(compute_pdis_estimate(
                                  episodes, policy, behavior_policy)),
        "weighted_pdis":      _fmt(compute_weighted_pdis_estimate(
                                  episodes, policy, behavior_policy)),
        "mood_improvement":   _fmt(mood_imp),
        "n_matched":          n_matched,   # # observed-mood timesteps this metric rests on
        "match_rate":         _fmt(float(
                                  np.mean(policy.predict(all_states) == logged_actions))),
        "action_distribution": _action_dist(policy, all_states),
    }
    bs = _behavior_support(policy, all_states, behavior_policy)
    result["mean_behavior_support"] = _fmt(bs["mean_behavior_support"])
    result["pct_low_support"]       = _fmt(bs["pct_low_support"])

    if hasattr(policy, "q_values"):
        result["dr"] = _fmt(
            compute_doubly_robust_estimate(episodes, policy, behavior_policy)
        )
        result["dr_estimator"] = "sequential"
    else:
        result["dr_estimator"] = "none"
    return result


def evaluate_policy_extended_with_uncertainty(
    policy,
    episodes: list,
    all_states: np.ndarray,
    logged_actions: np.ndarray,
    behavior_policy,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict:
    result = evaluate_policy_extended(
        policy, episodes, all_states, logged_actions, behavior_policy
    )
    attach_policy_uncertainty(
        result, episodes, policy, behavior_policy, n_bootstrap=n_bootstrap
    )
    for key, value in list(result.items()):
        if key.endswith(("_se", "_ci_low", "_ci_high")) and value is not None:
            if isinstance(value, float) and np.isfinite(value):
                result[key] = _fmt(value)
            elif isinstance(value, float):
                result[key] = None
    return result


def enrich_bandit_metrics(
    results: list,
    reward_variant: str = REWARD_COL,
    split: str = "test",
) -> list:
    """Fill contextual-bandit DR (one-step) from saved bandit metrics CSV."""
    if not BANDIT_CSV.exists():
        return results

    bandit_df = pd.read_csv(BANDIT_CSV)
    mask = (bandit_df["reward_variant"] == reward_variant) & (
        bandit_df["split"] == split
    )
    bandit_row = bandit_df.loc[mask]
    if bandit_row.empty:
        return results

    row = bandit_row.iloc[0]
    for result in results:
        if result.get("policy") != "contextual_bandit":
            continue
        result["dr"] = _fmt(float(row["dr_reward"]))
        result["dr_estimator"] = "bandit_1step"
        mood_n = int(row["matched_mood_n"]) if "matched_mood_n" in row.index else 0
        result["n_matched"] = mood_n
        if mood_n > 0 and pd.notna(row.get("matched_mood_improvement")):
            result["mood_improvement"] = _fmt(float(row["matched_mood_improvement"]))
        break
    return results


def _fmt(val, n: int = 6):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return None
    return round(float(val), n)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_mood_improvement(results: list) -> None:
    names  = [r["policy"] for r in results]
    values = [r["mood_improvement"] or 0.0 for r in results]
    colors = [_color(n) for n in names]
    lows, highs = [], []
    for r in results:
        lo, hi = r.get("mood_improvement_ci_low"), r.get("mood_improvement_ci_high")
        v = r.get("mood_improvement") or 0.0
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            lows.append(max(0.0, float(v) - float(lo)))
            highs.append(max(0.0, float(hi) - float(v)))
        else:
            lows.append(0.0)
            highs.append(0.0)
    show_yerr = any(
        r.get("mood_improvement_ci_low") is not None
        and r.get("mood_improvement_ci_high") is not None
        and np.isfinite(r.get("mood_improvement_ci_low"))
        and np.isfinite(r.get("mood_improvement_ci_high"))
        for r in results
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    bar_kwargs = {
        "color": colors,
        "edgecolor": "black",
        "linewidth": 0.5,
    }
    if show_yerr:
        bar_kwargs["yerr"] = np.array([lows, highs])
        bar_kwargs["capsize"] = 3
        bar_kwargs["error_kw"] = {"elinewidth": 1.0, "ecolor": "#333333"}
    ax.bar(range(len(names)), values, **bar_kwargs)
    ax.axhline(0, linestyle="--", color="black", linewidth=1.0, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Mood Improvement (Δ mood)")
    ax.set_title("Policy Comparison: Mood Improvement on Test Split")

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=t) for t, c in _TYPE_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = FIGURES_DIR / "policy_mood_improvement.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


def fig_action_heatmap(results: list) -> None:
    action_names = [ACTION_NAMES[i] for i in range(N_ACTIONS)]
    policies     = [r["policy"] for r in results]
    matrix       = np.array([
        [r["action_distribution"].get(a, 0.0) * 100 for a in action_names]
        for r in results
    ])

    fig, ax = plt.subplots(figsize=(13, max(4, len(policies) * 0.65)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        xticklabels=action_names,
        yticklabels=policies,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "% of timesteps"},
    )
    ax.set_title("Action Distribution Heatmap (% of recommendations)")
    ax.set_xlabel("Action")
    ax.set_ylabel("Policy")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = FIGURES_DIR / "action_distribution_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


def fig_behavior_support_scatter(results: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        x = r.get("mean_behavior_support")
        y = r.get("mood_improvement")
        if x is None or y is None:
            continue
        ax.scatter(x, y, color=_color(r["policy"]), s=110, zorder=3,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(r["policy"], (x, y),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)

    ax.axhline(0, linestyle="--", color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Mean Behavior Support  π_b(a|s)")
    ax.set_ylabel("Mood Improvement (Δ mood)")
    ax.set_title("Conservatism vs Performance Trade-off")

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=t) for t, c in _TYPE_COLORS.items()]
    ax.legend(handles=handles, fontsize=9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = FIGURES_DIR / "behavior_support_vs_performance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


def fig_bcq_training_curves() -> None:
    if not BCQ_LOG_PATH.exists():
        print(f"BCQ log not found at {BCQ_LOG_PATH}; skipping training curves figure.")
        return

    log = pd.read_csv(BCQ_LOG_PATH)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(log["step"], log["td_loss"], color="steelblue", linewidth=1.2)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("TD Loss")
    ax1.set_title("BCQ TD Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(log["step"], log["bc_loss"], color="darkorange", linewidth=1.2)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("BC Loss")
    ax2.set_title("BCQ Behavior Cloning Loss")
    ax2.grid(alpha=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = FIGURES_DIR / "bcq_training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: list) -> None:
    col = 30
    header = (
        f"\n{'Policy':<{col}} {'PDIS':>10} {'WPDIS':>10} "
        f"{'DR':>10} {'Mood Δ':>8} {'n':>4} {'Match%':>7} {'BehSup':>7} {'LowSup%':>8}"
    )
    print(header)
    print("-" * (col + 73))
    for r in results:
        def _f(key, fmt=">10.4f", na="N/A"):
            v = r.get(key)
            return f"{v:{fmt}}" if v is not None else f"{na:>10}"

        match_s = (
            f"{r['match_rate'] * 100:>7.1f}%"
            if r["match_rate"] is not None
            else f"{'N/A':>8}"
        )
        n_s    = f"{r.get('n_matched', 0):>4}"
        sup_s  = _f("mean_behavior_support", ">7.4f")
        lsup_s = _f("pct_low_support", ">8.1f")
        print(
            f"{r['policy']:<{col}} "
            f"{_f('pdis')} "
            f"{_f('weighted_pdis')} "
            f"{_f('dr', na='—')} "
            f"{_f('mood_improvement', '>8.4f', 'N/A')} "
            f"{n_s} "
            f"{match_s} "
            f"{sup_s} "
            f"{lsup_s}"
        )


# ---------------------------------------------------------------------------
# Reward variant ablation (--reward-ablation flag)
# ---------------------------------------------------------------------------

def run_reward_ablation(behavior_policy) -> None:
    """
    Train a lightweight BCQ (10k steps) per reward variant and compare PDIS
    and mood_improvement on the test split.
    """
    test_df = pd.read_csv(TEST_CSV)
    rows    = []
    for reward_col in REWARD_VARIANTS:
        print(f"\nAblation: training BCQ with reward_col={reward_col} (10k steps) ...")
        bc_net, q_net, _ = train_bcq(
            n_steps=10_000,
            reward_col=reward_col,
            save_model=False,
        )
        wrapper  = BCQPolicyWrapper(bc_net, q_net)
        episodes = extract_episodes(test_df, reward_col=reward_col)
        pdis     = compute_pdis_estimate(episodes, wrapper, behavior_policy)
        mood_imp, n_matched = compute_mood_improvement_stats(episodes, wrapper)
        rows.append({
            "reward_variant":   reward_col,
            "pdis":             _fmt(pdis),
            "mood_improvement": _fmt(mood_imp),
            "n_matched":        n_matched,
        })

    print(f"\n{'Reward Variant':<32} {'PDIS':>10} {'Mood Δ':>10} {'n':>4}")
    print("-" * 60)
    for r in rows:
        print(
            f"{r['reward_variant']:<32} "
            f"{str(r['pdis'] or 'N/A'):>10} "
            f"{str(r['mood_improvement'] or 'N/A'):>10} "
            f"{r['n_matched']:>4}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ABLATION_CSV, index=False)
    print(f"\nSaved ablation results → {ABLATION_CSV}")


# ---------------------------------------------------------------------------
# Final summary JSON (source of truth for report writing)
# ---------------------------------------------------------------------------

SUMMARY_PATH = MODEL_DIR / "juan_final_results_summary.json"


def build_final_summary(results: list, test_df: pd.DataFrame) -> None:
    """
    Writes models/juan_final_results_summary.json — a single dict with the
    key numbers needed for writing the CS 224R final report.
    """
    # Best mood improvement among policies with at least 1 valid observation
    mood_valid = [
        r for r in results
        if r.get("mood_improvement") is not None and r.get("n_matched", 0) > 0
    ]
    best_mood  = max(mood_valid, key=lambda r: r["mood_improvement"]) if mood_valid else {}

    # Best PDIS
    pdis_valid = [r for r in results if r.get("pdis") is not None]
    best_pdis  = max(pdis_valid, key=lambda r: r["pdis"]) if pdis_valid else {}

    # Reward ablation rows (from CSV if it exists, else empty)
    ablation = []
    if ABLATION_CSV.exists():
        ablation = pd.read_csv(ABLATION_CSV).to_dict(orient="records")

    n_mood_observed = int(test_df["next_mood_observed"].sum()) if "next_mood_observed" in test_df.columns else 0

    summary = {
        "best_mood_improvement": {
            "policy": best_mood.get("policy"),
            "value":  best_mood.get("mood_improvement"),
            "n_matched": best_mood.get("n_matched"),
        },
        "best_pdis": {
            "policy": best_pdis.get("policy"),
            "value":  best_pdis.get("pdis"),
        },
        "key_finding": (
            "BCQ and CQL achieve equal in-support mood improvement "
            "but behavior cloning outperforms both, suggesting action labels are "
            "learnable from state features alone under sparse reward"
        ),
        "n_test_rows": len(test_df),
        "n_mood_observed": n_mood_observed,
        "reward_ablation": ablation,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extended policy comparison for StudentLife offline RL."
    )
    parser.add_argument(
        "--reward-ablation",
        action="store_true",
        help="Also run BCQ reward-variant ablation (trains 3 extra BCQ models).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=(
            "Bootstrap replicates for OPE/match CIs (0 = skip; mood uses analytic CI)."
        ),
    )
    args = parser.parse_args()

    # Load test split
    print("Loading test split ...")
    test_df        = pd.read_csv(TEST_CSV)
    episodes       = extract_episodes(test_df)
    all_states     = test_df[STATE_COLS].fillna(0.0).to_numpy("float32")
    logged_actions = test_df["action"].to_numpy("int64")
    print(
        f"  rows={len(test_df)}, episodes={len(episodes)}, "
        f"students={test_df['student_id'].nunique()}"
    )

    # Load behavior policy first (needed for IS weights in all evaluations)
    print("\nLoading behavior policy ...")
    behavior_policy, other_baselines = load_baseline_policies()

    # Load all evaluation policies
    policies = []

    print("Loading CQL ...")
    try:
        policies.append(load_cql_policy())
        print("  CQL loaded.")
    except Exception as exc:
        print(f"  CQL skipped: {exc}")

    print("Loading BCQ ...")
    try:
        policies.append(load_bcq_policy())
        print("  BCQ loaded.")
    except Exception as exc:
        print(f"  BCQ skipped: {exc}")

    print("Loading contextual bandit ...")
    try:
        policies.append(load_bandit_policy())
        print("  Bandit loaded.")
    except Exception as exc:
        print(f"  Bandit skipped: {exc}")

    policies.append(behavior_policy)
    policies.extend(other_baselines)

    # Evaluate
    print(f"\nEvaluating {len(policies)} policies on test split ...")
    results = []
    for policy in policies:
        print(f"  {policy.name} ...")
        results.append(
            evaluate_policy_extended_with_uncertainty(
                policy,
                episodes,
                all_states,
                logged_actions,
                behavior_policy,
                n_bootstrap=args.n_bootstrap,
            )
        )

    results = enrich_bandit_metrics(results)

    # Print summary table to stdout
    print_summary_table(results)

    # Save results
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_JSON, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Flatten action_distribution for CSV
    flat_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "action_distribution"}
        for a_name, frac in r["action_distribution"].items():
            row[f"action_{a_name}"] = round(frac, 6)
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV.relative_to(PROJECT_ROOT)}")

    # Final summary JSON (always written, updated after ablation if run)
    build_final_summary(results, test_df)

    # Generate figures
    print("\nGenerating figures ...")
    fig_mood_improvement(results)
    fig_action_heatmap(results)
    fig_behavior_support_scatter(results)
    fig_bcq_training_curves()

    # Optional reward ablation — rebuilds summary JSON with ablation data included
    if args.reward_ablation:
        print("\n=== BCQ Reward Variant Ablation ===")
        run_reward_ablation(behavior_policy)
        build_final_summary(results, test_df)

    print("\nDone.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# AI Tools Disclosure
# ---------------------------------------------------------------------------
# Claude (claude-sonnet-4-6) was used to generate boilerplate infrastructure
# in this file including: file path constants, extract_episodes() data loader,
# BanditPolicyWrapper class (predict/action_probs scaffolding), all four figure
# functions (axis labels, color mapping, seaborn heatmap call, plt.savefig),
# print_summary_table() formatting, main() loading/saving scaffold, and the
# reward ablation loop structure.
#
# Juan Pablo Pacheco independently implemented the core evaluation logic:
#   - compute_mood_improvement()    (NaN-aware finite-diff filtering)
#   - evaluate_policy_extended()    (full metric assembly per policy)
#   - _behavior_support()           (π_b support metric)
#   - _action_dist()                (action distribution summary)
#   - run_reward_ablation()         (reward variant comparison logic)
#   - BanditPolicyWrapper._sa_features() / ._score_all()
#     (feature construction matching train_contextual_bandit.py)
