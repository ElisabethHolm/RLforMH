"""
Offline Policy Evaluation (OPE) for StudentLife RL policies.
CS 224R Spring 2026 — Juan Pablo Pacheco

Compares the trained Discrete CQL policy against baselines using:
  - Per-Decision Importance Sampling (PDIS): re-weights observed returns
  - Direct Method (DM): estimates initial value using CQL Q(s0, pi(s0))
  - Mood improvement: mean next-day mood delta on steps where the
    policy's action matches the logged action

Run (from repo root, with venv active):
    python algorithms/evaluate_policies.py

AI Tools Disclosure — see bottom of file.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH    = PROJECT_ROOT / "daily_studentlife.csv"
MODEL_DIR    = PROJECT_ROOT / "models"
CQL_PATH     = MODEL_DIR   / "studentlife_discrete_cql.d3"
BASELINE_PKL = MODEL_DIR   / "studentlife_baseline_models.pkl"
OUT_PATH     = MODEL_DIR   / "ope_metrics.json"

STATE_COLS = [
    "mood", "sleep_z", "activity_z", "social_z",
    "mood_lag1", "sleep_z_lag1", "activity_z_lag1", "social_z_lag1",
    "mood_lag2", "sleep_z_lag2", "activity_z_lag2", "social_z_lag2",
    "mood_lag3", "sleep_z_lag3", "activity_z_lag3", "social_z_lag3",
    "mood_observed",
]

N_ACTIONS = 7
GAMMA     = 0.99   # must match CQL training (params.json)
IS_CLIP   = 5.0    # max cumulative IS weight (standard variance-reduction trick)
EPSILON   = 0.05   # soft epsilon for deterministic policies in IS computation
SEED      = 42

ACTION_NAMES = {
    0: "none",            1: "increase_activity", 2: "decrease_activity",
    3: "increase_sleep",  4: "decrease_sleep",
    5: "increase_social", 6: "decrease_social",
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_test_students(df: pd.DataFrame) -> pd.DataFrame:
    """Return the same test-student slice used in train_baselines.py (seed=42)."""
    rng = np.random.default_rng(SEED)
    students = np.array(sorted(df["student_id"].unique()))
    rng.shuffle(students)
    n_train = int(len(students) * 0.8)
    n_val   = int(len(students) * 0.1)
    test_students = set(students[n_train + n_val:])
    return df[df["student_id"].isin(test_students)].reset_index(drop=True)


def extract_episodes(df: pd.DataFrame) -> list:
    """
    Group rows into episodes by (student_id, episode_id).
    Each episode is a dict with arrays aligned by timestep.
    """
    episodes = []
    for (sid, eid), grp in df.groupby(["student_id", "episode_id"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        episodes.append({
            "student_id": sid,
            "episode_id":  eid,
            "states":      grp[STATE_COLS].to_numpy("float32"),
            "actions":     grp["action"].to_numpy("int64"),
            "rewards":     grp["reward"].to_numpy("float32"),
            "mood":        grp["mood"].to_numpy("float32"),
            "next_moods":  grp["next_mood"].to_numpy("float32"),
            "T":           len(grp),
        })
    return episodes

# ---------------------------------------------------------------------------
# Policy wrappers  (uniform interface: predict() + action_probs())
# ---------------------------------------------------------------------------

class CQLPolicyWrapper:
    """Wraps a trained d3rlpy DiscreteCQL model."""
    name = "cql"

    def __init__(self, model):
        self._model = model

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model.predict(states)

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        """Epsilon-greedy softening so IS weights are always finite."""
        actions = self.predict(states)
        probs = np.full((len(states), N_ACTIONS), EPSILON / N_ACTIONS)
        probs[np.arange(len(states)), actions] += (1.0 - EPSILON)
        return probs

    def q_values(self, states: np.ndarray) -> np.ndarray:
        """
        Return Q(s, a) for all actions — shape (n_states, N_ACTIONS).
        Uses d3rlpy 2.x predict_value(obs, actions) looped over each action.
        """
        return np.column_stack([
            self._model.predict_value(states, np.full(len(states), a, dtype="int64"))
            for a in range(N_ACTIONS)
        ])


class SklearnPolicyWrapper:
    """Wraps sklearn classifiers (dummy baselines, behavior cloning)."""
    def __init__(self, name: str, model):
        self.name   = name
        self._model = model

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model.predict(states)

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(states)
        actions = self.predict(states)
        probs = np.full((len(states), N_ACTIONS), EPSILON / N_ACTIONS)
        probs[np.arange(len(states)), actions] += (1.0 - EPSILON)
        return probs


class RuleBasedPolicyWrapper:
    """
    Heuristic: pick the behavioral dimension farthest from the student's baseline
    and recommend correcting it. Implements the same logic as train_baselines.py.
    """
    name = "rule_based"
    _feature_to_actions = {
        "activity_z": (1, 2),
        "sleep_z":    (3, 4),
        "social_z":   (5, 6),
    }

    def __init__(self, threshold: float = 0.6):
        self._threshold   = threshold
        self._feature_idx = {
            feat: STATE_COLS.index(feat) for feat in self._feature_to_actions
        }

    def predict(self, states: np.ndarray) -> np.ndarray:
        actions = []
        for row in states:
            values = {f: float(row[idx]) for f, idx in self._feature_idx.items()}
            sel    = max(values, key=lambda f: abs(values[f]))
            if abs(values[sel]) < self._threshold:
                actions.append(0)
                continue
            inc, dec = self._feature_to_actions[sel]
            actions.append(inc if values[sel] < 0 else dec)
        return np.array(actions, dtype="int64")

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        actions = self.predict(states)
        probs   = np.full((len(states), N_ACTIONS), EPSILON / N_ACTIONS)
        probs[np.arange(len(states)), actions] += (1.0 - EPSILON)
        return probs


class RandomPolicyWrapper:
    """Uniform random policy."""
    name = "random_uniform"

    def predict(self, states: np.ndarray) -> np.ndarray:
        return np.random.default_rng(SEED).integers(0, N_ACTIONS, len(states)).astype("int64")

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        return np.full((len(states), N_ACTIONS), 1.0 / N_ACTIONS)

# ---------------------------------------------------------------------------
# Core OPE estimators — (Juan) implement the four functions below
# ---------------------------------------------------------------------------

def compute_discounted_return(rewards: np.ndarray, gamma: float = GAMMA) -> float:
    """
    Compute the discounted cumulative return for one episode.

    G = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^{T-1} * r_{T-1}

    Args:
        rewards: 1-D array of per-step rewards, shape (T,)
        gamma:   discount factor

    Returns:
        scalar float G

    """
    # -----------------------------------------------------------------------
    T = rewards.shape[0]
    return np.dot(np.full(T, gamma) ** np.arange(T), rewards)
    # -----------------------------------------------------------------------


def compute_pdis_estimate(
    episodes: list,
    eval_policy,
    behavior_policy,
    gamma:  float = GAMMA,
    clip:   float = IS_CLIP,
) -> float:
    """
    Per-Decision Importance Sampling (PDIS) estimator.

    For each step t in each episode:
        rho_t       = pi_eval(a_t | s_t) / pi_behavior(a_t | s_t)
        cum_weight_t = clip( product_{i=0}^{t} rho_i, max=clip )
        contribution = gamma^t * cum_weight_t * r_t

    The PDIS estimate is the mean of all contributions across
    every (episode, step) pair in the dataset.

    Why PDIS instead of trajectory-level IS?
      PDIS multiplies IS ratios only up to step t when weighting r_t,
      which gives lower variance than trajectory-level IS while remaining
      an unbiased estimator (Precup et al. 2000).

    Args:
        episodes:        list of episode dicts from extract_episodes()
        eval_policy:     policy being evaluated  — needs .action_probs(states)
        behavior_policy: logging / behavior policy — needs .action_probs(states)
        gamma:           discount factor
        clip:            max allowed cumulative IS weight (clips outliers)

    Returns:
        scalar float: PDIS estimate of eval_policy's expected return

    """
    # -----------------------------------------------------------------------

    contributions = []
    for episode in episodes:
        eval_probs = eval_policy.action_probs(episode["states"])
        behavior_probs = behavior_policy.action_probs(episode["states"])
        a_t = episode["actions"]
        T = a_t.shape[0]
        rho_t = eval_probs[range(T), a_t] / (behavior_probs[range(T), a_t]+ 1e-10)
        weights = np.clip(np.cumprod(rho_t), 0, clip)
        discounts = gamma ** np.arange(T)
        contributions.extend(discounts * weights * episode["rewards"])
    return np.mean(contributions)

    # -----------------------------------------------------------------------


def _importance_ratios(episode: dict, eval_policy, behavior_policy) -> np.ndarray:
    """Return per-step pi_eval(a_t|s_t) / pi_behavior(a_t|s_t)."""
    eval_probs = eval_policy.action_probs(episode["states"])
    behavior_probs = behavior_policy.action_probs(episode["states"])
    actions = episode["actions"]
    idx = np.arange(actions.shape[0])
    return eval_probs[idx, actions] / (behavior_probs[idx, actions] + 1e-10)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(weights * values) / denom)


def compute_importance_weight_diagnostics(
    episodes: list,
    eval_policy,
    behavior_policy,
    clip: float = IS_CLIP,
) -> dict:
    """Summarize final trajectory weights for variance/support diagnostics."""
    weights = []
    nonzero = 0
    for episode in episodes:
        ratios = _importance_ratios(episode, eval_policy, behavior_policy)
        weight = float(np.clip(np.prod(ratios), 0.0, clip))
        weights.append(weight)
        if weight > 0.0:
            nonzero += 1

    weights = np.asarray(weights, dtype="float64")
    weight_sq_sum = float(np.sum(weights**2))
    ess = (
        float((np.sum(weights) ** 2) / weight_sq_sum)
        if weight_sq_sum > 0.0
        else 0.0
    )
    return {
        "effective_sample_size": ess,
        "weight_mean": float(np.mean(weights)) if len(weights) else float("nan"),
        "weight_max": float(np.max(weights)) if len(weights) else float("nan"),
        "nonzero_weight_episodes": nonzero,
        "num_episodes": int(len(weights)),
    }


def compute_trajectory_is_estimate(
    episodes: list,
    eval_policy,
    behavior_policy,
    gamma: float = GAMMA,
    clip: float = IS_CLIP,
) -> float:
    """
    Full-trajectory importance sampling estimate.

    This is mainly diagnostic: multiplying ratios across an entire episode can
    have high variance when the learned policy has limited logged-action support.
    """
    returns = []
    for episode in episodes:
        ratios = _importance_ratios(episode, eval_policy, behavior_policy)
        weight = float(np.clip(np.prod(ratios), 0.0, clip))
        returns.append(weight * compute_discounted_return(episode["rewards"], gamma))
    return float(np.mean(returns)) if returns else float("nan")


def compute_weighted_is_estimate(
    episodes: list,
    eval_policy,
    behavior_policy,
    gamma: float = GAMMA,
    clip: float = IS_CLIP,
) -> float:
    """Self-normalized full-trajectory IS estimate."""
    returns = []
    weights = []
    for episode in episodes:
        ratios = _importance_ratios(episode, eval_policy, behavior_policy)
        weights.append(float(np.clip(np.prod(ratios), 0.0, clip)))
        returns.append(compute_discounted_return(episode["rewards"], gamma))
    if not returns:
        return float("nan")
    return _weighted_mean(
        np.asarray(returns, dtype="float64"),
        np.asarray(weights, dtype="float64"),
    )


def compute_weighted_pdis_estimate(
    episodes: list,
    eval_policy,
    behavior_policy,
    gamma: float = GAMMA,
    clip: float = IS_CLIP,
) -> float:
    """
    Self-normalized per-decision IS on the same per-step scale as PDIS.

    For each timestep t, cumulative weights are normalized across episodes that
    have a t-th transition, then the per-timestep estimates are averaged with the
    same step-count weighting as the existing PDIS implementation.
    """
    if not episodes:
        return float("nan")

    max_t = max(ep["T"] for ep in episodes)
    total_steps = sum(ep["T"] for ep in episodes)
    weighted_terms = []

    for t in range(max_t):
        weights = []
        rewards = []
        for episode in episodes:
            if episode["T"] <= t:
                continue
            ratios = _importance_ratios(episode, eval_policy, behavior_policy)
            weights.append(float(np.clip(np.prod(ratios[: t + 1]), 0.0, clip)))
            rewards.append(float(episode["rewards"][t]))

        weights = np.asarray(weights, dtype="float64")
        rewards = np.asarray(rewards, dtype="float64")
        if len(rewards) == 0 or np.sum(weights) <= 0.0:
            continue

        timestep_value = (gamma**t) * _weighted_mean(rewards, weights)
        weighted_terms.append(timestep_value * (len(rewards) / total_steps))

    return float(np.sum(weighted_terms)) if weighted_terms else float("nan")


def compute_doubly_robust_estimate(
    episodes: list,
    eval_policy,
    behavior_policy,
    gamma: float = GAMMA,
    clip: float = IS_CLIP,
) -> float:
    """
    Sequential doubly robust estimate using the policy's Q-function.

    Requires eval_policy.q_values(states). DR is useful as a sensitivity check,
    but it inherits bias when learned Q-values are optimistic.
    """
    estimates = []
    for episode in episodes:
        states = episode["states"]
        actions = episode["actions"]
        rewards = episode["rewards"]
        T = episode["T"]

        eval_probs = eval_policy.action_probs(states)
        behavior_probs = behavior_policy.action_probs(states)
        q_values = eval_policy.q_values(states)
        v_values = np.sum(eval_probs * q_values, axis=1)

        ratios = eval_probs[np.arange(T), actions] / (
            behavior_probs[np.arange(T), actions] + 1e-10
        )
        cumulative_weights = np.clip(np.cumprod(ratios), 0.0, clip)

        estimate = float(v_values[0]) if T else 0.0
        for t in range(T):
            next_v = float(v_values[t + 1]) if t + 1 < T else 0.0
            bellman_residual = (
                float(rewards[t])
                + gamma * next_v
                - float(q_values[t, actions[t]])
            )
            estimate += (gamma**t) * float(cumulative_weights[t]) * bellman_residual
        estimates.append(estimate)

    return float(np.mean(estimates)) if estimates else float("nan")


def compute_direct_method(
    episodes:   list,
    cql_policy: CQLPolicyWrapper,
    gamma:      float = GAMMA,
) -> float:
    """
    Direct Method (DM) estimator.

    Estimates the value of the CQL policy as V(s_0) = Q(s_0, pi(s_0)),
    averaged over all episode starting states s_0.

    This is valid only for the CQL policy because we have its Q-function.
    For all other policies, use PDIS.

    Args:
        episodes:   list of episode dicts
        cql_policy: CQLPolicyWrapper  — needs .q_values(states) and .predict(states)
        gamma:      discount factor (unused for Q-based estimate, kept for signature)

    Returns:
        scalar float: mean V(s_0) across episodes
    """ 
    # -----------------------------------------------------------------------
    max_v0_list = []
    for episode in episodes:
        s0 = episode["states"][[0]]
        q_vals = cql_policy.q_values(s0)
        best_a = np.argmax(q_vals[0])
        max_v0_list.append(q_vals[0, best_a])
    return np.mean(max_v0_list)
    # -----------------------------------------------------------------------


def compute_mood_improvement(episodes: list, policy) -> float:
    """
    Proxy for causal mood improvement (within-support estimation).

    For each timestep where the policy's chosen action matches the
    logged action (i.e., we have an observed outcome for that action),
    compute:   delta = next_mood - mood

    Steps where the policy disagrees with the logged action are excluded —
    we can't observe counterfactual outcomes in offline data.

    Args:
        episodes: list of episode dicts from extract_episodes()
        policy:   any policy with .predict(states)

    Returns:
        mean mood delta (float) over matched steps, or np.nan if no matches

    """
    # -----------------------------------------------------------------------
    differences = []
    for episode in episodes:
        pred_actions = policy.predict(episode["states"])
        # where predicted actions match with actions we actually observed
        matched_actions = np.where(pred_actions == episode["actions"])[0]
        for i in matched_actions:
            differences.append(episode["next_moods"][i] - episode["mood"][i])
    return np.mean(differences) if differences else np.nan

    # -----------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Evaluation driver (boilerplate — no TODOs below this line)
# ---------------------------------------------------------------------------

def evaluate_all(
    episodes:        list,
    policies:        list,
    cql_policy:      CQLPolicyWrapper,
    behavior_policy,
) -> list:
    results = []
    for policy in policies:
        print(f"  Evaluating {policy.name} ...")
        pdis = compute_pdis_estimate(episodes, policy, behavior_policy)
        trajectory_is = compute_trajectory_is_estimate(episodes, policy, behavior_policy)
        trajectory_wis = compute_weighted_is_estimate(episodes, policy, behavior_policy)
        weighted_pdis = compute_weighted_pdis_estimate(episodes, policy, behavior_policy)
        weight_diagnostics = compute_importance_weight_diagnostics(
            episodes, policy, behavior_policy
        )
        mood_imp = compute_mood_improvement(episodes, policy)
        result = {
            "policy": policy.name,
            "pdis_estimate": _round_or_none(pdis),
            "trajectory_is": _round_or_none(trajectory_is),
            "trajectory_wis": _round_or_none(trajectory_wis),
            "weighted_pdis": _round_or_none(weighted_pdis),
            "effective_sample_size": _round_or_none(
                weight_diagnostics["effective_sample_size"]
            ),
            "weight_max": _round_or_none(weight_diagnostics["weight_max"]),
            "mood_improvement": _round_or_none(mood_imp),
        }
        if policy.name == "cql":
            dm = compute_direct_method(episodes, cql_policy)
            dr = compute_doubly_robust_estimate(episodes, cql_policy, behavior_policy)
            result["direct_method_v0"] = _round_or_none(dm)
            result["dr"] = _round_or_none(dr)
        results.append(result)
    return results


def _round_or_none(value: float, ndigits: int = 6):
    return round(float(value), ndigits) if np.isfinite(value) else None


def print_results(results: list) -> None:
    col = 30
    print(
        f"\n{'Policy':<{col}} {'PDIS':>12} {'WPDIS':>12} "
        f"{'Traj WIS':>12} {'DR':>12} {'Mood Δ':>10} {'DM V(s0)':>12}"
    )
    print("-" * (col + 84))
    for r in results:
        pdis_s = f"{r['pdis_estimate']:>12.4f}"   if r["pdis_estimate"]    is not None else f"{'N/A':>12}"
        wpdis_s = f"{r['weighted_pdis']:>12.4f}" if r["weighted_pdis"] is not None else f"{'N/A':>12}"
        wis_s = f"{r['trajectory_wis']:>12.4f}" if r["trajectory_wis"] is not None else f"{'N/A':>12}"
        dr_s = f"{r['dr']:>12.4f}" if r.get("dr") is not None else f"{'—':>12}"
        mood_s = f"{r['mood_improvement']:>10.4f}" if r["mood_improvement"] is not None else f"{'N/A':>10}"
        dm_s   = (
            f"{r['direct_method_v0']:>12.4f}"
            if r.get("direct_method_v0") is not None
            else f"{'—':>12}"
        )
        print(
            f"{r['policy']:<{col}} {pdis_s} {wpdis_s} {wis_s} "
            f"{dr_s} {mood_s} {dm_s}"
        )


def save_results(results: list, path: Path = OUT_PATH) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "notes": (
            "PDIS = Per-Decision IS using behavior_cloning_logistic as logging policy. "
            f"IS weights clipped at {IS_CLIP}. epsilon={EPSILON} soft policy for deterministic policies. "
            "WPDIS and trajectory_wis are self-normalized IS diagnostics. "
            "DR = sequential doubly robust estimate, reported when Q-values are available. "
            "DM = Direct Method using CQL Q(s0, pi(s0)), reported only for CQL. "
            "mood_improvement = mean next-day mood delta on matched-action steps."
        ),
        "gamma":  GAMMA,
        "is_clip": IS_CLIP,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved OPE results → {path}")


def main() -> None:
    # 1. Load data & extract test split
    df       = pd.read_csv(DATA_PATH)
    test_df  = load_test_students(df)
    print(f"Test rows: {len(test_df)}  |  students: {test_df['student_id'].nunique()}")
    episodes = extract_episodes(test_df)
    print(f"Episodes:  {len(episodes)}")

    # 2. Load CQL model — must build_with_dataset first because save_model
    #    only serializes weights, not the network architecture.
    full_dataset = MDPDataset(
        observations=df[STATE_COLS].to_numpy("float32"),
        actions=df["action"].to_numpy("int64"),
        rewards=df["reward"].to_numpy("float32"),
        terminals=df["done"].to_numpy("bool"),
    )
    cql_model = DiscreteCQLConfig().create(device="cpu")
    cql_model.build_with_dataset(full_dataset)
    cql_model.load_model(str(CQL_PATH))
    cql_policy = CQLPolicyWrapper(cql_model)

    # 3. Load sklearn baselines
    with open(BASELINE_PKL, "rb") as f:
        baseline_payload = pickle.load(f)
    sklearn_models = baseline_payload["models"]

    # Behavior cloning is our best proxy for the logging policy
    behavior_policy = SklearnPolicyWrapper(
        "behavior_cloning_logistic",
        sklearn_models["behavior_cloning_logistic"],
    )

    # 4. All eval policies
    policies = [
        cql_policy,
        RandomPolicyWrapper(),
        SklearnPolicyWrapper("majority_action",  sklearn_models["majority_action"]),
        SklearnPolicyWrapper("action_frequency", sklearn_models["action_frequency"]),
        behavior_policy,
        RuleBasedPolicyWrapper(
            threshold=baseline_payload["rule_based_baseline"]["rule_threshold"]
        ),
    ]

    # 5. Evaluate
    print("\nRunning offline policy evaluation ...")
    results = evaluate_all(episodes, policies, cql_policy, behavior_policy)

    # 6. Report
    print_results(results)
    save_results(results)

    # 7. Highlight best policy by PDIS
    valid = [r for r in results if r["pdis_estimate"] is not None]
    if valid:
        best = max(valid, key=lambda r: r["pdis_estimate"])
        print(f"\nBest policy by PDIS: {best['policy']}  ({best['pdis_estimate']:.4f})")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# AI Tools Disclosure
# ---------------------------------------------------------------------------
# Claude (claude-sonnet-4-6) was used to generate the boilerplate
# infrastructure in this file: data loading (load_test_students,
# extract_episodes), all policy wrapper classes (CQLPolicyWrapper,
# SklearnPolicyWrapper, RuleBasedPolicyWrapper, RandomPolicyWrapper),
# the evaluation driver (evaluate_all, print_results, save_results),
# and the main() entry point.
#
# Juan Pablo Pacheco independently implemented the four
# core OPE functions, given hints provided by Claude:
#   - compute_discounted_return  (discounted return formula)
#   - compute_pdis_estimate      (per-decision importance sampling)
#   - compute_direct_method      (Q-value-based value estimate)
#   - compute_mood_improvement   (within-support mood delta)
#

# ---------------------------------------------------------------------------
