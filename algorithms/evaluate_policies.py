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
        pdis     = compute_pdis_estimate(episodes, policy, behavior_policy)
        mood_imp = compute_mood_improvement(episodes, policy)
        result   = {
            "policy":           policy.name,
            "pdis_estimate":    round(float(pdis),     6) if not np.isnan(pdis)     else None,
            "mood_improvement": round(float(mood_imp), 6) if not np.isnan(mood_imp) else None,
        }
        if policy.name == "cql":
            dm = compute_direct_method(episodes, cql_policy)
            result["direct_method_v0"] = round(float(dm), 6) if not np.isnan(dm) else None
        results.append(result)
    return results


def print_results(results: list) -> None:
    col = 30
    print(f"\n{'Policy':<{col}} {'PDIS':>12} {'Mood Δ':>10} {'DM V(s0)':>12}")
    print("-" * (col + 36))
    for r in results:
        pdis_s = f"{r['pdis_estimate']:>12.4f}"   if r["pdis_estimate"]    is not None else f"{'N/A':>12}"
        mood_s = f"{r['mood_improvement']:>10.4f}" if r["mood_improvement"] is not None else f"{'N/A':>10}"
        dm_s   = (
            f"{r['direct_method_v0']:>12.4f}"
            if r.get("direct_method_v0") is not None
            else f"{'—':>12}"
        )
        print(f"{r['policy']:<{col}} {pdis_s} {mood_s} {dm_s}")


def save_results(results: list, path: Path = OUT_PATH) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "notes": (
            "PDIS = Per-Decision IS using behavior_cloning_logistic as logging policy. "
            f"IS weights clipped at {IS_CLIP}. epsilon={EPSILON} soft policy for deterministic policies. "
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
