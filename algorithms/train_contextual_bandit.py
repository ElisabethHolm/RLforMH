"""
Contextual bandit baseline for StudentLife offline RL.

This baseline ignores episode dynamics and learns an immediate reward model:

    q_hat(s, a) = E[reward | state=s, action=a]

At evaluation time it scores all seven actions for each state and chooses the
action with the highest predicted immediate reward. This gives a one-step
comparison point for DQN/CQL: if a sequential method only slightly improves on
this baseline, the gain may mostly come from dense reward correlations rather
than long-horizon planning.

Run from the repo root:
    python algorithms/train_contextual_bandit.py
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "final_datasets"
DATASET_BASENAME = "daily_studentlife"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "contextual_bandit_models.pkl"
METRICS_JSON_PATH = MODEL_DIR / "contextual_bandit_metrics.json"
METRICS_CSV_PATH = MODEL_DIR / "contextual_bandit_metrics.csv"

STATE_COLS = [
    "mood",
    "sleep_z",
    "activity_z",
    "social_z",
    "mood_lag1",
    "sleep_z_lag1",
    "activity_z_lag1",
    "social_z_lag1",
    "mood_lag2",
    "sleep_z_lag2",
    "activity_z_lag2",
    "social_z_lag2",
    "mood_lag3",
    "sleep_z_lag3",
    "activity_z_lag3",
    "social_z_lag3",
    "mood_observed",
]

ACTION_NAMES = {
    0: "none",
    1: "increase_activity",
    2: "decrease_activity",
    3: "increase_sleep",
    4: "decrease_sleep",
    5: "increase_social",
    6: "decrease_social",
}

N_ACTIONS = len(ACTION_NAMES)
REWARD_VARIANTS = ["reward_sparse", "reward_dense", "reward_observed_only"]
SEED = 42
PROPENSITY_FLOOR = 1e-6
MAX_IMPORTANCE_WEIGHT = 10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a contextual bandit baseline on StudentLife splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing <basename>.{train,val,test}.csv.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=DATASET_BASENAME,
        help="Split file basename.",
    )
    parser.add_argument(
        "--reward-variants",
        nargs="+",
        choices=REWARD_VARIANTS,
        default=REWARD_VARIANTS,
        help="Reward columns to fit and evaluate.",
    )
    parser.add_argument(
        "--model",
        choices=["ridge"],
        default="ridge",
        help="Immediate reward model type.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for behavior cloning.",
    )
    parser.add_argument(
        "--max-importance-weight",
        type=float,
        default=MAX_IMPORTANCE_WEIGHT,
        help="Clip IPS/DR importance weights for variance control.",
    )
    return parser.parse_args()


def load_splits(data_dir: Path, basename: str) -> dict:
    paths = {
        split: data_dir / f"{basename}.{split}.csv"
        for split in ("train", "val", "test")
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing split files: "
            + ", ".join(missing)
            + ". Generate them with dataset_prep/prepare_rl_dataset.py."
        )
    return {split: pd.read_csv(path) for split, path in paths.items()}


def state_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[STATE_COLS].fillna(0.0).to_numpy("float32")


def rewards(df: pd.DataFrame, reward_col: str) -> np.ndarray:
    return df[reward_col].fillna(0.0).to_numpy("float32")


def action_one_hot(actions: np.ndarray) -> np.ndarray:
    one_hot = np.zeros((len(actions), N_ACTIONS), dtype="float32")
    one_hot[np.arange(len(actions)), actions.astype(int)] = 1.0
    return one_hot


def state_action_features(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """
    Build features for q_hat(s, a).

    The interaction block lets Ridge learn action-specific state weights while
    keeping a shared state/action representation:
        [state, one_hot(action), state x one_hot(action)]
    """
    one_hot = action_one_hot(actions)
    interactions = (states[:, :, None] * one_hot[:, None, :]).reshape(
        len(states),
        states.shape[1] * N_ACTIONS,
    )
    return np.concatenate([states, one_hot, interactions], axis=1)


def all_action_features(states: np.ndarray) -> np.ndarray:
    repeated_states = np.repeat(states, N_ACTIONS, axis=0)
    tiled_actions = np.tile(np.arange(N_ACTIONS, dtype="int64"), len(states))
    return state_action_features(repeated_states, tiled_actions)


def fit_reward_model(train_df: pd.DataFrame, reward_col: str, alpha: float):
    states = state_matrix(train_df)
    actions = train_df["action"].to_numpy("int64")
    x_train = state_action_features(states, actions)
    y_train = rewards(train_df, reward_col)
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    model.fit(x_train, y_train)
    return model


def score_all_actions(model, states: np.ndarray) -> np.ndarray:
    features = all_action_features(states)
    return model.predict(features).reshape(len(states), N_ACTIONS)


def predict_actions(model, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_values = score_all_actions(model, states)
    return q_values.argmax(axis=1).astype("int64"), q_values


def fit_behavior_policy(train_df: pd.DataFrame, seed: int):
    x_train = state_matrix(train_df)
    y_train = train_df["action"].to_numpy("int64")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    model.fit(x_train, y_train)
    return model


def behavior_action_probs(model, states: np.ndarray) -> np.ndarray:
    raw_probs = model.predict_proba(states)
    probs = np.full((len(states), N_ACTIONS), PROPENSITY_FLOOR, dtype="float64")
    for col, action in enumerate(model.classes_):
        probs[:, int(action)] = raw_probs[:, col]
    probs = probs / probs.sum(axis=1, keepdims=True)
    return np.clip(probs, PROPENSITY_FLOOR, 1.0)


def action_distribution(actions: np.ndarray) -> dict:
    counts = np.bincount(actions, minlength=N_ACTIONS)
    total = counts.sum()
    return {
        ACTION_NAMES[action]: float(count / total) if total else 0.0
        for action, count in enumerate(counts)
    }


def matched_mood_improvement(df: pd.DataFrame, matches: np.ndarray) -> float | None:
    deltas = df.loc[matches, "next_mood"].to_numpy("float32") - df.loc[
        matches, "mood"
    ].to_numpy("float32")
    deltas = deltas[np.isfinite(deltas)]
    return float(deltas.mean()) if len(deltas) else None


def evaluate_split(
    split_name: str,
    df: pd.DataFrame,
    reward_col: str,
    reward_model,
    behavior_model,
    max_importance_weight: float,
) -> dict:
    states = state_matrix(df)
    logged_actions = df["action"].to_numpy("int64")
    observed_rewards = rewards(df, reward_col)
    chosen_actions, q_values = predict_actions(reward_model, states)

    behavior_probs = behavior_action_probs(behavior_model, states)
    logged_propensities = behavior_probs[
        np.arange(len(df)),
        logged_actions,
    ]

    matches = chosen_actions == logged_actions
    raw_weights = matches.astype("float64") / logged_propensities
    weights = np.clip(raw_weights, 0.0, max_importance_weight)
    selected_q = q_values[np.arange(len(df)), chosen_actions]
    logged_q = q_values[np.arange(len(df)), logged_actions]

    dm_reward = float(selected_q.mean())
    ips_reward = float(np.mean(weights * observed_rewards))
    snips_reward = (
        float(np.sum(weights * observed_rewards) / np.sum(weights))
        if weights.sum() > 0
        else None
    )
    dr_reward = float(
        np.mean(selected_q + weights * (observed_rewards - logged_q))
    )

    matched_rewards = observed_rewards[matches]
    matched_reward_mean = (
        float(matched_rewards.mean()) if len(matched_rewards) else None
    )

    return {
        "reward_variant": reward_col,
        "split": split_name,
        "dm_reward": dm_reward,
        "ips_reward": ips_reward,
        "snips_reward": snips_reward,
        "dr_reward": dr_reward,
        "matched_logged_reward_mean": matched_reward_mean,
        "matched_mood_improvement": matched_mood_improvement(df, matches),
        "matched_logged_action_count": int(matches.sum()),
        "action_match": float(matches.mean()),
        "importance_weight_mean": float(weights.mean()),
        "importance_weight_max": float(weights.max()) if len(weights) else 0.0,
        "num_examples": int(len(df)),
        "action_distribution": action_distribution(chosen_actions),
    }


def fit_and_evaluate(args, splits: dict, behavior_model) -> tuple[dict, list]:
    reward_models = {}
    metrics = []

    for reward_col in args.reward_variants:
        reward_model = fit_reward_model(
            splits["train"],
            reward_col=reward_col,
            alpha=args.alpha,
        )
        reward_models[reward_col] = reward_model

        for split_name, split_df in [
            ("validation", splits["val"]),
            ("test", splits["test"]),
        ]:
            metrics.append(
                evaluate_split(
                    split_name,
                    split_df,
                    reward_col,
                    reward_model,
                    behavior_model,
                    args.max_importance_weight,
                )
            )

    return reward_models, metrics


def clean_for_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_for_json(val) for val in value]
    return value


def save_outputs(args, reward_models, behavior_model, metrics) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_payload = {
        "reward_models": reward_models,
        "behavior_model": behavior_model,
        "state_cols": STATE_COLS,
        "action_names": ACTION_NAMES,
        "reward_variants": args.reward_variants,
        "model_type": args.model,
        "alpha": args.alpha,
        "seed": args.seed,
        "max_importance_weight": args.max_importance_weight,
        "notes": (
            "Contextual bandit baseline. Reward models estimate immediate "
            "E[reward | state, action] using state features, action one-hot "
            "features, and state-action interactions."
        ),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_payload, f)

    nested = {}
    for metric in metrics:
        nested.setdefault(metric["reward_variant"], {})[metric["split"]] = metric

    metrics_payload = {
        "notes": (
            "DM/IPS/SNIPS/DR are one-step contextual-bandit OPE metrics. "
            "IPS/SNIPS/DR use a behavior-cloning logistic model as the logging "
            "policy, with clipped importance weights for variance control. "
            "Sparse and observed-only rewards may be identical after NaN "
            "rewards are filled with zero."
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": args.model,
        "alpha": args.alpha,
        "seed": args.seed,
        "max_importance_weight": args.max_importance_weight,
        "metrics": clean_for_json(nested),
    }
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    flat = []
    for metric in metrics:
        row = {
            key: val
            for key, val in metric.items()
            if key != "action_distribution"
        }
        row.update(
            {
                f"action_rate_{name}": metric["action_distribution"][name]
                for name in ACTION_NAMES.values()
            }
        )
        flat.append(row)
    pd.DataFrame(flat).to_csv(METRICS_CSV_PATH, index=False)

    print(f"Saved contextual bandit models to {MODEL_PATH}")
    print(f"Saved contextual bandit metrics to {METRICS_JSON_PATH}")
    print(f"Saved flat contextual bandit metrics to {METRICS_CSV_PATH}")


def print_leaderboard(metrics: list) -> None:
    rows = [
        metric
        for metric in metrics
        if metric["split"] == "validation"
    ]
    rows = sorted(rows, key=lambda row: row["dr_reward"], reverse=True)

    print("\n================ CONTEXTUAL BANDIT VALIDATION LEADERBOARD ================")
    header = (
        f"{'reward':<22} {'DR':>10} {'IPS':>10} {'SNIPS':>10} "
        f"{'mood d':>10} {'match':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        snips = row["snips_reward"]
        snips_text = f"{snips:>10.4f}" if snips is not None else f"{'NA':>10}"
        mood = row["matched_mood_improvement"]
        mood_text = f"{mood:>10.4f}" if mood is not None else f"{'NA':>10}"
        print(
            f"{row['reward_variant']:<22} "
            f"{row['dr_reward']:>10.4f} "
            f"{row['ips_reward']:>10.4f} "
            f"{snips_text} "
            f"{mood_text} "
            f"{row['action_match']:>8.4f}"
        )


def main() -> None:
    args = parse_args()
    splits = load_splits(args.data_dir, args.basename)
    behavior_model = fit_behavior_policy(splits["train"], args.seed)
    reward_models, metrics = fit_and_evaluate(args, splits, behavior_model)
    print_leaderboard(metrics)
    save_outputs(args, reward_models, behavior_model, metrics)


if __name__ == "__main__":
    main()
