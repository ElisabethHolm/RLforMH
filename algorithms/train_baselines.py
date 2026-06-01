import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "daily_studentlife.csv"
MODEL_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_PATH = MODEL_DIR / "studentlife_baseline_models.pkl"
BC_MODEL_PATH = MODEL_DIR / "studentlife_behavior_cloning.pkl"
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"

#Possible states we can have
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


"""
A function to parse the arguments for the training. Written by GPT. 
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train standard baseline policies for StudentLife offline RL."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--rule-threshold",
        type=float,
        default=0.6,
        help="Absolute z-score threshold used by the rule-based baseline.",
    )
    return parser.parse_args()


"""
A function to split the data by student in order to create a train + validation set. Written by GPT. 
"""
def split_by_student(df: pd.DataFrame, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    students = np.array(sorted(df["student_id"].unique()))
    rng.shuffle(students)

    n_train = int(len(students) * train_frac)
    n_val = int(len(students) * val_frac)

    train_students = set(students[:n_train])
    val_students = set(students[n_train:n_train + n_val])
    test_students = set(students[n_train + n_val:])

    train = df[df["student_id"].isin(train_students)].reset_index(drop=True)
    val = df[df["student_id"].isin(val_students)].reset_index(drop=True)
    test = df[df["student_id"].isin(test_students)].reset_index(drop=True)
    return train, val, test


"""
A function to get the features and labels from the dataframe. Written by GPT. 
"""
def features_and_labels(df: pd.DataFrame):
    x = df[STATE_COLS].to_numpy("float32")
    y = df["action"].to_numpy("int64")
    return x, y

"""
Function to fit the relevant baseline models. 
"""
def fit_baseline_models(train_df: pd.DataFrame, seed: int, max_iter: int):
    x_train, y_train = features_and_labels(train_df)

    models = {
        #Use dummy classifiers to fit initial baselines
        "random_uniform": DummyClassifier(
            strategy="uniform",
            random_state=seed,
        ),
        "majority_action": DummyClassifier(
            strategy="most_frequent",
        ),
        "action_frequency": DummyClassifier(
            strategy="stratified",
            random_state=seed,
        ),
        #Behavior cloning, simple logistic regression model
        "behavior_cloning_logistic": make_pipeline(
            StandardScaler(), #stadnard scaler used to stadnardize features so that they're all on the same scale. 
            LogisticRegression(max_iter=max_iter, random_state=seed), #Fit logreg on the scaled features 
        ),
    }

    for model in models.values():
        model.fit(x_train, y_train)

    return models


"""
A function to implement the rule-based baseline. Written by GPT. 
"""
def rule_based_policy(df: pd.DataFrame, threshold: float) -> np.ndarray:
    feature_to_actions = {
        "activity_z": (1, 2),
        "sleep_z": (3, 4),
        "social_z": (5, 6),
    }
    actions = []

    for _, row in df.iterrows():
        values = {
            feature: float(row[feature])
            for feature in feature_to_actions
        }
        selected_feature = max(values, key=lambda feature: abs(values[feature]))
        selected_value = values[selected_feature]

        if abs(selected_value) < threshold:
            actions.append(0)
            continue

        increase_action, decrease_action = feature_to_actions[selected_feature]
        actions.append(increase_action if selected_value < 0 else decrease_action)

    return np.array(actions, dtype=np.int64)


"""
A function to get the action distribution. Written by GPT. 
"""
def action_distribution(actions: np.ndarray):
    counts = np.bincount(actions, minlength=len(ACTION_NAMES))
    total = counts.sum()
    return {
        ACTION_NAMES[action]: float(count / total) if total else 0.0
        for action, count in enumerate(counts)
    }


"""
A function to evaluate the policy. Written by GPT. 
"""
def evaluate_policy(name: str, predictions: np.ndarray, df: pd.DataFrame):
    logged_actions = df["action"].to_numpy()
    rewards = df["reward"].to_numpy()
    matches = predictions == logged_actions

    matched_rewards = rewards[matches]
    matched_reward = (
        float(matched_rewards.mean())
        if len(matched_rewards) > 0
        else None
    )

    return {
        "name": name,
        "accuracy": float(accuracy_score(logged_actions, predictions)),
        "macro_f1": float(
            f1_score(
                logged_actions,
                predictions,
                labels=list(ACTION_NAMES),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                logged_actions,
                predictions,
                labels=list(ACTION_NAMES),
                average="weighted",
                zero_division=0,
            )
        ),
        "matched_logged_reward_mean": matched_reward,
        "matched_logged_action_count": int(matches.sum()),
        "num_examples": int(len(df)),
        "action_distribution": action_distribution(predictions),
    }


"""
A function to print the metrics. Written by GPT. 
"""
def print_metrics(metrics):
    for metric in metrics:
        reward = metric["matched_logged_reward_mean"]
        reward_text = "NA" if reward is None else f"{reward:.4f}"
        print(
            f"{metric['name']}: "
            f"accuracy={metric['accuracy']:.4f}, "
            f"macro_f1={metric['macro_f1']:.4f}, "
            f"matched_reward={reward_text}, "
            f"matched_count={metric['matched_logged_action_count']}"
        )


"""
A function to evaluate the split. Written by GPT. 
"""
def evaluate_split(split_name: str, df: pd.DataFrame, models, rule_threshold: float):
    x, _ = features_and_labels(df)
    metrics = []

    for name, model in models.items():
        predictions = model.predict(x)
        metric = evaluate_policy(name, predictions, df)
        metric["split"] = split_name
        metrics.append(metric)

    rule_metric = evaluate_policy(
        "rule_based_baseline",
        rule_based_policy(df, rule_threshold),
        df,
    )
    rule_metric["split"] = split_name
    metrics.append(rule_metric)

    return metrics


"""
A function to save the baseline models. Written by GPT. 
"""
def save_baseline_models(models, args):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    baseline_payload = {
        "models": models,
        "rule_based_baseline": {
            "type": "z_score_threshold_policy",
            "rule_threshold": args.rule_threshold,
            "description": (
                "Selects the sleep/activity/social feature farthest from baseline; "
                "if its absolute z-score is below the threshold, predicts none; "
                "otherwise recommends moving that feature back toward baseline."
            ),
        },
        "state_cols": STATE_COLS,
        "action_names": ACTION_NAMES,
        "seed": args.seed,
        "max_iter": args.max_iter,
    }

    with open(BASELINE_MODELS_PATH, "wb") as f:
        pickle.dump(baseline_payload, f)
    print(f"Saved all baseline models to {BASELINE_MODELS_PATH}")

    with open(BC_MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": models["behavior_cloning_logistic"],
                "state_cols": STATE_COLS,
                "action_names": ACTION_NAMES,
                "model_type": "sklearn_logistic_regression_behavior_cloning",
                "max_iter": args.max_iter,
            },
            f,
        )
    print(f"Saved behavior cloning model to {BC_MODEL_PATH}")


"""
A function to save the metrics. Written by GPT. 
"""
def save_metrics(metrics, split_sizes):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_sizes": split_sizes,
        "metrics": metrics,
        "notes": (
            "Baselines imitate logged inferred actions. matched_logged_reward_mean "
            "only averages rewards where the baseline chose the logged action; it is "
            "not an online causal estimate."
        ),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved baseline metrics to {METRICS_PATH}")


def main():
    args = parse_args()

    df = pd.read_csv(args.data)
    train_df, val_df, test_df = split_by_student(
        df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    split_sizes = {
        "train": len(train_df),
        "validation": len(val_df),
        "test": len(test_df),
    }
    print(
        f"Split rows: train={split_sizes['train']}, "
        f"val={split_sizes['validation']}, test={split_sizes['test']}"
    )

    models = fit_baseline_models(train_df, seed=args.seed, max_iter=args.max_iter)

    metrics = []
    for split_name, split_df in [
        ("validation", val_df),
        ("test", test_df),
    ]:
        split_metrics = evaluate_split(
            split_name,
            split_df,
            models,
            rule_threshold=args.rule_threshold,
        )
        print(f"\n{split_name} metrics")
        print_metrics(split_metrics)
        metrics.extend(split_metrics)

    save_baseline_models(models, args)
    save_metrics(metrics, split_sizes)


if __name__ == "__main__":
    main()
