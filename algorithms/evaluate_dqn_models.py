"""
Evaluate saved best DQN / Double DQN models with sequential OPE estimators.

This script does not retrain the grid. It loads the best model files written by
hyperparameter_search_dqn.py and evaluates them on train/validation/test splits.

Run from the repo root, with the project venv active:
    python algorithms/evaluate_dqn_models.py
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from d3rlpy.metrics import (
    AverageValueEstimationEvaluator,
    DiscreteActionMatchEvaluator,
    TDErrorEvaluator,
)
from d3rlpy.models.encoders import VectorEncoderFactory

# Make sibling modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_policies import (  # noqa: E402
    ACTION_NAMES,
    GAMMA,
    N_ACTIONS,
    compute_doubly_robust_estimate,
    compute_importance_weight_diagnostics,
    compute_mood_improvement_stats,
    compute_pdis_estimate,
    compute_trajectory_is_estimate,
    compute_weighted_is_estimate,
    compute_weighted_pdis_estimate,
)
from ope_uncertainty import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    attach_policy_uncertainty,
)
from hyperparameter_search_dqn import (  # noqa: E402
    ALGO_CONFIGS,
    DATASET_BASENAME,
    DEFAULT_DATA_DIR,
    MODEL_DIR,
    RESULTS_JSON,
    REWARD_VARIANTS,
    DQNPolicyWrapper,
    STATE_COLS,
    build_mdp_dataset,
    direct_method_v0,
    extract_episodes,
    fit_behavior_policy,
    load_splits,
    mood_improvement,
    mood_improvement_stats,
    train_and_eval,
)


logging.getLogger("d3rlpy").setLevel(logging.WARNING)
try:
    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
except Exception:  # pragma: no cover - structlog is a d3rlpy dependency
    pass


OUT_JSON = MODEL_DIR / "dqn_saved_model_ope_metrics.json"
OUT_CSV = MODEL_DIR / "dqn_saved_model_ope_metrics.csv"
ABLATION_CSV = MODEL_DIR / "double_dqn_reward_ablation.csv"
ABLATION_JSON = MODEL_DIR / "double_dqn_reward_ablation.json"
SUBGROUP_JSON = MODEL_DIR / "dqn_subgroup_policy_analysis.json"
SUBGROUP_CSV = MODEL_DIR / "dqn_subgroup_policy_analysis.csv"
STUDENT_CSV = MODEL_DIR / "dqn_student_policy_analysis.csv"

LOW_STATE_THRESHOLD = -0.5
HIGH_STATE_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved best DQN / Double DQN models with OPE."
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
        "--results-json",
        type=Path,
        default=RESULTS_JSON,
        help="DQN hyperparameter search JSON containing best_per_variant.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Splits to evaluate.",
    )
    parser.add_argument(
        "--reward-variants",
        nargs="+",
        default=None,
        help="Optional subset of reward variants from best_per_variant.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device passed to d3rlpy.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUT_JSON,
        help="Path for nested JSON metrics.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUT_CSV,
        help="Path for flat CSV metrics.",
    )
    parser.add_argument(
        "--subgroup-analysis",
        action="store_true",
        help="Compute state-conditioned subgroup policy/OPE diagnostics.",
    )
    parser.add_argument(
        "--subgroups",
        nargs="+",
        default=None,
        help="Optional subgroup families to include, e.g. mood sleep activity.",
    )
    parser.add_argument(
        "--min-subgroup-rows",
        type=int,
        default=20,
        help="Minimum rows before subgroup OPE is treated as reliable.",
    )
    parser.add_argument(
        "--student-analysis",
        action="store_true",
        help="Compute per-student policy diagnostics.",
    )
    parser.add_argument(
        "--subgroup-output-json",
        type=Path,
        default=SUBGROUP_JSON,
        help="Path for nested subgroup JSON metrics.",
    )
    parser.add_argument(
        "--subgroup-output-csv",
        type=Path,
        default=SUBGROUP_CSV,
        help="Path for flat subgroup CSV metrics.",
    )
    parser.add_argument(
        "--student-output-csv",
        type=Path,
        default=STUDENT_CSV,
        help="Path for per-student diagnostic CSV metrics.",
    )
    parser.add_argument(
        "--best-per-algo",
        action="store_true",
        help=(
            "Evaluate the best validation-PDIS config for each algo (dqn, "
            "double_dqn) separately instead of only the overall best model."
        ),
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["dqn", "double_dqn"],
        help="Algos to include when --best-per-algo is set.",
    )
    parser.add_argument(
        "--retrain-missing",
        action="store_true",
        help="Retrain configs whose per-algo checkpoint is missing.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=(
            "Bootstrap replicates for OPE/match CIs (0 = skip; mood uses analytic CI)."
        ),
    )
    parser.add_argument(
        "--reward-ablation",
        action="store_true",
        help=(
            "Train Double DQN once per reward variant with fixed best dense-reward "
            "hyperparameters and evaluate on the test split (mirrors BCQ ablation)."
        ),
    )
    parser.add_argument(
        "--ablation-n-steps",
        type=int,
        default=None,
        help="Training steps for --reward-ablation (default: n_steps from search JSON).",
    )
    parser.add_argument(
        "--ablation-output-csv",
        type=Path,
        default=ABLATION_CSV,
        help="CSV path for --reward-ablation results.",
    )
    parser.add_argument(
        "--ablation-output-json",
        type=Path,
        default=ABLATION_JSON,
        help="JSON path for --reward-ablation results.",
    )
    return parser.parse_args()


def load_search_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing DQN search results: {path}")
    with open(path) as f:
        return json.load(f)


def _is_better_pdis(candidate: dict, incumbent: dict) -> bool:
    c, i = candidate.get("pdis"), incumbent.get("pdis")
    if c is None or (isinstance(c, float) and not np.isfinite(c)):
        return False
    if i is None or (isinstance(i, float) and not np.isfinite(i)):
        return True
    return c > i


def best_configs_per_algo(payload: dict, reward_col: str, algos: list[str]) -> dict:
    """Best validation-PDIS row per algo for one reward variant."""
    best = {}
    for row in payload["results"]:
        if row["reward_variant"] != reward_col or row["algo"] not in algos:
            continue
        current = best.get(row["algo"])
        if current is None or _is_better_pdis(row, current):
            best[row["algo"]] = row
    missing = sorted(set(algos) - set(best))
    if missing:
        raise ValueError(
            f"No search results for algos={missing} on reward_variant={reward_col}"
        )
    return best


def resolve_model_path(reward_col: str, algo: str, best_per_variant: dict) -> Path | None:
    per_algo = MODEL_DIR / f"dqn_best_{reward_col}_{algo}.d3"
    if per_algo.exists():
        return per_algo
    overall = best_per_variant.get(reward_col, {})
    if overall.get("algo") == algo:
        legacy = MODEL_DIR / f"dqn_best_{reward_col}.d3"
        if legacy.exists():
            return legacy
    return None


def ensure_model(
    config: dict,
    reward_col: str,
    splits: dict,
    behavior_policy,
    payload: dict,
    device: str,
    seed: int,
    retrain_missing: bool,
):
    algo = config["algo"]
    model_path = resolve_model_path(reward_col, algo, payload["best_per_variant"])
    train_ds = build_mdp_dataset(splits["train"], reward_col)
    model = build_model(config, train_ds, device)

    if model_path is not None:
        model.load_model(str(model_path))
        return model

    if not retrain_missing:
        raise FileNotFoundError(
            f"Missing checkpoint for {reward_col}/{algo}. Re-run with "
            f"--retrain-missing or retrain via hyperparameter_search_dqn.py."
        )

    n_steps = payload.get("n_steps", 5000)
    n_steps_per_epoch = payload.get("n_steps_per_epoch", n_steps)
    val_ds = build_mdp_dataset(splits["val"], reward_col)
    val_episodes = extract_episodes(splits["val"], reward_col)
    print(
        f"Retraining {reward_col}/{algo} "
        f"(lr={config['learning_rate']}, bs={config['batch_size']}, "
        f"hidden={config['hidden_units']}) ..."
    )
    model, _ = train_and_eval(
        algo,
        config,
        train_ds,
        val_ds,
        val_episodes,
        behavior_policy,
        n_steps,
        n_steps_per_epoch,
        device,
        seed,
    )
    save_path = MODEL_DIR / f"dqn_best_{reward_col}_{algo}.d3"
    model.save_model(str(save_path))
    print(f"Saved retrained checkpoint -> {save_path}")
    return model


def action_rate_columns(model, split_df: pd.DataFrame) -> dict:
    policy_actions = DQNPolicyWrapper(model).predict(state_matrix(split_df))
    rates = action_distribution(policy_actions, "policy")
    return {
        f"action_{name}": rates[f"policy_action_rate_{name}"]
        for name in ACTION_NAMES.values()
    }


def build_model(best_config: dict, train_ds, device: str):
    algo = best_config["algo"]
    config = ALGO_CONFIGS[algo](
        batch_size=best_config["batch_size"],
        learning_rate=best_config["learning_rate"],
        gamma=GAMMA,
        target_update_interval=best_config["target_update_interval"],
        encoder_factory=VectorEncoderFactory(hidden_units=best_config["hidden_units"]),
    )
    model = config.create(device=device)
    model.build_with_dataset(train_ds)
    return model


def evaluate_model_on_split(
    model,
    split_df,
    reward_col: str,
    behavior_policy,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict:
    ds = build_mdp_dataset(split_df, reward_col)
    episodes = extract_episodes(split_df, reward_col)
    policy = DQNPolicyWrapper(model)
    weight_diagnostics = compute_importance_weight_diagnostics(
        episodes, policy, behavior_policy
    )
    mood_imp, mood_n = compute_mood_improvement_stats(episodes, policy)

    metrics = {
        "num_rows": int(len(split_df)),
        "num_episodes": int(len(episodes)),
        "td_error": float(TDErrorEvaluator(episodes=ds.episodes)(model, ds)),
        "action_match": float(
            DiscreteActionMatchEvaluator(episodes=ds.episodes)(model, ds)
        ),
        "value": float(AverageValueEstimationEvaluator(episodes=ds.episodes)(model, ds)),
        "pdis": float(compute_pdis_estimate(episodes, policy, behavior_policy)),
        "trajectory_is": float(
            compute_trajectory_is_estimate(episodes, policy, behavior_policy)
        ),
        "trajectory_wis": float(
            compute_weighted_is_estimate(episodes, policy, behavior_policy)
        ),
        "weighted_pdis": float(
            compute_weighted_pdis_estimate(episodes, policy, behavior_policy)
        ),
        "dr": float(compute_doubly_robust_estimate(episodes, policy, behavior_policy)),
        "effective_sample_size": weight_diagnostics["effective_sample_size"],
        "weight_mean": weight_diagnostics["weight_mean"],
        "weight_max": weight_diagnostics["weight_max"],
        "nonzero_weight_episodes": weight_diagnostics["nonzero_weight_episodes"],
        "mood_improvement": mood_imp,
        "mood_n_matched": mood_n,
        "direct_method_v0": direct_method_v0(episodes, policy),
    }
    attach_policy_uncertainty(
        metrics,
        episodes,
        policy,
        behavior_policy,
        n_bootstrap=n_bootstrap,
    )
    for suffix in ("_se", "_ci_low", "_ci_high"):
        ci_key = f"match_rate{suffix}"
        if ci_key in metrics:
            metrics[f"action_match{suffix}"] = metrics[ci_key]
    return metrics


def state_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[STATE_COLS].fillna(0.0).to_numpy("float32")


def action_distribution(actions: np.ndarray, prefix: str) -> dict:
    counts = np.bincount(actions.astype("int64"), minlength=N_ACTIONS)
    total = int(counts.sum())
    if total == 0:
        return {f"{prefix}_action_rate_{name}": float("nan") for name in ACTION_NAMES.values()}
    return {
        f"{prefix}_action_rate_{ACTION_NAMES[action]}": float(count / total)
        for action, count in enumerate(counts)
    }


def add_subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive subgroup columns from already available transition features."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["is_weekend"] = out["date"].dt.dayofweek >= 5

    global_mood_median = out["mood"].median(skipna=True)
    student_mood_median = out.groupby("student_id")["mood"].transform("median")
    out["student_mood_median"] = student_mood_median.fillna(global_mood_median)
    out["low_mood"] = (
        out["mood_observed"].astype(bool)
        & out["mood"].notna()
        & (out["mood"] <= out["student_mood_median"])
    )
    out["high_mood"] = (
        out["mood_observed"].astype(bool)
        & out["mood"].notna()
        & (out["mood"] > out["student_mood_median"])
    )
    out["mood_missing"] = ~out["mood_observed"].astype(bool)

    out["low_sleep"] = out["sleep_z"] < LOW_STATE_THRESHOLD
    out["normal_sleep"] = out["sleep_z"] >= LOW_STATE_THRESHOLD
    out["low_activity"] = out["activity_z"] < LOW_STATE_THRESHOLD
    out["high_activity"] = out["activity_z"] > HIGH_STATE_THRESHOLD
    out["low_social"] = out["social_z"] < LOW_STATE_THRESHOLD
    out["high_social"] = out["social_z"] > HIGH_STATE_THRESHOLD

    day_rank = out.groupby("student_id")["date"].rank(method="dense")
    day_count = out.groupby("student_id")["date"].transform("nunique").clip(lower=1)
    normalized_day = (day_rank - 1) / day_count
    out["early_term"] = normalized_day <= (1.0 / 3.0)
    out["late_term"] = normalized_day >= (2.0 / 3.0)
    return out


def build_subgroup_masks(df: pd.DataFrame, requested_families: list[str] | None) -> dict:
    families = requested_families or [
        "mood",
        "sleep",
        "activity",
        "social",
        "calendar",
        "term",
    ]
    available = {
        "mood": {
            "mood_observed": df["mood_observed"].astype(bool),
            "mood_missing": df["mood_missing"],
            "low_mood": df["low_mood"],
            "high_mood": df["high_mood"],
        },
        "sleep": {
            "low_sleep": df["low_sleep"],
            "normal_sleep": df["normal_sleep"],
        },
        "activity": {
            "low_activity": df["low_activity"],
            "high_activity": df["high_activity"],
        },
        "social": {
            "low_social": df["low_social"],
            "high_social": df["high_social"],
        },
        "calendar": {
            "weekday": ~df["is_weekend"],
            "weekend": df["is_weekend"],
        },
        "term": {
            "early_term": df["early_term"],
            "late_term": df["late_term"],
        },
    }
    unknown = sorted(set(families) - set(available))
    if unknown:
        raise ValueError(f"Unknown subgroup families: {unknown}")

    masks = {}
    for family in families:
        masks.update(available[family])
    return masks


def policy_behavior_metrics(
    model,
    df: pd.DataFrame,
    min_rows: int,
) -> dict:
    if df.empty:
        empty_metrics = {
            "num_rows": 0,
            "num_students": 0,
            "action_match": float("nan"),
            "dominant_policy_action": None,
            "dominant_policy_action_rate": float("nan"),
            "policy_collapsed": False,
            "increase_sleep_rate": float("nan"),
            "increase_activity_rate": float("nan"),
            "increase_social_rate": float("nan"),
        }
        empty_metrics.update(action_distribution(np.array([], dtype="int64"), "policy"))
        empty_metrics.update(action_distribution(np.array([], dtype="int64"), "logged"))
        return empty_metrics

    states = state_matrix(df)
    logged_actions = df["action"].to_numpy("int64")
    policy_actions = DQNPolicyWrapper(model).predict(states)
    dominant_action = int(np.bincount(policy_actions, minlength=N_ACTIONS).argmax())
    dominant_rate = float(np.mean(policy_actions == dominant_action))
    matches = policy_actions == logged_actions

    metrics = {
        "num_rows": int(len(df)),
        "num_students": int(df["student_id"].nunique()) if len(df) else 0,
        "action_match": float(matches.mean()) if len(df) else float("nan"),
        "dominant_policy_action": ACTION_NAMES[dominant_action],
        "dominant_policy_action_rate": dominant_rate,
        "policy_collapsed": bool(dominant_rate >= 0.8 and len(df) >= min_rows),
        "increase_sleep_rate": float(np.mean(policy_actions == 3)) if len(df) else float("nan"),
        "increase_activity_rate": float(np.mean(policy_actions == 1)) if len(df) else float("nan"),
        "increase_social_rate": float(np.mean(policy_actions == 5)) if len(df) else float("nan"),
    }
    metrics.update(action_distribution(policy_actions, "policy"))
    metrics.update(action_distribution(logged_actions, "logged"))
    return metrics


def evaluate_model_on_subset(
    model,
    subset_df: pd.DataFrame,
    reward_col: str,
    behavior_policy,
    min_rows: int,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict:
    metrics = policy_behavior_metrics(model, subset_df, min_rows)
    metrics["reliable_support"] = bool(metrics["num_rows"] >= min_rows)
    if subset_df.empty:
        return metrics

    ope = evaluate_model_on_split(
        model,
        subset_df,
        reward_col,
        behavior_policy,
        n_bootstrap=n_bootstrap,
    )
    metrics.update(ope)
    return metrics


def subgroup_threshold_metadata(min_rows: int) -> dict:
    return {
        "low_state_threshold": LOW_STATE_THRESHOLD,
        "high_state_threshold": HIGH_STATE_THRESHOLD,
        "low_mood_definition": "observed mood <= student median mood, fallback global median",
        "high_mood_definition": "observed mood > student median mood, fallback global median",
        "weekend_definition": "date day_of_week in Saturday/Sunday",
        "early_term_definition": "student normalized date rank <= 1/3",
        "late_term_definition": "student normalized date rank >= 2/3",
        "collapse_definition": "dominant policy action rate >= 0.8",
        "min_subgroup_rows": min_rows,
        "stress": "not available in current final_datasets transition schema",
    }


def compute_subgroup_analysis(
    model,
    split_df: pd.DataFrame,
    reward_col: str,
    split: str,
    behavior_policy,
    args,
) -> list:
    df = add_subgroup_columns(split_df)
    masks = build_subgroup_masks(df, args.subgroups)
    rows = []
    for subgroup, mask in masks.items():
        subset = df.loc[mask].copy()
        metrics = evaluate_model_on_subset(
            model,
            subset,
            reward_col,
            behavior_policy,
            args.min_subgroup_rows,
            n_bootstrap=args.n_bootstrap,
        )
        rows.append(
            {
                "reward_variant": reward_col,
                "split": split,
                "subgroup": subgroup,
                **metrics,
            }
        )
    return rows


def compute_student_analysis(
    model,
    split_df: pd.DataFrame,
    reward_col: str,
    split: str,
    behavior_policy,
    min_rows: int,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> list:
    rows = []
    df = split_df.copy()
    for student_id, subset in df.groupby("student_id"):
        metrics = evaluate_model_on_subset(
            model,
            subset.copy(),
            reward_col,
            behavior_policy,
            min_rows,
            n_bootstrap=n_bootstrap,
        )
        rows.append(
            {
                "reward_variant": reward_col,
                "split": split,
                "student_id": student_id,
                **metrics,
            }
        )
    return rows


def clean_for_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_for_json(val) for val in value]
    return value


def save_outputs(args, payload: dict, rows: list) -> None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    nested = {}
    for row in rows:
        nested.setdefault(row["reward_variant"], {}).setdefault(row["algo"], {})[
            row["split"]
        ] = row

    json_payload = {
        "notes": (
            "Saved best DQN / Double DQN model evaluation. WPDIS/WIS/DR are "
            "diagnostic robustness checks for the PDIS-selected models."
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_results_json": str(args.results_json),
        "gamma": payload.get("gamma", GAMMA),
        "seed": payload.get("seed"),
        "metrics": clean_for_json(nested),
    }

    if args.output_json.exists():
        with open(args.output_json) as f:
            prior = json.load(f)
        prior_metrics = prior.get("metrics", {})
        for variant, algos in nested.items():
            prior_metrics.setdefault(variant, {})
            for algo, splits in algos.items():
                prior_metrics[variant].setdefault(algo, {})
                prior_metrics[variant][algo].update(splits)
        json_payload["metrics"] = clean_for_json(prior_metrics)

    with open(args.output_json, "w") as f:
        json.dump(json_payload, f, indent=2)

    df = pd.DataFrame(rows)
    df["hidden_units"] = df["hidden_units"].apply(lambda h: "x".join(map(str, h)))

    if args.output_csv.exists():
        prior_df = pd.read_csv(args.output_csv)
        key_cols = ["reward_variant", "algo", "split"]
        prior_df = prior_df[
            ~prior_df.set_index(key_cols).index.isin(df.set_index(key_cols).index)
        ]
        df = pd.concat([prior_df, df], ignore_index=True)

    df = df.sort_values(["reward_variant", "algo", "split"]).reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\nSaved saved-model OPE metrics -> {args.output_json}")
    print(f"Saved flat saved-model OPE metrics -> {args.output_csv}")


def save_subgroup_outputs(args, payload: dict, subgroup_rows: list, student_rows: list) -> None:
    if subgroup_rows:
        args.subgroup_output_json.parent.mkdir(parents=True, exist_ok=True)
        nested = {}
        for row in subgroup_rows:
            nested.setdefault(row["reward_variant"], {}).setdefault(row["split"], {})[
                row["subgroup"]
            ] = row

        json_payload = {
            "notes": (
                "State-conditioned saved DQN / Double DQN analysis. Subgroup OPE "
                "uses filtered transition pseudo-episodes and should be read as "
                "diagnostic, not causal."
            ),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_results_json": str(args.results_json),
            "gamma": payload.get("gamma", GAMMA),
            "seed": payload.get("seed"),
            "thresholds": subgroup_threshold_metadata(args.min_subgroup_rows),
            "metrics": clean_for_json(nested),
        }
        with open(args.subgroup_output_json, "w") as f:
            json.dump(json_payload, f, indent=2)

        subgroup_df = pd.DataFrame(subgroup_rows)
        subgroup_df["hidden_units"] = subgroup_df["hidden_units"].apply(
            lambda h: "x".join(map(str, h)) if isinstance(h, list) else h
        )
        subgroup_df.to_csv(args.subgroup_output_csv, index=False)
        print(f"Saved subgroup analysis -> {args.subgroup_output_json}")
        print(f"Saved flat subgroup analysis -> {args.subgroup_output_csv}")

    if student_rows:
        student_df = pd.DataFrame(student_rows)
        student_df["hidden_units"] = student_df["hidden_units"].apply(
            lambda h: "x".join(map(str, h)) if isinstance(h, list) else h
        )
        student_df.to_csv(args.student_output_csv, index=False)
        print(f"Saved per-student analysis -> {args.student_output_csv}")


def print_subgroup_summary(subgroup_rows: list, student_rows: list) -> None:
    if subgroup_rows:
        print("\n================ SUBGROUP POLICY ANALYSIS ================")
        header = (
            f"{'reward':<15} {'split':<5} {'subgroup':<18} {'rows':>5} "
            f"{'pdis':>8} {'wpdis':>8} {'dr':>8} {'ess':>7} "
            f"{'match':>7} {'dom action':<18} {'dom%':>6}"
        )
        print(header)
        print("-" * len(header))
        for row in subgroup_rows:
            print(
                f"{row['reward_variant']:<15} {row['split']:<5} "
                f"{row['subgroup']:<18} {row['num_rows']:>5} "
                f"{row.get('pdis', float('nan')):>8.4f} "
                f"{row.get('weighted_pdis', float('nan')):>8.4f} "
                f"{row.get('dr', float('nan')):>8.4f} "
                f"{row.get('effective_sample_size', float('nan')):>7.2f} "
                f"{row['action_match']:>7.4f} "
                f"{str(row['dominant_policy_action']):<18} "
                f"{row['dominant_policy_action_rate']:>6.2f}"
            )

    if student_rows:
        worst = sorted(
            student_rows,
            key=lambda r: (
                r.get("reliable_support", False),
                r.get("weighted_pdis", float("nan"))
                if np.isfinite(r.get("weighted_pdis", float("nan")))
                else float("-inf"),
                r.get("action_match", 0.0)
                if np.isfinite(r.get("action_match", float("nan")))
                else 0.0,
            ),
        )[:5]
        print("\n================ LOWEST-SUPPORT / WORST STUDENT SNAPSHOT ================")
        for row in worst:
            print(
                f"{row['reward_variant']} {row['split']} {row['student_id']}: "
                f"rows={row['num_rows']} wpdis={row.get('weighted_pdis', float('nan')):.4f} "
                f"match={row['action_match']:.4f} "
                f"dominant={row['dominant_policy_action']} "
                f"dominant_rate={row['dominant_policy_action_rate']:.2f}"
            )


def fixed_double_dqn_ablation_config(payload: dict) -> dict:
    """Best validation-PDIS Double DQN hyperparameters on reward_dense."""
    return best_configs_per_algo(payload, "reward_dense", ["double_dqn"])["double_dqn"]


def run_double_dqn_reward_ablation(
    args,
    payload: dict,
    splits: dict,
    behavior_policy,
) -> list:
    """
    Train Double DQN per reward variant with fixed hyperparameters from the
    best dense-reward model, then evaluate PDIS and mood improvement on test.
    """
    config = fixed_double_dqn_ablation_config(payload)
    hp = {
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "target_update_interval": config["target_update_interval"],
        "hidden_units": config["hidden_units"],
    }
    algo = "double_dqn"
    n_steps = args.ablation_n_steps or payload.get("n_steps", 5000)
    n_steps_per_epoch = min(payload.get("n_steps_per_epoch", n_steps), n_steps)
    seed = payload.get("seed", 42)
    test_df = splits["test"]
    rows = []

    print(
        "\n=== Double DQN Reward Variant Ablation ===\n"
        f"Fixed hyperparameters from best reward_dense Double DQN: {hp}\n"
        f"Training steps: {n_steps}"
    )

    for reward_col in REWARD_VARIANTS:
        print(f"\nAblation: training Double DQN with reward_col={reward_col} ...")
        train_ds = build_mdp_dataset(splits["train"], reward_col)
        val_ds = build_mdp_dataset(splits["val"], reward_col)
        val_episodes = extract_episodes(splits["val"], reward_col)
        model, _ = train_and_eval(
            algo,
            hp,
            train_ds,
            val_ds,
            val_episodes,
            behavior_policy,
            n_steps,
            n_steps_per_epoch,
            args.device,
            seed,
        )
        metrics = evaluate_model_on_split(
            model,
            test_df,
            reward_col,
            behavior_policy,
            n_bootstrap=0,
        )
        rows.append(
            {
                "reward_variant": reward_col,
                "algo": algo,
                "n_steps": n_steps,
                "learning_rate": hp["learning_rate"],
                "batch_size": hp["batch_size"],
                "target_update_interval": hp["target_update_interval"],
                "hidden_units": "x".join(map(str, hp["hidden_units"])),
                "pdis": metrics["pdis"],
                "weighted_pdis": metrics["weighted_pdis"],
                "dr": metrics["dr"],
                "action_match": metrics["action_match"],
                "mood_improvement": metrics["mood_improvement"],
                "n_matched": metrics["mood_n_matched"],
            }
        )

    print(f"\n{'Reward Variant':<32} {'PDIS':>10} {'WPDIS':>10} {'Mood Δ':>10} {'n':>4}")
    print("-" * 70)
    for row in rows:
        mood = row["mood_improvement"]
        mood_s = f"{mood:.6f}" if np.isfinite(mood) else "N/A"
        print(
            f"{row['reward_variant']:<32} "
            f"{row['pdis']:>10.6f} "
            f"{row['weighted_pdis']:>10.6f} "
            f"{mood_s:>10} "
            f"{row['n_matched']:>4}"
        )

    args.ablation_output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.ablation_output_csv, index=False)
    json_payload = {
        "notes": (
            "Double DQN reward ablation: fixed hyperparameters from best "
            "validation-PDIS Double DQN on reward_dense; one fresh train "
            "per reward variant; test-split OPE."
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_results_json": str(args.results_json),
        "fixed_hyperparameters": {**hp, "algo": algo, "n_steps": n_steps},
        "rows": clean_for_json(rows),
    }
    with open(args.ablation_output_json, "w") as f:
        json.dump(json_payload, f, indent=2)

    print(f"\nSaved ablation results -> {args.ablation_output_csv}")
    print(f"Saved ablation results -> {args.ablation_output_json}")
    return rows


def print_summary(rows: list) -> None:
    print("\n================ SAVED DQN MODEL OPE ================")
    header = (
        f"{'reward':<22} {'algo':<11} {'split':<6} {'pdis':>9} "
        f"{'wpdis':>9} {'wis':>9} {'dr':>9} {'ess':>8} {'a-match':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['reward_variant']:<22} {row['algo']:<11} {row['split']:<6} "
            f"{row['pdis']:>9.4f} {row['weighted_pdis']:>9.4f} "
            f"{row['trajectory_wis']:>9.4f} {row['dr']:>9.4f} "
            f"{row['effective_sample_size']:>8.2f} {row['action_match']:>8.4f}"
        )


def main() -> None:
    args = parse_args()
    payload = load_search_payload(args.results_json)
    splits = load_splits(args.data_dir, args.basename)
    behavior_policy = fit_behavior_policy(splits["train"], payload.get("seed", 42))

    if args.reward_ablation:
        run_double_dqn_reward_ablation(args, payload, splits, behavior_policy)
        return

    best_per_variant = payload["best_per_variant"]
    variants = args.reward_variants or list(best_per_variant.keys())
    rows = []
    subgroup_rows = []
    student_rows = []
    seed = payload.get("seed", 42)

    for reward_col in variants:
        if reward_col not in best_per_variant:
            raise ValueError(f"{reward_col} not found in best_per_variant")

        if args.best_per_algo:
            configs = list(best_configs_per_algo(payload, reward_col, args.algos).values())
        else:
            configs = [best_per_variant[reward_col]]

        for best in configs:
            model = ensure_model(
                best,
                reward_col,
                splits,
                behavior_policy,
                payload,
                args.device,
                seed,
                args.retrain_missing,
            )

            for split in args.splits:
                split_df = splits[split]
                metrics = evaluate_model_on_split(
                    model,
                    split_df,
                    reward_col,
                    behavior_policy,
                    n_bootstrap=args.n_bootstrap,
                )
                metadata = {
                    "reward_variant": reward_col,
                    "algo": best["algo"],
                    "split": split,
                    "learning_rate": best["learning_rate"],
                    "batch_size": best["batch_size"],
                    "target_update_interval": best["target_update_interval"],
                    "hidden_units": best["hidden_units"],
                }
                row = {
                    **metadata,
                    **metrics,
                    **action_rate_columns(model, split_df),
                }
                rows.append(row)

                if args.subgroup_analysis:
                    for subgroup_row in compute_subgroup_analysis(
                        model,
                        split_df,
                        reward_col,
                        split,
                        behavior_policy,
                        args,
                    ):
                        subgroup_rows.append({**metadata, **subgroup_row})

                if args.student_analysis:
                    for student_row in compute_student_analysis(
                        model,
                        split_df,
                        reward_col,
                        split,
                        behavior_policy,
                        args.min_subgroup_rows,
                        n_bootstrap=args.n_bootstrap,
                    ):
                        student_rows.append({**metadata, **student_row})

    print_summary(rows)
    save_outputs(args, payload, rows)
    print_subgroup_summary(subgroup_rows, student_rows)
    save_subgroup_outputs(args, payload, subgroup_rows, student_rows)


if __name__ == "__main__":
    main()
