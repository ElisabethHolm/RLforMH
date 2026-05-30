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
    GAMMA,
    compute_doubly_robust_estimate,
    compute_importance_weight_diagnostics,
    compute_pdis_estimate,
    compute_trajectory_is_estimate,
    compute_weighted_is_estimate,
    compute_weighted_pdis_estimate,
)
from hyperparameter_search_dqn import (  # noqa: E402
    ALGO_CONFIGS,
    DATASET_BASENAME,
    DEFAULT_DATA_DIR,
    MODEL_DIR,
    RESULTS_JSON,
    DQNPolicyWrapper,
    build_mdp_dataset,
    direct_method_v0,
    extract_episodes,
    fit_behavior_policy,
    load_splits,
    mood_improvement,
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
    return parser.parse_args()


def load_search_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing DQN search results: {path}")
    with open(path) as f:
        return json.load(f)


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


def evaluate_model_on_split(model, split_df, reward_col: str, behavior_policy) -> dict:
    ds = build_mdp_dataset(split_df, reward_col)
    episodes = extract_episodes(split_df, reward_col)
    policy = DQNPolicyWrapper(model)
    weight_diagnostics = compute_importance_weight_diagnostics(
        episodes, policy, behavior_policy
    )

    return {
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
        "mood_improvement": mood_improvement(episodes, policy),
        "direct_method_v0": direct_method_v0(episodes, policy),
    }


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
        nested.setdefault(row["reward_variant"], {})[row["split"]] = row

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
    with open(args.output_json, "w") as f:
        json.dump(json_payload, f, indent=2)

    df = pd.DataFrame(rows)
    df["hidden_units"] = df["hidden_units"].apply(lambda h: "x".join(map(str, h)))
    df.to_csv(args.output_csv, index=False)

    print(f"\nSaved saved-model OPE metrics -> {args.output_json}")
    print(f"Saved flat saved-model OPE metrics -> {args.output_csv}")


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

    best_per_variant = payload["best_per_variant"]
    variants = args.reward_variants or list(best_per_variant.keys())
    rows = []

    for reward_col in variants:
        if reward_col not in best_per_variant:
            raise ValueError(f"{reward_col} not found in best_per_variant")

        best = best_per_variant[reward_col]
        model_path = MODEL_DIR / f"dqn_best_{reward_col}.d3"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing saved model: {model_path}")

        train_ds = build_mdp_dataset(splits["train"], reward_col)
        model = build_model(best, train_ds, args.device)
        model.load_model(str(model_path))

        for split in args.splits:
            metrics = evaluate_model_on_split(
                model,
                splits[split],
                reward_col,
                behavior_policy,
            )
            rows.append(
                {
                    "reward_variant": reward_col,
                    "algo": best["algo"],
                    "split": split,
                    "learning_rate": best["learning_rate"],
                    "batch_size": best["batch_size"],
                    "target_update_interval": best["target_update_interval"],
                    "hidden_units": best["hidden_units"],
                    **metrics,
                }
            )

    print_summary(rows)
    save_outputs(args, payload, rows)


if __name__ == "__main__":
    main()
