"""
Evaluate saved best discrete IQL / AWAC checkpoints with sequential OPE.

Run from repo root (project venv active):
    python algorithms/evaluate_iql_awac_models.py
    python algorithms/evaluate_iql_awac_models.py --splits test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from discrete_awac import load_awac_policy
from discrete_iql import load_iql_policy
from evaluate_policies import (
    ACTION_NAMES,
    N_ACTIONS,
    compute_doubly_robust_estimate,
    compute_importance_weight_diagnostics,
    compute_mood_improvement_stats,
    compute_pdis_estimate,
    compute_trajectory_is_estimate,
    compute_weighted_is_estimate,
    compute_weighted_pdis_estimate,
)
from hyperparameter_search_iql_awac import RESULTS_JSON
from offline_rl_common import (
    DATASET_BASENAME,
    DEFAULT_DATA_DIR,
    GAMMA,
    MODEL_DIR,
    REWARD_VARIANTS,
    STATE_COLS,
    action_match_rate,
    extract_episodes,
    fit_behavior_policy,
    load_splits,
)
from ope_uncertainty import DEFAULT_BOOTSTRAP_SAMPLES, attach_policy_uncertainty

OUT_JSON = MODEL_DIR / "iql_awac_ope_metrics.json"
OUT_CSV = MODEL_DIR / "iql_awac_ope_metrics.csv"
ALGOS = ["iql", "awac"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved best IQL / AWAC models with OPE."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--basename", type=str, default=DATASET_BASENAME)
    parser.add_argument(
        "--results-json",
        type=Path,
        default=RESULTS_JSON,
        help="IQL/AWAC hyperparameter search JSON.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    parser.add_argument(
        "--reward-variants",
        nargs="+",
        choices=REWARD_VARIANTS,
        default=["reward_dense"],
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        choices=ALGOS,
        default=ALGOS,
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--output-json", type=Path, default=OUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=OUT_CSV)
    return parser.parse_args()


def load_search_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing IQL/AWAC search results: {path}")
    with open(path) as f:
        return json.load(f)


def resolve_checkpoint(algo: str, reward_col: str) -> Path:
    path = MODEL_DIR / f"{algo}_best_{reward_col}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {path}. Run hyperparameter_search_iql_awac.py first."
        )
    return path


def load_policy(algo: str, path: Path):
    if algo == "iql":
        return load_iql_policy(path)
    return load_awac_policy(path)


def direct_method_v0(episodes: list, policy) -> float:
    values = []
    for ep in episodes:
        q = policy.q_values(ep["states"][[0]])
        values.append(float(np.max(q[0])))
    return float(np.mean(values)) if values else float("nan")


def action_rate_columns(policy, split_df: pd.DataFrame) -> dict:
    states = split_df[STATE_COLS].fillna(0.0).to_numpy("float32")
    actions = policy.predict(states)
    counts = np.bincount(actions.astype("int64"), minlength=N_ACTIONS)
    total = int(counts.sum())
    if total == 0:
        return {f"action_{name}": float("nan") for name in ACTION_NAMES.values()}
    return {
        f"action_{ACTION_NAMES[a]}": float(c / total)
        for a, c in enumerate(counts)
    }


def evaluate_policy_on_split(
    policy,
    split_df: pd.DataFrame,
    reward_col: str,
    behavior_policy,
    n_bootstrap: int,
) -> dict:
    episodes = extract_episodes(split_df, reward_col)
    states = split_df[STATE_COLS].fillna(0.0).to_numpy("float32")
    logged_actions = split_df["action"].to_numpy("int64")
    weight_diagnostics = compute_importance_weight_diagnostics(
        episodes, policy, behavior_policy
    )
    mood_imp, mood_n = compute_mood_improvement_stats(episodes, policy)

    metrics = {
        "num_rows": int(len(split_df)),
        "num_episodes": int(len(episodes)),
        "action_match": action_match_rate(states, logged_actions, policy),
        "pdis": float(compute_pdis_estimate(episodes, policy, behavior_policy)),
        "weighted_pdis": float(
            compute_weighted_pdis_estimate(episodes, policy, behavior_policy)
        ),
        "trajectory_is": float(
            compute_trajectory_is_estimate(episodes, policy, behavior_policy)
        ),
        "trajectory_wis": float(
            compute_weighted_is_estimate(episodes, policy, behavior_policy)
        ),
        "dr": float(compute_doubly_robust_estimate(episodes, policy, behavior_policy)),
        "effective_sample_size": weight_diagnostics["effective_sample_size"],
        "weight_mean": weight_diagnostics["weight_mean"],
        "weight_max": weight_diagnostics["weight_max"],
        "mood_improvement": mood_imp,
        "mood_n_matched": mood_n,
        "direct_method_v0": direct_method_v0(episodes, policy),
    }
    attach_policy_uncertainty(
        metrics, episodes, policy, behavior_policy, n_bootstrap=n_bootstrap
    )
    for suffix in ("_se", "_ci_low", "_ci_high"):
        ci_key = f"match_rate{suffix}"
        am_key = f"action_match{suffix}"
        if ci_key in metrics:
            metrics[am_key] = metrics[ci_key]
    return metrics


def clean_for_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    return value


def print_summary(rows: list) -> None:
    print("\n================ SAVED IQL / AWAC OPE ================")
    header = (
        f"{'reward':<22} {'algo':<6} {'split':<6} {'pdis':>9} "
        f"{'wpdis':>9} {'dr':>9} {'ess':>8} {'a-match':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['reward_variant']:<22} {row['algo']:<6} {row['split']:<6} "
            f"{row.get('pdis', float('nan')):>9.4f} "
            f"{row.get('weighted_pdis', float('nan')):>9.4f} "
            f"{row.get('dr', float('nan')):>9.4f} "
            f"{row.get('effective_sample_size', float('nan')):>8.2f} "
            f"{row.get('action_match', float('nan')):>8.4f}"
        )


def main() -> None:
    args = parse_args()
    payload = load_search_payload(args.results_json)
    splits = load_splits(args.data_dir, args.basename)
    behavior_policy = fit_behavior_policy(splits["train"], payload.get("seed", 42))
    best = payload.get("best_per_algo_variant", {})

    rows = []
    for reward_col in args.reward_variants:
        for algo in args.algos:
            key = f"{algo}_{reward_col}"
            meta = best.get(key, {})
            ckpt = resolve_checkpoint(algo, reward_col)
            policy = load_policy(algo, ckpt)
            print(f"Loaded {algo} ({reward_col}) from {ckpt.name}")

            for split in args.splits:
                split_df = splits[split]
                metrics = evaluate_policy_on_split(
                    policy,
                    split_df,
                    reward_col,
                    behavior_policy,
                    args.n_bootstrap,
                )
                row = {
                    "reward_variant": reward_col,
                    "algo": algo,
                    "split": split,
                    "expectile": meta.get("expectile"),
                    "temperature": meta.get("temperature"),
                    "lam": meta.get("lam"),
                    "max_weight": meta.get("max_weight"),
                    "critic_lr": meta.get("critic_lr"),
                    "actor_lr": meta.get("actor_lr"),
                    "batch_size": meta.get("batch_size"),
                    "hidden_units": meta.get("hidden_units"),
                    **metrics,
                    **action_rate_columns(policy, split_df),
                }
                rows.append(row)

    print_summary(rows)

    nested = {}
    for row in rows:
        nested.setdefault(row["reward_variant"], {}).setdefault(row["algo"], {})[
            row["split"]
        ] = clean_for_json(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "notes": "Saved best discrete IQL / AWAC evaluation on chronological splits.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_results_json": str(args.results_json),
        "gamma": payload.get("gamma", GAMMA),
        "seed": payload.get("seed"),
        "metrics": nested,
    }
    with open(args.output_json, "w") as f:
        json.dump(json_payload, f, indent=2)

    df = pd.DataFrame(rows)
    if "hidden_units" in df.columns:
        df["hidden_units"] = df["hidden_units"].apply(
            lambda h: "x".join(map(str, h)) if isinstance(h, list) else h
        )
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved -> {args.output_json}")
    print(f"Saved -> {args.output_csv}")


if __name__ == "__main__":
    main()
