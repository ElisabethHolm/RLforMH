"""
Hyperparameter search for discrete IQL and AWAC on chronological StudentLife splits.

Run from repo root:
    python algorithms/hyperparameter_search_iql_awac.py
    python algorithms/hyperparameter_search_iql_awac.py --quick
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from discrete_awac import save_awac_checkpoint, train_discrete_awac
from discrete_iql import save_iql_checkpoint, train_discrete_iql
from evaluate_policies import (
    compute_doubly_robust_estimate,
    compute_importance_weight_diagnostics,
    compute_mood_improvement_stats,
    compute_pdis_estimate,
    compute_trajectory_is_estimate,
    compute_weighted_is_estimate,
    compute_weighted_pdis_estimate,
)
from offline_rl_common import (
    DATASET_BASENAME,
    DEFAULT_DATA_DIR,
    GAMMA,
    MODEL_DIR,
    REWARD_VARIANTS,
    SEED,
    STATE_COLS,
    action_match_rate,
    extract_episodes,
    fit_behavior_policy,
    load_splits,
)
from ope_uncertainty import attach_policy_uncertainty

RESULTS_JSON = MODEL_DIR / "iql_awac_hparam_search_results.json"
RESULTS_CSV = MODEL_DIR / "iql_awac_hparam_search_results.csv"

ALGOS = ["iql", "awac"]

IQL_FULL_GRID = {
    "expectile": [0.7, 0.9],
    "temperature": [1.0, 3.0],
    "critic_lr": [1e-4],
    "batch_size": [64],
    "hidden_units": [[256, 256]],
    "target_update_interval": [100],
}
IQL_QUICK_GRID = {
    "expectile": [0.7],
    "temperature": [3.0],
    "critic_lr": [1e-4],
    "batch_size": [64],
    "hidden_units": [[256, 256]],
    "target_update_interval": [100],
}

AWAC_FULL_GRID = {
    "expectile": [0.7],
    "lam": [0.5, 1.0],
    "max_weight": [100.0],
    "critic_lr": [1e-4],
    "actor_lr": [1e-4],
    "batch_size": [64],
    "hidden_units": [[256, 256]],
    "target_update_interval": [100],
}
AWAC_QUICK_GRID = {
    "expectile": [0.7],
    "lam": [1.0],
    "max_weight": [100.0],
    "critic_lr": [1e-4],
    "actor_lr": [1e-4],
    "batch_size": [64],
    "hidden_units": [[256, 256]],
    "target_update_interval": [100],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid-search discrete IQL / AWAC (reward_dense first)."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--basename", type=str, default=DATASET_BASENAME)
    parser.add_argument("--algos", nargs="+", choices=ALGOS, default=ALGOS)
    parser.add_argument(
        "--reward-variants",
        nargs="+",
        choices=REWARD_VARIANTS,
        default=["reward_dense"],
    )
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=300,
        help="Bootstrap replicates for val OPE CIs (0=skip).",
    )
    return parser.parse_args()


def build_hp_configs(algo: str, quick: bool) -> list[dict]:
    grid = IQL_QUICK_GRID if quick else IQL_FULL_GRID
    if algo == "awac":
        grid = AWAC_QUICK_GRID if quick else AWAC_FULL_GRID
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


def direct_method_v0(episodes: list, policy) -> float:
    values = []
    for ep in episodes:
        q = policy.q_values(ep["states"][[0]])
        values.append(float(np.max(q[0])))
    return float(np.mean(values)) if values else float("nan")


def train_and_eval(
    algo: str,
    hp: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_episodes: list,
    behavior_policy,
    n_steps: int,
    seed: int,
    n_bootstrap: int,
) -> tuple[dict, object]:
    if algo == "iql":
        q_net, v_net, config = train_discrete_iql(
            train_df,
            hp.get("reward_col", "reward_dense"),
            n_steps=n_steps,
            batch_size=hp["batch_size"],
            critic_lr=hp["critic_lr"],
            expectile=hp["expectile"],
            temperature=hp["temperature"],
            hidden_units=hp["hidden_units"],
            target_update_interval=hp["target_update_interval"],
            seed=seed,
        )
        from discrete_iql import IQLPolicyWrapper

        policy = IQLPolicyWrapper(q_net, v_net, temperature=hp["temperature"])
        artifact = (q_net, v_net, config)
    else:
        policy_net, q_net, v_net, config = train_discrete_awac(
            train_df,
            hp.get("reward_col", "reward_dense"),
            n_steps=n_steps,
            batch_size=hp["batch_size"],
            critic_lr=hp["critic_lr"],
            actor_lr=hp["actor_lr"],
            expectile=hp["expectile"],
            lam=hp["lam"],
            max_weight=hp["max_weight"],
            hidden_units=hp["hidden_units"],
            target_update_interval=hp["target_update_interval"],
            seed=seed,
        )
        from discrete_awac import AWACPolicyWrapper

        policy = AWACPolicyWrapper(policy_net, q_net)
        artifact = (policy_net, q_net, v_net, config)

    val_states = val_df[STATE_COLS].fillna(0.0).to_numpy("float32")
    val_actions = val_df["action"].to_numpy("int64")

    weight_diagnostics = compute_importance_weight_diagnostics(
        val_episodes, policy, behavior_policy
    )
    mood_imp, mood_n = compute_mood_improvement_stats(val_episodes, policy)

    metrics = {
        "action_match": action_match_rate(val_states, val_actions, policy),
        "pdis": float(compute_pdis_estimate(val_episodes, policy, behavior_policy)),
        "weighted_pdis": float(
            compute_weighted_pdis_estimate(val_episodes, policy, behavior_policy)
        ),
        "trajectory_is": float(
            compute_trajectory_is_estimate(val_episodes, policy, behavior_policy)
        ),
        "trajectory_wis": float(
            compute_weighted_is_estimate(val_episodes, policy, behavior_policy)
        ),
        "dr": float(compute_doubly_robust_estimate(val_episodes, policy, behavior_policy)),
        "effective_sample_size": weight_diagnostics["effective_sample_size"],
        "weight_mean": weight_diagnostics["weight_mean"],
        "weight_max": weight_diagnostics["weight_max"],
        "mood_improvement": mood_imp,
        "mood_n_matched": mood_n,
        "direct_method_v0": direct_method_v0(val_episodes, policy),
    }
    result_row = {"algo": algo, **hp, **metrics}
    attach_policy_uncertainty(
        result_row, val_episodes, policy, behavior_policy, n_bootstrap=n_bootstrap
    )
    return result_row, artifact


def _is_better(candidate: dict, incumbent: dict) -> bool:
    c, i = candidate.get("pdis"), incumbent.get("pdis")
    if not np.isfinite(c):
        return False
    if not np.isfinite(i):
        return True
    return c > i


def _clean(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_checkpoint(algo: str, reward_col: str, artifact: object) -> Path:
    path = MODEL_DIR / f"{algo}_best_{reward_col}.pt"
    if algo == "iql":
        q_net, v_net, config = artifact
        save_iql_checkpoint(path, q_net, v_net, config)
    else:
        policy_net, q_net, v_net, config = artifact
        save_awac_checkpoint(path, policy_net, q_net, v_net, config)
    return path


def main() -> None:
    args = parse_args()
    n_steps = args.n_steps if args.n_steps is not None else (500 if args.quick else 5000)

    splits = load_splits(args.data_dir, args.basename)
    behavior_policy = fit_behavior_policy(splits["train"], args.seed)

    all_results = []
    best_per_algo_variant: dict[tuple[str, str], dict] = {}
    best_artifacts: dict[tuple[str, str], object] = {}

    for reward_col in args.reward_variants:
        train_df = splits["train"]
        val_df = splits["val"]
        val_episodes = extract_episodes(val_df, reward_col)

        for algo in args.algos:
            hp_configs = build_hp_configs(algo, args.quick)
            if args.max_configs is not None:
                hp_configs = hp_configs[: args.max_configs]

            for hp in hp_configs:
                hp = {**hp, "reward_col": reward_col}
                print(f"Training {reward_col} | {algo} | {hp}")
                row, artifact = train_and_eval(
                    algo,
                    hp,
                    train_df,
                    val_df,
                    val_episodes,
                    behavior_policy,
                    n_steps,
                    args.seed,
                    args.n_bootstrap,
                )
                print(
                    f"  pdis={row.get('pdis', float('nan')):.4f} "
                    f"wpdis={row.get('weighted_pdis', float('nan')):.4f} "
                    f"dr={row.get('dr', float('nan')):.4f} "
                    f"match={row.get('action_match', float('nan')):.4f}"
                )
                all_results.append(row)

                key = (algo, reward_col)
                if key not in best_per_algo_variant or _is_better(row, best_per_algo_variant[key]):
                    best_per_algo_variant[key] = row
                    best_artifacts[key] = artifact

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_steps": n_steps,
        "gamma": GAMMA,
        "seed": args.seed,
        "results": [{k: _clean(v) for k, v in r.items()} for r in all_results],
        "best_per_algo_variant": {
            f"{algo}_{reward}": {k: _clean(v) for k, v in row.items()}
            for (algo, reward), row in best_per_algo_variant.items()
        },
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
    print(f"Saved {RESULTS_JSON.relative_to(MODEL_DIR.parent)}")
    print(f"Saved {RESULTS_CSV.relative_to(MODEL_DIR.parent)}")

    if not args.no_save_models:
        for (algo, reward_col), artifact in best_artifacts.items():
            path = save_checkpoint(algo, reward_col, artifact)
            print(f"Saved best {algo} ({reward_col}) -> {path.relative_to(MODEL_DIR.parent)}")


if __name__ == "__main__":
    main()
