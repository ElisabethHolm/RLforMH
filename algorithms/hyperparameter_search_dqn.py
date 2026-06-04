"""
DQN / Double DQN hyperparameter search for StudentLife offline RL.

Grid-searches Discrete DQN and Double DQN over the three reward variants
(reward_sparse, reward_dense, reward_observed_only) using the chronological
splits in final_datasets/. Each config is evaluated two ways:

  A) d3rlpy built-in offline evaluators on the validation split
     (TD error, discrete action match, average value estimation)
  B) the project's offline policy evaluation (OPE):
     - PDIS (per-decision importance sampling) against a behavior-cloning
       logging policy fit on the train split
     - weighted/self-normalized IS variants and doubly robust OPE diagnostics
     - matched-action next-day mood improvement
     - direct-method V(s0) = mean over episodes of max_a Q(s0, a)

The best config per reward variant is selected by PDIS, with the built-in
metrics and mood improvement reported alongside.

Run (from repo root, with venv active):
    python algorithms/hyperparameter_search_dqn.py
    python algorithms/hyperparameter_search_dqn.py --quick
    python algorithms/hyperparameter_search_dqn.py --reward-variants reward_dense --algos dqn

Note: reward_sparse / reward_observed_only are NaN on most rows (mood is rarely
observed) and are filled with 0 for training. PDIS for those variants can be
near-degenerate because almost every reward is 0; that is expected and is why
the built-in metrics and mood improvement are logged alongside it.
"""

import argparse
import itertools
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import d3rlpy
from d3rlpy.algos import DQNConfig, DoubleDQNConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.logging import NoopAdapterFactory
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.metrics import (
    AverageValueEstimationEvaluator,
    DiscreteActionMatchEvaluator,
    TDErrorEvaluator,
)
# Make the sibling evaluate_policies module importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_policies import (  # noqa: E402
    EPSILON,
    GAMMA,
    N_ACTIONS,
    compute_doubly_robust_estimate,
    compute_importance_weight_diagnostics,
    compute_pdis_estimate,
    compute_trajectory_is_estimate,
    compute_weighted_is_estimate,
    compute_weighted_pdis_estimate,
)
from offline_rl_common import (  # noqa: E402
    BehaviorPolicy,
    STATE_COLS,
    build_mdp_dataset,
    extract_episodes,
    fit_behavior_policy,
    load_splits,
)

# Keep the d3rlpy console output quiet during the (potentially large) sweep.
# d3rlpy logs through structlog, so filter that as well as stdlib logging.
logging.getLogger("d3rlpy").setLevel(logging.WARNING)
try:
    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
except Exception:  # pragma: no cover - structlog is a d3rlpy dependency
    pass

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "final_datasets"
DATASET_BASENAME = "daily_studentlife"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_JSON = MODEL_DIR / "dqn_hparam_search_results.json"
RESULTS_CSV = MODEL_DIR / "dqn_hparam_search_results.csv"

REWARD_VARIANTS = ["reward_sparse", "reward_dense", "reward_observed_only"]
ALGOS = ["dqn", "double_dqn"]
ALGO_CONFIGS = {"dqn": DQNConfig, "double_dqn": DoubleDQNConfig}

# Grid axes (each is a list of candidate values).
FULL_GRID = {
    "learning_rate": [1e-4, 6.25e-5],
    "batch_size": [32, 64],
    "target_update_interval": [1000, 8000],
    "hidden_units": [[64, 64], [256, 256]],
}
QUICK_GRID = {
    "learning_rate": [1e-4],
    "batch_size": [64],
    "target_update_interval": [1000],
    "hidden_units": [[64, 64]],
}

SEED = 42

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid-search DQN / Double DQN over the StudentLife reward variants."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the <basename>.{train,val,test}.csv splits.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=DATASET_BASENAME,
        help="Split file basename (default: daily_studentlife).",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        choices=ALGOS,
        default=ALGOS,
        help="Which algorithms to include in the sweep.",
    )
    parser.add_argument(
        "--reward-variants",
        nargs="+",
        choices=REWARD_VARIANTS,
        default=REWARD_VARIANTS,
        help="Which reward columns to train and evaluate on.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Gradient steps per run (default: 5000, or 500 with --quick).",
    )
    parser.add_argument(
        "--n-steps-per-epoch",
        type=int,
        default=None,
        help="Steps per epoch (default: equal to --n-steps, i.e. a single epoch).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device passed to d3rlpy (e.g. cpu, cuda:0).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shrink the grid to one value per axis and lower n_steps for a smoke test.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Cap the number of (algo, hyperparameter) configs evaluated per reward variant.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for d3rlpy, numpy, and the behavior-cloning policy.",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Skip writing the best model per reward variant to disk.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Policy wrappers
# ---------------------------------------------------------------------------


class DQNPolicyWrapper:
    """Uniform OPE interface around a trained d3rlpy Discrete (Double) DQN."""

    name = "dqn"

    def __init__(self, model):
        self._model = model

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model.predict(states)

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        """Epsilon-greedy softening so importance weights stay finite."""
        actions = self.predict(states)
        probs = np.full((len(states), N_ACTIONS), EPSILON / N_ACTIONS)
        probs[np.arange(len(states)), actions] += (1.0 - EPSILON)
        return probs

    def q_values(self, states: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                self._model.predict_value(
                    states, np.full(len(states), a, dtype="int64")
                )
                for a in range(N_ACTIONS)
            ]
        )


# ---------------------------------------------------------------------------
# OPE estimators (IS/DR estimators are imported; these are NaN-aware variants)
# ---------------------------------------------------------------------------


def mood_improvement_stats(episodes: list, policy) -> tuple[float, int]:
    """Mean next-day mood delta on matched-action steps with observed mood."""
    deltas = []
    for ep in episodes:
        predicted = policy.predict(ep["states"])
        matched = np.where(predicted == ep["actions"])[0]
        for i in matched:
            delta = ep["next_moods"][i] - ep["mood"][i]
            if np.isfinite(delta):
                deltas.append(float(delta))
    if not deltas:
        return float("nan"), 0
    return float(np.mean(deltas)), len(deltas)


def mood_improvement(episodes: list, policy) -> float:
    """Mean next-day mood delta over steps where the policy matches the log."""
    return mood_improvement_stats(episodes, policy)[0]


def direct_method_v0(episodes: list, policy: DQNPolicyWrapper) -> float:
    """Mean over episodes of max_a Q(s0, a)."""
    values = []
    for ep in episodes:
        q = policy.q_values(ep["states"][[0]])
        values.append(float(np.max(q[0])))
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def build_hp_configs(grid: dict) -> list:
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def build_run_configs(algos: list, hp_configs: list, max_configs) -> list:
    combos = [
        {"algo": algo, "hp": hp}
        for algo, hp in itertools.product(algos, hp_configs)
    ]
    if max_configs is not None:
        combos = combos[:max_configs]
    return combos


def train_and_eval(
    algo: str,
    hp: dict,
    train_ds: MDPDataset,
    val_ds: MDPDataset,
    val_episodes: list,
    behavior_policy: BehaviorPolicy,
    n_steps: int,
    n_steps_per_epoch: int,
    device: str,
    seed: int,
):
    """Train one (algo, hp) config and return (model, metrics dict)."""
    d3rlpy.seed(seed)
    np.random.seed(seed)

    config = ALGO_CONFIGS[algo](
        batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
        gamma=GAMMA,
        target_update_interval=hp["target_update_interval"],
        encoder_factory=VectorEncoderFactory(hidden_units=hp["hidden_units"]),
    )
    model = config.create(device=device)

    history = model.fit(
        train_ds,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        show_progress=False,
        logger_adapter=NoopAdapterFactory(),
        evaluators={
            "td_error": TDErrorEvaluator(episodes=val_ds.episodes),
            "action_match": DiscreteActionMatchEvaluator(episodes=val_ds.episodes),
            "value": AverageValueEstimationEvaluator(episodes=val_ds.episodes),
        },
    )
    last_metrics = history[-1][1] if history else {}

    policy = DQNPolicyWrapper(model)
    weight_diagnostics = compute_importance_weight_diagnostics(
        val_episodes, policy, behavior_policy
    )
    metrics = {
        "td_error": float(last_metrics.get("td_error", float("nan"))),
        "action_match": float(last_metrics.get("action_match", float("nan"))),
        "value": float(last_metrics.get("value", float("nan"))),
        "loss": float(last_metrics.get("loss", float("nan"))),
        "pdis": float(compute_pdis_estimate(val_episodes, policy, behavior_policy)),
        "trajectory_is": float(
            compute_trajectory_is_estimate(val_episodes, policy, behavior_policy)
        ),
        "trajectory_wis": float(
            compute_weighted_is_estimate(val_episodes, policy, behavior_policy)
        ),
        "weighted_pdis": float(
            compute_weighted_pdis_estimate(val_episodes, policy, behavior_policy)
        ),
        "dr": float(
            compute_doubly_robust_estimate(val_episodes, policy, behavior_policy)
        ),
        "effective_sample_size": weight_diagnostics["effective_sample_size"],
        "weight_mean": weight_diagnostics["weight_mean"],
        "weight_max": weight_diagnostics["weight_max"],
        "nonzero_weight_episodes": weight_diagnostics["nonzero_weight_episodes"],
        "mood_improvement": mood_improvement(val_episodes, policy),
        "direct_method_v0": direct_method_v0(val_episodes, policy),
    }
    return model, metrics


def run_search(args) -> dict:
    grid = QUICK_GRID if args.quick else FULL_GRID
    hp_configs = build_hp_configs(grid)
    run_configs = build_run_configs(args.algos, hp_configs, args.max_configs)

    n_steps = args.n_steps if args.n_steps is not None else (500 if args.quick else 5000)
    n_steps_per_epoch = args.n_steps_per_epoch or n_steps
    n_steps_per_epoch = min(n_steps_per_epoch, n_steps)

    splits = load_splits(args.data_dir, args.basename)
    behavior_policy = fit_behavior_policy(splits["train"], args.seed)

    total = len(args.reward_variants) * len(run_configs)
    print(
        f"Reward variants: {args.reward_variants}\n"
        f"Algos: {args.algos} | hp configs/algo: {len(hp_configs)} | "
        f"configs/variant: {len(run_configs)} | total runs: {total}\n"
        f"n_steps={n_steps} (per_epoch={n_steps_per_epoch}) | device={args.device}\n"
    )

    all_results = []
    best_per_variant = {}
    best_models = {}
    run_idx = 0

    for reward_col in args.reward_variants:
        train_ds = build_mdp_dataset(splits["train"], reward_col)
        val_ds = build_mdp_dataset(splits["val"], reward_col)
        val_episodes = extract_episodes(splits["val"], reward_col)

        for cfg in run_configs:
            run_idx += 1
            algo, hp = cfg["algo"], cfg["hp"]
            print(
                f"[{run_idx}/{total}] {reward_col} | {algo} | "
                f"lr={hp['learning_rate']:g} bs={hp['batch_size']} "
                f"tui={hp['target_update_interval']} hidden={hp['hidden_units']}"
            )
            model, metrics = train_and_eval(
                algo,
                hp,
                train_ds,
                val_ds,
                val_episodes,
                behavior_policy,
                n_steps,
                n_steps_per_epoch,
                args.device,
                args.seed,
            )
            print(
                f"      pdis={metrics['pdis']:.4f} "
                f"wpdis={metrics['weighted_pdis']:.4f} "
                f"dr={metrics['dr']:.4f} "
                f"mood_delta={metrics['mood_improvement']:.4f} "
                f"action_match={metrics['action_match']:.4f} "
                f"td_error={metrics['td_error']:.4f}"
            )

            result = {
                "reward_variant": reward_col,
                "algo": algo,
                "learning_rate": hp["learning_rate"],
                "batch_size": hp["batch_size"],
                "target_update_interval": hp["target_update_interval"],
                "hidden_units": hp["hidden_units"],
                **metrics,
            }
            all_results.append(result)

            current_best = best_per_variant.get(reward_col)
            if current_best is None or _is_better(result, current_best):
                best_per_variant[reward_col] = result
                best_models[reward_col] = model

    return {
        "n_steps": n_steps,
        "n_steps_per_epoch": n_steps_per_epoch,
        "gamma": GAMMA,
        "seed": args.seed,
        "results": all_results,
        "best_per_variant": best_per_variant,
        "_best_models": best_models,
    }


def _is_better(candidate: dict, incumbent: dict) -> bool:
    """Higher PDIS wins; NaN PDIS always loses to a finite one."""
    c, i = candidate["pdis"], incumbent["pdis"]
    if not np.isfinite(c):
        return False
    if not np.isfinite(i):
        return True
    return c > i


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _clean(value):
    """Replace non-finite floats with None for valid JSON."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_results(payload: dict, save_models: bool) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_models = payload.pop("_best_models", {})

    json_payload = {
        "notes": (
            "DQN / Double DQN grid search over StudentLife reward variants. "
            "Built-in metrics (td_error, action_match, value) computed on the val "
            "split; OPE (pdis, weighted_pdis, trajectory_is, trajectory_wis, dr, "
            "mood_improvement, direct_method_v0) computed on val with a "
            "behavior-cloning logging policy. Best config per variant = max PDIS. "
            "WPDIS/WIS/DR are diagnostic robustness checks, not the selection "
            "metric. reward_sparse / reward_observed_only are NaN on most rows "
            "and filled with 0, so their IS estimates can be near-degenerate."
        ),
        "n_steps": payload["n_steps"],
        "n_steps_per_epoch": payload["n_steps_per_epoch"],
        "gamma": payload["gamma"],
        "seed": payload["seed"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": [
            {k: _clean(v) for k, v in r.items()} for r in payload["results"]
        ],
        "best_per_variant": {
            variant: {k: _clean(v) for k, v in best.items()}
            for variant, best in payload["best_per_variant"].items()
        },
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"\nSaved full results -> {RESULTS_JSON}")

    df = pd.DataFrame(payload["results"])
    df["hidden_units"] = df["hidden_units"].apply(lambda h: "x".join(map(str, h)))
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved flat results  -> {RESULTS_CSV}")

    if save_models:
        for variant, model in best_models.items():
            path = MODEL_DIR / f"dqn_best_{variant}.d3"
            model.save_model(str(path))
            print(f"Saved best model ({variant}) -> {path}")


def print_leaderboard(best_per_variant: dict) -> None:
    print("\n================ BEST CONFIG PER REWARD VARIANT ================")
    header = (
        f"{'reward variant':<22} {'algo':<11} {'lr':>9} {'bs':>4} "
        f"{'tui':>6} {'hidden':>10} {'pdis':>9} {'wpdis':>9} "
        f"{'dr':>9} {'mood Δ':>9} {'a-match':>8}"
    )
    print(header)
    print("-" * len(header))
    for variant, r in best_per_variant.items():
        hidden = "x".join(map(str, r["hidden_units"]))
        pdis = r["pdis"] if np.isfinite(r["pdis"]) else float("nan")
        wpdis = r["weighted_pdis"] if np.isfinite(r["weighted_pdis"]) else float("nan")
        dr = r["dr"] if np.isfinite(r["dr"]) else float("nan")
        mood = r["mood_improvement"] if np.isfinite(r["mood_improvement"]) else float("nan")
        print(
            f"{variant:<22} {r['algo']:<11} {r['learning_rate']:>9.2e} "
            f"{r['batch_size']:>4} {r['target_update_interval']:>6} {hidden:>10} "
            f"{pdis:>9.4f} {wpdis:>9.4f} {dr:>9.4f} "
            f"{mood:>9.4f} {r['action_match']:>8.4f}"
        )


def main() -> None:
    args = parse_args()
    payload = run_search(args)
    print_leaderboard(payload["best_per_variant"])
    save_results(payload, save_models=not args.no_save_models)


if __name__ == "__main__":
    main()
