"""
Bootstrap and analytic uncertainty for offline policy evaluation metrics.
"""

from __future__ import annotations

import numpy as np

from evaluate_policies import (
    compute_doubly_robust_estimate,
    compute_mood_improvement_deltas,
    compute_pdis_estimate,
    compute_weighted_pdis_estimate,
)

DEFAULT_BOOTSTRAP_SAMPLES = 300
DEFAULT_ALPHA = 0.05


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float, float]:
    if samples.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo = float(np.percentile(samples, 100.0 * alpha / 2.0))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    se = float(np.std(samples, ddof=1)) if samples.size > 1 else float("nan")
    return lo, hi, se


def bootstrap_episode_ci(
    episodes: list,
    estimator,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> dict[str, float]:
    """Nonparametric bootstrap over episodes (cluster-respecting)."""
    if not episodes:
        return {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "se": float("nan"),
        }

    point = float(estimator(episodes))
    if len(episodes) < 2 or n_bootstrap <= 0:
        return {"ci_low": point, "ci_high": point, "se": float("nan")}

    rng = np.random.default_rng(seed)
    n = len(episodes)
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = [episodes[i] for i in idx]
        boots[b] = float(estimator(sample))

    lo, hi, se = _percentile_ci(boots, alpha)
    return {"ci_low": lo, "ci_high": hi, "se": se}


def match_rate_on_episodes(episodes: list, policy) -> float:
    matches = 0
    total = 0
    for episode in episodes:
        pred = policy.predict(episode["states"])
        acts = episode["actions"]
        matches += int(np.sum(pred == acts))
        total += int(acts.shape[0])
    return float(matches / total) if total else float("nan")


def mood_improvement_ci(
    deltas: list[float],
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float]:
    """Normal-approximation CI for mean mood delta (matched, observed mood)."""
    if not deltas:
        return {
            "mean": float("nan"),
            "n": 0,
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    arr = np.asarray(deltas, dtype=np.float64)
    n = arr.size
    mean = float(np.mean(arr))
    if n < 2:
        return {
            "mean": mean,
            "n": int(n),
            "se": float("nan"),
            "ci_low": mean,
            "ci_high": mean,
        }

    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    z = 1.96  # approximate 95% for moderate n
    return {
        "mean": mean,
        "n": int(n),
        "se": se,
        "ci_low": mean - z * se,
        "ci_high": mean + z * se,
    }


def _attach_ci_fields(result: dict, prefix: str, ci: dict[str, float]) -> None:
    result[f"{prefix}_se"] = ci.get("se")
    result[f"{prefix}_ci_low"] = ci.get("ci_low")
    result[f"{prefix}_ci_high"] = ci.get("ci_high")


def attach_policy_uncertainty(
    result: dict,
    episodes: list,
    policy,
    behavior_policy,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
) -> dict:
    """Add *_se, *_ci_low, *_ci_high for OPE, match rate, and mood metrics."""
    if n_bootstrap > 0:
        for metric, estimator in (
            ("pdis", lambda eps: compute_pdis_estimate(eps, policy, behavior_policy)),
            (
                "weighted_pdis",
                lambda eps: compute_weighted_pdis_estimate(eps, policy, behavior_policy),
            ),
            (
                "match_rate",
                lambda eps: match_rate_on_episodes(eps, policy),
            ),
        ):
            ci = bootstrap_episode_ci(
                episodes,
                estimator,
                n_bootstrap=n_bootstrap,
                seed=seed + abs(hash(metric)) % 10_000,
            )
            _attach_ci_fields(result, metric, ci)

        if hasattr(policy, "q_values"):
            dr_ci = bootstrap_episode_ci(
                episodes,
                lambda eps: compute_doubly_robust_estimate(eps, policy, behavior_policy),
                n_bootstrap=n_bootstrap,
                seed=seed + 17,
            )
            _attach_ci_fields(result, "dr", dr_ci)

    deltas = compute_mood_improvement_deltas(episodes, policy)
    mood_ci = mood_improvement_ci(deltas)
    _attach_ci_fields(result, "mood_improvement", mood_ci)
    return result
