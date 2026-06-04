"""
Shared data loading, replay buffer, networks, and policy wrappers for
offline RL on StudentLife chronological splits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from d3rlpy.dataset import MDPDataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "final_datasets"
DATASET_BASENAME = "daily_studentlife"
MODEL_DIR = PROJECT_ROOT / "models"

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
NEXT_STATE_COLS = [f"next_{c}" for c in STATE_COLS]

N_ACTIONS = 7
N_STATE = len(STATE_COLS)
GAMMA = 0.99
EPSILON = 0.05
SEED = 42

REWARD_VARIANTS = ["reward_sparse", "reward_dense", "reward_observed_only"]


class MLP(nn.Module):
    """Two-hidden-layer MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: list | None = None,
    ):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]
        dims = [input_dim] + list(hidden) + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """In-memory transitions with next-state columns from split CSVs."""

    def __init__(self, df: pd.DataFrame, reward_col: str):
        df = df.sort_values(["student_id", "episode_id", "date"]).reset_index(drop=True)
        self.states = torch.FloatTensor(
            df[STATE_COLS].fillna(0.0).to_numpy("float32")
        )
        self.next_states = torch.FloatTensor(
            df[NEXT_STATE_COLS].fillna(0.0).to_numpy("float32")
        )
        self.actions = torch.LongTensor(df["action"].to_numpy("int64"))
        self.rewards = torch.FloatTensor(
            df[reward_col].fillna(0.0).to_numpy("float32")
        )
        self.dones = torch.FloatTensor(df["done"].to_numpy("float32"))
        self.size = len(df)

    def sample(self, batch_size: int) -> tuple:
        idx = torch.randint(0, self.size, (batch_size,))
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


def load_splits(data_dir: Path, basename: str) -> dict[str, pd.DataFrame]:
    paths = {
        split: data_dir / f"{basename}.{split}.csv"
        for split in ("train", "val", "test")
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing split files: "
            + ", ".join(missing)
            + ". Generate them with dataset_prep/prepare_rl_dataset.py."
        )
    return {split: pd.read_csv(p) for split, p in paths.items()}


def build_mdp_dataset(df: pd.DataFrame, reward_col: str) -> MDPDataset:
    df = df.sort_values(["student_id", "episode_id", "date"]).reset_index(drop=True)
    terminals = (
        ~df.duplicated(subset=["student_id", "episode_id"], keep="last")
    ).to_numpy()
    return MDPDataset(
        observations=df[STATE_COLS].fillna(0.0).to_numpy("float32"),
        actions=df["action"].to_numpy("int64"),
        rewards=df[reward_col].fillna(0.0).to_numpy("float32"),
        terminals=terminals,
    )


def extract_episodes(df: pd.DataFrame, reward_col: str) -> list[dict]:
    episodes = []
    for _, grp in df.groupby(["student_id", "episode_id"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        episodes.append(
            {
                "states": grp[STATE_COLS].fillna(0.0).to_numpy("float32"),
                "actions": grp["action"].to_numpy("int64"),
                "rewards": grp[reward_col].fillna(0.0).to_numpy("float32"),
                "mood": grp["mood"].to_numpy("float32"),
                "next_moods": grp["next_mood"].to_numpy("float32"),
                "T": len(grp),
            }
        )
    return episodes


class BehaviorPolicy:
    name = "behavior_cloning_logistic"

    def __init__(self, model, classes: np.ndarray):
        self._model = model
        self._classes = [int(c) for c in classes]

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        raw = self._model.predict_proba(states)
        probs = np.full((len(states), N_ACTIONS), 1e-8, dtype="float64")
        for col, action in enumerate(self._classes):
            probs[:, action] = raw[:, col]
        return probs / probs.sum(axis=1, keepdims=True)


def fit_behavior_policy(train_df: pd.DataFrame, seed: int = SEED) -> BehaviorPolicy:
    x = train_df[STATE_COLS].fillna(0.0).to_numpy("float32")
    y = train_df["action"].to_numpy("int64")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    model.fit(x, y)
    return BehaviorPolicy(model, model.classes_)


class QLearningPolicyWrapper:
    """Interface for discrete Q-network policies used in OPE."""

    def __init__(
        self,
        q_net: nn.Module,
        name: str = "qlearning",
        epsilon: float = EPSILON,
        n_actions: int = N_ACTIONS,
    ):
        self._q_net = q_net.eval()
        self.name = name
        self._epsilon = epsilon
        self._n_actions = n_actions

    def _tensor(self, states: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(np.asarray(states, dtype="float32"))

    def predict(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            q = self._q_net(self._tensor(states))
            return q.argmax(dim=1).cpu().numpy().astype("int64")

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        actions = self.predict(states)
        probs = np.full(
            (len(states), self._n_actions),
            self._epsilon / self._n_actions,
            dtype="float64",
        )
        probs[np.arange(len(states)), actions] += 1.0 - self._epsilon
        return probs

    def q_values(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._q_net(self._tensor(states)).cpu().numpy()


def action_match_rate(states: np.ndarray, actions: np.ndarray, policy) -> float:
    pred = policy.predict(states)
    return float(np.mean(pred == actions))
