"""
Discrete Implicit Q-Learning (IQL) for StudentLife offline RL.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl_common import (
    GAMMA,
    MLP,
    N_ACTIONS,
    N_STATE,
    EPSILON,
    ReplayBuffer,
    QLearningPolicyWrapper,
)


class IQLPolicyWrapper(QLearningPolicyWrapper):
    """IQL policy: greedy Q with advantage-softmax action_probs for OPE."""

    def __init__(
        self,
        q_net: nn.Module,
        v_net: nn.Module,
        temperature: float = 3.0,
        epsilon: float = EPSILON,
    ):
        super().__init__(q_net, name="iql", epsilon=epsilon)
        self._v_net = v_net.eval()
        self._temperature = max(float(temperature), 1e-6)

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = self._tensor(states)
            q = self._q_net(s)
            v = self._v_net(s)
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            adv = q - v
            logits = adv / self._temperature
            probs = F.softmax(logits, dim=1).cpu().numpy()
        floor = self._epsilon / self._n_actions
        probs = (1.0 - self._epsilon) * probs + floor
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric L2 for IQL value fitting."""
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def train_discrete_iql(
    train_df: pd.DataFrame,
    reward_col: str,
    *,
    n_steps: int = 5000,
    batch_size: int = 64,
    critic_lr: float = 1e-4,
    expectile: float = 0.7,
    temperature: float = 3.0,
    gamma: float = GAMMA,
    hidden_units: list | None = None,
    target_update_interval: int = 100,
    seed: int = 42,
) -> tuple[nn.Module, nn.Module, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden = hidden_units or [256, 256]
    buffer = ReplayBuffer(train_df, reward_col)

    q_net = MLP(N_STATE, N_ACTIONS, hidden)
    v_net = MLP(N_STATE, 1, hidden)
    target_q = copy.deepcopy(q_net)
    target_v = copy.deepcopy(v_net)

    q_opt = torch.optim.Adam(q_net.parameters(), lr=critic_lr)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=critic_lr)

    for step in range(1, n_steps + 1):
        s, a, r, s_next, done = buffer.sample(batch_size)

        with torch.no_grad():
            target = r + gamma * (1.0 - done) * target_v(s_next).squeeze(1)

        q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_sa, target)

        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        with torch.no_grad():
            q_all = q_net(s)
            q_logged = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        v_pred = v_net(s).squeeze(1)
        v_loss = expectile_loss(q_logged - v_pred, expectile)

        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()

        if step % target_update_interval == 0:
            target_q.load_state_dict(q_net.state_dict())
            target_v.load_state_dict(v_net.state_dict())

    config = {
        "algo": "iql",
        "reward_col": reward_col,
        "n_state": N_STATE,
        "n_actions": N_ACTIONS,
        "hidden": hidden,
        "expectile": expectile,
        "temperature": temperature,
        "gamma": gamma,
        "critic_lr": critic_lr,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "target_update_interval": target_update_interval,
    }
    return q_net, v_net, config


def save_iql_checkpoint(
    path: Path,
    q_net: nn.Module,
    v_net: nn.Module,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "q_net": q_net.state_dict(),
            "v_net": v_net.state_dict(),
        },
        path,
    )


def load_iql_policy(
    path: Path,
    *,
    epsilon: float = EPSILON,
) -> IQLPolicyWrapper:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    hidden = cfg["hidden"]
    q_net = MLP(cfg["n_state"], cfg["n_actions"], hidden)
    v_net = MLP(cfg["n_state"], 1, hidden)
    q_net.load_state_dict(ckpt["q_net"])
    v_net.load_state_dict(ckpt["v_net"])
    return IQLPolicyWrapper(
        q_net,
        v_net,
        temperature=cfg.get("temperature", 3.0),
        epsilon=epsilon,
    )
