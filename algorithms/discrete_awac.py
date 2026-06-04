"""
Discrete AWAC (advantage-weighted actor-critic) for StudentLife offline RL.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from discrete_iql import expectile_loss, train_discrete_iql
from offline_rl_common import (
    GAMMA,
    MLP,
    N_ACTIONS,
    N_STATE,
    EPSILON,
    ReplayBuffer,
)


class AWACPolicyWrapper:
    """Softmax policy with epsilon floor for OPE."""

    name = "awac"

    def __init__(
        self,
        policy_net: nn.Module,
        q_net: nn.Module,
        epsilon: float = EPSILON,
        n_actions: int = N_ACTIONS,
    ):
        self._policy_net = policy_net.eval()
        self._q_net = q_net.eval()
        self._epsilon = epsilon
        self._n_actions = n_actions

    def _tensor(self, states: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(np.asarray(states, dtype="float32"))

    def predict(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self._policy_net(self._tensor(states))
            return logits.argmax(dim=1).cpu().numpy().astype("int64")

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self._policy_net(self._tensor(states))
            probs = F.softmax(logits, dim=1).cpu().numpy()
        floor = self._epsilon / self._n_actions
        probs = (1.0 - self._epsilon) * probs + floor
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def q_values(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._q_net(self._tensor(states)).cpu().numpy()


def train_discrete_awac(
    train_df: pd.DataFrame,
    reward_col: str,
    *,
    n_steps: int = 5000,
    batch_size: int = 64,
    critic_lr: float = 1e-4,
    actor_lr: float = 1e-4,
    expectile: float = 0.7,
    lam: float = 1.0,
    max_weight: float = 100.0,
    gamma: float = GAMMA,
    hidden_units: list | None = None,
    target_update_interval: int = 100,
    critic_warmup: int = 1000,
    seed: int = 42,
) -> tuple[nn.Module, nn.Module, nn.Module, dict]:
    """Train critic (IQL-style) then advantage-weighted policy."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden = hidden_units or [256, 256]
    buffer = ReplayBuffer(train_df, reward_col)

    q_net, v_net, critic_cfg = train_discrete_iql(
        train_df,
        reward_col,
        n_steps=critic_warmup,
        batch_size=batch_size,
        critic_lr=critic_lr,
        expectile=expectile,
        gamma=gamma,
        hidden_units=hidden,
        target_update_interval=target_update_interval,
        seed=seed,
    )

    policy_net = MLP(N_STATE, N_ACTIONS, hidden)
    target_q = copy.deepcopy(q_net)
    target_v = copy.deepcopy(v_net)

    q_opt = torch.optim.Adam(q_net.parameters(), lr=critic_lr)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=critic_lr)
    pi_opt = torch.optim.Adam(policy_net.parameters(), lr=actor_lr)

    remaining = max(n_steps - critic_warmup, 0)
    for step in range(1, remaining + 1):
        s, a, r, s_next, done = buffer.sample(batch_size)

        with torch.no_grad():
            target = r + gamma * (1.0 - done) * target_v(s_next).squeeze(1)
        q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_sa, target)
        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        with torch.no_grad():
            q_logged = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        v_loss = expectile_loss(q_logged - v_net(s).squeeze(1), expectile)
        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()

        with torch.no_grad():
            q_all = q_net(s)
            v = v_net(s).squeeze(1)
            adv = q_all.gather(1, a.unsqueeze(1)).squeeze(1) - v
            weights = torch.exp(adv / max(lam, 1e-6)).clamp(max=max_weight)

        log_probs = F.log_softmax(policy_net(s), dim=1)
        log_pi_a = log_probs.gather(1, a.unsqueeze(1)).squeeze(1)
        pi_loss = -(weights * log_pi_a).mean()

        pi_opt.zero_grad()
        pi_loss.backward()
        pi_opt.step()

        if step % target_update_interval == 0:
            target_q.load_state_dict(q_net.state_dict())
            target_v.load_state_dict(v_net.state_dict())

    config = {
        "algo": "awac",
        "reward_col": reward_col,
        "n_state": N_STATE,
        "n_actions": N_ACTIONS,
        "hidden": hidden,
        "expectile": expectile,
        "lam": lam,
        "max_weight": max_weight,
        "gamma": gamma,
        "critic_lr": critic_lr,
        "actor_lr": actor_lr,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "critic_warmup": critic_warmup,
        "target_update_interval": target_update_interval,
    }
    return policy_net, q_net, v_net, config


def save_awac_checkpoint(
    path: Path,
    policy_net: nn.Module,
    q_net: nn.Module,
    v_net: nn.Module,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "policy_net": policy_net.state_dict(),
            "q_net": q_net.state_dict(),
            "v_net": v_net.state_dict(),
        },
        path,
    )


def load_awac_policy(
    path: Path,
    *,
    epsilon: float = EPSILON,
) -> AWACPolicyWrapper:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    hidden = cfg["hidden"]
    policy_net = MLP(cfg["n_state"], cfg["n_actions"], hidden)
    q_net = MLP(cfg["n_state"], cfg["n_actions"], hidden)
    policy_net.load_state_dict(ckpt["policy_net"])
    q_net.load_state_dict(ckpt["q_net"])
    return AWACPolicyWrapper(policy_net, q_net, epsilon=epsilon)
