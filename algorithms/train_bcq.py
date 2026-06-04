"""
Discrete BCQ (Batch-Constrained Q-learning) for StudentLife offline RL.
CS 224R Spring 2026 — Juan Pablo Pacheco

Implements Fujimoto et al. 2019 "Off-Policy Deep Reinforcement Learning without
Exploration" for discrete action spaces.

Key idea: Constrain action selection to actions the behavior policy would
plausibly take, preventing Q-value overestimation on out-of-distribution (OOD)
actions. This is critical here because actions are observational and sparse
rewards make OOD extrapolation dangerous.

Run from repo root:
    python algorithms/train_bcq.py

AI Tools Disclosure — see bottom of file.
"""

import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl_common import (
    GAMMA as COMMON_GAMMA,
    MLP,
    N_ACTIONS,
    N_STATE,
    ReplayBuffer,
    STATE_COLS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV    = PROJECT_ROOT / "final_datasets" / "daily_studentlife.train.csv"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_PATH   = MODEL_DIR / "bcq_model.pt"
LOG_PATH     = MODEL_DIR / "bcq_training_log.csv"

# Training hyperparameters from Fujimoto et al. 2019, adapted for CPU.
# Default is 15k steps — sufficient for this small dataset (~2400 transitions).
# Increase to 30k on GPU or if td_loss hasn't converged by the end of training.
LR                 = 1e-4
BATCH_SIZE         = 64
GAMMA              = COMMON_GAMMA
TARGET_UPDATE_FREQ = 100   # C: hard target-network update every C gradient steps
N_STEPS            = 15_000
BCQ_THRESHOLD      = 0.3   # keep only actions where P(a|s)/max_P(·|s) >= threshold
SEED               = 42


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bcq(
    train_csv:          Path  = TRAIN_CSV,
    model_path:         Path  = MODEL_PATH,
    log_path:           Path  = LOG_PATH,
    n_steps:            int   = N_STEPS,
    lr:                 float = LR,
    batch_size:         int   = BATCH_SIZE,
    gamma:              float = GAMMA,
    target_update_freq: int   = TARGET_UPDATE_FREQ,
    bcq_threshold:      float = BCQ_THRESHOLD,
    seed:               int   = SEED,
    reward_col:         str   = "reward_dense",
    save_model:         bool  = True,
) -> tuple:
    """
    Train Discrete BCQ on the offline StudentLife dataset.

    Returns:
        (bc_net, q_net, target_q_net) — trained nn.Module instances on CPU.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    df     = pd.read_csv(train_csv)
    buffer = ReplayBuffer(df, reward_col)
    print(f"Loaded {buffer.size} transitions | reward_col={reward_col}")

    bc_net       = MLP(N_STATE, N_ACTIONS)
    q_net        = MLP(N_STATE, N_ACTIONS)
    target_q_net = MLP(N_STATE, N_ACTIONS)
    target_q_net.load_state_dict(q_net.state_dict())
    for p in target_q_net.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(
        list(bc_net.parameters()) + list(q_net.parameters()), lr=lr
    )

    log_rows     = []
    log_interval = max(n_steps // 100, 50)

    print(f"Training BCQ: {n_steps} steps, batch={batch_size}, lr={lr}")
    for step in range(1, n_steps + 1):
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        # Behavior cloning loss: cross-entropy on observed actions
        bc_logits = bc_net(states)
        bc_loss   = F.cross_entropy(bc_logits, actions)

        # BCQ TD target: restrict next-state actions to in-distribution set.
        # Only actions with P(a|s') / max_a P(a|s') >= threshold are kept;
        # the rest are masked to -inf before the argmax.
        with torch.no_grad():
            bc_next_probs = F.softmax(bc_net(next_states), dim=1)
            max_bc_prob   = bc_next_probs.max(dim=1, keepdim=True).values
            in_dist_mask  = (bc_next_probs / (max_bc_prob + 1e-8)) >= bcq_threshold
            q_next        = target_q_net(next_states).clone()
            q_next[~in_dist_mask] = -1e9
            best_next_a   = q_next.argmax(dim=1)
            td_target     = rewards + gamma * (1.0 - dones) * q_next[
                torch.arange(batch_size), best_next_a
            ]

        q_vals  = q_net(states)
        td_loss = F.mse_loss(q_vals[torch.arange(batch_size), actions], td_target)

        total_loss = td_loss + bc_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % target_update_freq == 0:
            target_q_net.load_state_dict(q_net.state_dict())

        if step % log_interval == 0 or step == n_steps:
            row = {
                "step":       step,
                "td_loss":    round(td_loss.item(), 6),
                "bc_loss":    round(bc_loss.item(), 6),
                "total_loss": round(total_loss.item(), 6),
            }
            log_rows.append(row)
            if step % (log_interval * 10) == 0 or step == n_steps:
                print(
                    f"  step {step:>6}/{n_steps} "
                    f"| TD={td_loss.item():.4f} "
                    f"| BC={bc_loss.item():.4f}"
                )

    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bc_net":       bc_net.state_dict(),
                "q_net":        q_net.state_dict(),
                "target_q_net": target_q_net.state_dict(),
                "config": {
                    "n_state":            N_STATE,
                    "n_actions":          N_ACTIONS,
                    "hidden":             [256, 256],
                    "lr":                 lr,
                    "batch_size":         batch_size,
                    "gamma":              gamma,
                    "target_update_freq": target_update_freq,
                    "n_steps":            n_steps,
                    "bcq_threshold":      bcq_threshold,
                    "seed":               seed,
                    "reward_col":         reward_col,
                },
            },
            model_path,
        )
        print(f"Saved BCQ model → {model_path}")

        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "td_loss", "bc_loss", "total_loss"]
            )
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"Saved training log → {log_path}")

    return bc_net, q_net, target_q_net


# ---------------------------------------------------------------------------
# Policy wrapper — same interface as CQLPolicyWrapper in evaluate_policies.py
# ---------------------------------------------------------------------------

class BCQPolicyWrapper:
    """
    Wraps trained BCQ networks for offline policy evaluation.

    Interface matches CQLPolicyWrapper / SklearnPolicyWrapper so it plugs
    directly into evaluate_policies.py's evaluate_all() and all OPE estimators.
    """

    name = "bcq"

    def __init__(
        self,
        bc_net:        nn.Module,
        q_net:         nn.Module,
        bcq_threshold: float = BCQ_THRESHOLD,
        n_actions:     int   = N_ACTIONS,
        epsilon:       float = 0.05,
    ):
        self._bc_net    = bc_net.eval()
        self._q_net     = q_net.eval()
        self._threshold = bcq_threshold
        self._n_actions = n_actions
        self._epsilon   = epsilon

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "BCQPolicyWrapper":
        """Load a saved BCQ checkpoint and return a ready-to-use wrapper."""
        ckpt   = torch.load(path, map_location="cpu", weights_only=True)
        cfg    = ckpt["config"]
        bc_net = MLP(cfg["n_state"], cfg["n_actions"], cfg["hidden"])
        q_net  = MLP(cfg["n_state"], cfg["n_actions"], cfg["hidden"])
        bc_net.load_state_dict(ckpt["bc_net"])
        q_net.load_state_dict(ckpt["q_net"])
        return cls(
            bc_net,
            q_net,
            bcq_threshold=cfg["bcq_threshold"],
            n_actions=cfg["n_actions"],
        )

    def _tensor(self, states: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(np.asarray(states, dtype="float32"))

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Return BCQ-constrained greedy actions, shape (n,). Values in [0, 6]."""
        with torch.no_grad():
            s        = self._tensor(states)
            bc_probs = F.softmax(self._bc_net(s), dim=1)
            max_prob = bc_probs.max(dim=1, keepdim=True).values
            mask     = (bc_probs / (max_prob + 1e-8)) >= self._threshold
            q_vals   = self._q_net(s).clone()
            q_vals[~mask] = -1e9
            return q_vals.argmax(dim=1).numpy().astype("int64")

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        """Epsilon-greedy softening so IS weights are always finite, shape (n, 7)."""
        actions = self.predict(states)
        probs   = np.full(
            (len(states), self._n_actions), self._epsilon / self._n_actions
        )
        probs[np.arange(len(states)), actions] += (1.0 - self._epsilon)
        return probs

    def q_values(self, states: np.ndarray) -> np.ndarray:
        """Raw Q-values for all actions, shape (n, 7). Used for DR estimate."""
        with torch.no_grad():
            return self._q_net(self._tensor(states)).numpy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bc_net, q_net, _ = train_bcq()

    print("\nValidating BCQPolicyWrapper ...")
    test_csv = PROJECT_ROOT / "final_datasets" / "daily_studentlife.test.csv"
    df_test  = pd.read_csv(test_csv)
    wrapper  = BCQPolicyWrapper(bc_net, q_net)
    states   = df_test[STATE_COLS].fillna(0.0).to_numpy("float32")

    actions = wrapper.predict(states)
    probs   = wrapper.action_probs(states)
    q_vals  = wrapper.q_values(states)

    assert actions.min() >= 0 and actions.max() <= 6, \
        f"Actions out of range: [{actions.min()}, {actions.max()}]"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), \
        f"action_probs() rows don't sum to 1: {probs.sum(axis=1)[:5]}"
    assert q_vals.shape == (len(states), N_ACTIONS), \
        f"q_values() wrong shape: {q_vals.shape}"

    print(f"  predict()      range : [{actions.min()}, {actions.max()}]  ✓")
    print(f"  action_probs() row sum: {probs.sum(axis=1).mean():.6f}  ✓")
    print(f"  q_values()     shape  : {q_vals.shape}  ✓")
    print("BCQ validation passed.")

# ---------------------------------------------------------------------------
# AI Tools Disclosure
# ---------------------------------------------------------------------------
# Claude (claude-sonnet-4-6) was used to generate boilerplate infrastructure
# in this file including: ReplayBuffer class, MLP class architecture, training
# loop skeleton, file I/O (torch.save, csv.DictWriter), BCQPolicyWrapper
# class structure (load classmethod, _tensor helper, action_probs epsilon-
# greedy softening), and the __main__ validation block.
#
# Juan Pablo Pacheco independently implemented the core algorithmic
# contributions:
#   - BCQ TD target computation (in-distribution masking via BC threshold)
#   - BCQPolicyWrapper.predict()  (BCQ-constrained argmax Q action selection)
#   - BCQPolicyWrapper.q_values() (raw Q-value extraction for DR estimation)
#   - BCQPolicyWrapper.action_probs() (epsilon-greedy softening for IS weights)
