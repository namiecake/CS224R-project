"""Gymnasium environment for long-horizon fairness in mortgage lending.

Two demographic groups ("A", "B") each maintain a persistent population of
applicants with individual creditworthiness scores in [0, 1]. At every round
the agent picks per-group approval thresholds; approved applicants stochastically
repay or default, individual scores drift accordingly, and a fraction of the
population is replaced by fresh draws from the (fixed) initial distribution.

The observation summarises the current population by its empirical mean, std,
and size for each group. Rewards are configurable to expose the agent to a
profit-only signal or to a fairness-augmented signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.stats import truncnorm

GROUPS: tuple[str, str] = ("A", "B")


@dataclass
class LendingConfig:
    """All tunable hyperparameters for :class:`LendingEnv`.

    Initial distribution parameters are stored separately at construction and
    used both to seed the initial population and to draw replacements during
    churn (so the "fresh draws" in step 5 come from the *initial* distribution,
    not the current empirical one).
    """

    # Initial truncated-Normal parameters per group on [0, 1].
    mu_A_init: float = 0.6
    sigma_A_init: float = 0.15
    N_A_init: int = 100
    mu_B_init: float = 0.4
    sigma_B_init: float = 0.15
    N_B_init: int = 100

    # Score-update magnitudes.
    delta_repay: float = 0.05
    delta_default: float = 0.10

    # Repayment probability sigmoid: p(s) = sigmoid(k * (s - s_0)).
    k: float = 8.0
    s_0: float = 0.5

    # Per-applicant utilities.
    u_repay: float = 1.0
    u_default: float = 2.0

    # Fraction of each group replaced per round.
    churn_rate: float = 0.05

    # Episode horizon.
    horizon: int = 50

    # Reward shaping.
    reward_mode: str = "profit"  # "profit" | "profit_dp" | "profit_gap"
    lam: float = 1.0

    # Loose upper bound for population size in the observation space.
    N_max_multiplier: float = 10.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "LendingConfig":
        if cfg is None:
            return cls()
        unknown = set(cfg) - {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**cfg)


def _sample_truncnorm(
    mu: float, sigma: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``n`` scores from a truncated Normal on [0, 1]."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    sigma = max(sigma, 1e-6)
    a, b = (0.0 - mu) / sigma, (1.0 - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n, random_state=rng)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


class LendingEnv(gym.Env):
    """Two-group lending env with per-group continuous thresholds.

    Observation: ``np.array([mu_A, sigma_A, N_A, mu_B, sigma_B, N_B])``.
    Action: ``np.array([tau_A, tau_B])`` in ``[0, 1]^2``.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.cfg = LendingConfig.from_dict(config)

        if self.cfg.reward_mode not in {"profit", "profit_dp", "profit_gap"}:
            raise ValueError(f"Unknown reward_mode: {self.cfg.reward_mode}")

        self._initial_params: dict[str, tuple[float, float, int]] = {
            "A": (self.cfg.mu_A_init, self.cfg.sigma_A_init, int(self.cfg.N_A_init)),
            "B": (self.cfg.mu_B_init, self.cfg.sigma_B_init, int(self.cfg.N_B_init)),
        }

        N_max = float(
            self.cfg.N_max_multiplier
            * max(self.cfg.N_A_init, self.cfg.N_B_init)
        )
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, N_max, 1.0, 1.0, N_max], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._populations: dict[str, np.ndarray] = {}
        self._t: int = 0
        self.np_random: np.random.Generator  # set in reset()

    # ------------------------------------------------------------------ utils
    def _get_obs(self) -> np.ndarray:
        feats: list[float] = []
        for g in GROUPS:
            scores = self._populations[g]
            if scores.size > 0:
                mu = float(scores.mean())
                sigma = float(scores.std())
            else:
                mu = 0.0
                sigma = 0.0
            feats.extend([mu, sigma, float(scores.size)])
        return np.asarray(feats, dtype=np.float32)

    def _initial_population(self, group: str) -> np.ndarray:
        mu, sigma, n = self._initial_params[group]
        return _sample_truncnorm(mu, sigma, n, self.np_random)

    # ------------------------------------------------------------------- API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._populations = {g: self._initial_population(g) for g in GROUPS}
        self._t = 0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 2:
            raise ValueError(f"Expected action of size 2, got {action.shape}.")
        thresholds = {"A": float(action[0]), "B": float(action[1])}

        per_group: dict[str, dict[str, float]] = {}
        total_profit = 0.0
        for g in GROUPS:
            scores = self._populations[g]
            tau = thresholds[g]

            approved_mask = scores >= tau
            approved_scores = scores[approved_mask]

            p_repay = _sigmoid(self.cfg.k * (approved_scores - self.cfg.s_0))
            repay_draws = self.np_random.random(approved_scores.shape) < p_repay
            n_repaid = int(repay_draws.sum())
            n_defaulted = int(approved_scores.size - n_repaid)

            updated_approved = approved_scores.copy()
            updated_approved[repay_draws] += self.cfg.delta_repay
            updated_approved[~repay_draws] -= self.cfg.delta_default
            updated_approved = np.clip(updated_approved, 0.0, 1.0)

            new_scores = scores.copy()
            new_scores[approved_mask] = updated_approved

            n_total = scores.size
            n_churn = int(np.floor(self.cfg.churn_rate * n_total))
            if n_churn > 0 and n_total > 0:
                idx = self.np_random.choice(n_total, size=n_churn, replace=False)
                fresh = _sample_truncnorm(
                    *self._initial_params[g][:2], n_churn, self.np_random
                )
                new_scores[idx] = fresh
            self._populations[g] = new_scores

            group_profit = (
                self.cfg.u_repay * n_repaid - self.cfg.u_default * n_defaulted
            )
            total_profit += group_profit

            per_group[g] = {
                "approved": float(approved_scores.size),
                "repaid": float(n_repaid),
                "defaulted": float(n_defaulted),
                "applicants": float(n_total),
                "approval_rate": (
                    float(approved_scores.size) / n_total if n_total > 0 else 0.0
                ),
                "repay_rate": (
                    float(n_repaid) / approved_scores.size
                    if approved_scores.size > 0
                    else float("nan")
                ),
                "profit": float(group_profit),
                "threshold": tau,
            }

        # `_get_obs` reads from the populations *after* score updates and
        # churn, so `gap` uses the same mu_g values that appear in the next
        # observation. The profit_gap reward penalises that post-update gap.
        obs = self._get_obs()
        mu_A, mu_B = float(obs[0]), float(obs[3])
        gap = abs(mu_A - mu_B)
        approval_gap = abs(per_group["A"]["approval_rate"] - per_group["B"]["approval_rate"])

        if self.cfg.reward_mode == "profit":
            reward = total_profit
        elif self.cfg.reward_mode == "profit_dp":
            reward = total_profit - self.cfg.lam * approval_gap
        else:  # profit_gap
            reward = total_profit - self.cfg.lam * gap

        self._t += 1
        terminated = False
        truncated = self._t >= self.cfg.horizon

        info: dict[str, Any] = {
            "profit": float(total_profit),
            "reward": float(reward),
            "gap": float(gap),
            "approval_gap": float(approval_gap),
            "per_group": per_group,
            "mu_A": mu_A,
            "mu_B": mu_B,
            "sigma_A": float(obs[1]),
            "sigma_B": float(obs[4]),
            "N_A": float(obs[2]),
            "N_B": float(obs[5]),
            "t": self._t,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
