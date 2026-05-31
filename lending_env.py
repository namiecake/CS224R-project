"""Gymnasium environment for long-horizon fairness in mortgage lending.

Six demographic groups (Race × Income tier) each maintain a persistent
population of applicants with individual creditworthiness scores in [0, 1].
At every round the agent picks per-group approval thresholds; approved
applicants stochastically repay or default, individual scores drift
accordingly, and a fraction of the population is replaced by fresh draws
from the (fixed) initial distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.stats import truncnorm

from constants import GROUP_ORDER


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    """Population-weighted variance with weights normalised to sum to 1."""
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    total = w.sum()
    if total <= 0.0 or v.size == 0:
        return 0.0
    w = w / total
    mu_bar = float(np.sum(w * v))
    return float(np.sum(w * (v - mu_bar) ** 2))


def max_pairwise_gap(values: np.ndarray) -> float:
    """Maximum absolute difference between any pair of values."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return 0.0
    return float(np.max(v) - np.min(v))


def _default_groups() -> list[dict[str, Any]]:
    """Synthetic initial parameters when no HMDA config is supplied."""
    defaults = [
        ("White_Low", "White", "Low", 0.42, 0.13, 1850),
        ("White_Middle", "White", "Middle", 0.51, 0.14, 3200),
        ("White_Upper", "White", "Upper", 0.58, 0.13, 4100),
        ("Black_Low", "Black", "Low", 0.38, 0.12, 950),
        ("Black_Middle", "Black", "Middle", 0.45, 0.13, 1400),
        ("Black_Upper", "Black", "Upper", 0.52, 0.14, 1100),
    ]
    return [
        {
            "name": name,
            "race": race,
            "income_tier": tier,
            "mu_init": mu,
            "sigma_init": sigma,
            "N_init": n,
        }
        for name, race, tier, mu, sigma, n in defaults
    ]


@dataclass
class GroupInit:
    name: str
    race: str
    income_tier: str
    mu_init: float
    sigma_init: float
    N_init: int


@dataclass
class LendingConfig:
    """All tunable hyperparameters for :class:`LendingEnv`."""

    groups: list[GroupInit] = field(default_factory=lambda: [
        GroupInit(**g) for g in _default_groups()
    ])

    delta_repay: float = 0.05
    delta_default: float = 0.10

    k: float = 8.0
    s_0: float = 0.5

    u_repay: float = 1.0
    u_default: float = 2.0

    churn_rate: float = 0.05
    horizon: int = 50

    reward_mode: str = "profit"  # profit | profit_wvar_approval | profit_wvar_mean
    lam: float = 1.0

    N_max_multiplier: float = 10.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "LendingConfig":
        if cfg is None:
            return cls()
        cfg = dict(cfg)
        groups_raw = cfg.pop("groups", None)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        inst = cls(**cfg)
        if groups_raw is not None:
            inst.groups = [GroupInit(**g) for g in groups_raw]
        inst._validate_groups()
        return inst

    def _validate_groups(self) -> None:
        names = [g.name for g in self.groups]
        if names != list(GROUP_ORDER):
            raise ValueError(
                f"groups must appear in canonical GROUP_ORDER; got {names!r}, "
                f"expected {list(GROUP_ORDER)!r}"
            )


def _sample_truncnorm(
    mu: float, sigma: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    sigma = max(sigma, 1e-6)
    a, b = (0.0 - mu) / sigma, (1.0 - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n, random_state=rng)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


class LendingEnv(gym.Env):
    """Multi-group lending env with per-group continuous thresholds.

    Observation: ``np.array`` of shape ``(3 * G,)`` —
    ``[μ_0, σ_0, N_0, …, μ_{G-1}, σ_{G-1}, N_{G-1}]`` in ``GROUP_ORDER``.
    Action: ``np.array`` of shape ``(G,)`` — per-group thresholds in ``[0, 1]``.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.cfg = LendingConfig.from_dict(config)

        valid_modes = {"profit", "profit_wvar_approval", "profit_wvar_mean"}
        if self.cfg.reward_mode not in valid_modes:
            raise ValueError(f"Unknown reward_mode: {self.cfg.reward_mode!r}")

        self.groups = list(self.cfg.groups)
        self.G = len(self.groups)

        self._initial_params: list[tuple[float, float, int]] = [
            (g.mu_init, g.sigma_init, int(g.N_init)) for g in self.groups
        ]

        N_max = float(
            self.cfg.N_max_multiplier * max(g.N_init for g in self.groups)
        )
        obs_low = np.tile([0.0, 0.0, 0.0], self.G).astype(np.float32)
        obs_high = np.tile([1.0, 1.0, N_max], self.G).astype(np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.zeros(self.G, dtype=np.float32),
            high=np.ones(self.G, dtype=np.float32),
            dtype=np.float32,
        )

        self._populations: list[np.ndarray] = []
        self._t: int = 0
        self.np_random: np.random.Generator

    def _get_obs(self) -> np.ndarray:
        feats: list[float] = []
        for scores in self._populations:
            if scores.size > 0:
                mu = float(scores.mean())
                sigma = float(scores.std())
            else:
                mu = 0.0
                sigma = 0.0
            feats.extend([mu, sigma, float(scores.size)])
        return np.asarray(feats, dtype=np.float32)

    def _initial_population(self, g_idx: int) -> np.ndarray:
        mu, sigma, n = self._initial_params[g_idx]
        return _sample_truncnorm(mu, sigma, n, self.np_random)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._populations = [
            self._initial_population(i) for i in range(self.G)
        ]
        self._t = 0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != self.G:
            raise ValueError(
                f"Expected action of size {self.G}, got {action.shape}."
            )

        per_group: dict[str, dict[str, float]] = {}
        total_profit = 0.0

        for g_idx, group in enumerate(self.groups):
            scores = self._populations[g_idx]
            tau = float(action[g_idx])

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
                mu0, sigma0, _ = self._initial_params[g_idx]
                fresh = _sample_truncnorm(mu0, sigma0, n_churn, self.np_random)
                new_scores[idx] = fresh
            self._populations[g_idx] = new_scores

            group_profit = (
                self.cfg.u_repay * n_repaid - self.cfg.u_default * n_defaulted
            )
            total_profit += group_profit

            per_group[group.name] = {
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

        obs = self._get_obs()
        mus = np.array([obs[3 * i] for i in range(self.G)], dtype=np.float64)
        Ns = np.array([obs[3 * i + 2] for i in range(self.G)], dtype=np.float64)
        approval_rates = np.array(
            [per_group[g.name]["approval_rate"] for g in self.groups],
            dtype=np.float64,
        )

        wvar_mean = weighted_variance(mus, Ns)
        wvar_approval = weighted_variance(approval_rates, Ns)
        pairwise_gap = max_pairwise_gap(mus)

        if self.cfg.reward_mode == "profit":
            equity_penalty = 0.0
            reward = total_profit
        elif self.cfg.reward_mode == "profit_wvar_approval":
            equity_penalty = self.cfg.lam * wvar_approval
            reward = total_profit - equity_penalty
        else:  # profit_wvar_mean
            equity_penalty = self.cfg.lam * wvar_mean
            reward = total_profit - equity_penalty

        self._t += 1
        terminated = False
        truncated = self._t >= self.cfg.horizon

        info: dict[str, Any] = {
            "profit": float(total_profit),
            "profit_term": float(total_profit),
            "equity_penalty": float(equity_penalty),
            "reward": float(reward),
            "wvar_mean": float(wvar_mean),
            "wvar_approval": float(wvar_approval),
            "pairwise_gap": float(pairwise_gap),
            "per_group": per_group,
            "t": self._t,
        }
        for g_idx, group in enumerate(self.groups):
            info[f"mu_{group.name}"] = float(obs[3 * g_idx])
            info[f"sigma_{group.name}"] = float(obs[3 * g_idx + 1])
            info[f"N_{group.name}"] = float(obs[3 * g_idx + 2])

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
