"""Non-RL baseline threshold policies and a rollout helper.

Both policies operate on the parametric state ``(mu_g, sigma_g, N_g)`` per
group. Expectations are computed analytically against the truncated-Normal
model implied by the state.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from scipy.stats import truncnorm

from constants import GROUP_ORDER
from lending_env import LendingConfig, LendingEnv


class Policy(Protocol):
    def act(self, state: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------- analytics


def _truncnorm_args(mu: float, sigma: float) -> tuple[float, float, float, float]:
    sigma = max(float(sigma), 1e-6)
    a, b = (0.0 - mu) / sigma, (1.0 - mu) / sigma
    return a, b, float(mu), sigma


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _group_curves(
    mu: float,
    sigma: float,
    N: float,
    *,
    k: float,
    s_0: float,
    u_repay: float,
    u_default: float,
    taus: np.ndarray,
    n_integration: int = 2001,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(approval_rate, expected_profit)`` arrays evaluated at ``taus``."""
    a, b, loc, scale = _truncnorm_args(mu, sigma)
    s = np.linspace(0.0, 1.0, n_integration)
    pdf = truncnorm.pdf(s, a, b, loc=loc, scale=scale)
    p_repay = _sigmoid(k * (s - s_0))
    reward_density = pdf * (u_repay * p_repay - u_default * (1.0 - p_repay))

    sf = truncnorm.sf(taus, a, b, loc=loc, scale=scale)

    rev = reward_density[::-1]
    s_rev = s[::-1]
    diffs = np.diff(s_rev)
    seg = 0.5 * (rev[:-1] + rev[1:]) * diffs
    cum_from_right_rev = np.concatenate(([0.0], np.cumsum(seg)))
    cum_from_tau = -cum_from_right_rev[::-1]

    expected_reward_at_grid = np.interp(taus, s, cum_from_tau)
    return sf, N * expected_reward_at_grid


def _state_to_group_stats(
    state: np.ndarray, G: int
) -> list[tuple[float, float, float]]:
    """Parse ``(mu, sigma, N)`` triples from a state vector of length ``3*G``."""
    stats: list[tuple[float, float, float]] = []
    for i in range(G):
        base = 3 * i
        stats.append(
            (float(state[base]), float(state[base + 1]), float(state[base + 2]))
        )
    return stats


def _threshold_for_approval_rate(
    mu: float, sigma: float, target_rate: float
) -> float:
    """Return threshold ``tau`` such that ``P(score >= tau) ≈ target_rate``."""
    a, b, loc, scale = _truncnorm_args(mu, sigma)
    # sf(tau) = target_rate  =>  CDF(tau) = 1 - target_rate
    return float(truncnorm.ppf(1.0 - target_rate, a, b, loc=loc, scale=scale))


# ----------------------------------------------------------------- policies


class ProfitMaxThresholdPolicy:
    """Per-group expected-profit-maximising thresholds via 1D grid search."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.cfg = LendingConfig.from_dict(config)
        self.G = len(self.cfg.groups)
        self._taus = np.linspace(0.0, 1.0, 101)

    def act(self, state: np.ndarray) -> np.ndarray:
        stats = _state_to_group_stats(state, self.G)
        tau_out = np.zeros(self.G, dtype=np.float32)
        for i, (mu, sigma, N) in enumerate(stats):
            _, expected_profit = _group_curves(
                mu,
                sigma,
                N,
                k=self.cfg.k,
                s_0=self.cfg.s_0,
                u_repay=self.cfg.u_repay,
                u_default=self.cfg.u_default,
                taus=self._taus,
            )
            tau_out[i] = self._taus[int(np.argmax(expected_profit))]
        return 2.0 * tau_out - 1.0


class DemographicParityPolicy:
    """Profit-maximising thresholds subject to equal approval rates across groups.

    Searches over a common target approval rate ``r*`` and picks per-group
    thresholds that achieve exactly ``r*`` under each group's distribution.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.cfg = LendingConfig.from_dict(config)
        self.G = len(self.cfg.groups)
        self._target_rates = np.linspace(0.01, 0.99, 99)

    def act(self, state: np.ndarray) -> np.ndarray:
        stats = _state_to_group_stats(state, self.G)

        best_profit = -np.inf
        best_thresholds = np.zeros(self.G, dtype=np.float64)

        for r_star in self._target_rates:
            thresholds = np.array(
                [
                    _threshold_for_approval_rate(mu, sigma, r_star)
                    for mu, sigma, _ in stats
                ],
                dtype=np.float64,
            )
            total_profit = 0.0
            for (mu, sigma, N), tau in zip(stats, thresholds):
                _, expected_profit = _group_curves(
                    mu,
                    sigma,
                    N,
                    k=self.cfg.k,
                    s_0=self.cfg.s_0,
                    u_repay=self.cfg.u_repay,
                    u_default=self.cfg.u_default,
                    taus=np.array([tau]),
                )
                total_profit += float(expected_profit[0])
            if total_profit > best_profit:
                best_profit = total_profit
                best_thresholds = thresholds

        return (2.0 * best_thresholds - 1.0).astype(np.float32)


# ------------------------------------------------------------------ rollout


def _log_keys(G: int) -> tuple[str, ...]:
    keys: list[str] = []
    for name in GROUP_ORDER[:G]:
        keys.extend(
            [
                f"mu_{name}",
                f"sigma_{name}",
                f"N_{name}",
                f"approval_rate_{name}",
                f"repay_rate_{name}",
                f"tau_{name}",
            ]
        )
    keys.extend(
        [
            "profit",
            "reward",
            "wvar_mean",
            "pairwise_gap",
            "profit_term",
            "equity_penalty",
        ]
    )
    return tuple(keys)


def simulate(
    policy: Policy,
    env: LendingEnv,
    n_seeds: int,
    n_rounds: int,
) -> dict[str, np.ndarray]:
    """Run ``policy`` through ``env`` for ``n_seeds`` episodes of ``n_rounds`` steps."""
    if n_rounds > env.cfg.horizon:
        raise ValueError(
            f"n_rounds={n_rounds} exceeds env horizon {env.cfg.horizon}."
        )

    log_keys = _log_keys(env.G)
    logs: dict[str, list[list[float]]] = {key: [] for key in log_keys}

    for seed in range(n_seeds):
        obs, _ = env.reset(seed=seed)
        seed_logs: dict[str, list[float]] = {key: [] for key in log_keys}
        for _ in range(n_rounds):
            action = np.asarray(policy.act(obs), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            per_group = info["per_group"]
            for name in GROUP_ORDER[: env.G]:
                seed_logs[f"mu_{name}"].append(info[f"mu_{name}"])
                seed_logs[f"sigma_{name}"].append(info[f"sigma_{name}"])
                seed_logs[f"N_{name}"].append(info[f"N_{name}"])
                seed_logs[f"approval_rate_{name}"].append(
                    per_group[name]["approval_rate"]
                )
                seed_logs[f"repay_rate_{name}"].append(
                    per_group[name]["repay_rate"]
                )
                seed_logs[f"tau_{name}"].append(per_group[name]["threshold"])
            seed_logs["profit"].append(info["profit"])
            seed_logs["reward"].append(reward)
            seed_logs["wvar_mean"].append(info["wvar_mean"])
            seed_logs["pairwise_gap"].append(info["pairwise_gap"])
            seed_logs["profit_term"].append(info["profit_term"])
            seed_logs["equity_penalty"].append(info["equity_penalty"])
            if terminated or truncated:
                break
        for k, v in seed_logs.items():
            logs[k].append(v)

    out: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.float64) for k, v in logs.items()
    }
    out["cumulative_wvar_mean"] = out["wvar_mean"].sum(axis=1)
    return out
