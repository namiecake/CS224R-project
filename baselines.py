"""Non-RL baseline threshold policies and a rollout helper.

Both policies operate on the parametric state ``(mu_g, sigma_g, N_g)`` and
search a 0.01-resolution grid over per-group thresholds. Expectations are
computed analytically against the *current* truncated-Normal model implied by
the state, which matches the env's per-round generative model up to score
drift within the round.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from scipy.stats import truncnorm

from lending_env import GROUPS, LendingConfig, LendingEnv


GRID_STEP: float = 0.01


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
    """Return ``(approval_rate, expected_profit)`` arrays evaluated at ``taus``.

    ``expected_profit`` is the per-group expected profit (already multiplied by
    ``N``) for each threshold value.
    """

    a, b, loc, scale = _truncnorm_args(mu, sigma)
    s = np.linspace(0.0, 1.0, n_integration)
    pdf = truncnorm.pdf(s, a, b, loc=loc, scale=scale)
    p_repay = _sigmoid(k * (s - s_0))
    reward_density = pdf * (u_repay * p_repay - u_default * (1.0 - p_repay))

    sf = truncnorm.sf(taus, a, b, loc=loc, scale=scale)

    # Cumulative reward from each grid point onward (right-Riemann via trapezoid).
    # We integrate from tau to 1: build cumulative trapezoid from the right.
    rev = reward_density[::-1]
    s_rev = s[::-1]
    # trapezoid cumulative sum from the right
    diffs = np.diff(s_rev)  # negative
    seg = 0.5 * (rev[:-1] + rev[1:]) * diffs
    cum_from_right_rev = np.concatenate(([0.0], np.cumsum(seg)))
    cum_from_tau = -cum_from_right_rev[::-1]  # since diffs were negative

    expected_reward_at_grid = np.interp(taus, s, cum_from_tau)
    return sf, N * expected_reward_at_grid


def _state_to_group_stats(state: np.ndarray) -> dict[str, tuple[float, float, float]]:
    return {
        "A": (float(state[0]), float(state[1]), float(state[2])),
        "B": (float(state[3]), float(state[4]), float(state[5])),
    }


# ----------------------------------------------------------------- policies


class ProfitMaxThresholdPolicy:
    """Per-round expected-profit-maximising thresholds via grid search.

    Profit decouples across groups, so we optimise each group independently.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.cfg = LendingConfig.from_dict(config)
        self._taus = np.round(np.arange(0.0, 1.0 + GRID_STEP / 2, GRID_STEP), 2)

    def act(self, state: np.ndarray) -> np.ndarray:
        stats = _state_to_group_stats(state)
        tau_out = np.zeros(2, dtype=np.float32)
        for i, g in enumerate(GROUPS):
            mu, sigma, N = stats[g]
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
        return tau_out


class DemographicParityPolicy:
    """Profit-maximising thresholds subject to (approximate) equal approval rates.

    For each ``tau_A`` on the grid we pick the ``tau_B`` whose approval rate is
    closest to ``approval_rate_A(tau_A)``. Among candidate pairs whose absolute
    approval-rate difference is within ``tolerance`` we take the joint-profit
    argmax. If no pair clears the tolerance we fall back to the closest pair.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        tolerance: float = 1e-3,
    ):
        self.cfg = LendingConfig.from_dict(config)
        self.tolerance = float(tolerance)
        self._taus = np.round(np.arange(0.0, 1.0 + GRID_STEP / 2, GRID_STEP), 2)

    def act(self, state: np.ndarray) -> np.ndarray:
        stats = _state_to_group_stats(state)
        mu_A, sigma_A, N_A = stats["A"]
        mu_B, sigma_B, N_B = stats["B"]

        sf_A, profit_A = _group_curves(
            mu_A, sigma_A, N_A,
            k=self.cfg.k, s_0=self.cfg.s_0,
            u_repay=self.cfg.u_repay, u_default=self.cfg.u_default,
            taus=self._taus,
        )
        sf_B, profit_B = _group_curves(
            mu_B, sigma_B, N_B,
            k=self.cfg.k, s_0=self.cfg.s_0,
            u_repay=self.cfg.u_repay, u_default=self.cfg.u_default,
            taus=self._taus,
        )

        # For each tau_A index i, find tau_B index j minimising |sf_A[i] - sf_B[j]|.
        diff_matrix = np.abs(sf_A[:, None] - sf_B[None, :])  # (n_taus, n_taus)
        j_best = np.argmin(diff_matrix, axis=1)
        min_diffs = diff_matrix[np.arange(len(self._taus)), j_best]

        joint_profit = profit_A + profit_B[j_best]

        feasible = min_diffs <= self.tolerance
        if np.any(feasible):
            candidate_profit = np.where(feasible, joint_profit, -np.inf)
            i_star = int(np.argmax(candidate_profit))
        else:
            # Closest-feasible fallback: prefer smaller diff, break ties by profit.
            min_d = float(min_diffs.min())
            mask = min_diffs <= min_d + 1e-12
            candidate_profit = np.where(mask, joint_profit, -np.inf)
            i_star = int(np.argmax(candidate_profit))

        return np.array(
            [self._taus[i_star], self._taus[j_best[i_star]]], dtype=np.float32
        )


# ------------------------------------------------------------------ rollout


_LOG_KEYS: tuple[str, ...] = (
    "mu_A",
    "mu_B",
    "sigma_A",
    "sigma_B",
    "N_A",
    "N_B",
    "approval_rate_A",
    "approval_rate_B",
    "repay_rate_A",
    "repay_rate_B",
    "tau_A",
    "tau_B",
    "profit",
    "reward",
    "gap",
)


def simulate(
    policy: Policy,
    env: LendingEnv,
    n_seeds: int,
    n_rounds: int,
) -> dict[str, np.ndarray]:
    """Run ``policy`` through ``env`` for ``n_seeds`` episodes of ``n_rounds`` steps.

    Returns a dict of arrays. Per-round metrics are shaped ``(n_seeds, n_rounds)``;
    per-run summary scalars are shaped ``(n_seeds,)``.

    Per-round keys:
        ``mu_A, mu_B, sigma_A, sigma_B, N_A, N_B,
        approval_rate_A, approval_rate_B, repay_rate_A, repay_rate_B,
        tau_A, tau_B, profit, reward, gap``.

    Per-run summary keys:
        ``cumulative_gap`` -- sum of per-round ``|mu_A - mu_B|`` over the episode.
    """
    if n_rounds > env.cfg.horizon:
        raise ValueError(
            f"n_rounds={n_rounds} exceeds env horizon {env.cfg.horizon}."
        )

    logs: dict[str, list[list[float]]] = {key: [] for key in _LOG_KEYS}

    for seed in range(n_seeds):
        obs, _ = env.reset(seed=seed)
        seed_logs: dict[str, list[float]] = {key: [] for key in _LOG_KEYS}
        for _ in range(n_rounds):
            action = np.asarray(policy.act(obs), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            per_group = info["per_group"]
            seed_logs["mu_A"].append(info["mu_A"])
            seed_logs["mu_B"].append(info["mu_B"])
            seed_logs["sigma_A"].append(info["sigma_A"])
            seed_logs["sigma_B"].append(info["sigma_B"])
            seed_logs["N_A"].append(info["N_A"])
            seed_logs["N_B"].append(info["N_B"])
            seed_logs["approval_rate_A"].append(per_group["A"]["approval_rate"])
            seed_logs["approval_rate_B"].append(per_group["B"]["approval_rate"])
            seed_logs["repay_rate_A"].append(per_group["A"]["repay_rate"])
            seed_logs["repay_rate_B"].append(per_group["B"]["repay_rate"])
            seed_logs["tau_A"].append(per_group["A"]["threshold"])
            seed_logs["tau_B"].append(per_group["B"]["threshold"])
            seed_logs["profit"].append(info["profit"])
            seed_logs["reward"].append(reward)
            seed_logs["gap"].append(info["gap"])
            if terminated or truncated:
                break
        for k, v in seed_logs.items():
            logs[k].append(v)

    out: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.float64) for k, v in logs.items()
    }
    out["cumulative_gap"] = out["gap"].sum(axis=1)
    return out
