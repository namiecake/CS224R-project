"""Plotting helpers for comparing policies in the lending env."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


def _mean_std(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seed-wise mean and std along axis 0, ignoring NaNs."""
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
    return mean, std


def _plot_with_band(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle)
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)


def plot_results(
    results: Mapping[str, Mapping[str, np.ndarray]],
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render a 2x2 comparison figure across policies.

    Panels:
        (a) ``mu_A`` and ``mu_B`` over time per policy
        (b) inter-group gap ``|mu_A - mu_B|`` over time per policy
        (c) cumulative profit over time per policy
        (d) approval rate per group over time per policy

    Each metric is mean +/- 1 std across seeds.
    """
    if not results:
        raise ValueError("`results` is empty.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_mu, ax_gap, ax_profit, ax_appr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, name in enumerate(results)}

    for name, logs in results.items():
        color = colors[name]

        mu_A, std_A = _mean_std(logs["mu_A"])
        mu_B, std_B = _mean_std(logs["mu_B"])
        T = mu_A.shape[0]
        rounds = np.arange(1, T + 1)

        _plot_with_band(ax_mu, rounds, mu_A, std_A, f"{name} (A)", color, "-")
        _plot_with_band(ax_mu, rounds, mu_B, std_B, f"{name} (B)", color, "--")

        gap_mean, gap_std = _mean_std(logs["gap"])
        _plot_with_band(ax_gap, rounds, gap_mean, gap_std, name, color)

        cum_profit = np.cumsum(logs["profit"], axis=1)
        cp_mean, cp_std = _mean_std(cum_profit)
        _plot_with_band(ax_profit, rounds, cp_mean, cp_std, name, color)

        appr_A, appr_A_std = _mean_std(logs["approval_rate_A"])
        appr_B, appr_B_std = _mean_std(logs["approval_rate_B"])
        _plot_with_band(ax_appr, rounds, appr_A, appr_A_std, f"{name} (A)", color, "-")
        _plot_with_band(ax_appr, rounds, appr_B, appr_B_std, f"{name} (B)", color, "--")

    ax_mu.set_title("(a) Group score means over time")
    ax_mu.set_ylabel("score mean")
    ax_mu.legend(fontsize=8, loc="best")

    ax_gap.set_title("(b) Inter-group gap |mu_A - mu_B|")
    ax_gap.set_ylabel("gap")
    ax_gap.legend(fontsize=8, loc="best")

    ax_profit.set_title("(c) Cumulative profit")
    ax_profit.set_xlabel("round")
    ax_profit.set_ylabel("cumulative profit")
    ax_profit.legend(fontsize=8, loc="best")

    ax_appr.set_title("(d) Approval rate per group")
    ax_appr.set_xlabel("round")
    ax_appr.set_ylabel("approval rate")
    ax_appr.set_ylim(-0.02, 1.02)
    ax_appr.legend(fontsize=8, loc="best")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig
