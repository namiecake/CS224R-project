"""Plotting helpers for comparing policies in the lending env."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from constants import GROUP_ORDER, RACE_COLORS, TIER_LINESTYLES, parse_group_name


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
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=1.5)
    ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=color)


def _group_style(name: str) -> tuple[str, str]:
    race, tier = parse_group_name(name)
    return RACE_COLORS[race], TIER_LINESTYLES[tier]


def _style_legend(ax: plt.Axes) -> None:
    """Add separate race-color and income-tier linestyle legends."""
    race_handles = [
        mlines.Line2D([], [], color=RACE_COLORS[r], linewidth=2, label=r)
        for r in RACE_COLORS
    ]
    tier_handles = [
        mlines.Line2D([], [], color="0.3", linestyle=TIER_LINESTYLES[t], linewidth=2, label=t)
        for t in TIER_LINESTYLES
    ]
    leg1 = ax.legend(handles=race_handles, title="Race", fontsize=7, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=tier_handles, title="Income tier", fontsize=7, loc="upper right")


def plot_results(
    results: Mapping[str, Mapping[str, np.ndarray]],
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render a 2x2 comparison figure across policies.

    Panels:
        (a) group score means over time (one line per group)
        (b) population-weighted variance of group means (+ max pairwise gap)
        (c) cumulative profit per policy
        (d) approval rate per group
    """
    if not results:
        raise ValueError("`results` is empty.")

    # Infer which groups are present from the first policy's logs.
    sample_logs = next(iter(results.values()))
    groups = [
        name
        for name in GROUP_ORDER
        if f"mu_{name}" in sample_logs
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    ax_mu, ax_gap, ax_profit, ax_appr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    cmap = plt.get_cmap("tab10")
    policy_colors = {name: cmap(i % 10) for i, name in enumerate(results)}

    for policy_name, logs in results.items():
        color = policy_colors[policy_name]

        # Use first group's mu to determine horizon length.
        T = logs[f"mu_{groups[0]}"].shape[1]
        rounds = np.arange(1, T + 1)

        for gname in groups:
            gcolor, gstyle = _group_style(gname)
            mu_mean, mu_std = _mean_std(logs[f"mu_{gname}"])
            _plot_with_band(
                ax_mu, rounds, mu_mean, mu_std, gname, gcolor, gstyle
            )

            appr_mean, appr_std = _mean_std(logs[f"approval_rate_{gname}"])
            _plot_with_band(
                ax_appr, rounds, appr_mean, appr_std, gname, gcolor, gstyle
            )

        wvar_mean, wvar_std = _mean_std(logs["wvar_mean"])
        _plot_with_band(
            ax_gap, rounds, wvar_mean, wvar_std,
            f"{policy_name} (wvar)", color, "-",
        )
        gap_mean, gap_std = _mean_std(logs["pairwise_gap"])
        _plot_with_band(
            ax_gap, rounds, gap_mean, gap_std,
            f"{policy_name} (max gap)", color, "--",
        )

        cum_profit = np.cumsum(logs["profit"], axis=1)
        cp_mean, cp_std = _mean_std(cum_profit)
        _plot_with_band(ax_profit, rounds, cp_mean, cp_std, policy_name, color)

    ax_mu.set_title("(a) Group score means over time")
    ax_mu.set_ylabel("score mean")
    _style_legend(ax_mu)

    ax_gap.set_title("(b) Equity metrics over time")
    ax_gap.set_ylabel("variance / gap")
    ax_gap.legend(fontsize=7, loc="best")

    ax_profit.set_title("(c) Cumulative profit")
    ax_profit.set_xlabel("round")
    ax_profit.set_ylabel("cumulative profit")
    ax_profit.legend(fontsize=8, loc="best")

    ax_appr.set_title("(d) Approval rate per group")
    ax_appr.set_xlabel("round")
    ax_appr.set_ylabel("approval rate")
    ax_appr.set_ylim(-0.02, 1.02)
    _style_legend(ax_appr)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig
