"""Generate poster-quality figures from PPO sweep results."""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR = HERE / "ppo_results_mon" / "ppo_results_stats"

LAMBDA_MIN = 10
LAMBDA_MAX = 10_000

# -- Poster style defaults --
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.dpi": 200,
})

# -- Load data from all CSVs in ppo_results_mon/ppo_results_stats/ --
csv_files = sorted(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
all_data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print(f"Loaded {len(csv_files)} CSV files, {len(all_data)} total rows")

# -- Extract baselines (constant across lambdas) --
pmax = all_data[all_data["policy"] == "ProfitMax"].iloc[0]
dp = all_data[all_data["policy"] == "DemographicParity"].iloc[0]

has_ppo_profit = "PPO_profit" in all_data["policy"].values
if has_ppo_profit:
    ppo_profit = all_data[all_data["policy"] == "PPO_profit"].iloc[0]

# -- PPO with fairness penalties (one row per lambda) --
mean_ppo = (
    all_data[all_data["policy"] == "PPO_profit_wvar_mean"]
    .query(f"lam >= {LAMBDA_MIN} and lam <= {LAMBDA_MAX}")
    .sort_values("lam")
    .drop_duplicates(subset="lam")
    .reset_index(drop=True)
)

appr_ppo = (
    all_data[all_data["policy"] == "PPO_profit_wvar_approval"]
    .query(f"lam >= {LAMBDA_MIN} and lam <= {LAMBDA_MAX}")
    .sort_values("lam")
    .drop_duplicates(subset="lam")
    .reset_index(drop=True)
)

print(f"mean_ppo lambdas: {mean_ppo['lam'].tolist()}")
print(f"appr_ppo lambdas: {appr_ppo['lam'].tolist()}")


def fmt_lam(lam: float) -> str:
    if lam >= 1_000_000:
        return f"{lam / 1e6:.0f}M"
    if lam >= 1_000:
        return f"{lam / 1e3:.0f}k"
    return f"{lam:.0f}"


COLORS = {"ppo": "#5B21B6", "pmax": "#DC2626", "dp": "#2563EB", "nofair": "#16A34A"}

# Build lambda colour map from actual lambdas present in the data
_all_lams = sorted(set(mean_ppo["lam"].tolist() + appr_ppo["lam"].tolist()))
_palette = ["#2563EB", "#7C3AED", "#DB2777", "#EA580C", "#EAB308", "#DC2626"]
LAM_COLORS = {
    lam: _palette[i % len(_palette)] for i, lam in enumerate(_all_lams)
}

# ============================================================================
# Figure 1: Pareto tradeoff — fairness vs. profit (two panels)
# ============================================================================
fig1, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

mean_ppo_pareto = mean_ppo[mean_ppo["lam"] < 1_000_000].reset_index(drop=True)
appr_ppo_pareto = appr_ppo[appr_ppo["lam"] < 1_000_000].reset_index(drop=True)

for ax, df, subtitle in [
    (ax_l, mean_ppo_pareto, r"Reward: profit $-\;\lambda\,\cdot$ wvar(group means)"),
    (ax_r, appr_ppo_pareto, r"Reward: profit $-\;\lambda\,\cdot$ wvar(approval rates)"),
]:
    for _, r in df.iterrows():
        lam_val = r["lam"]
        ax.scatter(
            r["mean_final_wvar"], r["mean_cum_profit"] / 1e6,
            color=LAM_COLORS.get(lam_val, "#888888"), s=80, edgecolor="black",
            linewidth=0.6, zorder=3,
        )

    if has_ppo_profit:
        ax.scatter(
            ppo_profit["mean_final_wvar"], ppo_profit["mean_cum_profit"] / 1e6,
            marker="^", s=120, color=COLORS["nofair"], edgecolor="black",
            linewidth=0.8, zorder=5,
        )
    ax.scatter(
        pmax["mean_final_wvar"], pmax["mean_cum_profit"] / 1e6,
        marker="D", s=90, color=COLORS["pmax"], edgecolor="black",
        linewidth=0.7, zorder=4,
    )
    ax.scatter(
        dp["mean_final_wvar"], dp["mean_cum_profit"] / 1e6,
        marker="D", s=90, color=COLORS["dp"], edgecolor="black",
        linewidth=0.7, zorder=4,
    )

    ax.set_xlabel("Fairness (variance in creditworthiness scores)")
    ax.set_title(subtitle, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, linewidth=0.5)

ax_l.set_ylabel("Cumulative profit (USD in millions)")

legend_handles = []
for lam_val in sorted(LAM_COLORS):
    if lam_val >= 1_000_000:
        continue
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LAM_COLORS[lam_val],
               markeredgecolor="black", markeredgewidth=0.6, markersize=8,
               label=f"PPO + fairness (λ={fmt_lam(lam_val)})"))
if has_ppo_profit:
    legend_handles.append(
        Line2D([0], [0], marker="^", color="w", markerfacecolor=COLORS["nofair"],
               markeredgecolor="black", markeredgewidth=0.8, markersize=8,
               label="PPO (profit only)"))
legend_handles.append(
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["pmax"],
           markeredgecolor="black", markeredgewidth=0.7, markersize=8,
           label="Profit-maximizing baseline"))
legend_handles.append(
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["dp"],
           markeredgecolor="black", markeredgewidth=0.7, markersize=8,
           label="Demographic parity baseline"))
ax_r.legend(handles=legend_handles, loc="lower right", framealpha=0.92, edgecolor="0.8",
            fontsize=8.5)

fig1.tight_layout()
fig1.savefig(FIG_DIR / "poster_pareto.png", dpi=200, bbox_inches="tight")
print(f"Saved {FIG_DIR / 'poster_pareto.png'}")


# ============================================================================
# Figure 2: Effect of lambda on fairness and profit (2x2 grid)
# ============================================================================
fig2, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="col")

for col, (df, label) in enumerate([
    (mean_ppo, "wvar(group means) penalty"),
    (appr_ppo, "wvar(approval rates) penalty"),
]):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    lam = df["lam"].values

    ax_top.errorbar(
        lam, df["mean_final_wvar"], yerr=df["std_final_wvar"],
        fmt="o-", color=COLORS["ppo"], capsize=3, linewidth=2, markersize=7,
        label="PPO + fairness",
    )
    if has_ppo_profit:
        ax_top.axhline(ppo_profit["mean_final_wvar"], color=COLORS["nofair"],
                       ls="--", lw=1.8, label="PPO (profit only)")
    ax_top.axhline(pmax["mean_final_wvar"], color=COLORS["pmax"],
                   ls=":", lw=1.5, label="Profit-max")
    ax_top.axhline(dp["mean_final_wvar"], color=COLORS["dp"],
                   ls=":", lw=1.5, label="Demo. parity")
    ax_top.set_xscale("log")
    ax_top.set_ylabel("Final weighted var.\nof group means")
    ax_top.set_title(label, fontsize=13, fontweight="bold")
    ax_top.grid(alpha=0.25, linewidth=0.5)
    if col == 0:
        ax_top.legend(loc="upper left", fontsize=9, framealpha=0.92, edgecolor="0.8")

    ax_bot.errorbar(
        lam, df["mean_cum_profit"] / 1e6, yerr=df["std_cum_profit"] / 1e6,
        fmt="o-", color=COLORS["ppo"], capsize=3, linewidth=2, markersize=7,
    )
    if has_ppo_profit:
        ax_bot.axhline(ppo_profit["mean_cum_profit"] / 1e6, color=COLORS["nofair"],
                       ls="--", lw=1.8)
    ax_bot.axhline(pmax["mean_cum_profit"] / 1e6, color=COLORS["pmax"],
                   ls=":", lw=1.5)
    ax_bot.axhline(dp["mean_cum_profit"] / 1e6, color=COLORS["dp"],
                   ls=":", lw=1.5)
    ax_bot.set_xscale("log")
    ax_bot.set_xlabel(r"$\lambda$ (fairness penalty weight)")
    ax_bot.set_ylabel("Cumulative profit\n(millions)")
    ax_bot.grid(alpha=0.25, linewidth=0.5)

fig2.tight_layout()
fig2.savefig(FIG_DIR / "poster_lambda_sweep.png", dpi=200, bbox_inches="tight")
print(f"Saved {FIG_DIR / 'poster_lambda_sweep.png'}")


# ============================================================================
# Figure 3: Bar chart — best PPO+fairness vs baselines
# ============================================================================
best_mean = mean_ppo.loc[mean_ppo["mean_cum_profit"].idxmax()]
best_appr = appr_ppo.loc[appr_ppo["mean_cum_profit"].idxmax()]

policies = [
    ("Demo. parity\nbaseline", dp["mean_cum_profit"] / 1e6, dp["std_cum_profit"] / 1e6,
     dp["mean_final_wvar"], dp["std_final_wvar"]),
    ("Profit-max\nbaseline", pmax["mean_cum_profit"] / 1e6, pmax["std_cum_profit"] / 1e6,
     pmax["mean_final_wvar"], pmax["std_final_wvar"]),
]
bar_colors = [COLORS["dp"], COLORS["pmax"]]

if has_ppo_profit:
    policies.append(
        ("PPO\n(profit only)", ppo_profit["mean_cum_profit"] / 1e6, ppo_profit["std_cum_profit"] / 1e6,
         ppo_profit["mean_final_wvar"], ppo_profit["std_final_wvar"]))
    bar_colors.append(COLORS["nofair"])

policies.extend([
    (f"PPO + means\n(λ={fmt_lam(best_mean['lam'])})",
     best_mean["mean_cum_profit"] / 1e6, best_mean["std_cum_profit"] / 1e6,
     best_mean["mean_final_wvar"], best_mean["std_final_wvar"]),
    (f"PPO + approval\n(λ={fmt_lam(best_appr['lam'])})",
     best_appr["mean_cum_profit"] / 1e6, best_appr["std_cum_profit"] / 1e6,
     best_appr["mean_final_wvar"], best_appr["std_final_wvar"]),
])
bar_colors.extend([COLORS["ppo"], COLORS["ppo"]])

names = [p[0] for p in policies]
profits = [p[1] for p in policies]
profit_errs = [p[2] for p in policies]
wvars = [p[3] for p in policies]
wvar_errs = [p[4] for p in policies]

fig3, (ax_p, ax_w) = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(names))
bar_w = 0.55

ax_p.bar(x, profits, width=bar_w, yerr=profit_errs, capsize=4, color=bar_colors,
         edgecolor="black", linewidth=0.6, alpha=0.85)
ax_p.set_xticks(x)
ax_p.set_xticklabels(names, fontsize=9)
ax_p.set_ylabel("Cumulative profit (millions)")
ax_p.set_title("Profit comparison", fontsize=13, fontweight="bold")
ax_p.grid(axis="y", alpha=0.25, linewidth=0.5)
ax_p.set_axisbelow(True)

ax_w.bar(x, wvars, width=bar_w, yerr=wvar_errs, capsize=4, color=bar_colors,
         edgecolor="black", linewidth=0.6, alpha=0.85)
ax_w.set_xticks(x)
ax_w.set_xticklabels(names, fontsize=9)
ax_w.set_ylabel("Final weighted var. of group means")
ax_w.set_title("Fairness comparison", fontsize=13, fontweight="bold")
ax_w.grid(axis="y", alpha=0.25, linewidth=0.5)
ax_w.set_axisbelow(True)
wvar_min = min(wvars) - max(wvar_errs) * 1.5
wvar_max = max(wvars) + max(wvar_errs) * 2
ax_w.set_ylim(wvar_min * 0.95, wvar_max * 1.05)

fig3.tight_layout()
fig3.savefig(FIG_DIR / "poster_bar_comparison.png", dpi=200, bbox_inches="tight")
print(f"Saved {FIG_DIR / 'poster_bar_comparison.png'}")

plt.close("all")
