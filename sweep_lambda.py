"""Sweep the fairness-penalty weight ``lambda`` for both baseline policies.

What this does and does not do
------------------------------

The non-RL baselines (`ProfitMaxThresholdPolicy`, `DemographicParityPolicy`)
choose thresholds from the state alone -- they do not consume the env reward.
Sweeping ``lambda`` therefore leaves their per-round actions unchanged. What
*does* change is the env's reward signal, which is what a PPO agent would see.

This script is useful for two things:

1. Picking a ``lambda`` for PPO training. Because ``|mu_A - mu_B| in [0, 1]``
   while profit scales with ``N``, the gap penalty needs to be sized
   appropriately. Inspecting the per-lambda reward distributions for the
   baselines tells you the order of magnitude at which the penalty actually
   shifts the reward signal.
2. Building a reference table for PPO comparisons later: baseline profit /
   gap / reward, by lambda and reward mode, mean +/- std across seeds.

Usage
-----

    python sweep_lambda.py
    python sweep_lambda.py --lambdas 0.1 1 10 100 1000 --reward-modes profit_dp profit_gap
    python sweep_lambda.py --out sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import numpy as np

from baselines import (
    DemographicParityPolicy,
    ProfitMaxThresholdPolicy,
    simulate,
)
from lending_env import LendingEnv

DEFAULT_LAMBDAS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_REWARD_MODES: tuple[str, ...] = ("profit_dp", "profit_gap")


@dataclass
class SweepRow:
    reward_mode: str
    lam: float
    policy: str
    mean_cum_profit: float
    std_cum_profit: float
    mean_cum_gap: float
    std_cum_gap: float
    mean_cum_reward: float
    std_cum_reward: float


def _summarise(name: str, lam: float, mode: str, logs: dict) -> SweepRow:
    cum_profit = logs["profit"].sum(axis=1)
    cum_gap = logs["cumulative_gap"]
    cum_reward = logs["reward"].sum(axis=1)
    return SweepRow(
        reward_mode=mode,
        lam=lam,
        policy=name,
        mean_cum_profit=float(cum_profit.mean()),
        std_cum_profit=float(cum_profit.std()),
        mean_cum_gap=float(cum_gap.mean()),
        std_cum_gap=float(cum_gap.std()),
        mean_cum_reward=float(cum_reward.mean()),
        std_cum_reward=float(cum_reward.std()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="Lambda values to sweep.",
    )
    parser.add_argument(
        "--reward-modes",
        type=str,
        nargs="+",
        default=list(DEFAULT_REWARD_MODES),
        choices=["profit", "profit_dp", "profit_gap"],
        help="Reward modes to evaluate (sweeping lambda is a no-op under 'profit').",
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=50)
    parser.add_argument(
        "--out", type=str, default="lambda_sweep.csv", help="CSV output path."
    )
    args = parser.parse_args()

    rows: list[SweepRow] = []

    for mode in args.reward_modes:
        for lam in args.lambdas:
            env_config = {
                "horizon": args.n_rounds,
                "reward_mode": mode,
                "lam": float(lam),
            }
            env = LendingEnv(env_config)
            policies = {
                "ProfitMax": ProfitMaxThresholdPolicy(env_config),
                "DemographicParity": DemographicParityPolicy(env_config),
            }
            for name, policy in policies.items():
                logs = simulate(
                    policy, env, n_seeds=args.n_seeds, n_rounds=args.n_rounds
                )
                row = _summarise(name, float(lam), mode, logs)
                rows.append(row)
                print(
                    f"mode={mode:<11} lam={lam:>8g}  policy={name:<18} "
                    f"cum_profit={row.mean_cum_profit:>8.1f}+/-{row.std_cum_profit:>5.1f}  "
                    f"cum_gap={row.mean_cum_gap:>6.3f}+/-{row.std_cum_gap:>5.3f}  "
                    f"cum_reward={row.mean_cum_reward:>9.1f}+/-{row.std_cum_reward:>6.1f}"
                )

    fieldnames = list(rows[0].__dict__.keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
