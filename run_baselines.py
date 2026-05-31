"""Run both baseline policies and produce the comparison figure.

Usage:
    python run_baselines.py [--out path/to/figure.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines import (
    DemographicParityPolicy,
    ProfitMaxThresholdPolicy,
    simulate,
)
from lending_env import LendingEnv
from plotting import plot_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="baseline_comparison.png",
        help="Output path for the figure.",
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=50)
    parser.add_argument(
        "--hmda-config",
        type=str,
        default="hmda_config.json",
        metavar="PATH",
        help="Path to hmda_config.json produced by hmda_calibrate.py.",
    )
    args = parser.parse_args()

    env_config: dict = {"horizon": args.n_rounds, "reward_mode": "profit"}
    config_path = Path(args.hmda_config)
    if config_path.exists():
        with open(config_path) as f:
            env_config.update(json.load(f))
        print(f"Loaded HMDA config from '{config_path}'")
    else:
        print(
            f"No config at '{config_path}'; using synthetic default group parameters."
        )

    env = LendingEnv(env_config)
    print(
        f"LendingEnv: G={env.G}, obs shape={env.observation_space.shape}, "
        f"action shape={env.action_space.shape}"
    )

    policies = {
        "ProfitMax": ProfitMaxThresholdPolicy(env_config),
        "DemographicParity": DemographicParityPolicy(env_config),
    }

    results = {}
    for name, policy in policies.items():
        print(f"Simulating {name}...")
        results[name] = simulate(policy, env, n_seeds=args.n_seeds, n_rounds=args.n_rounds)
        final_wvar = results[name]["wvar_mean"][:, -1].mean()
        cum_profit = results[name]["profit"].sum(axis=1).mean()
        print(
            f"  mean cum. profit = {cum_profit:.2f}, "
            f"mean final wvar(mean) = {final_wvar:.6f}"
        )

    plot_results(results, save_path=args.out, show=False)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
