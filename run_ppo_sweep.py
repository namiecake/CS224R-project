"""run full ppo sweep (3 reward modes × 5 lambdas × 5 seeds = 75 runs)

training --> evaluate all PPO models against both non-RL baselines -->
write results to CSV

Usage:
    python run_ppo_sweep.py                       # 75-run sweep
    python run_ppo_sweep.py --mini                # mini run (1 mode, 1 lambda, 1 seed)
    python run_ppo_sweep.py --eval-only           # skip training, evaluate saved models
    python run_ppo_sweep.py --timesteps 50000     # override training budget before run
    python run_ppo_sweep.py --lambdas 1 5 10      # custom lambda grid
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from baselines import DemographicParityPolicy, ProfitMaxThresholdPolicy, simulate
from lending_env import LendingEnv
from train_ppo import DEFAULT_OUT_DIR, DEFAULT_TIMESTEPS, build_env_config, train

REWARD_MODES = ["profit", "profit_wvar_approval", "profit_wvar_mean"]
LAMBDAS = [1.0, 2.0, 5.0, 7.0, 10.0]
SEEDS = list(range(5))
N_EVAL_SEEDS = 5
N_ROUNDS = 50


# ------------------------------------------------------------------


class PPOPolicy:
    """Wrap trained PPO policy to match Policy protocol in baselines.py.

    load policy weights from plain policy.pt file
    load VecNormalize stats separately (obs normalised identically to
    training w/o needing training live env)
    """

    def __init__(self, policy_pt_path: str, vecnorm_path: str, env_config: dict):
        env = LendingEnv(env_config)
        self._policy = ActorCriticPolicy(
            env.observation_space,
            env.action_space,
            lr_schedule=lambda _: 3e-4,
        )
        state = torch.load(policy_pt_path, map_location="cpu", weights_only=True)
        self._policy.load_state_dict(state)
        self._policy.set_training_mode(False)

        dummy = DummyVecEnv([lambda: LendingEnv(env_config)])
        self._vn = VecNormalize.load(vecnorm_path, dummy)
        self._vn.training = False
        self._vn.norm_reward = False

    def act(self, state: np.ndarray) -> np.ndarray:
        obs = self._vn.normalize_obs(state.reshape(1, -1).astype(np.float32))
        obs_tensor = torch.FloatTensor(obs)
        with torch.no_grad():
            action, _, _ = self._policy.forward(obs_tensor, deterministic=True)
        return action.squeeze(0).numpy().astype(np.float32)


# ------------------------------------------------------------------


@dataclass
class ResultRow:
    reward_mode: str
    lam: float
    policy: str
    mean_cum_profit: float
    std_cum_profit: float
    mean_final_wvar: float
    std_final_wvar: float
    mean_pairwise_gap: float
    std_pairwise_gap: float
    mean_cum_reward: float
    std_cum_reward: float


def _summarise(reward_mode: str, lam: float, policy_name: str, logs: dict) -> ResultRow:
    cum_profit = logs["profit"].sum(axis=1)
    final_wvar = logs["wvar_mean"][:, -1]
    final_gap = logs["pairwise_gap"][:, -1]
    cum_reward = logs["reward"].sum(axis=1)
    return ResultRow(
        reward_mode=reward_mode,
        lam=lam,
        policy=policy_name,
        mean_cum_profit=float(cum_profit.mean()),
        std_cum_profit=float(cum_profit.std()),
        mean_final_wvar=float(final_wvar.mean()),
        std_final_wvar=float(final_wvar.std()),
        mean_pairwise_gap=float(final_gap.mean()),
        std_pairwise_gap=float(final_gap.std()),
        mean_cum_reward=float(cum_reward.mean()),
        std_cum_reward=float(cum_reward.std()),
    )


# ------------------------------------------------------------------ eval


def eval_ppo(
    reward_mode: str,
    lam: float,
    seeds: list[int],
    out_dir: str,
    hmda_config_path: str,
) -> ResultRow:
    """load every seed model, run one eval ep each, aggregate across"""
    env_config = build_env_config(reward_mode, lam, hmda_config_path)
    env = LendingEnv(env_config)

    seed_log_lists: dict[str, list[np.ndarray]] = {}

    for seed in seeds:
        run_name = f"ppo_{reward_mode}_lam{lam}_seed{seed}"
        run_dir = Path(out_dir) / run_name
        vecnorm_pkl = run_dir / "vecnorm.pkl"

        policy_pt = run_dir / "policy.pt"
        if not policy_pt.exists():
            print(f"  [warn] missing {policy_pt}, skipping seed {seed}.")
            continue

        policy = PPOPolicy(str(policy_pt), str(vecnorm_pkl), env_config)
        # n_seeds=1: 1 env episode per model, comes from env
        logs = simulate(policy, env, n_seeds=1, n_rounds=N_ROUNDS)

        for key, arr in logs.items():
            seed_log_lists.setdefault(key, []).append(arr)

    if not seed_log_lists:
        raise RuntimeError(f"No trained PPO models found for ({reward_mode}, lam={lam})")

    # stack: each arr=(1, T), concatenate along axis 0 (n_seeds, T)
    merged = {k: np.concatenate(v, axis=0) for k, v in seed_log_lists.items()}
    return _summarise(reward_mode, lam, f"PPO_{reward_mode}", merged)


def eval_baselines(
    reward_mode: str,
    lam: float,
    hmda_config_path: str,
) -> list[ResultRow]:
    env_config = build_env_config(reward_mode, lam, hmda_config_path)
    env = LendingEnv(env_config)
    rows = []
    for name, PolicyClass in [
        ("ProfitMax", ProfitMaxThresholdPolicy),
        ("DemographicParity", DemographicParityPolicy),
    ]:
        logs = simulate(PolicyClass(env_config), env, n_seeds=N_EVAL_SEEDS, n_rounds=N_ROUNDS)
        rows.append(_summarise(reward_mode, lam, name, logs))
    return rows


# ------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Smoke test: 1 reward mode × 1 lambda × 1 seed × 10k steps.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate already-saved models.",
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-csv", type=str, default="ppo_sweep_results.csv")
    parser.add_argument("--hmda-config", type=str, default="hmda_config.json")
    parser.add_argument(
        "--reward-modes", nargs="+", default=REWARD_MODES, choices=REWARD_MODES
    )
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    reward_modes = args.reward_modes
    lambdas = args.lambdas
    seeds = args.seeds
    timesteps = args.timesteps

    if args.mini:
        reward_modes = ["profit_wvar_approval"]
        lambdas = [1.0]
        seeds = [0]
        timesteps = 10_000
        print("[mini] Smoke test: 1 mode × 1 lambda × 1 seed × 10,000 steps")

    # training
    if not args.eval_only:
        total = len(reward_modes) * len(lambdas) * len(seeds)
        done = 0
        for mode in reward_modes:
            for lam in lambdas:
                for seed in seeds:
                    train(
                        reward_mode=mode,
                        lam=lam,
                        seed=seed,
                        timesteps=timesteps,
                        out_dir=args.out_dir,
                        hmda_config_path=args.hmda_config,
                    )
                    done += 1
                    print(f"  [{done}/{total}] training runs complete\n")

    # eval
    all_rows: list[ResultRow] = []

    for mode in reward_modes:
        for lam in lambdas:
            print(f"  PPO  mode={mode}  lam={lam}")
            try:
                ppo_row = eval_ppo(mode, lam, seeds, args.out_dir, args.hmda_config)
                all_rows.append(ppo_row)
                print(
                    f"    cum_profit={ppo_row.mean_cum_profit:.1f}±{ppo_row.std_cum_profit:.1f}  "
                    f"final_wvar={ppo_row.mean_final_wvar:.4f}±{ppo_row.std_final_wvar:.4f}"
                )
            except RuntimeError as e:
                print(f"    [warn] {e}")

            print(f"  baselines mode={mode}  lam={lam}")
            for row in eval_baselines(mode, lam, args.hmda_config):
                all_rows.append(row)
                print(
                    f"    {row.policy:<20} cum_profit={row.mean_cum_profit:.1f}±{row.std_cum_profit:.1f}  "
                    f"final_wvar={row.mean_final_wvar:.4f}±{row.std_final_wvar:.4f}"
                )

    # write csv
    if all_rows:
        fieldnames = [f.name for f in fields(ResultRow)]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(vars(row))
        print(f"\nWrote {len(all_rows)} rows: {args.out_csv}")


if __name__ == "__main__":
    main()
