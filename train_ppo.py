"""Train single PPO agent on LendingEnv

Usage:
    python train_ppo.py --reward-mode profit_wvar_approval --lam 5.0 --seed 0
    python train_ppo.py --reward-mode profit --lam 1.0 --seed 0 --timesteps 50000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from lending_env import LendingEnv

DEFAULT_TIMESTEPS = 200_000
DEFAULT_OUT_DIR = "ppo_results"

def build_env_config(
    reward_mode: str,
    lam: float,
    hmda_config_path: str = "hmda_config.json",
) -> dict:
    """build LendingEnv config, layer HMDA params under reward/lam overrides"""
    cfg: dict = {"reward_mode": reward_mode, "lam": lam, "horizon": 50}
    p = Path(hmda_config_path)
    if p.exists():
        with open(p) as f:
            cfg.update(json.load(f))
        # so hdma config can't ruin reward shaping
        cfg["reward_mode"] = reward_mode
        cfg["lam"] = lam
    return cfg


def train(
    reward_mode: str,
    lam: float,
    seed: int,
    timesteps: int = DEFAULT_TIMESTEPS,
    out_dir: str = DEFAULT_OUT_DIR,
    hmda_config_path: str = "hmda_config.json",
) -> Path:
    """train single PPO agent-->return path to saved model

    skips training if model already exists
    saves both SB3 model and VecNormalize stats so the policy can be
    loaded for inference without regenerating training env
    """
    run_name = f"ppo_{reward_mode}_lam{lam}_seed{seed}"
    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model"
    vecnorm_path = run_dir / "vecnorm.pkl"

    if model_path.with_suffix(".zip").exists():
        print(f"[skip] {run_name} already trained.")
        return model_path

    env_config = build_env_config(reward_mode, lam, hmda_config_path)

    vec_env = make_vec_env(lambda: LendingEnv(env_config), n_envs=1, seed=seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )

    print(f"[train] {run_name}  timesteps={timesteps:,}")
    model.learn(total_timesteps=timesteps)

    model.save(str(model_path))
    vec_env.save(str(vecnorm_path))
    # save policy weights as plain .pt
    import torch
    torch.save(model.policy.state_dict(), str(run_dir / "policy.pt"))
    print(f"[done]  saved {model_path}.zip + {vecnorm_path}")

    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="profit",
        choices=["profit", "profit_wvar_approval", "profit_wvar_mean"],
    )
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hmda-config", type=str, default="hmda_config.json")
    args = parser.parse_args()

    train(
        reward_mode=args.reward_mode,
        lam=args.lam,
        seed=args.seed,
        timesteps=args.timesteps,
        out_dir=args.out_dir,
        hmda_config_path=args.hmda_config,
    )


if __name__ == "__main__":
    main()
