"""Diagnose PPO policy after action-space fix.

Trains a fresh agent for 50K steps with reward_mode="profit", rolls out one
episode, and prints per-step thresholds (post-mapping) and per-group approval
rates. Then runs two verification checks:
  1. profit-only cumulative profit within ~5% of ProfitMax (~1.17M).
  2. profit_wvar_mean with lam=1e6 achieves meaningfully lower final_wvar.

Usage:
    python diagnose_policy.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from baselines import ProfitMaxThresholdPolicy, simulate
from constants import GROUP_ORDER
from lending_env import LendingEnv
from train_ppo import build_env_config

DIAG_DIR = Path("ppo_diag")
TIMESTEPS = 50_000
SEED = 0
N_ROUNDS = 50


def train_fresh(reward_mode: str, lam: float, tag: str) -> tuple[PPO, VecNormalize]:
    """Train a fresh PPO agent and return (model, vec_env)."""
    run_dir = DIAG_DIR / tag
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    env_config = build_env_config(reward_mode, lam)
    vec_env = make_vec_env(lambda: LendingEnv(env_config), n_envs=1, seed=SEED)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=0,
        seed=SEED,
    )
    print(f"[train] {tag}  timesteps={TIMESTEPS:,}")
    model.learn(total_timesteps=TIMESTEPS)
    print(f"[done]  {tag}")
    return model, vec_env


def rollout_and_print(model: PPO, vec_env: VecNormalize, env_config: dict) -> dict:
    """Roll out one episode, print per-step thresholds and approval rates."""
    env = LendingEnv(env_config)
    obs, _ = env.reset(seed=SEED)

    header = f"{'t':>3}"
    for name in GROUP_ORDER:
        header += f" | tau_{name[:6]:>6}"
    for name in GROUP_ORDER:
        header += f" | app_{name[:6]:>6}"
    header += f" | {'profit':>8}"
    print(header)
    print("-" * len(header))

    cum_profit = 0.0
    all_thresholds = []

    for t in range(N_ROUNDS):
        obs_norm = vec_env.normalize_obs(obs.reshape(1, -1).astype(np.float32))
        action, _ = model.predict(obs_norm, deterministic=True)
        action = action.flatten()

        obs, reward, term, trunc, info = env.step(action)
        per_group = info["per_group"]
        cum_profit += info["profit"]

        thresholds = [per_group[name]["threshold"] for name in GROUP_ORDER]
        approvals = [per_group[name]["approval_rate"] for name in GROUP_ORDER]
        all_thresholds.append(thresholds)

        row = f"{t:>3}"
        for tau in thresholds:
            row += f" | {tau:>10.3f}"
        for ar in approvals:
            row += f" | {ar:>10.3f}"
        row += f" | {info['profit']:>8.0f}"
        print(row)

        if term or trunc:
            break

    all_thresholds = np.array(all_thresholds)
    print(f"\nCumulative profit: {cum_profit:.0f}")
    print(f"Mean threshold per group: {all_thresholds.mean(axis=0)}")
    print(f"Std  threshold per group: {all_thresholds.std(axis=0)}")

    return {
        "cum_profit": cum_profit,
        "final_wvar": info["wvar_mean"],
        "thresholds": all_thresholds,
    }


def check_non_degenerate(thresholds: np.ndarray) -> bool:
    """Confirm thresholds aren't stuck at boundaries."""
    mean_tau = thresholds.mean()
    at_low = (thresholds < 0.05).mean()
    at_high = (thresholds > 0.95).mean()
    ok = at_low < 0.8 and at_high < 0.8 and 0.1 < mean_tau < 0.9
    if ok:
        print(f"\n[PASS] Thresholds non-degenerate (mean={mean_tau:.3f}, "
              f"{at_low*100:.0f}% near 0, {at_high*100:.0f}% near 1)")
    else:
        print(f"\n[FAIL] Thresholds appear degenerate (mean={mean_tau:.3f}, "
              f"{at_low*100:.0f}% near 0, {at_high*100:.0f}% near 1)")
    return ok


def main() -> None:
    # --- Check 1: profit-only PPO vs ProfitMax ---
    print("=" * 70)
    print("CHECK 1: Train profit-only PPO, compare to ProfitMax baseline")
    print("=" * 70)

    env_config_profit = build_env_config("profit", 1.0)
    model_profit, vec_env_profit = train_fresh("profit", 1.0, "profit_only")

    print("\n--- Rollout: PPO (profit-only) ---")
    result_profit = rollout_and_print(model_profit, vec_env_profit, env_config_profit)

    ok_nondegen = check_non_degenerate(result_profit["thresholds"])

    env_bl = LendingEnv(env_config_profit)
    bl_logs = simulate(ProfitMaxThresholdPolicy(env_config_profit), env_bl, n_seeds=5, n_rounds=N_ROUNDS)
    profitmax_cum = bl_logs["profit"].sum(axis=1).mean()
    print(f"\nProfitMax baseline cumulative profit (5-seed avg): {profitmax_cum:.0f}")

    ppo_profit = result_profit["cum_profit"]
    ratio = ppo_profit / profitmax_cum if profitmax_cum > 0 else 0
    within_5pct = ratio >= 0.95
    print(f"PPO cumulative profit: {ppo_profit:.0f}")
    print(f"PPO / ProfitMax ratio: {ratio:.3f}")
    if within_5pct:
        print("[PASS] PPO profit within 5% of ProfitMax")
    else:
        print(f"[FAIL] PPO profit ratio {ratio:.3f} < 0.95")

    # --- Check 2: profit_wvar_mean with high lambda ---
    print("\n" + "=" * 70)
    print("CHECK 2: Train profit_wvar_mean (lam=1e6), compare final_wvar")
    print("=" * 70)

    lam_high = 1_000_000.0
    env_config_wvar = build_env_config("profit_wvar_mean", lam_high)
    model_wvar, vec_env_wvar = train_fresh("profit_wvar_mean", lam_high, "wvar_high_lam")

    print("\n--- Rollout: PPO (profit_wvar_mean, lam=1e6) ---")
    result_wvar = rollout_and_print(model_wvar, vec_env_wvar, env_config_wvar)

    wvar_profit_only = result_profit["final_wvar"]
    wvar_penalized = result_wvar["final_wvar"]
    print(f"\nfinal_wvar (profit-only PPO): {wvar_profit_only:.6f}")
    print(f"final_wvar (lam=1e6 PPO):     {wvar_penalized:.6f}")
    meaningfully_lower = wvar_penalized < wvar_profit_only * 0.8
    if meaningfully_lower:
        print("[PASS] lam=1e6 achieves meaningfully lower final_wvar")
    else:
        print(f"[FAIL] wvar not meaningfully lower ({wvar_penalized:.6f} vs {wvar_profit_only:.6f})")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = ok_nondegen and within_5pct and meaningfully_lower
    checks = [
        ("Non-degenerate thresholds", ok_nondegen),
        ("PPO profit within 5% of ProfitMax", within_5pct),
        ("High-lambda achieves lower wvar", meaningfully_lower),
    ]
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    if all_pass:
        print("\nAll checks passed. Safe to run full sweep.")
    else:
        print("\nSome checks FAILED. Do NOT run the full sweep until resolved.")


if __name__ == "__main__":
    main()
