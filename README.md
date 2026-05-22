# Lending Fairness Simulator

A Gymnasium environment for studying long-horizon fairness in mortgage lending,
together with two non-RL baseline threshold policies and a small plotting and
rollout harness. Designed to plug into Stable-Baselines3 PPO later, and to
accept externally-calibrated initial distribution parameters (e.g. from HMDA).

## Files

- `lending_env.py` &mdash; `LendingEnv`, a `gymnasium.Env` for two demographic
  groups (A, B). State is `[mu_A, sigma_A, N_A, mu_B, sigma_B, N_B]`; action is
  `[tau_A, tau_B]` in `[0, 1]^2`. Hyperparameters are passed via a `config`
  dict and validated against `LendingConfig`.
- `baselines.py` &mdash; `ProfitMaxThresholdPolicy`, `DemographicParityPolicy`,
  and `simulate(policy, env, n_seeds, n_rounds)` for non-RL rollouts.
- `plotting.py` &mdash; `plot_results(results, save_path=...)` produces the 2x2
  comparison figure (mu_g, gap, cumulative profit, approval rates).
- `run_baselines.py` &mdash; entry point that runs both baselines over 5 seeds and
  50 rounds with default config and saves the figure.
- `sweep_lambda.py` &mdash; sweeps the fairness-penalty weight `lam` over several
  orders of magnitude under one or both fairness reward modes and writes a CSV
  summary (mean/std cumulative profit, gap, reward across seeds, per policy).
- `requirements.txt` &mdash; minimal deps (`numpy`, `scipy`, `matplotlib`,
  `gymnasium`).

## How the pieces fit together

```
                    +----------------------+
config dict ---->   |     LendingEnv       |  <---- reset(seed) / step(action)
                    | (gymnasium.Env)      |
                    +----------+-----------+
                               |
                state, info    |  reward
                               v
   +------------------+    +-----------+    +---------------+
   | ProfitMaxPolicy  | -->|           |    |               |
   | DemoParityPolicy | -->| simulate()| -->|  plot_results |
   +------------------+    +-----------+    +---------------+
                                                  |
                                                  v
                                       baseline_comparison.png
```

The env owns a persistent per-group population of individual scores. Each call
to `step()`:

1. All current population members in each group apply this round.
2. Members with `score >= tau_g` are approved.
3. Each approved member repays with probability `sigmoid(k * (s - s_0))`.
4. Repaying members gain `delta_repay`, defaulters lose `delta_default`;
   scores are clipped to `[0, 1]`. Denied members are unchanged.
5. A `churn_rate` fraction per group is replaced with fresh draws from the
   *initial* truncated-Normal distribution.
6. `(mu_g, sigma_g, N_g)` in the next observation are the empirical moments of
   the updated population.

Reward mode is configurable: `"profit"`, `"profit_dp"` (profit minus
`lam * |approval_rate_A - approval_rate_B|`), or `"profit_gap"` (profit minus
`lam * |mu_A - mu_B|`). The gap used by `"profit_gap"` is computed from the
*post-update* population (i.e., after step 4's score updates and step 5's
churn), and is the same value that appears in the next observation.

`simulate()` logs per-round arrays of shape `(n_seeds, n_rounds)` for the
usual metrics (`mu_A`, `mu_B`, `gap`, `profit`, `reward`, approval/repay
rates, thresholds, ...), plus a per-run scalar `cumulative_gap` of shape
`(n_seeds,)` -- the episode-summed `|mu_A - mu_B|`, intended for summary
tables and cross-seed statistical comparisons.

## Running

```bash
pip install -r requirements.txt
python run_baselines.py --out baseline_comparison.png

# Sweep lambda (fairness penalty weight) over several orders of magnitude.
# Baselines do not consume the reward signal, so their actions are
# lambda-invariant; what this captures is the reward magnitude each baseline
# accumulates under each (reward_mode, lambda) pair -- useful for sizing
# lambda before running PPO.
python sweep_lambda.py --lambdas 0.1 1 10 100 1000 \
    --reward-modes profit_dp profit_gap \
    --out lambda_sweep.csv
```

## Plugging in calibrated initial parameters

Override these `config` keys to inject externally calibrated values:

- `mu_A_init`, `sigma_A_init`, `N_A_init`
- `mu_B_init`, `sigma_B_init`, `N_B_init`

The env stores these at construction and re-uses them both to seed initial
populations on `reset()` and to draw churn replacements.

## Plugging into Stable-Baselines3

```python
from stable_baselines3 import PPO
from lending_env import LendingEnv

env = LendingEnv({"reward_mode": "profit_dp", "lam": 5.0})
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
```

## Default hyperparameters

See `LendingConfig` in `lending_env.py` for the full list. Notable defaults:

| Param            | Default | Notes                              |
|------------------|---------|------------------------------------|
| `mu_A_init`      | 0.60    | Starting advantage for A           |
| `mu_B_init`      | 0.40    | Starting disadvantage for B        |
| `sigma_*_init`   | 0.15    |                                     |
| `N_*_init`       | 100     |                                     |
| `delta_repay`    | 0.05    |                                     |
| `delta_default`  | 0.10    | Asymmetric: defaults hurt more     |
| `k`              | 8.0     |                                     |
| `s_0`            | 0.5     |                                     |
| `u_repay`        | 1.0     |                                     |
| `u_default`      | 2.0     |                                     |
| `churn_rate`     | 0.05    |                                     |
| `horizon`        | 50      |                                     |
| `reward_mode`    | profit  | one of profit/profit_dp/profit_gap |
| `lam`            | 1.0     | fairness penalty weight            |

## Design decisions worth knowing

- **Persistent applicant population.** Step 1 of the spec ("sample `N_g`
  applicants from the current truncated Normal") is read as "all current
  population members apply", since steps 4 and 5 require tracking individuals.
- **`N_g` is constant.** Churn replaces removed members 1-for-1; nothing else
  removes applicants. Change this in `LendingEnv.step` if you want defaulters
  to exit the pool.
- **`(mu_g, sigma_g)` in state are empirical** moments of the current scores,
  matching step 6.
- **Baseline analytics use the parametric truncated-Normal implied by the
  state**, not the empirical distribution. This is a reasonable approximation
  and matches what a planner with model access would do; the small mismatch
  versus the empirical population is by design.
- **Demographic-parity policy** uses a 0.01 grid, picks `tau_B` per `tau_A`
  with the closest approval rate, accepts pairs within tolerance `1e-3` and
  among those maximises joint expected profit (falling back to the
  closest-difference pair if nothing clears tolerance).
