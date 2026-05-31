# Lending Fairness Simulator

A Gymnasium environment for studying long-horizon fairness in mortgage lending,
together with two non-RL baseline threshold policies and a small plotting and
rollout harness. Designed to plug into Stable-Baselines3 PPO later, and to
accept externally-calibrated initial distribution parameters (e.g. from HMDA).

## Files

- `constants.py` &mdash; shared constants: `GROUP_ORDER` (canonical 6-group
  ordering), HMDA race/ethnicity labels, plot colour/linestyle maps.
- `lending_env.py` &mdash; `LendingEnv`, a `gymnasium.Env` for six demographic
  groups (Race &times; Income tier). State is
  `[mu_0, sigma_0, N_0, ..., mu_5, sigma_5, N_5]` of shape `(18,)`; action is
  per-group thresholds in `[0, 1]^6`. Hyperparameters are passed via a `config`
  dict and validated against `LendingConfig`.
- `baselines.py` &mdash; `ProfitMaxThresholdPolicy`, `DemographicParityPolicy`,
  and `simulate(policy, env, n_seeds, n_rounds)` for non-RL rollouts.
- `plotting.py` &mdash; `plot_results(results, save_path=...)` produces the 2&times;2
  comparison figure (group means, equity metrics, cumulative profit, approval rates).
- `run_baselines.py` &mdash; entry point that runs both baselines over 5 seeds and
  50 rounds with default config and saves the figure.
- `sweep_lambda.py` &mdash; sweeps the fairness-penalty weight `lam` over several
  orders of magnitude under one or both fairness reward modes and writes a CSV
  summary (mean/std cumulative profit, weighted variance, reward across seeds,
  per policy).
- `hmda_calibrate.py` &mdash; downloads a filtered subset of HMDA loan data from
  the CFPB Data Browser API (no local file needed) and calibrates the initial
  population parameters for six Race &times; Income groups. Outputs a
  `hmda_config.json` you can pass directly to the env.
- `requirements.txt` &mdash; minimal deps (`numpy`, `scipy`, `matplotlib`,
  `gymnasium`, `requests`).

## Group structure

Six groups defined by the intersection of race (2 levels, non-Hispanic) and
income tier (3 levels via income-to-AMI ratio):

| Group          | Race  | Income tier |
|----------------|-------|-------------|
| `White_Low`    | White | Low         |
| `White_Middle` | White | Middle      |
| `White_Upper`  | White | Upper       |
| `Black_Low`    | Black | Low         |
| `Black_Middle` | Black | Middle      |
| `Black_Upper`  | Black | Upper       |

This ordering is defined once in `constants.GROUP_ORDER` and used everywhere.

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

Reward mode is configurable:

- `"profit"` &mdash; sum of per-group profit.
- `"profit_wvar_approval"` &mdash; profit minus
  `lam * weighted_variance(approval_rates, population_weights)`.
- `"profit_wvar_mean"` &mdash; profit minus
  `lam * weighted_variance(group_means, population_weights)`,
  computed from the post-update population.

`simulate()` logs per-round arrays of shape `(n_seeds, n_rounds)` for the
usual metrics (per-group means, approval rates, thresholds, profit, reward,
weighted variance, max pairwise gap, etc.), plus a per-run scalar
`cumulative_wvar_mean` of shape `(n_seeds,)`.

## Running

```bash
pip install -r requirements.txt

# --- HMDA calibration (optional but recommended) ---
# Downloads filtered loan data from the CFPB API (no local file needed) and
# fits per-group truncated-Normal parameters for six Race × Income groups.
# Defaults: CA/TX/FL/NY/IL, year 2022, up to 50 000 streamed rows.
python hmda_calibrate.py --out hmda_config.json

# Larger sample or different states/year:
python hmda_calibrate.py --states CA TX FL NY GA NC --year 2023 --n-rows 80000

# Nationwide sample (bigger; use a smaller --n-rows to keep it fast):
python hmda_calibrate.py --nationwide --n-rows 30000

# --- Baselines ---
# run_baselines.py automatically loads hmda_config.json if present,
# otherwise uses synthetic defaults:
python run_baselines.py --out baseline_comparison.png

# Sweep lambda (fairness penalty weight) over several orders of magnitude.
python sweep_lambda.py --lambdas 0.1 1 10 100 1000 \
    --reward-modes profit_wvar_approval profit_wvar_mean \
    --out lambda_sweep.csv
```

## Config schema

The config JSON uses a list-of-groups schema:

```json
{
  "groups": [
    {
      "name": "White_Low",
      "race": "White",
      "income_tier": "Low",
      "mu_init": 0.54,
      "sigma_init": 0.08,
      "N_init": 7643
    }
  ]
}
```

Groups must appear in the canonical `GROUP_ORDER`. The env stores these at
construction and re-uses them both to seed initial populations on `reset()`
and to draw churn replacements.

## Plugging into Stable-Baselines3

```python
from stable_baselines3 import PPO
from lending_env import LendingEnv

env = LendingEnv({"reward_mode": "profit_wvar_mean", "lam": 5.0})
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
```

## Default hyperparameters

See `LendingConfig` in `lending_env.py` for the full list. Notable defaults:

| Param            | Default        | Notes                                        |
|------------------|----------------|----------------------------------------------|
| `groups`         | 6 synthetic    | See `_default_groups()` in `lending_env.py`  |
| `delta_repay`    | 0.05           |                                              |
| `delta_default`  | 0.10           | Asymmetric: defaults hurt more               |
| `k`              | 8.0            |                                              |
| `s_0`            | 0.5            |                                              |
| `u_repay`        | 1.0            |                                              |
| `u_default`      | 2.0            |                                              |
| `churn_rate`     | 0.05           |                                              |
| `horizon`        | 50             |                                              |
| `reward_mode`    | `profit`       | one of profit / profit_wvar_approval / profit_wvar_mean |
| `lam`            | 1.0            | fairness penalty weight                      |

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
- **Demographic-parity policy** searches over a common target approval rate
  `r*` and picks the per-group threshold that yields exactly that rate, then
  selects the `r*` maximising total expected profit.
