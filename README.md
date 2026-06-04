# RLforMH

RLforMH is an offline reinforcement learning project built on the StudentLife
dataset. The goal is to explore whether passively collected student behavior
signals can be converted into a daily decision-making problem related to mood
and mental health.

The project aggregates StudentLife mood, sleep, activity, and social sensing
data into daily student states. It then creates proxy actions from changes in
behavior, such as `increase_sleep`, `decrease_activity`, or `none`, and defines
several reward variants from mood and behavior changes. These transitions are
used to train and evaluate offline RL policies, including Discrete CQL, DQN,
Double DQN, supervised baselines, rule-based policies, and a contextual bandit
baseline.

Important caveat: the actions are inferred from observational data. They are not
real interventions delivered to students, so evaluation results should be read
as exploratory offline estimates rather than causal claims about improving
mental health.

## Setup

### Clone the repo

```bash
git clone https://github.com/ElisabethHolm/RLforMH.git
cd RLforMH
```

### Create a virtual environment and install requirements

```bash
python3.10 -m venv cs224r
source cs224r/bin/activate
pip install -r requirements.txt
```

## Dataset Creation

### Download StudentLife

Download the original StudentLife dataset from Kaggle:

https://www.kaggle.com/datasets/dartweichen/student-life/data

Place it in `studentLifeDataset` so the repo has this structure:

```text
RLforMH
├── studentLifeDataset
│   ├── app_usage
│   ├── calendar
│   ├── EMA
│   ├── sensing
│   ├── survey
│   └── ...
├── dataset_prep
├── algorithms
└── ...
```

### Build Daily Student Features

```bash
python dataset_prep/build_aggregated_data.py
```

This creates `daily_studentlife_no_transitions.csv`. It aggregates daily mood,
sleep, activity, and social features, normalizes sensor features per student,
adds lagged history features, labels proxy actions, and computes mood-based
rewards.

### Build RL Transitions

```bash
python dataset_prep/prepare_rl_dataset.py
```

This creates `daily_studentlife.csv`, a transition-level dataset containing
current states, actions, reward variants, terminal flags, and next-state
columns. The current experiments use the copies and chronological train/val/test
splits under `final_datasets/`.

## Training

### Train Discrete CQL

```bash
python algorithms/train_cql.py
```

The trained CQL model is saved to `models/studentlife_discrete_cql.d3`. Training
logs are written under `d3rlpy_logs/`.

### Train Baselines

```bash
python algorithms/train_baselines.py
```

This trains several comparison policies:

- `random_uniform`
- `majority_action`
- `action_frequency`
- `behavior_cloning_logistic`
- `rule_based_baseline`

The baseline models are saved in `models/studentlife_baseline_models.pkl`, and
summary metrics are saved in `models/baseline_metrics.json`.

### DQN Hyperparameter Search

```bash
python algorithms/hyperparameter_search_dqn.py
```

This grid-searches Discrete DQN and Double DQN over learning rate, batch size,
target update interval, and hidden layer sizes, training on the chronological
splits in `final_datasets/` (`daily_studentlife.train.csv` for fitting,
`daily_studentlife.val.csv` for selection).

The search runs separately for each of the three reward variants in
`final_datasets/daily_studentlife.csv`:

- `reward_sparse` — short- plus long-term mood change; nonzero only when mood
  was observed.
- `reward_dense` — weighted blend of short-term mood change and daily
  sleep/activity/social deltas; populated on every row.
- `reward_observed_only` — the sparse reward restricted to days where both the
  current and previous mood were observed.

Because `reward_sparse`, `reward_observed_only`, and the `mood` state feature
are `NaN` on most rows (mood is rarely observed), missing observations and
rewards are filled with `0` before training (the `mood_observed` flag stays in
the state so the network can distinguish missing mood).

Each config is scored two ways on the validation split:

- d3rlpy built-in offline metrics: TD error, discrete action match, average
  value estimation.
- Offline policy evaluation (OPE): per-decision importance sampling (PDIS)
  against a behavior-cloning logging policy, weighted/self-normalized IS
  diagnostics, doubly robust (DR) OPE, matched-action next-day mood improvement,
  and a direct-method `V(s0)` estimate.

The best config per reward variant is chosen by PDIS. Full results are written
to `models/dqn_hparam_search_results.json` and `models/dqn_hparam_search_results.csv`,
and the best model per variant is saved to `models/dqn_best_<reward_variant>.d3`.

Useful flags:

- `--quick` — shrink the grid to one value per axis and lower `n_steps` for a
  fast smoke test.
- `--reward-variants` / `--algos` — restrict which reward columns or algorithms
  to run.
- `--n-steps`, `--device`, `--max-configs` — control training length, the torch
  device, and the number of configs per variant.

Note: PDIS for `reward_sparse` / `reward_observed_only` can be near-degenerate
because almost every reward is `0`; this is expected, and is why the built-in
metrics and mood improvement are logged alongside it.

### Environment (important)

Use the project virtualenv **`cs224r`**, not conda `base`. Base often has
NumPy 2.x (breaks matplotlib) and lacks `d3rlpy`:

```bash
source cs224r/bin/activate   # prompt should show (cs224r), not (base)
python -c "import d3rlpy, numpy; print(numpy.__version__)"  # expect 1.24.x
```

If the venv does not exist yet, create it from the install steps above.
One-shot pipeline (figures + LaTeX table):

```bash
bash scripts/run_poster_eval.sh
```

### Discrete IQL and AWAC (custom PyTorch)

d3rlpy's `IQLConfig` / `AWACConfig` are **continuous-action** only; this repo
implements **discrete** IQL and AWAC in PyTorch (same chronological splits and
OPE pipeline as DQN / BCQ).

```bash
python algorithms/hyperparameter_search_iql_awac.py --quick --reward-variants reward_dense
python algorithms/evaluate_iql_awac_models.py --splits test
```

Outputs: `models/iql_awac_hparam_search_results.{json,csv}`,
`models/iql_best_reward_dense.pt`, `models/awac_best_reward_dense.pt`,
`models/iql_awac_ope_metrics.{json,csv}`. Policies are included in
`extended_policy_comparison.py` and focus figures when those checkpoints exist.

### Weighted IS and Doubly Robust OPE

```bash
python algorithms/evaluate_dqn_models.py
```

This evaluates the saved best DQN / Double DQN models on the
`final_datasets/` train, validation, and test splits without rerunning the grid.
It writes `models/dqn_saved_model_ope_metrics.json` and
`models/dqn_saved_model_ope_metrics.csv`.

Evaluation exports and `visualize_policy_comparison.py` figures include **95%
error bars** when you re-run evaluation with bootstrap enabled (default
`--n-bootstrap 300` on `extended_policy_comparison.py` and
`evaluate_dqn_models.py`; use `0` for a fast point-estimate-only run):

- **PDIS / weighted PDIS / DR / match%**: episode-level bootstrap CIs (clustered
  by student episode).
- **Δmood**: normal-approximation CI on matched steps with observed mood.

Poster figures (`*_test_focus.png`) use **display caps** on whiskers (split PDIS/WPDIS
vs DR panels; DR axis clipped). Full CIs remain in `models/*.csv` and JSON.

The extra OPE metrics are robustness checks for the PDIS-selected models:

- `pdis`: the existing per-decision IS estimate. Useful, but noisy when the
  learned policy often disagrees with the logged action.
- `weighted_pdis`: a self-normalized per-decision IS estimate. This usually
  lowers variance, but can introduce bias.
- `trajectory_wis`: trajectory-level weighted IS. This is mostly diagnostic
  because full-trajectory weights can be unstable on small offline datasets.
- `dr`: sequential doubly robust OPE using the learned Q-function. DR can be
  more stable when either propensities or Q-values are good, but it can still be
  biased if Q-values are optimistic.
- `effective_sample_size` and `weight_max`: support/variance diagnostics. Low
  effective sample size or high max weight means the estimate is driven by a
  small amount of logged support.

The practical interpretation is agreement-based: if Double DQN has positive
PDIS, weighted PDIS, and DR on the hold-out split, the offline evidence is
stronger. If only one estimator is positive, the conclusion should stay
conservative.

### Subgroup Policy Analysis

```bash
python algorithms/evaluate_dqn_models.py \
  --reward-variants reward_dense \
  --subgroup-analysis \
  --student-analysis
```

This adds state-conditioned diagnostics for the saved best DQN / Double DQN
models. The goal is to check whether a policy behaves sensibly for different
student states instead of only looking at aggregate OPE.

The subgroup analysis is derived from columns already present in
`final_datasets/`:

- mood observed vs missing, plus low/high observed mood
- low vs normal sleep from `sleep_z`
- low/high activity from `activity_z`
- low/high social signal from `social_z`
- weekday vs weekend from `date`
- early vs late term from each student's date rank

Outputs are saved to:

- `models/dqn_subgroup_policy_analysis.json`
- `models/dqn_subgroup_policy_analysis.csv`
- `models/dqn_student_policy_analysis.csv`

Reported subgroup metrics include policy/logged action distributions, action
match, dominant action, action-collapse flags, recommendation rates for sleep,
activity, and social actions, plus PDIS, weighted PDIS, trajectory WIS, DR, ESS,
mood improvement, and direct-method value where support allows.

These are diagnostic, not causal subgroup effects. Useful checks include whether
low-sleep rows receive more `increase_sleep` recommendations, whether low-social
rows receive more `increase_social` recommendations, whether mood-missing rows
collapse to one action, and whether particular students have weak support or poor
estimated outcomes. Stress is not currently in the transition files, so
high/low-stress subgroup analysis would require a separate data-prep change.

Current best validation results are summarized below:

| Reward variant | Best model | Selection metric | Matched mood improvement | Action match |
| --- | --- | ---: | ---: | ---: |
| `reward_dense` | Double DQN (`lr=1e-4`, batch 32, target update 1000, `256x256`) | PDIS `0.0328` | `0.4667` | `0.1842` |
| `reward_sparse` | DQN (`lr=1e-4`, batch 64, target update 1000, `256x256`) | PDIS `0.0115` | `0.3889` | `0.1678` |
| `reward_observed_only` | DQN (`lr=1e-4`, batch 64, target update 1000, `256x256`) | PDIS `0.0115` | `0.3889` | `0.1678` |

The dense reward is the most useful signal in the current setup. After missing
rewards are filled with `0`, `reward_sparse` and `reward_observed_only` are
identical in the current split files, so those two experiments currently produce
the same best result.

### Contextual Bandit Baseline

```bash
python algorithms/train_contextual_bandit.py
```

This trains a one-step baseline that learns
`E[reward | state, action]` from the logged transitions and then picks the
action with the highest predicted immediate reward. Unlike DQN/CQL, it does not
model future rewards or episode dynamics.

The contextual bandit uses the same `final_datasets/` train/validation/test
splits and the same three reward variants as the DQN search. Missing state
values and missing rewards are filled with `0`, matching the DQN preprocessing.
The default reward model is Ridge regression over state features, action one-hot
features, and state-by-action interactions.

Metrics are saved to `models/contextual_bandit_metrics.json` and
`models/contextual_bandit_metrics.csv`, and the fitted reward/behavior models
are saved to `models/contextual_bandit_models.pkl`.

Reported metrics include:

- Direct Method (DM), IPS, self-normalized IPS (SNIPS), and doubly robust (DR)
  one-step bandit OPE estimates.
- Matched logged reward and matched next-day mood improvement where the bandit
  action equals the observed action.
- Action match rate and predicted action distribution.

Use this baseline to interpret DQN results: if DQN only slightly improves over
the contextual bandit, the apparent gain may mostly come from immediate dense
reward correlations rather than long-horizon planning. The IPS/DR estimates use
a behavior-cloning logging policy and clipped importance weights for stability
on the small offline dataset.

Current contextual bandit results:

| Reward variant | Split | DR | IPS | SNIPS | Matched mood improvement | Action match |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `reward_dense` | validation | `0.1144` | `-0.0127` | `-0.0397` | `0.8333` | `0.1776` |
| `reward_dense` | test | `0.1211` | `0.0150` | `0.0443` | N/A | `0.1639` |
| `reward_sparse` | validation | `0.1274` | `-0.0219` | `-0.1323` | `0.8333` | `0.0987` |
| `reward_observed_only` | validation | `0.1274` | `-0.0219` | `-0.1323` | `0.8333` | `0.0987` |

The contextual bandit is a useful sanity check, but its metrics should not be
compared one-to-one with DQN's sequential PDIS. The key practical read is that
DQN/Double DQN get positive dense-reward sequential OPE, while the one-step
bandit has weak validation IPS/SNIPS on dense reward. That suggests the RL
models may be using more than just immediate reward correlations, but the low
action-match rates mean this is still exploratory offline evidence.

## Evaluation

Run offline policy evaluation with:

```bash
python algorithms/evaluate_policies.py
```

This compares CQL against the baselines using:

- Per-decision importance sampling (PDIS)
- Direct Method value estimates for CQL
- Matched-action next-day mood improvement

Results are saved to `models/ope_metrics.json`.

## Current Interpretation

The project now has an end-to-end daily offline RL pipeline with DQN/Double DQN
hyperparameter search and a contextual bandit baseline on the new
`final_datasets/` splits.

The clearest result is that `reward_dense` is the most informative reward
variant. It gives feedback on almost every row by combining short-term mood
change with sleep, activity, and social behavior deltas:

```text
reward_dense =
  0.60 * mood_short_term
  + 0.15 * sleep_delta
  + 0.15 * social_delta
  + 0.10 * activity_delta
```

In plain language, dense reward favors days where a student's mood and
wellness-related behavior improve relative to their own recent baseline.

The best dense-reward RL result is Double DQN with validation PDIS `0.0328` and
matched mood improvement `0.4667`. The contextual bandit baseline provides a
one-step comparison point: it has positive DR on dense reward, but weak
validation IPS/SNIPS. This suggests that sequential RL may be adding value
beyond an immediate reward model, although the evidence is not definitive.

Important caveats:

- Actions are inferred from observational behavior, not assigned
  interventions.
- Offline policy estimates are fragile because learned policies match the logged
  action only about 16-18% of the time on dense reward.
- `reward_sparse` and `reward_observed_only` currently collapse to the same
  signal after missing rewards are filled with `0`.
- Results should be presented as exploratory offline estimates, not causal
  evidence that a policy improves mental health.

