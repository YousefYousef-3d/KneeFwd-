# Team 3 Task 1 Simulator

Reusable Python simulator for one year of daily step-count recovery trajectories after treatment such as total knee replacement in people with knee osteoarthritis.

The project is designed for Hackathon Task 1: simulation of post-treatment wearable or smartphone trajectories with realistic missingness. It creates synthetic patient metadata, complete daily step-count trajectories, realistic missingness, optional imputed datasets, summary tables, figures, and validation reports when run locally.

## Install

From the project root:

```bash
pip install -r requirements.txt
```

## Run the simulator locally

```bash
python scripts/run_simulation.py --config config/default_config.yaml
```

This writes generated datasets, tables, figures, and simulator parameters into `outputs/` according to the paths in `config/default_config.yaml`.

## Run validation locally

After running the simulator:

```bash
python scripts/run_validation.py --config config/default_config.yaml
```

Validation writes:

```text
outputs/validation/validation_report.md
outputs/validation/validation_metrics.csv
```

## Research

### Duong et al. 2023

Used mainly for post-treatment trajectory structure after total knee replacement. The results section reported three short-term step-count trajectories over approximately 12 weeks:

1. high and rapidly increasing, about 14%;
2. low and gradually increasing, about 56%;
3. high and gradually increasing, about 30%.

The simulator uses those results to define early post-treatment targets, class-probability priors, BMI and pain effects on recovery class assignment, and validation targets around days 21 and 84.

### Brisson et al. 2019/2020

Used mainly for baseline physical-activity coefficients in symptomatic knee OA. Their mixed-effects models found that age, BMI, and season were associated with steps/day, while KOOS-pain and P4-pain were not statistically associated with steps/day after covariate adjustment. The simulator therefore makes age, BMI, and season direct determinants of daily steps, while pain has only a weak direct effect on baseline steps. Pain can still influence post-treatment class probability and missingness burden.

## Why the results sections are central

The simulator is calibrated from reported numerical results, not only from narrative conclusions. In particular:

- Duong group sizes and step-count means/SDs inform early recovery targets and trajectory-class mapping.
- Duong BMI and baseline pain group differences inform recovery-class probability coefficients.
- Brisson mixed-effects model coefficients inform baseline step-count effects for age, BMI, season, and pain.
- The one-year portion is explicitly extrapolated because Duong reports approximately 12-week post-TKR trajectories, not full one-year daily patterns.

## Model overview

The simulator is a Monte Carlo model. Each run samples:

- patient characteristics;
- baseline activity level;
- patient-level coefficient multipliers;
- recovery trajectory class;
- recovery parameters;
- daily noise and autoregressive residual variation;
- weekly and seasonal effects;
- setbacks and relapses;
- wearable or smartphone missingness events;
- optional imputed values.

The code is deterministic given the configured random seed.

## Six trajectory classes

The simulator creates six one-year classes:

1. `stable_baseline`: starts near baseline and remains broadly stable.
2. `decreasing`: starts near baseline or early post-treatment level and declines.
3. `increasing`: improves toward a higher plateau.
4. `decrease_then_increase`: early post-treatment fall followed by recovery; common after surgery.
5. `increase_then_decrease`: early improvement followed by later decline.
6. `fluctuating_or_relapsing`: repeated ups and downs, setbacks, or unstable recovery.

Duong observed three 12-week groups, not these six one-year groups. The mapping is documented in `references/paper_analysis.md` and encoded in `config/recovery_trajectory_classes.yaml`.

## Coefficient-based model

Coefficients are loaded from `config/coefficients.yaml`. The baseline step model is multiplicative on the log scale. For example, Brisson's reported percentage effects are converted to log effects:

- older age reduces expected steps/day;
- higher BMI reduces expected steps/day;
- spring/fall and winter reduce steps relative to summer;
- pain has a near-zero direct baseline-step effect.

Additional assumed coefficients control recovery rate, treatment effectiveness, daily variability, setbacks, final plateau, and missingness. The simulator writes the actual coefficient values and contribution terms used for each patient to:

```text
outputs/data/patient_coefficients.csv
```

## Missingness model

Missingness is configured in `config/missingness_scenarios.yaml`. Each missingness reason has:

- a base daily probability or patient-level dropout probability;
- block types and length distributions;
- MCAR, MAR, MNAR-like, or structural dropout label;
- patient-level covariate effects;
- optional true-activity dependence;
- configurable effect sizes.

Default reasons:

1. `sleep_or_low_wear_time`
2. `device_failure`
3. `sync_or_connectivity_problem`
4. `forgot_or_nonwear`
5. `pain_fatigue_or_burden_related_nonwear`
6. `dropout`

The missingness model is intended to remove observations, not true underlying activity. The complete true dataset is preserved separately.

## Imputation methods

Configured by `selected_imputation_method` in `config/default_config.yaml`.

Implemented methods:

- `no_imputation`
- `forward_fill_with_limit`
- `patient_mean_imputation`
- `group_day_mean_imputation`
- `linear_interpolation`
- `simple_model_based_imputation`

Imputation never overwrites the original observed column. It writes a separate file:

```text
outputs/data/imputed_data.csv
```

with `observed_steps`, `imputed_steps`, `was_imputed`, and `imputation_method`.

## How to change parameters

Edit YAML files in `config/`:

```text
config/default_config.yaml                 # run settings and paths
config/patient_generation.yaml             # patient covariate distributions
config/coefficients.yaml                   # coefficients and effect sizes
config/recovery_trajectory_classes.yaml    # class probabilities and curve settings
config/missingness_scenarios.yaml          # missingness mechanisms
```

Examples:

- Increase sample size: edit `number_of_patients`.
- Change missingness burden: edit `base_probability` values.
- Disable figures: set `run_plotting: false`. The current plotting code creates one separate trajectory figure for each of the six recovery classes and focuses the remaining figures on missingness; it intentionally does not create a healthy-reference figure, imputation-comparison figure, heatmap, patient-covariate figure, or class-mean-curve figure.
- Disable imputation: set `run_imputation: false` or choose `no_imputation`.
- Make winter effects stronger: edit `baseline_step_model.season_effects.winter_log_effect`.

## Expected outputs after running the simulator

Data:

```text
outputs/data/complete_data.csv
outputs/data/observed_data.csv
outputs/data/imputed_data.csv
outputs/data/patient_metadata.csv
outputs/data/patient_coefficients.csv
outputs/data/healthy_reference_summary.csv
outputs/data/simulator_parameters.json
```

Tables:

```text
outputs/tables/group_summary.csv
outputs/tables/trajectory_class_summary.csv
outputs/tables/patient_archetype_summary.csv
outputs/tables/missingness_summary.csv
outputs/tables/missingness_by_reason.csv
outputs/tables/missingness_by_month.csv
outputs/tables/imputation_summary.csv
```

Figures:

```text
outputs/figures/trajectory_class_stable_baseline.png
outputs/figures/trajectory_class_decreasing.png
outputs/figures/trajectory_class_increasing.png
outputs/figures/trajectory_class_decrease_then_increase.png
outputs/figures/trajectory_class_increase_then_decrease.png
outputs/figures/trajectory_class_fluctuating_or_relapsing.png
outputs/figures/missingness_examples_with_gaps.png
outputs/figures/missingness_by_reason.png
outputs/figures/missingness_by_mechanism.png
outputs/figures/missingness_by_month.png
outputs/figures/missingness_timeline_by_reason.png
outputs/figures/missingness_block_lengths.png
outputs/figures/missingness_patient_burden_distribution.png
```

Validation:

```text
outputs/validation/validation_report.md
outputs/validation/validation_metrics.csv
```

## Assumptions and limitations

This is a disciplined but synthetic simulator. It is not a clinical prediction model. The largest limitation is that one-year daily post-treatment trajectories are extrapolated from a short-term post-TKR trajectory paper and a longitudinal mild-to-moderate knee OA activity paper. The default values are intended to be plausible starting points for hackathon analysis, not definitive estimates.
