# Validation report

## Basic data validity
- Patients: 300 expected 300.
- Rows per patient expected: 396.
- Missingness rate: 0.102.

## Paper-informed plausibility
- Day 21 and day 84 class means are compared with paper-informed targets using deliberately wide tolerances because the simulator extrapolates to six one-year classes.

## Trajectory class validation
- All configured classes should appear and proportions should be close to configured priors, allowing Monte Carlo variation.

## Coefficient validation
- Age correlation with baseline steps: -0.545.
- BMI correlation with baseline steps: -0.335.
- Pain correlation with baseline steps: 0.068; expected to be weaker than age/BMI direct effects.

## Missingness validation
- Missing reasons present: device_failure, dropout, forgot_or_nonwear, pain_fatigue_or_burden_related_nonwear, sleep_or_low_wear_time, sync_or_connectivity_problem.

## Imputation validation
- Imputation MAE against hidden true steps where imputed: 1342.6600115502022.
