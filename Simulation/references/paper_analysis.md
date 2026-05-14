# Paper analysis and simulator assumptions

This document converts the results sections of the two evidence papers into simulator assumptions, coefficient values, parameter ranges, and validation targets. It is not only a paper summary; it is the traceability layer between published results and the code/configuration.

## Paper 1 summary: Duong et al. 2023

Duong et al. studied step-count trajectories after total knee replacement using activity tracker data from the intervention arm of a trial. The analysis included 43 participants. The sample was mostly female, with average age about 67.7 years, average BMI about 30.3, and average baseline pain about 5.5 on a 0-10 numerical rating scale.

The key result for this simulator is the identification of three distinct approximately 12-week step-count trajectories:

1. high and rapidly increasing;
2. low and gradually increasing;
3. high and gradually increasing.

BMI and baseline pain differed significantly between trajectory groups. Higher BMI was associated with lower odds of being in the high and rapidly increasing group compared with the high and gradually increasing group.

## Paper 2 summary: Brisson et al. 2019/2020

Brisson et al. analysed repeated accelerometer-derived steps/day in 59 people with mild-to-moderate symptomatic knee osteoarthritis over up to three years. Their mixed-effects models used log-transformed physical activity and reported percentage changes in steps/day.

Age, BMI, and season were statistically significant predictors of physical activity. Older age and higher BMI were associated with fewer steps/day. Spring/fall and winter were associated with fewer steps/day compared with summer. Neither KOOS-pain nor P4-pain was statistically associated with steps/day after adjustment.

For the simulator, this means baseline activity should depend clearly on age, BMI, and season. Pain should not be the dominant direct determinant of daily steps in the baseline activity model.

## Key results extracted from Paper 1

| Quantity | Extracted value | Use in simulator |
|---|---:|---|
| Analysis sample | 43 participants | Evidence scale and limitation |
| Female proportion | 30/43, about 70% | Default sex distribution |
| Mean age | 67.7 years | Patient age distribution for post-TKR simulation |
| Age SD | 7.5 years | Patient age distribution |
| Mean BMI | 30.3 | Patient BMI distribution |
| BMI SD | 6.0 | Patient BMI distribution |
| Mean baseline pain NRS | 5.5 | Baseline pain distribution |
| Pain SD | 2.7 | Baseline pain distribution |
| PAM-13 mean | 66.3 | Patient activation distribution |
| PAM-13 SD | 20.4 | Patient activation distribution |
| Technology self-efficacy mean | 72.2 | Digital confidence distribution |
| Technology self-efficacy SD | 21.2 | Digital confidence distribution |
| High and rapidly increasing group | n=6, 14% | Prior information for rapid/increasing class components |
| Low and gradually increasing group | n=24, 56% | Prior information for gradual, post-surgical decrease-then-increase, and fluctuating classes |
| High and gradually increasing group | n=13, 30% | Prior information for stable_baseline and moderate increasing classes |
| High rapid early step count | 6251 ± 3508 | Day 21 target scale for rapid/increasing classes |
| High rapid weeks 9-11 step count | 12794 ± 3173 | Day 84 target scale for rapid/increasing classes |
| Low gradual early step count | 2843 ± 1058 | Day 21 target scale for slow surgical recovery classes |
| Low gradual >12 week step count | 6441 ± 1677 | Day 84 target scale for gradual recovery classes |
| High gradual early step count | 6299 ± 1777 | Day 21 target scale for stable/high classes |
| High gradual >12 week step count | 9614 ± 1436 | Day 84 target scale for stable/moderate increasing classes |
| BMI group difference | statistically significant | BMI affects recovery class probabilities and rate |
| Baseline pain group difference | statistically significant overall | Pain affects recovery class probabilities and missingness burden |
| BMI odds ratio | OR 0.72 per BMI unit for high rapid vs high gradual | Coefficient direction and optional log-OR reference |

## Key results extracted from Paper 2

| Quantity | Extracted value | Use in simulator |
|---|---:|---|
| Analysis sample | 59 participants | Baseline OA activity evidence scale |
| Mean age | 61.1 years | Reference age for baseline activity coefficients |
| Age SD | 6.4 years | Supporting plausible range |
| Mean BMI | 28.1 | Reference BMI for baseline activity coefficients |
| BMI SD | 5.6 | Supporting plausible range |
| Mean physical activity | 7158 steps/day | Baseline reference steps before post-TKR downshift |
| Physical activity SD | 3071 steps/day | Baseline variability scale |
| KOOS-pain mean | 24.6 / 100 inverted | Pain distribution context |
| P4-pain mean | 6.4 / 40 | Pain distribution context |
| Age effect | -3.65% steps/day per year | Direct baseline activity coefficient |
| BMI effect | -3.06% steps/day per BMI unit | Direct baseline activity coefficient |
| Spring/fall effect | -6.91% vs summer | Seasonal coefficient |
| Winter effect | -14.92% vs summer | Seasonal coefficient |
| KOOS-pain effect | +0.04%, not significant | Pain direct effect set near zero |
| P4-pain effect | -0.37%, not significant | Pain direct effect kept weak |

## How Paper 1 informs trajectory classes and early recovery

The simulator has six one-year classes, but Duong reports three approximately 12-week groups. The mapping is therefore partly direct for the first 12 weeks and partly extrapolated for the remaining year.

The three Duong groups inform:

- class priors;
- early day 21 target scales;
- day 84 target scales;
- between-patient variability around those targets;
- BMI and baseline pain effects on class probabilities;
- validation checks for early recovery scale.

The six one-year classes are designed so that the Duong three-group structure is still visible in early recovery while allowing clinically plausible year-long patterns such as later decline, fluctuating recovery, relapse, and stability.

## How Paper 2 informs baseline activity coefficients

Brisson's mixed-effects model results are used as direct baseline activity coefficients. The simulator treats their reported percentage effects as multiplicative effects on expected steps/day. These are encoded on the log scale in `config/coefficients.yaml`.

Age, BMI, and season have clear direct effects. Pain has only a weak direct baseline effect so that the simulator does not contradict the main Brisson finding that pain was not associated with steps/day after adjustment.

## Directly taken values versus inferred assumptions

Directly taken values include:

- Duong group proportions: 14%, 56%, 30%;
- Duong group-specific early and week-12 step count means/SDs;
- Duong age, BMI, pain, PAM-13, and technology self-efficacy means/SDs;
- Duong BMI odds ratio direction and approximate OR=0.72;
- Brisson mean steps/day, age, BMI, KOOS-pain, P4-pain, and mixed-effects coefficients for age, BMI, season, KOOS-pain, and P4-pain.

Inferred or assumed values include:

- six-class one-year proportions;
- all post-12-week curve shapes;
- daily autoregressive noise parameters;
- setback frequency and magnitude;
- missingness reason probabilities;
- imputation methods;
- comorbidity and OA severity distributions for the synthetic post-treatment cohort;
- the exact translation of pain into trajectory probabilities and missingness burden.

## Coefficient table

| coefficient_name | affected_model_component | value | source_paper | paper_result_used | directly_reported_or_assumed | rationale | config_parameter_name |
|---|---|---:|---|---|---|---|---|
| reference_steps_per_day | baseline activity | 7158 | Brisson | Mean physical activity 7158 steps/day | directly reported | Anchor expected steps in mild-to-moderate OA before post-TKR adjustment | baseline_step_model.reference_steps_per_day |
| reference_age | baseline activity | 61.1 | Brisson | Mean age 61.1 years | directly reported | Center age coefficient at Brisson sample mean | baseline_step_model.reference_age |
| reference_bmi | baseline activity | 28.1 | Brisson | Mean BMI 28.1 | directly reported | Center BMI coefficient at Brisson sample mean | baseline_step_model.reference_bmi |
| age_log_effect_per_year | baseline activity | -0.0372 | Brisson | Age β=-3.65% per year | transformed from reported | log(1 - 0.0365), preserving reported direction and scale | baseline_step_model.age_log_effect_per_year |
| bmi_log_effect_per_unit | baseline activity | -0.0311 | Brisson | BMI β=-3.06% per unit | transformed from reported | log(1 - 0.0306), preserving reported direction and scale | baseline_step_model.bmi_log_effect_per_unit |
| spring_fall_log_effect_vs_summer | seasonality | -0.0716 | Brisson | Spring/fall β=-6.91% vs summer | transformed from reported | log(1 - 0.0691) | baseline_step_model.spring_fall_log_effect_vs_summer |
| winter_log_effect_vs_summer | seasonality | -0.1616 | Brisson | Winter β=-14.92% vs summer | transformed from reported | log(1 - 0.1492) | baseline_step_model.winter_log_effect_vs_summer |
| pain_nrs_log_effect_per_unit | baseline activity | -0.0050 | Brisson + assumption | KOOS and P4 pain not significant | assumed weak | Keeps pain from dominating baseline steps while allowing small clinical variation | baseline_step_model.pain_nrs_log_effect_per_unit |
| duong_bmi_log_or_high_rapid_vs_high_gradual | trajectory probability | -0.3285 | Duong | BMI OR=0.72 for high rapid vs high gradual | transformed from reported | log(0.72), used as directional reference for BMI class effects | recovery_class_probability_model.duong_bmi_log_or_high_rapid_vs_high_gradual |
| high_rapid_prior | trajectory proportions | 0.14 | Duong | High rapid group 14% | directly reported | Allocated mainly to increasing/decrease_then_increase rapid recovery | recovery_trajectory_classes.yaml mapping |
| low_gradual_prior | trajectory proportions | 0.56 | Duong | Low gradual group 56% | directly reported | Allocated mainly to decrease_then_increase and fluctuating recovery | recovery_trajectory_classes.yaml mapping |
| high_gradual_prior | trajectory proportions | 0.30 | Duong | High gradual group 30% | directly reported | Allocated mainly to stable_baseline and increasing | recovery_trajectory_classes.yaml mapping |
| recovery_bmi_effect | recovery rate | -0.018 | Duong + assumption | BMI differed and OR direction disfavors rapid group | assumed | Higher BMI slows recovery rates | recovery_rate_model.bmi_log_rate_effect_per_unit |
| recovery_pain_effect | recovery rate | -0.025 | Duong + Brisson | Duong pain differed; Brisson pain not direct baseline driver | assumed | Pain can affect post-surgical recovery class/rate but should not dominate baseline steps | recovery_rate_model.pain_log_rate_effect_per_nrs |
| base_daily_cv | day-level variability | 0.16 | Assumption | Not directly reported | assumed | Creates realistic day-to-day wearable variability | variability_model.base_daily_cv |
| setback_base_event_probability | setbacks | 0.0025/day | Assumption | Not directly reported | assumed | Occasional setbacks over one year without overwhelming trajectory signal | setback_model.base_event_probability_per_day |
| adherence_missingness_effect | missingness | -0.85 | Assumption | Missingness not modelled in papers | assumed | Lower adherence increases nonwear/forgotten-device missingness | missingness_coefficients.adherence_effect |
| pain_missingness_effect | missingness | 0.18 | Assumption | Pain differed by Duong trajectory; Brisson not direct baseline driver | assumed | Pain/fatigue can affect wearable adherence without dominating true activity | missingness_coefficients.pain_effect |

## Trajectory proportion table

| trajectory_class | default_probability | source_or_rationale | relationship_to_duong_trajectory_group | notes_for_one_year_extrapolation |
|---|---:|---|---|---|
| stable_baseline | 0.18 | Derived from part of Duong high gradually increasing group | High gradual group starts high and changes gradually | Assumes some high early walkers remain broadly stable rather than continuing to rise |
| decreasing | 0.07 | Added as clinically plausible sensitivity class | Not a dominant Duong 12-week group | Represents later decline, complications, comorbidity, or reduced motivation over one year |
| increasing | 0.18 | Derived from high rapid plus part of high gradual groups | High rapid and some high gradual patterns | Extends early improvement to a higher plateau over one year |
| decrease_then_increase | 0.37 | Main post-surgical pattern, mostly from Duong low gradual group | Low gradual group starts low and increases | Adds explicit early treatment disruption followed by recovery |
| increase_then_decrease | 0.06 | Added as clinically plausible one-year relapse class | Not directly observed as a dominant Duong group | Allows early improvement followed by later decline/relapse |
| fluctuating_or_relapsing | 0.14 | Part of low gradual group extrapolated to unstable recovery | Low gradual group may include heterogeneous slower recoverers | Adds repeated setbacks and variability over one year |

These probabilities sum to 1.00 and are stored in `config/recovery_trajectory_classes.yaml`.

## Limitations of using these papers for a one-year simulator

1. Duong's sample was small and exploratory.
2. Duong reports approximately 12-week trajectories, not one-year daily trajectories.
3. Duong participants were in a digital intervention context and may not represent all TKR patients.
4. Brisson studied mild-to-moderate symptomatic knee OA, not immediate post-surgical recovery.
5. Brisson's coefficients are useful for baseline activity direction but cannot fully determine post-treatment recovery.
6. Missingness reasons and probabilities are not directly estimated in either paper and are therefore explicit simulation assumptions.
7. The simulator is calibrated to produce plausible synthetic data, not to predict clinical outcomes for individual patients.
