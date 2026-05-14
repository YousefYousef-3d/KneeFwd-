# Assumptions

This simulator is intended for Hackathon Task 1. It is a transparent, configurable simulator rather than a clinical prediction model.

## Evidence-to-simulation boundary

The Duong paper reports approximately 12-week post-total-knee-replacement step-count trajectories. The simulator extrapolates those early patterns to one year. Any class behaviour after 12 weeks is an assumption.

The Brisson paper reports associations between covariates and steps/day in mild-to-moderate symptomatic knee osteoarthritis over repeated measurements. The simulator uses these coefficients for baseline activity direction and scale, but the post-surgical population may differ.

## One-year extrapolation

The six one-year classes are not directly observed in either paper. They are constructed for simulation and downstream missing-data analysis. The default proportions preserve the broad short-term message from Duong: many patients follow gradual recovery, a smaller group improves rapidly, and a high-step group remains higher throughout early follow-up.

## Pain modelling

Pain has a weak direct effect on baseline steps because Brisson did not find KOOS-pain or P4-pain to be statistically associated with steps/day after adjustment. Pain is allowed to influence trajectory class and burden-related missingness because Duong found baseline pain differed between post-TKR trajectory groups and because pain/fatigue can plausibly affect device use after surgery.

## Healthy reference

Healthy reference data, if supplied, are used only to set plausible upper bounds and variability reference values. They are not used to make post-treatment patients equivalent to healthy controls.

## Missingness

Missingness mechanisms are labelled MCAR, MAR, MNAR-like, and structural dropout for simulation purposes. These labels are conceptual, not proof of identifiability in real data.

## Configurability

Every important assumption is stored in YAML. Users are encouraged to run sensitivity analyses by editing the YAML files rather than changing source code.
