from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .utils import get_season, safe_int_steps, softmax


@dataclass
class BaselineStepResult:
    steps: int
    contributions: Dict[str, float]


def baseline_log_steps(patient: pd.Series, coeffs: Dict[str, Any], season: str) -> BaselineStepResult:
    model = coeffs["baseline_step_model"]
    log_steps = np.log(float(model["reference_steps_per_day"]))
    contributions: Dict[str, float] = {"intercept_log_steps": float(log_steps)}

    age_term = float(model["age_log_effect_per_year"]) * (float(patient["age"]) - float(model["reference_age"]))
    bmi_term = float(model["bmi_log_effect_per_unit"]) * (float(patient["bmi"]) - float(model["reference_bmi"]))
    pain_term = float(model["pain_nrs_log_effect_per_unit"]) * (float(patient["baseline_pain_nrs"]) - float(model["reference_pain_nrs"]))
    comorbidity_term = float(model["comorbidity_log_effect_per_condition"]) * float(patient["comorbidity_burden"])
    severity_term = float(model["severity_log_effects"].get(patient["osteoarthritis_severity"], 0.0))
    sex_term = float(model["sex_log_effects"].get(patient["sex"], 0.0))

    if season == "winter":
        season_term = float(model["winter_log_effect_vs_summer"])
    elif season == "spring_fall":
        season_term = float(model["spring_fall_log_effect_vs_summer"])
    else:
        season_term = 0.0

    for name, value in [
        ("age_log_contribution", age_term),
        ("bmi_log_contribution", bmi_term),
        ("pain_log_contribution", pain_term),
        ("comorbidity_log_contribution", comorbidity_term),
        ("severity_log_contribution", severity_term),
        ("sex_log_contribution", sex_term),
        ("season_log_contribution", season_term),
    ]:
        contributions[name] = float(value)
        log_steps += value

    multiplier = float(patient.get("surgical_candidate_multiplier", 1.0))
    multiplier_term = np.log(multiplier)
    contributions["surgical_candidate_log_contribution"] = float(multiplier_term)
    log_steps += multiplier_term

    steps = int(safe_int_steps(np.exp(log_steps), min_steps=0, max_steps=int(model.get("max_steps", 25000))))
    contributions["expected_log_steps"] = float(log_steps)
    contributions["expected_steps_before_random_effect"] = float(np.exp(log_steps))
    return BaselineStepResult(steps=steps, contributions=contributions)


def assign_recovery_classes(
    patients: pd.DataFrame,
    recovery_cfg: Dict[str, Any],
    coeffs: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[pd.Series, pd.DataFrame]:
    classes_cfg = recovery_cfg["trajectory_classes"]
    class_names = list(classes_cfg.keys())
    priors = np.array([classes_cfg[c]["default_probability"] for c in class_names], dtype=float)
    priors = priors / priors.sum()
    base_logits = np.log(priors)

    model = coeffs["recovery_class_probability_model"]
    rows = []
    assigned = []
    for _, patient in patients.iterrows():
        logits = base_logits.copy()
        for idx, cls in enumerate(class_names):
            c = model["coefficients_by_class"].get(cls, {})
            logits[idx] += float(c.get("bmi", 0.0)) * (patient["bmi"] - model["bmi_center"])
            logits[idx] += float(c.get("pain", 0.0)) * (patient["baseline_pain_nrs"] - model["pain_center"])
            logits[idx] += float(c.get("age", 0.0)) * (patient["age"] - model["age_center"])
            logits[idx] += float(c.get("treatment_effectiveness", 0.0)) * (patient["treatment_effectiveness_score"] - 0.5)
            logits[idx] += float(c.get("adherence", 0.0)) * (patient["adherence_tendency"] - 0.5)
        probs = softmax(logits)
        chosen = rng.choice(class_names, p=probs)
        assigned.append(chosen)
        row = {"patient_id": patient["patient_id"], "assigned_recovery_trajectory_class": chosen}
        for cls, prob, logit in zip(class_names, probs, logits):
            row[f"class_probability_{cls}"] = float(prob)
            row[f"class_logit_{cls}"] = float(logit)
        rows.append(row)
    return pd.Series(assigned, index=patients.index, name="recovery_trajectory_class"), pd.DataFrame(rows)


def patient_recovery_rate_multiplier(patient: pd.Series, coeffs: Dict[str, Any]) -> float:
    model = coeffs["recovery_rate_model"]
    log_rate = 0.0
    log_rate += float(model["bmi_log_rate_effect_per_unit"]) * (patient["bmi"] - 30.3)
    log_rate += float(model["age_log_rate_effect_per_year"]) * (patient["age"] - 67.7)
    log_rate += float(model["pain_log_rate_effect_per_nrs"]) * (patient["baseline_pain_nrs"] - 5.5)
    log_rate += float(model["treatment_effectiveness_log_rate_effect"]) * (patient["treatment_effectiveness_score"] - 0.5)
    log_rate += float(model["recovery_sensitivity_log_rate_effect"]) * (patient["recovery_sensitivity_score"] - 0.5)
    log_rate += float(model["adherence_log_rate_effect"]) * (patient["adherence_tendency"] - 0.5)
    return float(np.exp(log_rate))


def patient_variability_cv(patient: pd.Series, coeffs: Dict[str, Any]) -> float:
    model = coeffs["variability_model"]
    cv = float(model["base_daily_cv"])
    cv += float(model["variability_tendency_effect"]) * float(patient["variability_tendency"])
    cv += float(model["pain_effect_on_cv"]) * float(patient["baseline_pain_nrs"])
    cv += float(model["low_digital_confidence_effect_on_cv"]) * (1.0 - float(patient["digital_confidence_score"]) / 100.0)
    return float(np.clip(cv, 0.05, float(model["high_variability_cv"])))


def coefficient_rows_for_patients(
    patients: pd.DataFrame,
    class_probabilities: pd.DataFrame,
    baseline_contribs: Dict[str, Dict[str, float]],
    coeffs: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for _, patient in patients.iterrows():
        pid = patient["patient_id"]
        base = {
            "patient_id": pid,
            "recovery_trajectory_class": patient["recovery_trajectory_class"],
            "age": patient["age"],
            "bmi": patient["bmi"],
            "baseline_pain_nrs": patient["baseline_pain_nrs"],
            "baseline_steps": patient["pre_treatment_baseline_steps"],
            "age_log_effect_per_year": coeffs["baseline_step_model"]["age_log_effect_per_year"],
            "bmi_log_effect_per_unit": coeffs["baseline_step_model"]["bmi_log_effect_per_unit"],
            "pain_nrs_log_effect_per_unit": coeffs["baseline_step_model"]["pain_nrs_log_effect_per_unit"],
            "recovery_rate_multiplier": patient_recovery_rate_multiplier(patient, coeffs),
            "variability_cv": patient_variability_cv(patient, coeffs),
        }
        base.update(baseline_contribs.get(pid, {}))
        rows.append(base)
    df = pd.DataFrame(rows)
    if not class_probabilities.empty:
        df = df.merge(class_probabilities, on="patient_id", how="left")
    return df
