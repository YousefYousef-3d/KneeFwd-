from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .coefficients import assign_recovery_classes, baseline_log_steps, coefficient_rows_for_patients
from .utils import beta01, clipped_normal, get_season, safe_int_steps, weighted_choice


def generate_patients(
    n: int,
    default_cfg: Dict[str, Any],
    patient_cfg: Dict[str, Any],
    coeffs: Dict[str, Any],
    recovery_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pcfg = patient_cfg["patient_generation"]
    start_date = pd.Timestamp(default_cfg.get("calendar_start_date", "2026-01-01"))
    baseline_season = get_season(start_date - pd.Timedelta(days=1))

    patients = pd.DataFrame({"patient_id": [f"P{idx:04d}" for idx in range(1, n + 1)]})
    patients["age"] = clipped_normal(rng, **_normal_args(pcfg["age"]), size=n).round(1)
    patients["sex"] = rng.choice(pcfg["sex"]["categories"], p=pcfg["sex"]["probabilities"], size=n)
    patients["bmi"] = clipped_normal(rng, **_normal_args(pcfg["bmi"]), size=n).round(1)
    patients["baseline_pain_nrs"] = clipped_normal(rng, **_normal_args(pcfg["baseline_pain_nrs"]), size=n).round(1)
    patients["baseline_function_score"] = clipped_normal(rng, **_normal_args(pcfg["baseline_function_score"]), size=n).round(1)
    patients["patient_activation_score"] = clipped_normal(rng, **_normal_args(pcfg["patient_activation_score"]), size=n).round(1)
    patients["digital_confidence_score"] = clipped_normal(rng, **_normal_args(pcfg["digital_confidence_score"]), size=n).round(1)

    cb = pcfg["comorbidity_burden"]
    patients["comorbidity_burden"] = np.clip(rng.poisson(cb["lambda"], size=n), cb["min"], cb["max"]).astype(int)
    sev = pcfg["osteoarthritis_severity"]
    patients["osteoarthritis_severity"] = rng.choice(sev["categories"], p=sev["probabilities"], size=n)

    traits = pcfg["latent_traits"]
    patients["adherence_tendency"] = beta01(rng, *traits["adherence_beta"], size=n)
    patients["treatment_effectiveness_score"] = beta01(rng, *traits["treatment_effectiveness_beta"], size=n)
    patients["recovery_sensitivity_score"] = beta01(rng, *traits["recovery_sensitivity_beta"], size=n)
    patients["variability_tendency"] = beta01(rng, *traits["variability_beta"], size=n)
    patients["setback_tendency"] = beta01(rng, *traits["setback_beta"], size=n)
    patients["missingness_tendency"] = beta01(rng, *traits["missingness_beta"], size=n)

    # Link adherence and missingness to digital confidence/activation without making them deterministic.
    activation_z = (patients["patient_activation_score"] - 66.3) / 20.4
    digital_z = (patients["digital_confidence_score"] - 72.2) / 21.2
    pain_z = (patients["baseline_pain_nrs"] - 5.5) / 2.7
    patients["adherence_tendency"] = np.clip(
        patients["adherence_tendency"] + 0.05 * activation_z + 0.04 * digital_z - 0.03 * pain_z,
        0.01,
        0.99,
    )
    patients["missingness_tendency"] = np.clip(
        patients["missingness_tendency"] - 0.04 * digital_z - 0.04 * activation_z + 0.04 * pain_z,
        0.01,
        0.99,
    )

    patients["surgical_candidate_multiplier"] = pcfg["baseline_activity"]["surgical_candidate_multiplier"]

    baseline_contribs: Dict[str, Dict[str, float]] = {}
    baseline_steps = []
    for _, patient in patients.iterrows():
        result = baseline_log_steps(patient, coeffs, baseline_season)
        random_effect = rng.normal(0.0, pcfg["baseline_activity"]["random_log_sd"])
        expected = result.contributions["expected_steps_before_random_effect"] * np.exp(random_effect)
        steps = int(safe_int_steps(
            expected,
            min_steps=pcfg["baseline_activity"]["min_steps"],
            max_steps=pcfg["baseline_activity"]["max_steps"],
        ))
        baseline_steps.append(steps)
        result.contributions["baseline_random_log_effect"] = float(random_effect)
        result.contributions["baseline_steps_after_random_effect"] = float(steps)
        baseline_contribs[patient["patient_id"]] = result.contributions
    patients["pre_treatment_baseline_steps"] = baseline_steps

    classes, class_probabilities = assign_recovery_classes(patients, recovery_cfg, coeffs, rng)
    patients["recovery_trajectory_class"] = classes

    patient_coefficients = coefficient_rows_for_patients(patients, class_probabilities, baseline_contribs, coeffs)
    return patients, patient_coefficients


def _normal_args(cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "mean": float(cfg["mean"]),
        "sd": float(cfg["sd"]),
        "low": float(cfg["min"]),
        "high": float(cfg["max"]),
    }
