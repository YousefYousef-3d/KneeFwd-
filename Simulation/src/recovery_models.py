from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .coefficients import patient_recovery_rate_multiplier, patient_variability_cv
from .utils import get_season, resolve_path, safe_int_steps, sigmoid, weekday_name


def load_healthy_reference(
    path: str,
    use_healthy_reference: bool,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path_obj = pd.io.common.stringify_path(resolve_path(path))
    df = pd.DataFrame()
    source = "synthetic_fallback"
    if use_healthy_reference:
        try:
            candidate = pd.read_csv(path_obj)
            if len(candidate) > 0 and "steps" in candidate.columns:
                df = candidate.copy()
                source = "external_file"
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = pd.DataFrame()

    if df.empty:
        n_people = 120
        days = pd.date_range("2026-01-01", periods=365, freq="D")
        rows = []
        for person_idx in range(n_people):
            age = int(np.clip(rng.normal(60, 12), 25, 85))
            sex = rng.choice(["female", "male"])
            base = rng.lognormal(np.log(9000), 0.28) * np.exp(-0.007 * (age - 60))
            for date in days:
                weekend = 1.05 if date.weekday() == 5 else (0.95 if date.weekday() == 6 else 1.0)
                season = get_season(date)
                season_mult = 0.88 if season == "winter" else (0.95 if season == "spring_fall" else 1.0)
                steps = int(safe_int_steps(base * weekend * season_mult * rng.lognormal(0, 0.18), 1000, 24000))
                rows.append({"person_id": f"H{person_idx+1:03d}", "date": date.date(), "age": age, "sex": sex, "steps": steps})
        df = pd.DataFrame(rows)

    summary = pd.DataFrame([
        {
            "source": source,
            "n_rows": int(len(df)),
            "n_people": int(df["person_id"].nunique()) if "person_id" in df.columns else np.nan,
            "mean_steps": float(df["steps"].mean()),
            "sd_steps": float(df["steps"].std()),
            "p50_steps": float(df["steps"].quantile(0.50)),
            "p85_steps": float(df["steps"].quantile(0.85)),
            "p95_steps": float(df["steps"].quantile(0.95)),
        }
    ])
    return df, summary


def simulate_recovery_trajectories(
    patients: pd.DataFrame,
    default_cfg: Dict[str, Any],
    recovery_cfg: Dict[str, Any],
    coeffs: Dict[str, Any],
    healthy_summary: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline_days = int(default_cfg["baseline_days"])
    post_days = int(default_cfg["post_treatment_days"])
    include_day0 = bool(default_cfg.get("include_treatment_day", True))
    start_date = pd.Timestamp(default_cfg.get("calendar_start_date", "2026-01-01"))

    days = list(range(-baseline_days, 0))
    if include_day0:
        days.append(0)
    days.extend(range(1, post_days + 1))

    rows = []
    param_rows = []
    healthy_p85 = float(healthy_summary["p85_steps"].iloc[0]) if len(healthy_summary) else 10500.0
    max_fraction = float(coeffs["final_outcome_model"].get("max_fraction_of_healthy_upper_plateau", 1.10))
    healthy_cap = healthy_p85 * max_fraction

    for _, patient in patients.iterrows():
        class_name = patient["recovery_trajectory_class"]
        class_cfg = recovery_cfg["trajectory_classes"][class_name]
        params = dict(class_cfg["parameters"])

        baseline_steps = float(patient["pre_treatment_baseline_steps"])
        rate_mult = patient_recovery_rate_multiplier(patient, coeffs)
        variability_cv = patient_variability_cv(patient, coeffs)
        treatment_mult = _treatment_multiplier(patient, coeffs)

        curve_params = _make_patient_curve_params(
            patient=patient,
            class_name=class_name,
            class_cfg=class_cfg,
            params=params,
            baseline_steps=baseline_steps,
            rate_mult=rate_mult,
            treatment_mult=treatment_mult,
            healthy_cap=healthy_cap,
            rng=rng,
        )
        setbacks = _sample_setbacks(patient, coeffs, class_cfg, post_days, rng)
        ar_noise = 0.0
        for day in days:
            date = start_date + pd.Timedelta(days=day)
            if day < 0:
                expected = baseline_steps
                phase = "pre_treatment_baseline"
            elif day == 0:
                expected = max(250.0, baseline_steps * rng.uniform(0.03, 0.10))
                phase = "treatment_day"
            else:
                expected = _class_curve(day, class_name, curve_params)
                phase = "post_treatment"

            expected *= _weekly_multiplier(date, coeffs)
            expected *= _season_multiplier(date, coeffs)
            expected *= _setback_multiplier(day, setbacks)

            phi = float(coeffs["variability_model"].get("ar1_phi", 0.60))
            innovation_sd = max(0.02, variability_cv * np.sqrt(1 - phi**2))
            ar_noise = phi * ar_noise + rng.normal(0.0, innovation_sd)
            noisy_steps = expected * np.exp(ar_noise) * rng.lognormal(0.0, variability_cv * 0.45)
            steps = int(safe_int_steps(noisy_steps, coeffs["final_outcome_model"]["min_steps"], coeffs["final_outcome_model"]["max_steps"]))

            rows.append({
                "patient_id": patient["patient_id"],
                "day": int(day),
                "date": date.date().isoformat(),
                "month": int(max(0, np.ceil(day / 30.4375))) if day > 0 else 0,
                "phase": phase,
                "true_steps": steps,
                "expected_steps_without_daily_noise": float(expected),
                "recovery_trajectory_class": class_name,
                "season": _season_label(date),
                "weekday": weekday_name(date),
                "age": patient["age"],
                "sex": patient["sex"],
                "bmi": patient["bmi"],
                "baseline_pain_nrs": patient["baseline_pain_nrs"],
                "baseline_function_score": patient["baseline_function_score"],
                "comorbidity_burden": patient["comorbidity_burden"],
                "osteoarthritis_severity": patient["osteoarthritis_severity"],
                "pre_treatment_baseline_steps": patient["pre_treatment_baseline_steps"],
                "digital_confidence_score": patient["digital_confidence_score"],
                "adherence_tendency": patient["adherence_tendency"],
                "treatment_effectiveness_score": patient["treatment_effectiveness_score"],
                "recovery_sensitivity_score": patient["recovery_sensitivity_score"],
                "variability_tendency": patient["variability_tendency"],
                "setback_tendency": patient["setback_tendency"],
                "missingness_tendency": patient["missingness_tendency"],
            })

        param_rows.append({
            "patient_id": patient["patient_id"],
            "recovery_trajectory_class": class_name,
            "recovery_parameters_json": json.dumps(curve_params, sort_keys=True),
            "n_setbacks": len(setbacks),
            "setbacks_json": json.dumps(setbacks, sort_keys=True),
            "variability_cv": variability_cv,
            "recovery_rate_multiplier": rate_mult,
        })

    return pd.DataFrame(rows), pd.DataFrame(param_rows)


def _treatment_multiplier(patient: pd.Series, coeffs: Dict[str, Any]) -> float:
    model = coeffs["treatment_effectiveness_model"]
    raw = 1.0
    raw += float(model["adherence_effect"]) * (patient["adherence_tendency"] - 0.5)
    raw += float(model["digital_confidence_effect"]) * (patient["digital_confidence_score"] / 100.0 - 0.5)
    raw += float(model["comorbidity_effect"]) * patient["comorbidity_burden"]
    raw += float(model["pain_effect"]) * (patient["baseline_pain_nrs"] - 5.5)
    raw += 0.35 * (patient["treatment_effectiveness_score"] - 0.5)
    return float(np.clip(raw, model["baseline_multiplier_low"], model["baseline_multiplier_high"]))


def _make_patient_curve_params(patient, class_name, class_cfg, params, baseline_steps, rate_mult, treatment_mult, healthy_cap, rng):
    day21_target = float(class_cfg.get("duong_early_target_day21_mean", baseline_steps))
    week12_target = float(class_cfg.get("duong_week12_target_mean", baseline_steps))
    early_sd = float(class_cfg.get("early_day_sd", 1500))
    week12_sd = float(class_cfg.get("week12_sd", 1800))

    early_target = 0.55 * baseline_steps * params.get("early_fraction_of_baseline", 0.8) + 0.45 * rng.normal(day21_target, early_sd * 0.35)
    week12_target_patient = 0.45 * baseline_steps + 0.55 * rng.normal(week12_target, week12_sd * 0.35)

    plateau_gain = params.get("plateau_gain_fraction", params.get("peak_gain_fraction", 0.20))
    plateau = max(week12_target_patient, baseline_steps * (1.0 + plateau_gain * treatment_mult))
    plateau = min(plateau, healthy_cap)
    rate = max(0.006, float(params.get("recovery_rate", 0.04)) * rate_mult)
    midpoint = float(params.get("midpoint_day", 70)) * rng.uniform(0.85, 1.15)

    return {
        "baseline_steps": float(baseline_steps),
        "early_target_day21": float(np.clip(early_target, 300, healthy_cap)),
        "week12_target": float(np.clip(week12_target_patient, 500, healthy_cap)),
        "plateau": float(np.clip(plateau, 500, healthy_cap)),
        "rate": float(rate),
        "midpoint_day": float(midpoint),
        "healthy_cap": float(healthy_cap),
        **{k: float(v) if isinstance(v, (int, float)) else v for k, v in params.items()},
    }


def _class_curve(day: int, class_name: str, p: Dict[str, float]) -> float:
    baseline = p["baseline_steps"]
    early = p["early_target_day21"]
    week12 = p["week12_target"]
    plateau = p["plateau"]
    rate = p["rate"]
    mid = p["midpoint_day"]

    if class_name == "stable_baseline":
        recovery = early + (plateau - early) * sigmoid(rate * (day - mid))
        drift = 1.0 + float(p.get("long_term_drift_per_day", 0.0)) * day
        return recovery * drift

    if class_name == "decreasing":
        early_recovery = early + (week12 - early) * sigmoid(rate * (day - mid))
        decline_start = 90
        decline = 1.0 - float(p.get("decline_fraction_year", 0.22)) * max(0, day - decline_start) / max(1, 365 - decline_start)
        return early_recovery * decline

    if class_name == "increasing":
        first = early + (plateau - early) * sigmoid(rate * (day - mid))
        late_gain = 1.0 + float(p.get("late_gain_fraction", 0.06)) * sigmoid(0.025 * (day - 160))
        return first * late_gain

    if class_name == "decrease_then_increase":
        immediate = baseline * float(p.get("immediate_post_treatment_fraction", 0.10))
        early_segment = immediate + (early - immediate) * sigmoid(0.18 * (day - 12))
        recovery_segment = early + (plateau - early) * sigmoid(rate * (day - mid))
        blend = sigmoid(0.18 * (day - 28))
        late_gain = 1.0 + float(p.get("late_gain_fraction", 0.08)) * sigmoid(0.020 * (day - 180))
        return ((1 - blend) * early_segment + blend * recovery_segment) * late_gain

    if class_name == "increase_then_decrease":
        peak_gain = float(p.get("peak_gain_fraction", 0.30))
        peak = min(p["healthy_cap"], baseline * (1.0 + peak_gain))
        rising = early + (peak - early) * sigmoid(rate * (day - mid))
        decline_start = float(p.get("decline_start_day", 170))
        decline_fraction = float(p.get("decline_fraction_after_peak", 0.23))
        decline = 1.0 - decline_fraction * sigmoid(0.035 * (day - decline_start))
        return rising * decline

    if class_name == "fluctuating_or_relapsing":
        base = early + (plateau - early) * sigmoid(rate * (day - mid))
        cycle = np.sin(2 * np.pi * day / float(p.get("relapse_cycle_days", 70)))
        amp = float(p.get("relapse_amplitude_fraction", 0.14))
        return base * (1.0 + amp * cycle)

    raise ValueError(f"Unknown trajectory class: {class_name}")


def _weekly_multiplier(date: pd.Timestamp, coeffs: Dict[str, Any]) -> float:
    pattern = coeffs["variability_model"].get("weekly_pattern", {})
    return 1.0 + float(pattern.get(weekday_name(date), 0.0))


def _season_label(date: pd.Timestamp) -> str:
    from .utils import get_season
    return get_season(date)


def _season_multiplier(date: pd.Timestamp, coeffs: Dict[str, Any]) -> float:
    season = _season_label(date)
    model = coeffs["baseline_step_model"]
    if season == "winter":
        return float(np.exp(model["winter_log_effect_vs_summer"]))
    if season == "spring_fall":
        return float(np.exp(model["spring_fall_log_effect_vs_summer"]))
    return 1.0


def _sample_setbacks(patient, coeffs, class_cfg, post_days, rng):
    model = coeffs["setback_model"]
    class_multiplier = float(class_cfg.get("parameters", {}).get("setback_multiplier", 1.0))
    p_day = float(model["base_event_probability_per_day"]) * class_multiplier
    p_day += float(model["setback_tendency_effect"]) * float(patient["setback_tendency"]) / 30.0
    p_day += float(model["pain_effect"]) * float(patient["baseline_pain_nrs"]) / 10.0
    p_day += float(model["comorbidity_effect"]) * float(patient["comorbidity_burden"])
    p_day = float(np.clip(p_day, 0.0002, 0.030))

    setbacks = []
    day = 1
    while day <= post_days:
        if rng.random() < p_day:
            duration = int(np.clip(rng.normal(model["default_duration_days_mean"], model["default_duration_days_sd"]), 2, 35))
            magnitude = float(np.clip(rng.normal(model["default_magnitude_fraction_mean"], model["default_magnitude_fraction_sd"]), 0.05, 0.65))
            setbacks.append({"start_day": int(day), "duration_days": int(duration), "magnitude_fraction": magnitude})
            day += duration + rng.integers(8, 35)
        else:
            day += 1
    return setbacks


def _setback_multiplier(day: int, setbacks) -> float:
    if day <= 0:
        return 1.0
    mult = 1.0
    for event in setbacks:
        start = event["start_day"]
        end = start + event["duration_days"]
        if start <= day < end:
            progress = (day - start) / max(1, event["duration_days"])
            severity = event["magnitude_fraction"] * (1.0 - 0.55 * progress)
            mult *= 1.0 - severity
    return max(0.15, mult)
