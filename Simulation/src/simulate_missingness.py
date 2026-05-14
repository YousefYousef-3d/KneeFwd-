from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .utils import sigmoid


def resolve_missingness_scenario(missingness_cfg: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    scenarios = missingness_cfg["scenarios"]
    scenario = dict(scenarios[scenario_name])
    parent_name = scenario.get("inherits")
    if parent_name:
        parent = resolve_missingness_scenario(missingness_cfg, parent_name)
        merged = dict(parent)
        for key, value in scenario.items():
            if key == "reasons" and isinstance(value, dict):
                reasons = dict(parent.get("reasons", {}))
                for reason, reason_cfg in value.items():
                    base = dict(reasons.get(reason, {}))
                    base.update(reason_cfg)
                    reasons[reason] = base
                merged["reasons"] = reasons
            else:
                merged[key] = value
        scenario = merged
    return scenario


def apply_missingness(
    complete_df: pd.DataFrame,
    patients: pd.DataFrame,
    missingness_cfg: Dict[str, Any],
    coeffs: Dict[str, Any],
    scenario_name: str,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scenario = resolve_missingness_scenario(missingness_cfg, scenario_name)
    df = complete_df.copy()
    df["observed_steps"] = df["true_steps"].astype(float)
    df["is_missing"] = False
    df["missing_reason"] = "observed"
    df["missing_mechanism"] = "observed"
    df["missing_block_id"] = pd.NA

    global_mult = float(scenario.get("global_probability_multiplier", 1.0))
    block_id = 0
    reason_rows = []

    patient_lookup = patients.set_index("patient_id")
    reasons = scenario["reasons"]

    for pid, patient in patient_lookup.iterrows():
        patient_mask = df["patient_id"].eq(pid)
        patient_idx = df.index[patient_mask].to_numpy()
        patient_days = df.loc[patient_idx, "day"].to_numpy()

        # Structural dropout first because it dominates later observations.
        dropout_cfg = reasons.get("dropout")
        if dropout_cfg:
            p_dropout = float(dropout_cfg.get("patient_probability", 0.0)) * float(scenario.get("dropout_probability_multiplier", 1.0))
            p_dropout = _adjust_probability(p_dropout, dropout_cfg, patient, None, coeffs, patient_level=True)
            if rng.random() < p_dropout:
                start = int(rng.integers(int(dropout_cfg.get("earliest_day", 45)), int(dropout_cfg.get("latest_day", 310)) + 1))
                affected = patient_idx[patient_days >= start]
                if len(affected):
                    block_id += 1
                    df.loc[affected, ["is_missing", "observed_steps", "missing_reason", "missing_mechanism", "missing_block_id"]] = [
                        True,
                        np.nan,
                        "dropout",
                        dropout_cfg.get("mechanism", "structural_dropout"),
                        block_id,
                    ]
                    reason_rows.append({"patient_id": pid, "reason": "dropout", "block_id": block_id, "start_day": start, "duration_days": int(len(affected))})

        for reason, reason_cfg in reasons.items():
            if reason == "dropout":
                continue
            post_idx = patient_idx[(patient_days >= 1)]
            if len(post_idx) == 0:
                continue
            for idx in post_idx:
                if bool(df.at[idx, "is_missing"]):
                    continue
                true_steps = float(df.at[idx, "true_steps"])
                p_event = float(reason_cfg.get("base_probability", 0.0)) * global_mult
                p_event = _adjust_probability(p_event, reason_cfg, patient, true_steps, coeffs, patient_level=False)
                if rng.random() < p_event:
                    block_type = _sample_block_type(reason_cfg, rng)
                    length = _sample_length(reason_cfg, block_type, rng)
                    start_day = int(df.at[idx, "day"])
                    affected = df.index[
                        df["patient_id"].eq(pid)
                        & df["day"].between(start_day, start_day + length - 1)
                        & (~df["is_missing"])
                    ].to_numpy()
                    if len(affected):
                        block_id += 1
                        df.loc[affected, "observed_steps"] = np.nan
                        df.loc[affected, "is_missing"] = True
                        df.loc[affected, "missing_reason"] = reason
                        df.loc[affected, "missing_mechanism"] = reason_cfg.get("mechanism", "unknown")
                        df.loc[affected, "missing_block_id"] = block_id
                        reason_rows.append({"patient_id": pid, "reason": reason, "block_id": block_id, "start_day": start_day, "duration_days": int(len(affected))})

    observed_cols = [
        "patient_id", "day", "date", "month", "phase", "true_steps", "observed_steps", "is_missing",
        "missing_reason", "missing_mechanism", "missing_block_id", "recovery_trajectory_class",
        "season", "weekday", "age", "sex", "bmi", "baseline_pain_nrs", "baseline_function_score",
        "comorbidity_burden", "osteoarthritis_severity", "pre_treatment_baseline_steps",
        "digital_confidence_score", "adherence_tendency", "treatment_effectiveness_score",
        "recovery_sensitivity_score", "variability_tendency", "setback_tendency", "missingness_tendency",
    ]
    return df[observed_cols], pd.DataFrame(reason_rows)


def _adjust_probability(base_p: float, cfg: Dict[str, Any], patient: pd.Series, true_steps: float | None, coeffs: Dict[str, Any], patient_level: bool) -> float:
    if base_p <= 0:
        return 0.0
    logit = np.log(base_p / max(1e-9, 1.0 - base_p))
    for var, effect in cfg.get("patient_variable_effects", {}).items():
        if var == "digital_confidence_score":
            x = (float(patient[var]) - 72.2) / 21.2
        elif var == "baseline_pain_nrs":
            x = (float(patient[var]) - 5.5) / 2.7
        elif var in ("adherence_tendency", "missingness_tendency", "variability_tendency"):
            x = float(patient[var]) - 0.5
        elif var == "comorbidity_burden":
            x = float(patient[var])
        else:
            x = float(patient.get(var, 0.0))
        logit += float(effect) * x
    if true_steps is not None:
        baseline = max(500.0, float(patient["pre_treatment_baseline_steps"]))
        low_activity = max(0.0, (baseline - true_steps) / baseline)
        abrupt_drop = 1.0 if true_steps < 0.55 * baseline else 0.0
        effects = cfg.get("true_activity_effects", {})
        logit += float(effects.get("low_activity", 0.0)) * low_activity
        logit += float(effects.get("abrupt_drop", 0.0)) * abrupt_drop
    return float(np.clip(sigmoid(logit), 0.0, 0.95 if patient_level else 0.50))


def _sample_block_type(cfg: Dict[str, Any], rng: np.random.Generator) -> str:
    types = cfg.get("block_types", ["isolated"])
    probs = cfg.get("block_type_probabilities", None)
    if probs is None:
        probs = np.ones(len(types)) / len(types)
    else:
        probs = np.asarray(probs, dtype=float)
        probs = probs / probs.sum()
    return str(rng.choice(types, p=probs))


def _sample_length(cfg: Dict[str, Any], block_type: str, rng: np.random.Generator) -> int:
    dist = cfg.get("block_length_distribution", {}).get(block_type, {"min": 1, "max": 1})
    return int(rng.integers(int(dist.get("min", 1)), int(dist.get("max", 1)) + 1))
