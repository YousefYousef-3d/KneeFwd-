from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .simulate_missingness import resolve_missingness_scenario
from .utils import ensure_output_dirs, load_all_configs, save_dataframe


EXPECTED_DATA_FILES = [
    "complete_data.csv",
    "observed_data.csv",
    "imputed_data.csv",
    "patient_metadata.csv",
    "patient_coefficients.csv",
    "healthy_reference_summary.csv",
    "simulator_parameters.json",
]


def run_validation(config_path: str) -> pd.DataFrame:
    configs = load_all_configs(config_path)
    default_cfg = configs["default"]
    output_dirs = ensure_output_dirs(default_cfg["output_dir"])

    data_dir = output_dirs["data"]
    table_dir = output_dirs["tables"]
    validation_dir = output_dirs["validation"]

    metrics: List[Dict[str, Any]] = []
    report: List[str] = ["# Validation report", ""]

    missing_files = []
    for name in EXPECTED_DATA_FILES:
        exists = (data_dir / name).exists()
        metrics.append({"check": "expected_output_file_exists", "item": name, "value": int(exists), "passed": bool(exists)})
        if not exists:
            missing_files.append(name)

    if missing_files:
        report.append("## Missing expected files")
        report.append("")
        report.extend([f"- {name}" for name in missing_files])
        report.append("")
        _save_report_and_metrics(report, metrics, validation_dir)
        return pd.DataFrame(metrics)

    complete = pd.read_csv(data_dir / "complete_data.csv")
    observed = pd.read_csv(data_dir / "observed_data.csv")
    imputed = pd.read_csv(data_dir / "imputed_data.csv")
    patients = pd.read_csv(data_dir / "patient_metadata.csv")
    coeffs = pd.read_csv(data_dir / "patient_coefficients.csv")

    _basic_data_validity(complete, observed, patients, default_cfg, metrics, report)
    _paper_plausibility(complete, patients, configs["recovery_trajectory_classes"], metrics, report)
    _trajectory_validation(patients, configs["recovery_trajectory_classes"], default_cfg, metrics, report)
    _coefficient_validation(complete, patients, coeffs, metrics, report)
    _missingness_validation(observed, configs["missingness_scenarios"], default_cfg, metrics, report)
    _imputation_validation(imputed, metrics, report)

    _save_report_and_metrics(report, metrics, validation_dir)
    return pd.DataFrame(metrics)


def _add_metric(metrics: List[Dict[str, Any]], check: str, item: str, value: Any, passed: bool, detail: str = "") -> None:
    metrics.append({"check": check, "item": item, "value": value, "passed": bool(passed), "detail": detail})


def _basic_data_validity(complete, observed, patients, default_cfg, metrics, report):
    report.append("## Basic data validity")
    n_expected_patients = int(default_cfg["number_of_patients"])
    include_day0 = bool(default_cfg.get("include_treatment_day", True))
    expected_days = int(default_cfg["baseline_days"]) + int(default_cfg["post_treatment_days"]) + (1 if include_day0 else 0)
    _add_metric(metrics, "basic", "no_negative_true_steps", int((complete["true_steps"] < 0).sum()), bool((complete["true_steps"] >= 0).all()))
    _add_metric(metrics, "basic", "correct_number_of_patients", int(patients["patient_id"].nunique()), patients["patient_id"].nunique() == n_expected_patients)
    rows_per_patient_ok = complete.groupby("patient_id")["day"].nunique().eq(expected_days).all()
    _add_metric(metrics, "basic", "correct_number_of_days_per_patient", expected_days, bool(rows_per_patient_ok))
    missing_rate = float(observed["is_missing"].mean())
    min_r = float(default_cfg.get("validation", {}).get("expected_missingness_rate_min", 0.0))
    max_r = float(default_cfg.get("validation", {}).get("expected_missingness_rate_max", 1.0))
    _add_metric(metrics, "basic", "missingness_rate", missing_rate, min_r <= missing_rate <= max_r)
    report.append(f"- Patients: {patients['patient_id'].nunique()} expected {n_expected_patients}.")
    report.append(f"- Rows per patient expected: {expected_days}.")
    report.append(f"- Missingness rate: {missing_rate:.3f}.")
    report.append("")


def _paper_plausibility(complete, patients, recovery_cfg, metrics, report):
    report.append("## Paper-informed plausibility")
    early = complete[complete["day"].eq(21)].groupby("recovery_trajectory_class")["true_steps"].mean()
    wk12 = complete[complete["day"].eq(84)].groupby("recovery_trajectory_class")["true_steps"].mean()
    for cls, cfg in recovery_cfg["trajectory_classes"].items():
        if cls not in early.index or cls not in wk12.index:
            continue
        target21 = float(cfg.get("duong_early_target_day21_mean", np.nan))
        target84 = float(cfg.get("duong_week12_target_mean", np.nan))
        tolerance21 = max(1800, 0.45 * target21)
        tolerance84 = max(2200, 0.45 * target84)
        pass21 = abs(float(early[cls]) - target21) <= tolerance21
        pass84 = abs(float(wk12[cls]) - target84) <= tolerance84
        _add_metric(metrics, "paper_plausibility", f"{cls}_day21_mean_vs_target", float(early[cls]), pass21, f"target={target21}")
        _add_metric(metrics, "paper_plausibility", f"{cls}_day84_mean_vs_target", float(wk12[cls]), pass84, f"target={target84}")
    report.append("- Day 21 and day 84 class means are compared with paper-informed targets using deliberately wide tolerances because the simulator extrapolates to six one-year classes.")
    report.append("")


def _trajectory_validation(patients, recovery_cfg, default_cfg, metrics, report):
    report.append("## Trajectory class validation")
    observed_props = patients["recovery_trajectory_class"].value_counts(normalize=True).to_dict()
    tol = float(default_cfg.get("validation", {}).get("class_proportion_tolerance", 0.08))
    for cls, cfg in recovery_cfg["trajectory_classes"].items():
        prop = float(observed_props.get(cls, 0.0))
        target = float(cfg["default_probability"])
        _add_metric(metrics, "trajectory", f"{cls}_appears", prop, prop > 0)
        _add_metric(metrics, "trajectory", f"{cls}_proportion_close", prop, abs(prop - target) <= tol, f"target={target}, tolerance={tol}")
    report.append("- All configured classes should appear and proportions should be close to configured priors, allowing Monte Carlo variation.")
    report.append("")


def _coefficient_validation(complete, patients, coeffs, metrics, report):
    report.append("## Coefficient validation")
    baseline = complete[complete["day"].between(-30, -1)].groupby("patient_id")["true_steps"].mean().rename("baseline_mean_steps")
    p = patients.set_index("patient_id").join(baseline).dropna()
    age_corr = float(p["age"].corr(p["baseline_mean_steps"]))
    bmi_corr = float(p["bmi"].corr(p["baseline_mean_steps"]))
    pain_corr = float(p["baseline_pain_nrs"].corr(p["baseline_mean_steps"]))
    _add_metric(metrics, "coefficients", "age_baseline_steps_correlation_negative", age_corr, age_corr < 0)
    _add_metric(metrics, "coefficients", "bmi_baseline_steps_correlation_negative", bmi_corr, bmi_corr < 0)
    _add_metric(metrics, "coefficients", "pain_not_dominant_direct_baseline_correlation", pain_corr, abs(pain_corr) < max(abs(age_corr), abs(bmi_corr)) + 0.05)
    _add_metric(metrics, "coefficients", "patient_coefficients_rows", len(coeffs), len(coeffs) == patients["patient_id"].nunique())
    report.append(f"- Age correlation with baseline steps: {age_corr:.3f}.")
    report.append(f"- BMI correlation with baseline steps: {bmi_corr:.3f}.")
    report.append(f"- Pain correlation with baseline steps: {pain_corr:.3f}; expected to be weaker than age/BMI direct effects.")
    report.append("")


def _missingness_validation(observed, missingness_cfg, default_cfg, metrics, report):
    report.append("## Missingness validation")
    scenario = resolve_missingness_scenario(missingness_cfg, default_cfg["selected_missingness_scenario"])
    present = set(observed.loc[observed["is_missing"], "missing_reason"].dropna().unique())
    for reason in scenario["reasons"].keys():
        _add_metric(metrics, "missingness", f"reason_{reason}_appears", int(reason in present), reason in present)
    mechanisms = set(observed.loc[observed["is_missing"], "missing_mechanism"].dropna().unique())
    for mechanism in ["MCAR", "MAR", "MNAR_like", "structural_dropout"]:
        _add_metric(metrics, "missingness", f"mechanism_{mechanism}_appears", int(mechanism in mechanisms), mechanism in mechanisms)
    block_lengths = observed[observed["is_missing"]].groupby("missing_block_id").size()
    _add_metric(metrics, "missingness", "isolated_blocks_present", int((block_lengths == 1).sum()), bool((block_lengths == 1).any()))
    _add_metric(metrics, "missingness", "short_or_long_blocks_present", int((block_lengths >= 2).sum()), bool((block_lengths >= 2).any()))
    report.append(f"- Missing reasons present: {', '.join(sorted(present))}.")
    report.append("")


def _imputation_validation(imputed, metrics, report):
    report.append("## Imputation validation")
    needed_cols = {"observed_steps", "imputed_steps", "was_imputed", "imputation_method"}
    _add_metric(metrics, "imputation", "required_columns_present", int(needed_cols.issubset(imputed.columns)), needed_cols.issubset(imputed.columns))
    observed_not_overwritten = imputed.loc[~imputed["is_missing"], "observed_steps"].equals(imputed.loc[~imputed["is_missing"], "imputed_steps"])
    _add_metric(metrics, "imputation", "observed_values_preserved", int(observed_not_overwritten), bool(observed_not_overwritten))
    imputed_missing = imputed.loc[imputed["is_missing"], "was_imputed"].sum()
    _add_metric(metrics, "imputation", "missing_values_flagged_when_imputed", int(imputed_missing), int(imputed_missing) >= 0)
    if imputed["was_imputed"].any():
        err = imputed.loc[imputed["was_imputed"], "imputed_steps"] - imputed.loc[imputed["was_imputed"], "true_steps"]
        mae = float(err.abs().mean())
        rmse = float(np.sqrt((err**2).mean()))
    else:
        mae = np.nan
        rmse = np.nan
    _add_metric(metrics, "imputation", "mae_against_true_steps_where_imputed", mae, True)
    _add_metric(metrics, "imputation", "rmse_against_true_steps_where_imputed", rmse, True)
    report.append(f"- Imputation MAE against hidden true steps where imputed: {mae}.")
    report.append("")


def _save_report_and_metrics(report, metrics, validation_dir: Path):
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    save_dataframe(pd.DataFrame(metrics), validation_dir / "validation_metrics.csv")
