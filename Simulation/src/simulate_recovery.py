from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .generate_patients import generate_patients
from .impute_missingness import imputation_summary, impute_missingness
from .plotting import plot_all
from .recovery_models import load_healthy_reference, simulate_recovery_trajectories
from .simulate_missingness import apply_missingness
from .utils import ensure_output_dirs, flatten_dict, load_all_configs, save_dataframe, write_json


def run_simulation(config_path: str) -> Dict[str, pd.DataFrame]:
    import numpy as np

    configs = load_all_configs(config_path)
    default_cfg = configs["default"]
    rng = np.random.default_rng(int(default_cfg["random_seed"]))
    output_dirs = ensure_output_dirs(default_cfg["output_dir"])

    patient_cfg = configs["patient_generation"]
    coeffs = configs["coefficients"]
    recovery_cfg = configs["recovery_trajectory_classes"]
    missingness_cfg = configs["missingness_scenarios"]

    healthy_reference, healthy_summary = load_healthy_reference(
        path=default_cfg.get("healthy_reference_path", "data/external/healthy_steps_reference.csv"),
        use_healthy_reference=bool(default_cfg.get("use_healthy_reference", True)),
        rng=rng,
    )
    save_dataframe(healthy_summary, output_dirs["data"] / "healthy_reference_summary.csv")

    patients, patient_coefficients = generate_patients(
        n=int(default_cfg["number_of_patients"]),
        default_cfg=default_cfg,
        patient_cfg=patient_cfg,
        coeffs=coeffs,
        recovery_cfg=recovery_cfg,
        rng=rng,
    )

    complete_df, recovery_params = simulate_recovery_trajectories(
        patients=patients,
        default_cfg=default_cfg,
        recovery_cfg=recovery_cfg,
        coeffs=coeffs,
        healthy_summary=healthy_summary,
        rng=rng,
    )
    patients = patients.merge(recovery_params, on=["patient_id", "recovery_trajectory_class"], how="left")

    observed_df, missingness_blocks = apply_missingness(
        complete_df=complete_df,
        patients=patients,
        missingness_cfg=missingness_cfg,
        coeffs=coeffs,
        scenario_name=default_cfg["selected_missingness_scenario"],
        rng=rng,
    )

    save_dataframe(complete_df, output_dirs["data"] / "complete_data.csv")
    save_dataframe(observed_df, output_dirs["data"] / "observed_data.csv")
    save_dataframe(patients, output_dirs["data"] / "patient_metadata.csv")
    save_dataframe(patient_coefficients, output_dirs["data"] / "patient_coefficients.csv")

    imputed_df = None
    if bool(default_cfg.get("run_imputation", True)):
        imputed_df = impute_missingness(observed_df, default_cfg.get("selected_imputation_method", "linear_interpolation"), default_cfg)
    else:
        imputed_df = impute_missingness(observed_df, "no_imputation", default_cfg)
    save_dataframe(imputed_df, output_dirs["data"] / "imputed_data.csv")

    _write_summary_tables(complete_df, observed_df, imputed_df, patients, output_dirs["tables"])
    _write_simulator_parameters(configs, config_path, output_dirs["data"] / "simulator_parameters.json")

    if bool(default_cfg.get("run_plotting", True)):
        plot_all(
            complete_df=complete_df,
            observed_df=observed_df,
            imputed_df=imputed_df,
            patients=patients,
            healthy_reference=healthy_reference,
            output_dir=default_cfg["output_dir"],
            figure_settings=default_cfg.get("figure_settings", {}),
        )

    return {
        "patients": patients,
        "complete_data": complete_df,
        "observed_data": observed_df,
        "imputed_data": imputed_df,
        "patient_coefficients": patient_coefficients,
        "healthy_reference_summary": healthy_summary,
        "missingness_blocks": missingness_blocks,
    }


def _write_summary_tables(complete_df: pd.DataFrame, observed_df: pd.DataFrame, imputed_df: pd.DataFrame, patients: pd.DataFrame, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)

    group_summary = complete_df[complete_df["day"] >= 1].groupby("recovery_trajectory_class").agg(
        n_patients=("patient_id", "nunique"),
        mean_true_steps=("true_steps", "mean"),
        median_true_steps=("true_steps", "median"),
        sd_true_steps=("true_steps", "std"),
    ).reset_index()
    save_dataframe(group_summary, table_dir / "group_summary.csv")

    proportions = patients["recovery_trajectory_class"].value_counts(normalize=True).rename_axis("recovery_trajectory_class").reset_index(name="observed_proportion")
    counts = patients["recovery_trajectory_class"].value_counts().rename_axis("recovery_trajectory_class").reset_index(name="n_patients")
    save_dataframe(counts.merge(proportions, on="recovery_trajectory_class"), table_dir / "trajectory_class_summary.csv")

    archetypes = patients.groupby("recovery_trajectory_class").agg(
        n_patients=("patient_id", "nunique"),
        mean_age=("age", "mean"),
        mean_bmi=("bmi", "mean"),
        mean_baseline_pain=("baseline_pain_nrs", "mean"),
        mean_baseline_steps=("pre_treatment_baseline_steps", "mean"),
        mean_adherence=("adherence_tendency", "mean"),
        mean_treatment_effectiveness=("treatment_effectiveness_score", "mean"),
    ).reset_index()
    save_dataframe(archetypes, table_dir / "patient_archetype_summary.csv")

    missingness_summary = pd.DataFrame([
        {
            "n_rows": len(observed_df),
            "n_missing": int(observed_df["is_missing"].sum()),
            "missingness_rate": float(observed_df["is_missing"].mean()),
            "n_patients_with_any_missingness": int(observed_df.loc[observed_df["is_missing"], "patient_id"].nunique()),
        }
    ])
    save_dataframe(missingness_summary, table_dir / "missingness_summary.csv")

    by_reason = observed_df[observed_df["is_missing"]].groupby(["missing_reason", "missing_mechanism"]).size().reset_index(name="n_missing_rows")
    by_reason["missing_fraction_of_all_rows"] = by_reason["n_missing_rows"] / len(observed_df)
    save_dataframe(by_reason, table_dir / "missingness_by_reason.csv")

    by_month = observed_df[observed_df["day"] >= 1].groupby("month").agg(
        n_rows=("patient_id", "size"),
        n_missing=("is_missing", "sum"),
        missingness_rate=("is_missing", "mean"),
    ).reset_index()
    save_dataframe(by_month, table_dir / "missingness_by_month.csv")

    save_dataframe(imputation_summary(imputed_df), table_dir / "imputation_summary.csv")


def _write_simulator_parameters(configs: Dict[str, Any], config_path: str, out_path: Path) -> None:
    payload = {
        "config_path_used": config_path,
        "configs": configs,
        "flat_parameters": flatten_dict(configs),
    }
    write_json(payload, out_path)
