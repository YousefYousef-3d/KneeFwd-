from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_all(
    complete_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    imputed_df: pd.DataFrame | None,
    patients: pd.DataFrame,
    healthy_reference: pd.DataFrame,
    output_dir: str | Path,
    figure_settings: Dict[str, Any],
) -> None:
    out = Path(output_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    dpi = int(figure_settings.get("dpi", 150))
    max_per_class = int(figure_settings.get("max_patients_per_class_plot", 3))
    rolling = int(figure_settings.get("rolling_window_days", 7))

    plot_example_trajectories_by_class(complete_df, out / "example_trajectories_by_class.png", max_per_class, rolling, dpi)
    plot_example_trajectories_with_missingness(observed_df, out / "example_trajectories_with_missingness.png", max_per_class, rolling, dpi)
    plot_class_mean_curves(complete_df, out / "trajectory_class_mean_curves.png", rolling, dpi)
    plot_patient_covariate_effects(complete_df, patients, out / "patient_covariate_effects.png", dpi)
    plot_missingness_heatmap(observed_df, out / "missingness_heatmap.png", dpi)
    plot_missingness_by_reason(observed_df, out / "missingness_by_reason.png", dpi)
    plot_missingness_by_month(observed_df, out / "missingness_by_month.png", dpi)
    if imputed_df is not None:
        plot_imputation_comparison(imputed_df, out / "imputation_comparison.png", dpi)
    plot_healthy_reference_distribution(healthy_reference, out / "healthy_reference_distribution.png", dpi)


def plot_example_trajectories_by_class(df: pd.DataFrame, path: Path, max_per_class: int, rolling: int, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for cls, group in df[df["day"] >= 1].groupby("recovery_trajectory_class"):
        for pid in group["patient_id"].drop_duplicates().head(max_per_class):
            sub = group[group["patient_id"] == pid].sort_values("day")
            y = sub["true_steps"].rolling(rolling, min_periods=1).mean()
            ax.plot(sub["day"], y, linewidth=1.0, alpha=0.75, label=cls if pid == group["patient_id"].iloc[0] else None)
    ax.set_xlabel("Day after treatment")
    ax.set_ylabel("Daily steps, rolling mean")
    ax.set_title("Example complete trajectories by recovery class")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_example_trajectories_with_missingness(df: pd.DataFrame, path: Path, max_per_class: int, rolling: int, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    chosen = []
    for _, group in df[df["day"] >= 1].groupby("recovery_trajectory_class"):
        chosen.extend(group["patient_id"].drop_duplicates().head(max_per_class).tolist())
    subdf = df[df["patient_id"].isin(chosen) & (df["day"] >= 1)].copy()
    for pid, sub in subdf.groupby("patient_id"):
        sub = sub.sort_values("day")
        ax.plot(sub["day"], sub["true_steps"].rolling(rolling, min_periods=1).mean(), linewidth=0.8, alpha=0.35)
        obs = sub[~sub["is_missing"]]
        ax.scatter(obs["day"], obs["observed_steps"], s=3, alpha=0.45)
    ax.set_xlabel("Day after treatment")
    ax.set_ylabel("Steps")
    ax.set_title("Example trajectories with observed points after missingness")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_class_mean_curves(df: pd.DataFrame, path: Path, rolling: int, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    post = df[df["day"] >= 1]
    means = post.groupby(["recovery_trajectory_class", "day"])["true_steps"].mean().reset_index()
    for cls, sub in means.groupby("recovery_trajectory_class"):
        y = sub["true_steps"].rolling(rolling, min_periods=1).mean()
        ax.plot(sub["day"], y, linewidth=2, label=cls)
    ax.set_xlabel("Day after treatment")
    ax.set_ylabel("Mean true steps")
    ax.set_title("Mean complete trajectory by recovery class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_patient_covariate_effects(df: pd.DataFrame, patients: pd.DataFrame, path: Path, dpi: int) -> None:
    post_mean = df[df["day"].between(1, 84)].groupby("patient_id")["true_steps"].mean().rename("mean_steps_days_1_84")
    p = patients.set_index("patient_id").join(post_mean).reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(p["bmi"], p["mean_steps_days_1_84"], s=18, alpha=0.7)
    ax.set_xlabel("BMI")
    ax.set_ylabel("Mean steps days 1-84")
    ax.set_title("Patient covariate effect check: BMI vs early steps")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_missingness_heatmap(df: pd.DataFrame, path: Path, dpi: int) -> None:
    post = df[df["day"] >= 1]
    pivot = post.pivot_table(index="patient_id", columns="day", values="is_missing", aggfunc="max", fill_value=False).astype(int)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Day after treatment")
    ax.set_ylabel("Patient")
    ax.set_title("Missingness heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_missingness_by_reason(df: pd.DataFrame, path: Path, dpi: int) -> None:
    counts = df[df["is_missing"]].groupby("missing_reason").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Missingness reason")
    ax.set_ylabel("Missing rows")
    ax.set_title("Missingness by reason")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_missingness_by_month(df: pd.DataFrame, path: Path, dpi: int) -> None:
    post = df[df["day"] >= 1]
    rate = post.groupby("month")["is_missing"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    rate.plot(kind="bar", ax=ax)
    ax.set_xlabel("Month after treatment")
    ax.set_ylabel("Missingness rate")
    ax.set_title("Missingness by month")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_imputation_comparison(df: pd.DataFrame, path: Path, dpi: int) -> None:
    missing = df[df["was_imputed"]]
    fig, ax = plt.subplots(figsize=(7, 7))
    if len(missing):
        sample = missing.sample(min(4000, len(missing)), random_state=1)
        ax.scatter(sample["true_steps"], sample["imputed_steps"], s=8, alpha=0.5)
        lim = max(sample["true_steps"].max(), sample["imputed_steps"].max())
        ax.plot([0, lim], [0, lim], linewidth=1)
    ax.set_xlabel("True steps")
    ax.set_ylabel("Imputed steps")
    ax.set_title("Imputation comparison on artificially hidden true values")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_healthy_reference_distribution(df: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(df) and "steps" in df.columns:
        ax.hist(df["steps"].dropna(), bins=40)
    ax.set_xlabel("Steps/day")
    ax.set_ylabel("Frequency")
    ax.set_title("Healthy reference distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
