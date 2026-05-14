from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_ORDER = [
    "stable_baseline",
    "decreasing",
    "increasing",
    "decrease_then_increase",
    "increase_then_decrease",
    "fluctuating_or_relapsing",
]


def plot_all(
    complete_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    imputed_df: pd.DataFrame | None,
    patients: pd.DataFrame,
    healthy_reference: pd.DataFrame,
    output_dir: str | Path,
    figure_settings: Dict[str, Any],
) -> None:
    """Create only the Task 1 presentation plots requested by the team.

    This plotting suite deliberately avoids the old combined class-mean plot,
    patient-covariate plot, missingness heatmap, healthy-reference plot, and
    imputation-comparison plot. Instead it makes one clear trajectory figure per
    recovery class and several non-heatmap missingness diagnostics.
    """

    out = Path(output_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)

    dpi = int(figure_settings.get("dpi", 150))
    max_per_class = int(figure_settings.get("max_patients_per_class_plot", 6))
    max_missing_examples = int(figure_settings.get("max_patients_missingness_plot", 12))
    rolling = int(figure_settings.get("rolling_window_days", 7))

    plot_trajectory_classes_separately(
        complete_df=complete_df,
        output_dir=out,
        max_per_class=max_per_class,
        rolling=rolling,
        dpi=dpi,
    )

    plot_missingness_examples_with_gaps(
        observed_df=observed_df,
        path=out / "missingness_examples_with_gaps.png",
        max_patients=max_missing_examples,
        rolling=rolling,
        dpi=dpi,
    )
    plot_missingness_by_reason(observed_df, out / "missingness_by_reason.png", dpi)
    plot_missingness_by_mechanism(observed_df, out / "missingness_by_mechanism.png", dpi)
    plot_missingness_by_month(observed_df, out / "missingness_by_month.png", dpi)
    plot_missingness_timeline_by_reason(observed_df, out / "missingness_timeline_by_reason.png", dpi)
    plot_missingness_block_lengths(observed_df, out / "missingness_block_lengths.png", dpi)
    plot_missingness_patient_burden_distribution(
        observed_df,
        out / "missingness_patient_burden_distribution.png",
        dpi,
    )


# ---------------------------------------------------------------------------
# Recovery trajectory plots
# ---------------------------------------------------------------------------


def plot_trajectory_classes_separately(
    complete_df: pd.DataFrame,
    output_dir: Path,
    max_per_class: int,
    rolling: int,
    dpi: int,
) -> None:
    """Save one clean example trajectory figure for each recovery class."""

    post = complete_df[complete_df["day"] >= 1].copy()
    available_classes = list(post["recovery_trajectory_class"].dropna().unique())
    ordered_classes = [c for c in CLASS_ORDER if c in available_classes]
    ordered_classes += sorted(c for c in available_classes if c not in ordered_classes)

    for cls in ordered_classes:
        sub = post[post["recovery_trajectory_class"] == cls]
        path = output_dir / f"trajectory_class_{_safe_filename(cls)}.png"
        plot_single_class_trajectories(sub, cls, path, max_per_class, rolling, dpi)


def plot_single_class_trajectories(
    class_df: pd.DataFrame,
    class_name: str,
    path: Path,
    max_per_class: int,
    rolling: int,
    dpi: int,
) -> None:
    if class_df.empty:
        return

    selected_ids = _select_diverse_patients(class_df, max_per_class)
    selected = class_df[class_df["patient_id"].isin(selected_ids)].copy()

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for pid, sub in selected.groupby("patient_id", sort=False):
        sub = sub.sort_values("day")
        y_raw = sub["true_steps"].astype(float)
        y_smooth = y_raw.rolling(rolling, min_periods=1).mean()
        ax.plot(sub["day"], y_raw, linewidth=0.55, alpha=0.18)
        ax.plot(sub["day"], y_smooth, linewidth=1.8, label=f"patient {pid}")

    ax.axvspan(1, 84, alpha=0.06, label="first 12 weeks")
    ax.axvline(84, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(84, ax.get_ylim()[1] * 0.96, "week 12", ha="right", va="top", fontsize=9)

    ax.set_xlabel("Day after treatment")
    ax.set_ylabel(f"Daily steps ({rolling}-day rolling mean emphasized)")
    ax.set_title(f"Recovery class: {_pretty_label(class_name)}")
    ax.set_xlim(1, 365)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=False)

    note = (
        f"Showing up to {max_per_class} example patients. "
        "Thin lines are raw true daily steps; thicker lines are smoothed."
    )
    fig.text(0.01, 0.01, note, fontsize=8, alpha=0.8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Missingness plots: no heatmap
# ---------------------------------------------------------------------------


def plot_missingness_examples_with_gaps(
    observed_df: pd.DataFrame,
    path: Path,
    max_patients: int,
    rolling: int,
    dpi: int,
) -> None:
    """Show example trajectories with missing periods as visible gaps/rug marks.

    This replaces the old heatmap-style view. It is meant for explaining what
    missingness looks like on individual patient trajectories.
    """

    post = observed_df[observed_df["day"] >= 1].copy()
    if post.empty:
        return

    selected_ids = _select_patients_for_missingness_examples(post, max_patients)
    selected = post[post["patient_id"].isin(selected_ids)].copy()
    n = len(selected_ids)
    if n == 0:
        return

    height = max(5.5, min(18.0, 1.25 * n + 1.5))
    fig, axes = plt.subplots(n, 1, figsize=(12, height), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, pid in zip(axes, selected_ids):
        sub = selected[selected["patient_id"] == pid].sort_values("day")
        class_name = str(sub["recovery_trajectory_class"].iloc[0])
        y_true = sub["true_steps"].astype(float).rolling(rolling, min_periods=1).mean()
        obs = sub[~sub["is_missing"]]
        miss = sub[sub["is_missing"]]

        ax.plot(sub["day"], y_true, linewidth=1.4, alpha=0.85)
        ax.scatter(obs["day"], obs["observed_steps"], s=6, alpha=0.55)
        if len(miss):
            ymin, ymax = _safe_ylim_for_steps(sub["true_steps"])
            ax.vlines(miss["day"], ymin=ymin, ymax=ymin + 0.08 * (ymax - ymin), linewidth=0.8, alpha=0.8)
            ax.set_ylim(ymin, ymax)

        miss_rate = 100.0 * float(sub["is_missing"].mean())
        ax.set_ylabel(f"{pid}\n{miss_rate:.0f}% miss", rotation=0, ha="right", va="center", fontsize=8)
        ax.grid(True, axis="y", alpha=0.18)
        ax.text(0.995, 0.82, _pretty_label(class_name), transform=ax.transAxes, ha="right", va="top", fontsize=8)

    axes[-1].set_xlabel("Day after treatment")
    fig.suptitle("Example trajectories with observed points and missing-day rug marks", y=0.995)
    fig.text(
        0.01,
        0.01,
        "Line = smoothed true trajectory; points = observed step counts; bottom rug marks = missing days.",
        fontsize=8,
        alpha=0.8,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.98])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_missingness_by_reason(df: pd.DataFrame, path: Path, dpi: int) -> None:
    missing = df[df["is_missing"]].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing rows", ha="center", va="center")
    else:
        counts = missing.groupby("missing_reason").size().sort_values(ascending=True)
        total_rows = max(len(df), 1)
        counts.plot(kind="barh", ax=ax)
        for i, value in enumerate(counts.values):
            pct = 100.0 * value / total_rows
            ax.text(value, i, f" {value:,} rows ({pct:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("Missing rows")
    ax.set_ylabel("Missingness reason")
    ax.set_title("Missingness volume by reason")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_missingness_by_mechanism(df: pd.DataFrame, path: Path, dpi: int) -> None:
    missing = df[df["is_missing"]].copy()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing rows", ha="center", va="center")
    else:
        rate = missing.groupby("missing_mechanism").size().sort_values(ascending=False)
        rate = rate / max(len(df), 1)
        rate.plot(kind="bar", ax=ax)
        ax.set_ylim(0, max(rate.max() * 1.25, 0.01))
        for i, value in enumerate(rate.values):
            ax.text(i, value, f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Missingness mechanism")
    ax.set_ylabel("Fraction of all rows")
    ax.set_title("Missingness by mechanism label")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_missingness_by_month(df: pd.DataFrame, path: Path, dpi: int) -> None:
    post = df[df["day"] >= 1].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    if post.empty:
        ax.text(0.5, 0.5, "No post-treatment rows", ha="center", va="center")
    else:
        monthly = post.groupby("month")["is_missing"].mean().sort_index()
        monthly.plot(kind="bar", ax=ax)
        for i, value in enumerate(monthly.values):
            ax.text(i, value, f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(monthly.max() * 1.25, 0.01))
    ax.set_xlabel("Month after treatment")
    ax.set_ylabel("Missingness rate")
    ax.set_title("Missingness rate by month")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_missingness_timeline_by_reason(df: pd.DataFrame, path: Path, dpi: int) -> None:
    post = df[df["day"] >= 1].copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    if post.empty or not bool(post["is_missing"].any()):
        ax.text(0.5, 0.5, "No post-treatment missing rows", ha="center", va="center")
    else:
        missing = post[post["is_missing"]]
        rates = (
            missing.groupby(["day", "missing_reason"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=range(1, int(post["day"].max()) + 1), fill_value=0)
        )
        denominators = post.groupby("day").size().reindex(rates.index).replace(0, np.nan)
        rates = rates.div(denominators, axis=0).rolling(14, min_periods=1).mean()
        ax.stackplot(rates.index, [rates[col].values for col in rates.columns], labels=list(rates.columns), alpha=0.85)
        ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    ax.set_xlabel("Day after treatment")
    ax.set_ylabel("14-day rolling missingness rate")
    ax.set_title("Missingness over time by reason")
    ax.set_xlim(1, 365)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_missingness_block_lengths(df: pd.DataFrame, path: Path, dpi: int) -> None:
    blocks = _missingness_blocks_from_observed(df)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if blocks.empty:
        ax.text(0.5, 0.5, "No missingness blocks", ha="center", va="center")
    else:
        order = blocks.groupby("missing_reason")["duration_days"].median().sort_values().index.tolist()
        data = [blocks.loc[blocks["missing_reason"] == reason, "duration_days"].values for reason in order]
        ax.boxplot(data, vert=False, labels=order, showfliers=False)
        for i, values in enumerate(data, start=1):
            ax.text(max(values), i, f" n={len(values)}", va="center", fontsize=8)
    ax.set_xlabel("Missing block length, days")
    ax.set_ylabel("Missingness reason")
    ax.set_title("Distribution of missing block lengths by reason")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_missingness_patient_burden_distribution(df: pd.DataFrame, path: Path, dpi: int) -> None:
    post = df[df["day"] >= 1].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    if post.empty:
        ax.text(0.5, 0.5, "No post-treatment rows", ha="center", va="center")
    else:
        burden = 100.0 * post.groupby("patient_id")["is_missing"].mean()
        ax.hist(burden, bins=20, edgecolor="black", alpha=0.8)
        median = float(burden.median())
        ax.axvline(median, linestyle="--", linewidth=1.5)
        ax.text(median, ax.get_ylim()[1] * 0.95, f"median {median:.1f}%", ha="right", va="top", fontsize=9)
    ax.set_xlabel("Patient-level missingness burden, % of post-treatment days")
    ax.set_ylabel("Number of patients")
    ax.set_title("Distribution of patient-level missingness burden")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_diverse_patients(class_df: pd.DataFrame, max_n: int) -> List[Any]:
    """Pick deterministic examples that span low/medium/high final activity."""

    summary = (
        class_df.groupby("patient_id")
        .agg(final_steps=("true_steps", lambda x: float(pd.Series(x).tail(28).mean())))
        .sort_values("final_steps")
        .reset_index()
    )
    if len(summary) <= max_n:
        return summary["patient_id"].tolist()
    positions = np.linspace(0, len(summary) - 1, max_n).round().astype(int)
    return summary.iloc[positions]["patient_id"].tolist()


def _select_patients_for_missingness_examples(post: pd.DataFrame, max_patients: int) -> List[Any]:
    burden = post.groupby(["recovery_trajectory_class", "patient_id"])["is_missing"].mean().reset_index()
    selected: List[Any] = []
    per_class = max(1, int(np.ceil(max_patients / max(1, burden["recovery_trajectory_class"].nunique()))))
    for cls in [c for c in CLASS_ORDER if c in set(burden["recovery_trajectory_class"])] + sorted(set(burden["recovery_trajectory_class"]) - set(CLASS_ORDER)):
        sub = burden[burden["recovery_trajectory_class"] == cls].sort_values("is_missing", ascending=False)
        selected.extend(sub["patient_id"].head(per_class).tolist())
    return selected[:max_patients]


def _missingness_blocks_from_observed(df: pd.DataFrame) -> pd.DataFrame:
    missing = df[df["is_missing"]].copy()
    if missing.empty:
        return pd.DataFrame(columns=["patient_id", "missing_block_id", "missing_reason", "missing_mechanism", "start_day", "end_day", "duration_days"])

    if "missing_block_id" in missing.columns and missing["missing_block_id"].notna().any():
        blocks = (
            missing.groupby(["patient_id", "missing_block_id", "missing_reason", "missing_mechanism"])
            .agg(start_day=("day", "min"), end_day=("day", "max"), duration_days=("day", "size"))
            .reset_index()
        )
        return blocks

    # Fallback if block ids were not available: infer consecutive runs.
    rows = []
    for pid, sub in missing.sort_values(["patient_id", "day"]).groupby("patient_id"):
        run_id = (sub["day"].diff().fillna(1) != 1).cumsum()
        for _, run in sub.groupby(run_id):
            reason = run["missing_reason"].mode().iloc[0]
            mechanism = run["missing_mechanism"].mode().iloc[0]
            rows.append(
                {
                    "patient_id": pid,
                    "missing_block_id": pd.NA,
                    "missing_reason": reason,
                    "missing_mechanism": mechanism,
                    "start_day": int(run["day"].min()),
                    "end_day": int(run["day"].max()),
                    "duration_days": int(len(run)),
                }
            )
    return pd.DataFrame(rows)


def _safe_ylim_for_steps(values: Iterable[float]) -> tuple[float, float]:
    series = pd.Series(values).dropna().astype(float)
    if series.empty:
        return 0.0, 1.0
    ymax = max(1000.0, float(series.quantile(0.98)) * 1.15)
    return 0.0, ymax


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value.lower())


def _pretty_label(value: str) -> str:
    return value.replace("_", " ").title()
