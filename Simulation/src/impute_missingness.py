from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def impute_missingness(observed_df: pd.DataFrame, method: str, cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    df = observed_df.copy()
    df["imputed_steps"] = df["observed_steps"].copy()
    df["was_imputed"] = False
    df["imputation_method"] = method

    if method == "no_imputation":
        return df
    if method == "forward_fill_with_limit":
        return _forward_fill_with_limit(df, limit=7)
    if method == "patient_mean_imputation":
        return _patient_mean(df)
    if method == "group_day_mean_imputation":
        return _group_day_mean(df)
    if method == "linear_interpolation":
        return _linear_interpolation(df)
    if method == "simple_model_based_imputation":
        return _simple_model_based(df)
    raise ValueError(f"Unknown imputation method: {method}")


def _mark_and_clip(df: pd.DataFrame) -> pd.DataFrame:
    missing_mask = df["is_missing"].astype(bool)
    df.loc[missing_mask, "was_imputed"] = df.loc[missing_mask, "imputed_steps"].notna()
    df["imputed_steps"] = df["imputed_steps"].clip(lower=0, upper=25000).round()
    return df


def _forward_fill_with_limit(df: pd.DataFrame, limit: int = 7) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "day"]).copy()
    df["imputed_steps"] = df.groupby("patient_id")["observed_steps"].ffill(limit=limit)
    fallback = _group_day_values(df)
    df["imputed_steps"] = df["imputed_steps"].fillna(fallback)
    return _mark_and_clip(df)


def _patient_mean(df: pd.DataFrame) -> pd.DataFrame:
    means = df.groupby("patient_id")["observed_steps"].transform("mean")
    fallback = df["observed_steps"].mean()
    df.loc[df["is_missing"], "imputed_steps"] = means.fillna(fallback)
    return _mark_and_clip(df)


def _group_day_mean(df: pd.DataFrame) -> pd.DataFrame:
    fallback = _group_day_values(df)
    df.loc[df["is_missing"], "imputed_steps"] = fallback[df["is_missing"]]
    return _mark_and_clip(df)


def _linear_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "day"]).copy()
    interpolated = df.groupby("patient_id", group_keys=False)["observed_steps"].apply(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    df["imputed_steps"] = interpolated
    fallback = _group_day_values(df)
    df["imputed_steps"] = df["imputed_steps"].fillna(fallback)
    return _mark_and_clip(df)


def _group_day_values(df: pd.DataFrame) -> pd.Series:
    group_day = df.groupby(["recovery_trajectory_class", "day"])["observed_steps"].transform("mean")
    day_mean = df.groupby("day")["observed_steps"].transform("mean")
    patient_mean = df.groupby("patient_id")["observed_steps"].transform("mean")
    global_mean = df["observed_steps"].mean()
    return group_day.fillna(day_mean).fillna(patient_mean).fillna(global_mean)


def _simple_model_based(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception:
        return _linear_interpolation(df)

    features = [
        "day", "month", "age", "bmi", "baseline_pain_nrs", "baseline_function_score",
        "comorbidity_burden", "pre_treatment_baseline_steps", "digital_confidence_score",
        "adherence_tendency", "treatment_effectiveness_score", "recovery_sensitivity_score",
        "variability_tendency", "setback_tendency", "missingness_tendency",
        "recovery_trajectory_class", "season", "weekday", "sex", "osteoarthritis_severity",
    ]
    train = df[df["observed_steps"].notna()].copy()
    missing = df[df["observed_steps"].isna()].copy()
    if len(train) < 50 or len(missing) == 0:
        return _linear_interpolation(df)

    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in features if c not in numeric_features]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )
    model = Pipeline([("preprocess", pre), ("ridge", Ridge(alpha=5.0))])
    model.fit(train[features], np.log1p(train["observed_steps"]))
    preds = np.expm1(model.predict(missing[features]))
    df.loc[missing.index, "imputed_steps"] = preds
    return _mark_and_clip(df)


def imputation_summary(imputed_df: pd.DataFrame) -> pd.DataFrame:
    missing = imputed_df["is_missing"].astype(bool)
    if missing.sum() == 0:
        mae = np.nan
        rmse = np.nan
    else:
        err = imputed_df.loc[missing & imputed_df["imputed_steps"].notna(), "imputed_steps"] - imputed_df.loc[missing & imputed_df["imputed_steps"].notna(), "true_steps"]
        mae = float(err.abs().mean()) if len(err) else np.nan
        rmse = float(np.sqrt((err**2).mean())) if len(err) else np.nan
    return pd.DataFrame([
        {
            "imputation_method": imputed_df["imputation_method"].iloc[0] if len(imputed_df) else "unknown",
            "n_rows": int(len(imputed_df)),
            "n_missing_rows": int(missing.sum()),
            "n_imputed_rows": int(imputed_df["was_imputed"].sum()),
            "imputation_mae_against_true_steps": mae,
            "imputation_rmse_against_true_steps": rmse,
        }
    ])
