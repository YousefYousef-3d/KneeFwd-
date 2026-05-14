from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    base = Path(base_dir) if base_dir is not None else project_root()
    return base / path


def load_yaml(path_like: str | Path) -> Dict[str, Any]:
    path = resolve_path(path_like)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_json(obj: Mapping[str, Any], path_like: str | Path) -> None:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def ensure_output_dirs(output_dir: str | Path) -> Dict[str, Path]:
    out = resolve_path(output_dir)
    dirs = {
        "root": out,
        "data": out / "data",
        "figures": out / "figures",
        "tables": out / "tables",
        "validation": out / "validation",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def clipped_normal(rng: np.random.Generator, mean: float, sd: float, low: float, high: float, size: int | None = None):
    values = rng.normal(mean, sd, size=size)
    return np.clip(values, low, high)


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    total = e.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(e) / len(e)
    return e / total


def get_season(date: pd.Timestamp | str) -> str:
    month = pd.Timestamp(date).month
    if month in (6, 7, 8):
        return "summer"
    if month in (12, 1, 2):
        return "winter"
    return "spring_fall"


def weekday_name(date: pd.Timestamp | str) -> str:
    return pd.Timestamp(date).day_name().lower()


def month_index_from_day(day: int) -> int:
    if day <= 0:
        return 0
    return int(math.ceil(day / 30.4375))


def weighted_choice(rng: np.random.Generator, labels: Iterable[str], probabilities: Iterable[float]) -> str:
    labels_list = list(labels)
    probs = np.asarray(list(probabilities), dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(labels_list, p=probs))


def beta01(rng: np.random.Generator, alpha: float, beta: float, size: int | None = None) -> np.ndarray:
    return rng.beta(alpha, beta, size=size)


def log_effect_from_percent(percent_change: float) -> float:
    return float(np.log1p(percent_change / 100.0))


def safe_int_steps(values: np.ndarray | pd.Series | float, min_steps: int = 0, max_steps: int = 25000):
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=max_steps, neginf=min_steps)
    arr = np.clip(arr, min_steps, max_steps)
    return np.rint(arr).astype(int)


def normalise_probabilities(probabilities: Mapping[str, float]) -> Dict[str, float]:
    total = float(sum(probabilities.values()))
    if total <= 0:
        raise ValueError("Probabilities must sum to a positive value.")
    return {k: float(v) / total for k, v in probabilities.items()}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_all_configs(default_config_path: str | Path) -> Dict[str, Any]:
    default_cfg = load_yaml(default_config_path)
    cfg_paths = default_cfg.get("config_paths", {})
    configs = {"default": default_cfg}
    for name, rel_path in cfg_paths.items():
        configs[name] = load_yaml(rel_path)
    return configs


def flatten_dict(d: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_dict(value, name))
        else:
            out[name] = value
    return out


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
