from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


OMICS = ("MG", "MT", "MB")
NEAR_BLOCKS = (("MG", "MT"), ("MG", "MB"), ("MT", "MB"))
LAG_DIRECTIONS = tuple((x, y) for x in OMICS for y in OMICS if x != y)


_THREADPOOL_LIMITER: Any = None


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    root_value = Path(config.get("project_root", "."))
    if not root_value.is_absolute():
        root_value = (config_path.parent / root_value).resolve()
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(root_value)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing_omics = set(OMICS) - set(config.get("omics", {}))
    if missing_omics:
        raise ValueError(f"Configuration is missing omics: {sorted(missing_omics)}")
    metadata = config.get("metadata", {})
    columns = metadata.get("columns", {})
    required = {"sample_id", "participant_id", "omic", "group", "study_time"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"metadata.columns is missing: {sorted(missing)}")
    levels = [str(value) for value in metadata.get("group_levels", [])]
    if len(levels) != 2 or len(set(levels)) != 2:
        raise ValueError("RILoNet currently requires exactly two group_levels")
    reference = str(metadata.get("reference_group", ""))
    if reference not in levels:
        raise ValueError("metadata.reference_group must be one of metadata.group_levels")
    mode = str(config.get("modeling", {}).get("baseline_mode", "none"))
    if mode not in {"none", "adjust", "change"}:
        raise ValueError("modeling.baseline_mode must be none, adjust, or change")
    compute = config.get("compute", {})
    for key in ("workers", "blas_threads", "tasks_per_worker"):
        if key in compute and int(compute[key]) < 1:
            raise ValueError(f"compute.{key} must be at least 1")
    start_method = str(compute.get("start_method", "spawn"))
    if start_method not in {"spawn", "forkserver", "fork"}:
        raise ValueError("compute.start_method must be spawn, forkserver, or fork")


def configured_workers(
    config: dict[str, Any], override: int | None = None, task_count: int | None = None
) -> int:
    """Return the worker count, bounded by the number of tasks when provided."""
    workers = int(override if override is not None else config.get("compute", {}).get("workers", 1))
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if task_count is not None:
        workers = min(workers, max(1, int(task_count)))
    return workers


def worker_chunksize(config: dict[str, Any], task_count: int, workers: int) -> int:
    tasks_per_worker = int(config.get("compute", {}).get("tasks_per_worker", 4))
    return max(1, math.ceil(task_count / max(1, workers * tasks_per_worker)))


def configure_numeric_threads(threads: int) -> None:
    """Limit native BLAS/OpenMP pools inside a serial run or process worker."""
    global _THREADPOOL_LIMITER
    threads = int(threads)
    if threads < 1:
        raise ValueError("blas_threads must be at least 1")
    value = str(threads)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = value
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    _THREADPOOL_LIMITER = threadpool_limits(limits=threads)


def configure_worker(config: dict[str, Any]) -> None:
    configure_numeric_threads(int(config.get("compute", {}).get("blas_threads", 1)))


def process_context(config: dict[str, Any]) -> Any:
    method = str(config.get("compute", {}).get("start_method", "spawn"))
    if method not in mp.get_all_start_methods():
        raise ValueError(f"Process start method {method!r} is unavailable on this platform")
    return mp.get_context(method)


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def output_root(config: dict[str, Any]) -> Path:
    return resolve_path(config, config["paths"]["output_dir"])


def sample_split_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("screening", {}).get("sample_split", {}).get("enabled", False))


def split_subjects(config: dict[str, Any], role: str) -> set[str] | None:
    """Return subjects assigned to screening or inference, or None when disabled."""
    if not sample_split_enabled(config):
        return None
    if role not in {"screening", "inference"}:
        raise ValueError("role must be screening or inference")
    path = output_root(config) / "04_candidates" / "participant_split.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"Participant split is required but missing: {path}. Run candidate screening first."
        )
    frame = pd.read_csv(path, sep="\t", dtype=str)
    return set(frame.loc[frame["role"].eq(role), "subject_id"].astype(str))


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def write_json(value: Any, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def read_matrix(path: str | Path, orientation: str, sep: str = "\t") -> pd.DataFrame:
    """Read an abundance table and return profiles x features."""
    frame = pd.read_csv(path, sep=sep, index_col=0, low_memory=False)
    if orientation == "features_by_profiles":
        frame = frame.T
    elif orientation != "profiles_by_features":
        raise ValueError("orientation must be profiles_by_features or features_by_profiles")
    if not frame.index.is_unique:
        raise ValueError(f"Duplicate profile identifiers in {path}")
    if not frame.columns.is_unique:
        raise ValueError(f"Duplicate feature identifiers in {path}")
    frame.index = frame.index.astype(str)
    return frame.apply(pd.to_numeric, errors="coerce")


def load_prepared_manifest(config: dict[str, Any]) -> pd.DataFrame:
    path = output_root(config) / "01_prepared" / "sample_manifest.tsv"
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    return frame


def load_prepared_matrix(config: dict[str, Any], omic: str) -> pd.DataFrame:
    frame = pd.read_parquet(output_root(config) / "01_prepared" / "matrices" / f"{omic}.parquet")
    frame.index = frame.index.astype(str)
    return frame


def safe_standardize(values: Iterable[float]) -> tuple[np.ndarray, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.nanmean(array))
    sd = float(np.nanstd(array, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.full(array.shape, np.nan), mean, sd
    return (array - mean) / sd, mean, sd


def bh_fdr(values: Iterable[float]) -> np.ndarray:
    pvalues = np.asarray(list(values), dtype=float)
    result = np.full(pvalues.shape, np.nan)
    valid = np.isfinite(pvalues)
    p = pvalues[valid]
    if not p.size:
        return result
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    reverse = np.empty_like(order)
    reverse[order] = np.arange(len(order))
    result[valid] = adjusted[reverse]
    return result


def normal_two_sided_p(z_value: float) -> float:
    if not math.isfinite(z_value):
        return math.nan
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def categorical_covariates(config: dict[str, Any]) -> list[str]:
    return list(config.get("covariates", {}).get("categorical", []))


def numeric_covariates(config: dict[str, Any]) -> list[str]:
    return list(config.get("covariates", {}).get("numeric", []))


def all_covariates(config: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(categorical_covariates(config) + numeric_covariates(config)))
