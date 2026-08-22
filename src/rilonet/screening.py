from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import ensure_dir, load_prepared_manifest, output_root, sample_split_enabled, write_json


CANDIDATE_COLUMNS = [
    "analysis_type",
    "block",
    "window_name",
    "x_omic",
    "y_omic",
    "x_feature",
    "y_feature",
    "screen_correlation",
    "reference_screen_correlation",
    "condition_screen_correlation",
    "screen_correlation_difference",
    "pooled_abs_screen_correlation",
    "abs_screen_correlation",
    "n_pairs",
    "n_subjects",
]


def _residuals(config: dict[str, Any], omic: str) -> pd.DataFrame:
    frame = pd.read_parquet(output_root(config) / "02_residuals" / f"{omic}_residuals.parquet")
    frame.index = frame.index.astype(str)
    return frame


def _arrays(
    pairs: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.DataFrame,
    analysis_type: str,
    baseline_mode: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = pairs.copy()
    if analysis_type == "lagged" and baseline_mode in {"adjust", "change"}:
        work = work[work["has_baseline"].astype(str).str.lower().eq("true")].copy()
    exists = work["x_sample_id"].astype(str).isin(x.index) & work["y_sample_id"].astype(str).isin(y.index)
    work = work[exists].copy()
    x_values = x.loc[work["x_sample_id"].astype(str)].to_numpy(dtype=float)
    y_values = y.loc[work["y_sample_id"].astype(str)].to_numpy(dtype=float)
    if analysis_type == "lagged" and baseline_mode == "change":
        valid = work["baseline_sample_id"].astype(str).isin(y.index)
        work = work[valid].copy()
        x_values = x_values[valid.to_numpy()]
        y_values = y_values[valid.to_numpy()]
        baseline = y.loc[work["baseline_sample_id"].astype(str)].to_numpy(dtype=float)
        y_values = y_values - baseline
    return work, x_values, y_values


def _correlation(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_x = np.isfinite(x).all(axis=0) & (np.nanstd(x, axis=0, ddof=1) > 0)
    valid_y = np.isfinite(y).all(axis=0) & (np.nanstd(y, axis=0, ddof=1) > 0)
    x = x[:, valid_x]
    y = y[:, valid_y]
    if not x.shape[1] or not y.shape[1]:
        return np.empty((x.shape[1], y.shape[1])), valid_x, valid_y
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    result = (x.T @ y) / np.outer(np.sqrt((x**2).sum(axis=0)), np.sqrt((y**2).sum(axis=0)))
    return np.clip(result, -1.0, 1.0), valid_x, valid_y


def _group_correlation(
    x: np.ndarray,
    y: np.ndarray,
    row_mask: np.ndarray,
) -> np.ndarray:
    result = np.full((x.shape[1], y.shape[1]), np.nan, dtype=float)
    if int(np.sum(row_mask)) < 3:
        return result
    local, valid_x, valid_y = _correlation(x[row_mask], y[row_mask])
    if local.size:
        result[np.ix_(valid_x, valid_y)] = local
    return result


def _screening_matrices(
    work: pd.DataFrame,
    x_values: np.ndarray,
    y_values: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pooled, valid_x, valid_y = _correlation(x_values, y_values)
    x = x_values[:, valid_x]
    y = y_values[:, valid_y]
    levels = [str(value) for value in config["metadata"]["group_levels"]]
    reference = str(config["metadata"]["reference_group"])
    condition = next(level for level in levels if level != reference)
    groups = work["group"].astype(str).to_numpy()
    reference_correlation = _group_correlation(x, y, groups == reference)
    condition_correlation = _group_correlation(x, y, groups == condition)
    difference = condition_correlation - reference_correlation
    score = np.nanmax(
        np.stack(
            [
                np.abs(pooled),
                np.abs(reference_correlation),
                np.abs(condition_correlation),
                np.abs(difference),
            ]
        ),
        axis=0,
    )
    return (
        pooled,
        reference_correlation,
        condition_correlation,
        difference,
        score,
        valid_x,
        valid_y,
    )


def _top_edges(correlation: np.ndarray, minimum: float, top_k: int, maximum: int) -> list[tuple[int, int]]:
    if not correlation.size:
        return []
    absolute = np.abs(correlation)
    selected: set[tuple[int, int]] = set()
    row_k = min(top_k, absolute.shape[1])
    for row in range(absolute.shape[0]):
        indices = np.argpartition(absolute[row], -row_k)[-row_k:] if row_k else []
        selected.update((row, int(column)) for column in indices if absolute[row, column] >= minimum)
    col_k = min(top_k, absolute.shape[0])
    for column in range(absolute.shape[1]):
        indices = np.argpartition(absolute[:, column], -col_k)[-col_k:] if col_k else []
        selected.update((int(row), column) for row in indices if absolute[row, column] >= minimum)
    return sorted(selected, key=lambda index: absolute[index], reverse=True)[:maximum]


def screen_pair_file(
    config: dict[str, Any],
    path: Path,
    analysis_type: str,
    residual_cache: dict[str, pd.DataFrame] | None = None,
    allowed_subjects: set[str] | None = None,
) -> pd.DataFrame:
    pairs = pd.read_csv(path, sep="\t", low_memory=False)
    if allowed_subjects is not None:
        pairs = pairs[pairs["subject_id"].astype(str).isin(allowed_subjects)].copy()
    if pairs.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    x_omic = str(pairs["x_omic"].iloc[0])
    y_omic = str(pairs["y_omic"].iloc[0])
    if residual_cache is None:
        x = _residuals(config, x_omic)
        y = _residuals(config, y_omic)
    else:
        if x_omic not in residual_cache:
            residual_cache[x_omic] = _residuals(config, x_omic)
        if y_omic not in residual_cache:
            residual_cache[y_omic] = _residuals(config, y_omic)
        x = residual_cache[x_omic]
        y = residual_cache[y_omic]
    cfg = config["screening"]
    work, x_values, y_values = _arrays(
        pairs, x, y, analysis_type, str(cfg.get("baseline_mode", "none"))
    )
    if len(work) < int(cfg["min_pairs"]) or work["subject_id"].nunique() < int(cfg["min_subjects"]):
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    (
        correlation,
        reference_correlation,
        condition_correlation,
        correlation_difference,
        screening_score,
        valid_x,
        valid_y,
    ) = _screening_matrices(work, x_values, y_values, config)
    x_features = x.columns[valid_x]
    y_features = y.columns[valid_y]
    selected = _top_edges(
        screening_score,
        float(cfg["min_abs_correlation"]),
        int(cfg["top_k_per_feature"]),
        int(cfg["max_edges_per_block"]),
    )
    block = f"{x_omic}_{y_omic}" if analysis_type == "near" else f"{x_omic}_to_{y_omic}"
    rows = []
    for xi, yi in selected:
        value = float(correlation[xi, yi])
        reference_value = float(reference_correlation[xi, yi])
        condition_value = float(condition_correlation[xi, yi])
        difference_value = float(correlation_difference[xi, yi])
        score = float(screening_score[xi, yi])
        rows.append(
            {
                "analysis_type": analysis_type,
                "block": block,
                "window_name": str(pairs["window_name"].iloc[0]),
                "x_omic": x_omic,
                "y_omic": y_omic,
                "x_feature": str(x_features[xi]),
                "y_feature": str(y_features[yi]),
                "screen_correlation": value,
                "reference_screen_correlation": reference_value,
                "condition_screen_correlation": condition_value,
                "screen_correlation_difference": difference_value,
                "pooled_abs_screen_correlation": abs(value),
                "abs_screen_correlation": score,
                "n_pairs": int(len(work)),
                "n_subjects": int(work["subject_id"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def _create_participant_split(config: dict[str, Any], output_dir: Path) -> set[str] | None:
    if not sample_split_enabled(config):
        return None
    cfg = config["screening"]["sample_split"]
    fraction = float(cfg.get("screening_fraction", 0.50))
    if not 0 < fraction < 1:
        raise ValueError("screening.sample_split.screening_fraction must be between 0 and 1")
    manifest = load_prepared_manifest(config)
    manifest = manifest[manifest["valid_for_analysis"].astype(str).str.lower().eq("true")]
    subject_info = manifest[["subject_id", "group"]].drop_duplicates()
    if subject_info.groupby("subject_id")["group"].nunique().max() > 1:
        raise ValueError("A participant cannot belong to multiple groups in sample splitting")
    subject_info = subject_info.drop_duplicates("subject_id").copy()
    rng = np.random.default_rng(int(cfg.get("seed", 20260721)))
    minimum_inference = int(config["modeling"].get("min_subjects_per_group", 5))
    minimum_screening = int(cfg.get("min_screening_subjects_per_group", 3))
    assignments = []
    for group, subset in subject_info.groupby("group", sort=True):
        subjects = subset["subject_id"].astype(str).to_numpy()
        rng.shuffle(subjects)
        if len(subjects) < minimum_screening + minimum_inference:
            raise ValueError(
                f"Group {group!r} has {len(subjects)} participants; sample splitting requires "
                f"at least {minimum_screening + minimum_inference}"
            )
        n_screening = int(round(len(subjects) * fraction))
        n_screening = max(minimum_screening, n_screening)
        n_screening = min(n_screening, len(subjects) - minimum_inference)
        screening = set(subjects[:n_screening])
        assignments.extend(
            {
                "subject_id": subject,
                "group": str(group),
                "role": "screening" if subject in screening else "inference",
            }
            for subject in subjects
        )
    split = pd.DataFrame(assignments).sort_values(["group", "role", "subject_id"])
    split.to_csv(output_dir / "participant_split.tsv", sep="\t", index=False)
    return set(split.loc[split["role"].eq("screening"), "subject_id"].astype(str))


def screen_all(config: dict[str, Any]) -> dict[str, Any]:
    root = output_root(config) / "03_pairs"
    output_dir = ensure_dir(output_root(config) / "04_candidates")
    screening_subjects = _create_participant_split(config, output_dir)
    report: dict[str, Any] = {"near": {}, "lagged": {}}
    residual_cache: dict[str, pd.DataFrame] = {}
    for analysis_type, folder in (("near", "near"), ("lagged", "lagged")):
        for path in sorted((root / folder).glob("*.tsv")):
            candidates = screen_pair_file(
                config,
                path,
                analysis_type,
                residual_cache=residual_cache,
                allowed_subjects=screening_subjects,
            )
            output = output_dir / f"{analysis_type}_{path.stem}.tsv"
            candidates.to_csv(output, sep="\t", index=False)
            report[analysis_type][path.stem] = {
                "candidate_edges": int(len(candidates)),
                "output": str(output),
            }
    write_json(report, output_dir / "screening_qc.json")
    return report
