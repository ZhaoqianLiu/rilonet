from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import (
    LAG_DIRECTIONS,
    NEAR_BLOCKS,
    all_covariates,
    ensure_dir,
    load_prepared_manifest,
    output_root,
    write_json,
)


def _pairwise_delta_days(x: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
    x_date = pd.to_numeric(x["match_time_days"], errors="coerce").to_numpy(dtype=float)
    y_date = pd.to_numeric(y["match_time_days"], errors="coerce").to_numpy(dtype=float)
    delta = y_date[None, :] - x_date[:, None]
    x_study = pd.to_numeric(x["study_time_days"], errors="coerce").to_numpy(dtype=float)
    y_study = pd.to_numeric(y["study_time_days"], errors="coerce").to_numpy(dtype=float)
    fallback = y_study[None, :] - x_study[:, None]
    missing = ~np.isfinite(delta)
    delta[missing] = fallback[missing]
    return delta


def optimal_matches(
    x: pd.DataFrame,
    y: pd.DataFrame,
    min_delta_days: float,
    max_delta_days: float,
    target_delta_days: float,
    absolute_window: bool,
) -> list[tuple[int, int, float, bool]]:
    from scipy.optimize import linear_sum_assignment

    if x.empty or y.empty:
        return []
    delta = _pairwise_delta_days(x, y)
    valid = (
        np.abs(delta) <= max_delta_days
        if absolute_window
        else (delta >= min_delta_days) & (delta <= max_delta_days)
    )
    distance = np.abs(delta - target_delta_days)
    x_canonical = x["canonical_biosample_id"].fillna("").astype(str).to_numpy()
    y_canonical = y["canonical_biosample_id"].fillna("").astype(str).to_numpy()
    exact = (
        (x_canonical[:, None] != "")
        & (y_canonical[None, :] != "")
        & (x_canonical[:, None] == y_canonical[None, :])
    )
    # A large invalid-edge penalty produces maximum cardinality before distance
    # minimization; the small exact-biosample credit breaks equal-cost ties.
    invalid_cost = 1e12
    cost = np.where(valid & np.isfinite(distance), distance - exact * 1e-6, invalid_cost)
    rows, columns = linear_sum_assignment(cost)
    matches = []
    for row, column in zip(rows, columns):
        if cost[row, column] >= invalid_cost:
            continue
        matches.append((int(row), int(column), float(delta[row, column]), bool(exact[row, column])))
    return matches


def _baseline(
    targets: pd.DataFrame, source: pd.Series, lookback_days: float
) -> tuple[pd.Series | None, float]:
    candidates = targets[targets["subject_id"].eq(source["subject_id"])].copy()
    if candidates.empty:
        return None, np.nan
    source_frame = pd.DataFrame([source])
    lookback = _pairwise_delta_days(candidates, source_frame)[:, 0]
    valid = np.isfinite(lookback) & (lookback >= 0) & (lookback <= lookback_days)
    if not valid.any():
        return None, np.nan
    candidates = candidates.iloc[np.flatnonzero(valid)].copy()
    gaps = lookback[valid]
    exact = candidates["canonical_biosample_id"].fillna("").astype(str).eq(
        str(source.get("canonical_biosample_id", ""))
    )
    order = np.lexsort((gaps, ~exact.to_numpy()))
    position = int(order[0])
    return candidates.iloc[position], float(gaps[position])


def _pair_row(
    pair_id: str,
    pair_type: str,
    window_name: str,
    x_omic: str,
    y_omic: str,
    x: pd.Series,
    y: pd.Series,
    delta: float,
    exact: bool,
    covariates: list[str],
) -> dict[str, Any]:
    x_time = pd.to_numeric(x.get("study_time"), errors="coerce")
    y_time = pd.to_numeric(y.get("study_time"), errors="coerce")
    row: dict[str, Any] = {
        "pair_id": pair_id,
        "pair_type": pair_type,
        "window_name": window_name,
        "subject_id": str(x["subject_id"]),
        "group": str(x["group"]),
        "x_omic": x_omic,
        "y_omic": y_omic,
        "x_sample_id": str(x["sample_id"]),
        "y_sample_id": str(y["sample_id"]),
        "x_canonical_biosample_id": x.get("canonical_biosample_id", ""),
        "y_canonical_biosample_id": y.get("canonical_biosample_id", ""),
        "x_time": x_time,
        "y_time": y_time,
        "mid_time": float(np.nanmean([x_time, y_time])),
        "x_collection_date": x.get("collection_date", ""),
        "y_collection_date": y.get("collection_date", ""),
        "delta_days": float(delta),
        "abs_delta_days": abs(float(delta)),
        "exact_biosample": bool(exact),
    }
    for column in covariates:
        row[f"x_cov__{column}"] = x.get(column, np.nan)
        row[f"y_cov__{column}"] = y.get(column, np.nan)
    return row


def _valid_manifest(config: dict[str, Any]) -> pd.DataFrame:
    manifest = load_prepared_manifest(config)
    valid = manifest["valid_for_analysis"].astype(str).str.lower().eq("true")
    manifest = manifest[valid].copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    manifest["subject_id"] = manifest["subject_id"].astype(str)
    return manifest


def build_near_pairs(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    manifest = _valid_manifest(config)
    output_dir = ensure_dir(output_root(config) / "03_pairs" / "near")
    tolerance = float(config["matching"]["near_tolerance_days"])
    covariates = all_covariates(config)
    outputs: dict[str, pd.DataFrame] = {}
    for x_omic, y_omic in NEAR_BLOCKS:
        rows = []
        subjects = sorted(
            set(manifest.loc[manifest["omic"].eq(x_omic), "subject_id"])
            & set(manifest.loc[manifest["omic"].eq(y_omic), "subject_id"])
        )
        for subject in subjects:
            x = manifest[manifest["omic"].eq(x_omic) & manifest["subject_id"].eq(subject)].copy()
            y = manifest[manifest["omic"].eq(y_omic) & manifest["subject_id"].eq(subject)].copy()
            x = x.sort_values(["study_time", "sample_id"]).reset_index(drop=True)
            y = y.sort_values(["study_time", "sample_id"]).reset_index(drop=True)
            for index, (xi, yi, delta, exact) in enumerate(
                optimal_matches(x, y, -tolerance, tolerance, 0.0, True), 1
            ):
                rows.append(
                    _pair_row(
                        f"NEAR_{x_omic}_{y_omic}_{subject}_{index:03d}",
                        "near",
                        f"pm_{tolerance:g}d",
                        x_omic,
                        y_omic,
                        x.iloc[xi],
                        y.iloc[yi],
                        delta,
                        exact,
                        covariates,
                    )
                )
        frame = pd.DataFrame(rows)
        key = f"{x_omic}_{y_omic}"
        frame.to_csv(output_dir / f"{key}.tsv", sep="\t", index=False)
        outputs[key] = frame
    return outputs


def build_lag_pairs(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    manifest = _valid_manifest(config)
    output_dir = ensure_dir(output_root(config) / "03_pairs" / "lagged")
    cfg = config["matching"]
    lookback = float(cfg.get("baseline_lookback_days", 0))
    covariates = all_covariates(config)
    outputs: dict[str, pd.DataFrame] = {}
    for x_omic, y_omic in LAG_DIRECTIONS:
        targets = manifest[manifest["omic"].eq(y_omic)].copy()
        subjects = sorted(
            set(manifest.loc[manifest["omic"].eq(x_omic), "subject_id"])
            & set(targets["subject_id"])
        )
        for window in cfg["lag_windows"]:
            name = str(window["name"])
            rows = []
            for subject in subjects:
                x = manifest[manifest["omic"].eq(x_omic) & manifest["subject_id"].eq(subject)].copy()
                y = targets[targets["subject_id"].eq(subject)].copy()
                x = x.sort_values(["study_time", "sample_id"]).reset_index(drop=True)
                y = y.sort_values(["study_time", "sample_id"]).reset_index(drop=True)
                matches = optimal_matches(
                    x,
                    y,
                    float(window["min_days"]),
                    float(window["max_days"]),
                    float(window["target_days"]),
                    False,
                )
                for index, (xi, yi, delta, exact) in enumerate(matches, 1):
                    source = x.iloc[xi]
                    row = _pair_row(
                        f"LAG_{name}_{x_omic}_{y_omic}_{subject}_{index:03d}",
                        "lagged",
                        name,
                        x_omic,
                        y_omic,
                        source,
                        y.iloc[yi],
                        delta,
                        exact,
                        covariates,
                    )
                    baseline, gap = _baseline(targets, source, lookback) if lookback > 0 else (None, np.nan)
                    row["has_baseline"] = baseline is not None
                    row["baseline_sample_id"] = str(baseline["sample_id"]) if baseline is not None else ""
                    row["baseline_time"] = baseline.get("study_time", np.nan) if baseline is not None else np.nan
                    row["baseline_lookback_days"] = gap
                    rows.append(row)
            frame = pd.DataFrame(rows)
            key = f"{name}_{x_omic}_to_{y_omic}"
            frame.to_csv(output_dir / f"{key}.tsv", sep="\t", index=False)
            outputs[key] = frame
    return outputs


def build_all_pairs(config: dict[str, Any]) -> dict[str, Any]:
    near = build_near_pairs(config)
    lagged = build_lag_pairs(config)

    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"pairs": 0, "subjects": 0}
        result = {
            "pairs": int(len(frame)),
            "subjects": int(frame["subject_id"].nunique()),
            "pairs_by_group": {str(k): int(v) for k, v in frame["group"].value_counts().items()},
            "subjects_by_group": {
                str(k): int(v) for k, v in frame.groupby("group")["subject_id"].nunique().items()
            },
            "median_delta_days": float(frame["delta_days"].median()),
        }
        if "has_baseline" in frame:
            result["pairs_with_baseline"] = int(frame["has_baseline"].sum())
        return result

    report = {
        "near": {key: summarize(value) for key, value in near.items()},
        "lagged": {key: summarize(value) for key, value in lagged.items()},
    }
    write_json(report, output_root(config) / "03_pairs" / "pairing_qc.json")
    return report
