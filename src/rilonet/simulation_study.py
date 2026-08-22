from __future__ import annotations

import json
import re
import resource
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .common import (
    LAG_DIRECTIONS,
    NEAR_BLOCKS,
    load_config,
    output_root,
)
from .matching import _pairwise_delta_days, build_all_pairs
from .modeling import model_all
from .prepare import prepare_inputs
from .residualize import residualize_all
from .screening import screen_all
from .simulate import generate_simulation_dataset


def _edge_key(
    analysis_type: str,
    x_omic: str,
    x_feature: str,
    y_omic: str,
    y_feature: str,
) -> tuple[str, str, str, str, str]:
    return (
        str(analysis_type),
        str(x_omic),
        str(x_feature),
        str(y_omic),
        str(y_feature),
    )


def _truth_keys(truth: pd.DataFrame, differential_only: bool) -> set[tuple[str, str, str, str, str]]:
    subset = truth.copy()
    if differential_only:
        subset = subset[subset["is_differential"].astype(str).str.lower().eq("true")]
    return {
        _edge_key(
            row.analysis_type,
            row.source_omic,
            row.source_feature,
            row.target_omic,
            row.target_feature,
        )
        for row in subset.itertuples(index=False)
    }


def _candidate_keys(candidate_dir: Path) -> set[tuple[str, str, str, str, str]]:
    result: set[tuple[str, str, str, str, str]] = set()
    for path in candidate_dir.glob("*.tsv"):
        try:
            frame = pd.read_csv(path, sep="\t", low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        required = {"analysis_type", "x_omic", "x_feature", "y_omic", "y_feature"}
        if not required.issubset(frame.columns):
            continue
        for row in frame.itertuples(index=False):
            result.add(
                _edge_key(
                    row.analysis_type,
                    row.x_omic,
                    row.x_feature,
                    row.y_omic,
                    row.y_feature,
                )
            )
    return result


def _called_keys(
    network_dir: Path,
    condition_group: str,
    q_threshold: float,
) -> set[tuple[str, str, str, str, str]]:
    result: set[tuple[str, str, str, str, str]] = set()
    for path in network_dir.glob("*_edges.tsv"):
        try:
            frame = pd.read_csv(path, sep="\t", low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frame = frame[frame["group"].astype(str).eq(condition_group)].copy()
        frame = frame[pd.to_numeric(frame["contrast_q"], errors="coerce") <= q_threshold]
        for row in frame.itertuples(index=False):
            result.add(
                _edge_key(
                    row.analysis_type,
                    row.x_omic,
                    row.x_feature,
                    row.y_omic,
                    row.y_feature,
                )
            )
    return result


def _network_metrics(
    truth_differential: set[tuple[str, str, str, str, str]],
    candidates: set[tuple[str, str, str, str, str]],
    called: set[tuple[str, str, str, str, str]],
    analysis_type: str | None,
    planted_truth_differential: set[tuple[str, str, str, str, str]] | None = None,
) -> dict[str, Any]:
    planted = (
        set(truth_differential)
        if planted_truth_differential is None
        else set(planted_truth_differential)
    )
    if analysis_type is not None:
        planted = {edge for edge in planted if edge[0] == analysis_type}
        truth_differential = {edge for edge in truth_differential if edge[0] == analysis_type}
        candidates = {edge for edge in candidates if edge[0] == analysis_type}
        called = {edge for edge in called if edge[0] == analysis_type}
    screened_planted = truth_differential & candidates
    recovered_planted = truth_differential & called
    non_planted_calls = called - truth_differential
    missed_planted = truth_differential - called
    return {
        "analysis_subset": analysis_type or "all",
        "planted_differential_edges": len(planted),
        "testable_differential_edges": len(truth_differential),
        "feature_retention_recall": len(truth_differential) / len(planted)
        if planted
        else np.nan,
        "true_differential_edges": len(truth_differential),
        "candidate_edges": len(candidates),
        "called_edges": len(called),
        "screened_planted_edges": len(screened_planted),
        "candidate_planted_recall": len(screened_planted) / len(truth_differential)
        if truth_differential
        else np.nan,
        "recovered_planted_edges": len(recovered_planted),
        "non_planted_calls": len(non_planted_calls),
        "missed_planted_edges": len(missed_planted),
        "planted_edge_recall": len(recovered_planted) / len(truth_differential)
        if truth_differential
        else np.nan,
        "planted_edge_precision": len(recovered_planted) / len(called)
        if called
        else np.nan,
        "non_planted_call_fraction": len(non_planted_calls) / len(called)
        if called
        else 0.0,
        "any_non_planted_call": bool(non_planted_calls),
    }


def _visit_number(value: Any) -> int | None:
    match = re.search(r"_V(\d+)", str(value))
    return int(match.group(1)) if match else None


def _pair_correct(row: pd.Series, analysis_type: str, lag_visits: int = 3) -> bool:
    if analysis_type == "near":
        return str(row["x_canonical_biosample_id"]) == str(row["y_canonical_biosample_id"])
    x_visit = _visit_number(row["x_canonical_biosample_id"])
    y_visit = _visit_number(row["y_canonical_biosample_id"])
    return x_visit is not None and y_visit is not None and y_visit - x_visit == lag_visits


def _optimal_matching_metrics(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    root = output_root(config) / "03_pairs"
    for analysis_type, folder in (("near", "near"), ("lagged", "lagged")):
        for path in sorted((root / folder).glob("*.tsv")):
            frame = pd.read_csv(path, sep="\t", low_memory=False)
            correct = (
                frame.apply(lambda row: _pair_correct(row, analysis_type), axis=1)
                if not frame.empty
                else pd.Series(dtype=bool)
            )
            target_reuse = int(frame["y_sample_id"].duplicated().sum()) if not frame.empty else 0
            rows.append(
                {
                    "algorithm": "optimal_one_to_one",
                    "analysis_type": analysis_type,
                    "block": path.stem,
                    "pairs": int(len(frame)),
                    "correct_pairs": int(correct.sum()) if len(correct) else 0,
                    "correct_pair_fraction": float(correct.mean()) if len(correct) else np.nan,
                    "reused_target_assignments": target_reuse,
                    "median_abs_time_error_days": float(frame["abs_delta_days"].median())
                    if not frame.empty and analysis_type == "near"
                    else float(np.median(np.abs(frame["delta_days"] - 42.0)))
                    if not frame.empty
                    else np.nan,
                }
            )
    return rows


def _nearest_reuse_block(
    manifest: pd.DataFrame,
    x_omic: str,
    y_omic: str,
    analysis_type: str,
    min_days: float,
    max_days: float,
    target_days: float,
) -> pd.DataFrame:
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
        if x.empty or y.empty:
            continue
        delta = _pairwise_delta_days(x, y)
        valid = np.abs(delta) <= max_days if analysis_type == "near" else (
            (delta >= min_days) & (delta <= max_days)
        )
        cost = np.abs(delta - target_days)
        for source_index in range(len(x)):
            eligible = np.flatnonzero(valid[source_index] & np.isfinite(cost[source_index]))
            if not len(eligible):
                continue
            local_cost = cost[source_index, eligible]
            best_cost = np.min(local_cost)
            tied = eligible[np.isclose(local_cost, best_cost)]
            if len(tied) > 1:
                source_canonical = str(x.iloc[source_index]["canonical_biosample_id"])
                exact = [
                    target_index
                    for target_index in tied
                    if str(y.iloc[target_index]["canonical_biosample_id"]) == source_canonical
                ]
                target_index = int(exact[0] if exact else tied[0])
            else:
                target_index = int(tied[0])
            rows.append(
                {
                    "subject_id": subject,
                    "x_sample_id": x.iloc[source_index]["sample_id"],
                    "y_sample_id": y.iloc[target_index]["sample_id"],
                    "x_canonical_biosample_id": x.iloc[source_index]["canonical_biosample_id"],
                    "y_canonical_biosample_id": y.iloc[target_index]["canonical_biosample_id"],
                    "delta_days": float(delta[source_index, target_index]),
                    "abs_delta_days": abs(float(delta[source_index, target_index])),
                }
            )
    return pd.DataFrame(rows)


def _nearest_reuse_matching_metrics(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = output_root(config)
    manifest = pd.read_csv(root / "01_prepared" / "sample_manifest.tsv", sep="\t", low_memory=False)
    manifest = manifest[manifest["valid_for_analysis"].astype(str).str.lower().eq("true")].copy()
    manifest["subject_id"] = manifest["subject_id"].astype(str)
    rows = []
    tolerance = float(config["matching"]["near_tolerance_days"])
    for x_omic, y_omic in NEAR_BLOCKS:
        frame = _nearest_reuse_block(
            manifest, x_omic, y_omic, "near", -tolerance, tolerance, 0.0
        )
        correct = frame.apply(lambda row: _pair_correct(row, "near"), axis=1) if not frame.empty else []
        rows.append(
            {
                "algorithm": "nearest_with_reuse",
                "analysis_type": "near",
                "block": f"{x_omic}_{y_omic}",
                "pairs": int(len(frame)),
                "correct_pairs": int(np.sum(correct)) if len(correct) else 0,
                "correct_pair_fraction": float(np.mean(correct)) if len(correct) else np.nan,
                "reused_target_assignments": int(frame["y_sample_id"].duplicated().sum())
                if not frame.empty
                else 0,
                "median_abs_time_error_days": float(frame["abs_delta_days"].median())
                if not frame.empty
                else np.nan,
            }
        )
    for window in config["matching"]["lag_windows"]:
        for x_omic, y_omic in LAG_DIRECTIONS:
            frame = _nearest_reuse_block(
                manifest,
                x_omic,
                y_omic,
                "lagged",
                float(window["min_days"]),
                float(window["max_days"]),
                float(window["target_days"]),
            )
            correct = frame.apply(lambda row: _pair_correct(row, "lagged"), axis=1) if not frame.empty else []
            rows.append(
                {
                    "algorithm": "nearest_with_reuse",
                    "analysis_type": "lagged",
                    "block": f"{window['name']}_{x_omic}_to_{y_omic}",
                    "pairs": int(len(frame)),
                    "correct_pairs": int(np.sum(correct)) if len(correct) else 0,
                    "correct_pair_fraction": float(np.mean(correct)) if len(correct) else np.nan,
                    "reused_target_assignments": int(frame["y_sample_id"].duplicated().sum())
                    if not frame.empty
                    else 0,
                    "median_abs_time_error_days": float(
                        np.median(np.abs(frame["delta_days"] - float(window["target_days"])))
                    )
                    if not frame.empty
                    else np.nan,
                }
            )
    return rows


def _run_one_end_to_end(
    replicate_root: Path,
    seed: int,
    n_subjects: int,
    n_visits: int,
    condition_fraction: float,
    dimensions: dict[str, int],
    differential_truth: bool,
    screening_threshold: float,
    max_candidates_per_block: int,
    q_threshold: float,
    honest_sample_split: bool,
    screening_fraction: float,
    multiple_testing_scope: str,
    baseline_mode: str,
    near_association_method: str,
    signal_scale: float,
    innovation_sd: float,
    observation_probabilities: dict[str, float],
    metabolite_noise_sd: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stage_times: dict[str, float] = {}
    started_total = time.perf_counter()
    generated = generate_simulation_dataset(
        replicate_root,
        seed=seed,
        n_subjects=n_subjects,
        n_visits=n_visits,
        condition_fraction=condition_fraction,
        dimensions=dimensions,
        differential_truth=differential_truth,
        signal_scale=signal_scale,
        innovation_sd=innovation_sd,
        observation_probabilities=observation_probabilities,
        metabolite_noise_sd=metabolite_noise_sd,
    )
    config = load_config(generated["config"])
    config["screening"]["min_abs_correlation"] = screening_threshold
    config["screening"]["max_edges_per_block"] = max_candidates_per_block
    config["screening"]["top_k_per_feature"] = min(
        int(config["screening"]["top_k_per_feature"]), max(dimensions.values())
    )
    config["screening"]["baseline_mode"] = baseline_mode
    config["modeling"]["baseline_mode"] = baseline_mode
    config["modeling"]["multiple_testing_scope"] = multiple_testing_scope
    config["modeling"]["near_association_method"] = near_association_method
    config["screening"]["sample_split"] = {
        "enabled": honest_sample_split,
        "screening_fraction": screening_fraction,
        "min_screening_subjects_per_group": 3,
        "seed": seed + 17,
    }

    start = time.perf_counter()
    prepare_inputs(config)
    stage_times["prepare_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    residualize_all(config)
    stage_times["residualize_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    build_all_pairs(config)
    stage_times["matching_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    screen_all(config)
    stage_times["screening_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    model_all(config)
    stage_times["modeling_seconds"] = time.perf_counter() - start

    truth = pd.read_csv(generated["truth"], sep="\t", low_memory=False)
    planted_truth_differential = _truth_keys(truth, differential_only=True)
    output = output_root(config)
    retained_features = {
        omic: set(pd.read_parquet(output / "01_prepared" / "matrices" / f"{omic}.parquet").columns)
        for omic in ("MG", "MT", "MB")
    }
    truth_differential = {
        edge
        for edge in planted_truth_differential
        if edge[2] in retained_features[edge[1]] and edge[4] in retained_features[edge[3]]
    }
    candidates = _candidate_keys(output / "04_candidates")
    network_dir = (
        output / "05_networks"
        if baseline_mode == "none"
        else output / f"05_networks_baseline_{baseline_mode}"
    )
    called = _called_keys(network_dir, "condition", q_threshold)
    network_rows = []
    truth_label = "differential" if differential_truth else "global_null"
    for subset in (None, "near", "lagged"):
        network_rows.append(
            {
                "truth_mode": truth_label,
                **_network_metrics(
                    truth_differential,
                    candidates,
                    called,
                    subset,
                    planted_truth_differential=planted_truth_differential,
                ),
            }
        )
    matching_rows = _optimal_matching_metrics(config) + _nearest_reuse_matching_metrics(config)
    stage_times["total_seconds"] = time.perf_counter() - started_total
    stage_times["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    stage_times["candidate_edges"] = len(candidates)
    stage_times["called_edges"] = len(called)
    stage_times["planted_differential_edges"] = len(planted_truth_differential)
    stage_times["testable_differential_edges"] = len(truth_differential)
    return network_rows, matching_rows, stage_times


def run_end_to_end_study(
    output_dir: str | Path,
    replicates: int,
    seed: int,
    n_subjects: int = 60,
    n_visits: int = 12,
    condition_fraction: float = 0.65,
    dimensions: dict[str, int] | None = None,
    truth_modes: Iterable[str] = ("null", "differential"),
    screening_threshold: float = 0.10,
    max_candidates_per_block: int = 30,
    q_threshold: float = 0.05,
    retain_replicates: bool = False,
    inference_strategies: Iterable[str] = ("reuse",),
    screening_fraction: float = 0.50,
    multiple_testing_scope: str = "block",
    baseline_mode: str = "none",
    near_association_method: str = "max_p",
    signal_scale: float = 1.0,
    innovation_sd: float = 0.70,
    observation_probabilities: dict[str, float] | None = None,
    metabolite_noise_sd: float = 0.12,
) -> dict[str, Path]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dimensions = dimensions or {"MG": 10, "MT": 12, "MB": 8}
    # Normalize YAML null values to the null-network scenario name.
    modes = ["null" if value is None else str(value) for value in truth_modes]
    invalid = set(modes) - {"null", "differential"}
    if invalid:
        raise ValueError(f"Unknown truth modes: {sorted(invalid)}")
    strategies = [str(value) for value in inference_strategies]
    invalid_strategies = set(strategies) - {"reuse", "sample_split"}
    if invalid_strategies:
        raise ValueError(f"Unknown inference strategies: {sorted(invalid_strategies)}")
    if not 0 < screening_fraction < 1:
        raise ValueError("screening_fraction must be between 0 and 1")
    if multiple_testing_scope not in {"block", "global"}:
        raise ValueError("multiple_testing_scope must be block or global")
    if baseline_mode not in {"none", "adjust", "change"}:
        raise ValueError("baseline_mode must be none, adjust, or change")
    if near_association_method not in {"max_p", "symmetric_fisher_z"}:
        raise ValueError("near_association_method must be max_p or symmetric_fisher_z")
    signal_scale = float(signal_scale)
    innovation_sd = float(innovation_sd)
    metabolite_noise_sd = float(metabolite_noise_sd)
    observation_probabilities = {
        "MG": 0.90,
        "MT": 0.68,
        "MB": 0.55,
        **dict(observation_probabilities or {}),
    }
    seeds = np.random.SeedSequence(seed).spawn(len(modes) * replicates)
    network_rows = []
    matching_rows = []
    performance_rows = []
    for mode_index, mode in enumerate(modes):
        truth_label = "global_null" if mode == "null" else mode
        for strategy in strategies:
            for replicate in range(replicates):
                seed_index = mode_index * replicates + replicate
                replicate_seed = int(seeds[seed_index].generate_state(1)[0])
                if retain_replicates:
                    replicate_root = (
                        output / "replicates" / f"{mode}_{strategy}_{replicate:04d}"
                    )
                    if replicate_root.exists():
                        raise FileExistsError(
                            f"Replicate directory already exists: {replicate_root}"
                        )
                    replicate_root.mkdir(parents=True)
                    manager = None
                else:
                    manager = tempfile.TemporaryDirectory(
                        prefix=f"rilonet_{mode}_{strategy}_", dir="/tmp"
                    )
                    replicate_root = Path(manager.name)
                status = "ok"
                error = ""
                try:
                    networks, matching, performance = _run_one_end_to_end(
                        replicate_root,
                        replicate_seed,
                        n_subjects,
                        n_visits,
                        condition_fraction,
                        dimensions,
                        mode == "differential",
                        screening_threshold,
                        max_candidates_per_block,
                        q_threshold,
                        strategy == "sample_split",
                        screening_fraction,
                        multiple_testing_scope,
                        baseline_mode,
                        near_association_method,
                        signal_scale,
                        innovation_sd,
                        observation_probabilities,
                        metabolite_noise_sd,
                    )
                    for row in networks:
                        network_rows.append(
                            {
                                "truth_mode": truth_label,
                                "inference_strategy": strategy,
                                "replicate": replicate,
                                **row,
                            }
                        )
                    for row in matching:
                        matching_rows.append(
                            {
                                "truth_mode": truth_label,
                                "inference_strategy": strategy,
                                "replicate": replicate,
                                **row,
                            }
                        )
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}:{exc}"
                    performance = {}
                performance_rows.append(
                    {
                        "truth_mode": truth_label,
                        "inference_strategy": strategy,
                        "replicate": replicate,
                        "seed": replicate_seed,
                        "status": status,
                        "error": error,
                        **performance,
                    }
                )
                if manager is not None:
                    manager.cleanup()
                print(
                    f"[end-to-end] {mode}/{strategy} replicate "
                    f"{replicate + 1}/{replicates}: {status}",
                    flush=True,
                )

    network = pd.DataFrame(network_rows)
    matching = pd.DataFrame(matching_rows)
    performance = pd.DataFrame(performance_rows)
    network_path = output / "end_to_end_network_metrics.tsv"
    matching_path = output / "end_to_end_matching_metrics.tsv"
    performance_path = output / "end_to_end_performance.tsv"
    network.to_csv(network_path, sep="\t", index=False)
    matching.to_csv(matching_path, sep="\t", index=False)
    performance.to_csv(performance_path, sep="\t", index=False)

    if not network.empty:
        summary = (
            network.groupby(
                ["truth_mode", "inference_strategy", "analysis_subset"],
                as_index=False,
            )
            .agg(
                replicates=("replicate", "nunique"),
                mean_candidate_edges=("candidate_edges", "mean"),
                mean_called_edges=("called_edges", "mean"),
                mean_feature_retention_recall=("feature_retention_recall", "mean"),
                mean_candidate_planted_recall=("candidate_planted_recall", "mean"),
                mean_planted_edge_recall=("planted_edge_recall", "mean"),
                mean_planted_edge_precision=("planted_edge_precision", "mean"),
                mean_non_planted_call_fraction=("non_planted_call_fraction", "mean"),
                probability_any_non_planted_call=("any_non_planted_call", "mean"),
            )
        )
    else:
        summary = pd.DataFrame()
    summary_path = output / "end_to_end_network_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    run_config_path = output / "end_to_end_run.yaml"
    run_config_path.write_text(
        yaml.safe_dump(
            {
                "replicates_per_truth_mode_and_strategy": replicates,
                "truth_modes": modes,
                "inference_strategies": strategies,
                "seed": seed,
                "n_subjects": n_subjects,
                "n_visits": n_visits,
                "condition_fraction": condition_fraction,
                "dimensions": dimensions,
                "screening_threshold": screening_threshold,
                "max_candidates_per_block": max_candidates_per_block,
                "q_threshold": q_threshold,
                "multiple_testing_scope": multiple_testing_scope,
                "baseline_mode": baseline_mode,
                "near_association_method": near_association_method,
                "signal_scale": signal_scale,
                "innovation_sd": innovation_sd,
                "observation_probabilities": observation_probabilities,
                "metabolite_noise_sd": metabolite_noise_sd,
                "retain_replicates": retain_replicates,
                "screening_fraction": screening_fraction,
                "network_truth_definition": "planted_direct_structural_edges",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "network_metrics": network_path,
        "network_summary": summary_path,
        "matching_metrics": matching_path,
        "performance": performance_path,
        "configuration": run_config_path,
    }


def _positive_int(section: dict[str, Any], key: str, default: int) -> int:
    value = int(section.get(key, default))
    if value < 1:
        raise ValueError(f"{key} must be at least 1")
    return value


def run_simulation_study(study_config_path: str | Path) -> dict[str, Path]:
    """Run the configured end-to-end simulation and write a manifest."""
    config_path = Path(study_config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        study = yaml.safe_load(handle)
    if not isinstance(study, dict):
        raise ValueError("Simulation-study configuration must be a YAML mapping")
    output_value = Path(study.get("output_dir", "../simulation_results/study"))
    output = output_value if output_value.is_absolute() else config_path.parent / output_value
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = int(study.get("seed", 20260721))
    section_seeds = np.random.SeedSequence(seed).spawn(4)
    derived_seeds = [int(value.generate_state(1)[0]) for value in section_seeds]
    outputs: dict[str, Path] = {}
    timings: dict[str, float] = {}
    started_study = time.perf_counter()

    end_to_end = dict(study.get("end_to_end", {}))
    if bool(end_to_end.get("enabled", True)):
        split_residualization_scope = str(
            end_to_end.get("split_residualization_scope", "all_subjects")
        )
        if split_residualization_scope not in {"all_subjects", "screening_subjects"}:
            raise ValueError(
                "end_to_end.split_residualization_scope must be all_subjects "
                "or screening_subjects"
            )
        if split_residualization_scope == "screening_subjects":
            raise NotImplementedError(
                "end_to_end.split_residualization_scope=screening_subjects is not supported"
            )
        started = time.perf_counter()
        result = run_end_to_end_study(
            output / "04_end_to_end",
            replicates=_positive_int(end_to_end, "replicates", 200),
            seed=derived_seeds[3],
            n_subjects=_positive_int(end_to_end, "n_subjects", 60),
            n_visits=_positive_int(end_to_end, "visits", 12),
            condition_fraction=float(end_to_end.get("condition_fraction", 0.65)),
            dimensions={
                omic: int(value)
                for omic, value in dict(
                    end_to_end.get("dimensions", {"MG": 10, "MT": 12, "MB": 8})
                ).items()
            },
            truth_modes=end_to_end.get("truth_modes", ["null", "differential"]),
            screening_threshold=float(end_to_end.get("screening_threshold", 0.10)),
            max_candidates_per_block=_positive_int(
                end_to_end, "max_candidates_per_block", 30
            ),
            q_threshold=float(end_to_end.get("q_threshold", 0.05)),
            retain_replicates=bool(end_to_end.get("retain_replicates", False)),
            inference_strategies=end_to_end.get("inference_strategies", ["reuse"]),
            screening_fraction=float(end_to_end.get("screening_fraction", 0.50)),
            multiple_testing_scope=str(
                end_to_end.get("multiple_testing_scope", "block")
            ),
            baseline_mode=str(end_to_end.get("baseline_mode", "none")),
            near_association_method=str(
                end_to_end.get("near_association_method", "max_p")
            ),
            signal_scale=float(end_to_end.get("signal_scale", 1.0)),
            innovation_sd=float(end_to_end.get("innovation_sd", 0.70)),
            observation_probabilities={
                omic: float(value)
                for omic, value in dict(
                    end_to_end.get(
                        "observation_probabilities",
                        {"MG": 0.90, "MT": 0.68, "MB": 0.55},
                    )
                ).items()
            },
            metabolite_noise_sd=float(
                end_to_end.get("metabolite_noise_sd", 0.12)
            ),
        )
        outputs.update({f"end_to_end_{key}": value for key, value in result.items()})
        timings["end_to_end_seconds"] = time.perf_counter() - started

    timings["total_seconds"] = time.perf_counter() - started_study
    manifest_path = output / "simulation_study_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "study_config": str(config_path),
                "output_dir": str(output),
                "master_seed": seed,
                "derived_section_seeds": {
                    "end_to_end": derived_seeds[3],
                },
                "timings": timings,
                "outputs": {key: str(value) for key, value in outputs.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs["manifest"] = manifest_path
    return outputs
