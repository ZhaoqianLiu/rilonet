from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    configure_worker,
    configured_workers,
    ensure_dir,
    load_prepared_matrix,
    output_root,
    process_context,
    split_subjects,
    worker_chunksize,
)
from .modeling import fit_edge_frame, prepare_edge_frame


_VALIDATION_WORKER_STATE: dict[str, Any] = {}


def _pair_path(config: dict[str, Any], network: pd.DataFrame) -> Path:
    root = output_root(config) / "03_pairs"
    analysis_type = str(network["analysis_type"].iloc[0])
    x_omic = str(network["x_omic"].iloc[0])
    y_omic = str(network["y_omic"].iloc[0])
    if analysis_type == "near":
        return root / "near" / f"{x_omic}_{y_omic}.tsv"
    window = str(network["window_name"].iloc[0])
    return root / "lagged" / f"{window}_{x_omic}_to_{y_omic}.tsv"


def _bootstrap(pairs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    subject_info = pairs.groupby("subject_id", as_index=False).agg(group=("group", "first"))
    pieces = []
    copy_number = 0
    for group, subset in subject_info.groupby("group", sort=False):
        subjects = subset["subject_id"].astype(str).to_numpy()
        for subject in rng.choice(subjects, size=len(subjects), replace=True):
            copy_number += 1
            piece = pairs[pairs["subject_id"].astype(str).eq(str(subject))].copy()
            piece["subject_id"] = f"{subject}__bootstrap_{copy_number:04d}"
            piece["group"] = group
            pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pairs.iloc[0:0].copy()


def _permute_group(
    pairs: pd.DataFrame, rng: np.random.Generator, strata: list[str]
) -> pd.DataFrame:
    aggregation: dict[str, tuple[str, str]] = {"group": ("group", "first")}
    stratum_columns = []
    for name in strata:
        pair_column = f"x_cov__{name}"
        if pair_column in pairs:
            alias = f"stratum__{name}"
            aggregation[alias] = (pair_column, "first")
            stratum_columns.append(alias)
    info = pairs.groupby("subject_id", as_index=False).agg(**aggregation)
    for column in stratum_columns:
        info[column] = info[column].fillna("Missing").astype(str)
    mapping: dict[str, str] = {}
    grouped = info.groupby(stratum_columns, dropna=False, sort=False) if stratum_columns else [(None, info)]
    for _, subset in grouped:
        labels = subset["group"].astype(str).to_numpy().copy()
        rng.shuffle(labels)
        mapping.update(dict(zip(subset["subject_id"].astype(str), labels)))
    result = pairs.copy()
    result["group"] = result["subject_id"].astype(str).map(mapping)
    return result


def _circular_shift(pairs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    result = pairs.copy()
    for _, indices in result.groupby("subject_id").groups.items():
        subset = result.loc[indices]
        ordered = (
            subset[["x_sample_id", "x_time"]]
            .drop_duplicates()
            .sort_values(["x_time", "x_sample_id"])["x_sample_id"]
            .astype(str)
            .tolist()
        )
        if len(ordered) < 2:
            continue
        shift = int(rng.integers(1, len(ordered)))
        shifted = ordered[shift:] + ordered[:shift]
        mapping = dict(zip(ordered, shifted))
        result.loc[indices, "x_sample_id"] = result.loc[indices, "x_sample_id"].astype(str).map(mapping)
    return result


def _fit_edge(
    config: dict[str, Any],
    pairs: pd.DataFrame,
    x_matrix: pd.DataFrame,
    y_matrix: pd.DataFrame,
    edge: pd.Series,
) -> pd.DataFrame:
    analysis_type = str(edge["analysis_type"])
    baseline_mode = str(edge.get("baseline_mode", "none"))
    orientations = [False, True] if analysis_type == "near" and bool(
        config["modeling"].get("near_fit_both_orientations", True)
    ) else [False]
    fitted = []
    for reverse in orientations:
        frame = prepare_edge_frame(
            pairs,
            x_matrix,
            y_matrix,
            str(edge["x_feature"]),
            str(edge["y_feature"]),
            analysis_type,
            baseline_mode,
            reverse,
        )
        result, _ = fit_edge_frame(frame, analysis_type, config, baseline_mode)
        fitted.append(result.set_index("group"))
    result = fitted[0][["effect", "contrast_vs_reference"]].copy()
    if len(fitted) == 2:
        result = (result + fitted[1][["effect", "contrast_vs_reference"]]) / 2.0
    return result.reset_index()


def _selection_q_column(network: pd.DataFrame) -> str:
    if "contrast_q_global" in network.columns:
        return "contrast_q_global"
    if "contrast_q" in network.columns:
        return "contrast_q"
    raise ValueError("Network file must contain contrast_q or contrast_q_global")


def _as_boolean_series(values: pd.Series, default: bool = True) -> pd.Series:
    mapped = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "false": False,
                "0": False,
                "no": False,
            }
        )
    )
    return mapped.fillna(default).astype(bool)


def _primary_network_eligibility(network: pd.DataFrame) -> pd.Series:
    if "primary_network_eligible" in network:
        row_eligible = _as_boolean_series(network["primary_network_eligible"])
    elif {"analysis_type", "sign_consistent"}.issubset(network.columns):
        near = network["analysis_type"].astype(str).eq("near")
        row_eligible = ~near | _as_boolean_series(network["sign_consistent"])
    else:
        row_eligible = pd.Series(True, index=network.index, dtype=bool)
    edge_eligible = row_eligible.groupby(network["edge_id"]).all()
    return network["edge_id"].map(edge_eligible).fillna(False).astype(bool)


def _selected_edges(network: pd.DataFrame, threshold: float, maximum: int) -> pd.DataFrame:
    q_column = _selection_q_column(network)
    values = pd.to_numeric(network[q_column], errors="coerce")
    eligible = _primary_network_eligibility(network)
    significant_ids = (
        network.loc[(values <= threshold) & eligible, ["edge_id", q_column]]
        .drop_duplicates("edge_id")
        .sort_values(q_column)
        .head(maximum)["edge_id"]
    )
    if significant_ids.empty:
        raise ValueError(f"No edges meet the configured {q_column} threshold")
    return network[network["edge_id"].isin(significant_ids)].copy()


def _validation_output_dir(config: dict[str, Any], network_path: Path) -> Path:
    if network_path.parent.name == "05_networks":
        return ensure_dir(network_path.parent.parent / "06_validation" / network_path.stem)
    return ensure_dir(output_root(config) / "06_validation" / network_path.stem)


def _bootstrap_summary(observed: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q_column = _selection_q_column(observed)
    for (edge_id, group), target in observed.groupby(["edge_id", "group"]):
        subset = draws[(draws["edge_id"].eq(edge_id)) & (draws["group"].eq(group))]
        for statistic in ("effect", "contrast_vs_reference"):
            values = pd.to_numeric(subset[statistic], errors="coerce").dropna().to_numpy()
            observed_value = float(target[statistic].iloc[0])
            rows.append(
                {
                    "edge_id": edge_id,
                    "group": group,
                    "statistic": statistic,
                    "selection_q_column": q_column,
                    "selection_q": float(target[q_column].iloc[0]),
                    "observed": observed_value,
                    "successful_replicates": int(len(values)),
                    "bootstrap_median": float(np.median(values)) if len(values) else np.nan,
                    "ci_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                    "ci_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                    "sign_consistency": float(np.mean(np.sign(values) == np.sign(observed_value))) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _permutation_summary(
    observed: pd.DataFrame, draws: pd.DataFrame, statistic: str
) -> pd.DataFrame:
    rows = []
    q_column = _selection_q_column(observed)
    for (edge_id, group), target in observed.groupby(["edge_id", "group"]):
        values = pd.to_numeric(
            draws.loc[draws["edge_id"].eq(edge_id) & draws["group"].eq(group), statistic],
            errors="coerce",
        ).dropna().to_numpy()
        observed_value = float(target[statistic].iloc[0])
        empirical = (
            float((1 + np.sum(np.abs(values) >= abs(observed_value))) / (len(values) + 1))
            if len(values)
            else np.nan
        )
        rows.append(
            {
                "edge_id": edge_id,
                "group": group,
                "statistic": statistic,
                "selection_q_column": q_column,
                "selection_q": float(target[q_column].iloc[0]),
                "observed": observed_value,
                "successful_replicates": int(len(values)),
                "empirical_p": empirical,
            }
        )
    return pd.DataFrame(rows)


def _replicate_seeds(seed: int, stream: int, count: int) -> list[int]:
    root = np.random.SeedSequence([int(seed), int(stream)])
    return [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in root.spawn(int(count))
    ]


def _initialize_validation_worker(
    config: dict[str, Any],
    pairs: pd.DataFrame,
    x_matrix: pd.DataFrame,
    y_matrix: pd.DataFrame,
    edges: list[dict[str, Any]],
    strata: list[str],
    is_lagged: bool,
) -> None:
    configure_worker(config)
    _VALIDATION_WORKER_STATE.update(
        config=config,
        pairs=pairs,
        x_matrix=x_matrix,
        y_matrix=y_matrix,
        edges=edges,
        strata=strata,
        is_lagged=is_lagged,
    )


def _fit_validation_edges(pairs: pd.DataFrame, replicate: int) -> list[dict[str, Any]]:
    rows = []
    for edge in _VALIDATION_WORKER_STATE["edges"]:
        try:
            fitted = _fit_edge(
                _VALIDATION_WORKER_STATE["config"],
                pairs,
                _VALIDATION_WORKER_STATE["x_matrix"],
                _VALIDATION_WORKER_STATE["y_matrix"],
                edge,
            )
            rows.extend(
                {"replicate": replicate, "edge_id": edge["edge_id"], **row}
                for row in fitted.to_dict("records")
            )
        except Exception:
            pass
    return rows


def _bootstrap_replicate(task: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, seed = task
    rng = np.random.default_rng(seed)
    sampled = _bootstrap(_VALIDATION_WORKER_STATE["pairs"], rng)
    return _fit_validation_edges(sampled, replicate)


def _permutation_replicate(
    task: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replicate, seed = task
    rng = np.random.default_rng(seed)
    pairs = _VALIDATION_WORKER_STATE["pairs"]
    permuted = _permute_group(pairs, rng, _VALIDATION_WORKER_STATE["strata"])
    group_rows = _fit_validation_edges(permuted, replicate)
    temporal_rows = []
    if _VALIDATION_WORKER_STATE["is_lagged"]:
        shifted = _circular_shift(pairs, rng)
        temporal_rows = _fit_validation_edges(shifted, replicate)
    return group_rows, temporal_rows


def _run_validation_replicates(
    config: dict[str, Any],
    pairs: pd.DataFrame,
    x_matrix: pd.DataFrame,
    y_matrix: pd.DataFrame,
    edge_records: list[dict[str, Any]],
    strata: list[str],
    is_lagged: bool,
    seed: int,
    bootstrap_replicates: int,
    permutation_replicates: int,
    worker_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _initialize_validation_worker(
        config, pairs, x_matrix, y_matrix, edge_records, strata, is_lagged
    )
    if worker_count == 1:
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=process_context(config),
            initializer=_initialize_validation_worker,
            initargs=(config, pairs, x_matrix, y_matrix, edge_records, strata, is_lagged),
        )

    try:
        bootstrap_tasks = list(
            enumerate(_replicate_seeds(seed, stream=0, count=bootstrap_replicates))
        )
        if executor is None:
            bootstrap_results = map(_bootstrap_replicate, bootstrap_tasks)
        else:
            bootstrap_results = executor.map(
                _bootstrap_replicate,
                bootstrap_tasks,
                chunksize=worker_chunksize(config, bootstrap_replicates, worker_count),
            )
        bootstrap_rows = []
        for replicate, rows in enumerate(bootstrap_results, 1):
            bootstrap_rows.extend(rows)
            if replicate % 50 == 0 or replicate == bootstrap_replicates:
                print(
                    f"[validation] bootstrap {replicate}/{bootstrap_replicates} "
                    f"with {worker_count} worker(s)",
                    flush=True,
                )

        permutation_tasks = list(
            enumerate(_replicate_seeds(seed, stream=1, count=permutation_replicates))
        )
        if executor is None:
            permutation_results = map(_permutation_replicate, permutation_tasks)
        else:
            permutation_results = executor.map(
                _permutation_replicate,
                permutation_tasks,
                chunksize=worker_chunksize(config, permutation_replicates, worker_count),
            )
        group_rows = []
        temporal_rows = []
        for replicate, (replicate_group, replicate_temporal) in enumerate(
            permutation_results, 1
        ):
            group_rows.extend(replicate_group)
            temporal_rows.extend(replicate_temporal)
            if replicate % 50 == 0 or replicate == permutation_replicates:
                print(
                    f"[validation] permutation {replicate}/{permutation_replicates} "
                    f"with {worker_count} worker(s)",
                    flush=True,
                )
        return bootstrap_rows, group_rows, temporal_rows
    finally:
        if executor is not None:
            executor.shutdown()


def validate_network(
    config: dict[str, Any],
    network_path: str | Path,
    bootstrap_replicates: int | None = None,
    permutation_replicates: int | None = None,
    max_edges: int | None = None,
    workers: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    path = Path(network_path)
    network = pd.read_csv(path, sep="\t", low_memory=False)
    cfg = config["validation"]
    selected = _selected_edges(
        network,
        float(cfg.get("q_threshold", 0.05)),
        int(max_edges or cfg.get("max_edges", 100)),
    )
    unique_edges = selected.drop_duplicates("edge_id")
    pairs = pd.read_csv(_pair_path(config, network), sep="\t", low_memory=False)
    inference_subjects = split_subjects(config, "inference")
    if inference_subjects is not None:
        pairs = pairs[pairs["subject_id"].astype(str).isin(inference_subjects)].copy()
    x_matrix = load_prepared_matrix(config, str(network["x_omic"].iloc[0]))
    y_matrix = load_prepared_matrix(config, str(network["y_omic"].iloc[0]))
    bootstrap_replicates = int(bootstrap_replicates or cfg.get("bootstrap_replicates", 500))
    permutation_replicates = int(permutation_replicates or cfg.get("permutation_replicates", 1000))
    output_dir = (
        ensure_dir(Path(output_dir).resolve())
        if output_dir is not None
        else _validation_output_dir(config, path)
    )
    is_lagged = str(network["analysis_type"].iloc[0]) == "lagged"
    strata = list(cfg.get("permutation_strata", []))
    edge_records = unique_edges.to_dict("records")
    worker_count = configured_workers(
        config, workers, max(bootstrap_replicates, permutation_replicates)
    )
    seed = int(cfg.get("seed", 20260715))
    bootstrap_rows, group_rows, temporal_rows = _run_validation_replicates(
        config,
        pairs,
        x_matrix,
        y_matrix,
        edge_records,
        strata,
        is_lagged,
        seed,
        bootstrap_replicates,
        permutation_replicates,
        worker_count,
    )
    bootstrap_draws = pd.DataFrame(bootstrap_rows)
    bootstrap_draws.to_parquet(output_dir / "bootstrap_draws.parquet")
    bootstrap_summary = _bootstrap_summary(selected, bootstrap_draws)
    bootstrap_summary.to_csv(output_dir / "bootstrap_summary.tsv", sep="\t", index=False)

    group_draws = pd.DataFrame(group_rows)
    group_draws.to_parquet(output_dir / "group_permutation_draws.parquet")
    condition = next(
        level
        for level in config["metadata"]["group_levels"]
        if level != config["metadata"]["reference_group"]
    )
    group_summary = _permutation_summary(
        selected[selected["group"].eq(condition)],
        group_draws[group_draws["group"].eq(condition)],
        "contrast_vs_reference",
    )
    group_summary.to_csv(output_dir / "group_permutation_summary.tsv", sep="\t", index=False)
    outputs = {
        "bootstrap_summary": output_dir / "bootstrap_summary.tsv",
        "group_permutation_summary": output_dir / "group_permutation_summary.tsv",
    }
    if is_lagged:
        temporal_draws = pd.DataFrame(temporal_rows)
        temporal_draws.to_parquet(output_dir / "temporal_permutation_draws.parquet")
        temporal_summary = _permutation_summary(selected, temporal_draws, "effect")
        temporal_summary.to_csv(output_dir / "temporal_permutation_summary.tsv", sep="\t", index=False)
        outputs["temporal_permutation_summary"] = output_dir / "temporal_permutation_summary.tsv"
    return outputs
