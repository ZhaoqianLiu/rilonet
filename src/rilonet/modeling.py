from __future__ import annotations

import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    bh_fdr,
    categorical_covariates,
    configure_worker,
    configured_workers,
    ensure_dir,
    load_prepared_matrix,
    normal_two_sided_p,
    numeric_covariates,
    output_root,
    process_context,
    safe_standardize,
    split_subjects,
    worker_chunksize,
    write_json,
)
from .residualize import _full_rank


_MODEL_WORKER_STATE: dict[str, Any] = {}


def _z(values: pd.Series | np.ndarray) -> np.ndarray:
    return safe_standardize(np.asarray(values, dtype=float))[0]


def _near_association_method(config: dict[str, Any]) -> str:
    """Return the normalized near-synchronous association method."""
    value = (
        str(config.get("modeling", {}).get("near_association_method", "max_p"))
        .strip()
        .lower()
    )
    aliases = {
        "max_p": "max_p",
        "combined_max_p": "max_p",
        "bidirectional_max_p": "max_p",
        "symmetric_fisher_z": "symmetric_fisher_z",
    }
    if value not in aliases:
        raise ValueError(
            "modeling.near_association_method must be max_p or " "symmetric_fisher_z"
        )
    return aliases[value]


def prepare_edge_frame(
    pairs: pd.DataFrame,
    x_matrix: pd.DataFrame,
    y_matrix: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    analysis_type: str,
    baseline_mode: str,
    reverse: bool = False,
) -> pd.DataFrame:
    work = pairs.copy()
    exists = work["x_sample_id"].astype(str).isin(x_matrix.index) & work[
        "y_sample_id"
    ].astype(str).isin(y_matrix.index)
    work = work[exists].copy()
    x_ids = work["x_sample_id"].astype(str)
    y_ids = work["y_sample_id"].astype(str)
    if reverse:
        if analysis_type != "near":
            raise ValueError(
                "Reverse orientation is defined only for near-contemporaneous pairs"
            )
        work["predictor"] = y_matrix.loc[y_ids, y_feature].to_numpy(dtype=float)
        work["outcome_raw"] = x_matrix.loc[x_ids, x_feature].to_numpy(dtype=float)
    else:
        work["predictor"] = x_matrix.loc[x_ids, x_feature].to_numpy(dtype=float)
        work["outcome_raw"] = y_matrix.loc[y_ids, y_feature].to_numpy(dtype=float)

    if analysis_type == "lagged" and baseline_mode in {"adjust", "change"}:
        work = work[work["has_baseline"].astype(str).str.lower().eq("true")].copy()
        lookup = y_matrix[y_feature]
        work["baseline_outcome_raw"] = (
            work["baseline_sample_id"].astype(str).map(lookup)
        )
        work = work[
            np.isfinite(pd.to_numeric(work["baseline_outcome_raw"], errors="coerce"))
        ].copy()
        if baseline_mode == "change":
            work["outcome_raw"] = work["outcome_raw"] - work["baseline_outcome_raw"]

    finite = np.isfinite(work["predictor"]) & np.isfinite(work["outcome_raw"])
    work = work[finite].copy()
    work["outcome"] = _z(work["outcome_raw"])
    work["subject_id"] = work["subject_id"].astype(str)
    work["group"] = work["group"].astype(str)
    return work.reset_index(drop=True)


def make_design(
    frame: pd.DataFrame,
    analysis_type: str,
    config: dict[str, Any],
    baseline_mode: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    levels = list(config["metadata"]["group_levels"])
    reference = str(config["metadata"]["reference_group"])
    condition = next(level for level in levels if level != reference)
    participant_mean = frame.groupby("subject_id")["predictor"].transform("mean")
    within = frame["predictor"] - participant_mean
    design = pd.DataFrame(index=frame.index)
    design["Intercept"] = 1.0
    design["x_within"] = _z(within)
    design["x_between"] = _z(participant_mean)
    indicator = frame["group"].eq(condition).astype(float)
    design["group_condition"] = indicator
    design["x_within_by_condition"] = design["x_within"] * indicator

    time = pd.to_numeric(frame["mid_time"], errors="coerce")
    if time.notna().sum() > 1 and time.std(ddof=1) > 0:
        design["mid_time"] = _z(time.fillna(time.median()))
    gap_column = "abs_delta_days" if analysis_type == "near" else "delta_days"
    gap = pd.to_numeric(frame[gap_column], errors="coerce")
    if gap.notna().sum() > 1 and gap.std(ddof=1) > 0:
        design["time_gap"] = _z(gap.fillna(gap.median()))

    if analysis_type == "lagged" and baseline_mode == "adjust":
        baseline = pd.to_numeric(frame["baseline_outcome_raw"], errors="coerce")
        if baseline.notna().all() and baseline.std(ddof=1) > 0:
            baseline_mean = baseline.groupby(frame["subject_id"]).transform("mean")
            design["baseline_outcome_within"] = _z(baseline - baseline_mean)
            design["baseline_outcome_between"] = _z(baseline_mean)
            design["baseline_outcome_within_by_condition"] = (
                design["baseline_outcome_within"] * indicator
            )
            design["baseline_outcome_between_by_condition"] = (
                design["baseline_outcome_between"] * indicator
            )

    # Near-contemporaneous covariates remain anchored to the canonical x side
    # when the molecular orientation is reversed; lagged covariates are source-side.
    for column in categorical_covariates(config):
        values = (
            frame[f"x_cov__{column}"]
            .fillna("Missing")
            .astype(str)
            .replace("", "Missing")
        )
        dummies = pd.get_dummies(
            values, prefix=f"cov_{column}", drop_first=True, dtype=float
        )
        design = pd.concat([design, dummies], axis=1)
    for column in numeric_covariates(config):
        values = pd.to_numeric(frame[f"x_cov__{column}"], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        values = values.fillna(median)
        if values.std(ddof=1) > 0:
            design[f"cov_{column}"] = _z(values)

    design = _full_rank(design.astype(float))
    return design, frame["outcome"].to_numpy(dtype=float)


def _fit_model(
    outcome: np.ndarray,
    design: pd.DataFrame,
    groups: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import statsmodels.api as sm

    cfg = config["modeling"]
    x = design.to_numpy(dtype=float)
    diagnostics: dict[str, Any] = {"method": "failed", "converged": False, "error": ""}
    errors = []
    for optimizer in cfg.get("optimizers", ["lbfgs", "powell", "cg"]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.MixedLM(outcome, x, groups=groups).fit(
                    reml=False,
                    method=str(optimizer),
                    maxiter=int(cfg.get("max_iterations", 200)),
                    disp=False,
                )
            beta = np.asarray(fit.fe_params, dtype=float)
            covariance = np.asarray(fit.cov_params(), dtype=float)[
                : len(beta), : len(beta)
            ]
            finite = np.isfinite(beta).all() and np.isfinite(covariance).all()
            if finite and bool(fit.converged):
                diagnostics.update(
                    {
                        "method": "mixedlm_random_intercept",
                        "optimizer": optimizer,
                        "converged": bool(fit.converged),
                        "random_intercept_variance": float(
                            np.asarray(fit.cov_re)[0, 0]
                        ),
                        "error": ";".join(errors),
                    }
                )
                return beta, covariance, diagnostics
            errors.append(
                f"{optimizer}:nonconverged" if finite else f"{optimizer}:nonfinite"
            )
        except Exception as exc:
            errors.append(f"{optimizer}:{type(exc).__name__}:{exc}")
    try:
        fit = sm.OLS(outcome, x).fit(
            cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True}
        )
        diagnostics.update(
            {
                "method": "ols_participant_cluster_robust_fallback",
                "converged": True,
                "error": ";".join(errors),
            }
        )
        return np.asarray(fit.params), np.asarray(fit.cov_params()), diagnostics
    except Exception as exc:
        raise RuntimeError(
            ";".join(errors) + f";ols:{type(exc).__name__}:{exc}"
        ) from exc


def _contrast(
    beta: np.ndarray, covariance: np.ndarray, vector: np.ndarray
) -> tuple[float, float, float]:
    estimate = float(vector @ beta)
    variance = float(vector @ covariance @ vector)
    se = float(np.sqrt(max(variance, 0.0)))
    pvalue = normal_two_sided_p(estimate / se) if se > 0 else np.nan
    return estimate, se, pvalue


def fit_edge_frame(
    frame: pd.DataFrame,
    analysis_type: str,
    config: dict[str, Any],
    baseline_mode: str = "none",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if (
        analysis_type == "near"
        and _near_association_method(config) == "symmetric_fisher_z"
    ):
        return fit_symmetric_near_frame(frame, config)

    levels = list(config["metadata"]["group_levels"])
    reference = str(config["metadata"]["reference_group"])
    condition = next(level for level in levels if level != reference)
    cfg = config["modeling"]
    counts = frame.groupby("group")["subject_id"].nunique()
    if len(frame) < int(cfg["min_pairs"]) or frame["subject_id"].nunique() < int(
        cfg["min_subjects"]
    ):
        raise RuntimeError("insufficient_pairs_or_subjects")
    for level in levels:
        if int(counts.get(level, 0)) < int(cfg["min_subjects_per_group"]):
            raise RuntimeError(f"insufficient_subjects_in_{level}")
    design, outcome = make_design(frame, analysis_type, config, baseline_mode)
    if not np.isfinite(design.to_numpy()).all() or not np.isfinite(outcome).all():
        raise RuntimeError("nonfinite_design")
    beta, covariance, diagnostics = _fit_model(
        outcome, design, frame["subject_id"].to_numpy(), config
    )
    names = list(design.columns)
    within_index = names.index("x_within")
    interaction_index = names.index("x_within_by_condition")
    rows = []
    for level in levels:
        slope = np.zeros(len(names))
        slope[within_index] = 1.0
        difference = np.zeros(len(names))
        if level == condition:
            slope[interaction_index] = 1.0
            difference[interaction_index] = 1.0
        effect, effect_se, effect_p = _contrast(beta, covariance, slope)
        if level == reference:
            contrast, contrast_se, contrast_p = 0.0, np.nan, np.nan
        else:
            contrast, contrast_se, contrast_p = _contrast(beta, covariance, difference)
        rows.append(
            {
                "group": level,
                "effect": effect,
                "effect_se": effect_se,
                "effect_p": effect_p,
                "contrast_vs_reference": contrast,
                "contrast_se": contrast_se,
                "contrast_p": contrast_p,
                "n_pairs_model": int(len(frame)),
                "n_subjects_model": int(frame["subject_id"].nunique()),
                "n_subjects_group": int(counts.get(level, 0)),
                "fit_method": diagnostics["method"],
                "fit_converged": diagnostics["converged"],
            }
        )
    diagnostics["design_columns"] = names
    return pd.DataFrame(rows), diagnostics


def _fisher_z_from_centered_sums(x2: float, y2: float, xy: float) -> float:
    denominator = float(np.sqrt(max(x2, 0.0) * max(y2, 0.0)))
    if denominator <= 0:
        raise RuntimeError("insufficient_variation_for_symmetric_near_test")
    correlation = float(xy / denominator)
    if not np.isfinite(correlation):
        raise RuntimeError("nonfinite_symmetric_near_correlation")
    return float(np.arctanh(np.clip(correlation, -0.999999, 0.999999)))


def fit_symmetric_near_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit an orientation-invariant near-synchronous Fisher-z contrast.

    Feature values are centered within participant.  Group-specific Fisher-z
    correlations and their difference are assigned uncertainty with a
    group-stratified delete-one-participant jackknife.
    """
    levels = [str(value) for value in config["metadata"]["group_levels"]]
    reference = str(config["metadata"]["reference_group"])
    condition = next(level for level in levels if level != reference)
    cfg = config["modeling"]

    work = frame[["subject_id", "group", "predictor", "outcome_raw"]].copy()
    work["subject_id"] = work["subject_id"].astype(str)
    work["group"] = work["group"].astype(str)
    work["predictor"] = pd.to_numeric(work["predictor"], errors="coerce")
    work["outcome_raw"] = pd.to_numeric(work["outcome_raw"], errors="coerce")
    finite = np.isfinite(work["predictor"]) & np.isfinite(work["outcome_raw"])
    work = work[finite].copy()

    counts = work.groupby("group")["subject_id"].nunique()
    if len(work) < int(cfg["min_pairs"]) or work["subject_id"].nunique() < int(
        cfg["min_subjects"]
    ):
        raise RuntimeError("insufficient_pairs_or_subjects")
    for level in levels:
        if int(counts.get(level, 0)) < int(cfg["min_subjects_per_group"]):
            raise RuntimeError(f"insufficient_subjects_in_{level}")

    work["x_within"] = work["predictor"] - work.groupby("subject_id")[
        "predictor"
    ].transform("mean")
    work["y_within"] = work["outcome_raw"] - work.groupby("subject_id")[
        "outcome_raw"
    ].transform("mean")
    work["x2"] = work["x_within"] ** 2
    work["y2"] = work["y_within"] ** 2
    work["xy"] = work["x_within"] * work["y_within"]
    subject_stats = work.groupby(["group", "subject_id"], as_index=False)[
        ["x2", "y2", "xy"]
    ].sum()

    effects: dict[str, float] = {}
    variances: dict[str, float] = {}
    for level in levels:
        group_stats = subject_stats[subject_stats["group"].eq(level)].copy()
        if len(group_stats) < 3:
            raise RuntimeError(f"insufficient_subjects_for_symmetric_near_{level}")
        total = group_stats[["x2", "y2", "xy"]].sum()
        effect = _fisher_z_from_centered_sums(
            float(total["x2"]),
            float(total["y2"]),
            float(total["xy"]),
        )
        leave_one = []
        for row in group_stats.itertuples(index=False):
            leave_one.append(
                _fisher_z_from_centered_sums(
                    float(total["x2"] - row.x2),
                    float(total["y2"] - row.y2),
                    float(total["xy"] - row.xy),
                )
            )
        deleted = np.asarray(leave_one, dtype=float)
        variance = (
            (len(deleted) - 1)
            / len(deleted)
            * float(np.sum((deleted - deleted.mean()) ** 2))
        )
        effects[level] = effect
        variances[level] = max(variance, 0.0)

    contrast = effects[condition] - effects[reference]
    contrast_variance = variances[condition] + variances[reference]
    contrast_se = float(np.sqrt(max(contrast_variance, 0.0)))
    contrast_p = (
        normal_two_sided_p(contrast / contrast_se) if contrast_se > 0 else np.nan
    )

    rows = []
    for level in levels:
        effect_se = float(np.sqrt(variances[level]))
        effect_p = (
            normal_two_sided_p(effects[level] / effect_se) if effect_se > 0 else np.nan
        )
        rows.append(
            {
                "group": level,
                "effect": effects[level],
                "effect_se": effect_se,
                "effect_p": effect_p,
                "contrast_vs_reference": (contrast if level == condition else 0.0),
                "contrast_se": (contrast_se if level == condition else np.nan),
                "contrast_p": (contrast_p if level == condition else np.nan),
                "n_pairs_model": int(len(work)),
                "n_subjects_model": int(work["subject_id"].nunique()),
                "n_subjects_group": int(counts.get(level, 0)),
                "fit_method": "symmetric_fisher_z_participant_jackknife",
                "fit_converged": True,
            }
        )
    diagnostics = {
        "method": "symmetric_fisher_z_participant_jackknife",
        "converged": True,
        "jackknife_participants": int(work["subject_id"].nunique()),
        "error": "",
    }
    return pd.DataFrame(rows), diagnostics


def _pair_path(config: dict[str, Any], candidates: pd.DataFrame) -> Path:
    root = output_root(config) / "03_pairs"
    analysis_type = str(candidates["analysis_type"].iloc[0])
    x_omic = str(candidates["x_omic"].iloc[0])
    y_omic = str(candidates["y_omic"].iloc[0])
    if analysis_type == "near":
        return root / "near" / f"{x_omic}_{y_omic}.tsv"
    window = str(candidates["window_name"].iloc[0])
    return root / "lagged" / f"{window}_{x_omic}_to_{y_omic}.tsv"


def fit_candidate_file(
    config: dict[str, Any],
    candidate_path: Path,
    max_edges: int | None,
    baseline_mode: str,
    matrix_cache: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(candidate_path, sep="\t", low_memory=False)
    if max_edges is not None:
        candidates = candidates.head(int(max_edges))
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    analysis_type = str(candidates["analysis_type"].iloc[0])
    effective_baseline_mode = "none" if analysis_type == "near" else baseline_mode
    x_omic = str(candidates["x_omic"].iloc[0])
    y_omic = str(candidates["y_omic"].iloc[0])
    pairs = pd.read_csv(_pair_path(config, candidates), sep="\t", low_memory=False)
    inference_subjects = split_subjects(config, "inference")
    if inference_subjects is not None:
        pairs = pairs[pairs["subject_id"].astype(str).isin(inference_subjects)].copy()
    if matrix_cache is None:
        x_matrix = load_prepared_matrix(config, x_omic)
        y_matrix = load_prepared_matrix(config, y_omic)
    else:
        if x_omic not in matrix_cache:
            matrix_cache[x_omic] = load_prepared_matrix(config, x_omic)
        if y_omic not in matrix_cache:
            matrix_cache[y_omic] = load_prepared_matrix(config, y_omic)
        x_matrix = matrix_cache[x_omic]
        y_matrix = matrix_cache[y_omic]
    results_rows = []
    diagnostics_rows = []
    for edge_index, candidate in candidates.reset_index(drop=True).iterrows():
        edge_id = (
            f"{analysis_type}_{candidate['window_name']}_{x_omic}_{candidate['x_feature']}"
            f"__{y_omic}_{candidate['y_feature']}"
        )
        near_method = (
            _near_association_method(config)
            if analysis_type == "near"
            else "directional_lmm"
        )
        reverse_options = (
            [False, True]
            if (
                analysis_type == "near"
                and near_method == "max_p"
                and bool(config["modeling"].get("near_fit_both_orientations", True))
            )
            else [False]
        )
        fitted_directions = []
        failed = False
        for reverse in reverse_options:
            label = (
                "symmetric"
                if analysis_type == "near" and near_method == "symmetric_fisher_z"
                else ("reverse" if reverse else "forward")
            )
            try:
                frame = prepare_edge_frame(
                    pairs,
                    x_matrix,
                    y_matrix,
                    str(candidate["x_feature"]),
                    str(candidate["y_feature"]),
                    analysis_type,
                    effective_baseline_mode,
                    reverse,
                )
                fitted, diagnostics = fit_edge_frame(
                    frame, analysis_type, config, effective_baseline_mode
                )
                fitted_directions.append(fitted.set_index("group"))
                diagnostics_rows.append(
                    {
                        "edge_id": edge_id,
                        "orientation": label,
                        "status": "ok",
                        "error": "",
                        **diagnostics,
                    }
                )
            except Exception as exc:
                failed = True
                diagnostics_rows.append(
                    {
                        "edge_id": edge_id,
                        "orientation": label,
                        "status": "failed",
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
        if failed or not fitted_directions:
            continue
        combined = fitted_directions[0].copy()
        if len(fitted_directions) == 2:
            reverse = fitted_directions[1]
            for column in ("effect", "contrast_vs_reference"):
                combined[column] = (combined[column] + reverse[column]) / 2.0
            combined["effect_p"] = np.maximum(combined["effect_p"], reverse["effect_p"])
            combined["contrast_p"] = np.maximum(
                combined["contrast_p"], reverse["contrast_p"]
            )
            edge_sign_consistent = bool(
                (
                    np.sign(fitted_directions[0]["contrast_vs_reference"])
                    == np.sign(reverse["contrast_vs_reference"])
                ).all()
            )
        else:
            edge_sign_consistent = True
        combined["sign_consistent"] = edge_sign_consistent
        combined["primary_network_eligible"] = (
            analysis_type != "near" or edge_sign_consistent
        )
        for group, row in combined.reset_index().set_index("group").iterrows():
            results_rows.append(
                {
                    "edge_id": edge_id,
                    "analysis_type": analysis_type,
                    "block": candidate["block"],
                    "window_name": candidate["window_name"],
                    "baseline_mode": effective_baseline_mode,
                    "near_association_method": near_method,
                    "effect_scale": (
                        "fisher_z"
                        if near_method == "symmetric_fisher_z"
                        else "standardized_slope"
                    ),
                    "x_omic": x_omic,
                    "y_omic": y_omic,
                    "x_feature": candidate["x_feature"],
                    "y_feature": candidate["y_feature"],
                    "screen_correlation": candidate["screen_correlation"],
                    "group": group,
                    "effect": row["effect"],
                    "effect_p": row["effect_p"],
                    "contrast_vs_reference": row["contrast_vs_reference"],
                    "contrast_p": row["contrast_p"],
                    "sign_consistent": row["sign_consistent"],
                    "primary_network_eligible": row["primary_network_eligible"],
                    "n_pairs_model": row["n_pairs_model"],
                    "n_subjects_model": row["n_subjects_model"],
                    "fit_method": row["fit_method"],
                    "fit_converged": row["fit_converged"],
                }
            )
        if (edge_index + 1) % 100 == 0 or edge_index + 1 == len(candidates):
            print(
                f"[{candidate_path.stem}] modeled {edge_index + 1}/{len(candidates)}",
                flush=True,
            )

    results = pd.DataFrame(results_rows)
    if not results.empty:
        results["effect_q"] = np.nan
        for _, indices in results.groupby("group").groups.items():
            results.loc[indices, "effect_q"] = bh_fdr(results.loc[indices, "effect_p"])
        condition = next(
            level
            for level in config["metadata"]["group_levels"]
            if level != config["metadata"]["reference_group"]
        )
        condition_rows = results[results["group"].eq(condition)].drop_duplicates(
            "edge_id"
        )
        qvalues = bh_fdr(condition_rows["contrast_p"])
        contrast_map = dict(zip(condition_rows["edge_id"], qvalues))
        results["contrast_q_block"] = results["edge_id"].map(contrast_map)
        results["contrast_q"] = results["contrast_q_block"]
    return results, pd.DataFrame(diagnostics_rows)


def _apply_global_contrast_q(
    edge_paths: list[Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply BH once across all fitted edge contrasts and rewrite network files."""
    condition = next(
        level
        for level in config["metadata"]["group_levels"]
        if level != config["metadata"]["reference_group"]
    )
    loaded: list[tuple[Path, pd.DataFrame]] = []
    tests = []
    for path in edge_paths:
        try:
            frame = pd.read_csv(path, sep="\t", low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        loaded.append((path, frame))
        if frame.empty:
            continue
        required = {"edge_id", "group", "contrast_p"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Network file is missing global-BH columns: {path}")
        condition_rows = (
            frame[frame["group"].astype(str).eq(str(condition))]
            .drop_duplicates("edge_id")
            .loc[:, ["edge_id", "contrast_p"]]
        )
        tests.append(condition_rows)

    if not tests:
        return {
            "scope": "global",
            "tested_contrasts": 0,
            "network_files": len(loaded),
        }

    combined = pd.concat(tests, ignore_index=True)
    duplicate = combined["edge_id"].duplicated(keep=False)
    if duplicate.any():
        examples = combined.loc[duplicate, "edge_id"].astype(str).unique()[:5].tolist()
        raise ValueError(f"Duplicate edge IDs in global BH family: {examples}")
    combined["contrast_q_global"] = bh_fdr(combined["contrast_p"])
    q_map = dict(zip(combined["edge_id"], combined["contrast_q_global"]))

    for path, frame in loaded:
        if frame.empty:
            continue
        frame["contrast_q_global"] = frame["edge_id"].map(q_map)
        frame["contrast_q"] = frame["contrast_q_global"]
        frame.to_csv(path, sep="\t", index=False)

    return {
        "scope": "global",
        "tested_contrasts": int(len(combined)),
        "network_files": len(loaded),
    }


def _initialize_model_worker(
    config: dict[str, Any],
    max_edges: int | None,
    baseline_mode: str,
    output_dir: Path,
) -> None:
    configure_worker(config)
    _MODEL_WORKER_STATE.update(
        config=config,
        max_edges=max_edges,
        baseline_mode=baseline_mode,
        output_dir=output_dir,
        matrix_cache={},
    )


def _model_candidate_path(candidate_path: str) -> tuple[str, dict[str, Any]]:
    path = Path(candidate_path)
    config = _MODEL_WORKER_STATE["config"]
    total = len(pd.read_csv(path, sep="\t", usecols=["x_feature"]))
    results, diagnostics = fit_candidate_file(
        config,
        path,
        max_edges=_MODEL_WORKER_STATE["max_edges"],
        baseline_mode=_MODEL_WORKER_STATE["baseline_mode"],
        matrix_cache=_MODEL_WORKER_STATE["matrix_cache"],
    )
    output_dir = _MODEL_WORKER_STATE["output_dir"]
    results.to_csv(output_dir / f"{path.stem}_edges.tsv", sep="\t", index=False)
    diagnostics.to_csv(
        output_dir / f"{path.stem}_fit_diagnostics.tsv", sep="\t", index=False
    )
    effective_modes = (
        sorted(results["baseline_mode"].dropna().astype(str).unique())
        if "baseline_mode" in results
        else []
    )
    report = {
        "candidate_edges": int(total),
        "unique_edges_fitted": int(results["edge_id"].nunique())
        if not results.empty
        else 0,
        "failed_orientation_fits": (
            int((diagnostics["status"] == "failed").sum())
            if not diagnostics.empty
            else 0
        ),
        "baseline_mode": (
            effective_modes[0]
            if len(effective_modes) == 1
            else _MODEL_WORKER_STATE["baseline_mode"]
        ),
        "near_association_method": _near_association_method(config),
    }
    return path.stem, report


def _reuse_near_result(
    config: dict[str, Any],
    candidate_path: Path,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, Any] | None:
    edge_source = source_dir / f"{candidate_path.stem}_edges.tsv"
    if not edge_source.exists():
        return None
    try:
        frame = pd.read_csv(edge_source, sep="\t", low_memory=False)
    except pd.errors.EmptyDataError:
        return None
    if frame.empty:
        return None

    requested_method = _near_association_method(config)
    if "near_association_method" in frame:
        observed_methods = set(frame["near_association_method"].dropna().astype(str))
        observed_method = (
            next(iter(observed_methods)) if len(observed_methods) == 1 else ""
        )
    else:
        observed_method = "max_p"
    observed_method = {
        "combined_max_p": "max_p",
        "bidirectional_max_p": "max_p",
    }.get(observed_method, observed_method)
    if observed_method != requested_method:
        return None

    frame["baseline_mode"] = "none"
    frame["near_association_method"] = requested_method
    frame["effect_scale"] = (
        "fisher_z" if requested_method == "symmetric_fisher_z" else "standardized_slope"
    )
    if "sign_consistent" not in frame:
        frame["sign_consistent"] = True
    sign = (
        frame["sign_consistent"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(True)
    )
    edge_sign = sign.groupby(frame["edge_id"]).all()
    frame["sign_consistent"] = frame["edge_id"].map(edge_sign).astype(bool)
    frame["primary_network_eligible"] = frame["sign_consistent"]
    frame.to_csv(
        output_dir / f"{candidate_path.stem}_edges.tsv",
        sep="\t",
        index=False,
    )

    diagnostics_source = source_dir / f"{candidate_path.stem}_fit_diagnostics.tsv"
    if diagnostics_source.exists():
        shutil.copy2(
            diagnostics_source,
            output_dir / diagnostics_source.name,
        )
    candidate_edges = len(pd.read_csv(candidate_path, sep="\t", usecols=["x_feature"]))
    return {
        "candidate_edges": int(candidate_edges),
        "unique_edges_fitted": int(frame["edge_id"].nunique()),
        "failed_orientation_fits": 0,
        "baseline_mode": "none",
        "near_association_method": requested_method,
        "reused_from": str(source_dir),
    }


def model_all(
    config: dict[str, Any],
    max_edges: int | None = None,
    baseline_mode: str | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    mode = str(baseline_mode or config["modeling"].get("baseline_mode", "none"))
    if mode not in {"none", "adjust", "change"}:
        raise ValueError("baseline_mode must be none, adjust, or change")
    multiple_testing_scope = str(
        config["modeling"].get("multiple_testing_scope", "block")
    )
    if multiple_testing_scope not in {"block", "global"}:
        raise ValueError("modeling.multiple_testing_scope must be block or global")
    if max_edges is not None:
        folder = f"05_networks_smoke_{mode}"
    else:
        folder = "05_networks" if mode == "none" else f"05_networks_baseline_{mode}"
    output_dir = ensure_dir(output_root(config) / folder)
    all_candidate_paths = [
        path
        for path in sorted((output_root(config) / "04_candidates").glob("*.tsv"))
        if path.name.startswith(("near_", "lagged_"))
    ]
    candidate_paths = list(all_candidate_paths)
    report: dict[str, Any] = {}
    if mode != "none" and max_edges is None:
        source_dir = output_root(config) / "05_networks"
        candidate_paths = []
        for path in all_candidate_paths:
            if path.name.startswith("near_"):
                reused = _reuse_near_result(
                    config,
                    path,
                    source_dir,
                    output_dir,
                )
                if reused is not None:
                    report[path.stem] = reused
                    continue
            candidate_paths.append(path)

    worker_count = configured_workers(config, workers, len(candidate_paths))
    _initialize_model_worker(config, max_edges, mode, output_dir)
    if worker_count == 1:
        modeled = map(_model_candidate_path, map(str, candidate_paths))
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=process_context(config),
            initializer=_initialize_model_worker,
            initargs=(config, max_edges, mode, output_dir),
        )
        modeled = executor.map(
            _model_candidate_path,
            map(str, candidate_paths),
            chunksize=worker_chunksize(config, len(candidate_paths), worker_count),
        )
    try:
        report.update(dict(modeled))
    finally:
        if executor is not None:
            executor.shutdown()
    if multiple_testing_scope == "global":
        edge_paths = [
            output_dir / f"{candidate_path.stem}_edges.tsv"
            for candidate_path in all_candidate_paths
        ]
        multiple_testing_report = _apply_global_contrast_q(edge_paths, config)
    else:
        multiple_testing_report = {
            "scope": "block",
            "network_files": len(all_candidate_paths),
        }
    multiple_testing_report["primary_network_directory"] = str(output_dir)
    multiple_testing_report["near_blocks"] = sum(
        path.name.startswith("near_") for path in all_candidate_paths
    )
    multiple_testing_report["lagged_blocks"] = sum(
        path.name.startswith("lagged_") for path in all_candidate_paths
    )
    report["_multiple_testing"] = multiple_testing_report
    write_json(report, output_dir / "network_modeling_qc.json")
    return report
