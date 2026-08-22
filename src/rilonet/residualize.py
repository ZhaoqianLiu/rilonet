from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    OMICS,
    categorical_covariates,
    configure_worker,
    configured_workers,
    ensure_dir,
    load_prepared_manifest,
    load_prepared_matrix,
    numeric_covariates,
    output_root,
    process_context,
    safe_standardize,
    worker_chunksize,
    write_json,
)


_RESIDUAL_WORKER_STATE: dict[str, Any] = {}


def _dependencies() -> tuple[Any, Any]:
    try:
        import patsy
        import statsmodels.api as sm
    except ImportError as exc:
        raise RuntimeError("Residualization requires patsy and statsmodels") from exc
    return sm, patsy


def _full_rank(frame: pd.DataFrame) -> pd.DataFrame:
    from scipy.linalg import qr

    keep = [
        column
        for column in frame
        if column == "Intercept" or float(np.nanstd(frame[column].to_numpy(dtype=float))) > 0
    ]
    frame = frame[keep]
    _, r, pivot = qr(frame.to_numpy(dtype=float), mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    tolerance = max(frame.shape) * np.finfo(float).eps * (diagonal.max() if diagonal.size else 0.0)
    rank = int((diagonal > tolerance).sum())
    chosen = sorted(pivot[:rank])
    result = frame.iloc[:, chosen]
    if "Intercept" in frame and "Intercept" not in result:
        result.insert(0, "Intercept", frame["Intercept"])
        return _full_rank(result)
    return result


def build_design(manifest: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    _, patsy = _dependencies()
    cfg = config["residualization"]
    design = pd.DataFrame({"Intercept": 1.0}, index=manifest.index)
    imputation: dict[str, Any] = {}
    time = pd.to_numeric(manifest["study_time"], errors="coerce")
    if time.isna().any():
        fill = float(time.median())
        time = time.fillna(fill)
        imputation["study_time"] = fill
    spline_df = int(cfg.get("time_spline_df", 3))
    if time.nunique() >= max(4, spline_df + 1):
        spline = patsy.dmatrix(
            f"bs(x, df={spline_df}, degree=3, include_intercept=False) - 1",
            {"x": time.to_numpy(dtype=float)},
            return_type="dataframe",
        )
        spline.index = manifest.index
        spline.columns = [f"time_spline_{i + 1}" for i in range(spline.shape[1])]
        design = pd.concat([design, spline], axis=1)
    elif time.std(ddof=1) > 0:
        design["time_linear"] = (time - time.mean()) / time.std(ddof=1)

    if bool(cfg.get("include_group_main_effect", True)):
        levels = list(config["metadata"]["group_levels"])
        values = pd.Categorical(manifest["group"], categories=levels, ordered=True)
        dummies = pd.get_dummies(values, prefix="group", drop_first=True, dtype=float)
        dummies.index = manifest.index
        design = pd.concat([design, dummies], axis=1)

    for column in categorical_covariates(config):
        values = manifest[column].fillna("Missing").astype(str).replace("", "Missing")
        dummies = pd.get_dummies(values, prefix=f"cat_{column}", drop_first=True, dtype=float)
        design = pd.concat([design, dummies], axis=1)
        imputation[column] = "Missing"

    for column in numeric_covariates(config):
        values = pd.to_numeric(manifest[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        values = values.fillna(median)
        if values.std(ddof=1) > 0:
            design[f"num_{column}"] = (values - values.mean()) / values.std(ddof=1)
        imputation[column] = median

    design = _full_rank(design.astype(float))
    return design, {"columns": list(design.columns), "imputation": imputation}


def _subject_fixed_effect_residuals(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    dummies = pd.get_dummies(pd.Series(groups, dtype=str), drop_first=True, dtype=float).to_numpy()
    augmented = np.column_stack([x, dummies]) if dummies.size else x
    coefficients, *_ = np.linalg.lstsq(augmented, y, rcond=None)
    return y - augmented @ coefficients


def _fit_feature(
    y_all: np.ndarray,
    design_all: np.ndarray,
    groups_all: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    sm, _ = _dependencies()
    finite = np.isfinite(y_all) & np.isfinite(design_all).all(axis=1)
    y = y_all[finite]
    x = design_all[finite]
    groups = groups_all[finite]
    result = np.full(y_all.shape, np.nan)
    diagnostics: dict[str, Any] = {
        "n_observations": int(len(y)),
        "n_subjects": int(pd.unique(groups).size),
        "method": "failed",
        "converged": False,
        "error": "",
    }
    if len(y) < int(cfg["min_observations"]) or pd.unique(groups).size < int(cfg["min_subjects"]):
        diagnostics["error"] = "insufficient_observations_or_subjects"
        return result, diagnostics
    if np.nanstd(y, ddof=1) <= 0:
        diagnostics["error"] = "zero_variance"
        return result, diagnostics

    errors = []
    for optimizer in cfg.get("optimizers", ["lbfgs", "powell", "cg"]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.MixedLM(y, x, groups=groups).fit(
                    reml=bool(cfg.get("reml", True)),
                    method=str(optimizer),
                    maxiter=int(cfg.get("max_iterations", 200)),
                    disp=False,
                )
            fixed = x @ np.asarray(fit.fe_params, dtype=float)
            random = fit.random_effects
            fitted = fixed + np.asarray(
                [float(np.ravel(np.asarray(random[group], dtype=float))[0]) for group in groups]
            )
            residual = y - fitted
            result[finite] = residual
            diagnostics.update(
                {
                    "method": "mixedlm_random_intercept",
                    "optimizer": optimizer,
                    "converged": bool(fit.converged),
                    "random_intercept_variance": float(np.asarray(fit.cov_re)[0, 0]),
                    "error": ";".join(errors),
                }
            )
            return result, diagnostics
        except Exception as exc:
            errors.append(f"{optimizer}:{type(exc).__name__}:{exc}")

    try:
        residual = _subject_fixed_effect_residuals(y, x, groups)
        result[finite] = residual
        diagnostics.update(
            {
                "method": "subject_fixed_effect_fallback",
                "converged": True,
                "error": ";".join(errors),
            }
        )
    except Exception as exc:
        diagnostics["error"] = ";".join(errors) + f";fallback:{type(exc).__name__}:{exc}"
    return result, diagnostics


def _initialize_residual_worker(
    values: np.ndarray,
    design: np.ndarray,
    groups: np.ndarray,
    config: dict[str, Any],
) -> None:
    configure_worker(config)
    _RESIDUAL_WORKER_STATE.update(
        values=values,
        design=design,
        groups=groups,
        cfg=config["residualization"],
    )


def _residualize_feature(index: int) -> tuple[int, np.ndarray, np.ndarray, dict[str, Any], float, float]:
    residual, diagnostics = _fit_feature(
        _RESIDUAL_WORKER_STATE["values"][:, index],
        _RESIDUAL_WORKER_STATE["design"],
        _RESIDUAL_WORKER_STATE["groups"],
        _RESIDUAL_WORKER_STATE["cfg"],
    )
    standardized, mean, sd = safe_standardize(residual)
    return index, residual, standardized, diagnostics, mean, sd


def residualize_omic(
    config: dict[str, Any],
    omic: str,
    max_features: int | None = None,
    workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = load_prepared_manifest(config)
    valid = manifest["valid_for_analysis"].astype(str).str.lower().eq("true")
    manifest = manifest[valid & manifest["omic"].eq(omic)].copy()
    matrix = load_prepared_matrix(config, omic)
    manifest = manifest.set_index("sample_id").loc[matrix.index]
    if max_features is not None:
        matrix = matrix.iloc[:, : int(max_features)]
    design, design_info = build_design(manifest, config)
    values = matrix.to_numpy(dtype=float)
    design_values = design.to_numpy(dtype=float)
    groups = manifest["subject_id"].astype(str).to_numpy()
    worker_count = configured_workers(config, workers, matrix.shape[1])
    # Apply environment limits before spawning so child interpreters inherit
    # them even when threadpoolctl is not yet installed in a development tree.
    configure_worker(config)
    output_dir = ensure_dir(output_root(config) / "02_residuals")
    raw_values = np.empty(values.shape, dtype=float)
    standardized_values = np.empty(values.shape, dtype=float)
    diagnostics_rows = []
    scaling_rows = []

    indices = range(matrix.shape[1])
    if worker_count == 1:
        _initialize_residual_worker(values, design_values, groups, config)
        fitted = map(_residualize_feature, indices)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=process_context(config),
            initializer=_initialize_residual_worker,
            initargs=(values, design_values, groups, config),
        )
        fitted = executor.map(
            _residualize_feature,
            indices,
            chunksize=worker_chunksize(config, matrix.shape[1], worker_count),
        )

    try:
        for completed, (column_index, residual, z, diagnostics, mean, sd) in enumerate(fitted, 1):
            feature = matrix.columns[column_index]
            raw_values[:, column_index] = residual
            standardized_values[:, column_index] = z
            diagnostics_rows.append({"omic": omic, "feature": feature, **diagnostics})
            scaling_rows.append({"omic": omic, "feature": feature, "mean": mean, "sd": sd})
            if completed % 100 == 0 or completed == matrix.shape[1]:
                print(
                    f"[{omic}] residualized {completed}/{matrix.shape[1]} "
                    f"with {worker_count} worker(s)",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    raw = pd.DataFrame(raw_values, index=matrix.index, columns=matrix.columns)
    standardized = pd.DataFrame(standardized_values, index=matrix.index, columns=matrix.columns)
    raw.to_parquet(output_dir / f"{omic}_raw_residuals.parquet")
    standardized.to_parquet(output_dir / f"{omic}_residuals.parquet")
    diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostics.to_csv(output_dir / f"{omic}_fit_diagnostics.tsv", sep="\t", index=False)
    pd.DataFrame(scaling_rows).to_csv(output_dir / f"{omic}_scaling.tsv", sep="\t", index=False)
    write_json(design_info, output_dir / f"{omic}_design.json")
    return standardized, diagnostics


def residualize_all(
    config: dict[str, Any], max_features: int | None = None, workers: int | None = None
) -> None:
    for omic in OMICS:
        residualize_omic(config, omic, max_features=max_features, workers=workers)
