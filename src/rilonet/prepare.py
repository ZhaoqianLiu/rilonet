from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    OMICS,
    ensure_dir,
    output_root,
    read_matrix,
    resolve_path,
    write_json,
)


def _clean_identifier(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _matrix_lookup_id(sample_id: str, omic_cfg: dict[str, Any]) -> str:
    value = str(sample_id)
    suffix = str(omic_cfg.get("sample_id_strip_suffix", ""))
    if suffix and value.endswith(suffix):
        value = value[: -len(suffix)]
    for rule in omic_cfg.get("sample_id_replacements", []):
        value = re.sub(str(rule["pattern"]), str(rule.get("replacement", "")), value)
    return value


def _canonical_id(value: str, rules: list[dict[str, Any]]) -> str:
    result = str(value)
    for rule in rules:
        result = re.sub(str(rule["pattern"]), str(rule.get("replacement", "")), result)
    return result


def _consistent_row(frame: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    if frame.empty:
        return None
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].dropna().astype(str).str.strip()
        values = values[values.ne("")].unique()
        if len(values) > 1:
            return None
    return frame.iloc[0]


def _clr_transform(
    matrix: pd.DataFrame, delta: float | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = matrix.to_numpy(dtype=float)
    positive_row = np.isfinite(values).all(axis=1) & (np.nansum(values, axis=1) > 0)
    values = values[positive_row]
    sample_ids = matrix.index[positive_row]
    closure = values / values.sum(axis=1, keepdims=True)
    feature_count = closure.shape[1]
    replacement = float(delta) if delta is not None else 1.0 / (feature_count**2)
    zero_count = (closure == 0).sum(axis=1)
    if np.any(zero_count * replacement >= 1):
        raise ValueError("Multiplicative zero-replacement delta is too large")
    replaced = np.where(
        closure == 0,
        replacement,
        closure * (1.0 - zero_count[:, None] * replacement),
    )
    logged = np.log(replaced)
    clr = logged - logged.mean(axis=1, keepdims=True)
    result = pd.DataFrame(clr, index=sample_ids, columns=matrix.columns)
    return result, {
        "zero_replacement_delta": replacement,
        "profiles_excluded_nonpositive_or_nonfinite": int((~positive_row).sum()),
    }


def _pqn_log2_transform(matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = matrix.astype(float).copy()
    reference = values.replace(0, np.nan).median(axis=0, skipna=True)
    valid_reference = np.isfinite(reference.to_numpy(dtype=float)) & (
        reference.to_numpy(dtype=float) > 0
    )
    if not valid_reference.any():
        raise ValueError(
            "No positive metabolite reference values are available for PQN"
        )
    quotients = values.loc[:, valid_reference].div(
        reference.loc[valid_reference], axis=1
    )
    factors = quotients.median(axis=1, skipna=True)
    valid_profile = np.isfinite(factors.to_numpy(dtype=float)) & (
        factors.to_numpy(dtype=float) > 0
    )
    normalized = values.loc[valid_profile].div(factors.loc[valid_profile], axis=0)
    positive = normalized.where(normalized > 0).min().min()
    if not np.isfinite(positive) or positive <= 0:
        raise ValueError("No positive PQN-normalized intensity is available")
    pseudocount = float(positive / 2.0)
    transformed = np.log2(normalized + pseudocount)
    return transformed, {
        "pseudocount": pseudocount,
        "profiles_excluded_invalid_pqn_factor": int((~valid_profile).sum()),
    }


def _preprocess_matrix(
    matrix: pd.DataFrame,
    omic: str,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode = str(cfg.get("mode", "raw"))
    report: dict[str, Any] = {
        "mode": mode,
        "profiles_input": int(matrix.shape[0]),
        "features_input": int(matrix.shape[1]),
    }
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.loc[~matrix.isna().all(axis=1)].copy()
    if mode == "transformed":
        transformed = matrix
    elif mode == "raw":
        prevalence_threshold = float(cfg.get("prevalence", 0.0))
        prevalence = (matrix > 0).sum(axis=0) / len(matrix)
        matrix = matrix.loc[:, prevalence >= prevalence_threshold].copy()
        if matrix.shape[1] == 0:
            raise ValueError(f"No {omic} features remain after prevalence filtering")
        report["prevalence_threshold"] = prevalence_threshold
        report["features_after_prevalence"] = int(matrix.shape[1])
        transform = str(
            cfg.get("transform", "clr" if omic in {"MG", "MT"} else "pqn_log2")
        )
        if transform == "clr":
            transformed, details = _clr_transform(
                matrix, cfg.get("zero_replacement_delta")
            )
        elif transform == "pqn_log2":
            transformed, details = _pqn_log2_transform(matrix)
        elif transform == "none":
            transformed, details = matrix, {}
        else:
            raise ValueError(f"Unsupported transform for {omic}: {transform}")
        report["transform"] = transform
        report.update(details)
        iqr_percentile = cfg.get("iqr_bottom_percentile")
        if iqr_percentile is not None:
            iqr = transformed.quantile(0.75) - transformed.quantile(0.25)
            cutoff = float(iqr.quantile(float(iqr_percentile)))
            transformed = transformed.loc[:, iqr > cutoff].copy()
            report["iqr_bottom_percentile"] = float(iqr_percentile)
            report["iqr_cutoff"] = cutoff
    else:
        raise ValueError("preprocessing.mode must be raw or transformed")

    transformed = transformed.dropna(axis=0, how="all").dropna(axis=1, how="all")
    report["profiles_output"] = int(transformed.shape[0])
    report["features_output"] = int(transformed.shape[1])
    return transformed, report


def prepare_inputs(config: dict[str, Any]) -> pd.DataFrame:
    """Map profiles to metadata, preprocess each omic, and write canonical inputs."""
    metadata_cfg = config["metadata"]
    columns = metadata_cfg["columns"]
    metadata_path = resolve_path(config, config["paths"]["metadata"])
    metadata = pd.read_csv(
        metadata_path, sep=str(metadata_cfg.get("sep", ",")), low_memory=False
    )
    required_source = {
        columns[key]
        for key in ("sample_id", "participant_id", "omic", "group", "study_time")
    }
    missing = required_source - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing configured columns: {sorted(missing)}")

    omic_mapping = {
        str(k): str(v) for k, v in metadata_cfg.get("omic_mapping", {}).items()
    }
    group_mapping = {
        str(k): str(v) for k, v in metadata_cfg.get("group_mapping", {}).items()
    }
    metadata = metadata.copy()
    metadata["_sample_key"] = metadata[columns["sample_id"]].map(_clean_identifier)
    metadata["_omic"] = metadata[columns["omic"]].map(
        lambda value: omic_mapping.get(
            _clean_identifier(value), _clean_identifier(value)
        )
    )
    metadata["_subject"] = metadata[columns["participant_id"]].map(_clean_identifier)
    metadata["_group"] = metadata[columns["group"]].map(
        lambda value: group_mapping.get(
            _clean_identifier(value), _clean_identifier(value)
        )
    )
    canonical_column = columns.get("canonical_biosample_id")
    canonical_rules = list(metadata_cfg.get("canonical_id_replacements", []))
    if canonical_column and canonical_column in metadata:
        metadata["_canonical"] = metadata[canonical_column].map(_clean_identifier)
    else:
        metadata["_canonical"] = metadata["_sample_key"].map(
            lambda value: _canonical_id(value, canonical_rules)
        )

    levels = set(str(value) for value in metadata_cfg["group_levels"])
    static_covariates = set(config.get("covariates", {}).get("static", []))
    all_covariates = list(
        dict.fromkeys(
            config.get("covariates", {}).get("categorical", [])
            + config.get("covariates", {}).get("numeric", [])
        )
    )
    missing_covariates = set(all_covariates) - set(metadata.columns)
    if missing_covariates:
        raise ValueError(
            f"Metadata is missing configured covariates: {sorted(missing_covariates)}"
        )
    date_column = columns.get("collection_date")
    time_unit_days = float(metadata_cfg.get("study_time_unit_days", 7.0))
    output_dir = ensure_dir(output_root(config) / "01_prepared")
    matrix_dir = ensure_dir(output_dir / "matrices")
    rows: list[dict[str, Any]] = []
    matrices: dict[str, pd.DataFrame] = {}
    reports: dict[str, Any] = {}

    for omic in OMICS:
        omic_cfg = config["omics"][omic]
        matrix = read_matrix(
            resolve_path(config, omic_cfg["path"]),
            orientation=str(omic_cfg.get("orientation", "profiles_by_features")),
            sep=str(omic_cfg.get("sep", "\t")),
        )
        mapped_rows: list[dict[str, Any]] = []
        for sample_id in matrix.index:
            lookup = _matrix_lookup_id(sample_id, omic_cfg)
            canonical = _canonical_id(lookup, canonical_rules)
            exact = metadata[
                metadata["_omic"].eq(omic) & metadata["_sample_key"].eq(lookup)
            ]
            selected = _consistent_row(
                exact, ["_subject", "_group", columns["study_time"]]
            )
            method = "exact"
            if selected is None and bool(
                metadata_cfg.get("allow_canonical_fallback", False)
            ):
                fallback = metadata[metadata["_canonical"].eq(canonical)]
                selected = _consistent_row(
                    fallback, ["_subject", "_group", columns["study_time"]]
                )
                method = (
                    "canonical_fallback"
                    if selected is not None
                    else "unmapped_or_ambiguous"
                )
            elif selected is None:
                method = "unmapped_or_ambiguous"

            if selected is None:
                subject = group = ""
                study_time = np.nan
                collection_date = pd.NaT
                metadata_canonical = canonical
            else:
                subject = _clean_identifier(selected["_subject"])
                group = _clean_identifier(selected["_group"])
                if not group and subject:
                    subject_groups = metadata.loc[
                        metadata["_subject"].eq(subject), "_group"
                    ]
                    subject_groups = subject_groups[
                        subject_groups.astype(str).str.strip().ne("")
                    ].unique()
                    if len(subject_groups) == 1:
                        group = _clean_identifier(subject_groups[0])
                study_time = pd.to_numeric(
                    selected.get(columns["study_time"]), errors="coerce"
                )
                collection_date = (
                    pd.to_datetime(selected.get(date_column), errors="coerce")
                    if date_column
                    else pd.NaT
                )
                metadata_canonical = (
                    _clean_identifier(selected["_canonical"]) or canonical
                )

            mapped: dict[str, Any] = {
                "sample_id": str(sample_id),
                "metadata_sample_id": lookup,
                "omic": omic,
                "subject_id": subject,
                "group": group,
                "study_time": study_time,
                "collection_date": collection_date,
                "canonical_biosample_id": metadata_canonical,
                "mapping_method": method,
            }
            for covariate in all_covariates:
                value = (
                    selected.get(covariate, np.nan) if selected is not None else np.nan
                )
                if (
                    (pd.isna(value) or str(value).strip() == "")
                    and covariate in static_covariates
                    and subject
                ):
                    subject_values = metadata.loc[
                        metadata["_subject"].eq(subject), covariate
                    ]
                    subject_values = subject_values.dropna()
                    value = subject_values.iloc[0] if len(subject_values) else np.nan
                mapped[covariate] = value
            mapped["valid_metadata"] = bool(
                selected is not None
                and subject
                and group in levels
                and (pd.notna(study_time) or pd.notna(collection_date))
            )
            mapped["exclusion_reason"] = "" if mapped["valid_metadata"] else method
            mapped_rows.append(mapped)

        omic_manifest = pd.DataFrame(mapped_rows).set_index("sample_id")
        metadata_valid_ids = omic_manifest.index[omic_manifest["valid_metadata"]]
        matrix = matrix.loc[matrix.index.intersection(metadata_valid_ids)].copy()
        processed, process_report = _preprocess_matrix(
            matrix, omic, omic_cfg.get("preprocessing", {})
        )
        matrices[omic] = processed
        omitted_after_processing = set(matrix.index) - set(processed.index)
        if omitted_after_processing:
            omic_manifest.loc[list(omitted_after_processing), "exclusion_reason"] = (
                "excluded_during_preprocessing"
            )
        omic_manifest["valid_for_analysis"] = omic_manifest.index.isin(processed.index)
        omic_manifest = omic_manifest.reset_index()
        rows.extend(omic_manifest.to_dict("records"))
        processed.index.name = "sample_id"
        processed.to_parquet(matrix_dir / f"{omic}.parquet")
        processed.to_csv(matrix_dir / f"{omic}.tsv.gz", sep="\t", compression="gzip")
        reports[omic] = {
            **process_report,
            "metadata_mapping_methods": dict(Counter(omic_manifest["mapping_method"])),
            "participants_output": int(
                omic_manifest.loc[
                    omic_manifest["valid_for_analysis"], "subject_id"
                ].nunique()
            ),
        }

    manifest = pd.DataFrame(rows)
    valid = manifest[manifest["valid_for_analysis"]].copy()
    group_counts = valid.groupby("subject_id")["group"].nunique()
    inconsistent = group_counts[group_counts > 1]
    if len(inconsistent):
        raise ValueError(
            f"Participant-level group is inconsistent for: {inconsistent.index[:5].tolist()}"
        )
    manifest["collection_date"] = pd.to_datetime(
        manifest["collection_date"], errors="coerce"
    )
    manifest["match_time_days"] = manifest["collection_date"].map(
        lambda value: value.timestamp() / 86400.0 if pd.notna(value) else np.nan
    )
    manifest["study_time_days"] = (
        pd.to_numeric(manifest["study_time"], errors="coerce") * time_unit_days
    )
    manifest.to_csv(output_dir / "sample_manifest.tsv", sep="\t", index=False)
    manifest.loc[~manifest["valid_for_analysis"]].to_csv(
        output_dir / "excluded_profiles.tsv", sep="\t", index=False
    )
    write_json(
        {
            "metadata": str(metadata_path),
            "omics": reports,
            "total_valid_profiles": int(manifest["valid_for_analysis"].sum()),
            "participants": int(valid["subject_id"].nunique()),
            "participants_by_group": {
                str(k): int(v)
                for k, v in valid.groupby("subject_id")["group"]
                .first()
                .value_counts()
                .items()
            },
        },
        output_dir / "preparation_qc.json",
    )
    return manifest
