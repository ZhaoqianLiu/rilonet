from __future__ import annotations

import json
import re
import resource
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .common import (
    LAG_DIRECTIONS,
    NEAR_BLOCKS,
    load_config,
    normal_two_sided_p,
    output_root,
    safe_standardize,
)
from .matching import _pairwise_delta_days, build_all_pairs
from .modeling import _contrast, _fit_model, fit_edge_frame, model_all
from .prepare import prepare_inputs
from .residualize import residualize_all
from .screening import screen_all
from .simulate import generate_simulation_dataset


@dataclass(frozen=True)
class EdgeScenario:
    name: str
    beta_within: float
    beta_difference: float
    beta_between: float = 0.30
    beta_between_difference: float = 0.0
    beta_group: float = 0.20


DEFAULT_EDGE_SCENARIOS = (
    EdgeScenario("global_null", 0.0, 0.0, beta_between=0.0),
    EdgeScenario("shared_within", 0.40, 0.0),
    EdgeScenario("condition_only", 0.0, 0.40),
    EdgeScenario("weakened", 0.50, -0.40),
    EdgeScenario("strengthened", 0.10, 0.40),
    EdgeScenario("sign_reversed", 0.40, -0.80),
    EdgeScenario("between_only", 0.0, 0.0, beta_between=0.55),
    EdgeScenario(
        "between_rewiring_only",
        0.0,
        0.0,
        beta_between=0.10,
        beta_between_difference=0.60,
    ),
)


@dataclass(frozen=True)
class NearScenario:
    name: str
    rho_reference: float
    rho_condition: float


DEFAULT_NEAR_SCENARIOS = (
    NearScenario("near_null", 0.0, 0.0),
    NearScenario("near_shared", 0.40, 0.40),
    NearScenario("near_weakened", 0.55, 0.10),
    NearScenario("near_strengthened", 0.10, 0.55),
    NearScenario("near_reversed", 0.40, -0.40),
)


@dataclass(frozen=True)
class BaselineScenario:
    name: str
    direct_reference: float
    direct_difference: float
    persistence_reference: float
    persistence_condition: float
    predictor_baseline_correlation: float = 0.60


DEFAULT_BASELINE_SCENARIOS = (
    BaselineScenario("baseline_null_common_persistence", 0.0, 0.0, 0.60, 0.60),
    BaselineScenario("baseline_true_direct_difference", 0.10, 0.40, 0.60, 0.60),
    BaselineScenario("baseline_differential_persistence_only", 0.0, 0.0, 0.20, 0.80),
)


def _model_config(
    min_pairs: int = 30,
    min_subjects: int = 10,
    min_subjects_per_group: int = 5,
) -> dict[str, Any]:
    return {
        "metadata": {
            "group_levels": ["reference", "condition"],
            "reference_group": "reference",
        },
        "covariates": {"categorical": [], "numeric": [], "static": []},
        "modeling": {
            "min_pairs": min_pairs,
            "min_subjects": min_subjects,
            "min_subjects_per_group": min_subjects_per_group,
            "near_fit_both_orientations": True,
            "baseline_mode": "none",
            "max_iterations": 200,
            "optimizers": ["lbfgs", "powell", "cg"],
        },
    }


def _fixed_group_labels(
    n_subjects: int, condition_fraction: float, rng: np.random.Generator
) -> np.ndarray:
    n_condition = int(round(n_subjects * condition_fraction))
    n_condition = min(max(n_condition, 1), n_subjects - 1)
    labels = np.asarray(["condition"] * n_condition + ["reference"] * (n_subjects - n_condition))
    rng.shuffle(labels)
    return labels


def _ar1(length: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    values = np.zeros(length, dtype=float)
    innovation_sd = np.sqrt(max(1.0 - phi**2, 1e-8))
    values[0] = rng.normal()
    for index in range(1, length):
        values[index] = phi * values[index - 1] + rng.normal(scale=innovation_sd)
    return values


def simulate_edge_rows(
    scenario: EdgeScenario,
    rng: np.random.Generator,
    n_subjects: int,
    visits: int,
    condition_fraction: float,
    ar_phi: float,
    random_intercept_sd: float,
    residual_sd: float,
    irregular_visits: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    labels = _fixed_group_labels(n_subjects, condition_fraction, rng)
    participant_outcome_intercepts: dict[str, float] = {}
    for subject_index, group in enumerate(labels):
        subject = f"S{subject_index:04d}"
        count = int(rng.integers(max(3, visits - 3), visits + 1)) if irregular_visits else visits
        participant_mean = rng.normal(scale=0.80)
        within = _ar1(count, ar_phi, rng)
        participant_outcome_intercepts[subject] = rng.normal(scale=random_intercept_sd)
        for observation in range(count):
            rows.append(
                {
                    "subject_id": subject,
                    "group": str(group),
                    "predictor": participant_mean + within[observation],
                    "mid_time": observation * 2.0 + rng.uniform(-0.25, 0.25),
                    "abs_delta_days": abs(rng.normal(2.0, 1.5)),
                }
            )
    frame = pd.DataFrame(rows)
    participant_mean = frame.groupby("subject_id")["predictor"].transform("mean")
    within = frame["predictor"] - participant_mean
    within_z = safe_standardize(within)[0]
    between_z = safe_standardize(participant_mean)[0]
    condition = frame["group"].eq("condition").astype(float).to_numpy()
    outcome_raw = (
        scenario.beta_within * within_z
        + scenario.beta_difference * condition * within_z
        + scenario.beta_between * between_z
        + scenario.beta_between_difference * condition * between_z
        + scenario.beta_group * condition
        + frame["subject_id"].map(participant_outcome_intercepts).to_numpy(dtype=float)
        + rng.normal(scale=residual_sd, size=len(frame))
    )
    outcome_z, _, outcome_sd = safe_standardize(outcome_raw)
    frame["outcome"] = outcome_z
    truth = {
        "contrast": scenario.beta_difference / outcome_sd,
        "reference_effect": scenario.beta_within / outcome_sd,
        "condition_effect": (scenario.beta_within + scenario.beta_difference) / outcome_sd,
    }
    return frame, truth


def _fit_proposed(
    frame: pd.DataFrame, config: dict[str, Any], analysis_type: str = "near", baseline_mode: str = "none"
) -> dict[str, Any]:
    fitted, diagnostics = fit_edge_frame(frame, analysis_type, config, baseline_mode)
    reference = fitted[fitted["group"].eq("reference")].iloc[0]
    condition = fitted[fitted["group"].eq("condition")].iloc[0]
    estimate = float(condition["contrast_vs_reference"])
    se = float(condition["contrast_se"])
    return {
        "contrast_estimate": estimate,
        "contrast_se": se,
        "contrast_p": float(condition["contrast_p"]),
        "reference_effect": float(reference["effect"]),
        "condition_effect": float(condition["effect"]),
        "fit_method": str(condition["fit_method"]),
        "fit_converged": bool(condition["fit_converged"]),
    }


def _fit_naive(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    design = pd.DataFrame(index=frame.index)
    design["Intercept"] = 1.0
    design["predictor"] = safe_standardize(frame["predictor"])[0]
    indicator = frame["group"].eq("condition").astype(float)
    design["group_condition"] = indicator
    design["predictor_by_condition"] = design["predictor"] * indicator
    beta, covariance, diagnostics = _fit_model(
        frame["outcome"].to_numpy(dtype=float),
        design,
        frame["subject_id"].to_numpy(),
        config,
    )
    names = list(design.columns)
    interaction = np.zeros(len(names))
    interaction[names.index("predictor_by_condition")] = 1.0
    contrast, se, pvalue = _contrast(beta, covariance, interaction)
    reference_vector = np.zeros(len(names))
    reference_vector[names.index("predictor")] = 1.0
    condition_vector = reference_vector + interaction
    reference, _, _ = _contrast(beta, covariance, reference_vector)
    condition, _, _ = _contrast(beta, covariance, condition_vector)
    return {
        "contrast_estimate": contrast,
        "contrast_se": se,
        "contrast_p": pvalue,
        "reference_effect": reference,
        "condition_effect": condition,
        "fit_method": diagnostics["method"],
        "fit_converged": diagnostics["converged"],
    }


def _fit_ols_interaction(
    frame: pd.DataFrame, _config: dict[str, Any]
) -> dict[str, Any]:
    """Fit the usual row-independent OLS group-by-predictor interaction.

    This deliberately ignores participant clustering and within/between
    decomposition. It is a simple baseline, not a production fallback.
    """
    design = pd.DataFrame(index=frame.index)
    design["Intercept"] = 1.0
    design["predictor"] = safe_standardize(frame["predictor"])[0]
    indicator = frame["group"].eq("condition").astype(float)
    design["group_condition"] = indicator
    design["predictor_by_condition"] = design["predictor"] * indicator
    x = design.to_numpy(dtype=float)
    outcome = frame["outcome"].to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        raise RuntimeError("rank_deficient_ols_interaction")
    degrees_of_freedom = int(len(outcome) - rank)
    if degrees_of_freedom <= 0:
        raise RuntimeError("insufficient_ols_residual_degrees_of_freedom")
    beta, _, _, _ = np.linalg.lstsq(x, outcome, rcond=None)
    residual = outcome - x @ beta
    residual_variance = float(residual @ residual / degrees_of_freedom)
    covariance = residual_variance * np.linalg.inv(x.T @ x)
    names = list(design.columns)
    interaction = np.zeros(len(names))
    interaction[names.index("predictor_by_condition")] = 1.0
    contrast, se, pvalue = _contrast(beta, covariance, interaction)
    reference_vector = np.zeros(len(names))
    reference_vector[names.index("predictor")] = 1.0
    condition_vector = reference_vector + interaction
    reference, _, _ = _contrast(beta, covariance, reference_vector)
    condition, _, _ = _contrast(beta, covariance, condition_vector)
    return {
        "contrast_estimate": contrast,
        "contrast_se": se,
        "contrast_p": pvalue,
        "reference_effect": reference,
        "condition_effect": condition,
        "fit_method": "ols_independent_rows",
        "fit_converged": bool(
            np.isfinite(beta).all() and np.isfinite(covariance).all()
        ),
    }


def _summarize_draws(draws: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    draws = draws.copy()
    for column in (
        "contrast_estimate",
        "contrast_se",
        "contrast_p",
        "contrast_true",
        "fit_method",
    ):
        if column not in draws:
            draws[column] = np.nan
    rows = []
    for keys, subset in draws.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        valid = subset[np.isfinite(pd.to_numeric(subset["contrast_estimate"], errors="coerce"))].copy()
        estimate = pd.to_numeric(valid["contrast_estimate"], errors="coerce").to_numpy(dtype=float)
        truth = pd.to_numeric(valid["contrast_true"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(valid["contrast_se"], errors="coerce").to_numpy(dtype=float)
        pvalue = pd.to_numeric(valid["contrast_p"], errors="coerce").to_numpy(dtype=float)
        finite_p = np.isfinite(pvalue)
        finite_interval = np.isfinite(se) & np.isfinite(truth)
        rejection_count = int(np.sum(pvalue[finite_p] <= 0.05))
        rejection_n = int(np.sum(finite_p))
        if rejection_n:
            proportion = rejection_count / rejection_n
            denominator = 1.0 + 1.96**2 / rejection_n
            center = (proportion + 1.96**2 / (2 * rejection_n)) / denominator
            half_width = (
                1.96
                * np.sqrt(
                    proportion * (1 - proportion) / rejection_n
                    + 1.96**2 / (4 * rejection_n**2)
                )
                / denominator
            )
            rejection_low = max(0.0, center - half_width)
            rejection_high = min(1.0, center + half_width)
        else:
            rejection_low = rejection_high = np.nan
        null = bool(np.all(np.abs(truth) < 1e-12)) if len(truth) else False
        row.update(
            {
                "requested_replicates": int(subset["replicate"].nunique()),
                "successful_replicates": int(len(valid)),
                "success_rate": float(len(valid) / max(subset["replicate"].nunique(), 1)),
                "mean_true_contrast": float(np.mean(truth)) if len(truth) else np.nan,
                "mean_estimated_contrast": float(np.mean(estimate)) if len(estimate) else np.nan,
                "bias": float(np.mean(estimate - truth)) if len(estimate) else np.nan,
                "rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))) if len(estimate) else np.nan,
                "rejection_rate_0.05": float(np.mean(pvalue[finite_p] <= 0.05))
                if finite_p.any()
                else np.nan,
                "rejection_replicates": rejection_n,
                "rejection_rate_mc_ci_low": rejection_low,
                "rejection_rate_mc_ci_high": rejection_high,
                "metric_label": "type_I_error" if null else "power",
                "coverage_95": float(
                    np.mean(
                        (estimate[finite_interval] - 1.96 * se[finite_interval]
                         <= truth[finite_interval])
                        & (truth[finite_interval]
                           <= estimate[finite_interval] + 1.96 * se[finite_interval])
                    )
                )
                if finite_interval.any()
                else np.nan,
                "sign_recovery": float(np.mean(np.sign(estimate) == np.sign(truth)))
                if len(estimate) and not null
                else np.nan,
                "mixedlm_rate": float(
                    np.mean(valid["fit_method"].eq("mixedlm_random_intercept"))
                )
                if len(valid)
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _edge_method_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare each Tier-1 baseline with the proposed model by scenario.

    Replicates are generated once per scenario and shared by all methods, so
    these summary differences describe paired Monte Carlo experiments even
    though inferential uncertainty is reported separately in the main summary.
    """
    proposed = summary[summary["method"].eq("proposed")].copy()
    if proposed.empty:
        return pd.DataFrame()
    comparators = summary[~summary["method"].eq("proposed")].copy()
    if comparators.empty:
        return pd.DataFrame()
    merged = comparators.merge(
        proposed,
        on="scenario",
        how="inner",
        suffixes=("_comparator", "_proposed"),
        validate="many_to_one",
    )
    rows = []
    for _, row in merged.iterrows():
        metric_label = row["metric_label_proposed"]
        proposed_rejection = float(row["rejection_rate_0.05_proposed"])
        comparator_rejection = float(row["rejection_rate_0.05_comparator"])
        is_null = metric_label == "type_I_error"
        rows.append(
            {
                "scenario": row["scenario"],
                "metric_label": metric_label,
                "comparator": row["method_comparator"],
                "proposed_successful_replicates": int(
                    row["successful_replicates_proposed"]
                ),
                "comparator_successful_replicates": int(
                    row["successful_replicates_comparator"]
                ),
                "proposed_rejection_rate_0.05": proposed_rejection,
                "comparator_rejection_rate_0.05": comparator_rejection,
                "rejection_rate_difference_comparator_minus_proposed": (
                    comparator_rejection - proposed_rejection
                ),
                "proposed_type_I_error_abs_deviation_from_0.05": (
                    abs(proposed_rejection - 0.05) if is_null else np.nan
                ),
                "comparator_type_I_error_abs_deviation_from_0.05": (
                    abs(comparator_rejection - 0.05) if is_null else np.nan
                ),
                "proposed_bias": float(row["bias_proposed"]),
                "comparator_bias": float(row["bias_comparator"]),
                "absolute_bias_difference_comparator_minus_proposed": (
                    abs(float(row["bias_comparator"]))
                    - abs(float(row["bias_proposed"]))
                ),
                "proposed_rmse": float(row["rmse_proposed"]),
                "comparator_rmse": float(row["rmse_comparator"]),
                "rmse_difference_comparator_minus_proposed": (
                    float(row["rmse_comparator"])
                    - float(row["rmse_proposed"])
                ),
                "proposed_coverage_95": float(row["coverage_95_proposed"]),
                "comparator_coverage_95": float(row["coverage_95_comparator"]),
                "coverage_difference_comparator_minus_proposed": (
                    float(row["coverage_95_comparator"])
                    - float(row["coverage_95_proposed"])
                ),
                "proposed_sign_recovery": float(row["sign_recovery_proposed"]),
                "comparator_sign_recovery": float(row["sign_recovery_comparator"]),
                "sign_recovery_difference_comparator_minus_proposed": (
                    float(row["sign_recovery_comparator"])
                    - float(row["sign_recovery_proposed"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _edge_methods(names: Iterable[str] | None) -> list[tuple[str, Any]]:
    available = {
        "proposed": _fit_proposed,
        "naive_no_decomposition": _fit_naive,
        "naive_ols_interaction": _fit_ols_interaction,
    }
    requested = list(available) if names is None else [str(name) for name in names]
    if not requested:
        raise ValueError("edge_model.methods must contain at least one method")
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Unknown edge-model methods: {sorted(unknown)}")
    if len(requested) != len(set(requested)):
        raise ValueError("edge_model.methods must not contain duplicates")
    return [(name, available[name]) for name in requested]


def _effect_size_scenarios(section: dict[str, Any]) -> list[EdgeScenario] | None:
    values = section.get("effect_sizes")
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError("edge_model.effect_sizes must be a numeric sequence")
    effect_sizes = [float(value) for value in values]
    if not effect_sizes:
        raise ValueError("edge_model.effect_sizes must contain at least one value")
    if not np.isfinite(effect_sizes).all():
        raise ValueError("edge_model.effect_sizes must contain only finite values")
    if len(effect_sizes) != len(set(effect_sizes)):
        raise ValueError("edge_model.effect_sizes must not contain duplicates")

    beta_within = float(section.get("effect_curve_beta_within", 0.0))
    beta_between = float(section.get("effect_curve_beta_between", 0.30))
    beta_between_difference = float(
        section.get("effect_curve_beta_between_difference", 0.0)
    )
    beta_group = float(section.get("effect_curve_beta_group", 0.20))
    scenarios = []
    for value in effect_sizes:
        sign = "m" if value < 0 else "p"
        magnitude = f"{abs(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")
        scenarios.append(
            EdgeScenario(
                name=f"condition_only_effect_{sign}{magnitude}",
                beta_within=beta_within,
                beta_difference=value,
                beta_between=beta_between,
                beta_between_difference=beta_between_difference,
                beta_group=beta_group,
            )
        )
    return scenarios


def run_edge_model_study(
    output_dir: str | Path,
    replicates: int,
    seed: int,
    n_subjects: int = 120,
    visits: int = 12,
    condition_fraction: float = 0.65,
    ar_phi: float = 0.40,
    random_intercept_sd: float = 0.60,
    residual_sd: float = 1.0,
    scenarios: Iterable[EdgeScenario] = DEFAULT_EDGE_SCENARIOS,
    methods: Iterable[str] | None = None,
    checkpoint_every: int | None = None,
    resume: bool = False,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "edge_model_draws.checkpoint.tsv"
    config = _model_config()
    master = np.random.SeedSequence(seed)
    scenario_list = list(scenarios)
    method_list = _edge_methods(methods)
    seeds = master.spawn(len(scenario_list) * replicates)
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if resume and checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path, sep="\t", low_memory=False)
        required = {"scenario", "method", "replicate"}
        if not required.issubset(checkpoint.columns):
            raise ValueError(
                f"Invalid edge-model checkpoint; missing {sorted(required - set(checkpoint.columns))}"
            )
        rows = checkpoint.to_dict("records")
        completed = {
            (str(row["scenario"]), str(row["method"]), int(row["replicate"]))
            for row in rows
        }
    checkpoint_interval = int(checkpoint_every or 0)
    if checkpoint_interval < 0:
        raise ValueError("edge_model.checkpoint_every must be non-negative")

    def write_checkpoint() -> None:
        if not rows:
            return
        temporary = checkpoint_path.with_suffix(".tmp.tsv")
        pd.DataFrame(rows).to_csv(temporary, sep="\t", index=False)
        temporary.replace(checkpoint_path)

    seed_index = 0
    started = time.perf_counter()
    for scenario in scenario_list:
        for replicate in range(replicates):
            rng = np.random.default_rng(seeds[seed_index])
            seed_index += 1
            pending_methods = [
                (method, fitter)
                for method, fitter in method_list
                if (scenario.name, method, replicate) not in completed
            ]
            if not pending_methods:
                continue
            frame, truth = simulate_edge_rows(
                scenario,
                rng,
                n_subjects,
                visits,
                condition_fraction,
                ar_phi,
                random_intercept_sd,
                residual_sd,
            )
            for method, fitter in pending_methods:
                try:
                    fitted = fitter(frame, config)
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "method": method,
                            "replicate": replicate,
                            "contrast_true": truth["contrast"],
                            "status": "ok",
                            "error": "",
                            **fitted,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "method": method,
                            "replicate": replicate,
                            "contrast_true": truth["contrast"],
                            "status": "failed",
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )
            if checkpoint_interval and (replicate + 1) % checkpoint_interval == 0:
                write_checkpoint()
                print(
                    f"[edge-model] {scenario.name} replicate {replicate + 1}/{replicates}",
                    flush=True,
                )
        if checkpoint_interval:
            write_checkpoint()
            print(f"[edge-model] {scenario.name} complete", flush=True)
    draws = pd.DataFrame(rows)
    summary = _summarize_draws(draws, ["scenario", "method"])
    method_comparison = _edge_method_comparison(summary)
    elapsed = time.perf_counter() - started
    draws_path = output / "edge_model_draws.tsv"
    summary_path = output / "edge_model_summary.tsv"
    method_comparison_path = output / "edge_model_method_comparison.tsv"
    draws.to_csv(draws_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    method_comparison.to_csv(method_comparison_path, sep="\t", index=False)
    metadata_path = output / "edge_model_run.json"
    metadata = {
        "replicates_per_scenario": replicates,
        "scenarios": [asdict(value) for value in scenario_list],
        "n_subjects": n_subjects,
        "visits": visits,
        "condition_fraction": condition_fraction,
        "ar_phi": ar_phi,
        "comparison_methods": [name for name, _ in method_list],
        "paired_replicates_across_methods": len(method_list) > 1,
        "elapsed_seconds": elapsed,
        "fits_attempted": len(rows),
        "fits_per_second": len(rows) / elapsed if elapsed > 0 else None,
    }
    if "naive_ols_interaction" in metadata["comparison_methods"]:
        metadata["naive_ols_interaction"] = {
            "estimand": "condition-minus-reference predictor slope",
            "standard_error": "ordinary OLS treating repeated rows as independent",
            "participant_clustering": False,
            "within_between_decomposition": False,
        }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "draws": draws_path,
        "summary": summary_path,
        "method_comparison": method_comparison_path,
        "metadata": metadata_path,
    }


def simulate_near_rows(
    scenario: NearScenario,
    rng: np.random.Generator,
    n_subjects: int,
    visits: int,
    condition_fraction: float,
    participant_intercept_sd: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    labels = _fixed_group_labels(n_subjects, condition_fraction, rng)
    for subject_index, group in enumerate(labels):
        subject = f"S{subject_index:04d}"
        count = int(rng.integers(max(3, visits - 3), visits + 1))
        x_intercept = rng.normal(scale=participant_intercept_sd)
        y_intercept = rng.normal(scale=participant_intercept_sd)
        rho = scenario.rho_condition if group == "condition" else scenario.rho_reference
        covariance = np.asarray([[1.0, rho], [rho, 1.0]])
        within = rng.multivariate_normal([0.0, 0.0], covariance, size=count)
        for observation in range(count):
            rows.append(
                {
                    "subject_id": subject,
                    "group": str(group),
                    "x": x_intercept + within[observation, 0],
                    "y": y_intercept + within[observation, 1],
                    "mid_time": observation * 2.0,
                    "abs_delta_days": abs(rng.normal(2.0, 1.0)),
                }
            )
    data = pd.DataFrame(rows)
    x_within = data["x"] - data.groupby("subject_id")["x"].transform("mean")
    y_within = data["y"] - data.groupby("subject_id")["y"].transform("mean")
    x_within_sd = float(x_within.std(ddof=1))
    y_within_sd = float(y_within.std(ddof=1))
    x_sd = float(data["x"].std(ddof=1))
    y_sd = float(data["y"].std(ddof=1))
    difference = scenario.rho_condition - scenario.rho_reference
    return data, {
        "forward": difference * x_within_sd / y_sd,
        "reverse": difference * y_within_sd / x_sd,
        "combined": 0.5
        * (difference * x_within_sd / y_sd + difference * y_within_sd / x_sd),
        "symmetric_fisher_z": float(
            np.arctanh(np.clip(scenario.rho_condition, -0.999999, 0.999999))
            - np.arctanh(np.clip(scenario.rho_reference, -0.999999, 0.999999))
        ),
    }


def _near_orientation_fit(
    data: pd.DataFrame, config: dict[str, Any], predictor: str, outcome: str
) -> dict[str, Any]:
    frame = data[["subject_id", "group", "mid_time", "abs_delta_days"]].copy()
    frame["predictor"] = data[predictor].to_numpy(dtype=float)
    frame["outcome"] = safe_standardize(data[outcome])[0]
    return _fit_proposed(frame, config, analysis_type="near")


def _within_centered_near(data: pd.DataFrame) -> pd.DataFrame:
    work = data[["subject_id", "group", "x", "y"]].copy()
    work["subject_id"] = work["subject_id"].astype(str)
    work["group"] = work["group"].astype(str)
    work["x_within"] = work["x"] - work.groupby("subject_id")["x"].transform("mean")
    work["y_within"] = work["y"] - work.groupby("subject_id")["y"].transform("mean")
    finite = np.isfinite(work["x_within"]) & np.isfinite(work["y_within"])
    return work[finite].reset_index(drop=True)


def _fisher_correlation(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or float(x.std(ddof=1)) <= 0 or float(y.std(ddof=1)) <= 0:
        raise RuntimeError("insufficient_variation_for_symmetric_near_test")
    correlation = float(np.corrcoef(x.to_numpy(dtype=float), y.to_numpy(dtype=float))[0, 1])
    if not np.isfinite(correlation):
        raise RuntimeError("nonfinite_symmetric_near_correlation")
    return float(np.arctanh(np.clip(correlation, -0.999999, 0.999999)))


def _symmetric_near_effects(work: pd.DataFrame) -> tuple[float, float, float]:
    reference = work[work["group"].eq("reference")]
    condition = work[work["group"].eq("condition")]
    reference_effect = _fisher_correlation(reference["x_within"], reference["y_within"])
    condition_effect = _fisher_correlation(condition["x_within"], condition["y_within"])
    return reference_effect, condition_effect, condition_effect - reference_effect


def _fisher_from_centered_sums(x2: float, y2: float, xy: float) -> float:
    denominator = float(np.sqrt(max(x2, 0.0) * max(y2, 0.0)))
    if denominator <= 0:
        raise RuntimeError("insufficient_variation_for_symmetric_near_test")
    correlation = float(xy / denominator)
    return float(np.arctanh(np.clip(correlation, -0.999999, 0.999999)))


def _fit_symmetric_near(data: pd.DataFrame) -> dict[str, Any]:
    """Fit an orientation-invariant within-participant Fisher-z contrast.

    The standard error uses a group-stratified delete-one-participant jackknife,
    keeping all repeated observations from one participant in the same cluster.
    """
    work = _within_centered_near(data)
    work["x2"] = work["x_within"] ** 2
    work["y2"] = work["y_within"] ** 2
    work["xy"] = work["x_within"] * work["y_within"]
    subject_stats = (
        work.groupby(["group", "subject_id"], as_index=False)[["x2", "y2", "xy"]]
        .sum()
    )
    totals = subject_stats.groupby("group")[["x2", "y2", "xy"]].sum()
    if not {"reference", "condition"}.issubset(totals.index):
        raise RuntimeError("missing_group_for_symmetric_near_test")
    effects = {
        str(group): _fisher_from_centered_sums(row.x2, row.y2, row.xy)
        for group, row in totals.iterrows()
    }
    reference_effect = effects["reference"]
    condition_effect = effects["condition"]
    estimate = condition_effect - reference_effect
    variance = 0.0
    jackknife_replicates = 0
    for group, group_stats in subject_stats.groupby("group", sort=True):
        if len(group_stats) < 3:
            raise RuntimeError(f"insufficient_subjects_for_symmetric_near_{group}")
        leave_one = []
        total = totals.loc[group]
        for row in group_stats.itertuples(index=False):
            deleted_effect = _fisher_from_centered_sums(
                float(total["x2"] - row.x2),
                float(total["y2"] - row.y2),
                float(total["xy"] - row.xy),
            )
            if group == "condition":
                leave_one.append(deleted_effect - reference_effect)
            else:
                leave_one.append(condition_effect - deleted_effect)
        values = np.asarray(leave_one, dtype=float)
        variance += (len(values) - 1) / len(values) * float(
            np.sum((values - values.mean()) ** 2)
        )
        jackknife_replicates += len(values)
    se = float(np.sqrt(max(variance, 0.0)))
    pvalue = normal_two_sided_p(estimate / se) if se > 0 else np.nan
    return {
        "contrast_estimate": estimate,
        "contrast_se": se,
        "contrast_p": pvalue,
        "reference_effect": reference_effect,
        "condition_effect": condition_effect,
        "fit_method": "participant_cluster_jackknife",
        "fit_converged": bool(np.isfinite(estimate) and np.isfinite(se)),
        "jackknife_replicates": jackknife_replicates,
    }


def run_near_orientation_study(
    output_dir: str | Path,
    replicates: int,
    seed: int,
    n_subjects: int = 120,
    visits: int = 12,
    condition_fraction: float = 0.65,
    participant_intercept_sd: float = 0.60,
    scenarios: Iterable[NearScenario] = DEFAULT_NEAR_SCENARIOS,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = _model_config()
    scenario_list = list(scenarios)
    seeds = np.random.SeedSequence(seed).spawn(len(scenario_list) * replicates)
    seed_index = 0
    rows = []
    started = time.perf_counter()
    for scenario in scenario_list:
        for replicate in range(replicates):
            rng = np.random.default_rng(seeds[seed_index])
            seed_index += 1
            data, truth = simulate_near_rows(
                scenario,
                rng,
                n_subjects,
                visits,
                condition_fraction,
                participant_intercept_sd,
            )
            try:
                symmetric = _fit_symmetric_near(data)
                rows.append(
                    {
                        "scenario": scenario.name,
                        "method": "symmetric_fisher_z",
                        "replicate": replicate,
                        "contrast_true": truth["symmetric_fisher_z"],
                        "status": "ok",
                        "error": "",
                        **symmetric,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "scenario": scenario.name,
                        "method": "symmetric_fisher_z",
                        "replicate": replicate,
                        "contrast_true": truth["symmetric_fisher_z"],
                        "status": "failed",
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
            fitted: dict[str, dict[str, Any]] = {}
            for orientation, predictor, outcome in (
                ("forward", "x", "y"),
                ("reverse", "y", "x"),
            ):
                try:
                    result = _near_orientation_fit(data, config, predictor, outcome)
                    fitted[orientation] = result
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "method": orientation,
                            "replicate": replicate,
                            "contrast_true": truth[orientation],
                            "status": "ok",
                            "error": "",
                            **result,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "method": orientation,
                            "replicate": replicate,
                            "contrast_true": truth[orientation],
                            "status": "failed",
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )
            if set(fitted) == {"forward", "reverse"}:
                forward = fitted["forward"]
                reverse = fitted["reverse"]
                combined_estimate = 0.5 * (
                    forward["contrast_estimate"] + reverse["contrast_estimate"]
                )
                # The production rule combines p-values conservatively. A single
                # standard error is not defined for the averaged estimate.
                rows.append(
                    {
                        "scenario": scenario.name,
                        "method": "combined_max_p",
                        "replicate": replicate,
                        "contrast_true": truth["combined"],
                        "status": "ok",
                        "error": "",
                        "contrast_estimate": combined_estimate,
                        "contrast_se": np.nan,
                        "contrast_p": max(forward["contrast_p"], reverse["contrast_p"]),
                        "reference_effect": 0.5 * (
                            forward["reference_effect"] + reverse["reference_effect"]
                        ),
                        "condition_effect": 0.5 * (
                            forward["condition_effect"] + reverse["condition_effect"]
                        ),
                        "fit_method": "combined",
                        "fit_converged": True,
                        "orientation_sign_concordant": bool(
                            np.sign(forward["contrast_estimate"])
                            == np.sign(reverse["contrast_estimate"])
                        ),
                    }
                )
                rows.append(
                    {
                        "scenario": scenario.name,
                        "method": "combined_bonferroni_min_p",
                        "replicate": replicate,
                        "contrast_true": truth["combined"],
                        "status": "ok",
                        "error": "",
                        "contrast_estimate": combined_estimate,
                        "contrast_se": np.nan,
                        "contrast_p": min(
                            1.0,
                            2.0 * min(forward["contrast_p"], reverse["contrast_p"]),
                        ),
                        "reference_effect": 0.5 * (
                            forward["reference_effect"] + reverse["reference_effect"]
                        ),
                        "condition_effect": 0.5 * (
                            forward["condition_effect"] + reverse["condition_effect"]
                        ),
                        "fit_method": "combined",
                        "fit_converged": True,
                        "orientation_sign_concordant": bool(
                            np.sign(forward["contrast_estimate"])
                            == np.sign(reverse["contrast_estimate"])
                        ),
                    }
                )
    draws = pd.DataFrame(rows)
    summary = _summarize_draws(draws, ["scenario", "method"])
    # Coverage and mixed-model rates are undefined for P-value combination
    # rules because they do not have a covariance-aware combined standard error.
    p_combination = summary["method"].isin(
        ["combined_max_p", "combined_bonferroni_min_p"]
    )
    summary.loc[p_combination, ["coverage_95", "mixedlm_rate"]] = np.nan
    elapsed = time.perf_counter() - started
    draws_path = output / "near_orientation_draws.tsv"
    summary_path = output / "near_orientation_summary.tsv"
    draws.to_csv(draws_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    metadata_path = output / "near_orientation_run.json"
    metadata_path.write_text(
        json.dumps(
            {
                "replicates_per_scenario": replicates,
                "scenarios": [asdict(value) for value in scenario_list],
                "n_subjects": n_subjects,
                "visits": visits,
                "condition_fraction": condition_fraction,
                "participant_intercept_sd": participant_intercept_sd,
                "comparison_methods": [
                    "forward",
                    "reverse",
                    "combined_max_p",
                    "combined_bonferroni_min_p",
                    "symmetric_fisher_z",
                ],
                "symmetric_method": {
                    "estimand": "condition-minus-reference within-participant Fisher-z correlation",
                    "standard_error": "group-stratified delete-one-participant jackknife",
                },
                "elapsed_seconds": elapsed,
                "fits_attempted": int(len(scenario_list) * replicates * 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"draws": draws_path, "summary": summary_path, "metadata": metadata_path}


def simulate_baseline_rows(
    scenario: BaselineScenario,
    rng: np.random.Generator,
    n_subjects: int,
    visits: int,
    condition_fraction: float,
    random_intercept_sd: float,
    residual_sd: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    labels = _fixed_group_labels(n_subjects, condition_fraction, rng)
    outcome_intercepts: dict[str, float] = {}
    for subject_index, group in enumerate(labels):
        subject = f"S{subject_index:04d}"
        count = int(rng.integers(max(3, visits - 3), visits + 1))
        baseline_between = rng.normal(scale=0.70)
        baseline_within = _ar1(count, 0.35, rng)
        independent = _ar1(count, 0.35, rng)
        correlation = scenario.predictor_baseline_correlation
        predictor_within = correlation * baseline_within + np.sqrt(1 - correlation**2) * independent
        predictor_between = rng.normal(scale=0.70)
        outcome_intercepts[subject] = rng.normal(scale=random_intercept_sd)
        for observation in range(count):
            rows.append(
                {
                    "subject_id": subject,
                    "group": str(group),
                    "predictor": predictor_between + predictor_within[observation],
                    "baseline_outcome_raw": baseline_between + baseline_within[observation],
                    "mid_time": observation * 2.0,
                    "delta_days": 42.0 + rng.normal(scale=4.0),
                }
            )
    frame = pd.DataFrame(rows)
    x_mean = frame.groupby("subject_id")["predictor"].transform("mean")
    x_within_z = safe_standardize(frame["predictor"] - x_mean)[0]
    baseline_mean = frame.groupby("subject_id")["baseline_outcome_raw"].transform("mean")
    baseline_within_z = safe_standardize(frame["baseline_outcome_raw"] - baseline_mean)[0]
    condition = frame["group"].eq("condition").astype(float).to_numpy()
    persistence = np.where(
        condition == 1,
        scenario.persistence_condition,
        scenario.persistence_reference,
    )
    future_raw = (
        scenario.direct_reference * x_within_z
        + scenario.direct_difference * condition * x_within_z
        + persistence * baseline_within_z
        + frame["subject_id"].map(outcome_intercepts).to_numpy(dtype=float)
        + rng.normal(scale=residual_sd, size=len(frame))
    )
    future_z, _, outcome_sd = safe_standardize(future_raw)
    frame["future_outcome_raw"] = future_raw
    frame["outcome"] = future_z
    frame["abs_delta_days"] = np.abs(frame["delta_days"])
    return frame, {"contrast": scenario.direct_difference / outcome_sd}


def run_baseline_study(
    output_dir: str | Path,
    replicates: int,
    seed: int,
    n_subjects: int = 120,
    visits: int = 12,
    condition_fraction: float = 0.65,
    random_intercept_sd: float = 0.60,
    residual_sd: float = 1.0,
    scenarios: Iterable[BaselineScenario] = DEFAULT_BASELINE_SCENARIOS,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = _model_config()
    scenario_list = list(scenarios)
    seeds = np.random.SeedSequence(seed).spawn(len(scenario_list) * replicates)
    seed_index = 0
    rows = []
    started = time.perf_counter()
    for scenario in scenario_list:
        for replicate in range(replicates):
            rng = np.random.default_rng(seeds[seed_index])
            seed_index += 1
            frame, truth = simulate_baseline_rows(
                scenario,
                rng,
                n_subjects,
                visits,
                condition_fraction,
                random_intercept_sd,
                residual_sd,
            )
            for mode in ("none", "adjust", "change"):
                work = frame.copy()
                contrast_true = truth["contrast"]
                if mode == "change":
                    changed = work["future_outcome_raw"] - work["baseline_outcome_raw"]
                    changed_z, _, changed_sd = safe_standardize(changed)
                    work["outcome"] = changed_z
                    contrast_true = scenario.direct_difference / changed_sd
                try:
                    fitted = _fit_proposed(work, config, analysis_type="lagged", baseline_mode=mode)
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "method": mode,
                            "replicate": replicate,
                            "contrast_true": contrast_true,
                            "status": "ok",
                            "error": "",
                            **fitted,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "method": mode,
                            "replicate": replicate,
                            "contrast_true": contrast_true,
                            "status": "failed",
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )
    draws = pd.DataFrame(rows)
    summary = _summarize_draws(draws, ["scenario", "method"])
    elapsed = time.perf_counter() - started
    draws_path = output / "baseline_draws.tsv"
    summary_path = output / "baseline_summary.tsv"
    draws.to_csv(draws_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    metadata_path = output / "baseline_run.json"
    metadata_path.write_text(
        json.dumps(
            {
                "replicates_per_scenario": replicates,
                "scenarios": [asdict(value) for value in scenario_list],
                "n_subjects": n_subjects,
                "visits": visits,
                "elapsed_seconds": elapsed,
                "fits_attempted": int(len(scenario_list) * replicates * 3),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"draws": draws_path, "summary": summary_path, "metadata": metadata_path}


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
    config["validation"]["q_threshold"] = q_threshold
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
    # In YAML, an unquoted `null` is parsed as None. Accept that natural spelling
    # and normalize it to the intended null-network scenario.
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
                "non_planted_call_interpretation": (
                    "false discoveries only under the global-null mode; under differential "
                    "modes they may include induced marginal associations"
                ),
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


def _select_scenarios(values: Iterable[Any], names: Iterable[str] | None) -> list[Any]:
    available = {str(value.name): value for value in values}
    if names is None:
        return list(available.values())
    requested = [str(name) for name in names]
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Unknown simulation scenarios: {sorted(unknown)}")
    return [available[name] for name in requested]


def _positive_int(section: dict[str, Any], key: str, default: int) -> int:
    value = int(section.get(key, default))
    if value < 1:
        raise ValueError(f"{key} must be at least 1")
    return value


def run_simulation_study(study_config_path: str | Path) -> dict[str, Path]:
    """Run the configured simulation program and write a machine-readable manifest."""
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

    edge = dict(study.get("edge_model", {}))
    if bool(edge.get("enabled", True)):
        started = time.perf_counter()
        effect_scenarios = _effect_size_scenarios(edge)
        result = run_edge_model_study(
            output / "01_edge_model",
            replicates=_positive_int(edge, "replicates", 1000),
            seed=derived_seeds[0],
            n_subjects=_positive_int(edge, "n_subjects", 120),
            visits=_positive_int(edge, "visits", 12),
            condition_fraction=float(edge.get("condition_fraction", 0.65)),
            ar_phi=float(edge.get("ar_phi", 0.40)),
            random_intercept_sd=float(edge.get("random_intercept_sd", 0.60)),
            residual_sd=float(edge.get("residual_sd", 1.0)),
            scenarios=(
                effect_scenarios
                if effect_scenarios is not None
                else _select_scenarios(DEFAULT_EDGE_SCENARIOS, edge.get("scenarios"))
            ),
            methods=edge.get("methods"),
            checkpoint_every=int(edge.get("checkpoint_every", 0)),
            resume=bool(edge.get("resume", False)),
        )
        outputs.update({f"edge_model_{key}": value for key, value in result.items()})
        timings["edge_model_seconds"] = time.perf_counter() - started

    near = dict(study.get("near_orientation", {}))
    if bool(near.get("enabled", True)):
        started = time.perf_counter()
        result = run_near_orientation_study(
            output / "02_near_orientation",
            replicates=_positive_int(near, "replicates", 1000),
            seed=derived_seeds[1],
            n_subjects=_positive_int(near, "n_subjects", 120),
            visits=_positive_int(near, "visits", 12),
            condition_fraction=float(near.get("condition_fraction", 0.65)),
            participant_intercept_sd=float(near.get("participant_intercept_sd", 0.60)),
            scenarios=_select_scenarios(DEFAULT_NEAR_SCENARIOS, near.get("scenarios")),
        )
        outputs.update({f"near_orientation_{key}": value for key, value in result.items()})
        timings["near_orientation_seconds"] = time.perf_counter() - started

    baseline = dict(study.get("baseline", {}))
    if bool(baseline.get("enabled", True)):
        started = time.perf_counter()
        result = run_baseline_study(
            output / "03_baseline",
            replicates=_positive_int(baseline, "replicates", 1000),
            seed=derived_seeds[2],
            n_subjects=_positive_int(baseline, "n_subjects", 120),
            visits=_positive_int(baseline, "visits", 12),
            condition_fraction=float(baseline.get("condition_fraction", 0.65)),
            random_intercept_sd=float(baseline.get("random_intercept_sd", 0.60)),
            residual_sd=float(baseline.get("residual_sd", 1.0)),
            scenarios=_select_scenarios(DEFAULT_BASELINE_SCENARIOS, baseline.get("scenarios")),
        )
        outputs.update({f"baseline_{key}": value for key, value in result.items()})
        timings["baseline_seconds"] = time.perf_counter() - started

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
                "end_to_end.split_residualization_scope=screening_subjects is not "
                "implemented; refusing to run a mislabeled strict-split simulation"
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
                    "edge_model": derived_seeds[0],
                    "near_orientation": derived_seeds[1],
                    "baseline": derived_seeds[2],
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
