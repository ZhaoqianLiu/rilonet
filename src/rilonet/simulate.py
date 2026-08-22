from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


OMICS = ("MG", "MT", "MB")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _feature_names(dimensions: dict[str, int]) -> dict[str, list[str]]:
    return {
        omic: [f"{omic}_{index:03d}" for index in range(dimensions[omic])]
        for omic in OMICS
    }


def generate_simulation_dataset(
    output_dir: str | Path,
    seed: int = 20260715,
    n_subjects: int = 120,
    n_visits: int = 18,
    condition_fraction: float = 0.65,
    dimensions: dict[str, int] | None = None,
    differential_truth: bool = True,
    signal_scale: float = 1.0,
    innovation_sd: float = 0.70,
    observation_probabilities: dict[str, float] | None = None,
    metabolite_noise_sd: float = 0.12,
) -> dict[str, Path]:
    """Generate an irregular longitudinal MG-MT-MB dataset with known edges.

    Latent Gaussian trajectories contain group-dependent contemporaneous shocks
    and three-visit directional effects. MG and MT are emitted as sequencing
    counts; MB is emitted as positive intensities with nondetect zeros.
    """
    root = Path(output_dir).resolve()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    signal_scale = float(signal_scale)
    innovation_sd = float(innovation_sd)
    metabolite_noise_sd = float(metabolite_noise_sd)
    if not np.isfinite(signal_scale) or signal_scale <= 0:
        raise ValueError("signal_scale must be a positive finite number")
    if not np.isfinite(innovation_sd) or innovation_sd <= 0:
        raise ValueError("innovation_sd must be a positive finite number")
    if not np.isfinite(metabolite_noise_sd) or metabolite_noise_sd < 0:
        raise ValueError("metabolite_noise_sd must be a non-negative finite number")
    observation_probability = {
        "MG": 0.90,
        "MT": 0.68,
        "MB": 0.55,
        **dict(observation_probabilities or {}),
    }
    unknown_omics = set(observation_probability) - set(OMICS)
    if unknown_omics:
        raise ValueError(
            f"Unknown omics in observation_probabilities: {sorted(unknown_omics)}"
        )
    observation_probability = {
        omic: float(observation_probability[omic]) for omic in OMICS
    }
    if any(
        not np.isfinite(probability) or not 0 < probability <= 1
        for probability in observation_probability.values()
    ):
        raise ValueError("Each observation probability must be in (0, 1]")
    dimensions = dimensions or {"MG": 16, "MT": 20, "MB": 12}
    if any(dimensions.get(omic, 0) < 6 for omic in OMICS):
        raise ValueError(
            "Each omics layer needs at least six features for the default truth design"
        )
    names = _feature_names(dimensions)
    offsets: dict[str, int] = {}
    total = 0
    for omic in OMICS:
        offsets[omic] = total
        total += dimensions[omic]

    def index(omic: str, feature_index: int) -> int:
        return offsets[omic] + feature_index

    near_edges = [
        ("MG", 0, "MT", 0, "weakened", 0.70, 0.10),
        ("MG", 1, "MB", 0, "sign_reversed", 0.65, -0.65),
        ("MT", 1, "MB", 1, "strengthened", 0.15, 0.70),
        ("MG", 4, "MT", 4, "shared", 0.50, 0.50),
    ]
    lag_edges = [
        ("MG", 2, "MT", 2, "weakened", 0.55, 0.10),
        ("MT", 3, "MB", 2, "sign_reversed", 0.50, -0.50),
        ("MB", 3, "MG", 3, "strengthened", 0.10, 0.50),
        ("MG", 5, "MB", 4, "shared", 0.35, 0.35),
    ]
    if not differential_truth:
        near_edges = [
            (x_omic, xi, y_omic, yi, "shared", reference, reference)
            for x_omic, xi, y_omic, yi, _, reference, _ in near_edges
        ]
        lag_edges = [
            (x_omic, xi, y_omic, yi, "shared", reference, reference)
            for x_omic, xi, y_omic, yi, _, reference, _ in lag_edges
        ]
    near_edges = [
        (
            x_omic,
            xi,
            y_omic,
            yi,
            category,
            signal_scale * reference,
            signal_scale * condition,
        )
        for x_omic, xi, y_omic, yi, category, reference, condition in near_edges
    ]
    lag_edges = [
        (
            x_omic,
            xi,
            y_omic,
            yi,
            category,
            signal_scale * reference,
            signal_scale * condition,
        )
        for x_omic, xi, y_omic, yi, category, reference, condition in lag_edges
    ]
    truth_rows = []
    for x_omic, xi, y_omic, yi, category, reference, condition in near_edges:
        truth_rows.append(
            {
                "analysis_type": "near",
                "source_omic": x_omic,
                "source_feature": names[x_omic][xi],
                "target_omic": y_omic,
                "target_feature": names[y_omic][yi],
                "lag_visits": 0,
                "category": category,
                "reference_parameter": reference,
                "condition_parameter": condition,
                "is_differential": category != "shared",
            }
        )
    for x_omic, xi, y_omic, yi, category, reference, condition in lag_edges:
        truth_rows.append(
            {
                "analysis_type": "lagged",
                "source_omic": x_omic,
                "source_feature": names[x_omic][xi],
                "target_omic": y_omic,
                "target_feature": names[y_omic][yi],
                "lag_visits": 3,
                "category": category,
                "reference_parameter": reference,
                "condition_parameter": condition,
                "is_differential": category != "shared",
            }
        )

    matrices: dict[str, list[np.ndarray]] = {omic: [] for omic in OMICS}
    matrix_ids: dict[str, list[str]] = {omic: [] for omic in OMICS}
    metadata_rows = []
    start_date = pd.Timestamp("2020-01-01")
    visit_days = np.arange(n_visits) * 14
    for subject_index in range(n_subjects):
        subject = f"S{subject_index:04d}"
        group = "condition" if rng.random() < condition_fraction else "reference"
        group_index = int(group == "condition")
        # Add moderate site imbalance for the covariate-adjustment scenario.
        site_probability = 0.65 if group == "condition" else 0.35
        site = "Site_B" if rng.random() < site_probability else "Site_A"
        age = float(np.clip(rng.normal(42 + 4 * group_index, 12), 18, 80))
        sex = rng.choice(["Female", "Male"])
        intercept = rng.normal(0, 0.65, size=total)
        latent = np.zeros((n_visits, total), dtype=float)

        for visit in range(n_visits):
            innovations = rng.normal(0, innovation_sd, size=total)
            for (
                x_omic,
                xi,
                y_omic,
                yi,
                _,
                reference_loading,
                condition_loading,
            ) in near_edges:
                shock = rng.normal()
                target_loading = condition_loading if group_index else reference_loading
                innovations[index(x_omic, xi)] += 0.70 * shock
                innovations[index(y_omic, yi)] += target_loading * shock
            previous = latent[visit - 1] if visit else intercept
            latent[visit] = intercept + 0.40 * (previous - intercept) + innovations
            latent[visit] += (visit / max(n_visits - 1, 1)) * rng.normal(
                0, 0.15, size=total
            )
            if visit >= 3:
                for (
                    x_omic,
                    xi,
                    y_omic,
                    yi,
                    _,
                    reference_beta,
                    condition_beta,
                ) in lag_edges:
                    beta = condition_beta if group_index else reference_beta
                    source_index = index(x_omic, xi)
                    target_index = index(y_omic, yi)
                    latent[visit, target_index] += beta * (
                        latent[visit - 3, source_index] - intercept[source_index]
                    )

        for visit, nominal_day in enumerate(visit_days):
            biosample = f"{subject}_V{visit:02d}"
            exposure = "Yes" if rng.random() < (0.08 + 0.14 * group_index) else "No"
            for omic in OMICS:
                if rng.random() > observation_probability[omic]:
                    continue
                jitter = int(rng.integers(-3, 4))
                day = int(nominal_day + jitter)
                sample_id = f"{subject}_V{visit:02d}_{omic}"
                vector = latent[visit, offsets[omic] : offsets[omic] + dimensions[omic]]
                if omic in {"MG", "MT"}:
                    depth = int(rng.lognormal(mean=np.log(15000), sigma=0.35))
                    observed = rng.multinomial(max(depth, 1000), _softmax(vector))
                else:
                    observed = np.exp2(
                        8.0
                        + 0.75 * vector
                        + rng.normal(0, metabolite_noise_sd, size=len(vector))
                    )
                    observed[rng.random(len(vector)) < 0.06] = 0.0
                matrices[omic].append(observed)
                matrix_ids[omic].append(sample_id)
                metadata_rows.append(
                    {
                        "sample_id": sample_id,
                        "participant_id": subject,
                        "omic": omic,
                        "group": group,
                        "collection_date": start_date + pd.to_timedelta(day, unit="D"),
                        "study_week": day / 7.0,
                        "biosample_id": biosample,
                        "site": site,
                        "age": age,
                        "sex": sex,
                        "exposure": exposure,
                    }
                )

    metadata_path = data_dir / "metadata.tsv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, sep="\t", index=False)
    matrix_paths = {}
    for omic in OMICS:
        frame = pd.DataFrame(
            matrices[omic], index=matrix_ids[omic], columns=names[omic]
        )
        frame.index.name = "sample_id"
        path = data_dir / f"{omic}.tsv"
        frame.to_csv(path, sep="\t")
        matrix_paths[omic] = path
    truth_path = data_dir / "truth.tsv"
    pd.DataFrame(truth_rows).to_csv(truth_path, sep="\t", index=False)

    config = {
        "project_root": ".",
        "compute": {
            "workers": 1,
            "blas_threads": 1,
            "tasks_per_worker": 4,
            "start_method": "spawn",
        },
        "paths": {"metadata": "data/metadata.tsv", "output_dir": "analysis"},
        "metadata": {
            "sep": "\t",
            "columns": {
                "sample_id": "sample_id",
                "participant_id": "participant_id",
                "omic": "omic",
                "group": "group",
                "collection_date": "collection_date",
                "study_time": "study_week",
                "canonical_biosample_id": "biosample_id",
            },
            "omic_mapping": {"MG": "MG", "MT": "MT", "MB": "MB"},
            "group_levels": ["reference", "condition"],
            "reference_group": "reference",
            "group_mapping": {},
            "study_time_unit_days": 7,
            "allow_canonical_fallback": False,
            "canonical_id_replacements": [],
        },
        "omics": {
            "MG": {
                "path": "data/MG.tsv",
                "orientation": "profiles_by_features",
                "preprocessing": {
                    "mode": "raw",
                    "prevalence": 0.05,
                    "transform": "clr",
                    "iqr_bottom_percentile": 0.05,
                },
            },
            "MT": {
                "path": "data/MT.tsv",
                "orientation": "profiles_by_features",
                "preprocessing": {
                    "mode": "raw",
                    "prevalence": 0.05,
                    "transform": "clr",
                    "iqr_bottom_percentile": 0.05,
                },
            },
            "MB": {
                "path": "data/MB.tsv",
                "orientation": "profiles_by_features",
                "preprocessing": {
                    "mode": "raw",
                    "prevalence": 0.05,
                    "transform": "pqn_log2",
                    "iqr_bottom_percentile": 0.05,
                },
            },
        },
        "covariates": {
            "categorical": ["site", "sex", "exposure"],
            "numeric": ["age"],
            "static": ["site", "sex", "age"],
        },
        "residualization": {
            "include_group_main_effect": True,
            "time_spline_df": 3,
            "min_observations": 20,
            "min_subjects": 8,
            "reml": True,
            "max_iterations": 200,
            "optimizers": ["lbfgs", "powell", "cg"],
        },
        "matching": {
            "near_tolerance_days": 7,
            "lag_windows": [
                {"name": "lag_4_8w", "min_days": 28, "max_days": 56, "target_days": 42}
            ],
            "baseline_lookback_days": 28,
        },
        "screening": {
            "min_pairs": 30,
            "min_subjects": 10,
            "min_abs_correlation": 0.10,
            "top_k_per_feature": 10,
            "max_edges_per_block": 1000,
            "baseline_mode": "none",
            "sample_split": {
                "enabled": False,
                "screening_fraction": 0.50,
                "min_screening_subjects_per_group": 3,
                "seed": seed + 17,
            },
        },
        "modeling": {
            "reference_group": "reference",
            "min_pairs": 30,
            "min_subjects": 10,
            "min_subjects_per_group": 5,
            "near_fit_both_orientations": True,
            "baseline_mode": "none",
            "multiple_testing_scope": "block",
            "max_iterations": 200,
            "optimizers": ["lbfgs", "powell", "cg"],
        },
    }
    config_path = root / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return {
        "config": config_path,
        "metadata": metadata_path,
        "truth": truth_path,
        **{f"matrix_{omic}": path for omic, path in matrix_paths.items()},
    }
