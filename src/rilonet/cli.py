from __future__ import annotations

import argparse

from .common import load_config
from .interpretation import run_module_interpretation
from .matching import build_all_pairs
from .modeling import model_all
from .prepare import prepare_inputs
from .residualize import residualize_all, residualize_omic
from .screening import screen_all
from .simulation_study import run_simulation_study


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="rilonet",
        description="Longitudinal MG-MT-MB differential cross-omics network framework",
    )
    result.add_argument("--config", default="configs/analysis_example.yaml")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare", help="Map metadata and preprocess MG, MT, and MB matrices"
    )
    residual = commands.add_parser(
        "residualize", help="Fit feature-wise screening residual models"
    )
    residual.add_argument("--omic", choices=["MG", "MT", "MB", "all"], default="all")
    residual.add_argument("--max-features", type=int)
    residual.add_argument("--workers", type=int)
    commands.add_parser(
        "pair", help="Build near-contemporaneous and directional lagged pairs"
    )
    commands.add_parser("screen", help="Screen cross-omics candidate edges")
    model = commands.add_parser(
        "model", help="Fit edge-wise differential association models"
    )
    model.add_argument("--max-edges", type=int)
    model.add_argument("--baseline-mode", choices=["none", "adjust", "change"])
    model.add_argument("--workers", type=int)
    run = commands.add_parser("run-core", help="Run prepare through edge modeling")
    run.add_argument("--max-features", type=int)
    run.add_argument("--max-edges", type=int)
    run.add_argument("--workers", type=int)
    simulation_study = commands.add_parser(
        "simulate-study", help="Run the end-to-end simulation study"
    )
    simulation_study.add_argument("--study-config", required=True)
    interpretation = commands.add_parser(
        "interpret-modules",
        help="Annotate discovery edges and construct cross-omics modules",
    )
    interpretation.add_argument("--analysis-dir", required=True)
    interpretation.add_argument("--network-dir")
    interpretation.add_argument("--output-dir")
    interpretation.add_argument("--reference-group", default="nonIBD")
    interpretation.add_argument("--condition-group", default="IBD")
    interpretation.add_argument(
        "--q-column",
        default="contrast_q",
        help="Adjusted contrast column used to select discovery edges",
    )
    interpretation.add_argument("--q-threshold", type=float, default=0.05)
    interpretation.add_argument("--near-zero-threshold", type=float, default=0.10)
    interpretation.add_argument("--resolution", type=float, default=1.0)
    interpretation.add_argument("--seed", type=int, default=20260721)
    interpretation.add_argument("--leiden-runs", type=int, default=10)
    interpretation.add_argument("--consensus-threshold", type=float, default=0.55)
    interpretation.add_argument("--min-module-size", type=int, default=3)
    interpretation.add_argument("--min-module-stability", type=float, default=0.60)
    interpretation.add_argument("--offline", action="store_true")
    interpretation.add_argument("--hmdb-xml")
    interpretation.add_argument("--ncbi-email")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "simulate-study":
        outputs = run_simulation_study(args.study_config)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    if args.command == "interpret-modules":
        outputs = run_module_interpretation(
            args.analysis_dir,
            network_dir=args.network_dir,
            output_dir=args.output_dir,
            reference_group=args.reference_group,
            condition_group=args.condition_group,
            q_column=args.q_column,
            q_threshold=args.q_threshold,
            near_zero_threshold=args.near_zero_threshold,
            resolution=args.resolution,
            seed=args.seed,
            leiden_runs=args.leiden_runs,
            consensus_threshold=args.consensus_threshold,
            min_module_size=args.min_module_size,
            min_module_stability=args.min_module_stability,
            offline=args.offline,
            hmdb_xml=args.hmdb_xml,
            ncbi_email=args.ncbi_email,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    config = load_config(args.config)
    if args.command == "prepare":
        prepare_inputs(config)
    elif args.command == "residualize":
        if args.omic == "all":
            residualize_all(
                config, max_features=args.max_features, workers=args.workers
            )
        else:
            residualize_omic(
                config, args.omic, max_features=args.max_features, workers=args.workers
            )
    elif args.command == "pair":
        build_all_pairs(config)
    elif args.command == "screen":
        screen_all(config)
    elif args.command == "model":
        model_all(
            config,
            max_edges=args.max_edges,
            baseline_mode=args.baseline_mode,
            workers=args.workers,
        )
    elif args.command == "run-core":
        prepare_inputs(config)
        residualize_all(config, max_features=args.max_features, workers=args.workers)
        build_all_pairs(config)
        screen_all(config)
        model_all(config, max_edges=args.max_edges, workers=args.workers)
