from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rilonet.common import bh_fdr, ensure_dir, load_config, output_root, split_subjects
from rilonet.matching import build_all_pairs, optimal_matches
from rilonet.modeling import _apply_global_contrast_q, make_design, model_all
from rilonet.prepare import _clr_transform, _pqn_log2_transform, prepare_inputs
from rilonet.residualize import residualize_all
from rilonet.screening import _create_participant_split, _screening_matrices, screen_all
from rilonet.simulate import generate_simulation_dataset
from rilonet.simulation_study import _network_metrics


class PreprocessingTests(unittest.TestCase):
    def test_clr_rows_sum_to_zero(self) -> None:
        frame = pd.DataFrame([[1, 0, 3], [4, 2, 0]], index=["A", "B"])
        transformed, _ = _clr_transform(frame, None)
        np.testing.assert_allclose(transformed.sum(axis=1), 0.0, atol=1e-10)

    def test_pqn_log_output_is_finite(self) -> None:
        frame = pd.DataFrame([[1, 0, 3], [2, 4, 6], [1, 2, 3]], index=["A", "B", "C"])
        transformed, report = _pqn_log2_transform(frame)
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertGreater(report["pseudocount"], 0)


class MatchingTests(unittest.TestCase):
    def test_profiles_are_not_reused(self) -> None:
        def frame(ids: list[str], days: list[float]) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "sample_id": ids,
                    "canonical_biosample_id": ids,
                    "match_time_days": days,
                    "study_time_days": days,
                }
            )

        matches = optimal_matches(
            frame(["A", "B", "C"], [0, 10, 20]),
            frame(["A", "B", "D"], [0, 11, 40]),
            -2,
            2,
            0,
            True,
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual(len({row for row, _, _, _ in matches}), 2)
        self.assertEqual(len({column for _, column, _, _ in matches}), 2)


class ScreeningTests(unittest.TestCase):
    def test_group_specific_screening_retains_sign_reversal(self) -> None:
        x = np.tile(np.asarray([-2.0, -1.0, 1.0, 2.0]), 2)[:, None]
        y = np.concatenate([x[:4, 0], -x[4:, 0]])[:, None]
        work = pd.DataFrame(
            {"group": ["reference"] * 4 + ["condition"] * 4}
        )
        config = {
            "metadata": {
                "group_levels": ["reference", "condition"],
                "reference_group": "reference",
            }
        }
        pooled, reference, condition, difference, score, _, _ = _screening_matrices(
            work, x, y, config
        )
        self.assertAlmostEqual(float(pooled[0, 0]), 0.0, places=12)
        self.assertAlmostEqual(float(reference[0, 0]), 1.0, places=12)
        self.assertAlmostEqual(float(condition[0, 0]), -1.0, places=12)
        self.assertAlmostEqual(float(difference[0, 0]), -2.0, places=12)
        self.assertAlmostEqual(float(score[0, 0]), 2.0, places=12)

    def test_participant_split_is_disjoint_and_group_stratified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = generate_simulation_dataset(
                directory,
                seed=31,
                n_subjects=40,
                n_visits=7,
                condition_fraction=0.50,
            )
            config = load_config(outputs["config"])
            config["screening"]["sample_split"]["enabled"] = True
            prepare_inputs(config)
            candidate_dir = ensure_dir(output_root(config) / "04_candidates")
            _create_participant_split(config, candidate_dir)
            screening = split_subjects(config, "screening")
            inference = split_subjects(config, "inference")
            self.assertTrue(screening)
            self.assertTrue(inference)
            self.assertFalse(screening & inference)
            split = pd.read_csv(candidate_dir / "participant_split.tsv", sep="\t")
            counts = split.groupby(["group", "role"])["subject_id"].nunique()
            self.assertTrue((counts > 0).all())


class ModelingTests(unittest.TestCase):
    def test_baseline_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = generate_simulation_dataset(directory, seed=9, n_subjects=16, n_visits=8)
            config = load_config(outputs["config"])
            rows = []
            rng = np.random.default_rng(4)
            for group in ("reference", "condition"):
                for subject_index in range(6):
                    subject = f"{group}_{subject_index}"
                    for time in range(4):
                        rows.append(
                            {
                                "subject_id": subject,
                                "group": group,
                                "predictor": rng.normal(),
                                "outcome": rng.normal(),
                                "mid_time": time,
                                "delta_days": 42,
                                "baseline_outcome_raw": rng.normal(),
                                "x_cov__site": "A",
                                "x_cov__sex": "F",
                                "x_cov__exposure": "No",
                                "x_cov__age": 40,
                            }
                        )
            frame = pd.DataFrame(rows)
            none_design, _ = make_design(frame, "lagged", config, "none")
            adjusted_design, _ = make_design(frame, "lagged", config, "adjust")
            self.assertNotIn("baseline_outcome_within", none_design)
            self.assertIn("baseline_outcome_within", adjusted_design)
            self.assertIn("baseline_outcome_between", adjusted_design)
            self.assertIn("baseline_outcome_within_by_condition", adjusted_design)
            self.assertIn("baseline_outcome_between_by_condition", adjusted_design)

    def test_global_bh_uses_all_network_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "metadata": {
                    "group_levels": ["reference", "condition"],
                    "reference_group": "reference",
                }
            }
            paths = []
            edge_groups = (["A", "B"], ["C"])
            pvalue_groups = ([0.01, 0.04], [0.20])
            for file_index, (edges, pvalues) in enumerate(
                zip(edge_groups, pvalue_groups)
            ):
                rows = []
                for edge, pvalue in zip(edges, pvalues):
                    for group in ("reference", "condition"):
                        rows.append(
                            {
                                "edge_id": edge,
                                "group": group,
                                "contrast_p": pvalue,
                                "contrast_q_block": pvalue,
                                "contrast_q": pvalue,
                            }
                        )
                path = root / f"block_{file_index}_edges.tsv"
                pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
                paths.append(path)

            report = _apply_global_contrast_q(paths, config)
            expected = dict(zip(["A", "B", "C"], bh_fdr([0.01, 0.04, 0.20])))

            self.assertEqual(report["tested_contrasts"], 3)
            for path in paths:
                result = pd.read_csv(path, sep="\t")
                observed = result.set_index("edge_id")["contrast_q_global"].to_dict()
                for edge, qvalue in observed.items():
                    self.assertAlmostEqual(qvalue, expected[edge])
                np.testing.assert_allclose(
                    result["contrast_q"],
                    result["contrast_q_global"],
                )


class IntegrationTests(unittest.TestCase):
    def test_simulated_inputs_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = generate_simulation_dataset(directory, seed=11, n_subjects=20, n_visits=8)
            config = load_config(outputs["config"])
            manifest = prepare_inputs(config)
            self.assertEqual(set(manifest["omic"]), {"MG", "MT", "MB"})
            self.assertGreater(int(manifest["valid_for_analysis"].sum()), 0)
            truth = pd.read_csv(outputs["truth"], sep="\t")
            self.assertTrue({"near", "lagged"}.issubset(set(truth["analysis_type"])))

    def test_parallel_execution_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = generate_simulation_dataset(
                directory, seed=19, n_subjects=20, n_visits=8
            )
            config = load_config(outputs["config"])
            config["screening"]["min_abs_correlation"] = 0.0
            config["screening"]["max_edges_per_block"] = 9
            root = Path(directory) / "analysis"

            prepare_inputs(config)
            residualize_all(config, max_features=3, workers=1)
            serial_residuals = {
                omic: pd.read_parquet(
                    root / "02_residuals" / f"{omic}_residuals.parquet"
                )
                for omic in ("MG", "MT", "MB")
            }
            residualize_all(config, max_features=3, workers=2)
            for omic, expected in serial_residuals.items():
                pd.testing.assert_frame_equal(
                    expected,
                    pd.read_parquet(
                        root / "02_residuals" / f"{omic}_residuals.parquet"
                    ),
                    check_exact=False,
                    rtol=1e-12,
                    atol=1e-12,
                )

            build_all_pairs(config)
            screen_all(config)
            serial_report = model_all(config, max_edges=1, workers=1)
            network_dir = root / "05_networks_smoke_none"
            serial_networks = {
                path.name: pd.read_csv(path, sep="\t")
                for path in network_dir.glob("*_edges.tsv")
            }
            parallel_report = model_all(config, max_edges=1, workers=2)
            self.assertEqual(serial_report, parallel_report)
            for name, expected in serial_networks.items():
                pd.testing.assert_frame_equal(
                    expected,
                    pd.read_csv(network_dir / name, sep="\t"),
                    check_exact=False,
                    rtol=1e-12,
                    atol=1e-12,
                )

class SimulationStudyTests(unittest.TestCase):
    def test_null_generator_has_no_differential_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = generate_simulation_dataset(
                directory,
                seed=23,
                n_subjects=12,
                n_visits=7,
                dimensions={"MG": 6, "MT": 6, "MB": 6},
                differential_truth=False,
            )
            truth = pd.read_csv(outputs["truth"], sep="\t")
            self.assertFalse(truth["is_differential"].astype(bool).any())
            self.assertTrue(truth["category"].eq("shared").all())

    def test_differential_generator_has_six_differential_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = generate_simulation_dataset(
                directory,
                seed=24,
                n_subjects=12,
                n_visits=7,
                dimensions={"MG": 8, "MT": 8, "MB": 8},
                differential_truth=True,
            )
            truth = pd.read_csv(outputs["truth"], sep="\t")
            differential = truth[truth["is_differential"].astype(bool)]
            shared = truth[~truth["is_differential"].astype(bool)]
            self.assertEqual(len(differential), 6)
            self.assertEqual(len(shared), 2)
            counts = differential.groupby(["analysis_type", "category"]).size()
            for analysis_type in ("near", "lagged"):
                for category in ("weakened", "strengthened", "sign_reversed"):
                    self.assertEqual(int(counts.loc[(analysis_type, category)]), 1)

    def test_network_metrics_use_differential_truth(self) -> None:
        true_a = ("near", "MG", "MG_000", "MT", "MT_000")
        false_b = ("near", "MG", "MG_001", "MT", "MT_001")
        metrics = _network_metrics({true_a}, {true_a, false_b}, {true_a, false_b}, "near")
        self.assertEqual(metrics["recovered_planted_edges"], 1)
        self.assertEqual(metrics["non_planted_calls"], 1)
        self.assertAlmostEqual(metrics["planted_edge_precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
