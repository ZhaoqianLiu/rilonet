from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

import pandas as pd

from rilonet.interpretation import (
    build_edge_master,
    classify_edges,
    discover_modules,
)


class InterpretationTests(unittest.TestCase):
    def test_edge_master_uses_requested_q_column(self) -> None:
        rows = []
        for group, effect in (("reference", 0.6), ("condition", 0.1)):
            rows.append(
                {
                    "edge_id": "near::MG::a::MT::b",
                    "group": group,
                    "effect": effect,
                    "contrast_vs_reference": -0.5,
                    "contrast_p": 0.01,
                    "contrast_q": 0.02,
                    "contrast_q_block": 0.02,
                    "analysis_type": "near",
                    "block": "MG_MT",
                    "x_omic": "MG",
                    "y_omic": "MT",
                    "x_feature": "a",
                    "y_feature": "b",
                    "sign_consistent": True,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "near_MG_MT_edges.tsv"
            pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
            master = build_edge_master(
                directory,
                reference_group="reference",
                condition_group="condition",
                q_column="contrast_q",
            )

        self.assertEqual(len(master), 1)
        self.assertAlmostEqual(float(master.loc[0, "contrast_q"]), 0.02)
        self.assertEqual(master.loc[0, "contrast_q_source"], "contrast_q")
        self.assertTrue(pd.isna(master.loc[0, "contrast_q_global"]))

    def test_edge_classification_uses_group_effects(self) -> None:
        edges = pd.DataFrame(
            {
                "effect_reference": [0.6, 0.0, 0.4],
                "effect_condition": [0.1, 0.4, -0.2],
                "contrast": [-0.5, 0.4, -0.6],
            }
        )
        classified = classify_edges(edges, near_zero_threshold=0.15)
        self.assertEqual(
            classified["edge_class"].tolist(),
            ["weakened_or_lost", "condition_emergent", "sign_reversed"],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("igraph") is not None
        and importlib.util.find_spec("leidenalg") is not None,
        "interpretation extras are not installed",
    )
    def test_consensus_leiden_is_deterministic(self) -> None:
        nodes = [("MG", "a"), ("MG", "b"), ("MT", "c"), ("MB", "d")]
        rows = []
        for left, (x_omic, x_feature) in enumerate(nodes):
            for y_omic, y_feature in nodes[left + 1 :]:
                rows.append(
                    {
                        "edge_id": f"edge_{len(rows)}",
                        "analysis_type": "near",
                        "x_omic": x_omic,
                        "x_feature": x_feature,
                        "y_omic": y_omic,
                        "y_feature": y_feature,
                        "contrast": 1.0,
                    }
                )
        edges = pd.DataFrame(rows)
        arguments = {
            "resolution": 0.1,
            "seed": 7,
            "leiden_runs": 3,
            "consensus_threshold": 0.5,
            "min_module_size": 2,
            "min_module_stability": 0.5,
        }
        first = discover_modules(edges, **arguments)
        second = discover_modules(edges, **arguments)
        for observed, expected in zip(first, second):
            pd.testing.assert_frame_equal(observed, expected)
        self.assertEqual(len(first[0]), 4)
        self.assertEqual(first[0]["module_id"].nunique(), 1)
        self.assertFalse(first[2].empty)


if __name__ == "__main__":
    unittest.main()
