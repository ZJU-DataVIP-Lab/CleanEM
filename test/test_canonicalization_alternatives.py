import unittest

import pandas as pd

from canonicalization import (
    PlanDrivenCanonicalizationAgent,
    _split_top_level_regex_alternatives,
)
from cleanem_models import CellRecord
from agent import CanonicalizationPlanningAgent


class CanonicalizationAlternativeTests(unittest.TestCase):
    def test_splits_only_top_level_alternatives(self):
        self.assertEqual(
            _split_top_level_regex_alternatives(r"oz$|OZ\.|ounce$"),
            [r"oz$", r"OZ\.", r"ounce$"],
        )
        self.assertEqual(
            _split_top_level_regex_alternatives(r"(?:oz|ounce)$"),
            [r"(?:oz|ounce)$"],
        )
        self.assertEqual(
            _split_top_level_regex_alternatives(r"[|]$|ounce$"),
            [r"[|]$", r"ounce$"],
        )

    def test_combined_source_surfaces_are_validated_independently(self):
        values = (
            ["12.0 oz."] * 924
            + ["12.0 oz"] * 621
            + ["12.0 OZ."] * 154
            + ["12.0 ounce"] * 570
        )
        operation = {
            "mechanism": "unit_variant",
            "operation": "regex_replace",
            "params": {
                "pattern": r"oz$|OZ\.|ounce$",
                "replacement": "oz.",
                "target_pattern": "numeric_with_suffix",
            },
            "confidence": 0.98,
            "repairability": 0.95,
            "rationale": [],
        }
        agent = PlanDrivenCanonicalizationAgent(
            pd.DataFrame({"ounces": values}),
            {"ounces": {"semantics": {"archetype": "unit_measure"}}},
            {},
            {
                "ounces": {
                    "applicable": True,
                    "confidence": 0.99,
                    "operations": [operation],
                }
            },
        )

        diagnostics = agent.validate_plans()["ounces"]

        self.assertEqual(len(diagnostics), 3)
        self.assertTrue(all(item["accepted"] for item in diagnostics))
        self.assertEqual(len(agent.validated_operations["ounces"]), 3)

    def test_completes_observed_punctuation_variant_of_dominant_suffix(self):
        values = ["12.0 oz."] * 924 + ["12.0 oz"] * 621 + ["12.0 ounce"] * 570
        operations = [{
            "mechanism": "unit_variant",
            "operation": "regex_replace",
            "params": {
                "pattern": "ounce$",
                "replacement": "oz.",
                "target_pattern": "numeric_with_suffix",
            },
            "confidence": 0.94,
            "repairability": 0.94,
            "rationale": [],
        }]
        agent = PlanDrivenCanonicalizationAgent(
            pd.DataFrame({"ounces": values}),
            {"ounces": {"semantics": {"archetype": "unit_measure"}}},
            {},
            {
                "ounces": {
                    "applicable": True,
                    "confidence": 0.99,
                    "operations": operations,
                }
            },
        )

        diagnostics = agent.validate_plans()["ounces"]

        derived = [
            item for item in diagnostics
            if item.get("derived_from_observed_suffix") == "oz"
        ]
        self.assertEqual(len(derived), 1)
        self.assertTrue(derived[0]["accepted"])

    def test_rare_surface_confidence_is_not_a_hard_gate_when_repairable(self):
        values = ["male"] * 20 + ["mƟale"]
        operation = {
            "mechanism": "encoding_artifact",
            "operation": "regex_replace",
            "params": {
                "pattern": "mƟale",
                "replacement": "male",
                "target_pattern": "lexical",
            },
            # The planner uses this small value for a one-row surface ratio.
            "confidence": 0.01,
            "repairability": 0.90,
            "rationale": [],
        }
        agent = PlanDrivenCanonicalizationAgent(
            pd.DataFrame({"gender": values}),
            {"gender": {"semantics": {"archetype": "closed_enum"}}},
            {
                "20::gender": CellRecord(
                    key="20::gender",
                    row_index=20,
                    column="gender",
                    raw_value="mƟale",
                    value_key="gender::mƟale",
                )
            },
            {
                "gender": {
                    "applicable": True,
                    "confidence": 0.98,
                    "operations": [operation],
                }
            },
        )

        diagnostics = agent.validate_plans()["gender"]

        self.assertTrue(diagnostics[0]["accepted"])
        self.assertGreaterEqual(diagnostics[0]["effective_confidence"], 0.50)
        candidates = agent.generate()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates["20::gender"][0].candidate_value, "male")

    def test_low_repairability_still_respects_operation_confidence_gate(self):
        operation = {
            "mechanism": "new_surface",
            "operation": "regex_replace",
            "params": {
                "pattern": "rare",
                "replacement": "common",
                "target_pattern": "lexical",
            },
            "confidence": 0.01,
            "repairability": 0.20,
            "rationale": [],
        }
        agent = PlanDrivenCanonicalizationAgent(
            pd.DataFrame({"value": ["common"] * 10 + ["rare"]}),
            {"value": {"semantics": {"archetype": "open_entity"}}},
            {},
            {
                "value": {
                    "applicable": True,
                    "confidence": 0.98,
                    "operations": [operation],
                }
            },
        )

        diagnostics = agent.validate_plans()["value"]

        self.assertFalse(diagnostics[0]["accepted"])
        self.assertEqual(diagnostics[0]["reason"], "low_planner_confidence")

    def test_operation_name_in_mechanism_is_normalized_for_audit(self):
        planner = object.__new__(CanonicalizationPlanningAgent)
        validated = planner._validate_operation({
            "mechanism": "regex_replace",
            "operation": "regex_replace",
            "params": {"pattern": "x", "replacement": "y"},
            "confidence": 0.9,
            "repairability": 0.9,
        })

        self.assertEqual(validated["mechanism"], "surface_variant")
        self.assertEqual(validated["reported_mechanism"], "regex_replace")


if __name__ == "__main__":
    unittest.main()
