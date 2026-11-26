import unittest

import pandas as pd

from cleanem_models import CellRecord
from main import _build_corrected_value_standardization_candidates


class CorrectedValueStandardizationTests(unittest.TestCase):
    def test_corrected_values_define_a_general_numeric_unit_format(self):
        dirty = pd.DataFrame({
            "ounces": [
                "12.0 OZ.",
                "24.0 oz. Alumi-Tek",
                "16.0 ounce",
                "16.0 oz.",
                "0.05%",
                "12.0 oz.",
            ]
        })
        clean = pd.DataFrame({
            "ounces": [
                "12.0 oz.",
                "24.0 oz.",
                "16.0 oz.",
                "16.0 oz.",
                "12.0 oz.",
                "16.0 oz.",
            ]
        })
        registry = {
            f"{row}::ounces": CellRecord(
                key=f"{row}::ounces",
                row_index=row,
                column="ounces",
                raw_value=dirty.loc[row, "ounces"],
                value_key=f"ounces::{dirty.loc[row, 'ounces'].lower()}",
            )
            for row in range(len(dirty))
        }

        observations = {}
        candidates, summary = _build_corrected_value_standardization_candidates(
            dirty,
            clean,
            registry,
            ["0::ounces", "1::ounces"],
            {"0::ounces": 1, "1::ounces": 1},
            detection_observations=observations,
        )

        self.assertEqual(
            candidates["2::ounces"][0].candidate_value,
            "16.0 oz.",
        )
        self.assertNotIn("3::ounces", candidates)
        self.assertNotIn("4::ounces", candidates)
        self.assertNotIn("5::ounces", candidates)
        self.assertEqual(summary["learned_format_templates"], 1)
        self.assertEqual(summary["format_template_candidates"], 1)
        self.assertTrue(summary["used_in_detection"])
        self.assertEqual(observations["2::ounces"][0].polarity, "dirty")
        self.assertEqual(observations["3::ounces"][0].polarity, "clean")
        self.assertEqual(observations["4::ounces"][0].polarity, "dirty")

    def test_corrected_format_can_reverse_the_unit_direction(self):
        dirty = pd.DataFrame({"measure": ["12 oz", "16 ounce", "20"]})
        clean = pd.DataFrame({"measure": ["12", "16", "20"]})
        registry = {
            f"{row}::measure": CellRecord(
                key=f"{row}::measure",
                row_index=row,
                column="measure",
                raw_value=dirty.loc[row, "measure"],
                value_key=f"measure::{dirty.loc[row, 'measure'].lower()}",
            )
            for row in range(len(dirty))
        }
        observations = {}

        candidates, summary = _build_corrected_value_standardization_candidates(
            dirty,
            clean,
            registry,
            ["0::measure"],
            {"0::measure": 1},
            detection_observations=observations,
        )

        self.assertEqual(candidates["1::measure"][0].candidate_value, "16")
        self.assertNotIn("2::measure", candidates)
        self.assertEqual(observations["1::measure"][0].polarity, "dirty")
        self.assertEqual(observations["2::measure"][0].polarity, "clean")
        self.assertTrue(summary["used_in_detection"])

    def test_clean_labels_alone_define_numeric_format_contract(self):
        dirty = pd.DataFrame({"abv": ["0.037", "0.06", "0.07%"]})
        clean = pd.DataFrame({"abv": ["0.037", "0.06", "0.07"]})
        registry = {
            f"{row}::abv": CellRecord(
                key=f"{row}::abv",
                row_index=row,
                column="abv",
                raw_value=dirty.loc[row, "abv"],
                value_key=f"abv::{dirty.loc[row, 'abv']}",
            )
            for row in range(len(dirty))
        }
        observations = {}

        candidates, summary = _build_corrected_value_standardization_candidates(
            dirty,
            clean,
            registry,
            ["0::abv", "1::abv"],
            {"0::abv": 0, "1::abv": 0},
            detection_observations=observations,
        )

        self.assertEqual(candidates, {})
        self.assertEqual(observations["0::abv"][0].polarity, "clean")
        self.assertEqual(observations["1::abv"][0].polarity, "clean")
        self.assertEqual(observations["2::abv"][0].polarity, "dirty")
        self.assertTrue(observations["2::abv"][0].hard)
        self.assertEqual(summary["learned_detection_contracts"], 1)
        self.assertTrue(summary["used_in_detection"])


if __name__ == "__main__":
    unittest.main()
