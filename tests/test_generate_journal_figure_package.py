import sys
import unittest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from generate_journal_figure_package import (  # noqa: E402
    MAIN_FIGURE_STEMS,
    SUPPLEMENTARY_FIGURE_STEMS,
    assert_output_scope,
    build_all_figures,
    build_claim_rows,
    build_legend_texts,
    build_source_map_rows,
    format_p_value,
    parse_confusion_matrix,
    required_source_paths,
)


class JournalFigurePackageTests(unittest.TestCase):
    def test_inventory_is_exactly_five_main_and_three_supplementary_figures(self):
        self.assertEqual(
            MAIN_FIGURE_STEMS,
            (
                "figure1_overall_framework",
                "figure2_duration_clean_ablations",
                "figure3_hierarchical_clean_prototypes",
                "figure4_simulated_noise_robustness",
                "figure5_failure_selective_prediction",
            ),
        )
        self.assertEqual(
            SUPPLEMENTARY_FIGURE_STEMS,
            (
                "figure_s1_clean_ablations",
                "figure_s2_demand_environment_snr",
                "figure_s3_confusion_calibration_fold_deltas",
            ),
        )

    def test_source_map_has_required_columns_and_every_planned_panel(self):
        rows = build_source_map_rows()
        required_columns = {
            "figure",
            "panel",
            "source_csv",
            "source_columns",
            "transformation",
            "filtering",
            "statistic",
            "output_file",
        }
        self.assertTrue(rows)
        self.assertTrue(required_columns.issubset(rows[0]))

        covered = {(row["figure"], row["panel"]) for row in rows}
        expected = {
            *(('Figure 1', panel) for panel in 'abcd'),
            *(('Figure 2', panel) for panel in 'abc'),
            *(('Figure 3', panel) for panel in 'abc'),
            *(('Figure 4', panel) for panel in 'abc'),
            *(('Figure 5', panel) for panel in 'abc'),
            ('Supplementary Figure S1', 'a'),
            *(('Supplementary Figure S2', panel) for panel in 'abcd'),
            *(('Supplementary Figure S3', panel) for panel in 'abcd'),
        }
        self.assertEqual(covered, expected)
        self.assertTrue(all(row["source_csv"] for row in rows))

    def test_required_sources_exist_in_locked_repository(self):
        missing = [path for path in required_source_paths(ROOT) if not path.is_file()]
        self.assertEqual(missing, [])

    def test_source_map_records_exact_noise_row_filters(self):
        rows = {
            (row["figure"], row["panel"]): row for row in build_source_map_rows()
        }
        detailed_curve_panels = [
            ("Figure 4", "b"),
            ("Figure 5", "a"),
            *(("Supplementary Figure S2", panel) for panel in "abcd"),
        ]
        for key in detailed_curve_panels:
            filtering = rows[key]["filtering"]
            self.assertIn("DWASHING, TBUS, STRAFFIC", filtering)
            self.assertIn("exclude GROUP aggregate rows", filtering)
        self.assertIn("ALL_NOISY/GROUP", rows[("Figure 5", "c")]["filtering"])

    def test_protocol_legend_marks_n_as_not_applicable(self):
        english, chinese = build_legend_texts(ROOT)
        self.assertIn("n is not applicable", english)
        self.assertIn("n 不适用", chinese)

    def test_claim_bearing_cohort_and_audio_source_are_directly_traced(self):
        source_rows = {
            (row["figure"], row["panel"]): row for row in build_source_map_rows()
        }
        claim_rows = {
            (row["figure"], row["panel"]): row for row in build_claim_rows()
        }
        for rows in (source_rows, claim_rows):
            self.assertIn(
                "paper/manuscript/paper_en_full_story_polished.md",
                rows[("Figure 1", "a")][
                    "source_csv" if rows is source_rows else "evidence_file"
                ],
            )
            self.assertIn(
                "reports/prototype_noise_demand_w05_final/final_runs.csv",
                rows[("Figure 3", "c")][
                    "source_csv" if rows is source_rows else "evidence_file"
                ],
            )

    def test_parse_confusion_matrix_validates_labels_and_shape(self):
        matrix = parse_confusion_matrix(
            "cough|calm_grunt|feeding|stress_vocal",
            "[[5, 2, 1, 1042], [0, 1050, 0, 0], [0, 0, 853, 197], [0, 0, 64, 986]]",
        )
        self.assertEqual(matrix.shape, (4, 4))
        np.testing.assert_array_equal(matrix.sum(axis=1), [1050, 1050, 1050, 1050])

        with self.assertRaisesRegex(ValueError, "label count"):
            parse_confusion_matrix("a|b", "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]")

    def test_output_scope_rejects_paths_outside_approved_directories(self):
        approved = (
            ROOT / "paper" / "figures_journal",
            ROOT / "paper" / "supplementary" / "figures",
            ROOT / "paper" / "figure_source_data",
            ROOT / "paper" / "review",
        )
        assert_output_scope(ROOT / "paper" / "figures_journal" / "figure1.svg", approved)
        with self.assertRaisesRegex(ValueError, "approved output directories"):
            assert_output_scope(ROOT / "paper" / "tables" / "table3_duration_comparison.csv", approved)

    def test_p_values_are_exact_numeric_text_without_significance_stars(self):
        self.assertEqual(format_p_value(0.00016230344772338867), "P=0.000162303")
        self.assertEqual(format_p_value(0.058253), "P=0.058253")
        self.assertNotIn("*", format_p_value(9.14928944129577e-05))

    def test_all_figure_builders_return_fixed_width_matplotlib_figures(self):
        figures = build_all_figures(ROOT)
        try:
            self.assertEqual(tuple(figures), MAIN_FIGURE_STEMS + SUPPLEMENTARY_FIGURE_STEMS)
            for figure in figures.values():
                self.assertAlmostEqual(figure.get_figwidth() * 25.4, 183.0, places=5)
                self.assertGreater(len(figure.axes), 0)
        finally:
            for figure in figures.values():
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
