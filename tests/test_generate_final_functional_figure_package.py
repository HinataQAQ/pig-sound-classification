from __future__ import annotations

import re
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

from tools.generate_final_functional_figure_package import (  # noqa: E402
    DERIVED_TABLE_FILENAMES,
    MAIN_FIGURE_STEMS,
    SUPPLEMENTARY_FIGURE_STEMS,
    aggregate_deployable_margin,
    aggregate_hierarchy_utility,
    build_all_figures,
    build_claim_rows,
    build_legend_texts,
    build_source_map_rows,
    collect_frozen_source_paths,
    compute_run_deployable_margin,
    compute_run_hierarchy_utility,
    configure_style,
    discover_pre_analysis_source_paths,
    inspect_export,
    load_locked_lambda_selection,
    load_analysis,
    predicted_distance_margin,
    required_output_paths,
    save_figure_bundle,
    select_final_case_studies,
    validate_locked_lambda_reproduction,
    _promote_staged_outputs,
)


MAIN_LABELS = ("cough", "calm_grunt", "feeding", "stress_vocal")
AUX_LABELS = (
    "dry_cough",
    "abdominal_cough",
    "calm_grunt",
    "feeding",
    "frightened_stress",
    "anxious_stress",
)


def _fixture() -> pd.DataFrame:
    rows = [
        {
            "source_id": "s1",
            "y_true": "cough",
            "prototype_pred": "cough",
            "hierarchical_pred": "cough",
            "raw_softmax_pred": "cough",
            "softmax_aux_pred": "dry_cough",
            "prototype_aux_pred": "dry_cough",
            "main": [0.10, 0.40, 0.70, 0.90],
            "aux": [0.10, 0.30, 0.60, 0.80, 0.90, 1.00],
        },
        {
            "source_id": "s2",
            "y_true": "feeding",
            "prototype_pred": "feeding",
            "hierarchical_pred": "stress_vocal",
            "raw_softmax_pred": "feeding",
            "softmax_aux_pred": "feeding",
            "prototype_aux_pred": "frightened_stress",
            "main": [0.80, 0.70, 0.20, 0.30],
            "aux": [1.00, 0.90, 0.80, 0.20, 0.25, 0.50],
        },
        {
            "source_id": "s3",
            "y_true": "stress_vocal",
            "prototype_pred": "feeding",
            "hierarchical_pred": "feeding",
            "raw_softmax_pred": "feeding",
            "softmax_aux_pred": "anxious_stress",
            "prototype_aux_pred": "feeding",
            "main": [0.90, 0.80, 0.10, 0.35],
            "aux": [1.00, 0.90, 0.80, 0.10, 0.40, 0.50],
        },
        {
            "source_id": "s4",
            "y_true": "calm_grunt",
            "prototype_pred": "calm_grunt",
            "hierarchical_pred": "calm_grunt",
            "raw_softmax_pred": "calm_grunt",
            "softmax_aux_pred": "feeding",
            "prototype_aux_pred": "feeding",
            "main": [0.70, 0.10, 0.60, 0.80],
            "aux": [0.90, 0.80, 0.70, 0.20, 0.60, 0.50],
        },
    ]
    output: list[dict[str, object]] = []
    for row in rows:
        record = {
            key: value for key, value in row.items() if key not in {"main", "aux"}
        }
        for label, value in zip(MAIN_LABELS, row["main"], strict=True):
            record[f"main_proto_cosine_distance_{label}"] = value
        for label, value in zip(AUX_LABELS, row["aux"], strict=True):
            record[f"aux_proto_cosine_distance_{label}"] = value
        output.append(record)
    return pd.DataFrame(output)


class DeployableStatisticsTests(unittest.TestCase):
    def test_predicted_margin_sorts_distances_without_truth_labels(self) -> None:
        values = np.array([[0.40, 0.10, 0.25, 0.90], [0.2, 0.2, 0.8, 0.9]])

        actual = predicted_distance_margin(values)

        np.testing.assert_allclose(actual, [0.15, 0.0])

    def test_predicted_margin_rejects_invalid_matrices(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            predicted_distance_margin(np.array([0.1, 0.2]))
        with self.assertRaisesRegex(ValueError, "at least two"):
            predicted_distance_margin(np.array([[0.1]]))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            predicted_distance_margin(np.array([[0.1, np.nan]]))

    def test_run_margin_uses_shared_main_geometry_and_method_specific_errors(self) -> None:
        rows = compute_run_deployable_margin(_fixture(), fold=0, seed=42)
        lookup = rows.set_index("method")

        self.assertEqual(set(rows["method"]), {"main_prototype", "hierarchical_prototype"})
        self.assertEqual(lookup.loc["main_prototype", "distance_level"], "main")
        self.assertEqual(lookup.loc["hierarchical_prototype", "distance_level"], "main")
        self.assertEqual(
            set(rows["margin_scope"]), {"shared_main_prototype_geometry"}
        )
        self.assertEqual(lookup.loc["main_prototype", "margin_definition"], "second_nearest_minus_nearest")
        self.assertEqual(
            lookup.loc["hierarchical_prototype", "margin_uses_true_label"], False
        )
        self.assertEqual(
            lookup.loc["hierarchical_prototype", "evaluation_uses_true_label"], True
        )
        self.assertAlmostEqual(
            float(lookup.loc["main_prototype", "correct_margin_median"]), 0.30
        )
        self.assertAlmostEqual(
            float(lookup.loc["main_prototype", "error_margin_median"]), 0.25
        )

    def test_margin_aggregates_include_run_fold_and_fold_cluster_ci(self) -> None:
        runs = pd.concat(
            [
                compute_run_deployable_margin(_fixture(), fold=fold, seed=seed)
                for fold in range(5)
                for seed in (42, 123, 777, 2024, 3407)
            ],
            ignore_index=True,
        )

        result = aggregate_deployable_margin(runs, n_boot=200, seed=3407)

        self.assertEqual((result["record_type"] == "run").sum(), 50)
        self.assertEqual((result["record_type"] == "fold_mean").sum(), 10)
        overall = result[result["record_type"] == "overall_run_mean"]
        self.assertEqual(len(overall), 2)
        self.assertTrue(
            {
                "correct_margin_median_fold_cluster_ci95_low",
                "correct_margin_median_fold_cluster_ci95_high",
                "error_detection_auroc_fold_cluster_ci95_low",
                "error_detection_auroc_fold_cluster_ci95_high",
            }.issubset(overall.columns)
        )

    def test_aggregate_rejects_an_incomplete_fold_seed_grid(self) -> None:
        runs = pd.concat(
            [
                compute_run_deployable_margin(_fixture(), fold=fold, seed=seed)
                for fold in range(5)
                for seed in (42, 123, 777, 2024, 3407)
            ],
            ignore_index=True,
        )
        incomplete = runs.drop(runs.index[0])

        with self.assertRaisesRegex(ValueError, "exact 5-fold x 5-seed grid"):
            aggregate_deployable_margin(incomplete, n_boot=200, seed=3407)


class HierarchyUtilityTests(unittest.TestCase):
    def test_run_utility_reports_prevalence_precision_recall_and_enrichment(self) -> None:
        rows = compute_run_hierarchy_utility(_fixture(), fold=0, seed=42)
        required = {
            "inconsistency_prevalence",
            "error_rate_inconsistent",
            "error_rate_consistent",
            "error_enrichment_ratio",
            "inconsistency_error_precision",
            "inconsistency_error_recall",
        }
        self.assertTrue(required.issubset(rows.columns))
        self.assertEqual(
            set(rows["method"]),
            {"raw_softmax", "main_prototype", "hierarchical_prototype"},
        )
        raw = rows.set_index("method").loc["raw_softmax"]
        self.assertAlmostEqual(float(raw["inconsistency_prevalence"]), 0.5)
        self.assertAlmostEqual(float(raw["error_rate_inconsistent"]), 0.5)
        self.assertAlmostEqual(float(raw["inconsistency_error_precision"]), 0.5)
        self.assertAlmostEqual(float(raw["inconsistency_error_recall"]), 1.0)
        self.assertTrue(np.isnan(float(raw["error_enrichment_ratio"])))

    def test_zero_inconsistency_precision_uses_zero_division_zero(self) -> None:
        frame = _fixture().copy()
        frame["softmax_aux_pred"] = (
            "dry_cough",
            "feeding",
            "feeding",
            "calm_grunt",
        )

        raw = compute_run_hierarchy_utility(frame, fold=0, seed=42).set_index(
            "method"
        ).loc["raw_softmax"]

        self.assertEqual(int(raw["n_inconsistent"]), 0)
        self.assertEqual(float(raw["inconsistency_error_precision"]), 0.0)
        self.assertEqual(float(raw["inconsistency_error_recall"]), 0.0)
        self.assertTrue(np.isnan(float(raw["error_rate_inconsistent"])))

    def test_utility_aggregates_include_all_metrics_and_fold_cluster_cis(self) -> None:
        runs = pd.concat(
            [
                compute_run_hierarchy_utility(_fixture(), fold=fold, seed=seed)
                for fold in range(5)
                for seed in (42, 123, 777, 2024, 3407)
            ],
            ignore_index=True,
        )

        result = aggregate_hierarchy_utility(runs, n_boot=200, seed=3407)

        self.assertEqual((result["record_type"] == "run").sum(), 75)
        self.assertEqual((result["record_type"] == "fold_mean").sum(), 15)
        overall = result[result["record_type"] == "overall_run_mean"]
        self.assertEqual(len(overall), 3)
        for metric in (
            "inconsistency_prevalence",
            "inconsistency_error_precision",
            "inconsistency_error_recall",
            "error_enrichment_ratio",
        ):
            self.assertIn(f"{metric}_fold_cluster_ci95_low", overall.columns)
            self.assertIn(f"{metric}_fold_cluster_ci95_high", overall.columns)
            self.assertIn(f"{metric}_n_runs_available", overall.columns)
        fold_means = result[result["record_type"] == "fold_mean"]
        for row in overall.itertuples(index=False):
            method_folds = fold_means[fold_means["method"].eq(row.method)]
            self.assertAlmostEqual(
                float(row.inconsistency_prevalence),
                float(method_folds["inconsistency_prevalence"].mean()),
            )


class LockedSelectionTests(unittest.TestCase):
    def _write_locked_selection(self, root: Path, *, test_used: bool = False) -> None:
        target = root / "paper" / "final_validation" / "lambda_selection_by_fold.csv"
        target.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "fold": [0, 1, 2, 3, 4],
                "selected_lambda": [1.0, 0.2, 0.2, 0.2, 0.5],
                "test_metrics_used_for_selection": [test_used] * 5,
            }
        ).to_csv(target, index=False)

    def test_locked_lambda_csv_is_authoritative_and_test_independent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_locked_selection(root)

            frame, selected = load_locked_lambda_selection(root)

        self.assertEqual(len(frame), 5)
        self.assertEqual(selected, {0: 1.0, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.5})

    def test_locked_lambda_csv_rejects_test_selected_rows(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_locked_selection(root, test_used=True)

            with self.assertRaisesRegex(ValueError, "test metrics"):
                load_locked_lambda_selection(root)

    def test_reproduction_must_match_locked_lambda_csv(self) -> None:
        locked = {0: 1.0, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.5}
        reproduced = {**locked, 4: 1.0}

        with self.assertRaisesRegex(ValueError, "does not reproduce locked"):
            validate_locked_lambda_reproduction(locked, reproduced)


class PackageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analysis = load_analysis(ROOT, n_boot=200, seed=3407)

    def test_output_inventory_is_exact(self) -> None:
        self.assertEqual(
            DERIVED_TABLE_FILENAMES,
            (
                "deployable_predicted_margin_summary.csv",
                "hierarchy_inconsistency_utility.csv",
                "prototype_functional_final_summary.csv",
            ),
        )
        self.assertEqual(
            MAIN_FIGURE_STEMS,
            (
                "figure1_final_framework",
                "figure2_primary_classification_evidence",
                "figure3_prototype_functional_value",
                "figure4_prototype_candidate_cases",
            ),
        )
        self.assertEqual(
            SUPPLEMENTARY_FIGURE_STEMS,
            (
                "figure_s1_complete_clean_ablations",
                "figure_s2_fixed_lambda_retrospective_cumulative",
                "figure_s3_lambda_selection_by_fold",
                "figure_s4_best_epoch_validation_test_gap",
                "figure_s5_demand_simulated_noise",
                "figure_s6_aurc_augrc",
                "figure_s7_label_aware_true_class_margin",
            ),
        )
        paths = {path.relative_to(ROOT).as_posix() for path in required_output_paths(ROOT)}
        self.assertEqual(len([path for path in paths if path.endswith(".svg")]), 11)
        self.assertEqual(len([path for path in paths if path.endswith(".pdf")]), 11)
        self.assertEqual(len([path for path in paths if path.endswith(".png")]), 11)
        self.assertEqual(len([path for path in paths if path.endswith(".tiff")]), 11)
        figure_outputs = {
            path
            for path in paths
            if path.endswith((".svg", ".pdf", ".png", ".tiff"))
        }
        self.assertTrue(
            all(
                path.startswith("paper/figures_final_nomenclature_v1/")
                for path in figure_outputs
            )
        )
        self.assertTrue(
            {
                "paper/figures_final_nomenclature_v1/FIGURE_LEGENDS_CN.md",
                "paper/figures_final_nomenclature_v1/FIGURE_LEGENDS_EN.md",
                "paper/figures_final_nomenclature_v1/FIGURE_SOURCE_MAP.csv",
                "paper/figures_final_nomenclature_v1/FIGURE_TO_CLAIM_MATRIX.csv",
                "paper/figures_final_nomenclature_v1/FIGURE_QA_REPORT.md",
            }.issubset(paths)
        )
        self.assertEqual(
            {
                path
                for path in paths
                if path.startswith("paper/final_validation_nomenclature_v1/")
            },
            {
                f"paper/final_validation_nomenclature_v1/{name}"
                for name in DERIVED_TABLE_FILENAMES
            },
        )

    def test_case_selection_is_complete_unique_deterministic_and_distance_based(self) -> None:
        first = select_final_case_studies(
            ROOT,
            self.analysis["selected_lambdas"],
            self.analysis["predictions"],
        )
        second = select_final_case_studies(
            ROOT,
            self.analysis["selected_lambdas"],
            self.analysis["predictions"],
        )
        expected = [
            "correct_cough",
            "correct_calm_grunt",
            "correct_feeding",
            "correct_stress_vocal",
            "feeding_stress_boundary_feeding",
            "feeding_stress_boundary_stress",
            "low_predicted_margin",
        ]
        self.assertEqual(first["case_type"].tolist(), expected)
        self.assertEqual(first["source_id"].nunique(), 7)
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        uncertain = first.set_index("case_type").loc["low_predicted_margin"]
        self.assertEqual(
            uncertain["selection_rule"],
            "minimum remaining shared main-prototype predicted-distance margin",
        )
        self.assertEqual(
            set(first["representative_definition"]),
            {"a training sample closest to the predicted class prototype"},
        )

    def test_source_and_claim_maps_cover_every_figure_and_panel(self) -> None:
        sources = build_source_map_rows(ROOT, self.analysis)
        claims = build_claim_rows()
        for rows in (sources, claims):
            covered_figures = set(rows["figure"])
            expected = {
                *(f"Figure {index}" for index in range(1, 5)),
                *(f"Supplementary Figure S{index}" for index in range(1, 8)),
            }
            self.assertEqual(covered_figures, expected)
            self.assertFalse(rows[["figure", "panel"]].duplicated().any())
        self.assertTrue(
            {
                "source_path",
                "source_sha256",
                "deployability",
                "transformation",
                "output_file",
            }.issubset(sources.columns)
        )
        figure2a = sources[
            sources["figure"].eq("Figure 2") & sources["panel"].eq("a")
        ].iloc[0]
        self.assertIn(
            "paper/final_validation/validation_selected_framework_runs.csv",
            figure2a["source_path"],
        )
        figure1c = sources[
            sources["figure"].eq("Figure 1") & sources["panel"].eq("c")
        ].iloc[0]
        self.assertIn("calibration/val_predictions.csv", figure1c["source_path"])
        figure1d_source = sources[
            sources["figure"].eq("Figure 1") & sources["panel"].eq("d")
        ].iloc[0]
        figure1d_claim = claims[
            claims["figure"].eq("Figure 1") & claims["panel"].eq("d")
        ].iloc[0]
        canonical_b3 = "B3 — Hierarchical-prototype Top-1 decision ablation"
        self.assertIn(canonical_b3, figure1d_source["transformation"])
        self.assertIn(canonical_b3, figure1d_claim["claim"])
        figure4a_source = sources[
            sources["figure"].eq("Figure 4") & sources["panel"].eq("a")
        ].iloc[0]
        for route_label in (
            "Primary Softmax route (Raw Softmax)",
            "Main-class prototype candidate route",
            "Hierarchical prototype candidate route",
        ):
            self.assertIn(route_label, figure4a_source["transformation"])

    def test_sha_audit_inventory_includes_every_loaded_frozen_source(self) -> None:
        source_map = build_source_map_rows(ROOT, self.analysis)
        audited = set(collect_frozen_source_paths(ROOT, self.analysis, source_map))
        self.assertTrue(set(self.analysis["source_paths"]).issubset(audited))
        self.assertGreaterEqual(len(audited), len(self.analysis["source_paths"]))
        self.assertTrue(all(path.is_file() for path in audited))

    def test_pre_analysis_sha_inventory_covers_every_loaded_source(self) -> None:
        discovered = set(discover_pre_analysis_source_paths(ROOT))
        self.assertTrue(set(self.analysis["source_paths"]).issubset(discovered))
        self.assertTrue(all(path.is_file() for path in discovered))

    def test_real_four_format_export_and_inspection_in_temporary_directory(self) -> None:
        configure_style()
        figure = plt.figure(
            figsize=(183.0 / 25.4, 45.0 / 25.4), facecolor="white"
        )
        axis = figure.add_subplot(111)
        axis.plot([0.0, 1.0], [0.0, 1.0])
        axis.text(0.5, 0.4, "editable text", ha="center")
        try:
            with TemporaryDirectory() as temporary:
                paths = save_figure_bundle(figure, Path(temporary), "smoke")
                self.assertEqual(
                    {path.suffix for path in paths},
                    {".svg", ".pdf", ".png", ".tiff"},
                )
                self.assertTrue(all(inspect_export(path)["status"] == "PASS" for path in paths))
                svg_text = next(path for path in paths if path.suffix == ".svg").read_text(
                    encoding="utf-8"
                )
                self.assertIsNone(
                    re.search(r"[ \t]+$", svg_text, flags=re.MULTILINE)
                )
        finally:
            plt.close(figure)

    def test_atomic_promotion_rolls_back_if_a_move_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "paper" / ".stage"
            stage_figures = stage / "figures_final"
            stage_validation = stage / "final_validation"
            stage_figures.mkdir(parents=True)
            stage_validation.mkdir(parents=True)
            (stage_figures / "figure.svg").write_text("figure", encoding="utf-8")
            final_validation = root / "paper" / "final_validation_nomenclature_v1"
            final_validation.mkdir(parents=True)
            table_pairs = []
            for name in DERIVED_TABLE_FILENAMES:
                source = stage_validation / name
                source.write_text("metric,value\nexample,1\n", encoding="utf-8")
                table_pairs.append((source, final_validation / name))

            calls = 0

            def fail_on_second_move(source: Path, target: Path) -> Path:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected promotion failure")
                return source.replace(target)

            with patch(
                "tools.generate_final_functional_figure_package._atomic_replace",
                side_effect=fail_on_second_move,
            ):
                with self.assertRaisesRegex(OSError, "injected promotion failure"):
                    _promote_staged_outputs(
                        root=root,
                        staged_figure_dir=stage_figures,
                        table_pairs=tuple(table_pairs),
                    )
            self.assertFalse(
                (root / "paper" / "figures_final_nomenclature_v1").exists()
            )
            self.assertTrue(
                all(not (final_validation / name).exists() for name in DERIVED_TABLE_FILENAMES)
            )

    def test_legends_define_statistics_deployability_and_limits(self) -> None:
        english, chinese = build_legend_texts(self.analysis)
        for token in (
            "n=",
            "mean",
            "SD",
            "fold-cluster bootstrap 95% CI",
            "Wilcoxon",
            "Holm",
            "deployable",
            "label-aware",
            "does not establish",
            "B3 — Hierarchical-prototype Top-1 decision ablation does not improve Top-1",
            "B3 — Hierarchical-prototype Top-1 decision ablation; Fixed "
            "λ=0.5 retrospective",
            "a training sample closest to the predicted class prototype",
        ):
            self.assertIn(token, english)
        for route_label in (
            "Primary Softmax route (Raw Softmax)",
            "Main-class prototype candidate route",
            "Hierarchical prototype candidate route",
        ):
            self.assertIn(route_label, english)
        figure1_legend = english.split("## Figure 2", maxsplit=1)[0]
        for route_label in (
            "Primary Softmax route (Raw Softmax)",
            "Main-class prototype candidate route",
            "Hierarchical prototype candidate route",
        ):
            self.assertIn(route_label, figure1_legend)
        self.assertIn("Holm", chinese)
        for token in ("可部署", "标签感知", "不证明"):
            self.assertIn(token, chinese)
        for corrupted in ("部署（閮ㄧ讲）", "标签（鏍囩）"):
            self.assertNotIn(corrupted, chinese)

    def test_figure_builders_return_four_main_and_seven_supplementary_figures(self) -> None:
        figures = build_all_figures(ROOT, self.analysis)
        try:
            self.assertEqual(
                tuple(figures),
                MAIN_FIGURE_STEMS + SUPPLEMENTARY_FIGURE_STEMS,
            )
            for figure in figures.values():
                self.assertAlmostEqual(figure.get_figwidth() * 25.4, 183.0, places=4)
                self.assertGreater(len(figure.axes), 0)
                self.assertEqual(tuple(figure.get_facecolor()), (1.0, 1.0, 1.0, 1.0))
            primary_axis = figures["figure2_primary_classification_evidence"].axes[0]
            self.assertLessEqual(primary_axis.get_ylim()[0], 0.0)
            self.assertGreaterEqual(primary_axis.get_ylim()[1], 1.0)
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_primary_figure_shows_all_runs_exact_adjacent_contrasts_and_p_values(self) -> None:
        figures = build_all_figures(ROOT, self.analysis)
        try:
            figure = figures["figure2_primary_classification_evidence"]
            panel_a, panel_b, panel_c = figure.axes[:3]
            point_count = sum(
                len(collection.get_offsets())
                for collection in panel_a.collections
                if hasattr(collection, "get_offsets")
            )
            self.assertGreaterEqual(point_count, 25 * 4)
            comparison_text = " ".join(
                tick.get_text() for tick in panel_b.get_yticklabels()
            )
            compact_comparisons = comparison_text.replace(" ", "").replace("\n", "")
            for comparison in ("B1-B0", "B2-B1"):
                self.assertIn(comparison, compact_comparisons)
            self.assertIn(
                "B3 — Hierarchical-prototype Top-1 decision ablation",
                " ".join(comparison_text.split()),
            )
            annotations = " ".join(text.get_text() for text in panel_b.texts)
            self.assertIn("P=", annotations)
            self.assertIn("Holm", annotations)
            self.assertIn("locked 7-test family", annotations)
            selected_points = sum(
                len(collection.get_offsets())
                for collection in panel_c.collections
                if hasattr(collection, "get_offsets")
            )
            self.assertGreaterEqual(selected_points, 5)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            canvas = figure.bbox
            b3_notes = [
                text
                for text in panel_c.texts
                if "Hierarchical-prototype Top-1 decision ablation"
                in " ".join(text.get_text().split())
            ]
            self.assertEqual(len(b3_notes), 1)
            note_bounds = b3_notes[0].get_window_extent(renderer)
            self.assertGreaterEqual(note_bounds.x0, canvas.x0 - 1.0)
            self.assertLessEqual(note_bounds.x1, canvas.x1 + 1.0)
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_primary_holm_values_preserve_the_locked_final_validation_family(self) -> None:
        locked = pd.read_csv(
            ROOT
            / "paper/final_validation/validation_selected_framework_paired_stats.csv"
        ).set_index("comparison")
        paired = self.analysis["framework_paired"].set_index("comparison")
        for comparison in ("B2_valsel - B1", "B3_valsel - B2_valsel"):
            self.assertAlmostEqual(
                float(paired.loc[comparison, "holm_adjusted_p"]),
                float(locked.loc[comparison, "holm_adjusted_p"]),
            )
            self.assertEqual(
                paired.loc[comparison, "holm_family"],
                "locked_final_validation_7_comparisons",
            )
        self.assertTrue(np.isnan(float(paired.loc["B1 - B0", "holm_adjusted_p"])))
        self.assertEqual(
            paired.loc["B1 - B0", "holm_family"],
            "not_applicable_prespecified_duration_contrast",
        )

    def test_framework_figure_shows_primary_and_complete_parallel_diagnostic_chain(self) -> None:
        figures = build_all_figures(ROOT, self.analysis)
        try:
            figure = figures["figure1_final_framework"]
            visible = " ".join(
                [
                    *(text.get_text() for axis in figure.axes for text in axis.texts),
                    *(axis.get_title(loc="left") for axis in figure.axes),
                ]
            ).lower()
            visible = " ".join(visible.split())
            for token in (
                "2 s",
                "log-mel",
                "shared hierarchical crnn",
                "primary softmax route (raw softmax)",
                "frozen embedding",
                "train-only",
                "main prototypes",
                "subtype prototypes",
                "top-k candidates",
                "distances",
                "predicted top-1/top-2 distance margin",
                "hierarchy consistency",
                "prototype representative",
                "main-class prototype candidate route",
                "hierarchical prototype candidate route",
                "b3 — hierarchical-prototype top-1 decision ablation",
            ):
                self.assertIn(token, visible)
            route_labels = [text.get_text() for text in figure.axes[3].texts]
            self.assertTrue(
                any("\n" in label and "Main-class" in label for label in route_labels)
            )
            self.assertEqual(route_labels.count("Candidate"), 2)
            self.assertEqual(route_labels.count("Ablation"), 1)
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_functional_summary_retains_disagreements_and_explicit_harm_balance(self) -> None:
        summary = self.analysis["functional_final"]
        disagreements = summary[
            summary["evidence_family"].eq("disagreement_analysis")
        ]
        self.assertEqual(
            set(disagreements["method"]),
            {"main_prototype", "hierarchical_prototype"},
        )
        self.assertTrue(
            {
                "raw_correct_prototype_wrong",
                "prototype_correct_raw_wrong",
                "both_wrong",
                "net_harm_minus_rescue",
            }.issubset(disagreements["metric"])
        )
        hierarchical = disagreements[
            disagreements["method"].eq("hierarchical_prototype")
        ].set_index("metric")
        self.assertGreater(
            float(
                hierarchical.loc[
                    "raw_correct_prototype_wrong", "repeated_prediction_total"
                ]
            ),
            float(
                hierarchical.loc[
                    "prototype_correct_raw_wrong", "repeated_prediction_total"
                ]
            ),
        )
        conclusion = " ".join(disagreements["interpretation"].dropna()).lower()
        self.assertIn(
            "hierarchical prototype candidate route harmed more", conclusion
        )

    def test_functional_figure_uses_the_requested_four_panel_evidence_logic(self) -> None:
        figures = build_all_figures(ROOT, self.analysis)
        try:
            figure = figures["figure3_prototype_functional_value"]
            titles = [
                axis.get_title(loc="left").lower()
                for axis in figure.axes[:4]
            ]
            self.assertIn("rank-1", titles[0])
            self.assertIn("rank-2", titles[0])
            self.assertIn("feeding", titles[1])
            self.assertIn("stress", titles[1])
            self.assertIn("inconsistency", titles[2])
            self.assertIn("prevalence", titles[2])
            self.assertIn("predicted-distance", titles[3])
            self.assertIn("auroc", titles[3])
            for axis in figure.axes[:3]:
                self.assertTrue(
                    any(
                        container.__class__.__name__ == "ErrorbarContainer"
                        for container in axis.containers
                    )
                )
            panel_c_text = " ".join(
                text.get_text() for text in figure.axes[2].texts
            )
            self.assertIn("%", panel_c_text)
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_case_figure_displays_every_requested_candidate_field(self) -> None:
        figures = build_all_figures(ROOT, self.analysis)
        try:
            figure = figures["figure4_prototype_candidate_cases"]
            visible = " ".join(
                [
                    *(text.get_text() for axis in figure.axes for text in axis.texts),
                    *(axis.get_title(loc="left") for axis in figure.axes),
                    *(tick.get_text() for axis in figure.axes for tick in axis.get_xticklabels()),
                    *(tick.get_text() for axis in figure.axes for tick in axis.get_yticklabels()),
                    *(
                        cell.get_text().get_text()
                        for axis in figure.axes
                        for table in axis.tables
                        for cell in table.get_celld().values()
                    ),
                ]
            ).lower()
            visible = " ".join(visible.split())
            for token in (
                "true class",
                "primary softmax route (raw softmax) top-2",
                "main-class prototype candidate route top-2",
                "hierarchical prototype candidate route subtype top-2",
                "distances",
                "predicted margin",
                "hierarchy consistency",
                "representative",
            ):
                self.assertIn(token, visible)
            for case_id in self.analysis["cases"]["case_id"]:
                self.assertIn(str(case_id).lower(), visible)
            self.assertIn(
                "a training sample closest to the predicted class prototype", visible
            )
            header_texts = [
                cell.get_text().get_text()
                for table in figure.axes[0].tables
                for (row_index, _), cell in table.get_celld().items()
                if row_index == 0
            ]
            self.assertTrue(any("\n" in text for text in header_texts))
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_case_selection_rules_use_canonical_route_display_name(self) -> None:
        selection_rules = " ".join(self.analysis["cases"]["selection_rule"])
        self.assertIn(
            "Primary Softmax route (Raw Softmax) confidence", selection_rules
        )
        self.assertIn(
            "Primary Softmax route (Raw Softmax) Top-2 margin", selection_rules
        )
        self.assertNotIn("raw Softmax", selection_rules)
        self.assertNotIn("raw Top-2", selection_rules)

    def test_package_documentation_uses_canonical_route_and_b3_labels(self) -> None:
        documentation = (
            ROOT / "docs/FINAL_FUNCTIONAL_FIGURE_PACKAGE.md"
        ).read_text(encoding="utf-8")
        for label in (
            "Primary Softmax route (Raw Softmax)",
            "Main-class prototype candidate route",
            "Hierarchical prototype candidate route",
            "B3 — Hierarchical-prototype Top-1 decision ablation",
        ):
            self.assertIn(label, documentation)

    def test_supplements_show_all_fixed_lambda_runs_and_all_demand_environments(self) -> None:
        figures = build_all_figures(ROOT, self.analysis)
        try:
            retrospective = figures[
                "figure_s2_fixed_lambda_retrospective_cumulative"
            ]
            self.assertGreaterEqual(len(retrospective.axes[0].lines), 25)
            self.assertEqual(
                sum("P=" in text.get_text() for text in retrospective.axes[1].texts),
                len(self.analysis["cumulative_paired"]),
            )
            retrospective_labels = " ".join(
                [
                    *(tick.get_text() for tick in retrospective.axes[0].get_xticklabels()),
                    *(tick.get_text() for tick in retrospective.axes[1].get_yticklabels()),
                ]
            )
            retrospective_labels = " ".join(retrospective_labels.split())
            self.assertIn(
                "B3 — Hierarchical-prototype Top-1 decision ablation; "
                "Fixed λ=0.5 retrospective",
                retrospective_labels,
            )
            demand = figures["figure_s5_demand_simulated_noise"]
            visible = " ".join(
                axis.get_title(loc="left") for axis in demand.axes
            ).upper()
            for environment in ("DWASHING", "TBUS", "STRAFFIC"):
                self.assertIn(environment, visible)
            label_aware = figures["figure_s7_label_aware_true_class_margin"]
            for axis in label_aware.axes[:2]:
                self.assertTrue(
                    any(
                        container.__class__.__name__ == "ErrorbarContainer"
                        for container in axis.containers
                    )
                )
        finally:
            for figure in figures.values():
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
