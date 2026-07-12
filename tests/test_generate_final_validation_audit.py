from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tools.generate_final_validation_audit as final_audit
from tools.generate_final_validation_audit import (
    AUX_LABELS,
    EXPECTED_FOLDS,
    EXPECTED_SEEDS,
    MAIN_LABELS,
    build_convergence_figure,
    build_margin_figure,
    compute_run_functional_metrics,
    compute_convergence_tables,
    favorable_distance_margin,
    holm_adjust,
    paired_statistics,
    rank_true_labels,
    refuse_existing,
    required_output_paths,
    safe_error_auroc,
    select_lambda_by_fold,
    sha256_file,
)


def _selection_records(values_by_lambda: dict[float, list[float]], fold: int = 0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate, values in values_by_lambda.items():
        for seed, value in zip(EXPECTED_SEEDS, values, strict=True):
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "lambda": candidate,
                    "best_val_macro_f1": value,
                    "source_path": f"fold{fold}/lambda{candidate}/seed{seed}/summary.json",
                    "sha256": f"{seed:064x}",
                }
            )
    return pd.DataFrame(rows)


class LambdaSelectionTests(unittest.TestCase):
    def test_highest_mean_wins_before_sd(self) -> None:
        records = _selection_records(
            {
                0.2: [0.80, 0.80, 0.80, 0.80, 0.80],
                0.5: [0.79, 0.79, 0.79, 0.79, 0.79],
                1.0: [0.78, 0.78, 0.78, 0.78, 0.78],
            }
        )

        table, selected = select_lambda_by_fold(records)

        self.assertEqual(selected, {0: 0.2})
        self.assertEqual(float(table.loc[0, "selected_lambda"]), 0.2)
        self.assertEqual(table.loc[0, "selection_reason"], "highest_validation_mean")

    def test_sd_breaks_mean_tie_within_tolerance(self) -> None:
        records = _selection_records(
            {
                0.2: [0.78, 0.79, 0.80, 0.81, 0.82],
                0.5: [0.8000004] * 5,
                1.0: [0.70] * 5,
            }
        )

        table, selected = select_lambda_by_fold(records, tie_tolerance=1e-6)

        self.assertEqual(selected, {0: 0.5})
        self.assertEqual(table.loc[0, "selection_reason"], "validation_mean_tie_within_1e-06_then_lowest_sd")

    def test_smallest_lambda_breaks_mean_and_sd_tie(self) -> None:
        records = _selection_records(
            {
                0.2: [0.80] * 5,
                0.5: [0.80] * 5,
                1.0: [0.70] * 5,
            }
        )

        table, selected = select_lambda_by_fold(records)

        self.assertEqual(selected, {0: 0.2})
        self.assertEqual(
            table.loc[0, "selection_reason"],
            "validation_mean_and_sd_tie_then_smallest_lambda",
        )

    def test_missing_seed_is_rejected(self) -> None:
        records = _selection_records(
            {0.2: [0.8] * 5, 0.5: [0.8] * 5, 1.0: [0.8] * 5}
        )
        records = records[records["seed"] != EXPECTED_SEEDS[-1]]

        with self.assertRaisesRegex(ValueError, "exactly five seeds"):
            select_lambda_by_fold(records)

    def test_non_finite_validation_metric_is_rejected(self) -> None:
        records = _selection_records(
            {0.2: [0.8] * 5, 0.5: [0.8] * 5, 1.0: [0.8] * 5}
        )
        records.loc[0, "best_val_macro_f1"] = np.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            select_lambda_by_fold(records)


class StatisticalCoreTests(unittest.TestCase):
    def test_holm_adjust_restores_input_order_and_is_monotone(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_paired_statistics_require_exact_fold_seed_grid(self) -> None:
        rows = [
            {"fold": fold, "seed": seed, "B1": 0.8, "B2_valsel": 0.81}
            for fold in EXPECTED_FOLDS
            for seed in EXPECTED_SEEDS
        ][:-1]

        with self.assertRaisesRegex(ValueError, "exact 25 fold-seed"):
            paired_statistics(
                pd.DataFrame(rows),
                [("B2_valsel", "B1")],
                n_boot=100,
                seed=3407,
            )

    def test_paired_statistics_report_all_requested_fields(self) -> None:
        rows = [
            {
                "fold": fold,
                "seed": seed,
                "B1": 0.80 + fold * 0.001,
                "B2_valsel": 0.81 + fold * 0.001,
                "B3_valsel": 0.82 + fold * 0.001,
            }
            for fold in EXPECTED_FOLDS
            for seed in EXPECTED_SEEDS
        ]

        paired, folds = paired_statistics(
            pd.DataFrame(rows),
            [("B2_valsel", "B1"), ("B3_valsel", "B2_valsel")],
            n_boot=200,
            seed=3407,
        )

        self.assertEqual(len(paired), 2)
        self.assertEqual(len(folds), 10)
        self.assertTrue(
            {
                "n",
                "mean_delta",
                "std_delta",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "wilcoxon_raw_p",
                "holm_adjusted_p",
                "wins",
                "ties",
                "losses",
                "fold_cluster_bootstrap_ci95_low",
                "fold_cluster_bootstrap_ci95_high",
                "fold_0_mean_delta",
                "fold_1_mean_delta",
                "fold_2_mean_delta",
                "fold_3_mean_delta",
                "fold_4_mean_delta",
            }.issubset(paired.columns)
        )
        self.assertTrue((paired["n"] == 25).all())
        self.assertTrue((paired["wins"] == 25).all())
        self.assertTrue((paired["ties"] == 0).all())
        self.assertTrue((paired["losses"] == 0).all())
        np.testing.assert_allclose(paired["mean_delta"], [0.01, 0.01])

    def test_all_zero_delta_has_wilcoxon_p_one(self) -> None:
        rows = [
            {"fold": fold, "seed": seed, "a": 0.8, "b": 0.8}
            for fold in EXPECTED_FOLDS
            for seed in EXPECTED_SEEDS
        ]

        paired, _ = paired_statistics(
            pd.DataFrame(rows), [("a", "b")], n_boot=100, seed=3407
        )

        self.assertEqual(float(paired.loc[0, "wilcoxon_raw_p"]), 1.0)
        self.assertEqual(int(paired.loc[0, "ties"]), 25)


class ProvenanceTests(unittest.TestCase):
    def test_hierarchical_reader_reconciles_canonical_family_and_context(self) -> None:
        payload = {
            "model_family": "hierarchical_supervision_crnn",
            "feature_mode": "logmel",
            "use_se": True,
            "seed": 42,
            "dur_s": 2.0,
            "context_seconds": 2.0,
            "hier_aux": True,
            "hier_aux_weight": 0.5,
            "rnn_type": "gru",
            "pooling_type": "mean",
            "main_labels": list(MAIN_LABELS),
            "aux_labels": list(AUX_LABELS),
            "best_val_macro_f1": 0.95,
            "best_epoch": 4,
            "test_macro_f1": 0.94,
        }

        final_audit._validate_hierarchical_summary(
            payload,
            path=Path("canonical-summary.json"),
            seed=42,
            candidate=0.5,
        )

        conflicting = {**payload, "context_seconds": 1.0}
        with self.assertRaisesRegex(ValueError, "context_seconds=2.0"):
            final_audit._validate_hierarchical_summary(
                conflicting,
                path=Path("conflicting-summary.json"),
                seed=42,
                candidate=0.5,
            )

    def test_sha256_file_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text("frozen", encoding="utf-8")

            digest = sha256_file(path)

        self.assertEqual(
            digest,
            "ffb304816a1090313e833215c08dae3d209cfad1ffd1f674f0909a2ae99e1394",
        )

    def test_refuse_existing_lists_existing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "already-there.csv"
            existing.write_text("x", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already-there.csv"):
                refuse_existing([existing, Path(tmp) / "new.csv"])

    def test_recorded_sha256_is_recomputed_from_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"original")
            recorded = sha256_file(path)
            path.write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                final_audit.verify_recorded_sha256(
                    path, recorded, context="synthetic artifact"
                )

    def test_canonical_raw_probabilities_are_joined_by_frozen_identity(self) -> None:
        manifest = pd.DataFrame(
            {
                "path": [r"C:\\audio\\a.wav", r"C:\\audio\\b.wav"],
                "source_id": ["a", "b"],
                "md5": ["a" * 32, "b" * 32],
                "label": ["cough", "feeding"],
            }
        )
        reference = pd.DataFrame(
            {
                "y_true": [0, 2],
                "y_pred": [0, 2],
                "prob_cough": [0.91, 0.02],
                "prob_calm_grunt": [0.03, 0.03],
                "prob_feeding": [0.04, 0.90],
                "prob_stress_vocal": [0.02, 0.05],
            }
        )
        reproduced = pd.DataFrame(
            {
                "path": [r"C:\\audio\\b.wav", r"C:\\audio\\a.wav"],
                "source_id": ["b", "a"],
                "md5": ["B" * 32, "A" * 32],
                "y_true": ["feeding", "cough"],
                "raw_softmax_pred": ["feeding", "cough"],
                "raw_softmax_prob_cough": [0.01, 0.90],
                "raw_softmax_prob_calm_grunt": [0.02, 0.04],
                "raw_softmax_prob_feeding": [0.92, 0.03],
                "raw_softmax_prob_stress_vocal": [0.05, 0.03],
            }
        )

        aligned, audit = final_audit.align_canonical_raw_probabilities(
            reproduced,
            reference,
            manifest,
            context="synthetic canonical probabilities",
        )

        self.assertEqual(list(aligned["source_id"]), ["b", "a"])
        self.assertAlmostEqual(float(aligned.loc[0, "raw_softmax_prob_feeding"]), 0.90)
        self.assertAlmostEqual(float(aligned.loc[1, "raw_softmax_prob_cough"]), 0.91)
        self.assertEqual(audit["identity_join"], "one_to_one")
        self.assertEqual(audit["prediction_mismatch_count"], 0)

    def test_frozen_membership_rejects_stale_subtype_truth(self) -> None:
        manifest = pd.DataFrame(
            {
                "path": ["a.wav", "b.wav"],
                "source_id": ["a", "b"],
                "md5": ["a" * 32, "b" * 32],
                "label": ["cough", "stress_vocal"],
                "subtype": ["dry_cough", "anxious_stress"],
            }
        )
        artifact = manifest.copy()
        artifact["y_true"] = artifact["label"]
        artifact["aux_true"] = ["dry_cough", "frightened_stress"]

        with self.assertRaisesRegex(ValueError, "Subtype truth"):
            final_audit.validate_frozen_split_membership(
                artifact,
                manifest,
                context="synthetic frozen test",
                y_true_column="y_true",
                aux_true_column="aux_true",
            )


def _set_scores(
    row: dict[str, object], prefix: str, labels: tuple[str, ...], ordered: list[str]
) -> None:
    scores = {label: 0.01 for label in labels}
    for position, label in enumerate(reversed(ordered)):
        scores[label] = 0.5 + position * 0.1
    for label, value in scores.items():
        row[f"{prefix}{label}"] = value


def _functional_fixture() -> pd.DataFrame:
    cases = [
        {
            "source_id": "s1",
            "y_true": "cough",
            "aux_true": "dry_cough",
            "raw": ["cough", "calm_grunt"],
            "proto": ["cough", "feeding"],
            "hier": ["cough", "feeding"],
            "soft_aux": ["dry_cough", "abdominal_cough"],
            "proto_aux": ["dry_cough", "abdominal_cough"],
            "main_dist": [0.10, 0.50, 0.80, 0.90],
            "aux_dist": [0.10, 0.20, 0.70, 0.80, 0.90, 1.00],
        },
        {
            "source_id": "s2",
            "y_true": "calm_grunt",
            "aux_true": "calm_grunt",
            "raw": ["feeding", "calm_grunt"],
            "proto": ["calm_grunt", "feeding"],
            "hier": ["calm_grunt", "feeding"],
            "soft_aux": ["feeding", "calm_grunt"],
            "proto_aux": ["calm_grunt", "feeding"],
            "main_dist": [0.40, 0.20, 0.60, 0.80],
            "aux_dist": [0.80, 0.90, 0.20, 0.40, 0.70, 1.00],
        },
        {
            "source_id": "s3",
            "y_true": "feeding",
            "aux_true": "feeding",
            "raw": ["feeding", "stress_vocal"],
            "proto": ["stress_vocal", "feeding"],
            "hier": ["feeding", "stress_vocal"],
            "soft_aux": ["feeding", "frightened_stress"],
            "proto_aux": ["frightened_stress", "feeding"],
            "main_dist": [0.90, 0.80, 0.60, 0.20],
            "aux_dist": [1.00, 0.90, 0.80, 0.60, 0.20, 0.40],
        },
        {
            "source_id": "s4",
            "y_true": "stress_vocal",
            "aux_true": "anxious_stress",
            "raw": ["feeding", "stress_vocal"],
            "proto": ["calm_grunt", "stress_vocal"],
            "hier": ["stress_vocal", "feeding"],
            "soft_aux": ["feeding", "anxious_stress"],
            "proto_aux": ["calm_grunt", "anxious_stress"],
            "main_dist": [0.90, 0.80, 0.10, 0.70],
            "aux_dist": [1.00, 0.90, 0.20, 0.80, 0.70, 0.60],
        },
    ]
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        row: dict[str, object] = {
            "path": f"audio/{case['source_id']}.wav",
            "source_id": case["source_id"],
            "md5": f"{index + 1:032x}",
            "y_true": case["y_true"],
            "aux_true": case["aux_true"],
        }
        _set_scores(row, "raw_softmax_prob_", MAIN_LABELS, case["raw"])
        _set_scores(row, "prototype_prob_", MAIN_LABELS, case["proto"])
        _set_scores(row, "hierarchical_prob_", MAIN_LABELS, case["hier"])
        _set_scores(row, "softmax_aux_prob_", AUX_LABELS, case["soft_aux"])
        _set_scores(row, "prototype_aux_prob_", AUX_LABELS, case["proto_aux"])
        for label, distance in zip(MAIN_LABELS, case["main_dist"], strict=True):
            row[f"main_proto_cosine_distance_{label}"] = distance
        for label, distance in zip(AUX_LABELS, case["aux_dist"], strict=True):
            row[f"aux_proto_cosine_distance_{label}"] = distance
        rows.append(row)
    return pd.DataFrame(rows)


class PrototypeFunctionalMetricTests(unittest.TestCase):
    def test_true_rank_uses_stable_label_order_for_ties(self) -> None:
        values = np.array([[0.9, 0.9, 0.1], [0.2, 0.3, 0.4]])
        ranks, ties = rank_true_labels(
            values,
            true_labels=["a", "b"],
            labels=("a", "b", "c"),
            higher_is_better=True,
        )

        np.testing.assert_array_equal(ranks, [1, 2])
        self.assertEqual(ties, 1)

    def test_favorable_margin_is_nearest_wrong_minus_true(self) -> None:
        distances = np.array([[0.2, 0.5, 0.7], [0.1, 0.6, 0.4]])

        margin = favorable_distance_margin(
            distances,
            true_labels=["a", "b"],
            labels=("a", "b", "c"),
        )

        np.testing.assert_allclose(margin, [0.3, -0.5])

    def test_error_auroc_handles_one_class_without_inventing_value(self) -> None:
        value, status = safe_error_auroc(
            errors=np.array([False, False]), scores=np.array([0.1, 0.2])
        )

        self.assertTrue(np.isnan(value))
        self.assertEqual(status, "unavailable_single_error_class")

    def test_run_functional_metrics_cover_topk_consistency_margin_and_disagreement(self) -> None:
        functional, margins, disagreements = compute_run_functional_metrics(
            _functional_fixture(), fold=0, seed=42
        )

        lookup = functional.set_index(["method", "metric"])["value"]
        self.assertAlmostEqual(
            float(lookup.loc[("raw_softmax", "main_top1_accuracy")]), 0.5
        )
        self.assertAlmostEqual(
            float(lookup.loc[("main_prototype", "main_top1_accuracy")]), 0.5
        )
        self.assertAlmostEqual(
            float(lookup.loc[("hierarchical_prototype", "main_top1_accuracy")]),
            1.0,
        )
        self.assertAlmostEqual(
            float(lookup.loc[("hierarchical_prototype", "main_top2_accuracy")]),
            1.0,
        )
        self.assertIn("hierarchy_consistency_rate", set(functional["metric"]))
        self.assertIn("feeding_stress_exact_pair_top2_coverage", set(functional["metric"]))
        self.assertEqual(set(margins["level"]), {"main", "subtype"})
        self.assertTrue(np.isfinite(margins["error_detection_auroc"].dropna()).all())

        main = disagreements.set_index("prototype_method")
        self.assertEqual(int(main.loc["main_prototype", "n_disagreements"]), 3)
        self.assertEqual(
            int(main.loc["main_prototype", "raw_correct_prototype_wrong"]), 1
        )
        self.assertEqual(
            int(main.loc["main_prototype", "prototype_correct_raw_wrong"]), 1
        )
        self.assertEqual(int(main.loc["main_prototype", "both_wrong"]), 1)


class OutputAndConvergenceTests(unittest.TestCase):
    def test_cli_entrypoint_runs_after_all_function_definitions(self) -> None:
        source = Path(final_audit.__file__).read_text(encoding="utf-8")

        self.assertGreater(
            source.rfind('if __name__ == "__main__":'),
            source.rfind("\ndef "),
        )

    def test_required_output_paths_use_versioned_nonhistorical_contract(self) -> None:
        output = Path("paper/final_validation_audit_nomenclature_v1")

        paths = {path.as_posix() for path in required_output_paths(output)}

        required = {
            "paper/final_validation_audit_nomenclature_v1/lambda_selection_by_fold.csv",
            "paper/final_validation_audit_nomenclature_v1/lambda_selection_provenance.json",
            "paper/final_validation_audit_nomenclature_v1/validation_selected_framework_runs.csv",
            "paper/final_validation_audit_nomenclature_v1/validation_selected_framework_summary.csv",
            "paper/final_validation_audit_nomenclature_v1/validation_selected_framework_paired_stats.csv",
            "paper/final_validation_audit_nomenclature_v1/validation_selected_framework_fold_stats.csv",
            "paper/final_validation_audit_nomenclature_v1/prototype_functional_value_summary.csv",
            "paper/final_validation_audit_nomenclature_v1/prototype_margin_error_detection.csv",
            "paper/final_validation_audit_nomenclature_v1/prototype_disagreement_analysis.csv",
            "paper/final_validation_audit_nomenclature_v1/convergence_best_epoch_summary.csv",
            "paper/final_validation_audit_nomenclature_v1/validation_test_gap_summary.csv",
            "paper/final_validation_audit_nomenclature_v1/FINAL_VALIDATION_REPORT.md",
        }
        figure_stems = {
            "lambda_selection_by_fold",
            "validation_selected_cumulative_framework",
            "prototype_functional_value",
            "prototype_margin_correct_vs_error",
            "best_epoch_and_val_test_gap",
        }
        required.update(
            f"paper/final_validation_audit_nomenclature_v1/figures/{stem}.{extension}"
            for stem in figure_stems
            for extension in ("svg", "pdf", "png")
        )
        self.assertEqual(paths, required)

    def test_b3_convergence_inherits_selected_b2_checkpoint(self) -> None:
        rows = []
        for fold in EXPECTED_FOLDS:
            for seed in EXPECTED_SEEDS:
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "B0_best_epoch": 5,
                        "B1_best_epoch": 6,
                        "B2_valsel_best_epoch": 7,
                        "B3_valsel_best_epoch": 7,
                        "B0_best_val_macro_f1": 0.80,
                        "B1_best_val_macro_f1": 0.85,
                        "B2_valsel_best_val_macro_f1": 0.86,
                        "B3_valsel_best_val_macro_f1": 0.86,
                        "B0": 0.78,
                        "B1": 0.83,
                        "B2_valsel": 0.84,
                        "B3_valsel": 0.85,
                        "B0_source": "b0.json",
                        "B1_source": "b1.json",
                        "B2_valsel_source": "b2.json",
                        "B3_valsel_source": "b3.json",
                    }
                )

        epochs, gaps, log_audit = compute_convergence_tables(pd.DataFrame(rows))

        self.assertEqual(len(epochs), 100)
        self.assertEqual(len(gaps), 100)
        b3 = epochs[epochs["stage"] == "B3"]
        self.assertTrue((b3["best_epoch"] == 7).all())
        self.assertTrue(
            (
                b3["epoch_provenance"]
                == "inherited_from_selected_B2_checkpoint"
            ).all()
        )
        self.assertTrue((b3["source"] == "b2.json").all())
        self.assertFalse(log_audit["complete_epoch_logs"])
        self.assertFalse(log_audit["training_curves_emitted"])

        _, margin_frame, _ = compute_run_functional_metrics(
            _functional_fixture(), fold=0, seed=42
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            margin_figure = build_margin_figure(margin_frame)
            convergence_figure = build_convergence_figure(epochs, gaps)
            convergence_figure.canvas.draw()
            renderer = convergence_figure.canvas.get_renderer()
            canvas = convergence_figure.bbox
            convergence_artists = [
                *convergence_figure.texts,
                *(
                    text
                    for axis in convergence_figure.axes
                    for text in (
                        *axis.texts,
                        *axis.get_xticklabels(),
                        *axis.get_yticklabels(),
                        axis.xaxis.label,
                        axis.yaxis.label,
                        axis.title,
                    )
                    if text.get_visible() and text.get_text()
                ),
                *(
                    legend
                    for axis in convergence_figure.axes
                    if (legend := axis.get_legend()) is not None
                ),
            ]
            for artist in convergence_artists:
                bounds = artist.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, canvas.x0 - 1.0)
                self.assertGreaterEqual(bounds.y0, canvas.y0 - 1.0)
                self.assertLessEqual(bounds.x1, canvas.x1 + 1.0)
                self.assertLessEqual(bounds.y1, canvas.y1 + 1.0)
            plt.close(margin_figure)
            plt.close(convergence_figure)
        matplotlib_deprecations = [
            warning
            for warning in caught
            if issubclass(warning.category, mpl.MatplotlibDeprecationWarning)
        ]
        self.assertEqual(matplotlib_deprecations, [])
        layout_warnings = [
            warning
            for warning in caught
            if "layout" in str(warning.message).lower()
            and (
                "not applied" in str(warning.message).lower()
                or "collapsed" in str(warning.message).lower()
            )
        ]
        self.assertEqual(layout_warnings, [])

    def test_epoch_log_coverage_is_searched_not_hard_coded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for fold in EXPECTED_FOLDS:
                for seed in EXPECTED_SEEDS:
                    sources = {}
                    for stage in ("B0", "B1", "B2"):
                        source = root / "reports" / stage / f"fold{fold}_seed{seed}" / "summary.json"
                        source.parent.mkdir(parents=True, exist_ok=True)
                        source.write_text("{}", encoding="utf-8")
                        sources[stage] = source.relative_to(root).as_posix()
                    rows.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "B0_best_epoch": 5,
                            "B1_best_epoch": 6,
                            "B2_valsel_best_epoch": 7,
                            "B3_valsel_best_epoch": 7,
                            "B0_best_val_macro_f1": 0.80,
                            "B1_best_val_macro_f1": 0.85,
                            "B2_valsel_best_val_macro_f1": 0.86,
                            "B3_valsel_best_val_macro_f1": 0.86,
                            "B0": 0.78,
                            "B1": 0.83,
                            "B2_valsel": 0.84,
                            "B3_valsel": 0.85,
                            "B0_source": sources["B0"],
                            "B1_source": sources["B1"],
                            "B2_valsel_source": sources["B2"],
                            "B3_valsel_source": "post_hoc_metrics.json",
                        }
                    )
            history = root / "reports" / "B0" / "fold0_seed42" / "history.csv"
            pd.DataFrame(
                {
                    "epoch": [1, 2],
                    "train_loss": [1.0, 0.5],
                    "val_macro_f1": [0.7, 0.8],
                }
            ).to_csv(history, index=False)

            _, _, log_audit = compute_convergence_tables(
                pd.DataFrame(rows), root=root
            )

        self.assertEqual(log_audit["epoch_log_coverage"]["B0"], "1/25")
        self.assertEqual(log_audit["epoch_log_coverage"]["B1"], "0/25")
        self.assertIn(history.as_posix(), log_audit["complete_history_files"])
        self.assertFalse(log_audit["complete_epoch_logs"])

    def test_real_validate_only_enforces_locked_integration_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        required = root / "reports" / "prototype_cv5_exact_valsel_w10_fold0_seed42" / "evaluation" / "metrics.json"
        if not required.is_file():
            self.skipTest("Local frozen final-validation artifacts are unavailable")

        result = final_audit.generate(
            root,
            Path("paper/final_validation_audit_nomenclature_v1"),
            n_boot=100,
            seed=3407,
            validate_only=True,
        )

        self.assertEqual(
            result["selection"].set_index("fold")["selected_lambda"].to_dict(),
            {0: 1.0, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.5},
        )
        self.assertEqual(len(result["runs"]), 25)
        self.assertEqual(len(result["paired"]), 7)
        self.assertEqual(result["provenance"]["selection_source_count"], 75)
        self.assertIn(
            "paper/final_validation_audit_nomenclature_v1/",
            result["report"],
        )
        self.assertTrue(
            result["provenance"]["artifact_hash_link_verification"][
                "all_selected_runs_verified"
            ]
        )
        self.assertTrue(
            result["provenance"]["split_membership_verification"][
                "path_source_id_md5_main_subtype_exact"
            ]
        )
        self.assertEqual(
            result["provenance"]["split_membership_verification"][
                "test_subtype_truth_mismatches"
            ],
            0,
        )
        self.assertEqual(
            result["provenance"]["softmax_reproduction"][
                "canonical_truth_mismatch_count"
            ],
            0,
        )
        self.assertEqual(
            result["provenance"]["softmax_reproduction"][
                "canonical_prediction_mismatch_count"
            ],
            0,
        )
        self.assertEqual(
            result["provenance"]["epoch_log_audit"]["epoch_log_coverage"],
            {
                "B0": "0/25",
                "B1": "0/25",
                "B2": "0/25",
                "B3": "not_applicable_post_hoc",
            },
        )
        self.assertTrue(result["source_hashes_unchanged"])
        self.assertEqual(result["acceptance"], "framework_functional_only")
        self.assertIn(
            "a training sample closest to the predicted class prototype",
            result["report"],
        )
        self.assertIn(
            "Primary Softmax route (Raw Softmax)", result["report"]
        )
        self.assertIn(
            "B3 — Hierarchical-prototype Top-1 decision ablation",
            result["report"],
        )
        self.assertNotIn("| raw_softmax |", result["report"])
        self.assertNotIn("| B3_valsel |", result["report"])
        forbidden_wording = "the query's nearest training " + "neighbour"
        self.assertNotIn(forbidden_wording, result["report"])


if __name__ == "__main__":
    unittest.main()
