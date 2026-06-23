import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from prototype_model_adapter import (  # noqa: E402
    DEFAULT_AUX_LABELS,
    DEFAULT_MAIN_LABELS,
    HierModelConfig,
    apply_hierarchy_confidence_penalty,
    align_prediction_frames_by_path,
    assert_no_leakage,
    audit_manifest_disjointness,
    calibration_relevant_parameters,
    calibration_float_param,
    calibration_metrics,
    compute_class_prototypes,
    compute_distance_statistics,
    coverage_risk_curve,
    file_sha256,
    fusion_mode_from_alpha,
    make_numpy_logmel_feature,
    map_aux_probabilities_to_main,
    multiclass_metrics,
    normalize_identity_path,
    result_qualification_fields,
    select_prediction_input_role,
    training_exact_feature_config,
    fusion_formula_metadata,
    prototype_scores_from_bundle,
    validate_expected_config,
    verify_manifest_matches_metadata,
)
from eval_hier_acoustic_prototype import load_and_validate_prediction_metadata  # noqa: E402
from summarize_cv5_exact_prototype import (  # noqa: E402
    aggregate_confusion_matrices,
    aggregate_method_metrics,
    paired_macro_f1_statistics,
    validate_aggregate_records,
)


class PrototypePipelineCoreTests(unittest.TestCase):
    def _aggregate_record(
        self,
        fold=0,
        seed=42,
        lambda_value=0.5,
        *,
        raw_macro=0.80,
        calibrated_macro=0.81,
        prototype_macro=0.82,
        hierarchical_macro=0.84,
        fused_macro=0.83,
        confusion_matrices_json=None,
    ):
        methods = {
            "raw_softmax": {
                "macro_f1": raw_macro,
                "top1_acc": raw_macro - 0.10,
                "top2_acc": 0.95,
                "ece": 0.10,
                "brier": 0.20,
                "nll": 0.30,
            },
            "calibrated_softmax": {
                "macro_f1": calibrated_macro,
                "top1_acc": calibrated_macro - 0.10,
                "top2_acc": 0.96,
                "ece": 0.09,
                "brier": 0.19,
                "nll": 0.29,
            },
            "prototype": {
                "macro_f1": prototype_macro,
                "top1_acc": prototype_macro - 0.10,
                "top2_acc": 0.97,
                "ece": 0.08,
                "brier": 0.18,
                "nll": 0.28,
            },
            "hierarchical": {
                "macro_f1": hierarchical_macro,
                "top1_acc": hierarchical_macro - 0.10,
                "top2_acc": 0.98,
                "ece": 0.07,
                "brier": 0.17,
                "nll": 0.27,
            },
            "fused": {
                "macro_f1": fused_macro,
                "top1_acc": fused_macro - 0.10,
                "top2_acc": 0.99,
                "ece": 0.06,
                "brier": 0.16,
                "nll": 0.26,
            },
        }
        return {
            "fold": fold,
            "seed": seed,
            "lambda": lambda_value,
            "eligible_for_cv_aggregation": True,
            "feature_backend": "training_exact",
            "feature_pipeline_equivalent": True,
            "input_role": "frozen_test",
            "unsafe_allow_checkpoint_sha_mismatch": False,
            "leakage_audit_ok": True,
            "manifest_sha_verified": True,
            "softmax_reproduction_passed": True,
            "methods": methods,
            "confusion_matrices_json": confusion_matrices_json,
        }

    def _aggregate_records(self, seeds, **kwargs):
        return [self._aggregate_record(fold=fold, seed=seed, **kwargs) for fold in range(5) for seed in seeds]

    def test_compute_class_prototypes_uses_normalized_class_means(self):
        embeddings = np.array(
            [
                [2.0, 0.0],
                [4.0, 0.0],
                [0.0, 3.0],
                [0.0, 6.0],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1], dtype=np.int64)

        prototypes, counts = compute_class_prototypes(embeddings, labels, num_classes=2)

        np.testing.assert_allclose(prototypes[0], [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(prototypes[1], [0.0, 1.0], atol=1e-6)
        np.testing.assert_array_equal(counts, [2, 2])

    def test_compute_class_prototypes_rejects_missing_class(self):
        embeddings = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        labels = np.array([0, 0], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "missing classes"):
            compute_class_prototypes(embeddings, labels, num_classes=2)

    def test_distance_statistics_use_train_embeddings_only_geometry(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [0.6, 0.8],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        prototypes, _ = compute_class_prototypes(embeddings, labels, num_classes=2)

        stats = compute_distance_statistics(
            embeddings,
            labels,
            prototypes,
            class_labels=["a", "b"],
            split_name="train",
        )

        self.assertEqual(stats["split"], "train")
        self.assertEqual(stats["classes"]["a"]["count"], 2)
        self.assertIn("cosine_distance", stats["classes"]["a"])
        self.assertIn("euclidean_distance", stats["classes"]["a"])
        self.assertIn("q95", stats["classes"]["a"]["cosine_distance"])

    def test_prototype_scores_include_similarity_and_distance_columns(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        bundle = {
            "main_prototypes": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "aux_prototypes": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "main_labels": ["calm_grunt", "feeding"],
            "aux_labels": ["calm_grunt", "feeding"],
        }

        scores = prototype_scores_from_bundle(embeddings, bundle, temperature=0.5, hier_aux_weight=0.5)

        np.testing.assert_allclose(scores["main_cosine_similarity"][0], [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(scores["main_cosine_distance"][0], [0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(scores["main_euclidean_distance"][0], [0.0, np.sqrt(2.0)], atol=1e-6)
        self.assertEqual(scores["nearest_main_id"].tolist(), [0, 1])

    def test_hierarchy_penalty_lowers_only_inconsistent_confidence(self):
        confidence = np.array([0.9, 0.8], dtype=np.float32)
        main_pred = ["cough", "feeding"]
        aux_pred = ["dry_cough", "frightened_stress"]

        adjusted, inconsistent = apply_hierarchy_confidence_penalty(
            confidence,
            main_pred,
            aux_pred,
            penalty=0.5,
        )

        np.testing.assert_allclose(adjusted, [0.9, 0.4], atol=1e-6)
        self.assertEqual(inconsistent.tolist(), [False, True])

    def test_validate_expected_config_rejects_wrong_phase1_values(self):
        config = HierModelConfig(
            main_labels=list(DEFAULT_MAIN_LABELS),
            aux_labels=list(DEFAULT_AUX_LABELS),
            feature_mode="logmel",
            seed=3407,
            dur_s=2.0,
            hier_aux_weight=1.0,
        )

        with self.assertRaisesRegex(ValueError, "hier_aux_weight"):
            validate_expected_config(
                config,
                fold=0,
                seed=3407,
                expected_hier_aux_weight=0.5,
                expected_dur_s=2.0,
                expected_feature_mode="logmel",
                expected_main_labels=DEFAULT_MAIN_LABELS,
                expected_aux_labels=DEFAULT_AUX_LABELS,
            )

    def test_training_exact_feature_config_records_librosa_implicit_params(self):
        config = HierModelConfig(
            main_labels=list(DEFAULT_MAIN_LABELS),
            aux_labels=list(DEFAULT_AUX_LABELS),
            feature_mode="logmel",
            seed=3407,
            dur_s=2.0,
            hier_aux_weight=0.5,
        )

        feature_config = training_exact_feature_config(config)

        self.assertEqual(feature_config["feature_backend"], "training_exact")
        self.assertEqual(feature_config["feature_mode"], "logmel")
        self.assertEqual(feature_config["window"], "hann")
        self.assertTrue(feature_config["center"])
        self.assertEqual(feature_config["pad_mode"], "constant")
        self.assertEqual(feature_config["power"], 2.0)
        self.assertFalse(feature_config["mel_htk"])
        self.assertEqual(feature_config["mel_norm"], "slaney")
        self.assertEqual(feature_config["power_to_db_ref"], 1.0)
        self.assertEqual(feature_config["power_to_db_top_db"], 80.0)
        self.assertEqual(feature_config["norm_map"], "per-sample z-score with eps=1e-6")

    def test_fusion_formula_metadata_explains_weight_direction(self):
        meta = fusion_formula_metadata()

        self.assertIn("alpha * softmax", meta["fusion_formula"])
        self.assertEqual(meta["fusion_weight_alpha_0"], "pure prototype")
        self.assertEqual(meta["fusion_weight_alpha_1"], "pure softmax")
        self.assertIn("penalized_confidence", meta["hierarchy_penalty_formula"])

    def test_calibration_relevant_parameters_drop_irrelevant_knobs(self):
        hierarchical = calibration_relevant_parameters(
            "hierarchical",
            prototype_temperature=0.1,
            softmax_temperature=2.0,
            softmax_weight=0.75,
            hier_aux_prob_weight=0.5,
        )
        prototype = calibration_relevant_parameters(
            "prototype",
            prototype_temperature=0.2,
            softmax_temperature=0.5,
            softmax_weight=0.25,
            hier_aux_prob_weight=0.5,
        )

        self.assertIsNone(hierarchical["softmax_temperature"])
        self.assertIsNone(hierarchical["softmax_weight"])
        self.assertIsNone(prototype["softmax_weight"])

    def test_fusion_mode_from_alpha_marks_pure_endpoints(self):
        self.assertEqual(fusion_mode_from_alpha(0.0), "pure_prototype")
        self.assertEqual(fusion_mode_from_alpha(1.0), "pure_softmax")
        self.assertEqual(fusion_mode_from_alpha(0.25), "mixed")

    def test_calibration_float_param_preserves_zero_alpha(self):
        calibration = {
            "softmax_weight": 0.75,
            "method_best_params": {
                "fused": {
                    "softmax_weight": 0.0,
                }
            },
        }

        alpha = calibration_float_param(
            calibration,
            "fused",
            "softmax_weight",
            fallback_keys=("softmax_weight",),
            default=1.0,
        )

        self.assertEqual(alpha, 0.0)

    def test_calibration_float_param_preserves_zero_hier_aux_weight(self):
        calibration = {
            "hier_aux_prob_weight": 0.5,
            "method_best_params": {
                "hierarchical": {
                    "hier_aux_prob_weight": 0.0,
                }
            },
        }

        weight = calibration_float_param(
            calibration,
            "hierarchical",
            "hier_aux_prob_weight",
            fallback_keys=("hier_aux_prob_weight",),
            default=0.5,
        )

        self.assertEqual(weight, 0.0)

    def test_select_prediction_input_role_requires_exactly_one_input(self):
        self.assertEqual(
            select_prediction_input_role(test_manifest="test.csv", manifest="", audio="")["input_role"],
            "frozen_test",
        )
        self.assertEqual(
            select_prediction_input_role(test_manifest="", manifest="infer.csv", audio="")["input_role"],
            "inference_manifest",
        )
        self.assertEqual(
            select_prediction_input_role(test_manifest="", manifest="", audio="x.wav")["input_role"],
            "single_audio",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_prediction_input_role(test_manifest="test.csv", manifest="infer.csv", audio="")

    def test_result_qualification_fields_mark_fold0_debug_not_paper_main(self):
        exact = result_qualification_fields(
            "training_exact",
            fold=0,
            seed=3407,
            input_role="frozen_test",
            unsafe_allow_checkpoint_sha_mismatch=False,
            manifest_sha_verified=True,
            leakage_audit_ok=True,
        )

        self.assertTrue(exact["feature_pipeline_equivalent"])
        self.assertTrue(exact["single_fold_debug"])
        self.assertTrue(exact["eligible_for_cv_aggregation"])
        self.assertFalse(exact["paper_main_result"])
        self.assertEqual(exact["feature_backend"], "training_exact")
        self.assertEqual(exact["run_scope"], "fold_seed")

    def test_result_qualification_fields_never_mark_single_fold_paper_main(self):
        exact = result_qualification_fields(
            "training_exact",
            fold=1,
            seed=42,
            input_role="frozen_test",
            unsafe_allow_checkpoint_sha_mismatch=False,
            manifest_sha_verified=True,
            leakage_audit_ok=True,
        )

        self.assertTrue(exact["eligible_for_cv_aggregation"])
        self.assertFalse(exact["paper_main_result"])
        self.assertEqual(exact["run_scope"], "fold_seed")

    def test_result_qualification_fields_reject_numpy_for_cv_aggregation(self):
        numpy = result_qualification_fields(
            "numpy_logmel",
            fold=2,
            seed=1,
            input_role="frozen_test",
            unsafe_allow_checkpoint_sha_mismatch=False,
            manifest_sha_verified=True,
            leakage_audit_ok=True,
        )

        self.assertFalse(numpy["feature_pipeline_equivalent"])
        self.assertFalse(numpy["eligible_for_cv_aggregation"])
        self.assertFalse(numpy["paper_main_result"])

    def test_result_qualification_fields_require_frozen_test_for_prediction_aggregation(self):
        infer = result_qualification_fields(
            "training_exact",
            fold=2,
            seed=1,
            input_role="inference_manifest",
            unsafe_allow_checkpoint_sha_mismatch=False,
            manifest_sha_verified=True,
            leakage_audit_ok=True,
        )

        self.assertFalse(infer["eligible_for_cv_aggregation"])
        self.assertFalse(infer["paper_main_result"])

    def test_evaluation_requires_prediction_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "test_predictions.csv"
            cal = root / "calibration.json"
            pred.write_text("path,y_true,y_pred\n", encoding="utf-8")
            cal.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "prediction_metadata.json"):
                load_and_validate_prediction_metadata(pred, cal)

    def test_evaluation_rejects_inference_manifest_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "test_predictions.csv"
            cal = root / "calibration.json"
            meta = root / "prediction_metadata.json"
            pred.write_text("path,y_true,y_pred\n", encoding="utf-8")
            cal.write_text("{}", encoding="utf-8")
            meta.write_text(
                json.dumps(
                    {
                        "input_role": "inference_manifest",
                        "prediction_csv_sha256": file_sha256(pred),
                        "calibration_json_sha256": file_sha256(cal),
                        "fold": 0,
                        "seed": 3407,
                        "feature_backend": "training_exact",
                        "feature_pipeline_equivalent": True,
                        "eligible_for_cv_aggregation": False,
                        "paper_main_result": False,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "input_role=inference_manifest"):
                load_and_validate_prediction_metadata(pred, cal)

    def test_evaluation_rejects_prediction_csv_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "test_predictions.csv"
            cal = root / "calibration.json"
            meta = root / "prediction_metadata.json"
            pred.write_text("path,y_true,y_pred\n", encoding="utf-8")
            cal.write_text("{}", encoding="utf-8")
            recorded_sha = file_sha256(pred)
            pred.write_text("path,y_true,y_pred\na.wav,cough,feeding\n", encoding="utf-8")
            meta.write_text(
                json.dumps(
                    {
                        "input_role": "frozen_test",
                        "prediction_csv_sha256": recorded_sha,
                        "calibration_json_sha256": file_sha256(cal),
                        "fold": 0,
                        "seed": 3407,
                        "feature_backend": "training_exact",
                        "feature_pipeline_equivalent": True,
                        "eligible_for_cv_aggregation": True,
                        "paper_main_result": False,
                        "run_scope": "fold_seed",
                        "leakage_audit_ok": True,
                        "manifest_sha_verified": True,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "prediction CSV SHA256 mismatch"):
                load_and_validate_prediction_metadata(pred, cal)

    def test_one_by_one_aggregate_must_fail(self):
        records = [self._aggregate_record(fold=0, seed=42)]

        with self.assertRaisesRegex(RuntimeError, "unsupported aggregate protocol"):
            validate_aggregate_records(
                records,
                expected_folds=[0],
                expected_seeds=[42],
                expected_lambda=0.5,
            )

    def test_screening_5x3_is_candidate_not_paper_main_result(self):
        records = self._aggregate_records(seeds=[42, 2024, 3407])

        aggregate = validate_aggregate_records(
            records,
            expected_folds=[0, 1, 2, 3, 4],
            expected_seeds=[42, 2024, 3407],
            expected_lambda=0.5,
        )

        self.assertEqual(aggregate["run_scope"], "aggregate")
        self.assertTrue(aggregate["screening_result"])
        self.assertTrue(aggregate["paper_candidate_result"])
        self.assertFalse(aggregate["paper_main_result"])
        self.assertFalse(aggregate["final_25_run_result"])
        self.assertEqual(aggregate["n_runs"], 15)

    def test_final_5x5_is_paper_main_result(self):
        records = self._aggregate_records(seeds=[42, 2024, 3407, 7, 99])

        aggregate = validate_aggregate_records(
            records,
            expected_folds=[0, 1, 2, 3, 4],
            expected_seeds=[42, 2024, 3407, 7, 99],
            expected_lambda=0.5,
        )

        self.assertEqual(aggregate["run_scope"], "aggregate")
        self.assertFalse(aggregate["screening_result"])
        self.assertTrue(aggregate["paper_candidate_result"])
        self.assertTrue(aggregate["paper_main_result"])
        self.assertTrue(aggregate["final_25_run_result"])
        self.assertEqual(aggregate["n_runs"], 25)

    def test_four_by_five_aggregate_must_fail(self):
        records = [self._aggregate_record(fold=fold, seed=seed) for fold in range(4) for seed in [1, 2, 3, 4, 5]]

        with self.assertRaisesRegex(RuntimeError, "folds must be exactly"):
            validate_aggregate_records(
                records,
                expected_folds=[0, 1, 2, 3],
                expected_seeds=[1, 2, 3, 4, 5],
                expected_lambda=0.5,
            )

    def test_mixed_lambda_aggregate_must_fail(self):
        records = self._aggregate_records(seeds=[42, 2024, 3407])
        records[-1]["lambda"] = 1.0

        with self.assertRaisesRegex(RuntimeError, "mixed lambda"):
            validate_aggregate_records(
                records,
                expected_folds=[0, 1, 2, 3, 4],
                expected_seeds=[42, 2024, 3407],
                expected_lambda=0.5,
            )

    def test_duplicate_fold_seed_aggregate_must_fail(self):
        records = self._aggregate_records(seeds=[42, 2024, 3407])
        records.append(dict(records[0]))

        with self.assertRaisesRegex(RuntimeError, "duplicate fold-seed"):
            validate_aggregate_records(
                records,
                expected_folds=[0, 1, 2, 3, 4],
                expected_seeds=[42, 2024, 3407],
                expected_lambda=0.5,
            )

    def test_unexpected_fold_aggregate_must_fail(self):
        records = self._aggregate_records(seeds=[42, 2024, 3407])
        records[-1]["fold"] = 9

        with self.assertRaisesRegex(RuntimeError, "unexpected fold"):
            validate_aggregate_records(
                records,
                expected_folds=[0, 1, 2, 3, 4],
                expected_seeds=[42, 2024, 3407],
                expected_lambda=0.5,
            )

    def test_paired_stats_calculate_mean_std_and_win_tie_loss_counts(self):
        records = [
            self._aggregate_record(fold=0, seed=1, raw_macro=0.50, prototype_macro=0.50, hierarchical_macro=0.60),
            self._aggregate_record(fold=0, seed=2, raw_macro=0.70, prototype_macro=0.75, hierarchical_macro=0.70),
            self._aggregate_record(fold=0, seed=3, raw_macro=0.90, prototype_macro=0.95, hierarchical_macro=0.80),
        ]

        stats = paired_macro_f1_statistics(records, bootstrap_samples=100, random_seed=3407)
        hier_vs_raw = next(row for row in stats if row["comparison"] == "hierarchical - raw_softmax")

        self.assertEqual(hier_vs_raw["n"], 3)
        self.assertAlmostEqual(hier_vs_raw["mean_delta"], 0.0, places=7)
        self.assertAlmostEqual(hier_vs_raw["std_delta"], 0.1, places=7)
        self.assertEqual(hier_vs_raw["wins"], 1)
        self.assertEqual(hier_vs_raw["ties"], 1)
        self.assertEqual(hier_vs_raw["losses"], 1)

    def test_bootstrap_ci_is_reproducible_with_fixed_seed(self):
        records = [
            self._aggregate_record(fold=0, seed=1, raw_macro=0.50, hierarchical_macro=0.60),
            self._aggregate_record(fold=0, seed=2, raw_macro=0.70, hierarchical_macro=0.72),
            self._aggregate_record(fold=0, seed=3, raw_macro=0.80, hierarchical_macro=0.78),
        ]

        a = paired_macro_f1_statistics(records, bootstrap_samples=200, random_seed=3407)
        b = paired_macro_f1_statistics(records, bootstrap_samples=200, random_seed=3407)

        self.assertEqual(a, b)

    def test_confusion_aggregation_result_is_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm_path = Path(tmp) / "confusion_matrices.json"
            payload = {}
            for method in ["raw_softmax", "prototype", "hierarchical"]:
                payload[method] = {
                    "labels": ["cough", "calm_grunt", "feeding", "stress_vocal"],
                    "matrix": [
                        [2, 0, 0, 0],
                        [0, 2, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                    ],
                }
            cm_path.write_text(json.dumps(payload), encoding="utf-8")
            records = [self._aggregate_record(confusion_matrices_json=str(cm_path))]

            rows = aggregate_confusion_matrices(records)
            raw = next(row for row in rows if row["method"] == "raw_softmax")

            self.assertEqual(raw["feeding_to_stress"], 1)
            self.assertEqual(raw["stress_to_feeding"], 1)
            self.assertAlmostEqual(raw["feeding_recall"], 0.5, places=7)
            self.assertAlmostEqual(raw["stress_recall"], 0.5, places=7)

    def test_confusion_aggregation_requires_file(self):
        records = [self._aggregate_record(confusion_matrices_json="missing_confusion.json")]

        with self.assertRaisesRegex(FileNotFoundError, "confusion_matrices.json"):
            aggregate_confusion_matrices(records)

    def test_method_metric_summary_requires_every_method(self):
        records = [self._aggregate_record()]
        del records[0]["methods"]["fused"]

        with self.assertRaisesRegex(RuntimeError, "missing method metrics"):
            aggregate_method_metrics(records)

    def test_numpy_logmel_feature_is_finite_and_has_expected_shape(self):
        y = np.zeros(32000 * 2, dtype=np.float32)

        x = make_numpy_logmel_feature(
            y,
            sr=32000,
            n_mels=64,
            n_fft=1024,
            hop_length=320,
            win_length=800,
            fmin=50,
            fmax=8000,
        )

        self.assertEqual(x.shape[0], 1)
        self.assertEqual(x.shape[1], 64)
        self.assertGreaterEqual(x.shape[2], 190)
        self.assertTrue(np.isfinite(x).all())

    def test_map_aux_probabilities_to_main_sums_child_subtypes(self):
        aux_labels = [
            "dry_cough",
            "abdominal_cough",
            "calm_grunt",
            "feeding",
            "frightened_stress",
            "anxious_stress",
        ]
        main_labels = ["cough", "calm_grunt", "feeding", "stress_vocal"]
        aux_probs = np.array([[0.1, 0.2, 0.2, 0.2, 0.15, 0.15]], dtype=np.float32)

        mapped = map_aux_probabilities_to_main(aux_probs, aux_labels, main_labels)

        np.testing.assert_allclose(mapped, [[0.3, 0.2, 0.2, 0.3]], atol=1e-6)

    def test_multiclass_and_calibration_metrics_are_finite(self):
        y_true = np.array([0, 1, 1, 2])
        probs = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.2, 0.7, 0.1],
                [0.4, 0.5, 0.1],
                [0.2, 0.2, 0.6],
            ],
            dtype=np.float32,
        )

        cls = multiclass_metrics(y_true, probs, labels=["a", "b", "c"])
        cal = calibration_metrics(y_true, probs, n_bins=5)

        self.assertEqual(cls["top1_acc"], 1.0)
        self.assertEqual(cls["top2_acc"], 1.0)
        self.assertGreater(cls["macro_f1"], 0.99)
        self.assertTrue(0.0 <= cal["ece"] <= 1.0)
        self.assertTrue(np.isfinite(cal["brier"]))
        self.assertTrue(np.isfinite(cal["nll"]))

    def test_coverage_risk_curve_reports_selective_risk(self):
        y_true = np.array([0, 1, 1, 2])
        y_pred = np.array([0, 1, 0, 2])
        confidence = np.array([0.95, 0.80, 0.70, 0.60])

        curve = coverage_risk_curve(y_true, y_pred, confidence, thresholds=[0.75])

        self.assertEqual(len(curve), 1)
        self.assertAlmostEqual(curve[0]["coverage"], 0.5)
        self.assertAlmostEqual(curve[0]["selective_risk"], 0.0)

    def test_audit_manifest_disjointness_detects_md5_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.csv"
            val = root / "val.csv"
            test = root / "test.csv"
            pd.DataFrame(
                [{"path": "a.wav", "label": "cough", "source_id": "src-a", "md5": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]
            ).to_csv(train, index=False)
            pd.DataFrame(
                [{"path": "b.wav", "label": "cough", "source_id": "src-b", "md5": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}]
            ).to_csv(val, index=False)
            pd.DataFrame(
                [{"path": "c.wav", "label": "feeding", "source_id": "src-c", "md5": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}]
            ).to_csv(test, index=False)

            report = audit_manifest_disjointness(
                {"train": train, "val": val, "test": test},
                columns=("path", "source_id", "md5"),
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["overlaps"][0]["column"], "md5")

    def test_audit_manifest_disjointness_requires_md5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.csv"
            val = root / "val.csv"
            pd.DataFrame([{"path": "a.wav", "source_id": "a"}]).to_csv(train, index=False)
            pd.DataFrame([{"path": "b.wav", "source_id": "b"}]).to_csv(val, index=False)

            report = audit_manifest_disjointness({"train": train, "val": val})

        self.assertFalse(report["ok"])
        self.assertIn("md5", report["missing_columns"]["train"])

    def test_audit_manifest_disjointness_requires_source_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.csv"
            val = root / "val.csv"
            pd.DataFrame([{"path": "a.wav", "md5": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]).to_csv(train, index=False)
            pd.DataFrame([{"path": "b.wav", "md5": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}]).to_csv(val, index=False)

            report = audit_manifest_disjointness({"train": train, "val": val})

        self.assertFalse(report["ok"])
        self.assertIn("source_id", report["missing_columns"]["train"])

    def test_audit_manifest_disjointness_normalizes_windows_slashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.csv"
            val = root / "val.csv"
            pd.DataFrame(
                [{"path": r"data\a.wav", "source_id": "a", "md5": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]
            ).to_csv(train, index=False)
            pd.DataFrame(
                [{"path": "data/a.wav", "source_id": "b", "md5": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}]
            ).to_csv(val, index=False)

            report = audit_manifest_disjointness({"train": train, "val": val})

        self.assertFalse(report["ok"])
        self.assertEqual(report["overlaps"][0]["column"], "path")

    def test_audit_manifest_disjointness_normalizes_dotdot_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.csv"
            val = root / "val.csv"
            pd.DataFrame(
                [{"path": "data/a.wav", "source_id": "a", "md5": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]
            ).to_csv(train, index=False)
            pd.DataFrame(
                [{"path": "data/x/../a.wav", "source_id": "b", "md5": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}]
            ).to_csv(val, index=False)

            report = audit_manifest_disjointness({"train": train, "val": val})

        self.assertFalse(report["ok"])
        self.assertEqual(report["overlaps"][0]["column"], "path")
        self.assertEqual(normalize_identity_path("data/x/../a.wav"), "data/a.wav")

    def test_audit_manifest_disjointness_rejects_empty_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.csv"
            val = root / "val.csv"
            pd.DataFrame(
                [{"path": "", "source_id": "src-a", "md5": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]
            ).to_csv(train, index=False)
            pd.DataFrame(
                [{"path": "b.wav", "source_id": "src-b", "md5": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}]
            ).to_csv(val, index=False)

            report = audit_manifest_disjointness({"train": train, "val": val})

        self.assertFalse(report["ok"])
        self.assertEqual(report["invalid_values"][0]["column"], "path")

    def test_assert_no_leakage_error_includes_audit_sections(self):
        report = {
            "ok": False,
            "missing_columns": {"train": ["md5"]},
            "invalid_values": [{"split": "val", "column": "path"}],
            "overlaps": [{"column": "source_id"}],
        }

        with self.assertRaisesRegex(RuntimeError, "missing_columns.*invalid_values.*overlaps"):
            assert_no_leakage(report)

    def test_verify_manifest_matches_metadata_rejects_changed_val_content_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "val.csv"
            pd.DataFrame(
                [{"path": "a.wav", "source_id": "a", "md5": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]
            ).to_csv(path, index=False)
            metadata = {"manifest_sha256": {"val": file_sha256(path)}}
            pd.DataFrame(
                [{"path": "b.wav", "source_id": "b", "md5": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}]
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(RuntimeError, "val manifest SHA256 mismatch"):
                verify_manifest_matches_metadata(path, metadata, "val")

    def test_verify_manifest_matches_metadata_rejects_changed_test_content_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            pd.DataFrame(
                [{"path": "a.wav", "source_id": "a", "md5": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]
            ).to_csv(path, index=False)
            metadata = {"manifest_sha256": {"test": file_sha256(path)}}
            pd.DataFrame(
                [{"path": "c.wav", "source_id": "c", "md5": "cccccccccccccccccccccccccccccccc"}]
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(RuntimeError, "test manifest SHA256 mismatch"):
                verify_manifest_matches_metadata(path, metadata, "test")

    def test_align_prediction_frames_by_path_detects_missing_extra_and_matches(self):
        reference = pd.DataFrame(
            [
                {"path": r"data\a.wav", "y_true": "cough", "y_pred": "cough"},
                {"path": "data/b.wav", "y_true": "feeding", "y_pred": "feeding"},
            ]
        )
        reproduced = pd.DataFrame(
            [
                {"path": "data/a.wav", "y_true": "cough", "y_pred": "cough"},
                {"path": "data/c.wav", "y_true": "feeding", "y_pred": "feeding"},
            ]
        )

        report, summary = align_prediction_frames_by_path(reference, reproduced)

        self.assertFalse(summary["alignment_ok"])
        self.assertEqual(summary["matched_count"], 1)
        self.assertEqual(summary["missing_reference_paths"], ["data/b.wav"])
        self.assertEqual(summary["extra_reproduced_paths"], ["data/c.wav"])
        self.assertTrue(report.loc[report["path"] == "data/a.wav", "y_pred_match"].iloc[0])

    def test_file_sha256_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.txt"
            path.write_text("prototype", encoding="utf-8")

            digest = file_sha256(path)

        self.assertEqual(
            digest,
            "b34563e2ebeb771708ba0648c9385383f61cc11e86af6076848433059c14b2df",
        )


if __name__ == "__main__":
    unittest.main()
