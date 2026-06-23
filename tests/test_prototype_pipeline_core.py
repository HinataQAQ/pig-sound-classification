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
    audit_manifest_disjointness,
    calibration_metrics,
    compute_class_prototypes,
    compute_distance_statistics,
    coverage_risk_curve,
    file_sha256,
    make_numpy_logmel_feature,
    map_aux_probabilities_to_main,
    multiclass_metrics,
    training_exact_feature_config,
    fusion_formula_metadata,
    prototype_scores_from_bundle,
    validate_expected_config,
)


class PrototypePipelineCoreTests(unittest.TestCase):
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
                [{"path": "a.wav", "label": "cough", "source_id": "src-a", "md5": "same"}]
            ).to_csv(train, index=False)
            pd.DataFrame(
                [{"path": "b.wav", "label": "cough", "source_id": "src-b", "md5": "same"}]
            ).to_csv(val, index=False)
            pd.DataFrame(
                [{"path": "c.wav", "label": "feeding", "source_id": "src-c", "md5": "other"}]
            ).to_csv(test, index=False)

            report = audit_manifest_disjointness(
                {"train": train, "val": val, "test": test},
                columns=("path", "source_id", "md5"),
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["overlaps"][0]["column"], "md5")

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
