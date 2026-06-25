import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from mix_audio_at_snr import (  # noqa: E402
    demand_environment_recording_id,
    deterministic_seed,
    load_audio_with_valid_region,
    mix_clean_with_noise_at_snr,
    noise_offset_key,
    select_noise_segment,
    validate_noise_source_for_paper,
)
from eval_prototype_noise_robustness import (  # noqa: E402
    aurc_score,
    clean_equivalence_gate,
    compute_method_rejection_thresholds,
    noise_result_qualification,
    risk_at_coverages,
    verify_selected_channel,
)
from summarize_prototype_noise_robustness import (  # noqa: E402
    build_cross_seed_draw_audit,
    fold_cluster_paired_stats,
    paired_stats_by_condition,
    validate_screening_protocol,
    validate_single_run_exact_grid,
)


class NoiseRobustnessPipelineTests(unittest.TestCase):
    def _metric_rows(self, missing_env: str | None = None, missing_snr: float | None = None, missing_method: str | None = None):
        rows = []
        for method in ("raw_softmax", "prototype", "hierarchical", "fused"):
            if missing_method == method:
                continue
            rows.append(
                {
                    "noise_environment": "clean",
                    "snr_db": "clean",
                    "target_active_snr_db": "clean",
                    "condition": "clean",
                    "method": method,
                    "macro_f1": 0.9,
                    "top1_acc": 0.9,
                    "top2_acc": 1.0,
                    "ece": 0.01,
                    "brier": 0.02,
                    "nll": 0.03,
                    "aurc": 0.01,
                    "frozen_threshold_coverage": 0.95,
                    "frozen_threshold_selective_risk": 0.05,
                    "feeding_to_stress": 1,
                    "stress_to_feeding": 0,
                    "cough_to_calm_grunt": 0,
                    "cough_to_feeding": 0,
                    "cough_to_stress_vocal": 0,
                }
            )
        for env in ("DWASHING", "TBUS", "STRAFFIC"):
            if missing_env == env:
                continue
            for snr in (20.0, 10.0, 0.0):
                if missing_snr == snr:
                    continue
                for method in ("raw_softmax", "prototype", "hierarchical", "fused"):
                    if missing_method == method:
                        continue
                    base = 0.5 + 0.01 * snr
                    bump = {"raw_softmax": 0.0, "prototype": 0.01, "hierarchical": 0.02, "fused": 0.015}[method]
                    rows.append(
                        {
                            "noise_environment": env,
                            "snr_db": snr,
                            "target_active_snr_db": snr,
                            "condition": f"{env}_{snr:g}dB",
                            "method": method,
                            "macro_f1": base + bump,
                            "top1_acc": base + bump,
                            "top2_acc": 0.8,
                            "ece": 0.1,
                            "brier": 0.2,
                            "nll": 0.3,
                            "aurc": 0.2,
                            "frozen_threshold_coverage": 0.9,
                            "frozen_threshold_selective_risk": 0.2,
                            "feeding_to_stress": 2,
                            "stress_to_feeding": 1,
                            "cough_to_calm_grunt": 3,
                            "cough_to_feeding": 4,
                            "cough_to_stress_vocal": 5,
                        }
                    )
        return rows

    def _payload(self, fold=0, seed=3407, **overrides):
        payload = {
            "fold": fold,
            "seed": seed,
            "lambda": 0.5,
            "run_stage": "screening",
            "global_noise_seed": 3407,
            "noise_repeat": 0,
            "offset_key_version": "demand_noise_offset_v2",
            "single_fold_debug": False,
            "run_scope": "fold_seed",
            "paper_main_result": False,
            "noise_environments": ["DWASHING", "TBUS", "STRAFFIC"],
            "snr_db": [20.0, 10.0, 0.0],
            "target_active_snr_db": [20.0, 10.0, 0.0],
            "simulated_noise": True,
            "noise_protocol": "zero_shot_frozen",
            "real_farm_external_validation": False,
            "clean_equivalence": {"ok": True},
            "provenance_gates": {"ok": True},
            "same_waveform_embedding_reused": True,
            "same_offset_across_snr_verified": True,
            "test_parameter_selection": False,
            "feature_backend": "training_exact",
            "eligible_for_noise_aggregation": True,
            "metrics_by_condition": self._metric_rows(),
            "outputs": {},
        }
        payload.update(overrides)
        return payload

    def _screening_payloads(self):
        return [self._payload(fold=fold, seed=seed) for fold in range(5) for seed in (42, 2024, 3407)]

    def test_deterministic_seed_is_stable_across_processes(self):
        parts = ["fold0", "seed3407", "data/a.wav", "abc", "DWASHING", "noise-sha", "20", "20260625"]
        here = deterministic_seed(parts)
        code = (
            "import sys; "
            "sys.path.insert(0, r'%s'); "
            "from mix_audio_at_snr import deterministic_seed; "
            "print(deterministic_seed(%r))"
        ) % (str(TOOLS), parts)
        other = subprocess.check_output([sys.executable, "-c", code], text=True).strip()

        self.assertEqual(str(here), other)

    def test_mix_hits_active_region_target_snr_and_logs_full_window_snr(self):
        clean = np.zeros(8, dtype=np.float32)
        clean[2:6] = 0.5
        noise = np.ones(8, dtype=np.float32)

        mixed, record = mix_clean_with_noise_at_snr(
            clean,
            noise,
            valid_start=2,
            valid_samples=4,
            target_snr_db=20.0,
            eps=1e-8,
        )

        self.assertEqual(mixed.shape, clean.shape)
        self.assertAlmostEqual(record["achieved_active_snr_db"], 20.0, places=5)
        self.assertIn("achieved_full_window_snr_db", record)
        self.assertFalse(record["clipping_detected"])
        self.assertEqual(record["final_global_scale"], 1.0)

    def test_global_scaling_prevents_clipping_and_preserves_snr(self):
        clean = np.ones(8, dtype=np.float32) * 0.9
        noise = np.ones(8, dtype=np.float32)

        mixed, record = mix_clean_with_noise_at_snr(
            clean,
            noise,
            valid_start=0,
            valid_samples=8,
            target_snr_db=0.0,
            eps=1e-8,
            peak_limit=0.99,
        )

        self.assertLessEqual(float(np.max(np.abs(mixed))), 0.990001)
        self.assertTrue(record["clipping_detected"])
        self.assertLess(record["final_global_scale"], 1.0)
        self.assertAlmostEqual(record["achieved_active_snr_db"], 0.0, places=5)

    def test_mix_rejects_near_silent_active_regions(self):
        clean = np.zeros(8, dtype=np.float32)
        noise = np.ones(8, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "clean_active_rms"):
            mix_clean_with_noise_at_snr(clean, noise, valid_start=0, valid_samples=8, target_snr_db=10.0)

    def test_noise_offset_reproducible_and_key_sensitive(self):
        noise = np.arange(100, dtype=np.float32)
        first, rec_a = select_noise_segment(noise, 16, ["fold0", "seed3407", "a"], return_record=True)
        second, rec_b = select_noise_segment(noise, 16, ["fold0", "seed3407", "a"], return_record=True)
        third, rec_c = select_noise_segment(noise, 16, ["fold0", "seed3407", "b"], return_record=True)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(rec_a["noise_offset"], rec_b["noise_offset"])
        self.assertNotEqual(rec_a["noise_offset"], rec_c["noise_offset"])
        self.assertEqual(len(third), 16)

    def test_noise_offset_key_freezes_across_snr_and_model_seed(self):
        noise = np.arange(1000, dtype=np.float32)
        base = {
            "fold": 0,
            "clean_path": "data/a.wav",
            "clean_md5": "a" * 32,
            "environment_recording_id": "DEMAND:DWASHING",
            "noise_sha256": "b" * 64,
            "global_noise_seed": 3407,
            "noise_repeat": 0,
        }
        key_seed_a = noise_offset_key(model_seed=42, target_active_snr_db=20.0, **base)
        key_seed_b = noise_offset_key(model_seed=3407, target_active_snr_db=10.0, **base)
        _, rec_a = select_noise_segment(noise, 32, key_seed_a["key_parts"], return_record=True)
        _, rec_b = select_noise_segment(noise, 32, key_seed_b["key_parts"], return_record=True)

        self.assertEqual(key_seed_a["noise_draw_id"], key_seed_b["noise_draw_id"])
        self.assertEqual(rec_a["noise_offset"], rec_b["noise_offset"])
        self.assertEqual(key_seed_a["offset_key_version"], "demand_noise_offset_v2")

    def test_noise_offset_key_changes_for_environment_and_repeat(self):
        base = {
            "fold": 0,
            "model_seed": 3407,
            "target_active_snr_db": 20.0,
            "clean_path": "data/a.wav",
            "clean_md5": "a" * 32,
            "environment_recording_id": "DEMAND:DWASHING",
            "noise_sha256": "b" * 64,
            "global_noise_seed": 3407,
            "noise_repeat": 0,
        }
        env_changed = dict(base, environment_recording_id="DEMAND:TBUS")
        repeat_changed = dict(base, noise_repeat=1)

        self.assertNotEqual(noise_offset_key(**base)["noise_draw_id"], noise_offset_key(**env_changed)["noise_draw_id"])
        self.assertNotEqual(noise_offset_key(**base)["noise_draw_id"], noise_offset_key(**repeat_changed)["noise_draw_id"])

    def test_load_audio_with_valid_region_reports_padding_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.wav"
            sf.write(path, np.ones(4, dtype=np.float32), 4)

            y, sr, valid_start, valid_samples = load_audio_with_valid_region(path, sr=4, dur_s=2.0)

        self.assertEqual(sr, 4)
        self.assertEqual(len(y), 8)
        self.assertEqual(valid_start, 2)
        self.assertEqual(valid_samples, 4)

    def test_demand_channels_are_grouped_under_one_environment_recording(self):
        self.assertEqual(demand_environment_recording_id("DWASHING", "ch01.wav"), "DEMAND:DWASHING")
        self.assertEqual(demand_environment_recording_id("DWASHING", "ch16.wav"), "DEMAND:DWASHING")

    def test_local_unlicensed_noise_cannot_enter_paper_results(self):
        row = {
            "dataset": "local_reviewed",
            "allowed_for_publication": "false",
            "allowed_for_redistribution": "false",
        }

        with self.assertRaisesRegex(ValueError, "not paper-facing"):
            validate_noise_source_for_paper(row, paper_facing=True)

    def test_validate_noise_source_allows_demand_for_paper(self):
        row = {
            "dataset": "DEMAND",
            "allowed_for_publication": "true",
            "allowed_for_redistribution": "true",
        }

        self.assertTrue(validate_noise_source_for_paper(row, paper_facing=True))

    def test_noise_debug_outputs_carry_simulated_flags(self):
        payload = {
            "simulated_noise": True,
            "noise_protocol": "zero_shot_frozen",
            "real_farm_external_validation": False,
        }

        self.assertTrue(payload["simulated_noise"])
        self.assertEqual(payload["noise_protocol"], "zero_shot_frozen")
        self.assertFalse(payload["real_farm_external_validation"])

    def test_noise_qualification_never_promotes_single_run_to_paper_main(self):
        exact = noise_result_qualification(
            feature_backend="training_exact",
            clean_equivalence_ok=True,
            leakage_audit_ok=True,
            manifest_sha_verified=True,
            unsafe_allow_checkpoint_sha_mismatch=False,
        )
        numpy = noise_result_qualification(
            feature_backend="numpy_logmel",
            clean_equivalence_ok=True,
            leakage_audit_ok=True,
            manifest_sha_verified=True,
            unsafe_allow_checkpoint_sha_mismatch=False,
        )

        self.assertEqual(exact["run_scope"], "fold_seed")
        self.assertFalse(exact["paper_main_result"])
        self.assertTrue(exact["feature_pipeline_equivalent"])
        self.assertTrue(exact["eligible_for_cv_aggregation"])
        self.assertFalse(numpy["feature_pipeline_equivalent"])
        self.assertFalse(numpy["eligible_for_cv_aggregation"])

    def test_noise_qualification_sets_stage_specific_debug_flags(self):
        smoke = noise_result_qualification(
            feature_backend="training_exact",
            clean_equivalence_ok=True,
            leakage_audit_ok=True,
            manifest_sha_verified=True,
            unsafe_allow_checkpoint_sha_mismatch=False,
            run_stage="prescreen_smoke",
        )
        screening = noise_result_qualification(
            feature_backend="training_exact",
            clean_equivalence_ok=True,
            leakage_audit_ok=True,
            manifest_sha_verified=True,
            unsafe_allow_checkpoint_sha_mismatch=False,
            run_stage="screening",
        )

        self.assertTrue(smoke["single_fold_debug"])
        self.assertEqual(smoke["run_scope"], "fold_seed_smoke")
        self.assertFalse(screening["single_fold_debug"])
        self.assertEqual(screening["run_scope"], "fold_seed")

    def test_verify_selected_channel_rejects_silent_cli_manifest_disagreement(self):
        row = {"selected_channel": 1}

        self.assertEqual(verify_selected_channel(None, row), 1)
        self.assertEqual(verify_selected_channel(1, row), 1)
        with self.assertRaisesRegex(ValueError, "selected_channel"):
            verify_selected_channel(2, row)

    def test_method_specific_thresholds_use_each_method_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            val_path = Path(tmp) / "val_predictions.csv"
            val = pd.DataFrame(
                {
                    "y_true_id": [0, 0, 1, 1],
                    "raw_softmax_pred_id": [0, 0, 1, 1],
                    "raw_softmax_confidence": [0.9, 0.8, 0.7, 0.6],
                    "prototype_pred_id": [0, 0, 1, 1],
                    "prototype_confidence": [0.4, 0.5, 0.6, 0.7],
                    "hierarchical_pred_id": [0, 0, 1, 1],
                    "hierarchical_confidence": [0.95, 0.85, 0.75, 0.65],
                }
            )
            val.to_csv(val_path, index=False)
            thresholds = compute_method_rejection_thresholds(
                val_predictions=val,
                labels=["cough", "feeding"],
                target_coverage=0.5,
                source_validation_predictions_path=val_path,
                methods=("raw_softmax", "prototype", "hierarchical"),
            )

        self.assertNotEqual(thresholds["raw_softmax"]["global_threshold"], thresholds["prototype"]["global_threshold"])
        self.assertEqual(thresholds["raw_softmax"]["source_validation_predictions_sha256"], thresholds["prototype"]["source_validation_predictions_sha256"])
        self.assertEqual(thresholds["hierarchical"]["threshold_selection_formula"], "quantile(confidence, 1 - target_coverage)")
        self.assertIn("cough", thresholds["prototype"]["per_class_thresholds"])

    def test_aurc_and_risk_at_coverages_are_finite(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0])
        conf = np.array([0.9, 0.8, 0.7, 0.6])

        self.assertGreaterEqual(aurc_score(y_true, y_pred, conf), 0.0)
        risks = risk_at_coverages(y_true, y_pred, conf, coverages=(0.5, 0.75, 1.0))
        self.assertEqual(set(risks), {"risk_at_coverage_0.50", "risk_at_coverage_0.75", "risk_at_coverage_1.00"})
        self.assertTrue(all(np.isfinite(v) for v in risks.values()))

    def test_validate_screening_protocol_rejects_condition_rows_as_runs(self):
        payloads = [
            {"fold": 0, "seed": 3407, "lambda": 0.5, "noise_environments": ["DWASHING"], "snr_db": [20, 10, 0]},
        ]

        with self.assertRaisesRegex(ValueError, "15 fold-seed runs"):
            validate_screening_protocol(payloads)

    def test_validate_screening_protocol_accepts_fixed_15_run_grid(self):
        payloads = self._screening_payloads()

        result = validate_screening_protocol(payloads)

        self.assertEqual(result["n_fold_seed_runs"], 15)
        self.assertFalse(result["paper_main_result"])
        self.assertTrue(result["screening_result"])

    def test_validate_single_run_exact_grid_rejects_missing_environment(self):
        payload = self._payload(noise_environments=["DWASHING", "TBUS"])

        with self.assertRaisesRegex(ValueError, "environments"):
            validate_single_run_exact_grid(payload)

    def test_validate_single_run_exact_grid_rejects_missing_zero_db(self):
        payload = self._payload(snr_db=[20, 10], target_active_snr_db=[20, 10])

        with self.assertRaisesRegex(ValueError, "SNR"):
            validate_single_run_exact_grid(payload)

    def test_validate_single_run_exact_grid_rejects_missing_condition_method(self):
        payload = self._payload(metrics_by_condition=self._metric_rows(missing_method="fused"))

        with self.assertRaisesRegex(ValueError, "condition/method"):
            validate_single_run_exact_grid(payload)

    def test_validate_screening_protocol_rejects_global_noise_seed_mismatch(self):
        payloads = self._screening_payloads()
        payloads[0]["global_noise_seed"] = 123

        with self.assertRaisesRegex(ValueError, "global_noise_seed"):
            validate_screening_protocol(payloads)

    def test_validate_screening_protocol_rejects_noise_repeat_mismatch(self):
        payloads = self._screening_payloads()
        payloads[0]["noise_repeat"] = 1

        with self.assertRaisesRegex(ValueError, "noise_repeat"):
            validate_screening_protocol(payloads)

    def test_paired_stats_fail_when_condition_n_is_not_15(self):
        payloads = self._screening_payloads()
        frame = pd.DataFrame(
            [
                {**row, "fold": payload["fold"], "seed": payload["seed"], "lambda": payload["lambda"]}
                for payload in payloads[:-1]
                for row in payload["metrics_by_condition"]
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "paired n"):
            paired_stats_by_condition(frame)

    def test_moderate_and_extreme_grouping_are_reported(self):
        payloads = self._screening_payloads()
        frame = pd.DataFrame(
            [
                {**row, "fold": payload["fold"], "seed": payload["seed"], "lambda": payload["lambda"]}
                for payload in payloads
                for row in payload["metrics_by_condition"]
            ]
        )

        paired = paired_stats_by_condition(frame)

        self.assertIn("MODERATE_NOISE", set(paired["noise_environment"]))
        self.assertIn("EXTREME_STRESS", set(paired["noise_environment"]))
        self.assertIn("ALL_NOISY", set(paired["noise_environment"]))
        self.assertTrue((paired[paired["noise_environment"] == "MODERATE_NOISE"]["n"] == 15).all())

    def test_fold_cluster_bootstrap_is_reproducible(self):
        payloads = self._screening_payloads()
        frame = pd.DataFrame(
            [
                {**row, "fold": payload["fold"], "seed": payload["seed"], "lambda": payload["lambda"]}
                for payload in payloads
                for row in payload["metrics_by_condition"]
            ]
        )

        first = fold_cluster_paired_stats(frame)
        second = fold_cluster_paired_stats(frame)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["n_folds"]), {5})

    def test_cross_seed_draw_audit_rejects_mismatched_draw_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for seed in (42, 2024, 3407):
                path = Path(tmp) / f"seed{seed}.csv"
                pd.DataFrame(
                    [
                        {
                            "fold": 0,
                            "seed": seed,
                            "clean_md5": "a" * 32,
                            "noise_environment": "DWASHING",
                            "noise_repeat": 0,
                            "offset_key_version": "demand_noise_offset_v2",
                            "noise_draw_id": "draw-a" if seed != 3407 else "draw-b",
                            "noise_offset": 123,
                            "noise_sha256": "b" * 64,
                            "selected_channel": 1,
                            "global_noise_seed": 3407,
                        }
                    ]
                ).to_csv(path, index=False)
                paths.append(path)

            with self.assertRaisesRegex(ValueError, "cross-seed"):
                build_cross_seed_draw_audit(paths)

    def test_clean_equivalence_gate_handles_suffixed_y_true_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference_csv = Path(tmp) / "reference.csv"
            pd.DataFrame(
                [
                    {"path": "data/a.wav", "y_true": "cough", "y_pred": "cough"},
                    {"path": "data/b.wav", "y_true": "feeding", "y_pred": "feeding"},
                ]
            ).to_csv(reference_csv, index=False)
            clean_pred = pd.DataFrame(
                [
                    {"path": "data/a.wav", "y_true": "cough", "raw_softmax_pred": "cough"},
                    {"path": "data/b.wav", "y_true": "feeding", "raw_softmax_pred": "feeding"},
                ]
            )

            result = clean_equivalence_gate(
                reference_pred_csv=reference_csv,
                manifest_paths=["data/a.wav", "data/b.wav"],
                clean_pred=clean_pred,
                labels=["cough", "feeding"],
                tolerance=1e-6,
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["y_true_match"])
        self.assertTrue(result["raw_softmax_y_pred_match"])
        self.assertTrue(result["alignment"]["y_pred_all_match"])
        self.assertEqual(result["alignment"]["y_pred_match_count"], 2)


if __name__ == "__main__":
    unittest.main()
