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
    select_noise_segment,
    validate_noise_source_for_paper,
)
from eval_prototype_noise_robustness import clean_equivalence_gate, noise_result_qualification  # noqa: E402


class NoiseRobustnessPipelineTests(unittest.TestCase):
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
