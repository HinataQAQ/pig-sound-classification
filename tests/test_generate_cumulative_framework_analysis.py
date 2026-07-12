from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from tools import generate_cumulative_framework_analysis as cumulative
from tools.nomenclature import build_method_metadata


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ARTIFACTS_AVAILABLE = all(
    path.is_file()
    for path in (
        ROOT / "reports" / "cv5_expanded_cap3x_fold0_logmel_seed42" / "summary.json",
        ROOT
        / "reports"
        / "prototype_cv5_exact_w05_fold0_seed42"
        / "artifacts"
        / "prototype_metadata.json",
        ROOT
        / "paper_results"
        / "manifests"
        / "manifests_pigvocal_4class_expanded_train_cv5_cap3x"
        / "fold0"
        / "train.csv",
    )
)


class CumulativeFrameworkPureTests(unittest.TestCase):
    def test_stage_reader_reconciles_canonical_family_and_context(self) -> None:
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
            "test_macro_f1": 0.95,
        }

        self.assertAlmostEqual(
            cumulative._validate_stage_summary(
                payload,
                path=Path("canonical-summary.json"),
                stage="B2",
                seed=42,
            ),
            0.95,
        )

        conflicting = {**payload, "context_seconds": 1.0}
        with self.assertRaisesRegex(ValueError, "context_seconds=2.0"):
            cumulative._validate_stage_summary(
                conflicting,
                path=Path("conflicting-summary.json"),
                stage="B2",
                seed=42,
            )

        validation_selected = {
            **payload,
            **build_method_metadata(
                model_family="hierarchical_supervision_crnn",
                training_stage=(
                    "b2_validation_selected_hierarchical_crnn"
                ),
                context_seconds=2.0,
                selection_protocol=(
                    "foldwise_validation_selected_lambda"
                ),
                inference_route="primary_softmax",
                legacy_method_id="raw_softmax",
            ),
        }
        with self.assertRaisesRegex(ValueError, "selection_protocol"):
            cumulative._validate_stage_summary(
                validation_selected,
                path=Path("validation-selected-summary.json"),
                stage="B2",
                seed=42,
            )

    def test_stage_metadata_keeps_fixed_lambda_retrospective_distinct(self) -> None:
        b0 = cumulative.canonical_metadata_for_stage("B0")
        b1 = cumulative.canonical_metadata_for_stage("b1_2s_logmel_mainline")
        b2 = cumulative.canonical_metadata_for_stage("B2")
        b3 = cumulative.canonical_metadata_for_stage("B3")

        self.assertEqual(b0["selection_protocol"], "none")
        self.assertEqual(b1["selection_protocol"], "none")
        self.assertEqual(
            b2["selection_protocol"], "fixed_lambda_0_5_retrospective"
        )
        self.assertEqual(
            b3["selection_protocol"], "fixed_lambda_0_5_retrospective"
        )
        self.assertEqual(b2["inference_route"], "primary_softmax")
        self.assertEqual(
            b3["inference_route"], "hierarchical_prototype_candidate"
        )
        self.assertNotEqual(
            b2["selection_protocol"], "foldwise_validation_selected_lambda"
        )

    def test_stage_metadata_rejects_unknown_stage_names_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown cumulative framework stage"):
            cumulative.canonical_metadata_for_stage("B9")

    def test_fixed_cohort_rejects_validation_selected_source_metadata(self) -> None:
        self.assertEqual(
            cumulative._validate_fixed_selection_protocol(
                (
                    ("bundle", {"selection_protocol": "none"}),
                    (
                        "metrics",
                        {"selection_protocol": "fixed_lambda_0.5"},
                    ),
                )
            ),
            "fixed_lambda_0_5_retrospective",
        )
        with self.assertRaisesRegex(ValueError, "Conflicting selection_protocol"):
            cumulative._validate_fixed_selection_protocol(
                (
                    (
                        "metrics",
                        {
                            "selection_protocol": (
                                "foldwise_validation_selected_lambda"
                            )
                        },
                    ),
                )
            )

    def test_stage_summary_appends_metadata_without_renaming_legacy_columns(self) -> None:
        runs = pd.DataFrame(
            [
                {"fold": 0, "seed": 42, "B0": 0.8, "B1": 0.9, "B2": 0.91, "B3": 0.92},
                {"fold": 0, "seed": 123, "B0": 0.81, "B1": 0.9, "B2": 0.92, "B3": 0.93},
            ]
        )
        summary = cumulative.compute_stage_summary(runs)

        self.assertEqual(
            summary.columns[:7].tolist(),
            [
                "stage",
                "stage_name",
                "n",
                "mean_macro_f1",
                "std_macro_f1",
                "min_macro_f1",
                "max_macro_f1",
            ],
        )
        self.assertTrue(set(cumulative.METHOD_METADATA_FIELDS).issubset(summary.columns))
        protocols = summary.set_index("stage")["selection_protocol"].to_dict()
        self.assertEqual(
            protocols,
            {
                "B0": "none",
                "B1": "none",
                "B2": "fixed_lambda_0_5_retrospective",
                "B3": "fixed_lambda_0_5_retrospective",
            },
        )
        for row in summary.itertuples(index=False):
            if row.stage == "B2":
                self.assertNotIn("Validation-selected", row.stage_name)
                self.assertIn("2-s hierarchical-supervision CRNN", row.stage_name)
            else:
                self.assertEqual(
                    row.stage_name,
                    cumulative.display_label(
                        "training_stage", row.training_stage, language="en"
                    ),
                )

    def test_direct_statistics_use_the_same_synthetic_fold_seed_rows(self) -> None:
        rows = []
        for fold, seed in cumulative.EXPECTED_KEYS:
            offset = fold * 0.001 + cumulative.EXPECTED_SEEDS.index(seed) * 0.0001
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "B0": 0.80 + offset,
                    "B1": 0.82 + offset,
                    "B2": 0.825 + offset,
                    "B3": 0.83 + offset,
                }
            )
        runs = pd.DataFrame(rows)
        paired, folds = cumulative.compute_paired_statistics(runs, n_boot=200, seed=17)
        direct = paired.set_index("comparison").loc["B3 - B0"]
        expected = runs["B3"].to_numpy() - runs["B0"].to_numpy()
        self.assertEqual(int(direct["n"]), 25)
        self.assertAlmostEqual(float(direct["mean_delta"]), float(expected.mean()))
        self.assertEqual(folds[folds["comparison"] == "B3 - B0"].shape[0], 5)
        self.assertTrue(set(cumulative.METHOD_METADATA_FIELDS).issubset(paired.columns))
        self.assertTrue(set(cumulative.METHOD_METADATA_FIELDS).issubset(folds.columns))
        self.assertEqual(
            direct["selection_protocol"], "fixed_lambda_0_5_retrospective"
        )
        b1_b0 = paired.set_index("comparison").loc["B1 - B0"]
        self.assertEqual(b1_b0["selection_protocol"], "none")
        self.assertEqual(
            direct["baseline_canonical_method_id"],
            cumulative.canonical_metadata_for_stage("B0")["canonical_method_id"],
        )

    def test_lower_median_rank_is_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "score": [0.4, 0.1, 0.3, 0.2],
                "source_id": ["d", "a", "c", "b"],
            }
        )
        row, rank = cumulative._lower_median(frame, ("score", "source_id"))
        self.assertEqual(rank, 2)
        self.assertEqual(row["source_id"], "b")

    def test_prototype_metadata_binds_the_expected_b2_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = (
                root
                / "reports"
                / "cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed42"
                / "summary.json"
            )
            summary.parent.mkdir(parents=True)
            summary_payload = {
                "feature_mode": "logmel",
                "use_se": True,
                "seed": 42,
                "dur_s": 2.0,
                "n_mels": 64,
                "n_mfcc": 20,
                "fmin": 50,
                "fmax": 8000,
                "n_fft": 1024,
                "hop_length": 320,
                "win_length": 800,
                "sr": 32000,
                "rnn_type": "gru",
                "pooling_type": "mean",
                "hier_aux_weight": 0.5,
                "main_labels": list(cumulative.MAIN_LABELS),
                "aux_labels": list(cumulative.AUX_LABELS),
            }
            summary.write_text(json.dumps(summary_payload), encoding="utf-8")
            summary_sha = hashlib.sha256(summary.read_bytes()).hexdigest()
            checkpoint_relative = (
                "checkpoints/cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed42.pt"
            )
            summary_relative = summary.relative_to(root).as_posix()
            manifest_relatives = {
                split: (
                    "paper_results/manifests/"
                    "manifests_pigvocal_4class_expanded_train_cv5_cap3x/"
                    f"fold0/{split}.csv"
                )
                for split in ("train", "val", "test")
            }
            manifest_hashes = {}
            for split, relative in manifest_relatives.items():
                manifest_path = root / relative
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(f"{split}\n", encoding="utf-8")
                manifest_hashes[split] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            payload = {
                "artifact_type": "hier_acoustic_prototype_bundle",
                "fold": 0,
                "seed": 42,
                "inferred_fold_seed": {"fold": 0, "seed": 42},
                "checkpoint_sha256": "a" * 64,
                "checkpoint_summary_sha256": summary_sha,
                "project_relative_paths": {
                    "checkpoint": checkpoint_relative,
                    "checkpoint_summary": summary_relative,
                    **{f"{split}_manifest": value for split, value in manifest_relatives.items()},
                },
                "paths": {
                    "checkpoint": {"project_relative": checkpoint_relative, "sha256": "a" * 64},
                    "checkpoint_summary": {
                        "project_relative": summary_relative,
                        "sha256": summary_sha,
                    },
                    **{
                        f"{split}_manifest": {
                            "project_relative": value,
                            "sha256": manifest_hashes[split],
                        }
                        for split, value in manifest_relatives.items()
                    },
                },
                "input_hashes": {
                    "checkpoint": {"sha256": "a" * 64},
                    "checkpoint_summary": {"sha256": summary_sha},
                    **{
                        f"{split}_manifest": {"sha256": manifest_hashes[split]}
                        for split in manifest_relatives
                    },
                },
                "manifest_sha256": manifest_hashes,
                "model_config": dict(summary_payload),
                "feature_backend": "training_exact",
                "run_scope": "fold_seed",
                "feature_pipeline_equivalent": True,
                "single_fold_debug": False,
                "eligible_for_cv_aggregation": True,
                "dur_s": 2.0,
                "hier_aux_weight": 0.5,
                "prototype_source_split": "train",
                "calibration_source_split": "validation",
                "evaluation_source_split": "test",
                "main_labels": list(cumulative.MAIN_LABELS),
                "aux_labels": list(cumulative.AUX_LABELS),
            }
            cumulative._validate_prototype_metadata(
                payload,
                path=root / "prototype_metadata.json",
                root=root,
                fold=0,
                seed=42,
                summary_path=summary,
            )
            route_specific = {
                **payload,
                **build_method_metadata(
                    model_family="hierarchical_supervision_crnn",
                    training_stage=(
                        "b2_validation_selected_hierarchical_crnn"
                    ),
                    context_seconds=2.0,
                    selection_protocol=(
                        "fixed_lambda_0_5_retrospective"
                    ),
                    inference_route="primary_softmax",
                    legacy_method_id="raw_softmax",
                ),
            }
            with self.assertRaisesRegex(ValueError, "route-independent"):
                cumulative._validate_prototype_metadata(
                    route_specific,
                    path=root / "prototype_metadata.json",
                    root=root,
                    fold=0,
                    seed=42,
                    summary_path=summary,
                )
            stray_route_fields = {
                "canonical_method_id": (
                    "b3_hierarchical_prototype_top1_ablation::"
                    "fixed_lambda_0_5_retrospective::"
                    "hierarchical_prototype_candidate"
                ),
                "display_name_en": "Hierarchical prototype candidate route",
                "display_name_zh": "hierarchical route",
                "method": "hierarchical",
            }
            for field, value in stray_route_fields.items():
                with self.subTest(field=field):
                    with self.assertRaisesRegex(
                        ValueError, "route-independent"
                    ):
                        cumulative._validate_prototype_metadata(
                            {**payload, field: value},
                            path=root / "prototype_metadata.json",
                            root=root,
                            fold=0,
                            seed=42,
                            summary_path=summary,
                        )
            invalid = json.loads(json.dumps(payload))
            invalid["hier_aux_weight"] = 1.0
            with self.assertRaisesRegex(ValueError, "hier_aux_weight=0.5"):
                cumulative._validate_prototype_metadata(
                    invalid,
                    path=root / "prototype_metadata.json",
                    root=root,
                    fold=0,
                    seed=42,
                    summary_path=summary,
                )
            invalid_config = json.loads(json.dumps(payload))
            invalid_config["model_config"]["n_mels"] = 128
            with self.assertRaisesRegex(ValueError, "model_config n_mels matches B2 summary"):
                cumulative._validate_prototype_metadata(
                    invalid_config,
                    path=root / "prototype_metadata.json",
                    root=root,
                    fold=0,
                    seed=42,
                    summary_path=summary,
                )
            summary.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary SHA-256 match"):
                cumulative._validate_prototype_metadata(
                    payload,
                    path=root / "prototype_metadata.json",
                    root=root,
                    fold=0,
                    seed=42,
                    summary_path=summary,
                )
            summary.write_text(json.dumps(summary_payload), encoding="utf-8")
            (root / manifest_relatives["train"]).write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train manifest SHA-256 match"):
                cumulative._validate_prototype_metadata(
                    payload,
                    path=root / "prototype_metadata.json",
                    root=root,
                    fold=0,
                    seed=42,
                    summary_path=summary,
                )

    def test_prototype_metrics_require_locked_labels_and_feature_pipeline(self) -> None:
        payload = {
            "artifact_type": "hier_acoustic_prototype_test_evaluation",
            "fold": 0,
            "seed": 42,
            "labels": list(cumulative.MAIN_LABELS),
            "aux_labels": list(cumulative.AUX_LABELS),
            "input_role": "frozen_test",
            "selection_method": "hierarchical",
            "feature_backend": "training_exact",
            "feature_pipeline_equivalent": True,
            "run_scope": "fold_seed",
            "eligible_for_cv_aggregation": True,
            "leakage_audit_ok": True,
            "manifest_sha_verified": True,
            "test_used_for_parameter_selection": False,
            "method_best_params": {"hierarchical": {"hier_aux_prob_weight": 0.5}},
            "methods": {
                "raw_softmax": {"macro_f1": 0.9},
                "hierarchical": {"macro_f1": 0.91},
            },
        }
        cumulative._validate_prototype_metrics(
            payload, path=Path("metrics.json"), fold=0, seed=42
        )
        b2_metadata = {
            **payload,
            **build_method_metadata(
                model_family="hierarchical_supervision_crnn",
                training_stage=(
                    "b2_validation_selected_hierarchical_crnn"
                ),
                context_seconds=2.0,
                selection_protocol="fixed_lambda_0_5_retrospective",
                inference_route="primary_softmax",
                legacy_method_id="raw_softmax",
            ),
        }
        with self.assertRaisesRegex(
            ValueError,
            "artifact role|Conflicting inference method fields",
        ):
            cumulative._validate_prototype_metrics(
                b2_metadata,
                path=Path("b2-metadata-on-b3-metrics.json"),
                fold=0,
                seed=42,
            )
        payload["labels"] = list(reversed(cumulative.MAIN_LABELS))
        with self.assertRaisesRegex(ValueError, "main labels match"):
            cumulative._validate_prototype_metrics(
                payload, path=Path("metrics.json"), fold=0, seed=42
            )

    def test_prototype_metrics_accept_canonical_route_names_and_reject_unknowns(self) -> None:
        payload = {
            "artifact_type": "hier_acoustic_prototype_test_evaluation",
            "fold": 0,
            "seed": 42,
            "labels": list(cumulative.MAIN_LABELS),
            "aux_labels": list(cumulative.AUX_LABELS),
            "input_role": "frozen_test",
            "selection_method": "hierarchical_prototype_candidate",
            "feature_backend": "training_exact",
            "feature_pipeline_equivalent": True,
            "run_scope": "fold_seed",
            "eligible_for_cv_aggregation": True,
            "leakage_audit_ok": True,
            "manifest_sha_verified": True,
            "test_used_for_parameter_selection": False,
            "method_best_params": {
                "hierarchical_prototype_candidate": {"hier_aux_prob_weight": 0.5}
            },
            "methods": {
                "primary_softmax": {"macro_f1": 0.9},
                "hierarchical_prototype_candidate": {"macro_f1": 0.91},
            },
        }
        self.assertEqual(
            cumulative._validate_prototype_metrics(
                payload, path=Path("metrics.json"), fold=0, seed=42
            ),
            (0.9, 0.91),
        )

        payload["selection_method"] = "mystery_route"
        with self.assertRaisesRegex(ValueError, "Unknown inference_route ID"):
            cumulative._validate_prototype_metrics(
                payload, path=Path("metrics.json"), fold=0, seed=42
            )

        payload["selection_method"] = "hierarchical_prototype_candidate"
        payload["methods"]["mystery_route"] = {"macro_f1": 0.1}
        with self.assertRaisesRegex(ValueError, "Unknown inference method"):
            cumulative._validate_prototype_metrics(
                payload, path=Path("metrics.json"), fold=0, seed=42
            )

    def test_manifest_membership_rejects_validation_identity_leakage(self) -> None:
        train = pd.DataFrame(
            [
                {"path": "train/a.wav", "source_id": "train::a", "md5": "a" * 32, "label": "cough", "subtype": "dry_cough"},
                {"path": "train/b.wav", "source_id": "train::b", "md5": "b" * 32, "label": "feeding", "subtype": "feeding"},
            ]
        )
        val = pd.DataFrame(
            [{"path": "val/c.wav", "source_id": "val::c", "md5": "c" * 32, "label": "calm_grunt", "subtype": "calm_grunt"}]
        )
        test = pd.DataFrame(
            [{"path": "test/d.wav", "source_id": "test::d", "md5": "d" * 32, "label": "stress_vocal", "subtype": "anxious_stress"}]
        )
        predictions = test[["path", "source_id", "md5", "subtype"]].copy()
        predictions["y_true"] = test["label"]
        main_reps = train[["path", "source_id", "md5"]].copy()
        main_reps["main_label"] = train["label"]
        aux_reps = train[["path", "source_id", "md5"]].copy()
        aux_reps["aux_label"] = train["subtype"]
        cumulative._validate_canonical_membership(
            train=train,
            val=val,
            test=test,
            predictions=predictions,
            main_reps=main_reps,
            aux_reps=aux_reps,
        )
        leaky_val = val.copy()
        leaky_val.loc[0, "source_id"] = "train::a"
        with self.assertRaisesRegex(ValueError, "source_id overlap"):
            cumulative._validate_canonical_membership(
                train=train,
                val=leaky_val,
                test=test,
                predictions=predictions,
                main_reps=main_reps,
                aux_reps=aux_reps,
            )
        swapped = main_reps.copy()
        swapped.loc[:, "md5"] = list(reversed(swapped["md5"].tolist()))
        with self.assertRaisesRegex(ValueError, "identity membership differs"):
            cumulative._validate_canonical_membership(
                train=train,
                val=val,
                test=test,
                predictions=predictions,
                main_reps=swapped,
                aux_reps=aux_reps,
            )
        wrong_label = main_reps.copy()
        wrong_label.loc[0, "main_label"] = "feeding"
        with self.assertRaisesRegex(ValueError, "main_label does not match"):
            cumulative._validate_canonical_membership(
                train=train,
                val=val,
                test=test,
                predictions=predictions,
                main_reps=wrong_label,
                aux_reps=aux_reps,
            )


@unittest.skipUnless(
    LOCAL_ARTIFACTS_AVAILABLE,
    "requires the local frozen 25-run research artifacts (not stored in git)",
)
class LocalCumulativeFrameworkIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cumulative._source_paths(ROOT)
        except FileNotFoundError as exc:
            raise unittest.SkipTest(f"partial local artifact install: {exc}") from exc
        cls.runs = cumulative.load_cumulative_runs(ROOT)
        cls.summary = cumulative.compute_stage_summary(cls.runs)
        cls.paired, cls.folds = cumulative.compute_paired_statistics(cls.runs)
        cls.cases = cumulative.select_case_studies(ROOT)

    def test_locked_runs_are_exactly_25_matched_fold_seed_rows(self) -> None:
        self.assertEqual(len(self.runs), 25)
        self.assertEqual(self.runs[["fold", "seed"]].drop_duplicates().shape[0], 25)
        self.assertEqual(set(self.runs["fold"]), set(range(5)))
        self.assertEqual(set(self.runs["seed"]), {42, 123, 777, 2024, 3407})
        self.assertTrue({"B0", "B1", "B2", "B3"}.issubset(self.runs.columns))
        self.assertFalse(self.runs[["B0", "B1", "B2", "B3"]].isna().any().any())
        self.assertAlmostEqual(float(self.runs["B0"].mean()), 0.9226064673362835)
        self.assertAlmostEqual(float(self.runs["B1"].mean()), 0.9472373583266809)
        self.assertAlmostEqual(float(self.runs["B2"].mean()), 0.9510496732091448)
        self.assertAlmostEqual(float(self.runs["B3"].mean()), 0.9539862142299651)

    def test_b2_raw_softmax_matches_prototype_cohort_exactly(self) -> None:
        self.assertIn("B3_raw_softmax", self.runs.columns)
        pd.testing.assert_series_equal(
            self.runs["B2"],
            self.runs["B3_raw_softmax"],
            check_names=False,
            check_exact=True,
        )

    def test_debug_marker_is_disclosed_without_excluding_eligible_run(self) -> None:
        self.assertIn("B3_single_fold_debug", self.runs.columns)
        marked = self.runs[self.runs["B3_single_fold_debug"]]
        self.assertEqual(marked[["fold", "seed"]].to_records(index=False).tolist(), [(0, 3407)])

    def test_integrity_boundary_contains_all_240_frozen_sources(self) -> None:
        sources = cumulative._source_paths(ROOT)
        self.assertEqual(len(sources), 240)
        self.assertEqual(
            sum(path.name == "prototype_metadata.json" for path in sources),
            25,
        )
        self.assertEqual(sum(path.name in {"train.csv", "val.csv", "test.csv"} for path in sources), 15)

    def test_stage_summary_uses_macro_f1_and_sample_standard_deviation(self) -> None:
        self.assertEqual(self.summary["stage"].tolist(), ["B0", "B1", "B2", "B3"])
        self.assertEqual(self.summary["n"].tolist(), [25, 25, 25, 25])
        self.assertTrue(
            {
                "mean_macro_f1",
                "std_macro_f1",
                "min_macro_f1",
                "max_macro_f1",
            }.issubset(self.summary.columns)
        )
        self.assertNotIn("accuracy", " ".join(self.summary.columns).lower())
        protocols = self.summary.set_index("stage")["selection_protocol"].to_dict()
        self.assertEqual(protocols["B0"], "none")
        self.assertEqual(protocols["B1"], "none")
        self.assertEqual(protocols["B2"], "fixed_lambda_0_5_retrospective")
        self.assertEqual(protocols["B3"], "fixed_lambda_0_5_retrospective")

    def test_all_five_direct_paired_comparisons_are_reported(self) -> None:
        expected = ["B1 - B0", "B2 - B1", "B3 - B2", "B3 - B1", "B3 - B0"]
        self.assertEqual(self.paired["comparison"].tolist(), expected)
        self.assertEqual(self.paired["n"].tolist(), [25] * 5)
        self.assertEqual(self.folds.groupby("comparison").size().to_dict(), {name: 5 for name in expected})
        self.assertTrue(
            {
                "baseline_mean",
                "final_mean",
                "mean_delta",
                "std_delta",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "wilcoxon_p",
                "wins",
                "ties",
                "losses",
                "fold_0_mean_delta",
                "fold_1_mean_delta",
                "fold_2_mean_delta",
                "fold_3_mean_delta",
                "fold_4_mean_delta",
                "fold_cluster_bootstrap_ci95_low",
                "fold_cluster_bootstrap_ci95_high",
            }.issubset(self.paired.columns)
        )

    def test_b3_minus_b0_is_computed_from_direct_matched_rows(self) -> None:
        row = self.paired.set_index("comparison").loc["B3 - B0"]
        expected = self.runs["B3"].to_numpy() - self.runs["B0"].to_numpy()
        self.assertAlmostEqual(float(row["mean_delta"]), float(expected.mean()))
        for fold in range(5):
            mask = self.runs["fold"].to_numpy() == fold
            self.assertAlmostEqual(float(row[f"fold_{fold}_mean_delta"]), float(expected[mask].mean()))

    def test_case_selection_is_canonical_complete_unique_and_deterministic(self) -> None:
        expected_types = [
            "correct_cough",
            "correct_calm_grunt",
            "correct_feeding",
            "correct_stress_vocal",
            "feeding_stress_boundary_feeding",
            "feeding_stress_boundary_stress",
            "low_margin_uncertain",
        ]
        self.assertEqual(self.cases["case_type"].tolist(), expected_types)
        self.assertEqual(set(self.cases["fold"]), {0})
        self.assertEqual(set(self.cases["seed"]), {42})
        self.assertEqual(self.cases["source_id"].nunique(), 7)
        again = cumulative.select_case_studies(ROOT)
        pd.testing.assert_frame_equal(self.cases, again, check_exact=True)

    def test_correct_class_cases_are_correct_for_all_three_main_routes(self) -> None:
        correct = self.cases[self.cases["case_type"].str.startswith("correct_")]
        self.assertTrue((correct["raw_softmax_pred"] == correct["true_class"]).all())
        self.assertTrue((correct["main_prototype_pred"] == correct["true_class"]).all())
        self.assertTrue((correct["hierarchical_prototype_pred"] == correct["true_class"]).all())

    def test_boundary_cases_are_median_ranked_feeding_stress_errors(self) -> None:
        boundary = self.cases[self.cases["case_type"].str.startswith("feeding_stress_boundary_")]
        self.assertEqual(boundary["true_class"].tolist(), ["feeding", "stress_vocal"])
        any_incorrect = (
            (boundary["raw_softmax_pred"] != boundary["true_class"])
            | (boundary["main_prototype_pred"] != boundary["true_class"])
            | (boundary["hierarchical_prototype_pred"] != boundary["true_class"])
        )
        self.assertTrue(any_incorrect.all())
        self.assertEqual(set(boundary["selection_rule"]), {"lower median raw Top-2 margin among boundary errors"})

    def test_uncertain_case_has_smallest_remaining_b3_margin(self) -> None:
        prediction_path = (
            ROOT
            / "reports"
            / "prototype_cv5_exact_w05_fold0_seed42"
            / "evaluation"
            / "test_predictions.csv"
        )
        predictions = pd.read_csv(prediction_path)
        used = set(self.cases.iloc[:6]["source_id"].astype(str))
        remaining = predictions[~predictions["source_id"].astype(str).isin(used)].copy()
        probability_columns = [f"hierarchical_prob_{label}" for label in cumulative.MAIN_LABELS]
        probabilities = remaining[probability_columns].to_numpy(float)
        ordered = np.sort(probabilities, axis=1)
        expected_minimum = float(np.min(ordered[:, -1] - ordered[:, -2]))
        uncertain = self.cases.set_index("case_type").loc["low_margin_uncertain"]
        self.assertAlmostEqual(float(uncertain["margin"]), expected_minimum)
        self.assertEqual(
            uncertain["selection_rule"],
            "minimum remaining hierarchical prototype Top-2 margin",
        )

    def test_case_table_contains_requested_evidence_and_train_only_representatives(self) -> None:
        required = {
            "true_class",
            "raw_softmax_top2",
            "main_prototype_top2",
            "hierarchical_prototype_top2",
            "subtype_top2",
            "main_distances",
            "subtype_distances",
            "margin",
            "hierarchy_consistency",
            "closest_representative_sample",
            "closest_representative_source_id",
            "closest_representative_md5",
            "representative_split",
        }
        self.assertTrue(required.issubset(self.cases.columns))
        self.assertEqual(set(self.cases["representative_split"]), {"train_only"})
        self.assertTrue(self.cases[list(required)].notna().all().all())
        self.assertTrue(
            set(cumulative.METHOD_METADATA_FIELDS).issubset(self.cases.columns)
        )
        self.assertEqual(
            set(self.cases["selection_protocol"]),
            {"fixed_lambda_0_5_retrospective"},
        )
        self.assertEqual(
            set(self.cases["inference_route"]),
            {"hierarchical_prototype_candidate"},
        )

    def test_figure_builders_return_fixed_width_matplotlib_figures(self) -> None:
        first = cumulative.build_cumulative_figure(self.runs, self.summary, self.paired, self.folds)
        second = cumulative.build_case_figure(self.cases)
        try:
            self.assertAlmostEqual(float(first.get_size_inches()[0]), 183.0 / 25.4, places=6)
            self.assertAlmostEqual(float(second.get_size_inches()[0]), 183.0 / 25.4, places=6)
            self.assertEqual(len(first.axes), 4)
            self.assertGreaterEqual(len(second.axes), 2)
        finally:
            plt.close(first)
            plt.close(second)

    def test_dense_figure_keys_and_trace_labels_stay_inside_their_panels(self) -> None:
        first = cumulative.build_cumulative_figure(self.runs, self.summary, self.paired, self.folds)
        second = cumulative.build_case_figure(self.cases)
        try:
            paired_axis = first.axes[1]
            architecture_axis = first.axes[3]
            trace_axis = second.axes[3]
            first.canvas.draw()
            self.assertIsNone(paired_axis.get_legend())
            architecture_text = {text.get_text() for text in architecture_axis.texts}
            self.assertIn("Hierarchical-supervision\nCRNN", architecture_text)
            self.assertIn("Parallel candidate\nroutes", architecture_text)
            self.assertTrue(
                any("Primary Softmax route (Raw Softmax)" in text for text in architecture_text)
            )
            route_legend = "\n".join(text.get_text() for text in second.texts)
            self.assertIn("Primary Softmax route (Raw Softmax)", route_legend)
            self.assertIn("Main-class prototype candidate route", route_legend)
            self.assertIn("Hierarchical prototype candidate route", route_legend)
            self.assertTrue(
                all(
                    label.get_text().endswith((" P", " M", " H"))
                    for label in second.axes[0].get_yticklabels()
                )
            )
            self.assertTrue(all("\n" not in label.get_text() for label in trace_axis.get_yticklabels()))
            self.assertTrue(
                all(label.get_text().endswith(("[C]", "[I]")) for label in trace_axis.get_yticklabels())
            )
            self.assertEqual(sum(text.get_text().startswith("rep: ") for text in trace_axis.texts), 7)
            self.assertFalse(any(text.get_text().startswith("green:") for text in trace_axis.texts))
            self.assertTrue(
                all(len(text.get_text()) <= 18 for text in trace_axis.texts if text.get_text().startswith("rep: "))
            )
            for patch in architecture_axis.patches:
                if isinstance(patch, FancyBboxPatch):
                    bounds = patch.get_path().get_extents(patch.get_transform()).transformed(
                        architecture_axis.transAxes.inverted()
                    )
                    self.assertGreaterEqual(bounds.x0, 0.0)
                    self.assertLessEqual(bounds.x1, 1.0)
            first.canvas.draw()
            renderer = first.canvas.get_renderer()
            canvas = first.bbox
            b3_footers = [
                text
                for text in architecture_axis.texts
                if "hierarchical-prototype Top-1 decision ablation"
                in " ".join(text.get_text().split())
            ]
            self.assertEqual(len(b3_footers), 1)
            footer_bounds = b3_footers[0].get_window_extent(renderer)
            self.assertGreaterEqual(footer_bounds.x0, canvas.x0 - 1.0)
            self.assertLessEqual(footer_bounds.x1, canvas.x1 + 1.0)
        finally:
            plt.close(first)
            plt.close(second)

    def test_required_outputs_are_limited_to_requested_new_paths(self) -> None:
        outputs = {path.relative_to(ROOT).as_posix() for path in cumulative.required_output_paths(ROOT)}
        expected = {
            "paper/tables/cumulative_framework_runs_nomenclature_v1.csv",
            "paper/tables/cumulative_framework_summary_nomenclature_v1.csv",
            "paper/tables/cumulative_framework_paired_stats_nomenclature_v1.csv",
            "paper/tables/cumulative_framework_fold_stats_nomenclature_v1.csv",
            "paper/tables/prototype_prediction_case_studies_nomenclature_v1.csv",
            "paper/figures_journal/figure_cumulative_framework_nomenclature_v1.svg",
            "paper/figures_journal/figure_cumulative_framework_nomenclature_v1.pdf",
            "paper/figures_journal/figure_cumulative_framework_nomenclature_v1.png",
            "paper/figures_journal/figure_prototype_prediction_cases_nomenclature_v1.svg",
            "paper/figures_journal/figure_prototype_prediction_cases_nomenclature_v1.pdf",
            "paper/figures_journal/figure_prototype_prediction_cases_nomenclature_v1.png",
        }
        self.assertEqual(outputs, expected)

    def test_paper_ready_archive_paths_are_explicit_and_separate(self) -> None:
        outputs = {path.relative_to(ROOT).as_posix() for path in cumulative.archive_output_paths(ROOT)}
        self.assertEqual(
            outputs,
            {
                "paper_results/scripts/generate_cumulative_framework_analysis_nomenclature_v1.py",
                "paper_results/tables/cumulative_framework_runs_nomenclature_v1.csv",
                "paper_results/tables/cumulative_framework_summary_nomenclature_v1.csv",
                "paper_results/tables/cumulative_framework_paired_stats_nomenclature_v1.csv",
                "paper_results/tables/cumulative_framework_fold_stats_nomenclature_v1.csv",
                "paper_results/tables/prototype_prediction_case_studies_nomenclature_v1.csv",
            },
        )

    def test_versioned_outputs_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = cumulative.required_output_paths(root)[0]
            target.parent.mkdir(parents=True)
            target.write_text("historical", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                cumulative.refuse_existing(
                    (*cumulative.required_output_paths(root), *cumulative.archive_output_paths(root))
                )

    def test_exported_svgs_keep_editable_text_without_embedded_rasters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            figures = (
                cumulative.build_cumulative_figure(
                    self.runs, self.summary, self.paired, self.folds
                ),
                cumulative.build_case_figure(self.cases),
            )
            for figure, relative_stem in zip(
                figures, cumulative.FIGURE_STEMS, strict=True
            ):
                exported = cumulative._export_figure(
                    figure, Path(tmp) / Path(relative_stem).name
                )
                text = exported[0].read_text(encoding="utf-8")
                self.assertIn("<text", text)
                self.assertNotIn("<image", text)


if __name__ == "__main__":
    unittest.main()
