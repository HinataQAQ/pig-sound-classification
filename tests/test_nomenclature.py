from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from tools.nomenclature import (
    CANONICAL_IDS,
    DISPLAY_LABELS_EN,
    DISPLAY_LABELS_ZH,
    INFERENCE_ROUTE_IDS,
    LEGACY_ALIASES,
    METHOD_METADATA_FIELDS,
    MODEL_METADATA_FIELDS,
    MODEL_FAMILY_IDS,
    NOMENCLATURE_SCHEMA_VERSION,
    SELECTION_PROTOCOL_IDS,
    TRAINING_STAGE_IDS,
    build_method_metadata,
    build_model_metadata,
    canonicalize_id,
    canonicalize_inference_route,
    canonicalize_selection_protocol,
    display_label,
    metadata_from_legacy_record,
    preferred_legacy_inference_route_id,
    reconcile_selection_protocol,
    resolve_inference_method,
    resolve_inference_method_fields,
    try_canonicalize_inference_route,
    validate_id,
    validate_method_metadata,
    validate_model_metadata,
    validate_expected_artifact_role,
    validate_recorded_nomenclature_metadata,
    validate_recorded_artifact_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CanonicalIdTests(unittest.TestCase):
    def test_every_canonical_id_validates(self) -> None:
        expected = {
            "model_family": MODEL_FAMILY_IDS,
            "training_stage": TRAINING_STAGE_IDS,
            "inference_route": INFERENCE_ROUTE_IDS,
            "selection_protocol": SELECTION_PROTOCOL_IDS,
        }
        self.assertEqual(CANONICAL_IDS, expected)
        for namespace, identifiers in expected.items():
            with self.subTest(namespace=namespace):
                self.assertTrue(identifiers)
            for identifier in identifiers:
                with self.subTest(namespace=namespace, identifier=identifier):
                    self.assertEqual(validate_id(namespace, identifier), identifier)
                    self.assertEqual(canonicalize_id(namespace, identifier), identifier)

    def test_required_legacy_route_aliases_resolve(self) -> None:
        expected = {
            "raw_softmax": "primary_softmax",
            "prototype": "main_class_prototype_candidate",
            "main_prototype": "main_class_prototype_candidate",
            "hierarchical": "hierarchical_prototype_candidate",
            "hierarchical_prototype": "hierarchical_prototype_candidate",
        }
        self.assertEqual(LEGACY_ALIASES["inference_route"], expected)
        for legacy, canonical in expected.items():
            with self.subTest(legacy=legacy):
                self.assertEqual(canonicalize_inference_route(legacy), canonical)

    def test_every_declared_legacy_alias_resolves(self) -> None:
        for namespace, aliases in LEGACY_ALIASES.items():
            self.assertTrue(aliases)
            for legacy, canonical in aliases.items():
                with self.subTest(namespace=namespace, legacy=legacy):
                    self.assertEqual(
                        canonicalize_id(namespace, legacy), canonical
                    )

    def test_legacy_alias_can_emit_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = canonicalize_inference_route(
                "raw_softmax", warn_on_legacy=True
            )
        self.assertEqual(resolved, "primary_softmax")
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, DeprecationWarning)
        self.assertIn("raw_softmax", str(caught[0].message))
        self.assertIn("primary_softmax", str(caught[0].message))

    def test_unknown_ids_fail_clearly(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Unknown inference_route ID 'mystery_route'.*primary_softmax",
        ):
            canonicalize_inference_route("mystery_route")
        with self.assertRaisesRegex(ValueError, r"Unknown nomenclature namespace"):
            validate_id("optimizer", "adam")

    def test_fixed_lambda_is_not_validation_selected(self) -> None:
        fixed = canonicalize_selection_protocol("fixed_lambda_0.5")
        selected = canonicalize_selection_protocol("validation_selected")
        self.assertEqual(fixed, "fixed_lambda_0_5_retrospective")
        self.assertEqual(selected, "foldwise_validation_selected_lambda")
        self.assertNotEqual(fixed, selected)

        fixed_metadata = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol=fixed,
            inference_route="primary_softmax",
        )
        selected_metadata = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol=selected,
            inference_route="primary_softmax",
        )
        self.assertNotEqual(
            fixed_metadata["canonical_method_id"],
            selected_metadata["canonical_method_id"],
        )

    def test_semantically_impossible_stage_metadata_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires context_seconds=2.0"):
            build_model_metadata(
                model_family="hierarchical_supervision_crnn",
                training_stage="b2_validation_selected_hierarchical_crnn",
                context_seconds=1.0,
                selection_protocol="none",
            )
        with self.assertRaisesRegex(ValueError, "requires model_family"):
            build_model_metadata(
                model_family="logmel_crnn",
                training_stage="b2_validation_selected_hierarchical_crnn",
                context_seconds=2.0,
                selection_protocol="none",
            )
        with self.assertRaisesRegex(ValueError, "requires inference_route"):
            build_method_metadata(
                model_family="hierarchical_supervision_crnn",
                training_stage="b3_hierarchical_prototype_top1_ablation",
                context_seconds=2.0,
                selection_protocol="fixed_lambda_0_5_retrospective",
                inference_route="primary_softmax",
            )

    def test_bilingual_display_labels_are_complete_and_route_labels_are_canonical(self) -> None:
        for namespace, identifiers in CANONICAL_IDS.items():
            for identifier in identifiers:
                with self.subTest(namespace=namespace, identifier=identifier):
                    self.assertTrue(DISPLAY_LABELS_EN[namespace][identifier].strip())
                    self.assertTrue(DISPLAY_LABELS_ZH[namespace][identifier].strip())
        self.assertEqual(
            display_label("inference_route", "primary_softmax", language="en"),
            "Primary Softmax route (Raw Softmax)",
        )
        self.assertEqual(
            display_label(
                "inference_route",
                "main_class_prototype_candidate",
                language="zh",
            ),
            "主类别原型候选路由",
        )
        self.assertEqual(
            display_label(
                "training_stage",
                "b3_hierarchical_prototype_top1_ablation",
                language="en",
            ),
            "B3 — Hierarchical-prototype Top-1 decision ablation",
        )

    def test_registry_script_supports_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "nomenclature.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout.lower())
        self.assertIn(NOMENCLATURE_SCHEMA_VERSION, completed.stdout)


class CanonicalArtifactRoleTests(unittest.TestCase):
    def test_expected_artifact_role_rejects_b2_metadata_for_b3(self) -> None:
        b2 = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol="fixed_lambda_0_5_retrospective",
            inference_route="primary_softmax",
            legacy_method_id="raw_softmax",
        )
        with self.assertRaisesRegex(ValueError, "artifact role"):
            validate_expected_artifact_role(
                b2,
                expected_model_family="hierarchical_supervision_crnn",
                expected_training_stage=(
                    "b3_hierarchical_prototype_top1_ablation"
                ),
                expected_context_seconds=2.0,
                expected_selection_protocol=(
                    "fixed_lambda_0_5_retrospective"
                ),
                expected_inference_route=(
                    "hierarchical_prototype_candidate"
                ),
                context="B3 evaluation",
            )

    def test_expected_artifact_role_rejects_route_field_conflicts(self) -> None:
        role = {
            "expected_model_family": "hierarchical_supervision_crnn",
            "expected_training_stage": (
                "b3_hierarchical_prototype_top1_ablation"
            ),
            "expected_context_seconds": 2.0,
            "expected_selection_protocol": (
                "fixed_lambda_0_5_retrospective"
            ),
            "expected_inference_route": (
                "hierarchical_prototype_candidate"
            ),
            "context": "B3 evaluation",
        }
        with self.assertRaisesRegex(
            ValueError, "Conflicting inference method fields"
        ):
            validate_expected_artifact_role(
                {
                    "selection_method": "hierarchical",
                    "inference_route": "primary_softmax",
                },
                **role,
            )
        with self.assertRaisesRegex(
            ValueError, "Conflicting inference method fields"
        ):
            validate_expected_artifact_role(
                {
                    "legacy_method_id": "prototype",
                    "inference_route": "hierarchical_prototype_candidate",
                },
                **role,
            )

    def test_expected_artifact_role_rejects_consistent_wrong_route(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "inference route"):
            validate_expected_artifact_role(
                {
                    "dur_s": 2.0,
                    "selection_method": "raw_softmax",
                },
                expected_model_family="hierarchical_supervision_crnn",
                expected_training_stage=(
                    "b3_hierarchical_prototype_top1_ablation"
                ),
                expected_context_seconds=2.0,
                expected_selection_protocol=(
                    "fixed_lambda_0_5_retrospective"
                ),
                expected_inference_route=(
                    "hierarchical_prototype_candidate"
                ),
                context="B3 evaluation",
            )

    def test_expected_artifact_role_rejects_selection_protocol_conflict(
        self,
    ) -> None:
        validation_selected = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol="foldwise_validation_selected_lambda",
            inference_route="primary_softmax",
            legacy_method_id="raw_softmax",
        )
        with self.assertRaisesRegex(ValueError, "selection_protocol"):
            validate_expected_artifact_role(
                validation_selected,
                expected_model_family="hierarchical_supervision_crnn",
                expected_training_stage=(
                    "b2_validation_selected_hierarchical_crnn"
                ),
                expected_context_seconds=2.0,
                expected_selection_protocol=(
                    "fixed_lambda_0_5_retrospective"
                ),
                expected_inference_route="primary_softmax",
                context="fixed-lambda B2 summary",
            )

    def test_expected_artifact_role_accepts_consistent_legacy_only_record(
        self,
    ) -> None:
        validated = validate_expected_artifact_role(
            {
                "dur_s": 2.0,
                "selection_method": "hierarchical",
            },
            expected_model_family="hierarchical_supervision_crnn",
            expected_training_stage="b3_hierarchical_prototype_top1_ablation",
            expected_context_seconds=2.0,
            expected_selection_protocol="fixed_lambda_0_5_retrospective",
            expected_inference_route="hierarchical_prototype_candidate",
            context="legacy B3 evaluation",
        )
        self.assertEqual(validated, {"context_seconds": 2.0})


class CompatibilityMetadataTests(unittest.TestCase):
    def test_explicit_schema_blocks_are_strict_but_legacy_is_permissive(self) -> None:
        method = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b3_hierarchical_prototype_top1_ablation",
            context_seconds=2.0,
            selection_protocol="fixed_lambda_0_5_retrospective",
            inference_route="hierarchical_prototype_candidate",
            legacy_method_id="hierarchical",
        )
        model = build_model_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol="fixed_lambda_0_5_retrospective",
        )
        self.assertEqual(
            validate_recorded_nomenclature_metadata(method), method
        )
        self.assertEqual(
            validate_recorded_nomenclature_metadata(model), model
        )
        self.assertIsNone(
            validate_recorded_nomenclature_metadata(
                {"selection_method": "hierarchical"}
            )
        )

        corrupt = {**method, "canonical_method_id": "corrupt"}
        with self.assertRaisesRegex(
            ValueError, "Invalid canonical method metadata"
        ):
            validate_recorded_nomenclature_metadata(corrupt)
        with self.assertRaisesRegex(
            ValueError, "Unsupported nomenclature_schema_version"
        ):
            validate_recorded_nomenclature_metadata(
                {**model, "nomenclature_schema_version": "future.v9"}
            )
        incomplete_method = {
            **model,
            "inference_route": "",
            "canonical_method_id": None,
            "display_name_en": "",
            "display_name_zh": "",
            "legacy_method_id": "",
            "selection_method": "hierarchical",
        }
        with self.assertRaisesRegex(
            ValueError,
            "Invalid canonical method metadata|Missing canonical|non-empty",
        ):
            validate_recorded_nomenclature_metadata(incomplete_method)

        with self.assertRaisesRegex(ValueError, "artifact role"):
            validate_recorded_artifact_identity(
                method,
                expected_model_family="hierarchical_supervision_crnn",
                expected_training_stage=(
                    "b2_validation_selected_hierarchical_crnn"
                ),
                expected_context_seconds=2.0,
                context="synthetic calibration",
            )
        with self.assertRaisesRegex(ValueError, "artifact role"):
            validate_recorded_artifact_identity(
                {
                    "model_family": "logmel_crnn",
                    "dur_s": 2.0,
                },
                expected_model_family="hierarchical_supervision_crnn",
                expected_training_stage=(
                    "b2_validation_selected_hierarchical_crnn"
                ),
                expected_context_seconds=2.0,
                context="legacy bundle",
            )
        with self.assertRaisesRegex(ValueError, "dur_s"):
            validate_recorded_artifact_identity(
                {**model, "dur_s": 1.0},
                expected_model_family="hierarchical_supervision_crnn",
                expected_training_stage=(
                    "b2_validation_selected_hierarchical_crnn"
                ),
                expected_context_seconds=2.0,
                context="versioned bundle",
            )

    def test_inference_method_fields_reconcile_aliases_and_reject_conflicts(self) -> None:
        self.assertEqual(
            resolve_inference_method_fields(
                {
                    "selection_method": "raw_softmax",
                    "inference_route": "primary_softmax",
                }
            ),
            ("raw_softmax", "primary_softmax"),
        )
        with self.assertRaisesRegex(ValueError, "Conflicting inference method fields"):
            resolve_inference_method_fields(
                {
                    "selection_method": "hierarchical",
                    "inference_route": "primary_softmax",
                }
            )

    def test_selection_protocol_reconciliation_rejects_concrete_conflicts(self) -> None:
        self.assertEqual(
            reconcile_selection_protocol(
                "fixed_lambda_0_5_retrospective", requested=None
            ),
            "fixed_lambda_0_5_retrospective",
        )
        self.assertEqual(
            reconcile_selection_protocol(
                "none", requested="fixed_lambda_0.5"
            ),
            "fixed_lambda_0_5_retrospective",
        )
        with self.assertRaisesRegex(ValueError, "Conflicting selection_protocol"):
            reconcile_selection_protocol(
                "fixed_lambda_0_5_retrospective",
                requested="foldwise_validation_selected_lambda",
            )

    def test_model_metadata_contains_only_applicable_common_fields(self) -> None:
        metadata = build_model_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2,
            selection_protocol="foldwise_validation_selected_lambda",
        )
        self.assertEqual(
            metadata,
            {
                "nomenclature_schema_version": NOMENCLATURE_SCHEMA_VERSION,
                "model_family": "hierarchical_supervision_crnn",
                "training_stage": "b2_validation_selected_hierarchical_crnn",
                "context_seconds": 2.0,
                "selection_protocol": "foldwise_validation_selected_lambda",
            },
        )
        self.assertEqual(set(metadata), set(MODEL_METADATA_FIELDS))
        self.assertEqual(validate_model_metadata(metadata), metadata)

    def test_canonical_method_metadata_round_trips_through_json(self) -> None:
        metadata = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol="foldwise_validation_selected_lambda",
            inference_route="raw_softmax",
            legacy_method_id="raw_softmax",
        )
        self.assertEqual(set(metadata), set(METHOD_METADATA_FIELDS))
        self.assertEqual(
            metadata["canonical_method_id"],
            "b2_validation_selected_hierarchical_crnn::"
            "foldwise_validation_selected_lambda::primary_softmax",
        )
        restored = json.loads(json.dumps(metadata, ensure_ascii=False))
        self.assertEqual(validate_method_metadata(restored), metadata)

    def test_old_result_sample_can_be_read_without_schema_rewrite(self) -> None:
        path = ROOT / "paper" / "tables" / (
            "table5_clean_softmax_main_prototype_hierarchical_prototype.csv"
        )
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            [row["method"] for row in rows],
            [
                "raw_softmax",
                "calibrated_softmax",
                "prototype",
                "hierarchical",
                "fused",
            ],
        )

        canonical_rows = []
        for row in rows:
            route = try_canonicalize_inference_route(row["method"])
            if route is None:
                continue
            enriched = metadata_from_legacy_record(
                row,
                model_family="hierarchical_supervision_crnn",
                training_stage=(
                    "b3_hierarchical_prototype_top1_ablation"
                    if route == "hierarchical_prototype_candidate"
                    else "b2_validation_selected_hierarchical_crnn"
                ),
                context_seconds=2.0,
                selection_protocol="fixed_lambda_0_5_retrospective",
            )
            self.assertEqual(enriched["method"], row["method"])
            self.assertEqual(enriched["legacy_method_id"], row["method"])
            canonical_rows.append(enriched["inference_route"])

        self.assertEqual(
            canonical_rows,
            [
                "primary_softmax",
                "main_class_prototype_candidate",
                "hierarchical_prototype_candidate",
            ],
        )
        self.assertIsNone(try_canonicalize_inference_route("calibrated_softmax"))
        self.assertIsNone(try_canonicalize_inference_route("fused"))

    def test_metadata_enrichment_does_not_mutate_legacy_record(self) -> None:
        legacy = {"selection_method": "hierarchical", "macro_f1": 0.95}
        original = dict(legacy)
        enriched = metadata_from_legacy_record(
            legacy,
            model_family="hierarchical_supervision_crnn",
            training_stage="b3_hierarchical_prototype_top1_ablation",
            context_seconds=2.0,
            selection_protocol="fixed_lambda_0.5",
        )
        self.assertEqual(legacy, original)
        self.assertEqual(enriched["selection_method"], "hierarchical")
        self.assertEqual(
            enriched["selection_protocol"], "fixed_lambda_0_5_retrospective"
        )
        self.assertEqual(
            enriched["inference_route"], "hierarchical_prototype_candidate"
        )

    def test_canonical_record_uses_preferred_legacy_storage_id(self) -> None:
        record = {"inference_route": "primary_softmax", "score": 0.9}
        enriched = metadata_from_legacy_record(
            record,
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol="foldwise_validation_selected_lambda",
        )
        self.assertEqual(enriched["inference_route"], "primary_softmax")
        self.assertEqual(enriched["legacy_method_id"], "raw_softmax")

    def test_legacy_record_rejects_populated_noncanonical_or_unknown_method_fields(self) -> None:
        common = {
            "model_family": "hierarchical_supervision_crnn",
            "training_stage": "b2_validation_selected_hierarchical_crnn",
            "context_seconds": 2.0,
            "selection_protocol": "foldwise_validation_selected_lambda",
        }
        with self.assertRaisesRegex(ValueError, "noncanonical method fields"):
            metadata_from_legacy_record(
                {"method": "fused", "inference_route": "primary_softmax"},
                **common,
            )
        with self.assertRaisesRegex(ValueError, "Unknown inference method"):
            metadata_from_legacy_record(
                {"method": "mystery", "inference_route": "primary_softmax"},
                **common,
            )

    def test_preferred_legacy_ids_keep_existing_storage_schema(self) -> None:
        expected = {
            "primary_softmax": "raw_softmax",
            "main_class_prototype_candidate": "prototype",
            "hierarchical_prototype_candidate": "hierarchical",
        }
        for canonical, legacy in expected.items():
            with self.subTest(canonical=canonical):
                self.assertEqual(
                    preferred_legacy_inference_route_id(canonical), legacy
                )

    def test_inference_method_resolution_preserves_legacy_storage_and_cli(self) -> None:
        expected = {
            "raw_softmax": ("raw_softmax", "primary_softmax"),
            "primary_softmax": ("raw_softmax", "primary_softmax"),
            "prototype": (
                "prototype",
                "main_class_prototype_candidate",
            ),
            "main_class_prototype_candidate": (
                "prototype",
                "main_class_prototype_candidate",
            ),
            "hierarchical": (
                "hierarchical",
                "hierarchical_prototype_candidate",
            ),
            "hierarchical_prototype_candidate": (
                "hierarchical",
                "hierarchical_prototype_candidate",
            ),
            "softmax": ("calibrated_softmax", None),
            "calibrated_softmax": ("calibrated_softmax", None),
            "fused": ("fused", None),
        }
        for value, resolution in expected.items():
            with self.subTest(value=value):
                self.assertEqual(resolve_inference_method(value), resolution)
        with self.assertRaisesRegex(ValueError, "Unknown inference method"):
            resolve_inference_method("unknown_method")


class ImmutableArtifactCompatibilityTests(unittest.TestCase):
    def test_existing_checkpoint_loads_with_identical_keys(self) -> None:
        baseline_name = os.environ.get(
            "PIGSOUND_NOMENCLATURE_CHECKPOINT_BASELINE", ""
        )
        if not baseline_name:
            self.skipTest("local checkpoint compatibility baseline not provided")
        baseline_path = Path(baseline_name)
        baseline = json.loads(
            baseline_path.read_text(encoding="utf-8-sig")
        )
        checkpoint = Path(str(baseline["path"]))
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(_sha256(checkpoint), baseline["sha256"])

        import torch

        state_dict = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        keys = sorted(str(key) for key in state_dict)
        signatures = [
            f"{key}|{tuple(state_dict[key].shape)}|{state_dict[key].dtype}"
            for key in keys
        ]
        self.assertEqual(len(keys), baseline["state_dict_key_count"])
        self.assertEqual(
            hashlib.sha256("\n".join(keys).encode()).hexdigest(),
            baseline["state_dict_keys_sha256"],
        )
        self.assertEqual(
            hashlib.sha256("\n".join(signatures).encode()).hexdigest(),
            baseline["state_dict_signature_sha256"],
        )
        self.assertTrue(any(key.startswith("main_head.") for key in keys))
        self.assertTrue(any(key.startswith("aux_head.") for key in keys))

        tools_dir = ROOT / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        from train_hier_longcontext_crnn import HierCRNN

        model = HierCRNN(
            num_main_classes=4,
            num_aux_classes=6,
            in_channels=1,
            use_se=True,
            rnn_type="gru",
            pooling_type="mean",
        )
        model.load_state_dict(state_dict, strict=True)

    def test_tracked_historical_artifact_sha_manifest_is_unchanged(self) -> None:
        manifest = ROOT / "tests" / "data" / (
            "protected_artifacts_nomenclature_v1.sha256.json"
        )
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        self.assertEqual(
            payload["schema_version"],
            "pig_sound_protected_artifacts.sha256.v1",
        )
        self.assertEqual(
            payload["base_commit_sha"],
            "3b336996dba15c6227d652b276c9882503714659",
        )
        self.assertEqual(
            payload["source_tree_sha"],
            "417344e8a3c5fcdad715cfcd30e73525e340c311",
        )
        self.assertEqual(payload["hash_algorithm"], "sha256")
        self.assertEqual(payload["hash_domain"], "git_blob_bytes")
        self.assertEqual(
            payload["verification_requires"],
            "full_git_objects_for_base_commit_and_source_tree",
        )
        self.assertEqual(payload["entry_count"], 2624)

        entries = payload["entries"]
        self.assertEqual(len(entries), 2624)
        self.assertTrue(
            all(set(entry) == {"path", "sha256"} for entry in entries)
        )
        protected_paths = [str(entry["path"]) for entry in entries]
        self.assertEqual(protected_paths, sorted(protected_paths))
        self.assertEqual(len(set(protected_paths)), 2624)
        for relative in protected_paths:
            with self.subTest(relative=relative):
                self.assertNotIn("\\", relative)
                self.assertFalse(Path(relative).is_absolute())
                self.assertNotIn("..", Path(relative).parts)
        path_set_sha256 = hashlib.sha256(
            ("\n".join(protected_paths) + "\n").encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            path_set_sha256,
            "6016600eccf5c6a13b00c1d64ecb4efe0e388f9fa4e80057ef0d3432156ddcf5",
        )
        self.assertEqual(payload["path_set_sha256"], path_set_sha256)

        def git_tree_entries(treeish: str) -> dict[str, str]:
            completed = subprocess.run(
                ["git", "ls-tree", "-r", treeish],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            result: dict[str, str] = {}
            for line in completed.stdout.splitlines():
                metadata, relative = line.split("\t", 1)
                result[relative] = metadata.split()[2]
            return result

        base_tree = git_tree_entries(payload["base_commit_sha"])
        source_tree = git_tree_entries(payload["source_tree_sha"])

        def is_protected(relative: str) -> bool:
            return (
                (
                    relative.startswith("data/")
                    and not relative.startswith("data/embeddings_")
                )
                or relative.startswith("det")
                or relative.startswith("eval")
                or relative.startswith("paper/")
                or relative.startswith("paper_results/")
                or relative.startswith("reports/")
            )

        expected_paths = sorted(
            relative for relative in base_tree if is_protected(relative)
        )
        self.assertEqual(protected_paths, expected_paths)
        self.assertEqual(
            {relative: base_tree[relative] for relative in protected_paths},
            {relative: source_tree[relative] for relative in protected_paths},
        )
        current_tree = git_tree_entries("HEAD")
        self.assertEqual(
            {relative: base_tree[relative] for relative in protected_paths},
            {relative: current_tree[relative] for relative in protected_paths},
        )

        for diff_args in (
            ("diff", "--name-only", "--"),
            ("diff", "--cached", "--name-only", "--"),
        ):
            completed = subprocess.run(
                ["git", *diff_args],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            changed_protected = sorted(
                relative
                for relative in completed.stdout.splitlines()
                if is_protected(relative)
            )
            self.assertEqual(changed_protected, [])

        process = subprocess.Popen(
            ["git", "cat-file", "--batch"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self.assertIsNotNone(process.stdin)
        self.assertIsNotNone(process.stdout)
        failures: list[str] = []
        assert process.stdin is not None
        assert process.stdout is not None
        for entry in entries:
            relative = str(entry["path"])
            if not (ROOT / relative).is_file():
                failures.append(f"missing: {relative}")
                continue
            object_id = base_tree[relative]
            process.stdin.write(f"{object_id}\n".encode("ascii"))
            process.stdin.flush()
            header = process.stdout.readline().decode("ascii").strip()
            returned_id, object_type, size_text = header.split()
            self.assertEqual(returned_id, object_id)
            self.assertEqual(object_type, "blob")
            content = process.stdout.read(int(size_text))
            self.assertEqual(process.stdout.read(1), b"\n")
            actual = hashlib.sha256(content).hexdigest()
            expected = str(entry["sha256"])
            if actual != expected:
                failures.append(
                    f"changed: {relative} expected={expected} actual={actual}"
                )
        process.stdin.close()
        return_code = process.wait(timeout=30)
        process.stdout.close()
        self.assertEqual(return_code, 0)
        self.assertEqual(failures, [])


class ActivePrototypeScriptCompatibilityTests(unittest.TestCase):
    def test_summary_reader_preserves_unversioned_source_nomenclature(self) -> None:
        from summarize_cv5_exact_prototype import record_from_metrics

        payload = {
            "model_family": "hierarchical_crnn",
            "dur_s": 2.0,
            "selection_protocol": "fixed_lambda_0.5",
            "methods": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            record = record_from_metrics(path)

        self.assertEqual(
            record["model_family"], "hierarchical_supervision_crnn"
        )
        self.assertEqual(record["context_seconds"], 2.0)
        self.assertEqual(
            record["selection_protocol"],
            "fixed_lambda_0_5_retrospective",
        )

    def test_summary_reader_rejects_conflicting_duration_fields(self) -> None:
        from summarize_cv5_exact_prototype import record_from_metrics

        payload = {
            "dur_s": 1.0,
            "context_seconds": 2.0,
            "methods": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "Conflicting dur_s/context_seconds"
            ):
                record_from_metrics(path)

    def test_summary_reader_normalizes_nested_method_keys(self) -> None:
        from summarize_cv5_exact_prototype import record_from_metrics

        payload = {
            "model_family": "hierarchical_supervision_crnn",
            "context_seconds": 2.0,
            "selection_protocol": "fixed_lambda_0_5_retrospective",
            "methods": {
                "primary_softmax": {"method": "raw_softmax"},
                "main_prototype": {"method": "prototype"},
                "hierarchical_prototype": {"method": "hierarchical"},
                "calibrated_softmax": {"method": "calibrated_softmax"},
                "fused": {"method": "fused"},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            record = record_from_metrics(path)

        self.assertEqual(
            tuple(record["methods"]),
            (
                "raw_softmax",
                "prototype",
                "hierarchical",
                "calibrated_softmax",
                "fused",
            ),
        )
        self.assertEqual(record["methods"]["raw_softmax"]["method"], "raw_softmax")

    def test_summary_reader_rejects_unknown_or_duplicate_method_keys(self) -> None:
        from summarize_cv5_exact_prototype import record_from_metrics

        for methods, message in (
            ({"mystery_route": {}}, "Unknown inference method"),
            (
                {"raw_softmax": {}, "primary_softmax": {}},
                "duplicate aliases",
            ),
        ):
            with self.subTest(methods=tuple(methods)):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "metrics.json"
                    path.write_text(
                        json.dumps({"methods": methods}), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        record_from_metrics(path)

    def test_summary_reader_validates_and_reconciles_nested_schema_blocks(self) -> None:
        from summarize_cv5_exact_prototype import record_from_metrics

        top = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b3_hierarchical_prototype_top1_ablation",
            context_seconds=2.0,
            selection_protocol="fixed_lambda_0_5_retrospective",
            inference_route="hierarchical_prototype_candidate",
            legacy_method_id="hierarchical",
        )
        primary = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol="fixed_lambda_0_5_retrospective",
            inference_route="primary_softmax",
            legacy_method_id="raw_softmax",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(
                json.dumps({**top, "methods": {"primary_softmax": primary}}),
                encoding="utf-8",
            )
            record = record_from_metrics(path)
        self.assertEqual(tuple(record["methods"]), ("raw_softmax",))
        self.assertEqual(
            record["methods"]["raw_softmax"]["inference_route"],
            "primary_softmax",
        )

        corrupt = {**primary, "canonical_method_id": "corrupt"}
        conflicting_route = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol="fixed_lambda_0_5_retrospective",
            inference_route="main_class_prototype_candidate",
            legacy_method_id="prototype",
        )
        conflicting_source = {
            **primary,
            "selection_protocol": "foldwise_validation_selected_lambda",
            "canonical_method_id": (
                "b2_validation_selected_hierarchical_crnn::"
                "foldwise_validation_selected_lambda::primary_softmax"
            ),
        }
        for block, message in (
            (corrupt, "Invalid canonical method metadata"),
            (conflicting_route, "does not match methods key"),
            (conflicting_source, "Conflicting nested selection_protocol"),
        ):
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "metrics.json"
                    path.write_text(
                        json.dumps({**top, "methods": {"raw_softmax": block}}),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        record_from_metrics(path)

        model_block = build_model_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol="fixed_lambda_0_5_retrospective",
        )
        writer_shaped_model_block = {
            **model_block,
            **{
                field: None
                for field in (
                    set(METHOD_METADATA_FIELDS) - set(MODEL_METADATA_FIELDS)
                )
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        **top,
                        "methods": {
                            "calibrated_softmax": writer_shaped_model_block,
                            "fused": writer_shaped_model_block,
                        },
                    }
                ),
                encoding="utf-8",
            )
            record = record_from_metrics(path)
        self.assertEqual(
            tuple(record["methods"]), ("calibrated_softmax", "fused")
        )
        self.assertEqual(
            record["methods"]["fused"]["selection_protocol"],
            "fixed_lambda_0_5_retrospective",
        )

        populated_partial = {
            **writer_shaped_model_block,
            "inference_route": "primary_softmax",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        **top,
                        "methods": {
                            "calibrated_softmax": populated_partial
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "route metadata fields"):
                record_from_metrics(path)

    def test_summary_inherits_and_validates_source_nomenclature(self) -> None:
        from summarize_cv5_exact_prototype import resolve_aggregate_nomenclature

        records = [
            {
                "model_family": "hierarchical_supervision_crnn",
                "context_seconds": 2.0,
                "selection_protocol": "fixed_lambda_0.5",
            },
            {
                "model_family": "hierarchical_supervision_crnn",
                "context_seconds": 2.0,
                "selection_protocol": "fixed_lambda_0_5_retrospective",
            },
        ]
        self.assertEqual(
            resolve_aggregate_nomenclature(records),
            build_model_metadata(
                model_family="hierarchical_supervision_crnn",
                training_stage=(
                    "b2_validation_selected_hierarchical_crnn"
                ),
                context_seconds=2.0,
                selection_protocol="fixed_lambda_0_5_retrospective",
            ),
        )
        with self.assertRaisesRegex(ValueError, "Conflicting selection_protocol"):
            resolve_aggregate_nomenclature(
                records,
                requested_selection_protocol=(
                    "foldwise_validation_selected_lambda"
                ),
            )
        with self.assertRaisesRegex(ValueError, "context_seconds"):
            resolve_aggregate_nomenclature(
                [records[0], {**records[1], "context_seconds": 1.5}]
            )
        with self.assertRaisesRegex(ValueError, "selection_protocol"):
            resolve_aggregate_nomenclature(
                [
                    records[0],
                    {
                        "model_family": "hierarchical_supervision_crnn",
                        "context_seconds": 2.0,
                    },
                ]
            )
        filled = resolve_aggregate_nomenclature(
            [
                records[0],
                {
                    "model_family": "hierarchical_supervision_crnn",
                    "context_seconds": 2.0,
                },
            ],
            requested_selection_protocol="fixed_lambda_0_5_retrospective",
        )
        self.assertEqual(
            filled["selection_protocol"],
            "fixed_lambda_0_5_retrospective",
        )
        with self.assertRaisesRegex(ValueError, "context_seconds"):
            resolve_aggregate_nomenclature(
                [
                    {**records[0], "context_seconds": 1.5},
                    {
                        "model_family": "hierarchical_supervision_crnn",
                        "selection_protocol": "fixed_lambda_0_5_retrospective",
                    },
                ]
            )

    def test_summary_resolver_canonicalizes_unversioned_dur_s_sources(self) -> None:
        from summarize_cv5_exact_prototype import resolve_aggregate_nomenclature

        self.assertEqual(
            resolve_aggregate_nomenclature(
                [
                    {
                        "model_family": "hierarchical_crnn",
                        "dur_s": 2.0,
                        "selection_protocol": "fixed_lambda_0.5",
                    }
                ]
            ),
            build_model_metadata(
                model_family="hierarchical_supervision_crnn",
                training_stage=(
                    "b2_validation_selected_hierarchical_crnn"
                ),
                context_seconds=2.0,
                selection_protocol="fixed_lambda_0_5_retrospective",
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "Conflicting dur_s/context_seconds"
        ):
            resolve_aggregate_nomenclature(
                [{"dur_s": 1.0, "context_seconds": 2.0}]
            )

    def test_summary_aggregate_schema_v1_blocks_are_complete_and_stable(
        self,
    ) -> None:
        from summarize_cv5_exact_prototype import (
            SUMMARY_METHODS,
            aggregate_metadata_for_method,
            resolve_aggregate_nomenclature,
        )

        incomplete = {
            "nomenclature_schema_version": NOMENCLATURE_SCHEMA_VERSION,
            "model_family": "hierarchical_supervision_crnn",
            "context_seconds": 2.0,
            "selection_protocol": "fixed_lambda_0_5_retrospective",
        }
        for artifact in ("run row", "aggregate provenance"):
            with self.subTest(artifact=artifact):
                with self.assertRaisesRegex(
                    ValueError, "training_stage"
                ):
                    validate_model_metadata(incomplete)

        model_metadata = resolve_aggregate_nomenclature(
            [
                {
                    "model_family": "hierarchical_supervision_crnn",
                    "context_seconds": 2.0,
                    "selection_protocol": "fixed_lambda_0.5",
                }
            ]
        )
        run_row = {"fold": 0, "seed": 42, **model_metadata}
        provenance = {
            "artifact_type": "cv5_exact_prototype_aggregate_provenance",
            **model_metadata,
        }
        self.assertEqual(
            validate_model_metadata(run_row), model_metadata
        )
        self.assertEqual(
            validate_model_metadata(provenance), model_metadata
        )

        canonical_methods = [
            aggregate_metadata_for_method(method, model_metadata)
            for method in SUMMARY_METHODS
            if resolve_inference_method(method)[1] is not None
        ]
        for method in canonical_methods:
            self.assertEqual(validate_method_metadata(method), method)
        stages = {
            method["inference_route"]: method["training_stage"]
            for method in canonical_methods
        }
        self.assertEqual(
            stages["primary_softmax"],
            "b2_validation_selected_hierarchical_crnn",
        )
        self.assertEqual(
            stages["main_class_prototype_candidate"],
            "b2_validation_selected_hierarchical_crnn",
        )
        self.assertEqual(
            stages["hierarchical_prototype_candidate"],
            "b3_hierarchical_prototype_top1_ablation",
        )

        restored = json.loads(
            json.dumps(
                {
                    "run": run_row,
                    "provenance": provenance,
                    "canonical_methods": canonical_methods,
                }
            )
        )
        validate_model_metadata(restored["run"])
        validate_model_metadata(restored["provenance"])
        for method in restored["canonical_methods"]:
            validate_method_metadata(method)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=run_row)
                writer.writeheader()
                writer.writerow(run_row)
            with path.open("r", encoding="utf-8", newline="") as handle:
                csv_row = next(csv.DictReader(handle))
        csv_row["context_seconds"] = float(csv_row["context_seconds"])
        self.assertEqual(
            validate_model_metadata(csv_row), model_metadata
        )

    def test_summary_main_writes_strict_aggregate_metadata(self) -> None:
        import summarize_cv5_exact_prototype as summary_module

        source_record = {
            "model_family": "hierarchical_supervision_crnn",
            "context_seconds": 2.0,
            "selection_protocol": "fixed_lambda_0.5",
        }
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "aggregate"
            args = summary_module.argparse.Namespace(
                metrics_json=[Path("metrics.json")],
                out_prefix=prefix,
                expected_folds="0",
                expected_seeds="42",
                expected_lambda=0.5,
                bootstrap_samples=10,
                bootstrap_seed=3407,
                selection_protocol="fixed_lambda_0_5_retrospective",
                allow_overwrite=False,
            )
            method_rows = [
                {"method": method}
                for method in summary_module.SUMMARY_METHODS
            ]
            confusion_rows = [
                {"method": method}
                for method in summary_module.CONFUSION_METHODS
            ]
            with (
                patch.object(summary_module, "parse_args", return_value=args),
                patch.object(
                    summary_module,
                    "record_from_metrics",
                    return_value=source_record,
                ),
                patch.object(
                    summary_module,
                    "validate_aggregate_records",
                    return_value={"screening_result": False},
                ),
                patch.object(
                    summary_module,
                    "flatten_run_record",
                    return_value={"fold": 0, "seed": 42},
                ),
                patch.object(
                    summary_module,
                    "aggregate_method_metrics",
                    return_value=method_rows,
                ),
                patch.object(
                    summary_module,
                    "paired_macro_f1_statistics",
                    return_value=[],
                ),
                patch.object(
                    summary_module,
                    "aggregate_confusion_matrices",
                    return_value=confusion_rows,
                ),
            ):
                summary_module.main()

            runs_path = Path(f"{prefix}_final_runs.csv")
            with runs_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                run_row = next(csv.DictReader(handle))
            run_row["context_seconds"] = float(
                run_row["context_seconds"]
            )
            validate_model_metadata(run_row)

            provenance = json.loads(
                Path(f"{prefix}_final_provenance.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            validate_model_metadata(provenance)
            for method in provenance["canonical_methods"]:
                validate_method_metadata(method)

            with Path(f"{prefix}_final_summary.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                summary_rows = list(csv.DictReader(handle))
        stages = {
            row["method"]: row["training_stage"]
            for row in summary_rows
        }
        self.assertEqual(
            stages["raw_softmax"],
            "b2_validation_selected_hierarchical_crnn",
        )
        self.assertEqual(
            stages["prototype"],
            "b2_validation_selected_hierarchical_crnn",
        )
        self.assertEqual(
            stages["hierarchical"],
            "b3_hierarchical_prototype_top1_ablation",
        )

    def test_summary_record_reader_preserves_existing_canonical_metadata(self) -> None:
        from summarize_cv5_exact_prototype import record_from_metrics

        payload = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b3_hierarchical_prototype_top1_ablation",
            context_seconds=2.0,
            selection_protocol="fixed_lambda_0_5_retrospective",
            inference_route="hierarchical_prototype_candidate",
            legacy_method_id="hierarchical",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            record = record_from_metrics(path)
        for field in (
            "nomenclature_schema_version",
            "model_family",
            "training_stage",
            "context_seconds",
            "selection_protocol",
            "inference_route",
            "canonical_method_id",
            "display_name_en",
            "display_name_zh",
            "legacy_method_id",
        ):
            self.assertEqual(record[field], payload[field])

        corrupt = dict(payload)
        corrupt["canonical_method_id"] = "corrupt"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps(corrupt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid canonical method metadata"):
                record_from_metrics(path)

    @classmethod
    def setUpClass(cls) -> None:
        tools_dir = ROOT / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))

    def test_calibrator_accepts_canonical_and_legacy_method_values(self) -> None:
        from calibrate_prototype_predictor import parse_args

        common = [
            "calibrate_prototype_predictor.py",
            "--prototype_bundle",
            "bundle.npz",
            "--val_manifest",
            "val.csv",
            "--ckpt",
            "model.pt",
            "--out_dir",
            "new-output",
        ]
        accepted = (
            "primary_softmax",
            "main_class_prototype_candidate",
            "hierarchical_prototype_candidate",
            "raw_softmax",
            "prototype",
            "main_prototype",
            "hierarchical",
            "hierarchical_prototype",
            "calibrated_softmax",
            "softmax",
            "fused",
        )
        for value in accepted:
            with self.subTest(value=value), patch.object(
                sys, "argv", [*common, "--selection_method", value]
            ):
                args = parse_args()
                self.assertEqual(args.selection_method, value)

    def test_predict_and_eval_reject_corrupt_explicit_schema_blocks(self) -> None:
        from eval_hier_acoustic_prototype import (
            load_and_validate_prediction_metadata,
        )
        from predict_hier_acoustic_prototype import (
            validate_prediction_input_nomenclature,
        )

        bundle_metadata = build_model_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol="fixed_lambda_0_5_retrospective",
        )
        bundle_metadata["model_config"] = {"dur_s": 2.0}
        calibration = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b3_hierarchical_prototype_top1_ablation",
            context_seconds=2.0,
            selection_protocol="fixed_lambda_0_5_retrospective",
            inference_route="hierarchical_prototype_candidate",
            legacy_method_id="hierarchical",
        )
        validate_prediction_input_nomenclature(
            bundle_metadata, calibration
        )
        corrupt = {**calibration, "canonical_method_id": "corrupt"}
        with self.assertRaisesRegex(
            ValueError, "Invalid calibration nomenclature metadata"
        ):
            validate_prediction_input_nomenclature(
                bundle_metadata, corrupt
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "test_predictions.csv"
            cal = root / "calibration.json"
            metadata = root / "prediction_metadata.json"
            pred.write_text("path,y_true,y_pred\n", encoding="utf-8")
            cal.write_text(json.dumps(calibration), encoding="utf-8")
            metadata.write_text(
                json.dumps({**corrupt, "input_role": "frozen_test"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "Invalid prediction nomenclature metadata"
            ):
                load_and_validate_prediction_metadata(
                    pred,
                    cal,
                    calibration=calibration,
                )

        unselected_bundle = build_model_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol="none",
        )
        unselected_bundle["model_config"] = {"dur_s": 2.0}
        b1_calibration = build_method_metadata(
            model_family="logmel_crnn",
            training_stage="b1_2s_logmel_mainline",
            context_seconds=2.0,
            selection_protocol="none",
            inference_route="primary_softmax",
            legacy_method_id="raw_softmax",
        )
        b1_calibration["model_config"] = {"dur_s": 2.0}
        with self.assertRaisesRegex(
            ValueError, "Invalid calibration nomenclature metadata"
        ):
            validate_prediction_input_nomenclature(
                unselected_bundle, b1_calibration
            )

        b2_prediction = build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage="b2_validation_selected_hierarchical_crnn",
            context_seconds=2.0,
            selection_protocol="none",
            inference_route="primary_softmax",
            legacy_method_id="raw_softmax",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "test_predictions.csv"
            cal = root / "calibration.json"
            metadata = root / "prediction_metadata.json"
            pred.write_text("path,y_true,y_pred\n", encoding="utf-8")
            cal.write_text(json.dumps(b1_calibration), encoding="utf-8")
            metadata.write_text(
                json.dumps({**b2_prediction, "input_role": "frozen_test"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "Invalid calibration nomenclature metadata"
            ):
                load_and_validate_prediction_metadata(
                    pred,
                    cal,
                    calibration=b1_calibration,
                )

    def test_prediction_preflight_protects_leakage_audit(self) -> None:
        from predict_hier_acoustic_prototype import prepare_evaluation_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            (evaluation / "leakage_audit.json").write_text(
                "historical", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                FileExistsError, "leakage_audit.json"
            ):
                prepare_evaluation_dir(root, allow_overwrite=False)

    def test_calibrator_rejects_selection_protocol_relabelling(self) -> None:
        from calibrate_prototype_predictor import (
            selection_protocol_for_bundle,
        )

        metadata = {
            "selection_protocol": "fixed_lambda_0_5_retrospective"
        }
        self.assertEqual(
            selection_protocol_for_bundle(metadata, requested=""),
            "fixed_lambda_0_5_retrospective",
        )
        with self.assertRaisesRegex(ValueError, "Conflicting selection_protocol"):
            selection_protocol_for_bundle(
                metadata,
                requested="foldwise_validation_selected_lambda",
            )

    def test_prototype_writers_accept_explicit_selection_protocol(self) -> None:
        from build_hier_acoustic_prototypes import parse_args as parse_build
        from summarize_cv5_exact_prototype import parse_args as parse_summary

        build_argv = [
            "build_hier_acoustic_prototypes.py",
            "--train_manifest",
            "train.csv",
            "--val_manifest",
            "val.csv",
            "--test_manifest",
            "test.csv",
            "--ckpt",
            "model.pt",
            "--summary_json",
            "summary.json",
            "--out_dir",
            "new-output",
            "--fold",
            "0",
            "--seed",
            "42",
            "--expected_hier_aux_weight",
            "0.5",
            "--selection_protocol",
            "validation_selected",
        ]
        with warnings.catch_warnings(record=True), patch.object(
            sys, "argv", build_argv
        ):
            warnings.simplefilter("always")
            build_args = parse_build()
        self.assertEqual(
            build_args.selection_protocol,
            "foldwise_validation_selected_lambda",
        )

        summary_argv = [
            "summarize_cv5_exact_prototype.py",
            "--metrics_json",
            "metrics.json",
            "--expected_seeds",
            "42,123,777",
            "--selection_protocol",
            "fixed_lambda_0.5",
        ]
        with warnings.catch_warnings(record=True), patch.object(
            sys, "argv", summary_argv
        ):
            warnings.simplefilter("always")
            summary_args = parse_summary()
        self.assertEqual(
            summary_args.selection_protocol,
            "fixed_lambda_0_5_retrospective",
        )

    def test_builder_inherits_summary_protocol_and_rejects_relabelling(self) -> None:
        from build_hier_acoustic_prototypes import (
            selection_protocol_for_summary,
        )

        summary = {
            "selection_protocol": "fixed_lambda_0_5_retrospective"
        }
        self.assertEqual(
            selection_protocol_for_summary(summary, requested=None),
            "fixed_lambda_0_5_retrospective",
        )
        with self.assertRaisesRegex(ValueError, "Conflicting selection_protocol"):
            selection_protocol_for_summary(
                summary,
                requested="foldwise_validation_selected_lambda",
            )
        self.assertEqual(
            selection_protocol_for_summary({}, requested=None), "none"
        )

    def test_optional_selection_protocol_defaults_preserve_old_commands(self) -> None:
        from build_hier_acoustic_prototypes import parse_args as parse_build
        from calibrate_prototype_predictor import parse_args as parse_calibrate
        from summarize_cv5_exact_prototype import parse_args as parse_summary

        with patch.object(
            sys,
            "argv",
            [
                "build_hier_acoustic_prototypes.py",
                "--train_manifest",
                "train.csv",
                "--val_manifest",
                "val.csv",
                "--test_manifest",
                "test.csv",
                "--ckpt",
                "model.pt",
                "--summary_json",
                "summary.json",
                "--out_dir",
                "new-output",
                "--fold",
                "0",
                "--seed",
                "42",
                "--expected_hier_aux_weight",
                "0.5",
            ],
        ):
            self.assertIsNone(parse_build().selection_protocol)
        with patch.object(
            sys,
            "argv",
            [
                "calibrate_prototype_predictor.py",
                "--prototype_bundle",
                "bundle.npz",
                "--val_manifest",
                "val.csv",
                "--ckpt",
                "model.pt",
                "--out_dir",
                "new-output",
            ],
        ):
            self.assertIsNone(parse_calibrate().selection_protocol)
        with patch.object(
            sys,
            "argv",
            [
                "summarize_cv5_exact_prototype.py",
                "--metrics_json",
                "metrics.json",
                "--expected_seeds",
                "42,123,777",
            ],
        ):
            self.assertIsNone(parse_summary().selection_protocol)

    def test_legacy_machine_ids_remain_unchanged(self) -> None:
        from eval_hier_acoustic_prototype import METHODS
        from summarize_cv5_exact_prototype import (
            CONFUSION_METHODS,
            SUMMARY_METHODS,
        )

        self.assertEqual(
            METHODS,
            [
                "raw_softmax",
                "calibrated_softmax",
                "prototype",
                "hierarchical",
                "fused",
            ],
        )
        self.assertEqual(
            SUMMARY_METHODS,
            (
                "raw_softmax",
                "calibrated_softmax",
                "prototype",
                "hierarchical",
                "fused",
            ),
        )
        self.assertEqual(
            CONFUSION_METHODS,
            ("raw_softmax", "prototype", "hierarchical"),
        )


class ActiveAnalysisScriptNomenclatureTests(unittest.TestCase):
    def test_final_validation_readers_accept_legacy_and_canonical_routes(self) -> None:
        from tools.generate_final_validation_audit import (
            _canonical_route_from_record,
            _method_payload_for_route,
            _recorded_selection_protocol,
        )

        self.assertEqual(
            _canonical_route_from_record(
                {"selection_method": "hierarchical"}, context="legacy"
            ),
            "hierarchical_prototype_candidate",
        )
        self.assertEqual(
            _canonical_route_from_record(
                {
                    "inference_route": (
                        "hierarchical_prototype_candidate"
                    )
                },
                context="canonical",
            ),
            "hierarchical_prototype_candidate",
        )
        payload = _method_payload_for_route(
            {
                "primary_softmax": {"macro_f1": 0.9},
                "hierarchical": {"macro_f1": 0.91},
                "fused": {"macro_f1": 0.92},
            },
            route="primary_softmax",
            context="metrics.methods",
        )
        self.assertEqual(payload["macro_f1"], 0.9)
        self.assertEqual(
            _recorded_selection_protocol(
                (
                    ("bundle", {"selection_protocol": "none"}),
                    (
                        "calibration",
                        {"selection_protocol": "fixed_lambda_0.5"},
                    ),
                )
            ),
            "fixed_lambda_0_5_retrospective",
        )

    def test_final_validation_readers_reject_unknown_or_duplicate_routes(self) -> None:
        from tools.generate_final_validation_audit import (
            _canonical_route_from_record,
            _method_payload_for_route,
            _recorded_selection_protocol,
        )

        with self.assertRaisesRegex(ValueError, "Unknown inference method"):
            _canonical_route_from_record(
                {"selection_method": "mystery_route"}, context="metrics"
            )
        with self.assertRaisesRegex(ValueError, "duplicate aliases"):
            _method_payload_for_route(
                {
                    "raw_softmax": {"macro_f1": 0.9},
                    "primary_softmax": {"macro_f1": 0.9},
                },
                route="primary_softmax",
                context="metrics.methods",
            )
        with self.assertRaisesRegex(ValueError, "Conflicting selection_protocol"):
            _recorded_selection_protocol(
                (
                    (
                        "calibration",
                        {
                            "selection_protocol": (
                                "fixed_lambda_0_5_retrospective"
                            )
                        },
                    ),
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

    def test_framework_stage_metadata_distinguishes_protocols_and_stages(self) -> None:
        from tools.generate_final_validation_audit import (
            canonical_metadata_for_framework_stage,
        )

        expected = {
            "B0": (
                "b0_1s_logmel_baseline",
                "none",
                "primary_softmax",
                1.0,
            ),
            "B1": (
                "b1_2s_logmel_mainline",
                "none",
                "primary_softmax",
                2.0,
            ),
            "B2_valsel": (
                "b2_validation_selected_hierarchical_crnn",
                "foldwise_validation_selected_lambda",
                "primary_softmax",
                2.0,
            ),
            "B3_valsel": (
                "b3_hierarchical_prototype_top1_ablation",
                "foldwise_validation_selected_lambda",
                "hierarchical_prototype_candidate",
                2.0,
            ),
            "B2_fixed_w05": (
                "b2_validation_selected_hierarchical_crnn",
                "fixed_lambda_0_5_retrospective",
                "primary_softmax",
                2.0,
            ),
            "B3_fixed_w05": (
                "b3_hierarchical_prototype_top1_ablation",
                "fixed_lambda_0_5_retrospective",
                "hierarchical_prototype_candidate",
                2.0,
            ),
        }
        method_ids: dict[str, str] = {}
        for stage, values in expected.items():
            with self.subTest(stage=stage):
                metadata = canonical_metadata_for_framework_stage(stage)
                self.assertEqual(
                    (
                        metadata["training_stage"],
                        metadata["selection_protocol"],
                        metadata["inference_route"],
                        metadata["context_seconds"],
                    ),
                    values,
                )
                method_ids[stage] = str(metadata["canonical_method_id"])
        self.assertNotEqual(method_ids["B2_valsel"], method_ids["B2_fixed_w05"])
        self.assertNotEqual(method_ids["B3_valsel"], method_ids["B3_fixed_w05"])

    def test_method_frames_gain_metadata_without_replacing_legacy_columns(self) -> None:
        import pandas as pd

        from tools.generate_final_validation_audit import (
            add_canonical_method_metadata,
        )

        legacy = pd.DataFrame(
            {
                "method": [
                    "raw_softmax",
                    "main_prototype",
                    "hierarchical_prototype",
                ],
                "score": [0.95, 0.94, 0.93],
            }
        )
        enriched = add_canonical_method_metadata(
            legacy,
            selection_protocol="foldwise_validation_selected_lambda",
        )
        self.assertEqual(enriched["method"].tolist(), legacy["method"].tolist())
        self.assertEqual(enriched["score"].tolist(), legacy["score"].tolist())
        self.assertNotIn("inference_route", legacy.columns)
        self.assertEqual(
            enriched["inference_route"].tolist(),
            [
                "primary_softmax",
                "main_class_prototype_candidate",
                "hierarchical_prototype_candidate",
            ],
        )
        self.assertEqual(
            enriched["training_stage"].tolist(),
            [
                "b2_validation_selected_hierarchical_crnn",
                "b2_validation_selected_hierarchical_crnn",
                "b3_hierarchical_prototype_top1_ablation",
            ],
        )
        self.assertTrue(
            set(METHOD_METADATA_FIELDS).issubset(set(enriched.columns))
        )
        conflicting = legacy.copy()
        conflicting["selection_protocol"] = (
            "fixed_lambda_0_5_retrospective"
        )
        with self.assertRaisesRegex(ValueError, "Conflicting selection_protocol"):
            add_canonical_method_metadata(
                conflicting,
                selection_protocol="foldwise_validation_selected_lambda",
            )

    def test_stage_frames_gain_metadata_without_replacing_stage_keys(self) -> None:
        import pandas as pd

        from tools.generate_final_validation_audit import (
            add_canonical_stage_metadata,
        )

        legacy = pd.DataFrame(
            {"stage": ["B0", "B1", "B2", "B3"], "best_epoch": [1, 2, 3, 4]}
        )
        enriched = add_canonical_stage_metadata(
            legacy,
            selection_protocol="foldwise_validation_selected_lambda",
        )
        self.assertEqual(enriched["stage"].tolist(), legacy["stage"].tolist())
        self.assertEqual(
            enriched["training_stage"].tolist(),
            [
                "b0_1s_logmel_baseline",
                "b1_2s_logmel_mainline",
                "b2_validation_selected_hierarchical_crnn",
                "b3_hierarchical_prototype_top1_ablation",
            ],
        )
        self.assertEqual(
            enriched["selection_protocol"].tolist(),
            [
                "none",
                "none",
                "foldwise_validation_selected_lambda",
                "foldwise_validation_selected_lambda",
            ],
        )
        with self.assertRaisesRegex(ValueError, "requires explicit selection_protocol"):
            add_canonical_stage_metadata(legacy)

        conflicting_stage = legacy.copy()
        conflicting_stage["selection_protocol"] = [
            "none",
            "none",
            "fixed_lambda_0_5_retrospective",
            "fixed_lambda_0_5_retrospective",
        ]
        with self.assertRaisesRegex(ValueError, "selection_protocol"):
            add_canonical_stage_metadata(
                conflicting_stage,
                selection_protocol="foldwise_validation_selected_lambda",
            )

        fixed = add_canonical_stage_metadata(
            pd.DataFrame({"stage": ["B2", "B3"]}),
            selection_protocol="fixed_lambda_0_5_retrospective",
        )
        self.assertEqual(
            fixed["selection_protocol"].tolist(),
            [
                "fixed_lambda_0_5_retrospective",
                "fixed_lambda_0_5_retrospective",
            ],
        )

    def test_comparison_frames_keep_legacy_keys_and_both_protocol_identities(self) -> None:
        import pandas as pd

        from tools.generate_final_validation_audit import (
            add_canonical_comparison_metadata,
        )

        legacy = pd.DataFrame(
            {
                "comparison": ["B2_fixed_w05 - B2_valsel"],
                "final_stage": ["B2_fixed_w05"],
                "baseline_stage": ["B2_valsel"],
                "mean_delta": [0.0],
            }
        )
        enriched = add_canonical_comparison_metadata(legacy)
        self.assertEqual(enriched["comparison"].tolist(), legacy["comparison"].tolist())
        self.assertEqual(enriched["final_stage"].tolist(), legacy["final_stage"].tolist())
        self.assertEqual(
            enriched.loc[0, "selection_protocol"],
            "fixed_lambda_0_5_retrospective",
        )
        self.assertEqual(
            enriched.loc[0, "baseline_selection_protocol"],
            "foldwise_validation_selected_lambda",
        )
        self.assertNotEqual(
            enriched.loc[0, "canonical_method_id"],
            enriched.loc[0, "baseline_canonical_method_id"],
        )

    def test_active_final_validation_uses_canonical_display_labels(self) -> None:
        from tools.generate_final_validation_audit import (
            FRAMEWORK_STAGE_DESCRIPTIONS,
            METHOD_DISPLAY_LABELS,
            framework_comparison_display,
        )

        expected = {
            "raw_softmax": "Primary Softmax route (Raw Softmax)",
            "prototype": "Main-class prototype candidate route",
            "hierarchical": "Hierarchical prototype candidate route",
        }
        self.assertEqual(METHOD_DISPLAY_LABELS["raw_softmax"], expected["raw_softmax"])
        self.assertEqual(
            METHOD_DISPLAY_LABELS["main_prototype"], expected["prototype"]
        )
        self.assertEqual(
            METHOD_DISPLAY_LABELS["hierarchical_prototype"],
            expected["hierarchical"],
        )
        self.assertNotIn(
            "Validation-selected",
            FRAMEWORK_STAGE_DESCRIPTIONS["B2_fixed_w05"],
        )
        comparison_display = framework_comparison_display(
            "B3_fixed_w05 - B3_valsel"
        )
        self.assertIn(
            DISPLAY_LABELS_EN["training_stage"][
                "b3_hierarchical_prototype_top1_ablation"
            ],
            comparison_display,
        )
        self.assertIn("Fixed λ=0.5 retrospective", comparison_display)
        self.assertIn("Fold-wise validation-selected λ", comparison_display)


if __name__ == "__main__":
    unittest.main()
