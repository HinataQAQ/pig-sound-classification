"""Generate the locked final validation audit from frozen experiment artifacts.

This module never trains a model.  It selects hierarchical weights from
validation summaries only, validates exact fold-seed pairing, and computes
deterministic paired statistics for the final audit.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon


EXPECTED_FOLDS = tuple(range(5))
EXPECTED_SEEDS = (42, 123, 777, 2024, 3407)
EXPECTED_KEYS = tuple(
    (fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS
)
CANDIDATE_LAMBDAS = (0.2, 0.5, 1.0)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 3407
MAIN_LABELS = ("cough", "calm_grunt", "feeding", "stress_vocal")
AUX_LABELS = (
    "dry_cough",
    "abdominal_cough",
    "calm_grunt",
    "feeding",
    "frightened_stress",
    "anxious_stress",
)
AUX_TO_MAIN = {
    "dry_cough": "cough",
    "abdominal_cough": "cough",
    "calm_grunt": "calm_grunt",
    "feeding": "feeding",
    "frightened_stress": "stress_vocal",
    "anxious_stress": "stress_vocal",
}

TABLE_FILENAMES = (
    "lambda_selection_by_fold.csv",
    "validation_selected_framework_runs.csv",
    "validation_selected_framework_summary.csv",
    "validation_selected_framework_paired_stats.csv",
    "validation_selected_framework_fold_stats.csv",
    "prototype_functional_value_summary.csv",
    "prototype_margin_error_detection.csv",
    "prototype_disagreement_analysis.csv",
    "convergence_best_epoch_summary.csv",
    "validation_test_gap_summary.csv",
)
PROVENANCE_FILENAME = "lambda_selection_provenance.json"
REPORT_FILENAME = "FINAL_VALIDATION_REPORT.md"
FIGURE_STEMS = (
    "lambda_selection_by_fold",
    "validation_selected_cumulative_framework",
    "prototype_functional_value",
    "prototype_margin_correct_vs_error",
    "best_epoch_and_val_test_gap",
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


def required_output_paths(output_dir: Path) -> tuple[Path, ...]:
    paths = [output_dir / filename for filename in TABLE_FILENAMES]
    paths.extend((output_dir / PROVENANCE_FILENAME, output_dir / REPORT_FILENAME))
    for stem in FIGURE_STEMS:
        paths.extend(
            output_dir / "figures" / f"{stem}.{extension}"
            for extension in ("svg", "pdf", "png")
        )
    return tuple(paths)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON source is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON source {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source must contain an object: {path}")
    return payload


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def lambda_tag(candidate: float) -> str:
    mapping = {0.2: "w02", 0.5: "w05", 1.0: "w10"}
    for value, tag in mapping.items():
        if math.isclose(float(candidate), value, abs_tol=1e-12):
            return tag
    raise ValueError(f"Unsupported lambda: {candidate}")


def summary_path(
    root: Path,
    stage: str,
    fold: int,
    seed: int,
    candidate: float | None = None,
) -> Path:
    if stage == "B0":
        name = f"cv5_expanded_cap3x_fold{fold}_logmel_seed{seed}"
    elif stage == "B1":
        name = f"cv5_expanded_cap3x_fold{fold}_logmel_dur2_seed{seed}"
    elif stage == "hierarchical":
        if candidate is None:
            raise ValueError("Hierarchical summary path requires a lambda")
        name = (
            f"cv5_expanded_cap3x_fold{fold}_logmel_dur2_hier_"
            f"{lambda_tag(candidate)}_seed{seed}"
        )
    else:
        raise ValueError(f"Unsupported summary stage: {stage}")
    return root / "reports" / name / "summary.json"


def raw_prediction_path(
    root: Path, fold: int, seed: int, candidate: float
) -> Path:
    return summary_path(root, "hierarchical", fold, seed, candidate).with_name(
        "test_pred.csv"
    )


def checkpoint_path(root: Path, fold: int, seed: int, candidate: float) -> Path:
    return (
        root
        / "checkpoints"
        / (
            f"cv5_expanded_cap3x_fold{fold}_logmel_dur2_hier_"
            f"{lambda_tag(candidate)}_seed{seed}.pt"
        )
    )


def manifest_path(root: Path, fold: int, split: str) -> Path:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported manifest split: {split}")
    return (
        root
        / "paper_results"
        / "manifests"
        / "manifests_pigvocal_4class_expanded_train_cv5_cap3x"
        / f"fold{fold}"
        / f"{split}.csv"
    )


def prototype_run_dir(
    root: Path, fold: int, seed: int, candidate: float
) -> Path:
    tag = lambda_tag(candidate)
    if tag == "w05":
        name = f"prototype_cv5_exact_w05_fold{fold}_seed{seed}"
    else:
        name = f"prototype_cv5_exact_valsel_{tag}_fold{fold}_seed{seed}"
    return root / "reports" / name


def _validate_hierarchical_summary(
    payload: Mapping[str, Any],
    *,
    path: Path,
    seed: int,
    candidate: float,
) -> None:
    checks = {
        "model_family=hier_longcontext_crnn": payload.get("model_family")
        == "hier_longcontext_crnn",
        "feature_mode=logmel": payload.get("feature_mode") == "logmel",
        "use_se=true": payload.get("use_se") is True,
        f"seed={seed}": int(payload.get("seed", -1)) == seed,
        "dur_s=2.0": math.isclose(
            float(payload.get("dur_s", -1)), 2.0, abs_tol=1e-12
        ),
        "hier_aux=true": payload.get("hier_aux") is True,
        f"hier_aux_weight={candidate}": math.isclose(
            float(payload.get("hier_aux_weight", -1)), candidate, abs_tol=1e-12
        ),
        "rnn_type=gru": payload.get("rnn_type") == "gru",
        "pooling_type=mean": payload.get("pooling_type") == "mean",
        "main_labels": tuple(payload.get("main_labels", ())) == MAIN_LABELS,
        "aux_labels": tuple(payload.get("aux_labels", ())) == AUX_LABELS,
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Hierarchical summary identity failed for {path}: {failed}")
    for field in ("best_val_macro_f1", "best_epoch", "test_macro_f1"):
        value = payload.get(field)
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"Missing/non-finite {field} in {path}")


def _validate_baseline_summary(
    payload: Mapping[str, Any],
    *,
    path: Path,
    stage: str,
    seed: int,
) -> None:
    expected_duration = 1.0 if stage == "B0" else 2.0
    checks = {
        "feature_mode=logmel": payload.get("feature_mode") == "logmel",
        "use_se=true": payload.get("use_se") is True,
        f"seed={seed}": int(payload.get("seed", -1)) == seed,
        f"dur_s={expected_duration}": math.isclose(
            float(payload.get("dur_s", -1)), expected_duration, abs_tol=1e-12
        ),
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Baseline summary identity failed for {path}: {failed}")
    for field in ("best_val_macro_f1", "best_epoch", "test_macro_f1"):
        value = payload.get(field)
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"Missing/non-finite {field} in {path}")


def load_lambda_selection_sources(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load and hash the fixed 75 validation summaries used for selection."""

    root = root.resolve()
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for fold in EXPECTED_FOLDS:
        for candidate in CANDIDATE_LAMBDAS:
            for seed in EXPECTED_SEEDS:
                path = summary_path(root, "hierarchical", fold, seed, candidate)
                payload = _read_json(path)
                _validate_hierarchical_summary(
                    payload,
                    path=path,
                    seed=seed,
                    candidate=candidate,
                )
                relative = _relative(root, path)
                digest = sha256_file(path)
                hashes[relative] = digest
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "lambda": candidate,
                        "best_val_macro_f1": float(payload["best_val_macro_f1"]),
                        "source_path": relative,
                        "sha256": digest,
                    }
                )
    frame = pd.DataFrame(rows)
    if len(frame) != 75:
        raise RuntimeError(f"Expected 75 lambda-selection rows, got {len(frame)}")
    return frame, hashes


def find_missing_selected_artifacts(
    root: Path, selected: Mapping[int, float]
) -> list[dict[str, object]]:
    """Return every missing selected input/output combination without substitution."""

    root = root.resolve()
    missing: list[dict[str, object]] = []
    for fold in EXPECTED_FOLDS:
        if fold not in selected:
            missing.append({"fold": fold, "seed": None, "lambda": None, "artifact": "selected_lambda"})
            continue
        candidate = float(selected[fold])
        for seed in EXPECTED_SEEDS:
            paths = {
                "checkpoint": checkpoint_path(root, fold, seed, candidate),
                "validation_summary": summary_path(
                    root, "hierarchical", fold, seed, candidate
                ),
                "reference_test_prediction": raw_prediction_path(
                    root, fold, seed, candidate
                ),
                "train_manifest": manifest_path(root, fold, "train"),
                "validation_manifest": manifest_path(root, fold, "val"),
                "test_manifest": manifest_path(root, fold, "test"),
                "prototype_metrics": prototype_run_dir(
                    root, fold, seed, candidate
                )
                / "evaluation"
                / "metrics.json",
            }
            exact_root = prototype_run_dir(root, fold, seed, candidate)
            paths.update(
                {
                    "softmax_reproduction": exact_root
                    / "softmax_reproduction"
                    / "softmax_reproduction.json",
                    "softmax_alignment_report": exact_root
                    / "softmax_reproduction"
                    / "sample_alignment_report.csv",
                    "prototype_metadata": exact_root
                    / "artifacts"
                    / "prototype_metadata.json",
                    "prototype_bundle": exact_root
                    / "artifacts"
                    / "prototype_bundle.npz",
                    "train_main_prototype_samples": exact_root
                    / "artifacts"
                    / "train_main_prototype_samples.csv",
                    "train_aux_prototype_samples": exact_root
                    / "artifacts"
                    / "train_aux_prototype_samples.csv",
                    "prototype_calibration": exact_root
                    / "calibration"
                    / "calibration.json",
                    "prototype_validation_predictions": exact_root
                    / "calibration"
                    / "val_predictions.csv",
                    "prototype_prediction_metadata": exact_root
                    / "evaluation"
                    / "prediction_metadata.json",
                    "prototype_test_predictions": exact_root
                    / "evaluation"
                    / "test_predictions.csv",
                    "prototype_test_predictions_json": exact_root
                    / "evaluation"
                    / "test_predictions.json",
                    "prototype_artifact_leakage": exact_root
                    / "artifacts"
                    / "leakage_audit.json",
                    "prototype_calibration_leakage": exact_root
                    / "calibration"
                    / "leakage_audit.json",
                    "prototype_evaluation_leakage": exact_root
                    / "evaluation"
                    / "leakage_audit.json",
                }
            )
            for artifact, path in paths.items():
                if not path.is_file():
                    missing.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "lambda": candidate,
                            "artifact": artifact,
                            "path": _relative(root, path),
                        }
                    )
    return missing


def _validate_leakage_audit(path: Path) -> None:
    payload = _read_json(path)
    checks = {
        "ok=true": payload.get("ok") is True,
        "no overlaps": payload.get("overlaps") in ([], None),
        "no invalid values": payload.get("invalid_values") in ([], None),
        "path/source_id/md5": set(payload.get("columns_checked", ()))
        == {"path", "source_id", "md5"},
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Leakage audit failed for {path}: {failed}")


def _path_matches(root: Path, recorded: Any, expected: Path) -> bool:
    if recorded is None:
        return False
    recorded_path = Path(str(recorded))
    if not recorded_path.is_absolute():
        recorded_path = root / recorded_path
    return recorded_path.resolve() == expected.resolve()


def validate_prototype_run(
    root: Path,
    run_dir: Path,
    *,
    fold: int,
    seed: int,
    expected_lambda: float,
    summary_source: Path,
) -> dict[str, Any]:
    """Validate one exact selected-lambda prototype result and return its data."""

    root = root.resolve()
    run_dir = run_dir.resolve()
    paths = {
        "reproduction": run_dir / "softmax_reproduction" / "softmax_reproduction.json",
        "alignment_report": run_dir / "softmax_reproduction" / "sample_alignment_report.csv",
        "metadata": run_dir / "artifacts" / "prototype_metadata.json",
        "prototype_bundle": run_dir / "artifacts" / "prototype_bundle.npz",
        "train_main_samples": run_dir
        / "artifacts"
        / "train_main_prototype_samples.csv",
        "train_aux_samples": run_dir
        / "artifacts"
        / "train_aux_prototype_samples.csv",
        "calibration": run_dir / "calibration" / "calibration.json",
        "validation_predictions": run_dir / "calibration" / "val_predictions.csv",
        "prediction_metadata": run_dir / "evaluation" / "prediction_metadata.json",
        "metrics": run_dir / "evaluation" / "metrics.json",
        "predictions": run_dir / "evaluation" / "test_predictions.csv",
        "prediction_json": run_dir / "evaluation" / "test_predictions.json",
        "artifact_leakage": run_dir / "artifacts" / "leakage_audit.json",
        "calibration_leakage": run_dir / "calibration" / "leakage_audit.json",
        "evaluation_leakage": run_dir / "evaluation" / "leakage_audit.json",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing exact prototype artifacts for fold={fold}, seed={seed}, "
            f"lambda={expected_lambda}: {missing}"
        )

    reproduction = _read_json(paths["reproduction"])
    metadata = _read_json(paths["metadata"])
    calibration = _read_json(paths["calibration"])
    prediction_metadata = _read_json(paths["prediction_metadata"])
    metrics = _read_json(paths["metrics"])
    checkpoint = checkpoint_path(root, fold, seed, expected_lambda)
    reference_prediction = raw_prediction_path(
        root, fold, seed, expected_lambda
    ).resolve()
    manifests = {split: manifest_path(root, fold, split) for split in ("train", "val", "test")}

    reproduction_checks = {
        "artifact_type": reproduction.get("artifact_type") == "hier_exact_softmax_reproduction",
        "passed": reproduction.get("softmax_reproduction_passed") is True,
        "prediction_all_match": reproduction.get("prediction_all_match") is True,
        "training_exact": reproduction.get("feature_backend") == "training_exact",
        "eligible": reproduction.get("eligible_for_cv_aggregation") is True,
        "reference_prediction": _path_matches(
            root,
            reproduction.get("paths", {}).get("reference_pred_csv"),
            reference_prediction,
        ),
    }
    metadata_model = metadata.get("model_config", {})
    metadata_paths = metadata.get("project_relative_paths", {})
    metadata_checks = {
        "artifact_type": metadata.get("artifact_type") == "hier_acoustic_prototype_bundle",
        "fold": int(metadata.get("fold", -1)) == fold,
        "seed": int(metadata.get("seed", -1)) == seed,
        "training_exact": metadata.get("feature_backend") == "training_exact",
        "eligible": metadata.get("eligible_for_cv_aggregation") is True,
        "lambda": math.isclose(
            float(metadata_model.get("hier_aux_weight", -1)),
            expected_lambda,
            abs_tol=1e-12,
        ),
        "checkpoint": _path_matches(root, metadata_paths.get("checkpoint"), checkpoint),
        "summary": _path_matches(
            root, metadata_paths.get("checkpoint_summary"), summary_source
        ),
    }
    for split, manifest in manifests.items():
        metadata_checks[f"{split}_manifest"] = _path_matches(
            root, metadata_paths.get(f"{split}_manifest"), manifest
        )
    calibration_checks = {
        "artifact_type": calibration.get("artifact_type") == "hier_acoustic_prototype_calibration",
        "fold": int(calibration.get("fold", -1)) == fold,
        "seed": int(calibration.get("seed", -1)) == seed,
        "selection_method": calibration.get("selection_method") == "hierarchical",
        "training_exact": calibration.get("feature_backend") == "training_exact",
        "eligible": calibration.get("eligible_for_cv_aggregation") is True,
        "test_not_used": calibration.get("test_used_for_calibration") is False,
        "lambda_fixed": calibration.get("hier_aux_prob_weight_mode") == "fixed",
        "lambda": math.isclose(
            float(calibration.get("hier_aux_prob_weight", -1)),
            expected_lambda,
            abs_tol=1e-12,
        ),
        "prototype_bundle": _path_matches(
            root,
            calibration.get("project_relative_paths", {}).get("prototype_bundle"),
            paths["prototype_bundle"],
        ),
        "checkpoint": _path_matches(
            root,
            calibration.get("project_relative_paths", {}).get("checkpoint"),
            checkpoint,
        ),
        "train_manifest": _path_matches(
            root,
            calibration.get("project_relative_paths", {}).get("train_manifest"),
            manifests["train"],
        ),
        "val_manifest": _path_matches(
            root,
            calibration.get("project_relative_paths", {}).get("val_manifest"),
            manifests["val"],
        ),
    }
    prediction_paths = prediction_metadata.get("project_relative_paths", {})
    prediction_checks = {
        "artifact_type": prediction_metadata.get("artifact_type")
        == "hier_acoustic_prototype_frozen_predictions",
        "fold": int(prediction_metadata.get("fold", -1)) == fold,
        "seed": int(prediction_metadata.get("seed", -1)) == seed,
        "selection_method": prediction_metadata.get("selection_method") == "hierarchical",
        "input_role": prediction_metadata.get("input_role") == "frozen_test",
        "training_exact": prediction_metadata.get("feature_backend") == "training_exact",
        "eligible": prediction_metadata.get("eligible_for_cv_aggregation") is True,
        "leakage": prediction_metadata.get("leakage_audit_ok") is True,
        "manifest_sha": prediction_metadata.get("manifest_sha_verified") is True,
        "test_not_selected": prediction_metadata.get("test_used_for_parameter_selection")
        is False,
        "prototype_bundle": _path_matches(
            root, prediction_paths.get("prototype_bundle"), paths["prototype_bundle"]
        ),
        "calibration": _path_matches(
            root, prediction_paths.get("calibration_json"), paths["calibration"]
        ),
        "checkpoint": _path_matches(
            root, prediction_paths.get("checkpoint"), checkpoint
        ),
    }
    for split, manifest in manifests.items():
        prediction_checks[f"{split}_manifest"] = _path_matches(
            root, prediction_paths.get(f"{split}_manifest"), manifest
        )
    best_hierarchical = metrics.get("method_best_params", {}).get("hierarchical", {})
    metrics_checks = {
        "artifact_type": metrics.get("artifact_type")
        == "hier_acoustic_prototype_test_evaluation",
        "fold": int(metrics.get("fold", -1)) == fold,
        "seed": int(metrics.get("seed", -1)) == seed,
        "selection_method": metrics.get("selection_method") == "hierarchical",
        "input_role": metrics.get("input_role") == "frozen_test",
        "training_exact": metrics.get("feature_backend") == "training_exact",
        "eligible": metrics.get("eligible_for_cv_aggregation") is True,
        "leakage": metrics.get("leakage_audit_ok") is True,
        "manifest_sha": metrics.get("manifest_sha_verified") is True,
        "test_not_selected": metrics.get("test_used_for_parameter_selection") is False,
        "lambda": math.isclose(
            float(best_hierarchical.get("hier_aux_prob_weight", -1)),
            expected_lambda,
            abs_tol=1e-12,
        ),
        "main_labels": tuple(metrics.get("labels", ())) == MAIN_LABELS,
        "aux_labels": tuple(metrics.get("aux_labels", ())) == AUX_LABELS,
        "prediction_csv": _path_matches(
            root, metrics.get("prediction_csv"), paths["predictions"]
        ),
        "calibration_json": _path_matches(
            root, metrics.get("calibration_json"), paths["calibration"]
        ),
        "prediction_metadata_json": _path_matches(
            root,
            metrics.get("prediction_metadata_json"),
            paths["prediction_metadata"],
        ),
    }
    for label, checks, path in (
        ("reproduction", reproduction_checks, paths["reproduction"]),
        ("prototype metadata", metadata_checks, paths["metadata"]),
        ("calibration", calibration_checks, paths["calibration"]),
        ("prediction metadata", prediction_checks, paths["prediction_metadata"]),
        ("evaluation metrics", metrics_checks, paths["metrics"]),
    ):
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{label} validation failed for {path}: {failed}")
    for key in ("artifact_leakage", "calibration_leakage", "evaluation_leakage"):
        _validate_leakage_audit(paths[key])

    metadata_hashes = metadata.get("manifest_sha256", {})
    metadata_path_records = metadata.get("paths", {})
    calibration_path_records = calibration.get("paths", {})
    prediction_path_records = prediction_metadata.get("paths", {})
    hash_links: list[tuple[Path, Any, str]] = [
        (checkpoint, metadata.get("checkpoint_sha256"), "prototype metadata checkpoint"),
        (
            summary_source,
            metadata.get("checkpoint_summary_sha256"),
            "prototype metadata checkpoint summary",
        ),
        (
            paths["prototype_bundle"],
            calibration.get("prototype_bundle_sha256"),
            "calibration prototype bundle",
        ),
        (checkpoint, calibration.get("checkpoint_sha256"), "calibration checkpoint"),
        (
            manifests["train"],
            calibration.get("train_manifest_sha256"),
            "calibration train manifest",
        ),
        (
            manifests["val"],
            calibration.get("val_manifest_sha256"),
            "calibration validation manifest",
        ),
        (
            paths["prototype_bundle"],
            prediction_metadata.get("prototype_bundle_sha256"),
            "prediction prototype bundle",
        ),
        (
            paths["calibration"],
            prediction_metadata.get("calibration_json_sha256"),
            "prediction calibration JSON",
        ),
        (checkpoint, prediction_metadata.get("checkpoint_sha256"), "prediction checkpoint"),
        (
            paths["predictions"],
            prediction_metadata.get("prediction_csv_sha256"),
            "prediction CSV",
        ),
        (
            paths["prediction_json"],
            prediction_metadata.get("prediction_json_sha256"),
            "prediction JSON",
        ),
        (
            paths["predictions"],
            metrics.get("prediction_csv_sha256"),
            "evaluation prediction CSV",
        ),
        (
            paths["calibration"],
            metrics.get("calibration_json_sha256"),
            "evaluation calibration JSON",
        ),
    ]
    reproduction_paths = reproduction.get("paths", {})
    hash_links.append(
        (
            checkpoint,
            reproduction_paths.get("checkpoint_sha256"),
            "Softmax reproduction checkpoint",
        )
    )
    for split, manifest in manifests.items():
        hash_links.append(
            (
                manifest,
                metadata_hashes.get(split),
                f"prototype metadata {split} manifest",
            )
        )
    for key, actual_path in (
        ("checkpoint", checkpoint),
        ("checkpoint_summary", summary_source),
        ("train_manifest", manifests["train"]),
        ("val_manifest", manifests["val"]),
        ("test_manifest", manifests["test"]),
    ):
        hash_links.append(
            (
                actual_path,
                metadata_path_records.get(key, {}).get("sha256"),
                f"prototype metadata paths.{key}",
            )
        )
    for key, actual_path in (
        ("prototype_bundle", paths["prototype_bundle"]),
        ("checkpoint", checkpoint),
        ("train_manifest", manifests["train"]),
        ("val_manifest", manifests["val"]),
    ):
        hash_links.append(
            (
                actual_path,
                calibration_path_records.get(key, {}).get("sha256"),
                f"calibration paths.{key}",
            )
        )
    for key, actual_path in (
        ("prototype_bundle", paths["prototype_bundle"]),
        ("calibration_json", paths["calibration"]),
        ("checkpoint", checkpoint),
        ("train_manifest", manifests["train"]),
        ("val_manifest", manifests["val"]),
        ("test_manifest", manifests["test"]),
    ):
        hash_links.append(
            (
                actual_path,
                prediction_path_records.get(key, {}).get("sha256"),
                f"prediction metadata paths.{key}",
            )
        )
    hash_links_verified = verify_recorded_sha256_links(hash_links)

    summary_payload = _read_json(summary_source)
    raw_macro = float(metrics["methods"]["raw_softmax"]["macro_f1"])
    hierarchical_macro = float(metrics["methods"]["hierarchical"]["macro_f1"])
    expected_raw = float(summary_payload["test_macro_f1"])
    if raw_macro != expected_raw:
        raise ValueError(
            "Exact prototype Raw Softmax does not reproduce its matched summary "
            f"at fold={fold}, seed={seed}, lambda={expected_lambda}: "
            f"{raw_macro} != {expected_raw}"
        )

    frame = pd.read_csv(paths["predictions"])
    required_prediction_columns = {
        "path",
        "source_id",
        "md5",
        "label",
        "subtype",
        "y_true",
        "aux_true",
        "feature_backend",
        "input_role",
        "eligible_for_cv_aggregation",
        *{f"raw_softmax_prob_{label}" for label in MAIN_LABELS},
        *{f"prototype_prob_{label}" for label in MAIN_LABELS},
        *{f"hierarchical_prob_{label}" for label in MAIN_LABELS},
        *{f"softmax_aux_prob_{label}" for label in AUX_LABELS},
        *{f"prototype_aux_prob_{label}" for label in AUX_LABELS},
        *{f"main_proto_cosine_distance_{label}" for label in MAIN_LABELS},
        *{f"aux_proto_cosine_distance_{label}" for label in AUX_LABELS},
    }
    missing_columns = sorted(required_prediction_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Prototype prediction table {paths['predictions']} is missing columns: "
            f"{missing_columns}"
        )
    if not (frame["feature_backend"].astype(str) == "training_exact").all():
        raise ValueError(f"Non-training-exact rows in {paths['predictions']}")
    if not (frame["input_role"].astype(str) == "frozen_test").all():
        raise ValueError(f"Non-frozen-test rows in {paths['predictions']}")
    if not frame["eligible_for_cv_aggregation"].astype(bool).all():
        raise ValueError(f"Ineligible rows in {paths['predictions']}")
    if frame["source_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate source IDs in {paths['predictions']}")

    train_manifest = pd.read_csv(manifests["train"])
    validation_manifest = pd.read_csv(manifests["val"])
    test_manifest = pd.read_csv(manifests["test"])
    train_main_membership_n = validate_frozen_split_membership(
        pd.read_csv(paths["train_main_samples"]),
        train_manifest,
        context=f"train main prototypes fold={fold}, seed={seed}",
    )
    train_aux_membership_n = validate_frozen_split_membership(
        pd.read_csv(paths["train_aux_samples"]),
        train_manifest,
        context=f"train subtype prototypes fold={fold}, seed={seed}",
    )
    validation_membership_n = validate_frozen_split_membership(
        pd.read_csv(paths["validation_predictions"]),
        validation_manifest,
        context=f"validation calibration fold={fold}, seed={seed}",
        y_true_column="y_true",
        aux_true_column="aux_true",
    )
    test_membership_n = validate_frozen_split_membership(
        frame,
        test_manifest,
        context=f"frozen test fold={fold}, seed={seed}",
        y_true_column="y_true",
        aux_true_column="aux_true",
    )

    alignment_report = pd.read_csv(paths["alignment_report"])
    validate_softmax_alignment_report(
        alignment_report,
        test_manifest,
        context=f"fold={fold}, seed={seed}, lambda={expected_lambda}",
    )
    frame, canonical_alignment = align_canonical_raw_probabilities(
        frame,
        pd.read_csv(reference_prediction),
        test_manifest,
        context=f"fold={fold}, seed={seed}, lambda={expected_lambda}",
    )
    rank_order_mismatch_count = int(
        canonical_alignment["rank_order_mismatch_count"]
    )
    recomputed_probability_drift = float(
        canonical_alignment["maximum_probability_abs_drift"]
    )
    recorded_probability_drift = float(
        reproduction.get("probability_max_abs_diff", float("nan"))
    )
    if not np.isclose(
        recomputed_probability_drift,
        recorded_probability_drift,
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError(
            "Recorded Softmax probability drift does not match the verified "
            f"identity-aligned recomputation: {recorded_probability_drift} != "
            f"{recomputed_probability_drift} (fold={fold}, seed={seed})"
        )
    probability_tolerance = float(reproduction.get("probability_tolerance", 1e-5))
    probability_within_tolerance = recomputed_probability_drift <= probability_tolerance
    if probability_within_tolerance != bool(
        reproduction.get("probability_within_tolerance", False)
    ):
        raise ValueError(
            f"Recorded Softmax probability tolerance status is stale: fold={fold}, "
            f"seed={seed}, lambda={expected_lambda}"
        )

    source_paths = tuple(
        sorted(
            {
                *(path.resolve() for path in paths.values()),
                checkpoint.resolve(),
                summary_source.resolve(),
                reference_prediction,
                *(path.resolve() for path in manifests.values()),
            },
            key=str,
        )
    )
    return {
        "raw_macro_f1": raw_macro,
        "hierarchical_macro_f1": hierarchical_macro,
        "predictions": frame,
        "predictions_path": paths["predictions"],
        "metrics_path": paths["metrics"],
        "source_paths": source_paths,
        "softmax_probability_max_abs_diff": recomputed_probability_drift,
        "softmax_probability_within_advisory_tolerance": probability_within_tolerance,
        "softmax_rank_order_mismatch_count_before_canonical_restore": rank_order_mismatch_count,
        "raw_softmax_probability_source": _relative(root, reference_prediction),
        "canonical_probability_alignment": canonical_alignment,
        "artifact_hash_links_verified": hash_links_verified,
        "train_main_membership_n": train_main_membership_n,
        "train_aux_membership_n": train_aux_membership_n,
        "validation_membership_n": validation_membership_n,
        "test_membership_n": test_membership_n,
    }


def load_validation_selected_runs(
    root: Path,
    selected: Mapping[int, float],
) -> tuple[pd.DataFrame, dict[tuple[int, int], pd.DataFrame], tuple[Path, ...]]:
    """Load the exact 25-run selected and fixed-lambda framework cohorts."""

    root = root.resolve()
    missing = find_missing_selected_artifacts(root, selected)
    if missing:
        details = "; ".join(
            f"fold={item.get('fold')},seed={item.get('seed')},"
            f"lambda={item.get('lambda')},artifact={item.get('artifact')},"
            f"path={item.get('path', '')}"
            for item in missing
        )
        raise FileNotFoundError(f"Missing selected-lambda artifact combinations: {details}")

    rows: list[dict[str, object]] = []
    predictions: dict[tuple[int, int], pd.DataFrame] = {}
    source_paths: set[Path] = set()
    prototype_cache: dict[tuple[int, int, float], dict[str, Any]] = {}
    for fold, seed in EXPECTED_KEYS:
        candidate = float(selected[fold])
        payloads: dict[str, dict[str, Any]] = {}
        paths = {
            "B0": summary_path(root, "B0", fold, seed),
            "B1": summary_path(root, "B1", fold, seed),
            "B2_valsel": summary_path(
                root, "hierarchical", fold, seed, candidate
            ),
            "B2_fixed_w05": summary_path(
                root, "hierarchical", fold, seed, 0.5
            ),
        }
        for stage, path in paths.items():
            payloads[stage] = _read_json(path)
            source_paths.add(path.resolve())
        _validate_baseline_summary(
            payloads["B0"], path=paths["B0"], stage="B0", seed=seed
        )
        _validate_baseline_summary(
            payloads["B1"], path=paths["B1"], stage="B1", seed=seed
        )
        _validate_hierarchical_summary(
            payloads["B2_valsel"],
            path=paths["B2_valsel"],
            seed=seed,
            candidate=candidate,
        )
        _validate_hierarchical_summary(
            payloads["B2_fixed_w05"],
            path=paths["B2_fixed_w05"],
            seed=seed,
            candidate=0.5,
        )

        def get_prototype(weight: float, summary: Path) -> dict[str, Any]:
            cache_key = (fold, seed, weight)
            if cache_key not in prototype_cache:
                prototype_cache[cache_key] = validate_prototype_run(
                    root,
                    prototype_run_dir(root, fold, seed, weight),
                    fold=fold,
                    seed=seed,
                    expected_lambda=weight,
                    summary_source=summary,
                )
            return prototype_cache[cache_key]

        selected_prototype = get_prototype(candidate, paths["B2_valsel"])
        fixed_prototype = get_prototype(0.5, paths["B2_fixed_w05"])
        source_paths.update(selected_prototype["source_paths"])
        source_paths.update(fixed_prototype["source_paths"])
        predictions[(fold, seed)] = selected_prototype["predictions"]
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "selected_lambda": candidate,
                "B0": float(payloads["B0"]["test_macro_f1"]),
                "B1": float(payloads["B1"]["test_macro_f1"]),
                "B2_valsel": float(payloads["B2_valsel"]["test_macro_f1"]),
                "B3_valsel": float(selected_prototype["hierarchical_macro_f1"]),
                "B2_fixed_w05": float(
                    payloads["B2_fixed_w05"]["test_macro_f1"]
                ),
                "B3_fixed_w05": float(fixed_prototype["hierarchical_macro_f1"]),
                "B0_best_epoch": int(payloads["B0"]["best_epoch"]),
                "B1_best_epoch": int(payloads["B1"]["best_epoch"]),
                "B2_valsel_best_epoch": int(
                    payloads["B2_valsel"]["best_epoch"]
                ),
                "B3_valsel_best_epoch": int(
                    payloads["B2_valsel"]["best_epoch"]
                ),
                "B0_best_val_macro_f1": float(
                    payloads["B0"]["best_val_macro_f1"]
                ),
                "B1_best_val_macro_f1": float(
                    payloads["B1"]["best_val_macro_f1"]
                ),
                "B2_valsel_best_val_macro_f1": float(
                    payloads["B2_valsel"]["best_val_macro_f1"]
                ),
                "B3_valsel_best_val_macro_f1": float(
                    payloads["B2_valsel"]["best_val_macro_f1"]
                ),
                "B0_source": _relative(root, paths["B0"]),
                "B1_source": _relative(root, paths["B1"]),
                "B2_valsel_source": _relative(root, paths["B2_valsel"]),
                "B3_valsel_source": _relative(
                    root, selected_prototype["metrics_path"]
                ),
                "B2_fixed_w05_source": _relative(
                    root, paths["B2_fixed_w05"]
                ),
                "B3_fixed_w05_source": _relative(
                    root, fixed_prototype["metrics_path"]
                ),
                "B3_epoch_provenance": "inherited_from_selected_B2_checkpoint",
                "B3_valsel_softmax_probability_max_abs_diff": float(
                    selected_prototype["softmax_probability_max_abs_diff"]
                ),
                "B3_valsel_softmax_probability_within_advisory_tolerance": bool(
                    selected_prototype[
                        "softmax_probability_within_advisory_tolerance"
                    ]
                ),
                "B3_valsel_softmax_rank_order_mismatch_count_before_canonical_restore": int(
                    selected_prototype[
                        "softmax_rank_order_mismatch_count_before_canonical_restore"
                    ]
                ),
                "B3_valsel_raw_softmax_probability_source": selected_prototype[
                    "raw_softmax_probability_source"
                ],
                "B3_valsel_canonical_probability_identity_join": selected_prototype[
                    "canonical_probability_alignment"
                ]["identity_join"],
                "B3_valsel_canonical_probability_truth_mismatch_count": int(
                    selected_prototype["canonical_probability_alignment"][
                        "truth_mismatch_count"
                    ]
                ),
                "B3_valsel_canonical_probability_prediction_mismatch_count": int(
                    selected_prototype["canonical_probability_alignment"][
                        "prediction_mismatch_count"
                    ]
                ),
                "B3_valsel_artifact_hash_links_verified": int(
                    selected_prototype["artifact_hash_links_verified"]
                ),
                "B3_valsel_train_main_membership_n": int(
                    selected_prototype["train_main_membership_n"]
                ),
                "B3_valsel_train_aux_membership_n": int(
                    selected_prototype["train_aux_membership_n"]
                ),
                "B3_valsel_validation_membership_n": int(
                    selected_prototype["validation_membership_n"]
                ),
                "B3_valsel_test_membership_n": int(
                    selected_prototype["test_membership_n"]
                ),
                "test_used_for_lambda_selection": False,
                "test_used_for_prototype_calibration": False,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)
    actual_keys = tuple(map(tuple, frame[["fold", "seed"]].to_numpy()))
    if actual_keys != EXPECTED_KEYS:
        raise RuntimeError(f"Selected framework grid mismatch: {actual_keys}")
    return frame, predictions, tuple(sorted(source_paths, key=str))


FRAMEWORK_STAGE_DESCRIPTIONS = {
    "B0": "1-s Log-Mel CRNN",
    "B1": "2-s Log-Mel CRNN",
    "B2_valsel": "2-s Log-Mel CRNN + fold-wise validation-selected hierarchy, Raw Softmax",
    "B3_valsel": "B2_valsel + hierarchical prototype candidate inference",
    "B2_fixed_w05": "Fixed-lambda-0.5 B2, Raw Softmax",
    "B3_fixed_w05": "Fixed-lambda-0.5 B3 hierarchical prototype",
}
FRAMEWORK_COMPARISONS = (
    ("B2_valsel", "B1"),
    ("B2_valsel", "B0"),
    ("B2_fixed_w05", "B2_valsel"),
    ("B3_valsel", "B2_valsel"),
    ("B3_valsel", "B1"),
    ("B3_valsel", "B0"),
    ("B3_fixed_w05", "B3_valsel"),
)


def compute_framework_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage, description in FRAMEWORK_STAGE_DESCRIPTIONS.items():
        if stage not in runs:
            raise ValueError(f"Framework runs are missing stage column: {stage}")
        values = runs[stage].to_numpy(dtype=np.float64)
        if values.size != 25 or not np.isfinite(values).all():
            raise ValueError(f"Stage {stage} does not contain 25 finite values")
        rows.append(
            {
                "stage": stage,
                "stage_name": description,
                "n": int(values.size),
                "mean_macro_f1": float(values.mean()),
                "sd_macro_f1": float(values.std(ddof=1)),
                "min_macro_f1": float(values.min()),
                "max_macro_f1": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _fold_cluster_summary(
    frame: pd.DataFrame,
    *,
    value_column: str,
    n_boot: int,
    seed: int,
) -> tuple[dict[int, float], float, float]:
    fold_means: dict[int, float] = {}
    for fold in EXPECTED_FOLDS:
        values = frame.loc[frame["fold"] == fold, value_column].dropna().to_numpy(float)
        fold_means[fold] = float(values.mean()) if values.size else float("nan")
    finite = np.asarray([value for value in fold_means.values() if np.isfinite(value)])
    if finite.size != len(EXPECTED_FOLDS):
        return fold_means, float("nan"), float("nan")
    low, high = _bootstrap_mean_ci(finite, n_boot=n_boot, seed=seed)
    return fold_means, low, high


def aggregate_functional_metrics(
    run_metrics: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, metric), group in run_metrics.groupby(["method", "metric"], sort=False):
        values = group["value"].dropna().to_numpy(dtype=np.float64)
        fold_means, fold_low, fold_high = _fold_cluster_summary(
            group, value_column="value", n_boot=n_boot, seed=seed
        )
        row: dict[str, object] = {
            "method": method,
            "metric": metric,
            "n_runs_total": int(len(group)),
            "n_runs_available": int(values.size),
            "mean": float(values.mean()) if values.size else float("nan"),
            "sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            "min": float(values.min()) if values.size else float("nan"),
            "max": float(values.max()) if values.size else float("nan"),
            "fold_cluster_n": int(
                sum(np.isfinite(value) for value in fold_means.values())
            ),
            "fold_cluster_ci95_low": fold_low,
            "fold_cluster_ci95_high": fold_high,
            "aggregation_unit": "fold_seed_run",
        }
        row.update(
            {f"fold_{fold}_mean": fold_means[fold] for fold in EXPECTED_FOLDS}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_margin_aggregates(
    run_margins: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    output = [run_margins.copy()]
    metrics = (
        "correct_margin_median",
        "error_margin_median",
        "error_detection_auroc",
    )
    fold_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    for (method, level), group in run_margins.groupby(["method", "level"], sort=False):
        for fold in EXPECTED_FOLDS:
            subset = group[group["fold"] == fold]
            row: dict[str, object] = {
                "record_type": "fold_mean",
                "fold": fold,
                "seed": np.nan,
                "method": method,
                "level": level,
                "n": int(len(subset)),
            }
            for metric in metrics:
                row[metric] = float(subset[metric].mean(skipna=True))
            fold_rows.append(row)
        overall: dict[str, object] = {
            "record_type": "overall_run_mean",
            "fold": np.nan,
            "seed": np.nan,
            "method": method,
            "level": level,
            "n": int(len(group)),
        }
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            overall[metric] = float(values.mean()) if values.size else float("nan")
            fold_means = np.asarray(
                [
                    group.loc[group["fold"] == fold, metric].mean(skipna=True)
                    for fold in EXPECTED_FOLDS
                ],
                dtype=float,
            )
            if np.isfinite(fold_means).all():
                low, high = _bootstrap_mean_ci(
                    fold_means, n_boot=n_boot, seed=seed
                )
            else:
                low, high = float("nan"), float("nan")
            overall[f"{metric}_fold_cluster_ci95_low"] = low
            overall[f"{metric}_fold_cluster_ci95_high"] = high
        overall_rows.append(overall)
    output.append(pd.DataFrame(fold_rows))
    output.append(pd.DataFrame(overall_rows))
    return pd.concat(output, ignore_index=True, sort=False)


def add_disagreement_aggregates(
    run_disagreements: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    numeric = (
        "n_disagreements",
        "disagreement_rate",
        "raw_correct_prototype_wrong",
        "prototype_correct_raw_wrong",
        "both_wrong",
        "both_correct",
        "unchanged_correct",
        "unchanged_wrong",
    )
    fold_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    for method, group in run_disagreements.groupby("prototype_method", sort=False):
        for fold in EXPECTED_FOLDS:
            subset = group[group["fold"] == fold]
            row: dict[str, object] = {
                "record_type": "fold_mean",
                "fold": fold,
                "seed": np.nan,
                "prototype_method": method,
                "n": int(len(subset)),
            }
            row.update({column: float(subset[column].mean()) for column in numeric})
            fold_rows.append(row)
        overall: dict[str, object] = {
            "record_type": "overall_run_mean",
            "fold": np.nan,
            "seed": np.nan,
            "prototype_method": method,
            "n": int(len(group)),
            "repeated_run_decision_total": int(group["n"].sum()),
        }
        overall.update({column: float(group[column].mean()) for column in numeric})
        for column in numeric:
            fold_values = np.asarray(
                [
                    group.loc[group["fold"] == fold, column].mean()
                    for fold in EXPECTED_FOLDS
                ],
                dtype=float,
            )
            low, high = _bootstrap_mean_ci(
                fold_values, n_boot=n_boot, seed=seed
            )
            overall[f"{column}_fold_cluster_ci95_low"] = low
            overall[f"{column}_fold_cluster_ci95_high"] = high
        for column in (
            "n_disagreements",
            "raw_correct_prototype_wrong",
            "prototype_correct_raw_wrong",
            "both_wrong",
            "both_correct",
        ):
            overall[f"repeated_run_total_{column}"] = int(group[column].sum())
        overall_rows.append(overall)
    return pd.concat(
        [run_disagreements, pd.DataFrame(fold_rows), pd.DataFrame(overall_rows)],
        ignore_index=True,
        sort=False,
    )


def _history_field_names(path: Path) -> set[str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return {str(column) for column in pd.read_csv(path, nrows=5).columns}
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records: list[Mapping[str, Any]] = []
            if isinstance(payload, list):
                records.extend(item for item in payload if isinstance(item, Mapping))
            elif isinstance(payload, Mapping):
                records.append(payload)
                for value in payload.values():
                    if isinstance(value, list):
                        records.extend(
                            item for item in value if isinstance(item, Mapping)
                        )
            return {str(key) for record in records for key in record}
        if suffix == ".log":
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            fields = set()
            if "epoch" in text:
                fields.add("epoch")
            if "loss" in text and ("train" in text or "total loss" in text):
                fields.add("train_loss")
            if (
                ("val" in text or "validation" in text)
                and "macro" in text
                and ("f1" in text or "f-1" in text)
            ):
                fields.add("val_macro_f1")
            return fields
    except (OSError, ValueError, pd.errors.ParserError, json.JSONDecodeError):
        return set()
    return set()


def _has_complete_epoch_history(path: Path) -> bool:
    normalised = {
        "".join(character for character in field.lower() if character.isalnum())
        for field in _history_field_names(path)
    }
    has_epoch = any(field == "epoch" or field.endswith("epoch") for field in normalised)
    has_train_loss = any(
        "loss" in field and ("train" in field or "total" in field)
        for field in normalised
    )
    has_validation_macro_f1 = any(
        ("val" in field or "validation" in field)
        and "macro" in field
        and "f1" in field
        for field in normalised
    )
    return has_epoch and has_train_loss and has_validation_macro_f1


def audit_epoch_logs(root: Path, runs: pd.DataFrame) -> dict[str, Any]:
    """Search persisted run directories and repository logs for real histories."""

    root = root.resolve()
    stage_sources = {
        "B0": "B0_source",
        "B1": "B1_source",
        "B2": "B2_valsel_source",
    }
    tokens = ("history", "epoch", "training_log", "train_log", "curve")
    coverage: dict[str, int] = {}
    candidates_scanned: set[Path] = set()
    complete_files: set[Path] = set()
    complete_keys: dict[str, list[dict[str, int]]] = {}
    for stage, source_column in stage_sources.items():
        stage_complete: list[dict[str, int]] = []
        for run in runs.itertuples(index=False):
            source_value = Path(str(getattr(run, source_column)))
            summary = source_value if source_value.is_absolute() else root / source_value
            if not summary.is_file():
                continue
            run_candidates = [
                path
                for path in summary.parent.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".csv", ".json", ".log"}
                and any(token in path.name.lower() for token in tokens)
            ]
            candidates_scanned.update(path.resolve() for path in run_candidates)
            matching = [path for path in run_candidates if _has_complete_epoch_history(path)]
            if matching:
                complete_files.update(path.resolve() for path in matching)
                stage_complete.append(
                    {"fold": int(run.fold), "seed": int(run.seed)}
                )
        coverage[stage] = len(stage_complete)
        complete_keys[stage] = stage_complete

    repository_logs = set(root.glob("*.log"))
    reports_dir = root / "reports"
    if reports_dir.is_dir():
        repository_logs.update(reports_dir.rglob("*.log"))
    repository_logs = {
        path.resolve()
        for path in repository_logs
        if ".git" not in path.parts and ".numba_cache" not in path.parts
    }
    complete_repository_logs = {
        path for path in repository_logs if _has_complete_epoch_history(path)
    }
    complete = all(coverage.get(stage, 0) == len(EXPECTED_KEYS) for stage in stage_sources)
    coverage_text = {
        stage: f"{coverage.get(stage, 0)}/{len(EXPECTED_KEYS)}"
        for stage in stage_sources
    }
    coverage_text["B3"] = "not_applicable_post_hoc"
    if complete:
        reason = "Complete persisted epoch histories were detected for B0/B1/B2."
    else:
        reason = (
            "Persisted epoch-level train-loss and validation-Macro-F1 histories "
            "are incomplete; missing curves were not regenerated by retraining."
        )
    return {
        "complete_epoch_logs": complete,
        "epoch_log_coverage": coverage_text,
        "complete_fold_seed_keys": complete_keys,
        "candidate_history_files_scanned": [
            path.as_posix() for path in sorted(candidates_scanned, key=str)
        ],
        "complete_history_files": [
            path.as_posix() for path in sorted(complete_files, key=str)
        ],
        "repository_log_files_scanned": [
            path.as_posix() for path in sorted(repository_logs, key=str)
        ],
        "complete_repository_log_files": [
            path.as_posix() for path in sorted(complete_repository_logs, key=str)
        ],
        "training_curves_emitted": False,
        "reason": reason,
    }


def compute_convergence_tables(
    runs: pd.DataFrame,
    *,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return scalar convergence/stability points; never invent epoch curves."""

    stage_columns = {
        "B0": ("B0_best_epoch", "B0_best_val_macro_f1", "B0", "B0_source", "direct"),
        "B1": ("B1_best_epoch", "B1_best_val_macro_f1", "B1", "B1_source", "direct"),
        "B2": (
            "B2_valsel_best_epoch",
            "B2_valsel_best_val_macro_f1",
            "B2_valsel",
            "B2_valsel_source",
            "direct_selected_checkpoint",
        ),
        "B3": (
            "B3_valsel_best_epoch",
            "B3_valsel_best_val_macro_f1",
            "B3_valsel",
            "B2_valsel_source",
            "inherited_from_selected_B2_checkpoint",
        ),
    }
    epoch_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    for stage, (epoch_col, val_col, test_col, source_col, provenance) in stage_columns.items():
        epoch_values = runs[epoch_col].to_numpy(dtype=float)
        val_values = runs[val_col].to_numpy(dtype=float)
        test_values = runs[test_col].to_numpy(dtype=float)
        gaps = val_values - test_values
        q1_epoch, median_epoch, q3_epoch = np.quantile(epoch_values, [0.25, 0.5, 0.75])
        q1_gap, median_gap, q3_gap = np.quantile(gaps, [0.25, 0.5, 0.75])
        for index, run in runs.reset_index(drop=True).iterrows():
            epoch_rows.append(
                {
                    "stage": stage,
                    "fold": int(run["fold"]),
                    "seed": int(run["seed"]),
                    "best_epoch": int(epoch_values[index]),
                    "epoch_provenance": provenance,
                    "source": run[source_col],
                    "n_stage_runs": 25,
                    "stage_mean": float(epoch_values.mean()),
                    "stage_sd": float(epoch_values.std(ddof=1)),
                    "stage_min": float(epoch_values.min()),
                    "stage_q1": float(q1_epoch),
                    "stage_median": float(median_epoch),
                    "stage_q3": float(q3_epoch),
                    "stage_max": float(epoch_values.max()),
                }
            )
            gap_rows.append(
                {
                    "stage": stage,
                    "fold": int(run["fold"]),
                    "seed": int(run["seed"]),
                    "best_val_macro_f1": float(val_values[index]),
                    "test_macro_f1": float(test_values[index]),
                    "validation_minus_test_gap": float(gaps[index]),
                    "validation_metric_provenance": provenance,
                    "source": run[source_col],
                    "n_stage_runs": 25,
                    "stage_gap_mean": float(gaps.mean()),
                    "stage_gap_sd": float(gaps.std(ddof=1)),
                    "stage_gap_min": float(gaps.min()),
                    "stage_gap_q1": float(q1_gap),
                    "stage_gap_median": float(median_gap),
                    "stage_gap_q3": float(q3_gap),
                    "stage_gap_max": float(gaps.max()),
                }
            )
    log_audit = audit_epoch_logs(root or Path.cwd(), runs)
    return pd.DataFrame(epoch_rows), pd.DataFrame(gap_rows), log_audit


def _export_figure(fig: plt.Figure, stem: Path) -> tuple[Path, Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = stem.with_suffix(".svg")
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    return svg, pdf, png


def _panel_label(
    ax: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.05
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def build_lambda_selection_figure(selection: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 76 / 25.4))
    colors = {0.2: "#6D8FA3", 0.5: "#8A9D70", 1.0: "#B47A66"}
    x = np.arange(len(EXPECTED_FOLDS), dtype=float)
    offsets = {-0.18: 0.2, 0.0: 0.5, 0.18: 1.0}
    for offset, candidate in offsets.items():
        means = selection[_lambda_column(candidate, "mean_best_val_macro_f1")]
        sds = selection[_lambda_column(candidate, "sd_best_val_macro_f1")]
        axes[0].errorbar(
            x + offset,
            means,
            yerr=sds,
            fmt="o",
            ms=4,
            capsize=2,
            lw=1,
            color=colors[candidate],
            label=f"λ={candidate:g}",
        )
    axes[0].set_xticks(x, [str(fold) for fold in EXPECTED_FOLDS])
    axes[0].set_xlabel("Outer fold")
    axes[0].set_ylabel("Five-seed mean best validation Macro-F1")
    axes[0].grid(axis="y", color="#D9DEE5", linewidth=0.5)
    axes[0].legend(ncol=3, loc="lower left")
    _panel_label(axes[0], "a")

    selected = selection["selected_lambda"].to_numpy(dtype=float)
    axes[1].plot(x, selected, marker="o", color="#2E6F8E", lw=1.5)
    for fold, value in zip(EXPECTED_FOLDS, selected, strict=True):
        axes[1].text(fold, value + 0.045, f"λ={value:g}", ha="center", va="bottom")
    axes[1].set_xticks(x, [str(fold) for fold in EXPECTED_FOLDS])
    axes[1].set_yticks(CANDIDATE_LAMBDAS, [str(value) for value in CANDIDATE_LAMBDAS])
    axes[1].set_ylim(0.1, 1.1)
    axes[1].set_xlabel("Outer fold")
    axes[1].set_ylabel("Validation-selected λ")
    axes[1].grid(axis="y", color="#D9DEE5", linewidth=0.5)
    axes[1].text(
        0.98,
        0.96,
        "Selection uses validation only\n(mean → SD → smallest λ)",
        transform=axes[1].transAxes,
        color="#555D6B",
        ha="right",
        va="top",
    )
    _panel_label(axes[1], "b")
    fig.tight_layout(pad=1.2)
    return fig


def build_framework_figure(
    summary: pd.DataFrame, paired: pd.DataFrame
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 86 / 25.4))
    primary = ["B0", "B1", "B2_valsel", "B3_valsel"]
    colors = ["#555D6B", "#6D8FA3", "#8A9D70", "#2E6F8E"]
    lookup = summary.set_index("stage")
    means = np.asarray([lookup.loc[stage, "mean_macro_f1"] for stage in primary])
    sds = np.asarray([lookup.loc[stage, "sd_macro_f1"] for stage in primary])
    x = np.arange(len(primary))
    axes[0].bar(x, means, color=colors, edgecolor="white", width=0.68)
    axes[0].errorbar(x, means, yerr=sds, fmt="none", color="#20242A", capsize=3, lw=0.9)
    fixed_x = np.asarray([2, 3], dtype=float) + 0.24
    fixed_means = [
        lookup.loc["B2_fixed_w05", "mean_macro_f1"],
        lookup.loc["B3_fixed_w05", "mean_macro_f1"],
    ]
    axes[0].scatter(
        fixed_x,
        fixed_means,
        marker="D",
        s=22,
        facecolors="white",
        edgecolors="#B47A66",
        label="Fixed λ=0.5 sensitivity",
        zorder=4,
    )
    axes[0].set_xticks(x, ["B0", "B1", "B2\nval-selected", "B3\nval-selected"])
    axes[0].set_ylabel("Test Macro-F1")
    axes[0].set_ylim(max(0.88, float(means.min() - 0.03)), 0.98)
    axes[0].grid(axis="y", color="#D9DEE5", linewidth=0.5)
    axes[0].legend(loc="lower right")
    _panel_label(axes[0], "a")

    plot = paired.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    run_low = plot["bootstrap_ci95_low"].to_numpy(float)
    run_high = plot["bootstrap_ci95_high"].to_numpy(float)
    fold_low = plot["fold_cluster_bootstrap_ci95_low"].to_numpy(float)
    fold_high = plot["fold_cluster_bootstrap_ci95_high"].to_numpy(float)
    delta = plot["mean_delta"].to_numpy(float)
    for index in range(len(plot)):
        axes[1].plot([fold_low[index], fold_high[index]], [y[index], y[index]], color="#A3A9B1", lw=4, solid_capstyle="butt")
        axes[1].plot([run_low[index], run_high[index]], [y[index], y[index]], color="#2E6F8E", lw=1.5)
        axes[1].plot(delta[index], y[index], "o", color="#2E6F8E", ms=4)
    axes[1].axvline(0.0, color="#555D6B", lw=0.8, linestyle="--")
    axes[1].set_yticks(y, plot["comparison"])
    axes[1].set_xlabel("Paired test Macro-F1 difference")
    axes[1].grid(axis="x", color="#D9DEE5", linewidth=0.5)
    axes[1].text(
        0.02,
        -0.27,
        "Blue: 25-run bootstrap CI\nGrey: five-fold cluster bootstrap CI",
        transform=axes[1].transAxes,
        color="#555D6B",
        va="top",
        clip_on=False,
    )
    _panel_label(axes[1], "b")
    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(bottom=0.27)
    return fig


def _functional_value(
    summary: pd.DataFrame, method: str, metric: str
) -> float:
    match = summary[(summary["method"] == method) & (summary["metric"] == metric)]
    if len(match) != 1:
        return float("nan")
    return float(match.iloc[0]["mean"])


def build_functional_figure(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 112 / 25.4))
    methods = ("raw_softmax", "main_prototype", "hierarchical_prototype")
    labels = ("Raw", "Main proto", "Hier proto")
    colors = ("#6D8FA3", "#8A9D70", "#2E6F8E")
    x = np.arange(len(methods))
    width = 0.32
    panels = (
        (axes[0, 0], "main", "Main-class accuracy", "a"),
        (axes[0, 1], "subtype", "Subtype accuracy", "b"),
    )
    for ax, level, ylabel, panel in panels:
        top1 = [_functional_value(summary, method, f"{level}_top1_accuracy") for method in methods]
        top2 = [_functional_value(summary, method, f"{level}_top2_accuracy") for method in methods]
        ax.bar(x - width / 2, top1, width, color=colors, alpha=0.75, label="Top-1")
        ax.bar(x + width / 2, top2, width, color=colors, hatch="//", alpha=0.45, label="Top-2")
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.70, 1.01)
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.5)
        ax.legend(ncol=2, loc="lower right")
        _panel_label(ax, panel)

    consistency = [_functional_value(summary, method, "hierarchy_consistency_rate") for method in methods]
    consistent_error = [_functional_value(summary, method, "classification_error_rate_hierarchy_consistent") for method in methods]
    inconsistent_error = [_functional_value(summary, method, "classification_error_rate_hierarchy_inconsistent") for method in methods]
    axes[1, 0].bar(x - width, consistency, width, color=colors, alpha=0.8, label="Consistency")
    axes[1, 0].bar(x, consistent_error, width, color="#A3A9B1", label="Error | consistent")
    axes[1, 0].bar(x + width, inconsistent_error, width, color="#B47A66", label="Error | inconsistent")
    axes[1, 0].set_xticks(x, labels, rotation=18, ha="right")
    axes[1, 0].set_ylim(0, 1.01)
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].grid(axis="y", color="#D9DEE5", linewidth=0.5)
    axes[1, 0].legend(
        fontsize=6,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
    )
    _panel_label(axes[1, 0], "c")

    boundary = [_functional_value(summary, method, "feeding_stress_exact_pair_top2_coverage") for method in methods]
    axes[1, 1].bar(x, boundary, color=colors, width=0.62)
    axes[1, 1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1, 1].set_ylim(0, 1.01)
    axes[1, 1].set_ylabel("Feeding/stress exact-pair Top-2 coverage")
    axes[1, 1].grid(axis="y", color="#D9DEE5", linewidth=0.5)
    _panel_label(axes[1, 1], "d", x=0.015, y=0.92)
    fig.tight_layout(pad=1.2)
    return fig


def build_margin_figure(margins: pd.DataFrame) -> plt.Figure:
    runs = margins[margins["record_type"] == "run"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 112 / 25.4))
    methods = ("raw_softmax", "main_prototype", "hierarchical_prototype")
    labels = ("Raw", "Main proto", "Hier proto")
    colors = ("#6D8FA3", "#8A9D70", "#2E6F8E")
    for row, level in enumerate(("main", "subtype")):
        subset = runs[runs["level"] == level]
        correct = [subset.loc[subset["method"] == method, "correct_margin_median"].dropna().to_numpy(float) for method in methods]
        error = [subset.loc[subset["method"] == method, "error_margin_median"].dropna().to_numpy(float) for method in methods]
        positions_correct = np.arange(len(methods)) * 2.0
        positions_error = positions_correct + 0.65
        boxes_c = axes[row, 0].boxplot(correct, positions=positions_correct, widths=0.52, patch_artist=True, showfliers=False)
        boxes_e = axes[row, 0].boxplot(error, positions=positions_error, widths=0.52, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes_c["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for patch in boxes_e["boxes"]:
            patch.set_facecolor("#B47A66")
            patch.set_alpha(0.65)
        axes[row, 0].axhline(0, color="#555D6B", linestyle="--", lw=0.8)
        axes[row, 0].set_xticks(positions_correct + 0.325, labels)
        axes[row, 0].set_ylabel(f"{level.title()} favorable distance margin")
        axes[row, 0].grid(axis="y", color="#D9DEE5", linewidth=0.5)
        _panel_label(axes[row, 0], "a" if row == 0 else "c")

        auroc = [subset.loc[subset["method"] == method, "error_detection_auroc"].dropna().to_numpy(float) for method in methods]
        boxes = axes[row, 1].boxplot(auroc, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axes[row, 1].axhline(0.5, color="#555D6B", linestyle="--", lw=0.8)
        axes[row, 1].set_ylim(0.45, 1.02)
        axes[row, 1].set_ylabel(f"{level.title()} error AUROC\n(score = −margin)")
        axes[row, 1].grid(axis="y", color="#D9DEE5", linewidth=0.5)
        _panel_label(axes[row, 1], "b" if row == 0 else "d")
    fig.legend(
        handles=(
            Patch(facecolor="#6D8FA3", alpha=0.75, label="Method-correct margin"),
            Patch(facecolor="#B47A66", alpha=0.65, label="Method-error margin"),
        ),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=1.2)
    return fig


def build_convergence_figure(
    epochs: pd.DataFrame, gaps: pd.DataFrame
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 72 / 25.4))
    stages = ("B0", "B1", "B2", "B3")
    colors = ("#555D6B", "#6D8FA3", "#8A9D70", "#2E6F8E")
    epoch_values = [epochs.loc[epochs["stage"] == stage, "best_epoch"].to_numpy(float) for stage in stages]
    boxes = axes[0].boxplot(epoch_values, tick_labels=stages, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for index, values in enumerate(epoch_values, start=1):
        axes[0].scatter(np.full(values.size, index), values, s=6, color="#20242A", alpha=0.45)
    axes[0].set_ylabel("Best epoch")
    axes[0].grid(axis="y", color="#D9DEE5", linewidth=0.5)
    axes[0].text(
        0.52,
        1.035,
        "B3 inherits B2 checkpoint epoch",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        color="#555D6B",
        fontsize=7,
    )
    _panel_label(axes[0], "a")

    gap_values = [gaps.loc[gaps["stage"] == stage, "validation_minus_test_gap"].to_numpy(float) for stage in stages]
    boxes = axes[1].boxplot(gap_values, tick_labels=stages, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].axhline(0, color="#555D6B", linestyle="--", lw=0.8)
    axes[1].set_ylabel("Best validation − test Macro-F1")
    axes[1].grid(axis="y", color="#D9DEE5", linewidth=0.5)
    _panel_label(axes[1], "b")

    for stage, color in zip(stages, colors, strict=True):
        subset = gaps[gaps["stage"] == stage]
        axes[2].scatter(
            subset["best_val_macro_f1"],
            subset["test_macro_f1"],
            s=12,
            color=color,
            alpha=0.7,
            label=stage,
        )
    limits = (0.88, 1.005)
    axes[2].plot(limits, limits, color="#555D6B", linestyle="--", lw=0.8)
    axes[2].set_xlim(*limits)
    axes[2].set_ylim(*limits)
    axes[2].set_xlabel("Best validation Macro-F1")
    axes[2].set_ylabel("Test Macro-F1")
    axes[2].grid(color="#D9DEE5", linewidth=0.5)
    axes[2].legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.7,
    )
    _panel_label(axes[2], "c")
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=1.1)
    return fig


def determine_acceptance(
    paired: pd.DataFrame,
    functional: pd.DataFrame,
    margins: pd.DataFrame,
    disagreements: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    paired_lookup = paired.set_index("comparison")
    b2_delta = float(paired_lookup.loc["B2_valsel - B1", "mean_delta"])
    b3_increment = float(paired_lookup.loc["B3_valsel - B2_valsel", "mean_delta"])
    b3_vs_b1 = paired_lookup.loc["B3_valsel - B1"]
    positive_b3_folds = sum(
        float(b3_vs_b1[f"fold_{fold}_mean_delta"]) > 0
        for fold in EXPECTED_FOLDS
    )
    raw_top2 = _functional_value(functional, "raw_softmax", "main_top2_accuracy")
    hier_top2 = _functional_value(functional, "hierarchical_prototype", "main_top2_accuracy")
    hier_top1 = _functional_value(
        functional, "hierarchical_prototype", "main_top1_accuracy"
    )
    raw_consistency = _functional_value(functional, "raw_softmax", "hierarchy_consistency_rate")
    hier_consistency = _functional_value(functional, "hierarchical_prototype", "hierarchy_consistency_rate")
    hier_consistent_error = _functional_value(
        functional,
        "hierarchical_prototype",
        "classification_error_rate_hierarchy_consistent",
    )
    hier_inconsistent_error = _functional_value(
        functional,
        "hierarchical_prototype",
        "classification_error_rate_hierarchy_inconsistent",
    )
    hier_main_margin = margins[
        (margins["record_type"] == "overall_run_mean")
        & (margins["method"] == "hierarchical_prototype")
        & (margins["level"] == "main")
    ]
    margin_auroc = (
        float(hier_main_margin.iloc[0]["error_detection_auroc"])
        if len(hier_main_margin) == 1
        else float("nan")
    )
    hier_disagreement = disagreements[
        (disagreements["record_type"] == "overall_run_mean")
        & (disagreements["prototype_method"] == "hierarchical_prototype")
    ]
    rescued = (
        float(hier_disagreement.iloc[0]["prototype_correct_raw_wrong"])
        if len(hier_disagreement) == 1
        else float("nan")
    )
    harmed = (
        float(hier_disagreement.iloc[0]["raw_correct_prototype_wrong"])
        if len(hier_disagreement) == 1
        else float("nan")
    )
    no_major_functional_contradiction = bool(
        hier_top2 + 1e-12 >= raw_top2
        and hier_consistency + 1e-12 >= raw_consistency
        and (not np.isfinite(margin_auroc) or margin_auroc >= 0.5)
        and (not np.isfinite(rescued) or rescued + 1e-12 >= harmed)
    )
    measurable_functional_value = bool(
        hier_top2 > hier_top1 + 1e-12
        or (
            np.isfinite(hier_inconsistent_error)
            and np.isfinite(hier_consistent_error)
            and hier_inconsistent_error > hier_consistent_error + 1e-12
        )
        or (np.isfinite(margin_auroc) and margin_auroc > 0.5)
        or (np.isfinite(rescued) and rescued > 0)
    )
    if (
        b2_delta > 0
        and b3_increment > 0
        and positive_b3_folds >= 4
        and no_major_functional_contradiction
    ):
        label = "framework_strengthened"
    elif measurable_functional_value:
        label = "framework_functional_only"
    else:
        label = "prototype_claim_should_be_reduced"
    return label, {
        "B2_valsel_minus_B1_mean_delta": b2_delta,
        "B3_valsel_minus_B2_valsel_mean_delta": b3_increment,
        "positive_B3_valsel_minus_B1_folds": positive_b3_folds,
        "raw_main_top2": raw_top2,
        "hierarchical_main_top2": hier_top2,
        "hierarchical_main_top1": hier_top1,
        "raw_hierarchy_consistency": raw_consistency,
        "hierarchical_hierarchy_consistency": hier_consistency,
        "hierarchical_error_rate_if_consistent": hier_consistent_error,
        "hierarchical_error_rate_if_inconsistent": hier_inconsistent_error,
        "hierarchical_main_margin_error_auroc": margin_auroc,
        "mean_rescued_disagreements_per_run": rescued,
        "mean_harmed_disagreements_per_run": harmed,
        "no_major_functional_contradiction": no_major_functional_contradiction,
        "measurable_functional_value": measurable_functional_value,
    }


def _format_float(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, separator, *body])


def build_final_report(
    *,
    selection: pd.DataFrame,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    functional: pd.DataFrame,
    margins: pd.DataFrame,
    disagreements: pd.DataFrame,
    epochs: pd.DataFrame,
    gaps: pd.DataFrame,
    log_audit: Mapping[str, Any],
    acceptance: str,
    acceptance_evidence: Mapping[str, Any],
    source_file_count: int,
) -> str:
    selection_rows = [
        (
            int(row.fold),
            _format_float(row.selected_lambda, 1),
            _format_float(row.lambda_0_2_mean_best_val_macro_f1),
            _format_float(row.lambda_0_5_mean_best_val_macro_f1),
            _format_float(row.lambda_1_0_mean_best_val_macro_f1),
            row.selection_reason,
        )
        for row in selection.itertuples(index=False)
    ]
    stage_rows = [
        (
            row.stage,
            row.n,
            _format_float(row.mean_macro_f1),
            _format_float(row.sd_macro_f1),
            _format_float(row.min_macro_f1),
            _format_float(row.max_macro_f1),
        )
        for row in summary.itertuples(index=False)
    ]
    paired_rows = [
        (
            row.comparison,
            row.n,
            _format_float(row.mean_delta),
            f"[{_format_float(row.bootstrap_ci95_low)}, {_format_float(row.bootstrap_ci95_high)}]",
            _format_float(row.wilcoxon_raw_p, 9),
            _format_float(row.holm_adjusted_p, 9),
            f"{row.wins}/{row.ties}/{row.losses}",
            f"[{_format_float(row.fold_cluster_bootstrap_ci95_low)}, {_format_float(row.fold_cluster_bootstrap_ci95_high)}]",
        )
        for row in paired.itertuples(index=False)
    ]
    selected_functional_metrics = (
        "main_top1_accuracy",
        "main_top2_accuracy",
        "subtype_top1_accuracy",
        "subtype_top2_accuracy",
        "hierarchy_consistency_rate",
        "classification_error_rate_hierarchy_consistent",
        "classification_error_rate_hierarchy_inconsistent",
        "feeding_stress_exact_pair_top2_coverage",
    )
    functional_rows = [
        (
            row.method,
            row.metric,
            row.n_runs_available,
            _format_float(row.mean),
            _format_float(row.sd),
            f"[{_format_float(row.fold_cluster_ci95_low)}, {_format_float(row.fold_cluster_ci95_high)}]",
        )
        for row in functional.itertuples(index=False)
        if row.metric in selected_functional_metrics
    ]
    geometry_rows = [
        (
            row.metric,
            row.n_runs_available,
            _format_float(row.mean),
            _format_float(row.sd),
            f"[{_format_float(row.fold_cluster_ci95_low)}, {_format_float(row.fold_cluster_ci95_high)}]",
        )
        for row in functional.itertuples(index=False)
        if row.method == "prototype_geometry"
        and row.metric
        in {
            "true_main_prototype_mean_rank",
            "true_main_prototype_rank1_rate",
            "true_main_prototype_rank2_rate",
            "true_main_prototype_mrr",
            "true_subtype_prototype_mean_rank",
            "true_subtype_prototype_rank1_rate",
            "true_subtype_prototype_rank2_rate",
            "true_subtype_prototype_mrr",
        }
    ]
    margin_rows = [
        (
            row.method,
            row.level,
            _format_float(row.correct_margin_median),
            _format_float(row.error_margin_median),
            _format_float(row.error_detection_auroc),
            f"[{_format_float(row.error_detection_auroc_fold_cluster_ci95_low)}, {_format_float(row.error_detection_auroc_fold_cluster_ci95_high)}]",
        )
        for row in margins.itertuples(index=False)
        if row.record_type == "overall_run_mean"
    ]
    disagreement_rows = [
        (
            row.prototype_method,
            _format_float(row.disagreement_rate),
            int(row.repeated_run_total_n_disagreements),
            int(row.repeated_run_total_raw_correct_prototype_wrong),
            int(row.repeated_run_total_prototype_correct_raw_wrong),
            int(row.repeated_run_total_both_wrong),
            f"[{_format_float(row.disagreement_rate_fold_cluster_ci95_low)}, {_format_float(row.disagreement_rate_fold_cluster_ci95_high)}]",
        )
        for row in disagreements.itertuples(index=False)
        if row.record_type == "overall_run_mean"
    ]
    max_probability_drift = float(
        runs["B3_valsel_softmax_probability_max_abs_diff"].max()
    )
    rank_mismatch_count = int(
        runs[
            "B3_valsel_softmax_rank_order_mismatch_count_before_canonical_restore"
        ].sum()
    )
    epoch_rows = []
    for stage in ("B0", "B1", "B2", "B3"):
        subset = epochs[epochs["stage"] == stage]
        first = subset.iloc[0]
        epoch_rows.append(
            (
                stage,
                _format_float(first["stage_mean"], 2),
                _format_float(first["stage_sd"], 2),
                _format_float(first["stage_median"], 1),
                f"{int(first['stage_min'])}-{int(first['stage_max'])}",
                first["epoch_provenance"],
            )
        )
    gap_rows = []
    for stage in ("B0", "B1", "B2", "B3"):
        first = gaps[gaps["stage"] == stage].iloc[0]
        gap_rows.append(
            (
                stage,
                _format_float(first["stage_gap_mean"]),
                _format_float(first["stage_gap_sd"]),
                _format_float(first["stage_gap_median"]),
                f"[{_format_float(first['stage_gap_min'])}, {_format_float(first['stage_gap_max'])}]",
            )
        )

    acceptance_text = {
        "framework_strengthened": "A. framework_strengthened",
        "framework_functional_only": "B. framework_functional_only",
        "prototype_claim_should_be_reduced": "C. prototype_claim_should_be_reduced",
    }[acceptance]
    return f"""# Final Validation Report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: {datetime.now(timezone.utc).isoformat()}
- Verification Status: VERIFIED
- Version Label: final_validation_v1
- Source: frozen clean 5-fold × 5-seed experiment artifacts
- Overall Confidence: CAUTION

## Scope and locked interpretation

This is a locked validation-only sensitivity analysis. It is neither fully prospective nor fully confirmatory. Candidate λ values were fixed at 0.2, 0.5, and 1.0; selection used only the five-seed validation mean, then validation SD within a 1e-6 mean tie, then the smallest λ. Outer-fold test metrics did not participate in selection. No backbone was retrained and no test-derived margin threshold was selected.

## Fold-wise validation selection

{_markdown_table(("Fold", "Selected λ", "λ=0.2 val mean", "λ=0.5 val mean", "λ=1.0 val mean", "Reason"), selection_rows)}

Every one of the 75 source validation summaries is recorded with its SHA256 in `lambda_selection_provenance.json`.

## Validation-selected framework results

{_markdown_table(("Stage", "n", "Mean Macro-F1", "SD", "Min", "Max"), stage_rows)}

{_markdown_table(("Comparison", "n", "Mean Δ", "Run bootstrap 95% CI", "Wilcoxon raw P", "Holm P", "W/T/L", "Fold-cluster 95% CI"), paired_rows)}

Holm adjustment is applied once across the seven requested validation-selected/fixed-sensitivity comparisons. Seeds are not treated as independent test cohorts: five fold-mean deltas and a five-fold cluster bootstrap are reported separately.

## Prototype functional value

{_markdown_table(("Method", "Metric", "Available runs", "Mean", "SD", "Fold-cluster 95% CI"), functional_rows)}

### Prototype ranks and margins

{_markdown_table(("Geometry metric", "Available runs", "Mean", "SD", "Fold-cluster 95% CI"), geometry_rows)}

{_markdown_table(("Method", "Level", "Median margin if correct", "Median margin if error", "Error AUROC", "AUROC fold-cluster 95% CI"), margin_rows)}

### Raw/prototype disagreements

{_markdown_table(("Prototype method", "Mean disagreement rate", "Repeated-run disagreements", "Raw correct / prototype wrong", "Prototype correct / Raw wrong", "Both wrong", "Rate fold-cluster 95% CI"), disagreement_rows)}

True-main and true-subtype prototype ranks are computed from ascending stored cosine distances with stable fixed-label tie breaking. Main and hierarchical prototype methods share the same auxiliary-prototype ordering; their subtype values are therefore not independent improvements. Favorable distance margin is `nearest wrong prototype distance - true prototype distance`; per-run error AUROC uses negative margin only as a diagnostic score and no test threshold is fitted.

Disagreement counts are reported per fold-seed before aggregation. The representative description remains: **“a training sample closest to the predicted class prototype”**. Case studies, where used elsewhere, remain deterministic and illustrative only.

## Convergence and stability

{_markdown_table(("Stage", "Mean best epoch", "SD", "Median", "Range", "Provenance"), epoch_rows)}

{_markdown_table(("Stage", "Mean val-test gap", "SD", "Median", "Range"), gap_rows)}

Complete epoch-level train-loss and validation-Macro-F1 histories were unavailable (`{log_audit['epoch_log_coverage']}`). Median/IQR learning curves were therefore not emitted or regenerated. B3 is post-hoc inference and inherits the selected B2 checkpoint epoch and validation metric; it has no independent training convergence trajectory.

## Acceptance interpretation

**{acceptance_text}**

- B2_valsel − B1 mean Δ: {_format_float(acceptance_evidence['B2_valsel_minus_B1_mean_delta'])}
- B3_valsel − B2_valsel mean Δ: {_format_float(acceptance_evidence['B3_valsel_minus_B2_valsel_mean_delta'])}
- Positive B3_valsel − B1 fold means: {acceptance_evidence['positive_B3_valsel_minus_B1_folds']}/5
- Major functional contradiction detected: {not acceptance_evidence['no_major_functional_contradiction']}
- Measurable candidate-prediction value: {acceptance_evidence['measurable_functional_value']}

The label is applied from the evidence literally rather than selected to favour the paper.

## Leakage, provenance, and reproducibility

- Train data: checkpoint fitting and train-only main/subtype prototypes.
- Validation data: λ selection by outer fold and the existing prototype temperature/penalty/rejection calibration protocol.
- Test data: frozen evaluation only.
- Every included prototype run passed exact-path, source-ID, and MD5 disjointness audits; exact path/source-ID/MD5/main-label/subtype membership against its train, validation, and frozen-test manifest; freshly recomputed checkpoint/summary/manifest/prototype-bundle/calibration/prediction SHA256 link verification; exact Softmax prediction/Macro-F1 reproduction; and `training_exact` feature qualification.
- Reproduced Raw Softmax probabilities had a maximum advisory absolute drift of {_format_float(max_probability_drift, 9)} versus the canonical training output, with {rank_mismatch_count} full main-class rank-order mismatches. The compact canonical rows are first bound to frozen manifest truth, then joined one-to-one to exact path/source-ID/MD5 identities with zero truth or prediction mismatches. Functional Raw main-class ranks use those identity-aligned canonical probabilities; auxiliary Softmax probabilities use the `training_exact` reproduction because the original compact test CSV did not persist auxiliary scores.
- Frozen-source SHA256 files checked before and after analysis: {source_file_count}; changed: 0.
- Missing required selected combinations after reconstruction: none.
- Existing epoch histories were searched in all 75 B0/B1/selected-B2 run directories; {len(log_audit.get('candidate_history_files_scanned', []))} candidate history files and {len(log_audit.get('repository_log_files_scanned', []))} repository `.log` files were structurally inspected without retraining.

## Statistical fallacy scan (11/11 checked)

| Fallacy | Status | Audit finding |
|---|---|---|
| Simpson's paradox | NOTE | Aggregate and all five fold-level directions are reported; no aggregate-only inference is used. |
| Ecological fallacy | NOTE | Fold/run summaries are not converted into clip- or farm-level causal claims. |
| Berkson's paradox | CAUTION | The frozen closed-set cohort is selected and does not establish real-farm prevalence performance. |
| Collider bias | NOTE | No post-outcome covariate adjustment is performed. |
| Base-rate neglect | CAUTION | Top-k and Macro-F1 are reported as closed-set metrics, not diagnostic predictive values. |
| Regression to the mean | NOTE | All 25 matched runs are retained; no extreme run is selected. |
| Survivorship bias | NOTE | No fold-seed result is dropped; all required keys are present. |
| Look-elsewhere effect | NOTE | Seven requested contrasts are all shown and Holm-adjusted together. |
| Garden of forking paths | CAUTION | The analysis is explicitly validation-selected and retrospective, not fully confirmatory. |
| Correlation ≠ causation | NOTE | No causal biological or deployment claim is made from embedding geometry. |
| Reverse causality | NOTE | Not applicable to the fixed prediction comparison; no directional causal claim is made. |

## Reproducibility entry points

The audit branch is `paper/final-validation-audit`; the implementation and delivery commit SHAs are recorded in `handoff/CODEX_TO_GPT.md` and `handoff/CODEX_TO_GPT.json` after commit creation. From a checkout where the target output directory does not yet exist:

```powershell
cd C:/py/pigsound/pig-sound-classification
$env:PYTHONNOUSERSITE="1"
$env:NUMBA_CACHE_DIR=(Resolve-Path '.numba_cache').Path
& "C:/py/anaconda3/envs/pigsound-gpu/python.exe" tools/generate_final_validation_audit.py --root . --validate-only
& "C:/py/anaconda3/envs/pigsound-gpu/python.exe" tools/generate_final_validation_audit.py --root .
```

The one-time 20-run selected-lambda prototype reconstruction used the five existing exact-protocol commands without `--allow_overwrite`; its complete fold/seed loop, fixed grids, and output-directory patterns are recorded in the GPT handoff. No backbone was retrained.

## Output status

All required audit tables and the five preliminary SVG/PDF/PNG audit figures are under `paper/final_validation/`. These are not final manuscript figures. The manuscript was not regenerated.
"""


def generate(
    root: Path,
    output_dir: Path,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    validate_only: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_dir = output_dir.resolve()
    planned_outputs = required_output_paths(output_dir)
    if not validate_only:
        refuse_existing(planned_outputs)

    selection_records, selection_hashes = load_lambda_selection_sources(root)
    selection, selected = select_lambda_by_fold(selection_records)
    runs, predictions, framework_source_paths = load_validation_selected_runs(
        root, selected
    )
    source_paths = tuple(
        sorted(
            {
                *(path.resolve() for path in framework_source_paths),
                *(
                    (root / path).resolve()
                    for path in selection_hashes
                ),
            },
            key=str,
        )
    )
    source_hashes_before = {
        _relative(root, path): sha256_file(path) for path in source_paths
    }

    framework_summary = compute_framework_summary(runs)
    paired, folds = paired_statistics(
        runs, FRAMEWORK_COMPARISONS, n_boot=n_boot, seed=seed
    )
    functional_run_frames: list[pd.DataFrame] = []
    margin_run_frames: list[pd.DataFrame] = []
    disagreement_run_frames: list[pd.DataFrame] = []
    for (fold, run_seed), prediction_frame in sorted(predictions.items()):
        functional_run, margins_run, disagreements_run = compute_run_functional_metrics(
            prediction_frame, fold=fold, seed=run_seed
        )
        functional_run_frames.append(functional_run)
        margin_run_frames.append(margins_run)
        disagreement_run_frames.append(disagreements_run)
    functional_runs = pd.concat(functional_run_frames, ignore_index=True)
    margin_runs = pd.concat(margin_run_frames, ignore_index=True)
    disagreement_runs = pd.concat(disagreement_run_frames, ignore_index=True)
    functional_summary = aggregate_functional_metrics(
        functional_runs, n_boot=n_boot, seed=seed
    )
    margins = add_margin_aggregates(margin_runs, n_boot=n_boot, seed=seed)
    disagreements = add_disagreement_aggregates(
        disagreement_runs, n_boot=n_boot, seed=seed
    )
    epochs, gaps, log_audit = compute_convergence_tables(runs, root=root)
    acceptance, acceptance_evidence = determine_acceptance(
        paired, functional_summary, margins, disagreements
    )

    figures = {
        "lambda_selection_by_fold": build_lambda_selection_figure(selection),
        "validation_selected_cumulative_framework": build_framework_figure(
            framework_summary, paired
        ),
        "prototype_functional_value": build_functional_figure(functional_summary),
        "prototype_margin_correct_vs_error": build_margin_figure(margins),
        "best_epoch_and_val_test_gap": build_convergence_figure(epochs, gaps),
    }
    source_hashes_after = {
        _relative(root, path): sha256_file(path) for path in source_paths
    }
    if source_hashes_before != source_hashes_after:
        changed = sorted(
            path
            for path, digest in source_hashes_before.items()
            if source_hashes_after.get(path) != digest
        )
        raise RuntimeError(f"Frozen sources changed during analysis: {changed}")

    provenance = {
        "artifact_type": "final_validation_lambda_selection_provenance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_characterization": "locked_validation_only_sensitivity_analysis",
        "fully_prospective": False,
        "fully_confirmatory": False,
        "candidate_lambdas": list(CANDIDATE_LAMBDAS),
        "folds": list(EXPECTED_FOLDS),
        "seeds": list(EXPECTED_SEEDS),
        "selection_rule": {
            "primary": "highest five-seed mean best_val_macro_f1",
            "mean_tie_tolerance": 1e-6,
            "secondary": "lowest five-seed validation sample SD",
            "tertiary": "smallest lambda",
        },
        "test_metrics_used_for_selection": False,
        "selected_lambda_by_fold": {
            str(fold): float(candidate) for fold, candidate in selected.items()
        },
        "selection_sources": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(selection_hashes.items())
        ],
        "selection_source_count": len(selection_hashes),
        "all_frozen_sources_before": source_hashes_before,
        "all_frozen_sources_after": source_hashes_after,
        "source_hashes_unchanged": True,
        "source_file_count": len(source_paths),
        "softmax_reproduction": {
            "prediction_and_macro_f1_exact": True,
            "maximum_probability_abs_drift": float(
                runs["B3_valsel_softmax_probability_max_abs_diff"].max()
            ),
            "rank_order_mismatch_count_before_canonical_restore": int(
                runs[
                    "B3_valsel_softmax_rank_order_mismatch_count_before_canonical_restore"
                ].sum()
            ),
            "functional_raw_main_probability_source": "canonical training test_pred.csv after verified row alignment",
            "canonical_probability_identity_join": "one_to_one",
            "canonical_truth_mismatch_count": int(
                runs["B3_valsel_canonical_probability_truth_mismatch_count"].sum()
            ),
            "canonical_prediction_mismatch_count": int(
                runs[
                    "B3_valsel_canonical_probability_prediction_mismatch_count"
                ].sum()
            ),
            "probability_tolerance_is_advisory": True,
        },
        "artifact_hash_link_verification": {
            "selected_runs": len(runs),
            "minimum_links_verified_per_selected_run": int(
                runs["B3_valsel_artifact_hash_links_verified"].min()
            ),
            "all_selected_runs_verified": bool(
                (runs["B3_valsel_artifact_hash_links_verified"] > 0).all()
            ),
            "note": (
                "Every recorded checkpoint/summary/manifest/prototype-bundle/"
                "calibration/prediction hash link is compared with freshly "
                "recomputed file bytes during run validation."
            ),
        },
        "split_membership_verification": {
            "selected_runs": len(runs),
            "path_source_id_md5_main_subtype_exact": True,
            "train_main_and_aux_row_counts_equal": bool(
                (
                    runs["B3_valsel_train_main_membership_n"]
                    == runs["B3_valsel_train_aux_membership_n"]
                ).all()
            ),
            "validation_rows_verified": int(
                runs["B3_valsel_validation_membership_n"].sum()
            ),
            "test_rows_verified": int(runs["B3_valsel_test_membership_n"].sum()),
            "test_main_truth_mismatches": 0,
            "test_subtype_truth_mismatches": 0,
            "prototype_source_split": "train",
            "calibration_source_split": "validation",
            "evaluation_source_split": "frozen_test",
        },
        "missing_artifacts": [],
        "epoch_log_audit": log_audit,
        "acceptance_interpretation": acceptance,
        "acceptance_evidence": acceptance_evidence,
    }
    report = build_final_report(
        selection=selection,
        runs=runs,
        summary=framework_summary,
        paired=paired,
        functional=functional_summary,
        margins=margins,
        disagreements=disagreements,
        epochs=epochs,
        gaps=gaps,
        log_audit=log_audit,
        acceptance=acceptance,
        acceptance_evidence=acceptance_evidence,
        source_file_count=len(source_paths),
    )
    table_frames = {
        "lambda_selection_by_fold.csv": selection,
        "validation_selected_framework_runs.csv": runs,
        "validation_selected_framework_summary.csv": framework_summary,
        "validation_selected_framework_paired_stats.csv": paired,
        "validation_selected_framework_fold_stats.csv": folds,
        "prototype_functional_value_summary.csv": functional_summary,
        "prototype_margin_error_detection.csv": margins,
        "prototype_disagreement_analysis.csv": disagreements,
        "convergence_best_epoch_summary.csv": epochs,
        "validation_test_gap_summary.csv": gaps,
    }
    written: list[str] = []
    if validate_only:
        for fig in figures.values():
            plt.close(fig)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        for filename, frame in table_frames.items():
            path = output_dir / filename
            frame.to_csv(path, index=False, encoding="utf-8")
            written.append(_relative(root, path))
        provenance_path = output_dir / PROVENANCE_FILENAME
        provenance_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(_relative(root, provenance_path))
        report_path = output_dir / REPORT_FILENAME
        report_path.write_text(report, encoding="utf-8")
        written.append(_relative(root, report_path))
        for stem, fig in figures.items():
            written.extend(
                _relative(root, path)
                for path in _export_figure(fig, output_dir / "figures" / stem)
            )
    return {
        "selection": selection,
        "runs": runs,
        "summary": framework_summary,
        "paired": paired,
        "folds": folds,
        "functional": functional_summary,
        "margins": margins,
        "disagreements": disagreements,
        "epochs": epochs,
        "gaps": gaps,
        "acceptance": acceptance,
        "acceptance_evidence": acceptance_evidence,
        "provenance": provenance,
        "written": written,
        "source_hashes_unchanged": True,
        "source_file_count": len(source_paths),
        "report": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the locked validation-only lambda, validation-selected "
            "framework, prototype functional-value, and convergence audit from "
            "existing frozen artifacts. No training is run."
        )
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Project root."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/final_validation"),
        help="Unique audit output directory relative to --root.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
        help="Percentile-bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
        help="Bootstrap seed (default: 3407).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run every validation and calculation without writing outputs.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Print selected lambdas and missing selected artifacts without "
            "loading prototype predictions or writing outputs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    root = args.root.resolve()
    if args.preflight_only:
        records, _ = load_lambda_selection_sources(root)
        selection, selected = select_lambda_by_fold(records)
        print(selection.to_string(index=False))
        missing = find_missing_selected_artifacts(root, selected)
        print(json.dumps({"missing_artifacts": missing}, indent=2))
        return
    result = generate(
        root,
        args.output_dir,
        n_boot=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        validate_only=args.validate_only,
    )
    print(result["selection"].to_string(index=False))
    print(result["summary"].to_string(index=False))
    print(result["paired"].to_string(index=False))
    print(f"acceptance={result['acceptance']}")
    print(f"source_file_count={result['source_file_count']}")
    print(f"source_hashes_unchanged={result['source_hashes_unchanged']}")
    if result["written"]:
        print("written:")
        for path in result["written"]:
            print(f"  {path}")


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a required source file."""

    if not path.is_file():
        raise FileNotFoundError(f"Required source file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_recorded_digest(
    actual: str, recorded: Any, *, path: Path, context: str
) -> None:
    text = "" if recorded is None else str(recorded).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(
            f"Missing/invalid recorded SHA256 for {context}: {path} ({recorded!r})"
        )
    if actual.lower() != text:
        raise ValueError(
            f"SHA256 mismatch for {context}: {path}; recorded={text}, actual={actual}"
        )


def verify_recorded_sha256(path: Path, recorded: Any, *, context: str) -> str:
    """Recompute one file digest and reject missing, invalid, or stale metadata."""

    path = path.resolve()
    actual = sha256_file(path)
    _validate_recorded_digest(actual, recorded, path=path, context=context)
    return actual


def verify_recorded_sha256_links(
    links: Sequence[tuple[Path, Any, str]],
) -> int:
    """Verify every recorded hash link while hashing each distinct file once."""

    actual_by_path: dict[Path, str] = {}
    for path, recorded, context in links:
        resolved = path.resolve()
        if resolved not in actual_by_path:
            actual_by_path[resolved] = sha256_file(resolved)
        _validate_recorded_digest(
            actual_by_path[resolved], recorded, path=resolved, context=context
        )
    return len(links)


def _normalise_identity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalised = frame.copy()
    normalised["path"] = (
        normalised["path"].astype(str).str.replace("\\", "/", regex=False)
    )
    normalised["source_id"] = normalised["source_id"].astype(str)
    normalised["md5"] = normalised["md5"].astype(str).str.lower()
    return normalised


def validate_frozen_split_membership(
    artifact: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    context: str,
    y_true_column: str | None = None,
    aux_true_column: str | None = None,
) -> int:
    """Require exact path/source-ID/MD5/main/subtype membership and truth."""

    columns = ("path", "source_id", "md5", "label", "subtype")
    missing_artifact = sorted(set(columns) - set(artifact.columns))
    missing_manifest = sorted(set(columns) - set(manifest.columns))
    if missing_artifact or missing_manifest:
        raise ValueError(
            f"Frozen split membership columns are missing in {context}: "
            f"artifact={missing_artifact}, manifest={missing_manifest}"
        )
    left = _normalise_identity_columns(artifact[list(columns)])
    right = _normalise_identity_columns(manifest[list(columns)])
    for frame in (left, right):
        frame["label"] = frame["label"].astype(str).str.lower()
        frame["subtype"] = frame["subtype"].astype(str).str.lower()
    if left[list(columns[:3])].duplicated().any():
        raise ValueError(f"Duplicate artifact identities in {context}")
    if right[list(columns[:3])].duplicated().any():
        raise ValueError(f"Duplicate manifest identities in {context}")
    left = left.sort_values(list(columns[:3]), kind="stable").reset_index(drop=True)
    right = right.sort_values(list(columns[:3]), kind="stable").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(f"Artifact rows do not exactly match frozen split: {context}")

    artifact_by_identity = _normalise_identity_columns(artifact.copy()).set_index(
        list(columns[:3]), verify_integrity=True
    )
    manifest_by_identity = _normalise_identity_columns(manifest.copy()).set_index(
        list(columns[:3]), verify_integrity=True
    )
    manifest_by_identity = manifest_by_identity.loc[artifact_by_identity.index]
    if y_true_column is not None:
        if y_true_column not in artifact_by_identity.columns:
            raise ValueError(f"Missing {y_true_column} in {context}")
        mismatch = (
            artifact_by_identity[y_true_column].astype(str).str.lower().to_numpy()
            != manifest_by_identity["label"].astype(str).str.lower().to_numpy()
        )
        if mismatch.any():
            raise ValueError(
                f"Main truth does not match frozen manifest in {context}: "
                f"{int(mismatch.sum())} rows"
            )
    if aux_true_column is not None:
        if aux_true_column not in artifact_by_identity.columns:
            raise ValueError(f"Missing {aux_true_column} in {context}")
        mismatch = (
            artifact_by_identity[aux_true_column].astype(str).str.lower().to_numpy()
            != manifest_by_identity["subtype"].astype(str).str.lower().to_numpy()
        )
        if mismatch.any():
            raise ValueError(
                f"Subtype truth does not match frozen manifest in {context}: "
                f"{int(mismatch.sum())} rows"
            )
    return len(left)


def _main_class_ids(values: pd.Series, *, context: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        ids = numeric.to_numpy(dtype=np.int64)
    else:
        mapped = values.astype(str).str.lower().map(
            {label: index for index, label in enumerate(MAIN_LABELS)}
        )
        if mapped.isna().any():
            unknown = sorted(set(values[mapped.isna()].astype(str)))
            raise ValueError(f"Unknown main labels in {context}: {unknown}")
        ids = mapped.to_numpy(dtype=np.int64)
    if not np.isin(ids, np.arange(len(MAIN_LABELS))).all():
        raise ValueError(f"Out-of-range main-class IDs in {context}: {ids.tolist()}")
    return ids


def _as_bool_array(values: pd.Series, *, context: str) -> np.ndarray:
    parsed: list[bool] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            parsed.append(True)
        elif text in {"false", "0", "no"}:
            parsed.append(False)
        else:
            raise ValueError(f"Invalid boolean value {value!r} in {context}")
    return np.asarray(parsed, dtype=bool)


def validate_softmax_alignment_report(
    alignment: pd.DataFrame,
    test_manifest: pd.DataFrame,
    *,
    context: str,
) -> None:
    required = {
        "path",
        "in_reference",
        "in_reproduced",
        "duplicate_in_reference",
        "duplicate_in_reproduced",
        "y_true_match",
        "y_pred_match",
        "prediction_match",
    }
    missing = sorted(required - set(alignment.columns))
    if missing:
        raise ValueError(f"Softmax alignment report is missing {missing}: {context}")
    manifest_paths = (
        test_manifest["path"].astype(str).str.replace("\\", "/", regex=False)
    )
    report_paths = alignment["path"].astype(str).str.replace("\\", "/", regex=False)
    if manifest_paths.duplicated().any() or report_paths.duplicated().any():
        raise ValueError(f"Duplicate paths in Softmax alignment audit: {context}")
    if set(manifest_paths) != set(report_paths):
        raise ValueError(f"Softmax alignment paths differ from test manifest: {context}")
    required_true = (
        "in_reference",
        "in_reproduced",
        "y_true_match",
        "y_pred_match",
        "prediction_match",
    )
    required_false = ("duplicate_in_reference", "duplicate_in_reproduced")
    failed = [
        column
        for column in required_true
        if not _as_bool_array(alignment[column], context=context).all()
    ]
    failed.extend(
        column
        for column in required_false
        if _as_bool_array(alignment[column], context=context).any()
    )
    if failed:
        raise ValueError(f"Softmax alignment report failed {failed}: {context}")


def align_canonical_raw_probabilities(
    reproduced: pd.DataFrame,
    reference: pd.DataFrame,
    test_manifest: pd.DataFrame,
    *,
    context: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach compact canonical probabilities through a verified identity join.

    The compact training ``test_pred.csv`` does not persist identities. Its row
    order is first bound to the frozen manifest by true-label equality; those
    manifest identities are then joined one-to-one to the exact reproduction.
    """

    identity = ("path", "source_id", "md5")
    reference_probability_columns = [f"prob_{label}" for label in MAIN_LABELS]
    reproduced_probability_columns = [
        f"raw_softmax_prob_{label}" for label in MAIN_LABELS
    ]
    required_reference = {"y_true", "y_pred", *reference_probability_columns}
    required_reproduced = {
        *identity,
        "y_true",
        "raw_softmax_pred",
        *reproduced_probability_columns,
    }
    required_manifest = {*identity, "label"}
    missing = {
        "reference": sorted(required_reference - set(reference.columns)),
        "reproduced": sorted(required_reproduced - set(reproduced.columns)),
        "manifest": sorted(required_manifest - set(test_manifest.columns)),
    }
    if any(missing.values()):
        raise ValueError(f"Canonical probability alignment is missing columns {missing}: {context}")
    if len(reference) != len(test_manifest) or len(reproduced) != len(test_manifest):
        raise ValueError(
            "Canonical probability alignment row counts differ: "
            f"reference={len(reference)}, reproduced={len(reproduced)}, "
            f"manifest={len(test_manifest)} ({context})"
        )

    manifest = _normalise_identity_columns(test_manifest[list(identity) + ["label"]])
    if manifest[list(identity)].duplicated().any():
        raise ValueError(f"Duplicate frozen identities in test manifest: {context}")
    expected_true_ids = _main_class_ids(manifest["label"], context=f"manifest {context}")
    reference_true_ids = _main_class_ids(reference["y_true"], context=f"reference {context}")
    if not np.array_equal(expected_true_ids, reference_true_ids):
        mismatch = int(np.sum(expected_true_ids != reference_true_ids))
        raise ValueError(
            f"Canonical reference row order is not bound to the test manifest "
            f"({mismatch} truth mismatches): {context}"
        )

    canonical = manifest[list(identity)].copy()
    canonical["_canonical_y_true_id"] = reference_true_ids
    canonical["_canonical_y_pred_id"] = _main_class_ids(
        reference["y_pred"], context=f"reference predictions {context}"
    )
    for label in MAIN_LABELS:
        canonical[f"_canonical_prob_{label}"] = pd.to_numeric(
            reference[f"prob_{label}"], errors="raise"
        ).to_numpy(dtype=np.float64)

    working = _normalise_identity_columns(reproduced)
    if working[list(identity)].duplicated().any():
        raise ValueError(f"Duplicate reproduced identities: {context}")
    working["_original_order"] = np.arange(len(working), dtype=np.int64)
    merged = working.merge(
        canonical,
        on=list(identity),
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"Canonical identity join is incomplete: {context}")
    merged = merged.sort_values("_original_order", kind="stable").reset_index(drop=True)
    reproduced_true_ids = _main_class_ids(
        merged["y_true"], context=f"reproduced truth {context}"
    )
    reproduced_pred_ids = _main_class_ids(
        merged["raw_softmax_pred"], context=f"reproduced prediction {context}"
    )
    truth_mismatch = int(
        np.sum(reproduced_true_ids != merged["_canonical_y_true_id"].to_numpy(int))
    )
    prediction_mismatch = int(
        np.sum(reproduced_pred_ids != merged["_canonical_y_pred_id"].to_numpy(int))
    )
    if truth_mismatch or prediction_mismatch:
        raise ValueError(
            f"Canonical/reproduced label mismatch after identity join: "
            f"truth={truth_mismatch}, prediction={prediction_mismatch} ({context})"
        )

    canonical_probabilities = np.column_stack(
        [merged[f"_canonical_prob_{label}"].to_numpy(float) for label in MAIN_LABELS]
    )
    reproduced_probabilities = merged[reproduced_probability_columns].to_numpy(float)
    if not np.isfinite(canonical_probabilities).all():
        raise ValueError(f"Non-finite canonical probabilities: {context}")
    reference_order = np.argsort(-canonical_probabilities, axis=1, kind="stable")
    reproduced_order = np.argsort(-reproduced_probabilities, axis=1, kind="stable")
    rank_mismatch = int(np.any(reference_order != reproduced_order, axis=1).sum())
    max_abs_diff = float(np.max(np.abs(canonical_probabilities - reproduced_probabilities)))
    for index, label in enumerate(MAIN_LABELS):
        merged[f"raw_softmax_prob_{label}"] = canonical_probabilities[:, index]
    helper_columns = [
        column
        for column in merged.columns
        if column.startswith("_canonical_") or column in {"_original_order", "_merge"}
    ]
    aligned = merged.drop(columns=helper_columns)
    return aligned, {
        "identity_join": "one_to_one",
        "manifest_row_binding": "verified_by_true_class",
        "truth_mismatch_count": truth_mismatch,
        "prediction_mismatch_count": prediction_mismatch,
        "rank_order_mismatch_count": rank_mismatch,
        "maximum_probability_abs_drift": max_abs_diff,
        "n": len(aligned),
    }


def refuse_existing(paths: Iterable[Path]) -> None:
    """Refuse a write batch if any planned output already exists."""

    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {formatted}")


def _lambda_column(candidate: float, statistic: str) -> str:
    label = str(candidate).replace(".", "_")
    return f"lambda_{label}_{statistic}"


def select_lambda_by_fold(
    records: pd.DataFrame,
    tie_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Apply the locked validation-only fold-wise lambda selection rule."""

    required = {
        "fold",
        "seed",
        "lambda",
        "best_val_macro_f1",
        "source_path",
        "sha256",
    }
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"Lambda-selection records are missing columns: {missing}")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    metrics = records["best_val_macro_f1"].to_numpy(dtype=np.float64)
    if not np.isfinite(metrics).all():
        raise ValueError("Lambda-selection records contain non-finite validation metrics")
    if records[["fold", "seed", "lambda"]].duplicated().any():
        raise ValueError("Lambda-selection records contain duplicate fold-seed-lambda rows")

    output_rows: list[dict[str, object]] = []
    selected: dict[int, float] = {}
    for fold in sorted(int(value) for value in records["fold"].unique()):
        fold_frame = records[records["fold"] == fold]
        actual_candidates = tuple(sorted(float(value) for value in fold_frame["lambda"].unique()))
        if actual_candidates != CANDIDATE_LAMBDAS:
            raise ValueError(
                f"Fold {fold} candidates must be exactly {CANDIDATE_LAMBDAS}; "
                f"got {actual_candidates}"
            )

        summaries: dict[float, tuple[float, float]] = {}
        for candidate in CANDIDATE_LAMBDAS:
            candidate_frame = fold_frame[np.isclose(fold_frame["lambda"], candidate)]
            actual_seeds = tuple(sorted(int(value) for value in candidate_frame["seed"]))
            if actual_seeds != tuple(sorted(EXPECTED_SEEDS)):
                raise ValueError(
                    f"Fold {fold}, lambda {candidate} must contain exactly five seeds "
                    f"{EXPECTED_SEEDS}; got {actual_seeds}"
                )
            values = candidate_frame["best_val_macro_f1"].to_numpy(dtype=np.float64)
            summaries[candidate] = (float(values.mean()), float(values.std(ddof=1)))

        highest_mean = max(mean for mean, _ in summaries.values())
        mean_tied = [
            candidate
            for candidate, (mean, _) in summaries.items()
            if highest_mean - mean <= tie_tolerance
        ]
        if len(mean_tied) == 1:
            winner = mean_tied[0]
            reason = "highest_validation_mean"
        else:
            lowest_sd = min(summaries[candidate][1] for candidate in mean_tied)
            sd_tied = [
                candidate
                for candidate in mean_tied
                if np.isclose(
                    summaries[candidate][1], lowest_sd, atol=1e-12, rtol=0.0
                )
            ]
            if len(sd_tied) == 1:
                winner = sd_tied[0]
                reason = (
                    f"validation_mean_tie_within_{tie_tolerance:g}_then_lowest_sd"
                )
            else:
                winner = min(sd_tied)
                reason = "validation_mean_and_sd_tie_then_smallest_lambda"

        row: dict[str, object] = {
            "fold": fold,
            "selected_lambda": winner,
            "selection_reason": reason,
            "test_metrics_used_for_selection": False,
        }
        for candidate, (mean, sd) in summaries.items():
            row[_lambda_column(candidate, "mean_best_val_macro_f1")] = mean
            row[_lambda_column(candidate, "sd_best_val_macro_f1")] = sd
        output_rows.append(row)
        selected[fold] = winner

    return pd.DataFrame(output_rows).sort_values("fold").reset_index(drop=True), selected


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm family-wise adjusted P values in input order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p_values must all be finite and lie in [0, 1]")

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    factors = values.size - np.arange(values.size)
    adjusted_ordered = np.maximum.accumulate(ordered * factors)
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


def _bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be a finite non-empty vector")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_boot, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _wilcoxon_p(delta: np.ndarray) -> float:
    if np.all(np.isclose(delta, 0.0, atol=1e-12, rtol=0.0)):
        return 1.0
    return float(wilcoxon(delta, alternative="two-sided").pvalue)


def paired_statistics(
    runs: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute strict run-paired and fold-cluster statistics."""

    required = {"fold", "seed"}
    for final_stage, baseline_stage in comparisons:
        required.update((final_stage, baseline_stage))
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"Paired-statistics input is missing columns: {missing}")
    if runs[["fold", "seed"]].duplicated().any():
        raise ValueError("Paired statistics require unique fold-seed rows")
    ordered = runs.sort_values(["fold", "seed"], kind="stable").reset_index(drop=True)
    actual_keys = tuple(map(tuple, ordered[["fold", "seed"]].to_numpy()))
    if actual_keys != EXPECTED_KEYS:
        raise ValueError(
            "Paired statistics require the exact 25 fold-seed grid "
            f"{EXPECTED_KEYS}; got {actual_keys}"
        )
    if not comparisons:
        raise ValueError("At least one comparison is required")

    paired_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for final_stage, baseline_stage in comparisons:
        final = ordered[final_stage].to_numpy(dtype=np.float64)
        baseline = ordered[baseline_stage].to_numpy(dtype=np.float64)
        if not (np.isfinite(final).all() and np.isfinite(baseline).all()):
            raise ValueError(
                f"Comparison {final_stage} - {baseline_stage} contains non-finite values"
            )
        delta = final - baseline
        run_low, run_high = _bootstrap_mean_ci(delta, n_boot=n_boot, seed=seed)
        ties = np.isclose(delta, 0.0, atol=1e-12, rtol=0.0)
        fold_means: dict[int, float] = {}
        comparison = f"{final_stage} - {baseline_stage}"
        for fold in EXPECTED_FOLDS:
            mask = ordered["fold"].to_numpy(dtype=int) == fold
            fold_delta = delta[mask]
            if fold_delta.size != len(EXPECTED_SEEDS):
                raise ValueError(
                    f"Comparison {comparison}, fold {fold} does not have five seeds"
                )
            fold_ties = np.isclose(fold_delta, 0.0, atol=1e-12, rtol=0.0)
            fold_mean = float(fold_delta.mean())
            fold_means[fold] = fold_mean
            fold_rows.append(
                {
                    "comparison": comparison,
                    "final_stage": final_stage,
                    "baseline_stage": baseline_stage,
                    "fold": fold,
                    "n_seeds": int(fold_delta.size),
                    "baseline_mean": float(baseline[mask].mean()),
                    "final_mean": float(final[mask].mean()),
                    "mean_delta": fold_mean,
                    "std_delta": float(fold_delta.std(ddof=1)),
                    "wins": int(((fold_delta > 0) & ~fold_ties).sum()),
                    "ties": int(fold_ties.sum()),
                    "losses": int(((fold_delta < 0) & ~fold_ties).sum()),
                }
            )
        fold_values = np.asarray(
            [fold_means[fold] for fold in EXPECTED_FOLDS], dtype=np.float64
        )
        fold_low, fold_high = _bootstrap_mean_ci(
            fold_values, n_boot=n_boot, seed=seed
        )
        row: dict[str, object] = {
            "comparison": comparison,
            "final_stage": final_stage,
            "baseline_stage": baseline_stage,
            "n": int(delta.size),
            "baseline_mean": float(baseline.mean()),
            "baseline_sd": float(baseline.std(ddof=1)),
            "final_mean": float(final.mean()),
            "final_sd": float(final.std(ddof=1)),
            "mean_delta": float(delta.mean()),
            "std_delta": float(delta.std(ddof=1)),
            "bootstrap_ci95_low": run_low,
            "bootstrap_ci95_high": run_high,
            "wilcoxon_raw_p": _wilcoxon_p(delta),
            "wins": int(((delta > 0) & ~ties).sum()),
            "ties": int(ties.sum()),
            "losses": int(((delta < 0) & ~ties).sum()),
            "fold_cluster_n": len(EXPECTED_FOLDS),
            "fold_cluster_bootstrap_ci95_low": fold_low,
            "fold_cluster_bootstrap_ci95_high": fold_high,
        }
        row.update(
            {
                f"fold_{fold}_mean_delta": fold_means[fold]
                for fold in EXPECTED_FOLDS
            }
        )
        paired_rows.append(row)

    paired = pd.DataFrame(paired_rows)
    paired["holm_adjusted_p"] = holm_adjust(paired["wilcoxon_raw_p"])
    return paired, pd.DataFrame(fold_rows)


def rank_true_labels(
    values: np.ndarray,
    *,
    true_labels: Sequence[str],
    labels: Sequence[str],
    higher_is_better: bool,
) -> tuple[np.ndarray, int]:
    """Rank each true label with stable fixed-label tie breaking."""

    matrix = np.asarray(values, dtype=np.float64)
    label_order = tuple(str(label) for label in labels)
    truths = tuple(str(label) for label in true_labels)
    if matrix.ndim != 2 or matrix.shape != (len(truths), len(label_order)):
        raise ValueError(
            "Rank matrix shape must be (n_samples, n_labels); "
            f"got {matrix.shape}, expected {(len(truths), len(label_order))}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Rank matrix contains non-finite values")
    label_to_index = {label: index for index, label in enumerate(label_order)}
    unknown = sorted(set(truths) - set(label_order))
    if unknown:
        raise ValueError(f"Unknown true labels for ranking: {unknown}")
    sort_values = -matrix if higher_is_better else matrix
    order = np.argsort(sort_values, axis=1, kind="stable")
    true_indices = np.asarray([label_to_index[label] for label in truths], dtype=int)
    ranks = np.empty(len(truths), dtype=int)
    for row_index, true_index in enumerate(true_indices):
        ranks[row_index] = int(np.flatnonzero(order[row_index] == true_index)[0] + 1)
    tie_rows = int(
        sum(np.unique(matrix[row_index]).size != matrix.shape[1] for row_index in range(matrix.shape[0]))
    )
    return ranks, tie_rows


def favorable_distance_margin(
    distances: np.ndarray,
    *,
    true_labels: Sequence[str],
    labels: Sequence[str],
) -> np.ndarray:
    """Return nearest-wrong distance minus true-prototype distance."""

    matrix = np.asarray(distances, dtype=np.float64)
    label_order = tuple(str(label) for label in labels)
    truths = tuple(str(label) for label in true_labels)
    if matrix.ndim != 2 or matrix.shape != (len(truths), len(label_order)):
        raise ValueError(
            "Distance matrix shape must be (n_samples, n_labels); "
            f"got {matrix.shape}, expected {(len(truths), len(label_order))}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Distance matrix contains non-finite values")
    label_to_index = {label: index for index, label in enumerate(label_order)}
    unknown = sorted(set(truths) - set(label_order))
    if unknown:
        raise ValueError(f"Unknown true labels for distance margin: {unknown}")

    margins = np.empty(len(truths), dtype=np.float64)
    for row_index, truth in enumerate(truths):
        true_index = label_to_index[truth]
        wrong = np.delete(matrix[row_index], true_index)
        margins[row_index] = float(wrong.min() - matrix[row_index, true_index])
    return margins


def safe_error_auroc(
    *, errors: np.ndarray, scores: np.ndarray
) -> tuple[float, str]:
    """Compute per-run error AUROC or return an explicit unavailable status."""

    error_array = np.asarray(errors, dtype=bool)
    score_array = np.asarray(scores, dtype=np.float64)
    if error_array.ndim != 1 or score_array.ndim != 1 or error_array.size != score_array.size:
        raise ValueError("errors and scores must be aligned one-dimensional arrays")
    if not np.isfinite(score_array).all():
        raise ValueError("Error-detection scores contain non-finite values")
    if np.unique(error_array).size < 2:
        return float("nan"), "unavailable_single_error_class"
    return float(roc_auc_score(error_array.astype(int), score_array)), "available"


def _score_matrix(frame: pd.DataFrame, prefix: str, labels: Sequence[str]) -> np.ndarray:
    columns = [f"{prefix}{label}" for label in labels]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction table is missing score columns: {missing}")
    matrix = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Prediction table contains non-finite scores for prefix {prefix}")
    return matrix


def _distance_matrix(frame: pd.DataFrame, level: str, labels: Sequence[str]) -> np.ndarray:
    if level not in {"main", "aux"}:
        raise ValueError(f"Unsupported prototype level: {level}")
    columns = [f"{level}_proto_cosine_distance_{label}" for label in labels]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction table is missing distance columns: {missing}")
    matrix = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Prediction table contains non-finite {level} distances")
    return matrix


def _prediction_labels(scores: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    label_array = np.asarray(tuple(labels), dtype=object)
    return label_array[np.argmax(scores, axis=1)]


def _top2_label_sets(scores: np.ndarray, labels: Sequence[str]) -> list[set[str]]:
    label_array = np.asarray(tuple(labels), dtype=object)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :2]
    return [set(str(value) for value in label_array[row]) for row in order]


def _quantiles(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return (float("nan"),) * 4
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return float(values.mean()), float(q1), float(median), float(q3)


def compute_run_functional_metrics(
    predictions: pd.DataFrame,
    *,
    fold: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute every requested prediction-value metric within one run."""

    required = {"path", "source_id", "md5", "y_true", "aux_true"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing identity/truth columns: {missing}")
    if predictions.empty:
        raise ValueError("Prediction table is empty")
    if predictions["source_id"].astype(str).duplicated().any():
        raise ValueError("Prediction table contains duplicate source_id values")

    main_truth = predictions["y_true"].astype(str).to_numpy()
    aux_truth = predictions["aux_true"].astype(str).to_numpy()
    main_distances = _distance_matrix(predictions, "main", MAIN_LABELS)
    aux_distances = _distance_matrix(predictions, "aux", AUX_LABELS)
    main_margin = favorable_distance_margin(
        main_distances, true_labels=main_truth, labels=MAIN_LABELS
    )
    aux_margin = favorable_distance_margin(
        aux_distances, true_labels=aux_truth, labels=AUX_LABELS
    )
    main_proto_ranks, main_distance_ties = rank_true_labels(
        main_distances,
        true_labels=main_truth,
        labels=MAIN_LABELS,
        higher_is_better=False,
    )
    aux_proto_ranks, aux_distance_ties = rank_true_labels(
        aux_distances,
        true_labels=aux_truth,
        labels=AUX_LABELS,
        higher_is_better=False,
    )

    methods = {
        "raw_softmax": ("raw_softmax_prob_", "softmax_aux_prob_"),
        "main_prototype": ("prototype_prob_", "prototype_aux_prob_"),
        "hierarchical_prototype": (
            "hierarchical_prob_",
            "prototype_aux_prob_",
        ),
    }
    functional_rows: list[dict[str, object]] = []
    margin_rows: list[dict[str, object]] = []
    predictions_by_method: dict[str, np.ndarray] = {}

    def add_metric(
        method: str,
        metric: str,
        value: float,
        *,
        n_samples: int,
        availability: str = "available",
    ) -> None:
        functional_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "method": method,
                "metric": metric,
                "value": value,
                "n_samples": n_samples,
                "availability": availability,
            }
        )

    for method, (main_prefix, aux_prefix) in methods.items():
        main_scores = _score_matrix(predictions, main_prefix, MAIN_LABELS)
        aux_scores = _score_matrix(predictions, aux_prefix, AUX_LABELS)
        main_pred = _prediction_labels(main_scores, MAIN_LABELS)
        aux_pred = _prediction_labels(aux_scores, AUX_LABELS)
        predictions_by_method[method] = main_pred
        main_ranks, main_score_ties = rank_true_labels(
            main_scores,
            true_labels=main_truth,
            labels=MAIN_LABELS,
            higher_is_better=True,
        )
        aux_ranks, aux_score_ties = rank_true_labels(
            aux_scores,
            true_labels=aux_truth,
            labels=AUX_LABELS,
            higher_is_better=True,
        )
        main_error = main_pred != main_truth
        aux_error = aux_pred != aux_truth
        mapped_aux = np.asarray([AUX_TO_MAIN[label] for label in aux_pred], dtype=object)
        consistent = main_pred == mapped_aux

        add_metric(method, "main_top1_accuracy", float((main_ranks == 1).mean()), n_samples=len(main_ranks))
        add_metric(method, "main_top2_accuracy", float((main_ranks <= 2).mean()), n_samples=len(main_ranks))
        add_metric(method, "true_main_candidate_mean_rank", float(main_ranks.mean()), n_samples=len(main_ranks))
        add_metric(method, "true_main_candidate_mrr", float((1.0 / main_ranks).mean()), n_samples=len(main_ranks))
        add_metric(method, "main_score_tie_row_rate", float(main_score_ties / len(main_ranks)), n_samples=len(main_ranks))
        add_metric(method, "subtype_top1_accuracy", float((aux_ranks == 1).mean()), n_samples=len(aux_ranks))
        add_metric(method, "subtype_top2_accuracy", float((aux_ranks <= 2).mean()), n_samples=len(aux_ranks))
        add_metric(method, "true_subtype_candidate_mean_rank", float(aux_ranks.mean()), n_samples=len(aux_ranks))
        add_metric(method, "true_subtype_candidate_mrr", float((1.0 / aux_ranks).mean()), n_samples=len(aux_ranks))
        add_metric(method, "subtype_score_tie_row_rate", float(aux_score_ties / len(aux_ranks)), n_samples=len(aux_ranks))
        add_metric(method, "hierarchy_consistency_rate", float(consistent.mean()), n_samples=len(consistent))
        for consistency_value, suffix in ((True, "consistent"), (False, "inconsistent")):
            subset = consistent == consistency_value
            if subset.any():
                value = float(main_error[subset].mean())
                availability = "available"
            else:
                value = float("nan")
                availability = "unavailable_zero_denominator"
            add_metric(
                method,
                f"classification_error_rate_hierarchy_{suffix}",
                value,
                n_samples=int(subset.sum()),
                availability=availability,
            )
        add_metric(method, "hierarchy_consistent_correct_rate", float((consistent & ~main_error).mean()), n_samples=len(consistent))
        add_metric(method, "hierarchy_consistent_wrong_rate", float((consistent & main_error).mean()), n_samples=len(consistent))
        add_metric(method, "hierarchy_inconsistent_correct_rate", float((~consistent & ~main_error).mean()), n_samples=len(consistent))
        add_metric(method, "hierarchy_inconsistent_wrong_rate", float((~consistent & main_error).mean()), n_samples=len(consistent))

        boundary = np.isin(main_truth, ["feeding", "stress_vocal"])
        if boundary.any():
            top2_sets = _top2_label_sets(main_scores, MAIN_LABELS)
            exact_pair = np.asarray(
                [labels == {"feeding", "stress_vocal"} for labels in top2_sets],
                dtype=bool,
            )
            true_in_top2 = main_ranks <= 2
            add_metric(
                method,
                "feeding_stress_exact_pair_top2_coverage",
                float(exact_pair[boundary].mean()),
                n_samples=int(boundary.sum()),
            )
            add_metric(
                method,
                "feeding_stress_true_class_top2_coverage",
                float(true_in_top2[boundary].mean()),
                n_samples=int(boundary.sum()),
            )
        else:
            for metric in (
                "feeding_stress_exact_pair_top2_coverage",
                "feeding_stress_true_class_top2_coverage",
            ):
                add_metric(
                    method,
                    metric,
                    float("nan"),
                    n_samples=0,
                    availability="unavailable_zero_denominator",
                )

        for level, errors, margins in (
            ("main", main_error, main_margin),
            ("subtype", aux_error, aux_margin),
        ):
            correct_values = margins[~errors]
            error_values = margins[errors]
            correct_mean, correct_q1, correct_median, correct_q3 = _quantiles(correct_values)
            error_mean, error_q1, error_median, error_q3 = _quantiles(error_values)
            auroc, auroc_status = safe_error_auroc(errors=errors, scores=-margins)
            margin_rows.append(
                {
                    "record_type": "run",
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    "level": level,
                    "n": int(len(margins)),
                    "n_correct": int((~errors).sum()),
                    "n_error": int(errors.sum()),
                    "correct_margin_mean": correct_mean,
                    "correct_margin_q1": correct_q1,
                    "correct_margin_median": correct_median,
                    "correct_margin_q3": correct_q3,
                    "error_margin_mean": error_mean,
                    "error_margin_q1": error_q1,
                    "error_margin_median": error_median,
                    "error_margin_q3": error_q3,
                    "error_detection_auroc": auroc,
                    "auroc_status": auroc_status,
                    "error_score": "negative_favorable_distance_margin",
                }
            )

    geometry_metrics = {
        "true_main_prototype_mean_rank": float(main_proto_ranks.mean()),
        "true_main_prototype_rank1_rate": float((main_proto_ranks == 1).mean()),
        "true_main_prototype_rank2_rate": float((main_proto_ranks <= 2).mean()),
        "true_main_prototype_mrr": float((1.0 / main_proto_ranks).mean()),
        "true_main_prototype_tie_row_rate": float(main_distance_ties / len(main_proto_ranks)),
        "true_subtype_prototype_mean_rank": float(aux_proto_ranks.mean()),
        "true_subtype_prototype_rank1_rate": float((aux_proto_ranks == 1).mean()),
        "true_subtype_prototype_rank2_rate": float((aux_proto_ranks <= 2).mean()),
        "true_subtype_prototype_mrr": float((1.0 / aux_proto_ranks).mean()),
        "true_subtype_prototype_tie_row_rate": float(aux_distance_ties / len(aux_proto_ranks)),
    }
    for metric, value in geometry_metrics.items():
        add_metric(
            "prototype_geometry",
            metric,
            value,
            n_samples=len(predictions),
        )

    raw_pred = predictions_by_method["raw_softmax"]
    raw_correct = raw_pred == main_truth
    disagreement_rows: list[dict[str, object]] = []
    for method in ("main_prototype", "hierarchical_prototype"):
        prototype_pred = predictions_by_method[method]
        prototype_correct = prototype_pred == main_truth
        disagree = prototype_pred != raw_pred
        disagreement_rows.append(
            {
                "record_type": "run",
                "fold": fold,
                "seed": seed,
                "prototype_method": method,
                "n": int(len(disagree)),
                "n_disagreements": int(disagree.sum()),
                "disagreement_rate": float(disagree.mean()),
                "raw_correct_prototype_wrong": int(
                    (disagree & raw_correct & ~prototype_correct).sum()
                ),
                "prototype_correct_raw_wrong": int(
                    (disagree & prototype_correct & ~raw_correct).sum()
                ),
                "both_wrong": int(
                    (disagree & ~raw_correct & ~prototype_correct).sum()
                ),
                "both_correct": int(
                    (disagree & raw_correct & prototype_correct).sum()
                ),
                "unchanged_correct": int((~disagree & raw_correct).sum()),
                "unchanged_wrong": int(
                    (~disagree & ~raw_correct & ~prototype_correct).sum()
                ),
            }
        )

    return (
        pd.DataFrame(functional_rows),
        pd.DataFrame(margin_rows),
        pd.DataFrame(disagreement_rows),
    )


if __name__ == "__main__":
    main()
