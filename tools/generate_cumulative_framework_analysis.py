"""Generate the locked cumulative framework analysis without model inference.

The script reads only existing 5-fold x 5-seed clean result artifacts. It
validates the B0-B3 cohort, computes direct matched statistics, selects seven
deterministic cases from one canonical fold-seed prediction artifact, and
writes only the explicitly named paper tables and figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import textwrap
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import wilcoxon

try:  # Package import under unittest.
    from tools.nomenclature import (
        METHOD_METADATA_FIELDS,
        build_method_metadata,
        canonicalize_inference_route,
        canonicalize_model_family,
        canonicalize_selection_protocol,
        canonicalize_training_stage,
        display_label,
        reconcile_selection_protocol,
        resolve_inference_method,
        validate_recorded_nomenclature_metadata,
    )
except ModuleNotFoundError:  # Direct ``python tools/<script>.py`` execution.
    from nomenclature import (  # type: ignore[no-redef]
        METHOD_METADATA_FIELDS,
        build_method_metadata,
        canonicalize_inference_route,
        canonicalize_model_family,
        canonicalize_selection_protocol,
        canonicalize_training_stage,
        display_label,
        reconcile_selection_protocol,
        resolve_inference_method,
        validate_recorded_nomenclature_metadata,
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


EXPECTED_FOLDS = tuple(range(5))
EXPECTED_SEEDS = (42, 123, 777, 2024, 3407)
EXPECTED_KEYS = tuple((fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS)
EXPECTED_LAMBDA = 0.5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 3407

STAGE_ORDER = ("B0", "B1", "B2", "B3")

_STAGE_SPECS = {
    "B0": {
        "model_family": "logmel_crnn",
        "training_stage": "b0_1s_logmel_baseline",
        "context_seconds": 1.0,
        "selection_protocol": "none",
        "inference_route": "primary_softmax",
        "legacy_method_id": "raw_softmax",
    },
    "B1": {
        "model_family": "logmel_crnn",
        "training_stage": "b1_2s_logmel_mainline",
        "context_seconds": 2.0,
        "selection_protocol": "none",
        "inference_route": "primary_softmax",
        "legacy_method_id": "raw_softmax",
    },
    "B2": {
        "model_family": "hierarchical_supervision_crnn",
        "training_stage": "b2_validation_selected_hierarchical_crnn",
        "context_seconds": 2.0,
        "selection_protocol": "fixed_lambda_0_5_retrospective",
        "inference_route": "primary_softmax",
        "legacy_method_id": "raw_softmax",
    },
    "B3": {
        "model_family": "hierarchical_supervision_crnn",
        "training_stage": "b3_hierarchical_prototype_top1_ablation",
        "context_seconds": 2.0,
        "selection_protocol": "fixed_lambda_0_5_retrospective",
        "inference_route": "hierarchical_prototype_candidate",
        "legacy_method_id": "hierarchical",
    },
}
_STAGE_BY_TRAINING_ID = {
    str(spec["training_stage"]): stage for stage, spec in _STAGE_SPECS.items()
}


def canonical_metadata_for_stage(stage: str) -> dict[str, object]:
    """Return fixed-cohort metadata while retaining the B0-B3 machine key."""

    try:
        training_stage = canonicalize_training_stage(stage)
        stage_key = _STAGE_BY_TRAINING_ID[training_stage]
    except (KeyError, ValueError) as exc:
        allowed = ", ".join((*STAGE_ORDER, *_STAGE_BY_TRAINING_ID))
        raise ValueError(
            f"Unknown cumulative framework stage {stage!r}; expected one of: {allowed}."
        ) from exc
    return build_method_metadata(**_STAGE_SPECS[stage_key])


def _validate_fixed_selection_protocol(
    records: Sequence[tuple[str, Mapping[str, Any]]],
) -> str:
    """Reject concrete source provenance outside the locked fixed cohort."""

    concrete: dict[str, str] = {}
    for name, record in records:
        try:
            validate_recorded_nomenclature_metadata(record)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
        value = record.get("selection_protocol")
        if value is None or not str(value).strip():
            continue
        protocol = canonicalize_selection_protocol(str(value))
        if protocol != "none":
            concrete[name] = protocol
    if len(set(concrete.values())) > 1:
        detail = ", ".join(
            f"{name}={protocol!r}" for name, protocol in concrete.items()
        )
        raise ValueError(
            f"Conflicting selection_protocol provenance: {detail}"
        )
    recorded = next(iter(concrete.values()), "none")
    return reconcile_selection_protocol(
        recorded,
        requested="fixed_lambda_0_5_retrospective",
    )


_PRIMARY_SOFTMAX_LABEL = display_label(
    "inference_route", "primary_softmax", language="en"
)
_MAIN_PROTOTYPE_LABEL = display_label(
    "inference_route", "main_class_prototype_candidate", language="en"
)
_HIERARCHICAL_PROTOTYPE_LABEL = display_label(
    "inference_route", "hierarchical_prototype_candidate", language="en"
)
_FIXED_PROTOCOL_LABEL = display_label(
    "selection_protocol", "fixed_lambda_0_5_retrospective", language="en"
)

STAGE_DESCRIPTIONS = {
    stage: display_label(
        "training_stage", str(spec["training_stage"]), language="en"
    )
    for stage, spec in _STAGE_SPECS.items()
}
STAGE_DESCRIPTIONS["B2"] = "B2 — 2-s hierarchical-supervision CRNN"
STAGE_LABELS = {
    stage: "\n".join(
        textwrap.wrap(
            description,
            width=28,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )
    + (f"\n{_FIXED_PROTOCOL_LABEL}" if stage in {"B2", "B3"} else "")
    for stage, description in STAGE_DESCRIPTIONS.items()
}
COMPARISONS = (
    ("B1", "B0"),
    ("B2", "B1"),
    ("B3", "B2"),
    ("B3", "B1"),
    ("B3", "B0"),
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

FIGURE_WIDTH_MM = 183.0
CUMULATIVE_HEIGHT_MM = 158.0
CASE_HEIGHT_MM = 168.0

STAGE_COLORS = {
    "B0": "#555D6B",
    "B1": "#6D8FA3",
    "B2": "#8A9D70",
    "B3": "#2E6F8E",
}
TEXT = "#20242A"
MUTED = "#69717E"
GRID = "#D9DEE5"
PALE_BLUE = "#DDEAF0"
PALE_GREEN = "#E3EAD9"
PALE_GOLD = "#EEE6D1"
ACCENT = "#2E6F8E"
FOLD_COLOR = "#8A9099"
WHITE = "#FFFFFF"

TABLE_RELATIVE_PATHS = (
    "paper/tables/cumulative_framework_runs_nomenclature_v1.csv",
    "paper/tables/cumulative_framework_summary_nomenclature_v1.csv",
    "paper/tables/cumulative_framework_paired_stats_nomenclature_v1.csv",
    "paper/tables/cumulative_framework_fold_stats_nomenclature_v1.csv",
    "paper/tables/prototype_prediction_case_studies_nomenclature_v1.csv",
)
FIGURE_STEMS = (
    "paper/figures_journal/figure_cumulative_framework_nomenclature_v1",
    "paper/figures_journal/figure_prototype_prediction_cases_nomenclature_v1",
)
ARCHIVE_TABLE_RELATIVE_PATHS = tuple(
    relative.replace("paper/tables/", "paper_results/tables/")
    for relative in TABLE_RELATIVE_PATHS
)
ARCHIVE_SCRIPT_RELATIVE_PATH = (
    "paper_results/scripts/"
    "generate_cumulative_framework_analysis_nomenclature_v1.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the locked B0-B3 cumulative Macro-F1 analysis and "
            "deterministic prototype case studies from existing 25-run artifacts. "
            "No training or model inference is run."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current working directory).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
        help="Percentile-bootstrap resamples (paper default: 10000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
        help="Bootstrap random seed (paper default: 3407).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all sources and build in-memory outputs without writing files.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"BLOCKER: required frozen source is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"BLOCKER: invalid JSON in frozen source {path}: {exc}") from exc


def _require_csv(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"BLOCKER: required frozen source is missing: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"BLOCKER: {path} is missing required columns: {missing}")
    return frame


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot interpret value as boolean: {value!r}")


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"BLOCKER: required frozen source is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_relative_path(value: Any) -> str:
    return str(value).replace("\\", "/")


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _summary_path(root: Path, stage: str, fold: int, seed: int) -> Path:
    if stage == "B0":
        name = f"cv5_expanded_cap3x_fold{fold}_logmel_seed{seed}"
    elif stage == "B1":
        name = f"cv5_expanded_cap3x_fold{fold}_logmel_dur2_seed{seed}"
    elif stage == "B2":
        name = f"cv5_expanded_cap3x_fold{fold}_logmel_dur2_hier_w05_seed{seed}"
    else:
        raise ValueError(f"Unsupported summary stage: {stage}")
    return root / "reports" / name / "summary.json"


def _prototype_run_dir(root: Path, fold: int, seed: int) -> Path:
    return root / "reports" / f"prototype_cv5_exact_w05_fold{fold}_seed{seed}"


def _manifest_path(root: Path, fold: int, split: str) -> Path:
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


def _validate_prototype_metadata(
    payload: dict[str, Any],
    *,
    path: Path,
    root: Path,
    fold: int,
    seed: int,
    summary_path: Path,
) -> None:
    """Bind a prototype bundle to the exact matched B2 fold-seed artifact."""

    try:
        validate_recorded_nomenclature_metadata(payload)
    except ValueError as exc:
        raise ValueError(
            f"Invalid prototype nomenclature metadata for {path}: {exc}"
        ) from exc

    model = payload.get("model_config", {})
    inferred = payload.get("inferred_fold_seed", {})
    relatives = payload.get("project_relative_paths", {})
    nested_paths = payload.get("paths", {})
    input_hashes = payload.get("input_hashes", {})
    manifest_hashes = payload.get("manifest_sha256", {})

    expected_summary = _relative(root, summary_path)
    expected_checkpoint = (
        f"checkpoints/cv5_expanded_cap3x_fold{fold}_logmel_dur2_hier_w05_seed{seed}.pt"
    )
    expected_manifests = {
        split: _relative(root, _manifest_path(root, fold, split))
        for split in ("train", "val", "test")
    }
    actual_summary_sha = _sha256_file(summary_path)
    summary_payload = _read_json(summary_path)
    checkpoint_sha = payload.get("checkpoint_sha256")

    checks = {
        "artifact_type=hier_acoustic_prototype_bundle": payload.get("artifact_type")
        == "hier_acoustic_prototype_bundle",
        f"fold={fold}": int(payload.get("fold", -1)) == fold,
        f"seed={seed}": int(payload.get("seed", -1)) == seed,
        "inferred fold/seed match": int(inferred.get("fold", -1)) == fold
        and int(inferred.get("seed", -1)) == seed,
        "run_scope=fold_seed": payload.get("run_scope") == "fold_seed",
        "feature_backend=training_exact": payload.get("feature_backend") == "training_exact",
        "feature_pipeline_equivalent=true": payload.get("feature_pipeline_equivalent") is True,
        "eligible_for_cv_aggregation=true": payload.get("eligible_for_cv_aggregation") is True,
        "dur_s=2.0": math.isclose(float(payload.get("dur_s", -1.0)), 2.0, abs_tol=1e-12),
        "hier_aux_weight=0.5": math.isclose(
            float(payload.get("hier_aux_weight", -1.0)), EXPECTED_LAMBDA, abs_tol=1e-12
        ),
        "model feature_mode=logmel": model.get("feature_mode") == "logmel",
        "model use_se=true": model.get("use_se") is True,
        "model seed match": int(model.get("seed", -1)) == seed,
        "model dur_s=2.0": math.isclose(float(model.get("dur_s", -1.0)), 2.0, abs_tol=1e-12),
        "model hier_aux_weight=0.5": math.isclose(
            float(model.get("hier_aux_weight", -1.0)), EXPECTED_LAMBDA, abs_tol=1e-12
        ),
        "prototype_source_split=train": payload.get("prototype_source_split") == "train",
        "calibration_source_split=validation": payload.get("calibration_source_split")
        == "validation",
        "evaluation_source_split=test": payload.get("evaluation_source_split") == "test",
        "checkpoint path match": _normalise_relative_path(relatives.get("checkpoint"))
        == expected_checkpoint,
        "summary path match": _normalise_relative_path(relatives.get("checkpoint_summary"))
        == expected_summary,
        "summary SHA-256 match": payload.get("checkpoint_summary_sha256")
        == actual_summary_sha,
        "checkpoint SHA-256 present": _is_sha256(checkpoint_sha),
        "top-level main labels match": tuple(payload.get("main_labels", ())) == MAIN_LABELS,
        "top-level auxiliary labels match": tuple(payload.get("aux_labels", ())) == AUX_LABELS,
    }

    model_identity_fields = (
        "feature_mode",
        "use_se",
        "seed",
        "dur_s",
        "n_mels",
        "n_mfcc",
        "fmin",
        "fmax",
        "n_fft",
        "hop_length",
        "win_length",
        "sr",
        "rnn_type",
        "pooling_type",
        "hier_aux_weight",
        "main_labels",
        "aux_labels",
    )
    for field in model_identity_fields:
        checks[f"model_config {field} matches B2 summary"] = model.get(field) == summary_payload.get(
            field
        )

    checkpoint_path_entry = nested_paths.get("checkpoint", {})
    summary_path_entry = nested_paths.get("checkpoint_summary", {})
    checks.update(
        {
            "nested checkpoint path match": _normalise_relative_path(
                checkpoint_path_entry.get("project_relative")
            )
            == expected_checkpoint,
            "nested checkpoint SHA-256 match": checkpoint_path_entry.get("sha256")
            == checkpoint_sha,
            "input checkpoint SHA-256 match": input_hashes.get("checkpoint", {}).get("sha256")
            == checkpoint_sha,
            "nested summary path match": _normalise_relative_path(
                summary_path_entry.get("project_relative")
            )
            == expected_summary,
            "nested summary SHA-256 match": summary_path_entry.get("sha256")
            == actual_summary_sha,
            "input summary SHA-256 match": input_hashes.get("checkpoint_summary", {}).get(
                "sha256"
            )
            == actual_summary_sha,
        }
    )
    for split, expected_manifest in expected_manifests.items():
        key = f"{split}_manifest"
        manifest_sha = manifest_hashes.get(split)
        actual_manifest_sha = _sha256_file(_manifest_path(root, fold, split))
        nested = nested_paths.get(key, {})
        checks.update(
            {
                f"{split} manifest path match": _normalise_relative_path(relatives.get(key))
                == expected_manifest,
                f"nested {split} manifest path match": _normalise_relative_path(
                    nested.get("project_relative")
                )
                == expected_manifest,
                f"{split} manifest SHA-256 present": _is_sha256(manifest_sha),
                f"{split} manifest SHA-256 match": manifest_sha == actual_manifest_sha,
                f"nested {split} manifest SHA-256 match": nested.get("sha256")
                == manifest_sha,
                f"input {split} manifest SHA-256 match": input_hashes.get(key, {}).get(
                    "sha256"
                )
                == manifest_sha,
            }
        )

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"BLOCKER: prototype metadata identity failed for {path}: {failed}")


def _validate_stage_summary(
    payload: dict[str, Any],
    *,
    path: Path,
    stage: str,
    seed: int,
) -> float:
    try:
        validate_recorded_nomenclature_metadata(payload)
    except ValueError as exc:
        raise ValueError(
            f"Invalid stage-summary nomenclature metadata for {path}: {exc}"
        ) from exc
    expected_duration = 1.0 if stage == "B0" else 2.0
    recorded_context = payload.get("context_seconds")
    context_matches = (
        recorded_context is None
        or not str(recorded_context).strip()
        or math.isclose(
            float(recorded_context), expected_duration, abs_tol=1e-12
        )
    )
    recorded_family = payload.get("model_family")
    expected_family = (
        "hierarchical_supervision_crnn" if stage == "B2" else "logmel_crnn"
    )
    if recorded_family is None or not str(recorded_family).strip():
        family_matches = stage != "B2"
    else:
        try:
            family_matches = (
                canonicalize_model_family(str(recorded_family))
                == expected_family
            )
        except ValueError:
            family_matches = False
    checks = {
        f"model_family={expected_family}": family_matches,
        "feature_mode=logmel": payload.get("feature_mode") == "logmel",
        "use_se=true": payload.get("use_se") is True,
        f"seed={seed}": int(payload.get("seed", -1)) == seed,
        f"dur_s={expected_duration}": math.isclose(
            float(payload.get("dur_s", -1.0)), expected_duration, abs_tol=1e-12
        ),
        f"context_seconds={expected_duration} when present": context_matches,
    }
    if stage == "B2":
        checks.update(
            {
                "hier_aux=true": payload.get("hier_aux") is True,
                "hier_aux_weight=0.5": math.isclose(
                    float(payload.get("hier_aux_weight", -1.0)), EXPECTED_LAMBDA, abs_tol=1e-12
                ),
                "rnn_type=gru": payload.get("rnn_type") == "gru",
                "pooling_type=mean": payload.get("pooling_type") == "mean",
            }
        )
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"BLOCKER: stage identity failed for {path}: {failed}")
    metric = payload.get("test_macro_f1")
    if metric is None or not np.isfinite(float(metric)):
        raise ValueError(f"BLOCKER: missing/non-finite test_macro_f1 in {path}")
    return float(metric)


def _method_payload_for_route(
    methods: Any,
    *,
    route: str,
    field: str,
    path: Path,
) -> dict[str, Any]:
    """Resolve legacy or canonical method keys without rewriting source data."""

    canonical_route = canonicalize_inference_route(route)
    if not isinstance(methods, dict):
        raise ValueError(f"BLOCKER: {field} must be an object in {path}")
    matches: list[tuple[str, dict[str, Any]]] = []
    for method_name, method_payload in methods.items():
        _, resolved_route = resolve_inference_method(str(method_name))
        if resolved_route != canonical_route:
            continue
        if not isinstance(method_payload, dict):
            raise ValueError(
                f"BLOCKER: {field}[{method_name!r}] must be an object in {path}"
            )
        matches.append((str(method_name), method_payload))
    if not matches:
        raise ValueError(
            f"BLOCKER: {field} has no method for canonical route "
            f"{canonical_route!r} in {path}"
        )
    if len(matches) > 1:
        aliases = [name for name, _ in matches]
        raise ValueError(
            f"BLOCKER: {field} contains duplicate aliases for canonical route "
            f"{canonical_route!r} in {path}: {aliases}"
        )
    return matches[0][1]


def _validate_prototype_metrics(
    payload: dict[str, Any],
    *,
    path: Path,
    fold: int,
    seed: int,
) -> tuple[float, float]:
    selected_route = canonicalize_inference_route(str(payload.get("selection_method", "")))
    best = _method_payload_for_route(
        payload.get("method_best_params", {}),
        route="hierarchical_prototype_candidate",
        field="method_best_params",
        path=path,
    )
    checks = {
        "artifact_type": payload.get("artifact_type") == "hier_acoustic_prototype_test_evaluation",
        f"fold={fold}": int(payload.get("fold", -1)) == fold,
        f"seed={seed}": int(payload.get("seed", -1)) == seed,
        "input_role=frozen_test": payload.get("input_role") == "frozen_test",
        "selection_method=hierarchical_prototype_candidate": (
            selected_route == "hierarchical_prototype_candidate"
        ),
        "feature_backend=training_exact": payload.get("feature_backend") == "training_exact",
        "feature_pipeline_equivalent=true": payload.get("feature_pipeline_equivalent") is True,
        "run_scope=fold_seed": payload.get("run_scope") == "fold_seed",
        "eligible_for_cv_aggregation=true": payload.get("eligible_for_cv_aggregation") is True,
        "leakage_audit_ok=true": payload.get("leakage_audit_ok") is True,
        "manifest_sha_verified=true": payload.get("manifest_sha_verified") is True,
        "test_used_for_parameter_selection=false": payload.get("test_used_for_parameter_selection") is False,
        "hierarchical aux probability weight=0.5": math.isclose(
            float(best.get("hier_aux_prob_weight", -1.0)), EXPECTED_LAMBDA, abs_tol=1e-12
        ),
        "main labels match": tuple(payload.get("labels", ())) == MAIN_LABELS,
        "auxiliary labels match": tuple(payload.get("aux_labels", ())) == AUX_LABELS,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"BLOCKER: prototype provenance failed for {path}: {failed}")
    methods = payload.get("methods", {})
    raw_payload = _method_payload_for_route(
        methods,
        route="primary_softmax",
        field="methods",
        path=path,
    )
    hierarchical_payload = _method_payload_for_route(
        methods,
        route="hierarchical_prototype_candidate",
        field="methods",
        path=path,
    )
    try:
        raw = float(raw_payload["macro_f1"])
        hierarchical = float(hierarchical_payload["macro_f1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"BLOCKER: missing B2/B3 Macro-F1 in {path}") from exc
    if not (np.isfinite(raw) and np.isfinite(hierarchical)):
        raise ValueError(f"BLOCKER: non-finite B2/B3 Macro-F1 in {path}")
    return raw, hierarchical


def _validate_leakage_audit(path: Path) -> None:
    payload = _read_json(path)
    if payload.get("ok") is not True:
        raise ValueError(f"BLOCKER: leakage audit did not pass: {path}")
    if payload.get("overlaps") not in ([], None):
        raise ValueError(f"BLOCKER: leakage overlaps were recorded in {path}")
    if payload.get("invalid_values") not in ([], None):
        raise ValueError(f"BLOCKER: invalid leakage-audit values were recorded in {path}")
    columns = set(payload.get("columns_checked", []))
    if columns != {"path", "source_id", "md5"}:
        raise ValueError(
            f"BLOCKER: leakage audit must check path/source_id/md5 in {path}; got {sorted(columns)}"
        )


def load_cumulative_runs(root: Path) -> pd.DataFrame:
    """Load and strictly validate all 25 matched B0-B3 Macro-F1 rows."""

    root = root.resolve()
    rows: list[dict[str, Any]] = []
    for fold, seed in EXPECTED_KEYS:
        sources = {stage: _summary_path(root, stage, fold, seed) for stage in ("B0", "B1", "B2")}
        values = {
            stage: _validate_stage_summary(
                _read_json(path), path=path, stage=stage, seed=seed
            )
            for stage, path in sources.items()
        }
        prototype_dir = _prototype_run_dir(root, fold, seed)
        metadata_path = prototype_dir / "artifacts" / "prototype_metadata.json"
        metrics_path = prototype_dir / "evaluation" / "metrics.json"
        audit_path = prototype_dir / "evaluation" / "leakage_audit.json"
        metadata_payload = _read_json(metadata_path)
        metrics_payload = _read_json(metrics_path)
        _validate_fixed_selection_protocol(
            (
                (f"prototype metadata {metadata_path}", metadata_payload),
                (f"evaluation metrics {metrics_path}", metrics_payload),
            )
        )
        _validate_prototype_metadata(
            metadata_payload,
            path=metadata_path,
            root=root,
            fold=fold,
            seed=seed,
            summary_path=sources["B2"],
        )
        raw_softmax, b3 = _validate_prototype_metrics(
            metrics_payload, path=metrics_path, fold=fold, seed=seed
        )
        metadata_debug = _as_bool(metadata_payload.get("single_fold_debug"))
        metrics_debug = _as_bool(metrics_payload.get("single_fold_debug"))
        if metadata_debug != metrics_debug:
            raise ValueError(
                "BLOCKER: prototype metadata/evaluation single_fold_debug mismatch "
                f"at fold={fold}, seed={seed}"
            )
        _validate_leakage_audit(audit_path)
        if values["B2"] != raw_softmax:
            raise ValueError(
                "BLOCKER: B2 raw Softmax does not exactly match the prototype cohort "
                f"at fold={fold}, seed={seed}: {values['B2']} != {raw_softmax}"
            )
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "B0": values["B0"],
                "B1": values["B1"],
                "B2": values["B2"],
                "B3": b3,
                "B3_raw_softmax": raw_softmax,
                "B3_single_fold_debug": metadata_debug,
                "lambda": EXPECTED_LAMBDA,
                "B0_source": _relative(root, sources["B0"]),
                "B1_source": _relative(root, sources["B1"]),
                "B2_source": _relative(root, sources["B2"]),
                "B3_source": _relative(root, metrics_path),
                "B3_metadata_source": _relative(root, metadata_path),
                "leakage_audit_source": _relative(root, audit_path),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["fold", "seed"], kind="stable").reset_index(drop=True)
    actual = tuple(map(tuple, frame[["fold", "seed"]].to_numpy()))
    if actual != EXPECTED_KEYS:
        raise ValueError(f"BLOCKER: strict fold-seed grid mismatch: {actual}")
    if frame[["fold", "seed"]].duplicated().any():
        raise ValueError("BLOCKER: duplicate fold-seed rows in cumulative cohort")
    return frame


def compute_stage_summary(runs: pd.DataFrame) -> pd.DataFrame:
    required = {"fold", "seed", *STAGE_ORDER}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"Cumulative runs are missing columns: {missing}")
    rows = []
    for stage in STAGE_ORDER:
        values = runs[stage].to_numpy(dtype=np.float64)
        rows.append(
            {
                "stage": stage,
                "stage_name": STAGE_DESCRIPTIONS[stage],
                "n": int(len(values)),
                "mean_macro_f1": float(values.mean()),
                "std_macro_f1": float(values.std(ddof=1)),
                "min_macro_f1": float(values.min()),
                "max_macro_f1": float(values.max()),
                **canonical_metadata_for_stage(stage),
            }
        )
    return pd.DataFrame(rows)


def _add_comparison_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe the final route and retain explicit baseline identity."""

    enriched = frame.copy()
    if enriched.empty:
        for field in METHOD_METADATA_FIELDS:
            enriched[field] = pd.Series(dtype=object)
        return enriched
    final_metadata = [
        canonical_metadata_for_stage(stage)
        for stage in enriched["final_stage"].astype(str)
    ]
    baseline_metadata = [
        canonical_metadata_for_stage(stage)
        for stage in enriched["baseline_stage"].astype(str)
    ]
    for field in METHOD_METADATA_FIELDS:
        enriched[field] = [metadata[field] for metadata in final_metadata]
    for field in (
        "training_stage",
        "selection_protocol",
        "inference_route",
        "canonical_method_id",
        "display_name_en",
        "display_name_zh",
        "legacy_method_id",
    ):
        enriched[f"baseline_{field}"] = [
            metadata[field] for metadata in baseline_metadata
        ]
    return enriched


def _bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    delta = np.asarray(values, dtype=np.float64)
    if delta.ndim != 1 or len(delta) == 0:
        raise ValueError("Bootstrap values must be a non-empty one-dimensional array")
    if n_boot <= 0:
        raise ValueError("bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, size=(n_boot, len(delta)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _wilcoxon_p(delta: np.ndarray) -> float:
    if np.allclose(delta, 0.0):
        return 1.0
    return float(wilcoxon(delta).pvalue)


def compute_paired_statistics(
    runs: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute every requested contrast directly from matched rows."""

    required = {"fold", "seed", *STAGE_ORDER}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"Cumulative runs are missing columns: {missing}")
    keys = tuple(map(tuple, runs.sort_values(["fold", "seed"])[["fold", "seed"]].to_numpy()))
    if keys != EXPECTED_KEYS:
        raise ValueError("Strict paired statistics require the exact 25 fold-seed keys")

    paired_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    ordered = runs.sort_values(["fold", "seed"], kind="stable").reset_index(drop=True)
    for final_stage, baseline_stage in COMPARISONS:
        comparison = f"{final_stage} - {baseline_stage}"
        delta = (
            ordered[final_stage].to_numpy(dtype=np.float64)
            - ordered[baseline_stage].to_numpy(dtype=np.float64)
        )
        run_low, run_high = _bootstrap_ci(delta, n_boot=n_boot, seed=seed)
        tie_mask = np.isclose(delta, 0.0)
        fold_means: dict[int, float] = {}
        for fold in EXPECTED_FOLDS:
            subset = ordered[ordered["fold"] == fold]
            fold_delta = (
                subset[final_stage].to_numpy(dtype=np.float64)
                - subset[baseline_stage].to_numpy(dtype=np.float64)
            )
            if len(fold_delta) != len(EXPECTED_SEEDS):
                raise ValueError(
                    f"BLOCKER: comparison {comparison} fold={fold} has {len(fold_delta)} seeds"
                )
            fold_ties = np.isclose(fold_delta, 0.0)
            fold_means[fold] = float(fold_delta.mean())
            fold_rows.append(
                {
                    "comparison": comparison,
                    "baseline_stage": baseline_stage,
                    "final_stage": final_stage,
                    "fold": fold,
                    "n_seeds": int(len(fold_delta)),
                    "baseline_mean": float(subset[baseline_stage].mean()),
                    "final_mean": float(subset[final_stage].mean()),
                    "mean_delta": float(fold_delta.mean()),
                    "std_delta": float(fold_delta.std(ddof=1)),
                    "wins": int(((fold_delta > 0) & ~fold_ties).sum()),
                    "ties": int(fold_ties.sum()),
                    "losses": int(((fold_delta < 0) & ~fold_ties).sum()),
                }
            )
        fold_values = np.array([fold_means[fold] for fold in EXPECTED_FOLDS], dtype=np.float64)
        fold_low, fold_high = _bootstrap_ci(fold_values, n_boot=n_boot, seed=seed)
        row: dict[str, Any] = {
            "comparison": comparison,
            "baseline_stage": baseline_stage,
            "final_stage": final_stage,
            "n": int(len(delta)),
            "baseline_mean": float(ordered[baseline_stage].mean()),
            "final_mean": float(ordered[final_stage].mean()),
            "mean_delta": float(delta.mean()),
            "std_delta": float(delta.std(ddof=1)),
            "bootstrap_ci95_low": run_low,
            "bootstrap_ci95_high": run_high,
            "wilcoxon_p": _wilcoxon_p(delta),
            "wins": int(((delta > 0) & ~tie_mask).sum()),
            "ties": int(tie_mask.sum()),
            "losses": int(((delta < 0) & ~tie_mask).sum()),
            "fold_cluster_n": len(EXPECTED_FOLDS),
            "fold_cluster_bootstrap_ci95_low": fold_low,
            "fold_cluster_bootstrap_ci95_high": fold_high,
        }
        row.update({f"fold_{fold}_mean_delta": fold_means[fold] for fold in EXPECTED_FOLDS})
        paired_rows.append(row)
    return (
        _add_comparison_metadata(pd.DataFrame(paired_rows)),
        _add_comparison_metadata(pd.DataFrame(fold_rows)),
    )


def _top2(
    row: pd.Series,
    *,
    prefix: str,
    labels: Sequence[str],
) -> tuple[str, float, list[tuple[str, float]]]:
    probabilities = np.array([float(row[f"{prefix}_prob_{label}"]) for label in labels])
    order = np.argsort(-probabilities, kind="stable")[:2]
    ranked = [(labels[index], float(probabilities[index])) for index in order]
    text = " | ".join(f"{label}:{probability:.6f}" for label, probability in ranked)
    return text, float(ranked[0][1] - ranked[1][1]), ranked


def _lower_median(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.Series, int]:
    if frame.empty:
        raise ValueError("BLOCKER: deterministic case-selection pool is empty")
    ordered = frame.sort_values(list(columns), kind="stable").reset_index(drop=True)
    index = (len(ordered) - 1) // 2
    return ordered.iloc[index], index + 1


def _representative(
    frame: pd.DataFrame,
    *,
    label_column: str,
    label: str,
    distance_column: str,
) -> pd.Series:
    candidates = frame[frame[label_column].astype(str) == label].copy()
    if candidates.empty:
        raise ValueError(f"BLOCKER: no train-only representative for {label_column}={label}")
    candidates[distance_column] = pd.to_numeric(candidates[distance_column], errors="raise")
    return candidates.sort_values([distance_column, "source_id", "path"], kind="stable").iloc[0]


def _validate_canonical_membership(
    *,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    main_reps: pd.DataFrame,
    aux_reps: pd.DataFrame,
) -> None:
    """Validate canonical case and representative membership by all leakage keys."""

    identity = ("path", "source_id", "md5")

    def normalise(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
        missing = [column for column in identity if column not in frame]
        if missing:
            raise ValueError(f"BLOCKER: {name} is missing identity columns: {missing}")
        result = frame.copy()
        if result[list(identity)].isna().any().any():
            raise ValueError(f"BLOCKER: {name} contains missing path/source_id/md5 values")
        for column in identity:
            result[column] = result[column].astype(str).str.strip()
            if column == "path":
                result[column] = result[column].str.replace("\\", "/", regex=False)
            if (result[column] == "").any():
                raise ValueError(f"BLOCKER: {name} contains an empty {column}")
            if result[column].duplicated().any():
                raise ValueError(f"BLOCKER: {name} contains duplicate {column} values")
        return result

    frames = {
        "train": normalise(train, name="train manifest"),
        "val": normalise(val, name="validation manifest"),
        "test": normalise(test, name="test manifest"),
        "predictions": normalise(predictions, name="canonical predictions"),
        "main_reps": normalise(main_reps, name="main representatives"),
        "aux_reps": normalise(aux_reps, name="subtype representatives"),
    }

    for field in identity:
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = set(frames[left][field]) & set(frames[right][field])
            if overlap:
                raise ValueError(
                    f"BLOCKER: {field} overlap between {left} and {right} manifests: "
                    f"{sorted(overlap)[:3]}"
                )

    def require_exact_membership(artifact: str, split: str) -> None:
        actual = set(map(tuple, frames[artifact][list(identity)].to_numpy()))
        expected = set(map(tuple, frames[split][list(identity)].to_numpy()))
        if actual != expected:
            missing = sorted(expected - actual)[:3]
            extra = sorted(actual - expected)[:3]
            raise ValueError(
                f"BLOCKER: {artifact} identity membership differs from {split}; "
                f"missing={missing}, extra={extra}"
            )

    require_exact_membership("predictions", "test")
    require_exact_membership("main_reps", "train")
    require_exact_membership("aux_reps", "train")

    def require_label_match(
        artifact: str,
        split: str,
        artifact_label: str,
        manifest_label: str,
    ) -> None:
        if artifact_label not in frames[artifact] or manifest_label not in frames[split]:
            raise ValueError(
                f"BLOCKER: cannot validate {artifact_label}/{manifest_label} labels for {artifact}"
            )
        merged = frames[artifact][[*identity, artifact_label]].merge(
            frames[split][[*identity, manifest_label]],
            on=list(identity),
            how="inner",
            validate="one_to_one",
            suffixes=("_artifact", "_manifest"),
        )
        left_column = (
            f"{artifact_label}_artifact" if artifact_label == manifest_label else artifact_label
        )
        right_column = (
            f"{manifest_label}_manifest" if artifact_label == manifest_label else manifest_label
        )
        if not (
            merged[left_column].astype(str).to_numpy()
            == merged[right_column].astype(str).to_numpy()
        ).all():
            raise ValueError(
                f"BLOCKER: {artifact} {artifact_label} does not match {split} {manifest_label}"
            )

    require_label_match("predictions", "test", "y_true", "label")
    require_label_match("predictions", "test", "subtype", "subtype")
    require_label_match("main_reps", "train", "main_label", "label")
    require_label_match("aux_reps", "train", "aux_label", "subtype")


def select_case_studies(root: Path) -> pd.DataFrame:
    """Select seven reproducible examples from the canonical fold0/seed42 run."""

    root = root.resolve()
    fold, seed = EXPECTED_KEYS[0]
    run_dir = _prototype_run_dir(root, fold, seed)
    summary_path = _summary_path(root, "B2", fold, seed)
    metadata_path = run_dir / "artifacts" / "prototype_metadata.json"
    metrics_path = run_dir / "evaluation" / "metrics.json"
    _validate_prototype_metadata(
        _read_json(metadata_path),
        path=metadata_path,
        root=root,
        fold=fold,
        seed=seed,
        summary_path=summary_path,
    )
    _validate_prototype_metrics(
        _read_json(metrics_path), path=metrics_path, fold=fold, seed=seed
    )
    _validate_leakage_audit(run_dir / "evaluation" / "leakage_audit.json")

    probability_columns = [
        *(f"raw_softmax_prob_{label}" for label in MAIN_LABELS),
        *(f"prototype_prob_{label}" for label in MAIN_LABELS),
        *(f"hierarchical_prob_{label}" for label in MAIN_LABELS),
        *(f"prototype_aux_prob_{label}" for label in AUX_LABELS),
    ]
    distance_columns = [
        *(f"main_proto_cosine_distance_{label}" for label in MAIN_LABELS),
        *(f"aux_proto_cosine_distance_{label}" for label in AUX_LABELS),
    ]
    prediction_columns = {
        "path",
        "source_id",
        "md5",
        "y_true",
        "raw_softmax_pred",
        "prototype_pred",
        "hierarchical_pred",
        "prototype_aux_pred",
        "prototype_aux_mapped_main",
        "input_role",
        "selected_method",
        *probability_columns,
        *distance_columns,
    }
    predictions_path = run_dir / "evaluation" / "test_predictions.csv"
    predictions = _require_csv(predictions_path, prediction_columns)
    if set(predictions["input_role"].astype(str)) != {"frozen_test"}:
        raise ValueError("BLOCKER: canonical case predictions are not exclusively frozen_test")
    selected_routes = {
        canonicalize_inference_route(method)
        for method in predictions["selected_method"].astype(str)
    }
    if selected_routes != {"hierarchical_prototype_candidate"}:
        raise ValueError(
            "BLOCKER: canonical case predictions do not use the "
            "hierarchical_prototype_candidate route"
        )
    if predictions[["source_id", "md5"]].duplicated().any():
        raise ValueError("BLOCKER: duplicate source_id/md5 rows in canonical test predictions")

    raw_margins = []
    hierarchical_margins = []
    for _, row in predictions.iterrows():
        _, margin, _ = _top2(row, prefix="raw_softmax", labels=MAIN_LABELS)
        raw_margins.append(margin)
        _, margin, _ = _top2(row, prefix="hierarchical", labels=MAIN_LABELS)
        hierarchical_margins.append(margin)
    predictions = predictions.copy()
    predictions["_raw_margin"] = raw_margins
    predictions["_hierarchical_margin"] = hierarchical_margins

    selected: list[tuple[str, pd.Series, str, int, int]] = []
    used: set[str] = set()
    for label in MAIN_LABELS:
        pool = predictions[
            (predictions["y_true"].astype(str) == label)
            & (predictions["raw_softmax_pred"].astype(str) == label)
            & (predictions["prototype_pred"].astype(str) == label)
            & (predictions["hierarchical_pred"].astype(str) == label)
            & ~predictions["source_id"].astype(str).isin(used)
        ].copy()
        pool["_raw_confidence"] = pool[
            [f"raw_softmax_prob_{class_label}" for class_label in MAIN_LABELS]
        ].max(axis=1)
        chosen, rank = _lower_median(pool, ("_raw_confidence", "source_id"))
        case_type = f"correct_{label}"
        selected.append((case_type, chosen, "lower median raw Softmax confidence", rank, len(pool)))
        used.add(str(chosen["source_id"]))

    boundary_set = {"feeding", "stress_vocal"}
    for true_label, suffix in (("feeding", "feeding"), ("stress_vocal", "stress")):
        indices = []
        for index, row in predictions.iterrows():
            raw_labels = {item[0] for item in _top2(row, prefix="raw_softmax", labels=MAIN_LABELS)[2]}
            hier_labels = {item[0] for item in _top2(row, prefix="hierarchical", labels=MAIN_LABELS)[2]}
            if (
                str(row["y_true"]) == true_label
                and (raw_labels == boundary_set or hier_labels == boundary_set)
                and (
                    str(row["raw_softmax_pred"]) != true_label
                    or str(row["prototype_pred"]) != true_label
                    or str(row["hierarchical_pred"]) != true_label
                )
                and str(row["source_id"]) not in used
            ):
                indices.append(index)
        pool = predictions.loc[indices].copy()
        chosen, rank = _lower_median(pool, ("_raw_margin", "source_id"))
        case_type = f"feeding_stress_boundary_{suffix}"
        selected.append(
            (
                case_type,
                chosen,
                "lower median raw Top-2 margin among boundary errors",
                rank,
                len(pool),
            )
        )
        used.add(str(chosen["source_id"]))

    remaining = predictions[~predictions["source_id"].astype(str).isin(used)].copy()
    remaining = remaining.sort_values(["_hierarchical_margin", "source_id"], kind="stable")
    if remaining.empty:
        raise ValueError("BLOCKER: no remaining row for low-margin uncertain case")
    chosen = remaining.iloc[0]
    selected.append(
        (
            "low_margin_uncertain",
            chosen,
            "minimum remaining hierarchical prototype Top-2 margin",
            1,
            len(remaining),
        )
    )

    main_reps_path = run_dir / "artifacts" / "train_main_prototype_samples.csv"
    aux_reps_path = run_dir / "artifacts" / "train_aux_prototype_samples.csv"
    main_reps = _require_csv(
        main_reps_path,
        {
            "path",
            "source_id",
            "md5",
            "main_label",
            "main_own_cosine_distance",
        },
    )
    aux_reps = _require_csv(
        aux_reps_path,
        {
            "path",
            "source_id",
            "md5",
            "aux_label",
            "aux_own_cosine_distance",
        },
    )
    train_manifest = _require_csv(
        _manifest_path(root, fold, "train"),
        {"path", "source_id", "md5", "label", "subtype"},
    )
    val_manifest = _require_csv(
        _manifest_path(root, fold, "val"),
        {"path", "source_id", "md5", "label", "subtype"},
    )
    test_manifest = _require_csv(
        _manifest_path(root, fold, "test"),
        {"path", "source_id", "md5", "label", "subtype"},
    )
    _validate_canonical_membership(
        train=train_manifest,
        val=val_manifest,
        test=test_manifest,
        predictions=predictions,
        main_reps=main_reps,
        aux_reps=aux_reps,
    )

    output_rows: list[dict[str, Any]] = []
    for case_index, (case_type, row, selection_rule, rank, pool_size) in enumerate(selected, start=1):
        raw_text, raw_margin, _ = _top2(row, prefix="raw_softmax", labels=MAIN_LABELS)
        main_text, main_margin, _ = _top2(row, prefix="prototype", labels=MAIN_LABELS)
        hierarchical_text, hierarchical_margin, _ = _top2(
            row, prefix="hierarchical", labels=MAIN_LABELS
        )
        subtype_text, subtype_margin, _ = _top2(
            row, prefix="prototype_aux", labels=AUX_LABELS
        )
        predicted_main = str(row["hierarchical_pred"])
        predicted_subtype = str(row["prototype_aux_pred"])
        main_rep = _representative(
            main_reps,
            label_column="main_label",
            label=predicted_main,
            distance_column="main_own_cosine_distance",
        )
        aux_rep = _representative(
            aux_reps,
            label_column="aux_label",
            label=predicted_subtype,
            distance_column="aux_own_cosine_distance",
        )
        main_distances = {
            label: float(row[f"main_proto_cosine_distance_{label}"]) for label in MAIN_LABELS
        }
        subtype_distances = {
            label: float(row[f"aux_proto_cosine_distance_{label}"]) for label in AUX_LABELS
        }
        output: dict[str, Any] = {
            "case_id": f"C{case_index}",
            "case_type": case_type,
            "selection_rule": selection_rule,
            "selection_rank": rank,
            "selection_pool_size": pool_size,
            "fold": fold,
            "seed": seed,
            "run_source": _relative(root, predictions_path),
            "path": str(row["path"]).replace("\\", "/"),
            "source_id": str(row["source_id"]),
            "md5": str(row["md5"]),
            "true_class": str(row["y_true"]),
            "raw_softmax_pred": str(row["raw_softmax_pred"]),
            "raw_softmax_top2": raw_text,
            "raw_softmax_margin": raw_margin,
            "main_prototype_pred": str(row["prototype_pred"]),
            "main_prototype_top2": main_text,
            "main_prototype_margin": main_margin,
            "hierarchical_prototype_pred": predicted_main,
            "hierarchical_prototype_top2": hierarchical_text,
            "hierarchical_prototype_margin": hierarchical_margin,
            "subtype_pred": predicted_subtype,
            "subtype_top2": subtype_text,
            "subtype_margin": subtype_margin,
            "main_distances": " | ".join(
                f"{label}:{main_distances[label]:.6f}" for label in MAIN_LABELS
            ),
            "subtype_distances": " | ".join(
                f"{label}:{subtype_distances[label]:.6f}" for label in AUX_LABELS
            ),
            "margin": hierarchical_margin,
            "margin_definition": "hierarchical prototype Top-1 probability minus Top-2 probability",
            "hierarchy_consistency": predicted_main == str(row["prototype_aux_mapped_main"]),
            "prototype_aux_mapped_main": str(row["prototype_aux_mapped_main"]),
            "closest_representative_sample": str(main_rep["path"]).replace("\\", "/"),
            "closest_representative_source_id": str(main_rep["source_id"]),
            "closest_representative_md5": str(main_rep["md5"]),
            "closest_representative_class": str(main_rep["main_label"]),
            "closest_representative_prototype_distance": float(
                main_rep["main_own_cosine_distance"]
            ),
            "closest_subtype_representative_sample": str(aux_rep["path"]).replace("\\", "/"),
            "closest_subtype_representative_source_id": str(aux_rep["source_id"]),
            "closest_subtype_representative_md5": str(aux_rep["md5"]),
            "closest_subtype_representative_prototype_distance": float(
                aux_rep["aux_own_cosine_distance"]
            ),
            "representative_split": "train_only",
            "representative_definition": "same-run training sample closest to the predicted class prototype",
        }
        for label in MAIN_LABELS:
            output[f"raw_softmax_prob_{label}"] = float(row[f"raw_softmax_prob_{label}"])
            output[f"main_prototype_prob_{label}"] = float(row[f"prototype_prob_{label}"])
            output[f"hierarchical_prototype_prob_{label}"] = float(
                row[f"hierarchical_prob_{label}"]
            )
            output[f"main_distance_{label}"] = main_distances[label]
        for label in AUX_LABELS:
            output[f"subtype_prob_{label}"] = float(row[f"prototype_aux_prob_{label}"])
            output[f"subtype_distance_{label}"] = subtype_distances[label]
        output_rows.append(output)

    cases = pd.DataFrame(output_rows)
    if cases["source_id"].nunique() != len(cases):
        raise ValueError("BLOCKER: deterministic case roles are not unique")
    for field, value in canonical_metadata_for_stage("B3").items():
        cases[field] = value
    return cases


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="bottom",
    )


def _format_p(value: float) -> str:
    if value < 0.001:
        return f"P={value:.2e}"
    return f"P={value:.3f}"


def build_cumulative_figure(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    folds: pd.DataFrame,
) -> plt.Figure:
    """Build the four-panel cumulative framework figure."""

    width = FIGURE_WIDTH_MM / 25.4
    height = CUMULATIVE_HEIGHT_MM / 25.4
    fig = plt.figure(figsize=(width, height), facecolor=WHITE)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.05, 1.15),
        height_ratios=(1.1, 0.9),
        left=0.09,
        right=0.985,
        bottom=0.08,
        top=0.96,
        wspace=0.42,
        hspace=0.5,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    x = np.arange(len(STAGE_ORDER), dtype=float)
    for _, row in runs.iterrows():
        values = row[list(STAGE_ORDER)].to_numpy(dtype=np.float64)
        ax_a.plot(x, values, color="#BFC5CD", linewidth=0.45, alpha=0.55, zorder=1)
        fold_offset = (int(row["fold"]) - 2) * 0.006
        seed_offset = (EXPECTED_SEEDS.index(int(row["seed"])) - 2) * 0.022
        jitter = fold_offset + seed_offset
        for stage_index, stage in enumerate(STAGE_ORDER):
            ax_a.scatter(
                stage_index + jitter,
                float(row[stage]),
                s=9,
                facecolor=STAGE_COLORS[stage],
                edgecolor=WHITE,
                linewidth=0.25,
                alpha=0.78,
                zorder=2,
            )
    means = summary.set_index("stage").loc[list(STAGE_ORDER), "mean_macro_f1"].to_numpy(float)
    ax_a.plot(x, means, color=TEXT, linewidth=1.35, marker="D", markersize=3.6, zorder=4)
    for index, value in enumerate(means):
        ax_a.text(index, value + 0.0045, f"{value:.4f}", ha="center", va="bottom", fontsize=6.2)
    data_min = float(runs[list(STAGE_ORDER)].min().min())
    data_max = float(runs[list(STAGE_ORDER)].max().max())
    ax_a.set_ylim(max(0.0, data_min - 0.012), min(1.0, data_max + 0.012))
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([STAGE_LABELS[stage] for stage in STAGE_ORDER], fontsize=6.2)
    ax_a.set_ylabel("Macro-F1")
    ax_a.set_title("Cumulative stage performance", loc="left", fontsize=8, fontweight="bold")
    ax_a.tick_params(axis="x", length=0)
    ax_a.yaxis.set_tick_params(labelsize=6)
    _panel_label(ax_a, "a")

    plot_order = paired["comparison"].tolist()
    ypos = np.arange(len(plot_order))[::-1]
    all_lows = np.r_[
        paired["bootstrap_ci95_low"].to_numpy(float),
        paired["fold_cluster_bootstrap_ci95_low"].to_numpy(float),
    ]
    all_highs = np.r_[
        paired["bootstrap_ci95_high"].to_numpy(float),
        paired["fold_cluster_bootstrap_ci95_high"].to_numpy(float),
    ]
    for y, (_, row) in zip(ypos, paired.iterrows()):
        ax_b.plot(
            [float(row["bootstrap_ci95_low"]), float(row["bootstrap_ci95_high"])],
            [y + 0.10, y + 0.10],
            color=ACCENT,
            linewidth=2.0,
            solid_capstyle="round",
        )
        ax_b.plot(float(row["mean_delta"]), y + 0.10, "o", color=ACCENT, markersize=3.4)
        ax_b.plot(
            [
                float(row["fold_cluster_bootstrap_ci95_low"]),
                float(row["fold_cluster_bootstrap_ci95_high"]),
            ],
            [y - 0.12, y - 0.12],
            color=FOLD_COLOR,
            linewidth=1.0,
        )
        ax_b.plot(
            float(row["mean_delta"]),
            y - 0.12,
            marker="s",
            markerfacecolor=WHITE,
            markeredgecolor=FOLD_COLOR,
            markersize=3.1,
        )
        ax_b.text(
            0.99,
            (y + 0.52) / len(plot_order),
            _format_p(float(row["wilcoxon_p"])),
            transform=ax_b.transAxes,
            ha="right",
            va="center",
            fontsize=5.7,
            color=MUTED,
        )
    margin = max(0.004, 0.08 * (float(all_highs.max()) - float(all_lows.min())))
    ax_b.set_xlim(float(all_lows.min()) - margin, float(all_highs.max()) + margin)
    ax_b.axvline(0.0, color=MUTED, linestyle="--", linewidth=0.8, zorder=0)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels(plot_order, fontsize=6.1)
    ax_b.set_xlabel(
        "Paired Macro-F1 difference\nblue: 25-run bootstrap CI; grey: 5-fold cluster bootstrap CI",
        fontsize=6.2,
    )
    ax_b.set_title("Direct paired increments", loc="left", fontsize=8, fontweight="bold")
    ax_b.tick_params(axis="x", labelsize=6)
    _panel_label(ax_b, "b")

    b3_b0 = folds[folds["comparison"] == "B3 - B0"].sort_values("fold")
    fold_x = b3_b0["fold"].to_numpy(int)
    fold_delta = b3_b0["mean_delta"].to_numpy(float)
    bars = ax_c.bar(
        fold_x,
        fold_delta,
        width=0.62,
        color=["#7899AA" if value >= 0 else "#C47A72" for value in fold_delta],
        edgecolor=WHITE,
        linewidth=0.6,
    )
    for bar, value in zip(bars, fold_delta):
        va = "bottom" if value >= 0 else "top"
        offset = 0.0012 if value >= 0 else -0.0012
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.4f}",
            ha="center",
            va=va,
            fontsize=5.7,
        )
    overall = float(paired.set_index("comparison").loc["B3 - B0", "mean_delta"])
    ax_c.axhline(overall, color=TEXT, linewidth=1.1, label=f"25-run mean {overall:+.4f}")
    ax_c.axhline(0.0, color=MUTED, linestyle="--", linewidth=0.7)
    ax_c.set_xticks(fold_x)
    ax_c.set_xticklabels([f"Fold {fold}" for fold in fold_x], fontsize=6)
    ax_c.set_ylabel("B3 - B0 Macro-F1")
    ax_c.set_title("Fold-level cumulative gain", loc="left", fontsize=8, fontweight="bold")
    ax_c.legend(fontsize=5.4, loc="best")
    ax_c.tick_params(axis="y", labelsize=6)
    _panel_label(ax_c, "c")

    ax_d.set_axis_off()
    ax_d.set_title(
        "Backbone and inference-route distinction",
        loc="left",
        fontsize=8,
        fontweight="bold",
        pad=8,
    )
    _panel_label(ax_d, "d")

    def box(x0: float, y0: float, width_box: float, height_box: float, color: str, title: str, body: str) -> None:
        patch = FancyBboxPatch(
            (x0, y0),
            width_box,
            height_box,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            transform=ax_d.transAxes,
            facecolor=color,
            edgecolor=TEXT,
            linewidth=0.7,
        )
        ax_d.add_patch(patch)
        ax_d.text(
            x0 + width_box / 2,
            y0 + height_box * 0.66,
            title,
            transform=ax_d.transAxes,
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=TEXT,
        )
        ax_d.text(
            x0 + width_box / 2,
            y0 + height_box * 0.31,
            body,
            transform=ax_d.transAxes,
            ha="center",
            va="center",
            fontsize=5.4,
            color=MUTED,
            linespacing=1.25,
            wrap=True,
        )

    box(0.02, 0.38, 0.19, 0.36, PALE_BLUE, "2-s Log-Mel", "64 bins\n2-s context")
    box(
        0.29,
        0.30,
        0.28,
        0.52,
        PALE_GREEN,
        "Hierarchical-supervision\nCRNN",
        "shared representation\n4 main classes\n6 supervised subtypes",
    )
    box(
        0.68,
        0.56,
        0.30,
        0.28,
        PALE_BLUE,
        "Primary classifier",
        _PRIMARY_SOFTMAX_LABEL,
    )
    box(
        0.68,
        0.12,
        0.30,
        0.31,
        PALE_GOLD,
        "Parallel candidate\nroutes",
        f"{_MAIN_PROTOTYPE_LABEL}\n{_HIERARCHICAL_PROTOTYPE_LABEL}",
    )
    for start, end, start_y, end_y in (
        (0.21, 0.29, 0.56, 0.56),
        (0.57, 0.68, 0.62, 0.70),
        (0.57, 0.68, 0.50, 0.28),
    ):
        ax_d.add_patch(
            FancyArrowPatch(
                (start, start_y),
                (end, end_y),
                transform=ax_d.transAxes,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color=ACCENT,
            )
        )
    ax_d.text(
        0.5,
        0.02,
        "B3 fixed λ=0.5 retrospective:\n"
        "hierarchical-prototype Top-1 decision ablation",
        transform=ax_d.transAxes,
        ha="center",
        va="center",
        fontsize=6.1,
        fontweight="bold",
        color=TEXT,
    )
    return fig


def _annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    cmap: str,
    value_format: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if matrix.ndim != 2:
        raise ValueError("Heatmap matrix must be two-dimensional")
    lower = float(np.nanmin(matrix)) if vmin is None else float(vmin)
    upper = float(np.nanmax(matrix)) if vmax is None else float(vmax)
    norm = mpl.colors.Normalize(vmin=lower, vmax=upper)
    color_map = mpl.colormaps[cmap]
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            facecolor = color_map(norm(value))
            ax.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=facecolor,
                    edgecolor=WHITE,
                    linewidth=0.25,
                )
            )
            red, green, blue, _ = facecolor
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(
                column,
                row,
                value_format.format(value),
                ha="center",
                va="center",
                fontsize=4.5,
                color=WHITE if luminance < 0.48 else TEXT,
            )
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=5.2)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=5.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_case_figure(cases: pd.DataFrame) -> plt.Figure:
    """Build a compact evidence matrix for the seven deterministic cases."""

    width = FIGURE_WIDTH_MM / 25.4
    height = CASE_HEIGHT_MM / 25.4
    fig = plt.figure(figsize=(width, height), facecolor=WHITE)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.22, 1.0),
        height_ratios=(1.12, 0.88),
        left=0.12,
        right=0.985,
        bottom=0.08,
        top=0.96,
        wspace=0.34,
        hspace=0.48,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    route_specs = (
        ("P", "raw_softmax"),
        ("M", "main_prototype"),
        ("H", "hierarchical_prototype"),
    )
    probability_rows = []
    route_labels = []
    for _, case in cases.iterrows():
        for route_label, prefix in route_specs:
            probability_rows.append(
                [float(case[f"{prefix}_prob_{label}"]) for label in MAIN_LABELS]
            )
            route_labels.append(f"{case['case_id']} {route_label}")
    probability_matrix = np.asarray(probability_rows, dtype=np.float64)
    _annotated_heatmap(
        ax_a,
        probability_matrix,
        xlabels=("cough", "calm", "feeding", "stress"),
        ylabels=route_labels,
        cmap="Blues",
        value_format="{:.2f}",
        vmin=0.0,
        vmax=1.0,
    )
    for boundary in range(3, len(route_labels), 3):
        ax_a.axhline(boundary - 0.5, color=WHITE, linewidth=1.2)
    ax_a.set_title("Main-class probabilities (Top-2 evidence)", loc="left", fontsize=8, fontweight="bold")
    _panel_label(ax_a, "a")

    main_distance_matrix = cases[
        [f"main_distance_{label}" for label in MAIN_LABELS]
    ].to_numpy(float)
    _annotated_heatmap(
        ax_b,
        main_distance_matrix,
        xlabels=("cough", "calm", "feeding", "stress"),
        ylabels=cases["case_id"].tolist(),
        cmap="YlGnBu_r",
        value_format="{:.2f}",
    )
    ax_b.set_title("Main prototype cosine distance\n(lower is closer)", loc="left", fontsize=8, fontweight="bold")
    _panel_label(ax_b, "b")

    subtype_distance_matrix = cases[
        [f"subtype_distance_{label}" for label in AUX_LABELS]
    ].to_numpy(float)
    _annotated_heatmap(
        ax_c,
        subtype_distance_matrix,
        xlabels=("dry", "abdom.", "calm", "feeding", "fright.", "anxious"),
        ylabels=cases["case_id"].tolist(),
        cmap="YlGnBu_r",
        value_format="{:.2f}",
    )
    ax_c.set_title("Subtype prototype cosine distance (lower is closer)", loc="left", fontsize=8, fontweight="bold")
    _panel_label(ax_c, "c")

    y = np.arange(len(cases))[::-1]
    margins = cases["margin"].to_numpy(float)
    consistency = cases["hierarchy_consistency"].map(_as_bool).to_numpy(bool)
    colors = np.where(consistency, "#70865A", "#A95449")
    ax_d.hlines(y, 0.0, margins, color=colors, linewidth=2.6)
    ax_d.scatter(margins, y, c=colors, s=20, edgecolor=WHITE, linewidth=0.5, zorder=3)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(
        [
            f"{row.case_id}  {row.true_class} {'[C]' if is_consistent else '[I]'}"
            for row, is_consistent in zip(cases.itertuples(), consistency)
        ],
        fontsize=5.2,
    )
    for ypos, row in zip(y, cases.itertuples()):
        representative_id = str(row.closest_representative_source_id).split("::")[-1]
        if len(representative_id) > 11:
            representative_id = f"...{representative_id[-10:]}"
        ax_d.text(
            1.03,
            ypos,
            f"rep: {representative_id}",
            fontsize=4.7,
            color=MUTED,
            ha="left",
            va="center",
            clip_on=True,
        )
    ax_d.set_xlim(left=0.0, right=1.42)
    ax_d.set_xlabel("Hierarchical Top-1 minus Top-2 probability")
    ax_d.set_title("Margin, hierarchy consistency, and train trace", loc="left", fontsize=8, fontweight="bold")
    ax_d.tick_params(axis="x", labelsize=5.5)
    _panel_label(ax_d, "d")

    legend_lines = [
        f"{row.case_id}: {row.case_type.replace('_', ' ')}"
        for row in cases.itertuples()
    ]
    fig.text(
        0.505,
        0.015,
        "\n".join(
            (
                "   ".join(legend_lines),
                (
                    f"P = {_PRIMARY_SOFTMAX_LABEL}; "
                    f"M = {_MAIN_PROTOTYPE_LABEL}; "
                    f"H = {_HIERARCHICAL_PROTOTYPE_LABEL}"
                ),
            )
        ),
        ha="center",
        va="bottom",
        fontsize=4.2,
        color=MUTED,
    )
    return fig


def required_output_paths(root: Path) -> tuple[Path, ...]:
    root = root.resolve()
    paths = [root / relative for relative in TABLE_RELATIVE_PATHS]
    for stem in FIGURE_STEMS:
        paths.extend(root / f"{stem}.{extension}" for extension in ("svg", "pdf", "png"))
    return tuple(paths)


def archive_output_paths(root: Path) -> tuple[Path, ...]:
    root = root.resolve()
    return tuple(
        [root / ARCHIVE_SCRIPT_RELATIVE_PATH]
        + [root / relative for relative in ARCHIVE_TABLE_RELATIVE_PATHS]
    )


def refuse_existing(paths: Iterable[Path]) -> None:
    """Protect versioned outputs from accidental replacement on reruns."""

    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing nomenclature-v1 outputs: "
            + "; ".join(str(path) for path in existing)
        )


def _source_paths(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for fold, seed in EXPECTED_KEYS:
        paths.extend(_summary_path(root, stage, fold, seed) for stage in ("B0", "B1", "B2"))
        run = _prototype_run_dir(root, fold, seed)
        paths.extend(
            (
                run / "artifacts" / "prototype_metadata.json",
                run / "evaluation" / "metrics.json",
                run / "evaluation" / "test_predictions.csv",
                run / "evaluation" / "leakage_audit.json",
                run / "artifacts" / "train_main_prototype_samples.csv",
                run / "artifacts" / "train_aux_prototype_samples.csv",
            )
        )
    paths.extend(
        _manifest_path(root, fold, split)
        for fold in EXPECTED_FOLDS
        for split in ("train", "val", "test")
    )
    unique = tuple(sorted(set(path.resolve() for path in paths), key=str))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"BLOCKER: missing frozen source files: {missing[:5]}")
    return unique


def _hash_sources(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): _sha256_file(path) for path in paths}


def _export_figure(fig: plt.Figure, stem: Path) -> tuple[Path, Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = stem.with_suffix(".svg")
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(svg, facecolor=WHITE)
    fig.savefig(pdf, facecolor=WHITE)
    fig.savefig(png, dpi=300, facecolor=WHITE)
    plt.close(fig)
    return svg, pdf, png


def generate(
    root: Path,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    validate_only: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    if not validate_only:
        refuse_existing(
            (*required_output_paths(root), *archive_output_paths(root))
        )
    source_paths = _source_paths(root)
    hashes_before = _hash_sources(source_paths)
    runs = load_cumulative_runs(root)
    summary = compute_stage_summary(runs)
    paired, folds = compute_paired_statistics(runs, n_boot=n_boot, seed=seed)
    cases = select_case_studies(root)
    cumulative_figure = build_cumulative_figure(runs, summary, paired, folds)
    case_figure = build_case_figure(cases)

    written: list[str] = []
    if validate_only:
        plt.close(cumulative_figure)
        plt.close(case_figure)
    else:
        table_frames = (runs, summary, paired, folds, cases)
        for relative, frame in zip(TABLE_RELATIVE_PATHS, table_frames):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False, encoding="utf-8")
            written.append(_relative(root, path))
        for source_relative, archive_relative in zip(
            TABLE_RELATIVE_PATHS, ARCHIVE_TABLE_RELATIVE_PATHS
        ):
            source = root / source_relative
            archive = root / archive_relative
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, archive)
            written.append(_relative(root, archive))
        canonical_script = root / "tools" / "generate_cumulative_framework_analysis.py"
        archive_script = root / ARCHIVE_SCRIPT_RELATIVE_PATH
        if not canonical_script.is_file():
            raise FileNotFoundError(f"BLOCKER: canonical generator is missing: {canonical_script}")
        archive_script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(canonical_script, archive_script)
        written.append(_relative(root, archive_script))
        for figure, stem in (
            (cumulative_figure, root / FIGURE_STEMS[0]),
            (case_figure, root / FIGURE_STEMS[1]),
        ):
            written.extend(_relative(root, path) for path in _export_figure(figure, stem))

    hashes_after = _hash_sources(source_paths)
    if hashes_before != hashes_after:
        changed = sorted(path for path in hashes_before if hashes_before[path] != hashes_after.get(path))
        raise RuntimeError(f"BLOCKER: frozen source files changed during generation: {changed[:5]}")
    return {
        "runs": runs,
        "summary": summary,
        "paired": paired,
        "folds": folds,
        "cases": cases,
        "written": written,
        "source_file_count": len(source_paths),
        "source_hashes_unchanged": True,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    result = generate(
        args.root,
        n_boot=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        validate_only=args.validate_only,
    )
    mode = "validated" if args.validate_only else "generated"
    print(f"[OK] {mode} cumulative framework analysis")
    print(f"[OK] frozen source files unchanged: {result['source_file_count']}")
    print("\n[STAGE SUMMARY]")
    print(result["summary"].to_string(index=False))
    print("\n[DIRECT PAIRED STATISTICS]")
    print(result["paired"].to_string(index=False))
    print("\n[DETERMINISTIC CASES]")
    print(result["cases"][["case_id", "case_type", "source_id", "margin"]].to_string(index=False))
    if result["written"]:
        print("\n[WRITTEN]")
        for path in result["written"]:
            print(path)


if __name__ == "__main__":
    main()
