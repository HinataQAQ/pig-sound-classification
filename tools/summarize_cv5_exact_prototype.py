from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from nomenclature import (
    METHOD_METADATA_FIELDS,
    MODEL_METADATA_FIELDS,
    build_method_metadata,
    build_model_metadata,
    canonicalize_inference_route,
    canonicalize_model_family,
    canonicalize_selection_protocol,
    display_label,
    reconcile_selection_protocol,
    resolve_inference_method,
    validate_method_metadata,
    validate_model_metadata,
)
from prototype_model_adapter import read_json, require_file, write_json

REQUIRED_FOLDS = (0, 1, 2, 3, 4)
SUMMARY_METHODS = ("raw_softmax", "calibrated_softmax", "prototype", "hierarchical", "fused")
CONFUSION_METHODS = ("raw_softmax", "prototype", "hierarchical")
PAIRED_COMPARISONS = (
    ("hierarchical", "raw_softmax"),
    ("prototype", "raw_softmax"),
    ("hierarchical", "prototype"),
)
METRIC_KEYS = ("macro_f1", "top1_acc", "top2_acc", "ece", "brier", "nll")


def parse_int_csv(text: str, *, name: str) -> list[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values: {values}")
    return values


def _canonical_combo(record: Mapping[str, Any]) -> tuple[int, int]:
    return int(record["fold"]), int(record["seed"])


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _metric_value(metrics: Mapping[str, Any], key: str) -> float:
    if key not in metrics or metrics[key] is None:
        raise RuntimeError(f"missing method metric {key}")
    return float(metrics[key])


def _method_metrics(record: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    methods = record.get("methods")
    if not isinstance(methods, Mapping) or method not in methods:
        combo = (record.get("fold"), record.get("seed"))
        raise RuntimeError(f"missing method metrics for {method} in fold{combo[0]} seed{combo[1]}")
    method_metrics = methods[method]
    if not isinstance(method_metrics, Mapping):
        raise RuntimeError(f"method metrics for {method} must be an object")
    return method_metrics


def validate_aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_folds: Sequence[int],
    expected_seeds: Sequence[int],
    expected_lambda: float | None = None,
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("No run records provided for aggregate summary.")

    required_folds = set(REQUIRED_FOLDS)
    expected_fold_set = {int(x) for x in expected_folds}
    expected_seed_set = {int(x) for x in expected_seeds}
    seen: dict[tuple[int, int], int] = {}
    errors: list[str] = []
    lambdas: set[float] = set()

    for idx, record in enumerate(records):
        try:
            combo = _canonical_combo(record)
        except KeyError as exc:
            errors.append(f"record {idx} missing {exc.args[0]}")
            continue
        seen[combo] = seen.get(combo, 0) + 1

        if not _as_bool(record.get("eligible_for_cv_aggregation", False)):
            errors.append(f"fold{combo[0]} seed{combo[1]} not eligible_for_cv_aggregation")
        if str(record.get("feature_backend")) != "training_exact":
            errors.append(f"fold{combo[0]} seed{combo[1]} feature_backend is not training_exact")
        if not _as_bool(record.get("feature_pipeline_equivalent", False)):
            errors.append(f"fold{combo[0]} seed{combo[1]} feature_pipeline_equivalent is false")
        if str(record.get("input_role")) != "frozen_test":
            errors.append(f"fold{combo[0]} seed{combo[1]} input_role is not frozen_test")
        if _as_bool(record.get("unsafe_allow_checkpoint_sha_mismatch", False)):
            errors.append(f"fold{combo[0]} seed{combo[1]} used unsafe checkpoint SHA mismatch")
        if not _as_bool(record.get("leakage_audit_ok", False)):
            errors.append(f"fold{combo[0]} seed{combo[1]} leakage audit is not ok")
        if not _as_bool(record.get("manifest_sha_verified", False)):
            errors.append(f"fold{combo[0]} seed{combo[1]} manifest SHA was not verified")
        if not _as_bool(record.get("softmax_reproduction_passed", False)):
            errors.append(f"fold{combo[0]} seed{combo[1]} Softmax reproduction did not pass")

        raw_lambda = record.get("lambda", record.get("hier_aux_weight"))
        if raw_lambda is None:
            errors.append(f"fold{combo[0]} seed{combo[1]} missing lambda/hier_aux_weight")
        else:
            lambdas.add(float(raw_lambda))

    actual_combos = set(seen)
    actual_folds = {fold for fold, _ in actual_combos}
    actual_seeds = {seed for _, seed in actual_combos}
    duplicate_combos = sorted(combo for combo, count in seen.items() if count > 1)
    if duplicate_combos:
        errors.append(f"duplicate fold-seed combinations: {duplicate_combos}")

    if expected_fold_set != required_folds:
        errors.append(f"folds must be exactly {list(REQUIRED_FOLDS)} for an exact prototype aggregate")
    if actual_folds != required_folds:
        unexpected_folds = sorted(actual_folds - required_folds)
        missing_folds = sorted(required_folds - actual_folds)
        if unexpected_folds:
            errors.append(f"unexpected fold values: {unexpected_folds}")
        if missing_folds:
            errors.append(f"missing required folds: {missing_folds}")

    if expected_seed_set != actual_seeds:
        unexpected_seeds = sorted(actual_seeds - expected_seed_set)
        missing_seeds = sorted(expected_seed_set - actual_seeds)
        if unexpected_seeds:
            errors.append(f"unexpected seed values: {unexpected_seeds}")
        if missing_seeds:
            errors.append(f"missing expected seed values: {missing_seeds}")

    if actual_seeds:
        expected_combos = {(fold, seed) for fold in required_folds for seed in actual_seeds}
        missing_combos = sorted(expected_combos - actual_combos)
        extra_combos = sorted(actual_combos - expected_combos)
        if missing_combos:
            errors.append(f"missing fold-seed combinations: {missing_combos}")
        if extra_combos:
            errors.append(f"unexpected fold-seed combinations: {extra_combos}")

    if expected_lambda is not None and lambdas != {float(expected_lambda)}:
        errors.append(f"lambda mismatch or mixed lambda values: expected {expected_lambda}, got {sorted(lambdas)}")
    elif len(lambdas) != 1:
        errors.append(f"mixed lambda values: {sorted(lambdas)}")

    n_runs = len(records)
    seed_count = len(actual_seeds)
    screening = actual_folds == required_folds and seed_count == 3 and n_runs == 15
    final_25 = actual_folds == required_folds and seed_count == 5 and n_runs == 25
    if not screening and not final_25:
        errors.append("unsupported aggregate protocol: only 5 folds x 3 seeds screening or 5 folds x 5 seeds final is legal")

    if errors:
        raise RuntimeError("Aggregate validation failed: " + "; ".join(errors))

    return {
        "run_scope": "aggregate",
        "screening_result": bool(screening),
        "final_25_run_result": bool(final_25),
        "paper_candidate_result": True,
        "paper_main_result": bool(final_25),
        "n_runs": int(n_runs),
        "folds": list(REQUIRED_FOLDS),
        "seeds": sorted(int(x) for x in actual_seeds),
        "lambda": float(next(iter(lambdas))),
    }


def aggregate_method_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = SUMMARY_METHODS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        values = {key: [] for key in METRIC_KEYS}
        for record in records:
            metrics = _method_metrics(record, method)
            for key in METRIC_KEYS:
                values[key].append(_metric_value(metrics, key))
        rows.append(
            {
                "method": method,
                "n": len(records),
                "mean_macro_f1": _mean(values["macro_f1"]),
                "std_macro_f1": _std(values["macro_f1"]),
                "min_macro_f1": float(np.min(values["macro_f1"])),
                "max_macro_f1": float(np.max(values["macro_f1"])),
                "mean_top1_acc": _mean(values["top1_acc"]),
                "std_top1_acc": _std(values["top1_acc"]),
                "mean_top2_acc": _mean(values["top2_acc"]),
                "mean_ece": _mean(values["ece"]),
                "mean_brier": _mean(values["brier"]),
                "mean_nll": _mean(values["nll"]),
            }
        )
    return rows


def _paired_deltas(records: Sequence[Mapping[str, Any]], left: str, right: str) -> np.ndarray:
    sorted_records = sorted(records, key=_canonical_combo)
    deltas = []
    for record in sorted_records:
        left_metrics = _method_metrics(record, left)
        right_metrics = _method_metrics(record, right)
        deltas.append(_metric_value(left_metrics, "macro_f1") - _metric_value(right_metrics, "macro_f1"))
    return np.asarray(deltas, dtype=float)


def _bootstrap_mean_ci(
    deltas: np.ndarray,
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[float, float]:
    if deltas.size == 0:
        raise RuntimeError("No paired deltas available for bootstrap.")
    rng = np.random.default_rng(random_seed)
    samples = rng.integers(0, deltas.size, size=(int(bootstrap_samples), deltas.size))
    means = deltas[samples].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _wilcoxon_p(deltas: np.ndarray) -> float:
    if deltas.size == 0 or np.allclose(deltas, 0.0):
        return 1.0
    try:
        return float(wilcoxon(deltas, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def paired_macro_f1_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    comparisons: Sequence[tuple[str, str]] = PAIRED_COMPARISONS,
    bootstrap_samples: int = 10000,
    random_seed: int = 3407,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        deltas = _paired_deltas(records, left, right)
        ci_low, ci_high = _bootstrap_mean_ci(
            deltas,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )
        rows.append(
            {
                "comparison": f"{left} - {right}",
                "n": int(deltas.size),
                "mean_delta": float(np.mean(deltas)),
                "std_delta": _std(deltas.tolist()),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wilcoxon_p": _wilcoxon_p(deltas),
                "wins": int(np.sum(deltas > 1e-12)),
                "ties": int(np.sum(np.isclose(deltas, 0.0, atol=1e-12))),
                "losses": int(np.sum(deltas < -1e-12)),
            }
        )
    return rows


def _class_prf(cm: np.ndarray, idx: int) -> dict[str, float]:
    tp = float(cm[idx, idx])
    fp = float(cm[:, idx].sum() - tp)
    fn = float(cm[idx, :].sum() - tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _read_confusion_json(record: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_path = record.get("confusion_matrices_json")
    if not raw_path:
        metrics_json = record.get("metrics_json")
        if metrics_json:
            raw_path = str(Path(metrics_json).parent / "confusion_matrices.json")
    if not raw_path:
        raise FileNotFoundError("confusion_matrices.json path is missing from aggregate record")
    path = Path(str(raw_path))
    if not path.exists():
        raise FileNotFoundError(f"confusion_matrices.json not found: {path}")
    return read_json(path)


def aggregate_confusion_matrices(
    records: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = CONFUSION_METHODS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        labels: list[str] | None = None
        total: np.ndarray | None = None
        for record in records:
            payload = _read_confusion_json(record)
            if method not in payload:
                raise RuntimeError(f"missing confusion matrix for {method}")
            item = payload[method]
            current_labels = [str(x) for x in item.get("labels", [])]
            matrix = np.asarray(item.get("matrix"), dtype=np.int64)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise RuntimeError(f"invalid confusion matrix shape for {method}: {matrix.shape}")
            if labels is None:
                labels = current_labels
                total = matrix.copy()
            else:
                if current_labels != labels:
                    raise RuntimeError(f"label order mismatch while aggregating {method}")
                total = total + matrix
        if labels is None or total is None:
            raise RuntimeError("No confusion matrices provided.")
        if "feeding" not in labels or "stress_vocal" not in labels:
            raise RuntimeError("Confusion labels must include feeding and stress_vocal.")
        f1s = [_class_prf(total, idx)["f1"] for idx in range(len(labels))]
        feeding_idx = labels.index("feeding")
        stress_idx = labels.index("stress_vocal")
        feeding = _class_prf(total, feeding_idx)
        stress = _class_prf(total, stress_idx)
        rows.append(
            {
                "method": method,
                "n_runs": len(records),
                "macro_f1_from_cm": float(np.mean(f1s)),
                "feeding_f1": feeding["f1"],
                "stress_f1": stress["f1"],
                "feeding_recall": feeding["recall"],
                "stress_recall": stress["recall"],
                "feeding_to_stress": int(total[feeding_idx, stress_idx]),
                "stress_to_feeding": int(total[stress_idx, feeding_idx]),
            }
        )
    return rows


def _resolve_confusion_path(metrics_path: Path, metrics: Mapping[str, Any]) -> str:
    default_path = metrics_path.parent / "confusion_matrices.json"
    if default_path.exists():
        return str(default_path)
    output_path = metrics.get("outputs", {}).get("confusion_matrices_json")
    return str(output_path) if output_path else str(default_path)


def _populated(record: Mapping[str, Any], field: str) -> bool:
    value = record.get(field)
    return value is not None and bool(str(value).strip())


def _context_from_record(record: Mapping[str, Any]) -> float | None:
    values: dict[str, float] = {}
    for field in ("dur_s", "context_seconds"):
        if not _populated(record, field):
            continue
        value = record[field]
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a positive finite number")
        try:
            context = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field} must be a positive finite number, got {value!r}"
            ) from exc
        if not np.isfinite(context) or context <= 0:
            raise ValueError(
                f"{field} must be a positive finite number, got {value!r}"
            )
        values[field] = context
    if len(values) == 2 and not np.isclose(
        values["dur_s"], values["context_seconds"], rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "Conflicting dur_s/context_seconds provenance: "
            f"dur_s={values['dur_s']}, "
            f"context_seconds={values['context_seconds']}."
        )
    if not values:
        return None
    return next(iter(values.values()))


def _source_nomenclature(record: Mapping[str, Any]) -> dict[str, object]:
    """Validate explicit metadata or preserve populated legacy provenance."""

    context = _context_from_record(record)
    if _populated(record, "nomenclature_schema_version"):
        route_fields = set(METHOD_METADATA_FIELDS) - set(MODEL_METADATA_FIELDS)
        present_route_fields = route_fields & set(record)
        if set(METHOD_METADATA_FIELDS).issubset(record):
            metadata = validate_method_metadata(record)
        elif not present_route_fields and set(MODEL_METADATA_FIELDS).issubset(
            record
        ):
            metadata = validate_model_metadata(record)
        else:
            missing = sorted(set(METHOD_METADATA_FIELDS) - set(record))
            raise ValueError(
                "Incomplete nomenclature schema-v1 metadata: "
                f"missing {missing}"
            )
        if context is not None and not np.isclose(
            context,
            float(metadata["context_seconds"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Conflicting dur_s/context_seconds provenance: "
                f"dur_s={record.get('dur_s')!r}, "
                f"context_seconds={metadata['context_seconds']!r}."
            )
        return metadata

    metadata: dict[str, object] = {}
    if _populated(record, "model_family"):
        metadata["model_family"] = canonicalize_model_family(
            str(record["model_family"])
        )
    if context is not None:
        metadata["context_seconds"] = context
    if _populated(record, "selection_protocol"):
        metadata["selection_protocol"] = canonicalize_selection_protocol(
            str(record["selection_protocol"])
        )
    return metadata


def _method_identity(storage_id: str, route: str | None) -> str:
    return route if route is not None else f"legacy:{storage_id}"


def _normalise_method_block(
    raw_key: str,
    block: Mapping[str, Any],
    *,
    storage_id: str,
    route: str | None,
    source: Mapping[str, object],
) -> dict[str, Any]:
    normalised = dict(block)
    if _populated(block, "nomenclature_schema_version"):
        route_fields = set(METHOD_METADATA_FIELDS) - set(MODEL_METADATA_FIELDS)
        populated_route_fields = {
            field for field in route_fields if _populated(block, field)
        }
        if route is None and populated_route_fields:
            raise ValueError(
                f"Schema-v1 compatibility method {raw_key!r} must use a "
                "model-only metadata block; unexpected populated route "
                f"metadata fields: {sorted(populated_route_fields)}."
            )
        validated = (
            validate_method_metadata(block)
            if route is not None
            else validate_model_metadata(block)
        )
        normalised.update(validated)
        nested_source: Mapping[str, object] = validated
    else:
        nested_source = _source_nomenclature(block)
        normalised.update(nested_source)

    key_identity = _method_identity(storage_id, route)
    for field in ("method", "selection_method", "legacy_method_id"):
        if not _populated(block, field):
            continue
        nested_storage, nested_route = resolve_inference_method(str(block[field]))
        if _method_identity(nested_storage, nested_route) != key_identity:
            raise ValueError(
                f"Nested {field}={block[field]!r} does not match methods key "
                f"{raw_key!r}."
            )
        normalised[field] = nested_storage
    if _populated(block, "inference_route"):
        nested_route = canonicalize_inference_route(
            str(block["inference_route"])
        )
        if route != nested_route:
            raise ValueError(
                f"Nested inference_route={block['inference_route']!r} does "
                f"not match methods key {raw_key!r}."
            )
        normalised["inference_route"] = nested_route

    for field in ("model_family", "context_seconds", "selection_protocol"):
        if field not in source or field not in nested_source:
            continue
        left = source[field]
        right = nested_source[field]
        matches = (
            bool(
                np.isclose(
                    float(left), float(right), rtol=0.0, atol=1e-12
                )
            )
            if field == "context_seconds"
            else left == right
        )
        if not matches:
            raise ValueError(
                f"Conflicting nested {field} provenance for methods key "
                f"{raw_key!r}: top-level={left!r}, nested={right!r}."
            )
    return normalised


def _normalise_methods(
    methods: Any, *, source: Mapping[str, object]
) -> Mapping[str, Any] | None:
    if methods is None:
        return None
    if not isinstance(methods, Mapping):
        raise ValueError("metrics.methods must be an object")
    normalised: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    for raw_key, block in methods.items():
        storage_id, route = resolve_inference_method(str(raw_key))
        if storage_id in normalised:
            raise ValueError(
                "metrics.methods contains duplicate aliases for "
                f"{storage_id!r}: {aliases[storage_id]!r} and {raw_key!r}."
            )
        if not isinstance(block, Mapping):
            raise ValueError(
                f"metrics.methods[{raw_key!r}] must be an object"
            )
        normalised[storage_id] = _normalise_method_block(
            str(raw_key),
            block,
            storage_id=storage_id,
            route=route,
            source=source,
        )
        aliases[storage_id] = str(raw_key)
    return normalised


def record_from_metrics(metrics_json: str | Path) -> dict[str, Any]:
    metrics_path = require_file(metrics_json, "metrics JSON")
    metrics = read_json(metrics_path)
    source_nomenclature = _source_nomenclature(metrics)
    record: dict[str, Any] = {
        "metrics_json": str(metrics_path),
        "fold": metrics.get("fold"),
        "seed": metrics.get("seed"),
        "feature_backend": metrics.get("feature_backend"),
        "feature_pipeline_equivalent": metrics.get("feature_pipeline_equivalent"),
        "eligible_for_cv_aggregation": metrics.get("eligible_for_cv_aggregation"),
        "input_role": metrics.get("input_role"),
        "leakage_audit_ok": metrics.get("leakage_audit_ok"),
        "manifest_sha_verified": metrics.get("manifest_sha_verified"),
        "softmax_reproduction_passed": metrics.get("softmax_reproduction_passed"),
        "methods": _normalise_methods(
            metrics.get("methods"), source=source_nomenclature
        ),
        "confusion_matrices_json": _resolve_confusion_path(metrics_path, metrics),
    }
    record.update(source_nomenclature)

    metadata_path = metrics_path.parent / "prediction_metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        record["unsafe_allow_checkpoint_sha_mismatch"] = metadata.get(
            "unsafe_allow_checkpoint_sha_mismatch", False
        )
        record["leakage_audit_ok"] = metadata.get("leakage_audit_ok", record["leakage_audit_ok"])
        record["manifest_sha_verified"] = metadata.get("manifest_sha_verified", record["manifest_sha_verified"])

    calibration_path = metrics_path.parent.parent / "calibration" / "calibration.json"
    if calibration_path.exists():
        calibration = read_json(calibration_path)
        record["lambda"] = calibration.get("model_config", {}).get("hier_aux_weight")

    softmax_path = metrics_path.parent.parent / "softmax_reproduction" / "softmax_reproduction.json"
    if softmax_path.exists():
        softmax = read_json(softmax_path)
        record["softmax_reproduction_passed"] = softmax.get(
            "softmax_reproduction_passed",
            record["softmax_reproduction_passed"],
        )
    return record


def flatten_run_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "metrics_json": record.get("metrics_json"),
        "fold": record.get("fold"),
        "seed": record.get("seed"),
        "lambda": record.get("lambda", record.get("hier_aux_weight")),
        "feature_backend": record.get("feature_backend"),
        "input_role": record.get("input_role"),
        "eligible_for_cv_aggregation": record.get("eligible_for_cv_aggregation"),
        "leakage_audit_ok": record.get("leakage_audit_ok"),
        "manifest_sha_verified": record.get("manifest_sha_verified"),
        "softmax_reproduction_passed": record.get("softmax_reproduction_passed"),
        "confusion_matrices_json": record.get("confusion_matrices_json"),
    }
    for method in SUMMARY_METHODS:
        metrics = _method_metrics(record, method)
        for key in METRIC_KEYS:
            row[f"{method}_{key}"] = _metric_value(metrics, key)
    return row


def resolve_aggregate_nomenclature(
    records: Sequence[Mapping[str, Any]],
    *,
    requested_selection_protocol: str | None = None,
) -> dict[str, object]:
    """Inherit homogeneous source metadata, with legacy-safe fallbacks."""

    if not records:
        raise ValueError("Cannot resolve aggregate nomenclature without records")
    sources = [_source_nomenclature(record) for record in records]
    model_families = {
        canonicalize_model_family(
            str(source.get("model_family", "hierarchical_supervision_crnn"))
        )
        for source in sources
    }
    if len(model_families) > 1:
        raise ValueError(
            "Conflicting model_family values in aggregate inputs: "
            f"{sorted(model_families)}"
        )
    model_family = next(iter(model_families))

    contexts: set[float] = set()
    for source in sources:
        value = source.get("context_seconds", 2.0)
        context = float(value)
        if not np.isfinite(context) or context <= 0:
            raise ValueError(
                f"Invalid context_seconds in aggregate input: {value!r}"
            )
        contexts.add(context)
    if len(contexts) > 1:
        raise ValueError(
            "Conflicting context_seconds values in aggregate inputs: "
            f"{sorted(contexts)}"
        )
    context_seconds = next(iter(contexts))

    protocols = {
        reconcile_selection_protocol(
            str(record.get("selection_protocol") or "none"),
            requested=requested_selection_protocol,
            warn_on_legacy=bool(requested_selection_protocol),
        )
        for record in sources
    }
    if len(protocols) > 1:
        raise ValueError(
            "Conflicting selection_protocol values in aggregate inputs: "
            f"{sorted(protocols)}"
        )
    selection_protocol = next(iter(protocols))
    return build_model_metadata(
        model_family=model_family,
        training_stage="b2_validation_selected_hierarchical_crnn",
        context_seconds=context_seconds,
        selection_protocol=selection_protocol,
    )


def aggregate_metadata_for_method(
    method: str,
    aggregate_model_metadata: Mapping[str, Any],
) -> dict[str, object]:
    """Return strictly validated metadata for one aggregate method row."""

    model_metadata = validate_model_metadata(aggregate_model_metadata)
    storage_id, route = resolve_inference_method(method)
    training_stage = (
        "b3_hierarchical_prototype_top1_ablation"
        if route == "hierarchical_prototype_candidate"
        else "b2_validation_selected_hierarchical_crnn"
    )
    if route is None:
        return build_model_metadata(
            model_family=str(model_metadata["model_family"]),
            training_stage=training_stage,
            context_seconds=float(model_metadata["context_seconds"]),
            selection_protocol=str(model_metadata["selection_protocol"]),
        )
    return validate_method_metadata(
        build_method_metadata(
            model_family=str(model_metadata["model_family"]),
            training_stage=training_stage,
            context_seconds=float(model_metadata["context_seconds"]),
            selection_protocol=str(model_metadata["selection_protocol"]),
            inference_route=route,
            legacy_method_id=storage_id,
        )
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize exact "
            + display_label(
                "training_stage",
                "b3_hierarchical_prototype_top1_ablation",
                language="en",
            )
            + " CV runs with aggregate-only paper result provenance."
        )
    )
    ap.add_argument("--metrics_json", nargs="+", required=True, help="Per-run evaluation metrics.json files.")
    ap.add_argument(
        "--out_prefix",
        default="reports/prototype_cv5_exact_w05",
        help="Output prefix before _screening/_final file suffixes.",
    )
    ap.add_argument("--expected_folds", default="0,1,2,3,4")
    ap.add_argument("--expected_seeds", required=True)
    ap.add_argument("--expected_lambda", type=float, default=None)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--bootstrap_seed", type=int, default=3407)
    ap.add_argument(
        "--selection_protocol",
        type=lambda value: canonicalize_selection_protocol(
            value, warn_on_legacy=True
        ),
        default=None,
        help=(
            "Canonical selection protocol for new metadata. Omitted inherits "
            "homogeneous source metadata, falling back to none; it is never "
            "inferred from --expected_lambda."
        ),
    )
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def _output_paths(out_prefix: str | Path, mode: str) -> dict[str, Path]:
    prefix = Path(out_prefix)
    base = f"{prefix}_{mode}"
    return {
        "runs_csv": Path(f"{base}_runs.csv"),
        "summary_csv": Path(f"{base}_summary.csv"),
        "paired_stats_csv": Path(f"{base}_paired_stats.csv"),
        "confusion_summary_csv": Path(f"{base}_confusion_summary.csv"),
        "provenance_json": Path(f"{base}_provenance.json"),
    }


def _check_overwrite(paths: Mapping[str, Path], *, allow_overwrite: bool) -> None:
    for name, path in paths.items():
        if path.exists() and not allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing {name}: {path}")


def main() -> None:
    args = parse_args()
    records = [record_from_metrics(path) for path in args.metrics_json]
    nomenclature = resolve_aggregate_nomenclature(
        records,
        requested_selection_protocol=args.selection_protocol or None,
    )
    aggregate = validate_aggregate_records(
        records,
        expected_folds=parse_int_csv(args.expected_folds, name="expected_folds"),
        expected_seeds=parse_int_csv(args.expected_seeds, name="expected_seeds"),
        expected_lambda=args.expected_lambda,
    )
    mode = "screening" if aggregate["screening_result"] else "final"
    outputs = _output_paths(args.out_prefix, mode)
    _check_overwrite(outputs, allow_overwrite=args.allow_overwrite)

    runs = [flatten_run_record(record) for record in records]
    for row in runs:
        row.update(nomenclature)
        validate_model_metadata(row)
    summary = aggregate_method_metrics(records)
    paired = paired_macro_f1_statistics(
        records,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.bootstrap_seed,
    )
    confusion = aggregate_confusion_matrices(records)

    for row in summary:
        row.update(
            aggregate_metadata_for_method(str(row["method"]), nomenclature)
        )
    for row in confusion:
        row.update(
            aggregate_metadata_for_method(str(row["method"]), nomenclature)
        )
    canonical_methods = [
        validate_method_metadata(
            aggregate_metadata_for_method(method, nomenclature)
        )
        for method in SUMMARY_METHODS
        if resolve_inference_method(method)[1] is not None
    ]
    provenance = {
        "artifact_type": "cv5_exact_prototype_aggregate_provenance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **nomenclature,
        "canonical_methods": canonical_methods,
        **aggregate,
        "methods": list(SUMMARY_METHODS),
        "paired_comparisons": [f"{left} - {right}" for left, right in PAIRED_COMPARISONS],
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "records": runs,
    }
    validate_model_metadata(provenance)

    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(outputs["runs_csv"], index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(outputs["summary_csv"], index=False, encoding="utf-8-sig")
    pd.DataFrame(paired).to_csv(outputs["paired_stats_csv"], index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion).to_csv(outputs["confusion_summary_csv"], index=False, encoding="utf-8-sig")
    write_json(outputs["provenance_json"], provenance)
    print(
        "[INFO] canonical inference routes -> "
        + "; ".join(str(item["display_name_en"]) for item in canonical_methods)
    )
    print(f"[OK] wrote aggregate {mode} summary files with prefix {args.out_prefix}")


if __name__ == "__main__":
    main()
