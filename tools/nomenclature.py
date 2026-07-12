"""Canonical scientific nomenclature for active pig-sound outputs.

The registry deliberately separates model/training identity from inference
route identity.  Historical artifacts keep their original machine keys; new
writers use the helpers here to append versioned canonical metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any


NOMENCLATURE_SCHEMA_VERSION = "pig_sound_nomenclature.v1"

MODEL_FAMILY_IDS = (
    "logmel_crnn",
    "hierarchical_supervision_crnn",
)

TRAINING_STAGE_IDS = (
    "b0_1s_logmel_baseline",
    "b1_2s_logmel_mainline",
    "b2_validation_selected_hierarchical_crnn",
    "b3_hierarchical_prototype_top1_ablation",
)

INFERENCE_ROUTE_IDS = (
    "primary_softmax",
    "main_class_prototype_candidate",
    "hierarchical_prototype_candidate",
)

SELECTION_PROTOCOL_IDS = (
    "none",
    "foldwise_validation_selected_lambda",
    "fixed_lambda_0_5_retrospective",
)

CANONICAL_IDS = {
    "model_family": MODEL_FAMILY_IDS,
    "training_stage": TRAINING_STAGE_IDS,
    "inference_route": INFERENCE_ROUTE_IDS,
    "selection_protocol": SELECTION_PROTOCOL_IDS,
}

_STAGE_MODEL_SEMANTICS = {
    "b0_1s_logmel_baseline": {
        "model_family": "logmel_crnn",
        "context_seconds": 1.0,
        "selection_protocols": {"none"},
        "inference_routes": {"primary_softmax"},
    },
    "b1_2s_logmel_mainline": {
        "model_family": "logmel_crnn",
        "context_seconds": 2.0,
        "selection_protocols": {"none"},
        "inference_routes": {"primary_softmax"},
    },
    "b2_validation_selected_hierarchical_crnn": {
        "model_family": "hierarchical_supervision_crnn",
        "context_seconds": 2.0,
        "selection_protocols": set(SELECTION_PROTOCOL_IDS),
        "inference_routes": {
            "primary_softmax",
            "main_class_prototype_candidate",
        },
    },
    "b3_hierarchical_prototype_top1_ablation": {
        "model_family": "hierarchical_supervision_crnn",
        "context_seconds": 2.0,
        "selection_protocols": set(SELECTION_PROTOCOL_IDS),
        "inference_routes": {"hierarchical_prototype_candidate"},
    },
}

LEGACY_ALIASES = {
    "model_family": {
        "hier_longcontext_crnn": "hierarchical_supervision_crnn",
        "hierarchical_crnn": "hierarchical_supervision_crnn",
    },
    "training_stage": {
        "b0": "b0_1s_logmel_baseline",
        "b1": "b1_2s_logmel_mainline",
        "b2": "b2_validation_selected_hierarchical_crnn",
        "b2_valsel": "b2_validation_selected_hierarchical_crnn",
        "b3": "b3_hierarchical_prototype_top1_ablation",
    },
    "inference_route": {
        "raw_softmax": "primary_softmax",
        "prototype": "main_class_prototype_candidate",
        "main_prototype": "main_class_prototype_candidate",
        "hierarchical": "hierarchical_prototype_candidate",
        "hierarchical_prototype": "hierarchical_prototype_candidate",
    },
    "selection_protocol": {
        "validation_selected": "foldwise_validation_selected_lambda",
        "foldwise_validation_selected": "foldwise_validation_selected_lambda",
        "fixed_lambda_0.5": "fixed_lambda_0_5_retrospective",
        "fixed_lambda=0.5": "fixed_lambda_0_5_retrospective",
        "fixed_lambda_0_5": "fixed_lambda_0_5_retrospective",
        "fixed_w05": "fixed_lambda_0_5_retrospective",
    },
}

DISPLAY_LABELS_EN = {
    "model_family": {
        "logmel_crnn": "Log-Mel CRNN",
        "hierarchical_supervision_crnn": "Hierarchical-supervision CRNN",
    },
    "training_stage": {
        "b0_1s_logmel_baseline": "B0 — 1-s Log-Mel baseline",
        "b1_2s_logmel_mainline": "B1 — 2-s Log-Mel mainline",
        "b2_validation_selected_hierarchical_crnn": (
            "B2 — Validation-selected 2-s hierarchical-supervision CRNN"
        ),
        "b3_hierarchical_prototype_top1_ablation": (
            "B3 — Hierarchical-prototype Top-1 decision ablation"
        ),
    },
    "inference_route": {
        "primary_softmax": "Primary Softmax route (Raw Softmax)",
        "main_class_prototype_candidate": (
            "Main-class prototype candidate route"
        ),
        "hierarchical_prototype_candidate": (
            "Hierarchical prototype candidate route"
        ),
    },
    "selection_protocol": {
        "none": "No selection protocol recorded",
        "foldwise_validation_selected_lambda": (
            "Fold-wise validation-selected λ"
        ),
        "fixed_lambda_0_5_retrospective": "Fixed λ=0.5 retrospective",
    },
}

DISPLAY_LABELS_ZH = {
    "model_family": {
        "logmel_crnn": "Log-Mel CRNN",
        "hierarchical_supervision_crnn": "层级辅助监督 CRNN",
    },
    "training_stage": {
        "b0_1s_logmel_baseline": "B0 — 1 秒 Log-Mel 基线",
        "b1_2s_logmel_mainline": "B1 — 2 秒 Log-Mel 主线",
        "b2_validation_selected_hierarchical_crnn": (
            "B2 — 验证集逐折选择的 2 秒层级辅助监督 CRNN"
        ),
        "b3_hierarchical_prototype_top1_ablation": (
            "B3 — 层级原型 Top-1 决策消融"
        ),
    },
    "inference_route": {
        "primary_softmax": "主 Softmax 路由（原始 Softmax）",
        "main_class_prototype_candidate": "主类别原型候选路由",
        "hierarchical_prototype_candidate": "层级原型候选路由",
    },
    "selection_protocol": {
        "none": "未记录选择协议",
        "foldwise_validation_selected_lambda": "逐折验证集选择 λ",
        "fixed_lambda_0_5_retrospective": "固定 λ=0.5 回顾性",
    },
}

PREFERRED_LEGACY_INFERENCE_ROUTE_IDS = {
    "primary_softmax": "raw_softmax",
    "main_class_prototype_candidate": "prototype",
    "hierarchical_prototype_candidate": "hierarchical",
}

NONCANONICAL_LEGACY_METHOD_IDS = (
    "calibrated_softmax",
    "fused",
)

MODEL_METADATA_FIELDS = (
    "nomenclature_schema_version",
    "model_family",
    "training_stage",
    "context_seconds",
    "selection_protocol",
)

METHOD_METADATA_FIELDS = (
    *MODEL_METADATA_FIELDS,
    "inference_route",
    "canonical_method_id",
    "display_name_en",
    "display_name_zh",
    "legacy_method_id",
)


class UnknownNomenclatureID(ValueError):
    """Raised when a namespace or identifier is outside the registry."""


def _namespace_ids(namespace: str) -> tuple[str, ...]:
    try:
        return CANONICAL_IDS[namespace]
    except KeyError as exc:
        allowed = ", ".join(sorted(CANONICAL_IDS))
        raise UnknownNomenclatureID(
            f"Unknown nomenclature namespace {namespace!r}. "
            f"Expected one of: {allowed}."
        ) from exc


def _normalise_value(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UnknownNomenclatureID(
            f"Nomenclature IDs must be non-empty strings, got {value!r}."
        )
    return value.strip().lower()


def validate_id(namespace: str, value: str) -> str:
    """Validate and return an already-canonical ID."""

    identifiers = _namespace_ids(namespace)
    if value not in identifiers:
        allowed = ", ".join(identifiers)
        raise UnknownNomenclatureID(
            f"Unknown {namespace} ID {value!r}. Expected one of: {allowed}."
        )
    return value


def canonicalize_id(
    namespace: str,
    value: str,
    *,
    warn_on_legacy: bool = False,
) -> str:
    """Resolve a canonical ID or a supported historical alias."""

    identifiers = _namespace_ids(namespace)
    normalised = _normalise_value(value)
    if normalised in identifiers:
        return normalised
    aliases = LEGACY_ALIASES.get(namespace, {})
    canonical = aliases.get(normalised)
    if canonical is None:
        allowed = ", ".join(identifiers)
        raise UnknownNomenclatureID(
            f"Unknown {namespace} ID {value!r}. Expected one of: {allowed}."
        )
    if warn_on_legacy:
        warnings.warn(
            f"Legacy {namespace} ID {value!r} is deprecated; use "
            f"{canonical!r}.",
            DeprecationWarning,
            stacklevel=2,
        )
    return canonical


def canonicalize_model_family(
    value: str, *, warn_on_legacy: bool = False
) -> str:
    return canonicalize_id(
        "model_family", value, warn_on_legacy=warn_on_legacy
    )


def canonicalize_training_stage(
    value: str, *, warn_on_legacy: bool = False
) -> str:
    return canonicalize_id(
        "training_stage", value, warn_on_legacy=warn_on_legacy
    )


def canonicalize_inference_route(
    value: str, *, warn_on_legacy: bool = False
) -> str:
    return canonicalize_id(
        "inference_route", value, warn_on_legacy=warn_on_legacy
    )


def try_canonicalize_inference_route(value: str) -> str | None:
    """Return ``None`` for legacy methods outside the canonical route set."""

    try:
        return canonicalize_inference_route(value)
    except UnknownNomenclatureID:
        return None


def canonicalize_selection_protocol(
    value: str, *, warn_on_legacy: bool = False
) -> str:
    return canonicalize_id(
        "selection_protocol", value, warn_on_legacy=warn_on_legacy
    )


def reconcile_selection_protocol(
    recorded: str | None,
    *,
    requested: str | None = None,
    warn_on_legacy: bool = False,
) -> str:
    """Reconcile stored provenance with an optional explicit annotation.

    Missing or ``none`` provenance may be annotated by a caller.  Once a
    concrete protocol has been recorded, a different explicit protocol is a
    provenance conflict and must not silently relabel the artifact.
    """

    recorded_text = str(recorded or "").strip()
    recorded_protocol = canonicalize_selection_protocol(
        recorded_text or "none"
    )
    requested_text = str(requested or "").strip()
    if not requested_text:
        return recorded_protocol
    requested_protocol = canonicalize_selection_protocol(
        requested_text, warn_on_legacy=warn_on_legacy
    )
    if (
        recorded_protocol != "none"
        and requested_protocol != recorded_protocol
    ):
        raise ValueError(
            "Conflicting selection_protocol provenance: artifact records "
            f"{recorded_protocol!r}, but the caller requested "
            f"{requested_protocol!r}."
        )
    return requested_protocol


def display_label(
    namespace: str,
    value: str,
    *,
    language: str = "en",
) -> str:
    """Return the central English or Chinese display label."""

    canonical = canonicalize_id(namespace, value)
    if language == "en":
        labels = DISPLAY_LABELS_EN
    elif language == "zh":
        labels = DISPLAY_LABELS_ZH
    else:
        raise ValueError(
            f"Unsupported display language {language!r}; expected 'en' or 'zh'."
        )
    return labels[namespace][canonical]


def preferred_legacy_inference_route_id(value: str) -> str:
    """Return the existing storage key for a canonical inference route."""

    canonical = canonicalize_inference_route(value)
    return PREFERRED_LEGACY_INFERENCE_ROUTE_IDS[canonical]


def resolve_inference_method(
    value: str, *, warn_on_legacy: bool = False
) -> tuple[str, str | None]:
    """Resolve CLI/result method text without changing legacy storage keys.

    The returned pair is ``(legacy_storage_id, canonical_route_or_none)``.
    Calibrated Softmax and fused results remain supported legacy-only methods;
    they are intentionally not coerced into a canonical inference route.
    """

    normalised = _normalise_value(value)
    if normalised == "softmax":
        if warn_on_legacy:
            warnings.warn(
                "Legacy inference method 'softmax' is deprecated; use "
                "'calibrated_softmax'.",
                DeprecationWarning,
                stacklevel=2,
            )
        return "calibrated_softmax", None
    if normalised in NONCANONICAL_LEGACY_METHOD_IDS:
        return normalised, None
    try:
        route = canonicalize_inference_route(
            normalised, warn_on_legacy=warn_on_legacy
        )
    except UnknownNomenclatureID as exc:
        supported = ", ".join(
            (*INFERENCE_ROUTE_IDS, *NONCANONICAL_LEGACY_METHOD_IDS, "softmax")
        )
        raise UnknownNomenclatureID(
            f"Unknown inference method {value!r}. Supported canonical routes "
            f"and compatibility methods: {supported}."
        ) from exc
    return preferred_legacy_inference_route_id(route), route


def resolve_inference_method_fields(
    record: Mapping[str, Any],
    *,
    fields: Sequence[str] = ("selection_method", "inference_route"),
    warn_on_legacy: bool = False,
) -> tuple[str, str | None]:
    """Resolve all populated method fields and reject contradictory values."""

    resolved: list[tuple[str, str, str | None]] = []
    for field in fields:
        value = record.get(field)
        if value is None or not str(value).strip():
            continue
        storage_id, route = resolve_inference_method(
            str(value), warn_on_legacy=warn_on_legacy
        )
        resolved.append((field, storage_id, route))
    if not resolved:
        raise UnknownNomenclatureID(
            "Record contains no populated inference method fields: "
            + ", ".join(fields)
            + "."
        )
    identities = {
        route if route is not None else f"legacy:{storage_id}"
        for _, storage_id, route in resolved
    }
    if len(identities) != 1:
        detail = ", ".join(
            f"{field}={route or f'legacy:{storage_id}'}"
            for field, storage_id, route in resolved
        )
        raise ValueError(f"Conflicting inference method fields: {detail}")
    _, storage_id, route = resolved[0]
    return storage_id, route


def _context_seconds(value: float | int) -> float:
    if isinstance(value, bool):
        raise ValueError("context_seconds must be a positive finite number")
    try:
        context = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "context_seconds must be a positive finite number"
        ) from exc
    if not math.isfinite(context) or context <= 0:
        raise ValueError("context_seconds must be a positive finite number")
    return context


def build_model_metadata(
    *,
    model_family: str,
    training_stage: str,
    context_seconds: float | int,
    selection_protocol: str,
) -> dict[str, object]:
    """Build canonical metadata for an artifact without one route."""

    family = canonicalize_model_family(model_family)
    stage = canonicalize_training_stage(training_stage)
    context = _context_seconds(context_seconds)
    protocol = canonicalize_selection_protocol(selection_protocol)
    semantics = _STAGE_MODEL_SEMANTICS[stage]
    if family != semantics["model_family"]:
        raise ValueError(
            f"training_stage {stage!r} requires model_family="
            f"{semantics['model_family']!r}, got {family!r}."
        )
    expected_context = float(semantics["context_seconds"])
    if not math.isclose(context, expected_context, abs_tol=1e-12, rel_tol=0.0):
        raise ValueError(
            f"training_stage {stage!r} requires "
            f"context_seconds={expected_context:.1f}, got {context}."
        )
    if protocol not in semantics["selection_protocols"]:
        allowed = ", ".join(sorted(semantics["selection_protocols"]))
        raise ValueError(
            f"training_stage {stage!r} does not permit selection_protocol "
            f"{protocol!r}; expected one of: {allowed}."
        )
    return {
        "nomenclature_schema_version": NOMENCLATURE_SCHEMA_VERSION,
        "model_family": family,
        "training_stage": stage,
        "context_seconds": context,
        "selection_protocol": protocol,
    }


def build_method_metadata(
    *,
    model_family: str,
    training_stage: str,
    context_seconds: float | int,
    selection_protocol: str,
    inference_route: str,
    legacy_method_id: str | None = None,
) -> dict[str, object]:
    """Build all canonical fields for one inference route."""

    metadata = build_model_metadata(
        model_family=model_family,
        training_stage=training_stage,
        context_seconds=context_seconds,
        selection_protocol=selection_protocol,
    )
    route = canonicalize_inference_route(inference_route)
    allowed_routes = _STAGE_MODEL_SEMANTICS[
        str(metadata["training_stage"])
    ]["inference_routes"]
    if route not in allowed_routes:
        allowed = ", ".join(sorted(allowed_routes))
        raise ValueError(
            f"training_stage {metadata['training_stage']!r} requires "
            f"inference_route to be one of: {allowed}; got {route!r}."
        )
    if legacy_method_id is None:
        legacy = preferred_legacy_inference_route_id(route)
    else:
        legacy = _normalise_value(legacy_method_id)
        legacy_route = canonicalize_inference_route(legacy)
        if legacy_route != route:
            raise ValueError(
                f"legacy_method_id {legacy_method_id!r} resolves to "
                f"{legacy_route!r}, not {route!r}."
            )

    stage = str(metadata["training_stage"])
    protocol = str(metadata["selection_protocol"])
    metadata.update(
        {
            "inference_route": route,
            "canonical_method_id": f"{stage}::{protocol}::{route}",
            "display_name_en": display_label(
                "inference_route", route, language="en"
            ),
            "display_name_zh": display_label(
                "inference_route", route, language="zh"
            ),
            "legacy_method_id": legacy,
        }
    )
    return metadata


def metadata_from_legacy_record(
    record: Mapping[str, Any],
    *,
    model_family: str,
    training_stage: str,
    context_seconds: float | int,
    selection_protocol: str,
    method_keys: Sequence[str] = (
        "legacy_method_id",
        "method",
        "selection_method",
        "inference_route",
    ),
    warn_on_legacy: bool = False,
) -> dict[str, Any]:
    """Return a copy of a legacy record enriched with canonical metadata."""

    resolved: list[tuple[str, str, str]] = []
    unsupported: list[tuple[str, str]] = []
    for key in method_keys:
        raw = record.get(key)
        if raw is None or not str(raw).strip():
            continue
        raw_text = str(raw).strip()
        _, route = resolve_inference_method(
            raw_text, warn_on_legacy=warn_on_legacy
        )
        if route is None:
            unsupported.append((key, raw_text))
            continue
        resolved.append((key, raw_text, route))
    if unsupported:
        detail = ", ".join(f"{key}={value!r}" for key, value in unsupported)
        raise ValueError(
            "Legacy record contains noncanonical method fields that cannot "
            f"be mapped to a canonical inference route: {detail}"
        )
    if not resolved:
        raise UnknownNomenclatureID(
            "Legacy record contains no canonical inference route."
        )
    routes = {route for _, _, route in resolved}
    if len(routes) != 1:
        detail = ", ".join(
            f"{key}={value!r}->{route!r}"
            for key, value, route in resolved
        )
        raise ValueError(f"Conflicting inference routes in legacy record: {detail}")
    _, legacy, route = resolved[0]
    if _normalise_value(legacy) in INFERENCE_ROUTE_IDS:
        legacy = preferred_legacy_inference_route_id(route)
    enriched = dict(record)
    enriched.update(
        build_method_metadata(
            model_family=model_family,
            training_stage=training_stage,
            context_seconds=context_seconds,
            selection_protocol=selection_protocol,
            inference_route=route,
            legacy_method_id=legacy,
        )
    )
    return enriched


def validate_method_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, object]:
    """Validate a serialized canonical method-metadata block."""

    missing = [field for field in METHOD_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"Missing canonical method metadata fields: {missing}")
    if metadata["nomenclature_schema_version"] != NOMENCLATURE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported nomenclature_schema_version "
            f"{metadata['nomenclature_schema_version']!r}; expected "
            f"{NOMENCLATURE_SCHEMA_VERSION!r}."
        )
    expected = build_method_metadata(
        model_family=str(metadata["model_family"]),
        training_stage=str(metadata["training_stage"]),
        context_seconds=metadata["context_seconds"],
        selection_protocol=str(metadata["selection_protocol"]),
        inference_route=str(metadata["inference_route"]),
        legacy_method_id=str(metadata["legacy_method_id"]),
    )
    mismatches = {
        field: (metadata[field], expected[field])
        for field in METHOD_METADATA_FIELDS
        if metadata[field] != expected[field]
    }
    if mismatches:
        raise ValueError(f"Invalid canonical method metadata: {mismatches}")
    return expected


def validate_model_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, object]:
    """Validate a serialized route-independent model-metadata block."""

    missing = [field for field in MODEL_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"Missing canonical model metadata fields: {missing}")
    if metadata["nomenclature_schema_version"] != NOMENCLATURE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported nomenclature_schema_version "
            f"{metadata['nomenclature_schema_version']!r}; expected "
            f"{NOMENCLATURE_SCHEMA_VERSION!r}."
        )
    expected = build_model_metadata(
        model_family=str(metadata["model_family"]),
        training_stage=str(metadata["training_stage"]),
        context_seconds=metadata["context_seconds"],
        selection_protocol=str(metadata["selection_protocol"]),
    )
    mismatches = {
        field: (metadata[field], expected[field])
        for field in MODEL_METADATA_FIELDS
        if metadata[field] != expected[field]
    }
    if mismatches:
        raise ValueError(f"Invalid canonical model metadata: {mismatches}")
    return expected


def validate_recorded_nomenclature_metadata(
    record: Mapping[str, Any],
) -> dict[str, object] | None:
    """Strictly validate an explicitly versioned metadata block.

    Legacy records without ``nomenclature_schema_version`` remain readable.
    Once a record declares a schema version, however, it must be a complete
    route-specific method block or a complete route-independent model block;
    callers must not silently regenerate corrupt canonical fields.
    """

    raw_version = record.get("nomenclature_schema_version")
    if raw_version is None or not str(raw_version).strip():
        return None
    if raw_version != NOMENCLATURE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported nomenclature_schema_version "
            f"{raw_version!r}; expected {NOMENCLATURE_SCHEMA_VERSION!r}."
        )
    route_fields = METHOD_METADATA_FIELDS[len(MODEL_METADATA_FIELDS) :]
    has_route_metadata = any(
        field in record
        and record[field] is not None
        and bool(str(record[field]).strip())
        for field in route_fields
    )
    if not has_route_metadata:
        for field in ("selection_method", "method"):
            value = record.get(field)
            if value is None or not str(value).strip():
                continue
            try:
                _, route = resolve_inference_method(str(value))
            except ValueError:
                continue
            if route is not None:
                has_route_metadata = True
                break
    if has_route_metadata:
        return validate_method_metadata(record)
    return validate_model_metadata(record)


def validate_recorded_artifact_identity(
    record: Mapping[str, Any],
    *,
    expected_model_family: str,
    expected_training_stage: str,
    expected_context_seconds: float | int,
    context: str,
) -> dict[str, object] | None:
    """Validate an explicit block against the artifact role that reads it."""

    validated = validate_recorded_nomenclature_metadata(record)
    if validated is None:
        legacy: dict[str, object] = {}
        if record.get("model_family") is not None and str(
            record["model_family"]
        ).strip():
            legacy["model_family"] = canonicalize_model_family(
                str(record["model_family"])
            )
        if record.get("training_stage") is not None and str(
            record["training_stage"]
        ).strip():
            legacy["training_stage"] = canonicalize_training_stage(
                str(record["training_stage"])
            )
        contexts: dict[str, float] = {}
        for field in ("dur_s", "context_seconds"):
            value = record.get(field)
            if value is None or not str(value).strip():
                continue
            contexts[field] = _context_seconds(value)
        if len(contexts) == 2 and not math.isclose(
            contexts["dur_s"],
            contexts["context_seconds"],
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise ValueError(
                f"{context} has conflicting dur_s/context_seconds: "
                f"{contexts}"
            )
        if contexts:
            legacy["context_seconds"] = next(iter(contexts.values()))
        protocol = record.get("selection_protocol")
        if protocol is not None and str(protocol).strip():
            legacy["selection_protocol"] = canonicalize_selection_protocol(
                str(protocol)
            )
        if not legacy:
            return None
        validated = legacy
    recorded_duration = record.get("dur_s")
    if (
        recorded_duration is not None
        and str(recorded_duration).strip()
        and "context_seconds" in validated
    ):
        duration = _context_seconds(recorded_duration)
        if not math.isclose(
            duration,
            float(validated["context_seconds"]),
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise ValueError(
                f"{context} has dur_s={duration}, which conflicts with "
                "canonical context_seconds="
                f"{validated['context_seconds']}."
            )
    expected = {
        "model_family": canonicalize_model_family(expected_model_family),
        "training_stage": canonicalize_training_stage(expected_training_stage),
        "context_seconds": _context_seconds(expected_context_seconds),
    }
    mismatches: dict[str, tuple[object, object]] = {}
    for field, expected_value in expected.items():
        if field not in validated:
            continue
        actual = validated[field]
        matches = (
            math.isclose(
                float(actual),
                float(expected_value),
                abs_tol=1e-12,
                rel_tol=0.0,
            )
            if field == "context_seconds"
            else actual == expected_value
        )
        if not matches:
            mismatches[field] = (actual, expected_value)
    if mismatches:
        raise ValueError(
            f"{context} canonical identity conflicts with its artifact role: "
            f"{mismatches}"
        )
    return validated


def validate_expected_artifact_role(
    record: Mapping[str, Any],
    *,
    expected_model_family: str,
    expected_training_stage: str,
    expected_context_seconds: float | int,
    expected_selection_protocol: str | None = None,
    expected_inference_route: str | None = None,
    route_fields: Sequence[str] = (
        "selection_method",
        "inference_route",
        "legacy_method_id",
    ),
    context: str,
) -> dict[str, object] | None:
    """Bind recorded canonical or legacy fields to one artifact role."""

    validated = validate_recorded_artifact_identity(
        record,
        expected_model_family=expected_model_family,
        expected_training_stage=expected_training_stage,
        expected_context_seconds=expected_context_seconds,
        context=context,
    )

    if expected_selection_protocol is not None:
        expected_protocol = canonicalize_selection_protocol(
            expected_selection_protocol
        )
        recorded_protocol = (
            validated.get("selection_protocol")
            if validated is not None
            else None
        )
        if (
            recorded_protocol is not None
            and recorded_protocol != expected_protocol
        ):
            raise ValueError(
                f"{context} selection_protocol conflicts with its artifact "
                f"role: recorded={recorded_protocol!r}, "
                f"expected={expected_protocol!r}."
            )

    populated_route_fields = tuple(
        field
        for field in route_fields
        if record.get(field) is not None
        and bool(str(record[field]).strip())
    )
    if populated_route_fields:
        _, recorded_route = resolve_inference_method_fields(
            record,
            fields=populated_route_fields,
        )
        if expected_inference_route is not None:
            expected_route = canonicalize_inference_route(
                expected_inference_route
            )
            if recorded_route != expected_route:
                raise ValueError(
                    f"{context} inference route conflicts with its artifact "
                    f"role: recorded={recorded_route!r}, "
                    f"expected={expected_route!r}."
                )

    return validated


def main(argv: Sequence[str] | None = None) -> int:
    """Print the registry for inspection; no project artifact is modified."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect canonical pig-sound model and inference nomenclature "
            f"({NOMENCLATURE_SCHEMA_VERSION})."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete canonical registry as JSON.",
    )
    args = parser.parse_args(argv)
    if args.json:
        payload = {
            "nomenclature_schema_version": NOMENCLATURE_SCHEMA_VERSION,
            "canonical_ids": CANONICAL_IDS,
            "display_labels_en": DISPLAY_LABELS_EN,
            "display_labels_zh": DISPLAY_LABELS_ZH,
            "legacy_aliases": LEGACY_ALIASES,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
    return 0


__all__ = [
    "CANONICAL_IDS",
    "DISPLAY_LABELS_EN",
    "DISPLAY_LABELS_ZH",
    "INFERENCE_ROUTE_IDS",
    "LEGACY_ALIASES",
    "METHOD_METADATA_FIELDS",
    "MODEL_METADATA_FIELDS",
    "MODEL_FAMILY_IDS",
    "NOMENCLATURE_SCHEMA_VERSION",
    "NONCANONICAL_LEGACY_METHOD_IDS",
    "PREFERRED_LEGACY_INFERENCE_ROUTE_IDS",
    "SELECTION_PROTOCOL_IDS",
    "TRAINING_STAGE_IDS",
    "UnknownNomenclatureID",
    "build_method_metadata",
    "build_model_metadata",
    "canonicalize_id",
    "canonicalize_inference_route",
    "canonicalize_model_family",
    "canonicalize_selection_protocol",
    "canonicalize_training_stage",
    "display_label",
    "metadata_from_legacy_record",
    "preferred_legacy_inference_route_id",
    "reconcile_selection_protocol",
    "resolve_inference_method",
    "resolve_inference_method_fields",
    "try_canonicalize_inference_route",
    "validate_id",
    "validate_method_metadata",
    "validate_model_metadata",
    "validate_expected_artifact_role",
    "validate_recorded_nomenclature_metadata",
    "validate_recorded_artifact_identity",
]


if __name__ == "__main__":
    raise SystemExit(main())
