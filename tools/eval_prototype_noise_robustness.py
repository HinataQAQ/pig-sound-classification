from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from mix_audio_at_snr import (
    deterministic_seed,
    file_md5,
    file_sha256,
    load_audio_with_valid_region,
    load_noise_channel,
    mix_clean_with_noise_at_snr,
    noise_offset_key,
    select_noise_segment,
    validate_noise_source_for_paper,
)
from prototype_model_adapter import (
    align_prediction_frames_by_path,
    assert_no_leakage,
    audit_manifest_disjointness,
    build_hier_dataset,
    calibration_float_param,
    calibration_metrics,
    config_from_summary,
    coverage_risk_curve,
    fuse_probabilities,
    load_hier_model,
    load_prototype_bundle,
    multiclass_metrics,
    normalize_identity_path,
    prediction_frame,
    prototype_scores_from_bundle,
    read_json,
    require_file,
    resolve_recorded_file,
    select_per_class_thresholds,
    select_threshold_for_target_coverage,
    temperature_scale_probabilities,
    verify_manifest_matches_metadata,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from train_mctafd_crnn_ablation import make_feature  # noqa: E402
from train_hier_longcontext_crnn import resolve_path as resolve_audio_path  # noqa: E402


METHODS = ("raw_softmax", "prototype", "hierarchical", "fused")
THRESHOLD_METHODS = ("raw_softmax", "prototype", "hierarchical", "fused")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_out_dir(path: Path, allow_overwrite: bool) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    planned = [
        path / "noise_predictions.csv",
        path / "noise_sample_provenance.csv",
        path / "noise_metrics.json",
        path / "metrics_by_condition.csv",
        path / "clean_equivalence.json",
        path / "confusion_by_condition_method.json",
        path / "snr_by_class.csv",
        path / "provenance_gates.json",
    ]
    for p in planned:
        if p.exists() and not allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {p}")
    return path


def feature_from_waveform(y: np.ndarray, config) -> np.ndarray:
    return make_feature(
        y,
        feature_mode=config.feature_mode,
        sr=config.sr,
        n_mels=config.n_mels,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        fmin=config.fmin,
        fmax=config.fmax,
    )


def infer_from_features(model, features: list[np.ndarray], device: str, batch_size: int = 32) -> dict[str, np.ndarray]:
    import torch

    embeddings: list[np.ndarray] = []
    main_probs: list[np.ndarray] = []
    aux_probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = np.stack(features[start : start + batch_size], axis=0)
            xb = torch.from_numpy(batch).float().to(device)
            z = model.encode(xb)
            main_logits = model.main_head(z)
            aux_logits = model.aux_head(z)
            embeddings.append(z.detach().cpu().numpy().astype(np.float32))
            main_probs.append(torch.softmax(main_logits, dim=1).detach().cpu().numpy().astype(np.float32))
            aux_probs.append(torch.softmax(aux_logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return {
        "embeddings": np.concatenate(embeddings, axis=0),
        "softmax_probs": np.concatenate(main_probs, axis=0),
        "aux_softmax_probs": np.concatenate(aux_probs, axis=0),
    }


def method_probabilities(extracted: Mapping[str, np.ndarray], bundle: Mapping[str, Any], calibration: Mapping[str, Any]) -> dict[str, np.ndarray]:
    calibrated_softmax_temperature = calibration_float_param(
        calibration,
        "calibrated_softmax",
        "softmax_temperature",
        fallback_keys=("softmax_temperature",),
        default=1.0,
    )
    calibrated_softmax_probs = temperature_scale_probabilities(
        extracted["softmax_probs"],
        calibrated_softmax_temperature,
    )
    prototype_temperature = calibration_float_param(
        calibration,
        "prototype",
        "prototype_temperature",
        fallback_keys=("prototype_temperature", "temperature"),
        default=1.0,
    )
    hierarchical_temperature = calibration_float_param(
        calibration,
        "hierarchical",
        "prototype_temperature",
        fallback_keys=("prototype_temperature", "temperature"),
        default=prototype_temperature,
    )
    hierarchical_aux_weight = calibration_float_param(
        calibration,
        "hierarchical",
        "hier_aux_prob_weight",
        fallback_keys=("hier_aux_prob_weight",),
        default=0.5,
    )
    proto_scores = prototype_scores_from_bundle(
        extracted["embeddings"],
        bundle,
        temperature=prototype_temperature,
        hier_aux_weight=hierarchical_aux_weight,
    )
    hier_scores = prototype_scores_from_bundle(
        extracted["embeddings"],
        bundle,
        temperature=hierarchical_temperature,
        hier_aux_weight=hierarchical_aux_weight,
    )
    fusion_alpha = calibration_float_param(
        calibration,
        "fused",
        "softmax_weight",
        fallback_keys=("softmax_weight",),
        default=0.0,
    )
    fused = fuse_probabilities(calibrated_softmax_probs, hier_scores["hierarchical"], softmax_weight=fusion_alpha)
    return {
        "raw_softmax": extracted["softmax_probs"],
        "calibrated_softmax": calibrated_softmax_probs,
        "prototype": proto_scores["prototype"],
        "hierarchical": hier_scores["hierarchical"],
        "fused": fused,
        "aux_prototype": hier_scores["aux_prototype"],
        "main_cosine_similarity": hier_scores["main_cosine_similarity"],
    }


def condition_label(environment: str, snr: float | None) -> str:
    return "clean" if environment == "clean" else f"{environment}_{snr:g}dB"


def _json_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _safe_std(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.astype(float).std(ddof=1))


def aurc_score(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    order = np.argsort(-confidence)
    correct = (y_true[order] == y_pred[order]).astype(np.float64)
    coverage = np.arange(1, len(y_true) + 1, dtype=np.float64) / float(len(y_true))
    risk = 1.0 - np.cumsum(correct) / np.arange(1, len(y_true) + 1, dtype=np.float64)
    return float(np.trapezoid(risk, coverage))


def risk_at_coverages(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    coverages: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=np.float64)
    if len(y_true) == 0:
        return {f"risk_at_coverage_{c:.2f}": 0.0 for c in coverages}
    order = np.argsort(-confidence)
    out: dict[str, float] = {}
    for coverage in coverages:
        k = max(1, min(len(y_true), int(math.ceil(float(coverage) * len(y_true)))))
        accepted = order[:k]
        risk = 1.0 - float((y_true[accepted] == y_pred[accepted]).mean())
        out[f"risk_at_coverage_{float(coverage):.2f}"] = risk
    return out


def compute_method_rejection_thresholds(
    *,
    val_predictions: pd.DataFrame,
    labels: list[str],
    target_coverage: float,
    source_validation_predictions_path: Path | None,
    methods: tuple[str, ...] = THRESHOLD_METHODS,
) -> dict[str, dict[str, Any]]:
    source_sha = file_sha256(source_validation_predictions_path) if source_validation_predictions_path else None
    thresholds: dict[str, dict[str, Any]] = {}
    for method in methods:
        pred_col = f"{method}_pred_id"
        conf_col = f"{method}_confidence"
        if pred_col not in val_predictions.columns or conf_col not in val_predictions.columns:
            raise RuntimeError(f"Validation predictions missing columns for {method}: {pred_col}, {conf_col}")
        y_pred = val_predictions[pred_col].to_numpy(dtype=np.int64)
        confidence = val_predictions[conf_col].to_numpy(dtype=np.float32)
        global_threshold = select_threshold_for_target_coverage(confidence, target_coverage)
        thresholds[method] = {
            "method": method,
            "global_threshold": float(global_threshold),
            "per_class_thresholds": select_per_class_thresholds(
                y_pred,
                confidence,
                labels,
                target_coverage,
                min_count=1,
                fallback_threshold=global_threshold,
            ),
            "source_validation_predictions": str(source_validation_predictions_path) if source_validation_predictions_path else None,
            "source_validation_predictions_sha256": source_sha,
            "target_coverage": float(target_coverage),
            "threshold_selection_formula": "quantile(confidence, 1 - target_coverage)",
            "source_split": "clean_validation",
            "noisy_validation_used": False,
            "noisy_test_used_for_threshold_selection": False,
        }
    return thresholds


def threshold_metrics_for_method(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    labels: list[str],
    threshold_info: Mapping[str, Any] | None,
) -> dict[str, float | None]:
    if not threshold_info:
        return {
            "frozen_global_threshold": None,
            "frozen_threshold_coverage": None,
            "frozen_threshold_selective_risk": None,
            "frozen_per_class_threshold_coverage": None,
            "frozen_per_class_threshold_selective_risk": None,
        }
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=np.float64)
    global_threshold = float(threshold_info["global_threshold"])
    accepted = confidence >= global_threshold

    def cov_risk(mask: np.ndarray) -> tuple[float, float | None]:
        coverage = float(mask.mean()) if len(mask) else 0.0
        if not mask.any():
            return coverage, None
        return coverage, float(1.0 - (y_true[mask] == y_pred[mask]).mean())

    global_cov, global_risk = cov_risk(accepted)
    per_class_thresholds = threshold_info.get("per_class_thresholds", {})
    per_class_accept = np.array(
        [
            confidence[i] >= float(per_class_thresholds.get(labels[int(pred)], global_threshold))
            for i, pred in enumerate(y_pred)
        ],
        dtype=bool,
    )
    per_class_cov, per_class_risk = cov_risk(per_class_accept)
    return {
        "frozen_global_threshold": global_threshold,
        "frozen_threshold_coverage": global_cov,
        "frozen_threshold_selective_risk": global_risk,
        "frozen_per_class_threshold_coverage": per_class_cov,
        "frozen_per_class_threshold_selective_risk": per_class_risk,
    }


def class_metrics_and_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, float | int], list[list[int]]]:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    label_ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    out: dict[str, float | int] = {}
    for idx, label in enumerate(labels):
        out[f"{label}_precision"] = float(precision[idx])
        out[f"{label}_recall"] = float(recall[idx])
        out[f"{label}_f1"] = float(f1[idx])
        out[f"{label}_support"] = int(support[idx])
    if "feeding" in labels and "stress_vocal" in labels:
        feeding = labels.index("feeding")
        stress = labels.index("stress_vocal")
        out["feeding_to_stress"] = int(cm[feeding, stress])
        out["stress_to_feeding"] = int(cm[stress, feeding])
    if "cough" in labels:
        cough = labels.index("cough")
        for idx, label in enumerate(labels):
            if idx != cough:
                out[f"cough_to_{label}"] = int(cm[cough, idx])
    return out, cm.astype(int).tolist()


def compute_snr_stats_by_class(provenance: pd.DataFrame) -> list[dict[str, Any]]:
    if provenance.empty or "target_active_snr_db" not in provenance.columns:
        return []
    noisy = provenance[provenance["simulated_noise"].astype(bool)].copy()
    if noisy.empty:
        return []
    for col in ("target_active_snr_db", "achieved_active_snr_db", "achieved_full_window_snr_db", "valid_duration_ratio"):
        noisy[col] = pd.to_numeric(noisy[col], errors="raise")
    rows: list[dict[str, Any]] = []
    group_cols = ["noise_environment", "target_active_snr_db", "y_true"]
    for keys, group in noisy.groupby(group_cols, dropna=False):
        environment, snr, y_true_label = keys
        rows.append(
            {
                "noise_environment": environment,
                "target_active_snr_db": float(snr),
                "snr_reference": "active_valid_region",
                "y_true": y_true_label,
                "n": int(len(group)),
                "active_snr_mean": float(group["achieved_active_snr_db"].mean()),
                "active_snr_std": _safe_std(group["achieved_active_snr_db"]),
                "active_snr_min": float(group["achieved_active_snr_db"].min()),
                "active_snr_max": float(group["achieved_active_snr_db"].max()),
                "full_window_snr_mean": float(group["achieved_full_window_snr_db"].mean()),
                "full_window_snr_std": _safe_std(group["achieved_full_window_snr_db"]),
                "full_window_snr_min": float(group["achieved_full_window_snr_db"].min()),
                "full_window_snr_max": float(group["achieved_full_window_snr_db"].max()),
                "valid_duration_ratio_mean": float(group["valid_duration_ratio"].mean()),
                "valid_duration_ratio_std": _safe_std(group["valid_duration_ratio"]),
            }
        )
    return rows


def verify_selected_channel(cli_selected_channel: int | None, row: Mapping[str, Any]) -> int:
    manifest_channel = int(row["selected_channel"])
    if cli_selected_channel is not None and int(cli_selected_channel) != manifest_channel:
        raise ValueError(
            f"selected_channel mismatch: CLI selected_channel={cli_selected_channel}, "
            f"manifest selected_channel={manifest_channel}"
        )
    return manifest_channel


def noise_result_qualification(
    *,
    feature_backend: str,
    clean_equivalence_ok: bool,
    leakage_audit_ok: bool,
    manifest_sha_verified: bool,
    unsafe_allow_checkpoint_sha_mismatch: bool,
    provenance_gates_ok: bool = True,
) -> dict[str, Any]:
    feature_pipeline_equivalent = feature_backend == "training_exact"
    eligible = bool(
        feature_pipeline_equivalent
        and clean_equivalence_ok
        and leakage_audit_ok
        and manifest_sha_verified
        and provenance_gates_ok
        and not unsafe_allow_checkpoint_sha_mismatch
    )
    return {
        "feature_backend": feature_backend,
        "feature_pipeline_equivalent": feature_pipeline_equivalent,
        "input_role": "frozen_test",
        "run_scope": "fold_seed",
        "single_fold_debug": True,
        "eligible_for_cv_aggregation": eligible,
        "paper_candidate_result": False,
        "paper_main_result": False,
        "screening_result": False,
        "final_25_run_result": False,
        "manifest_sha_verified": bool(manifest_sha_verified),
        "unsafe_allow_checkpoint_sha_mismatch": bool(unsafe_allow_checkpoint_sha_mismatch),
        "provenance_gates_ok": bool(provenance_gates_ok),
        "eligible_for_noise_aggregation": eligible,
    }


def compute_condition_metrics(
    *,
    y_true: np.ndarray,
    labels: list[str],
    probs_by_method: Mapping[str, np.ndarray],
    pred_frame: pd.DataFrame,
    calibration: Mapping[str, Any],
    method_thresholds: Mapping[str, Mapping[str, Any]],
    noise_environment: str,
    snr_db: float | None,
    clean_macro_by_method: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for method in METHODS:
        probs = probs_by_method[method]
        cls = multiclass_metrics(y_true, probs, labels=labels)
        cal = calibration_metrics(y_true, probs)
        pred = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        threshold_info = method_thresholds.get(method)
        class_fields, cm = class_metrics_and_confusion(y_true, pred, labels)
        threshold_fields = threshold_metrics_for_method(
            y_true=y_true,
            y_pred=pred,
            confidence=confidence,
            labels=labels,
            threshold_info=threshold_info,
        )
        row: dict[str, Any] = {
            "noise_environment": noise_environment,
            "snr_db": "clean" if snr_db is None else float(snr_db),
            "target_active_snr_db": "clean" if snr_db is None else float(snr_db),
            "snr_reference": "clean" if snr_db is None else "active_valid_region",
            "condition": condition_label(noise_environment, snr_db),
            "method": method,
            **cls,
            **cal,
            **threshold_fields,
            "aurc": aurc_score(y_true, pred, confidence),
            **risk_at_coverages(y_true, pred, confidence),
            **class_fields,
        }
        row["global_coverage"] = row["frozen_threshold_coverage"]
        row["selective_risk"] = row["frozen_threshold_selective_risk"]
        for class_name in ("feeding", "stress_vocal"):
            if class_name in labels:
                class_id = labels.index(class_name)
                support = y_true == class_id
                if threshold_info:
                    threshold = float(threshold_info["global_threshold"])
                else:
                    threshold = 0.0
                accepted = confidence >= threshold
                row[f"{class_name}_coverage"] = float(np.mean(accepted[support])) if np.any(support) else 0.0
                class_accepted = support & accepted
                if np.any(class_accepted):
                    row[f"{class_name}_selective_risk"] = float(np.mean(pred[class_accepted] != y_true[class_accepted]))
                else:
                    row[f"{class_name}_selective_risk"] = 0.0
        if clean_macro_by_method and method in clean_macro_by_method:
            row["macro_f1_degradation_vs_clean"] = float(clean_macro_by_method[method] - row["macro_f1"])
        rows.append(row)
        confusion_rows.append(
            {
                "noise_environment": noise_environment,
                "snr_db": "clean" if snr_db is None else float(snr_db),
                "target_active_snr_db": "clean" if snr_db is None else float(snr_db),
                "condition": condition_label(noise_environment, snr_db),
                "method": method,
                "labels": labels,
                "confusion_matrix": cm,
            }
        )

    if "prototype_pred" in pred_frame.columns and "prototype_aux_pred" in pred_frame.columns:
        rows.append(
            {
                "noise_environment": noise_environment,
                "snr_db": "clean" if snr_db is None else float(snr_db),
                "condition": condition_label(noise_environment, snr_db),
                "method": "hierarchy_audit",
                "hierarchy_consistency": float(1.0 - pred_frame["hierarchy_inconsistent"].astype(bool).mean()),
                "mean_prototype_distance_margin": float(
                    (
                        pred_frame["nearest_main_cosine_distance"].astype(float)
                        - pred_frame["nearest_other_main_cosine_distance"].astype(float)
                    ).mul(-1.0).mean()
                )
                if "nearest_other_main_cosine_distance" in pred_frame.columns
                else None,
            }
        )
    return rows, confusion_rows


def read_reference_predictions(reference_pred_csv: Path, manifest_paths: list[str]) -> pd.DataFrame:
    ref = pd.read_csv(reference_pred_csv)
    if "path" not in ref.columns:
        if len(ref) != len(manifest_paths):
            raise RuntimeError("reference predictions lack path column and row count differs from manifest.")
        ref.insert(0, "path", manifest_paths)
    return ref


def clean_equivalence_gate(
    *,
    reference_pred_csv: Path,
    manifest_paths: list[str],
    clean_pred: pd.DataFrame,
    labels: list[str],
    tolerance: float,
) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    ref = read_reference_predictions(reference_pred_csv, manifest_paths)
    report, summary = align_prediction_frames_by_path(ref, clean_pred)
    y_true_col = "y_true" if "y_true" in ref.columns else "label"
    if y_true_col not in ref.columns:
        raise RuntimeError(f"reference predictions missing y_true/label column: {reference_pred_csv}")
    ref_pred_col = "y_pred" if "y_pred" in ref.columns else "raw_softmax_pred"
    if ref_pred_col not in ref.columns:
        raise RuntimeError(f"reference predictions missing y_pred column: {reference_pred_csv}")
    reproduced = clean_pred[["path", "y_true", "raw_softmax_pred"]].copy()
    merged = ref.assign(_path=ref["path"].map(normalize_identity_path)).merge(
        reproduced.assign(_path=reproduced["path"].map(normalize_identity_path)),
        on="_path",
        suffixes=("_ref", "_rep"),
    )

    def merged_column(base: str, side: str) -> str:
        for candidate in (f"{base}_{side}", base):
            if candidate in merged.columns:
                return candidate
        raise RuntimeError(f"clean equivalence merge missing column for {base!r} ({side}); columns={list(merged.columns)}")

    ref_y_true_col = merged_column(y_true_col, "ref")
    rep_y_true_col = merged_column("y_true", "rep")
    ref_y_pred_col = merged_column(ref_pred_col, "ref")
    rep_y_pred_col = merged_column("raw_softmax_pred", "rep")

    y_true_matches = (
        merged[ref_y_true_col].astype(str).str.lower()
        == merged[rep_y_true_col].astype(str).str.lower()
    )
    y_pred_matches = (
        merged[ref_y_pred_col].astype(str).str.lower()
        == merged[rep_y_pred_col].astype(str).str.lower()
    )
    y_true_match = bool(y_true_matches.all())
    y_pred_match = bool(y_pred_matches.all())
    summary = dict(summary)
    summary["y_true_match_count"] = int(y_true_matches.sum())
    summary["y_pred_match_count"] = int(y_pred_matches.sum())
    summary["y_true_all_match"] = y_true_match
    summary["y_pred_all_match"] = y_pred_match
    ref_macro = float(f1_score(ref[y_true_col], ref[ref_pred_col], labels=labels, average="macro", zero_division=0))
    rep_macro = float(f1_score(clean_pred["y_true"], clean_pred["raw_softmax_pred"], labels=labels, average="macro", zero_division=0))
    macro_ok = abs(ref_macro - rep_macro) <= tolerance
    ok = bool(summary["alignment_ok"] and y_true_match and y_pred_match and macro_ok)
    return {
        "ok": ok,
        "alignment": summary,
        "y_true_match": y_true_match,
        "raw_softmax_y_pred_match": y_pred_match,
        "reference_macro_f1": ref_macro,
        "reproduced_macro_f1": rep_macro,
        "macro_f1_abs_diff": abs(ref_macro - rep_macro),
        "macro_f1_tolerance": tolerance,
        "sample_count": int(len(clean_pred)),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate frozen exact prototype predictions under simulated DEMAND noise.")
    ap.add_argument("--prototype_bundle", required=True)
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--reference_pred_csv", required=True)
    ap.add_argument("--noise_manifest", required=True)
    ap.add_argument("--noise_provenance_json", default="", help="Defaults to PROVENANCE.json beside --noise_manifest.")
    ap.add_argument("--clean_prediction_metadata", default="", help="Defaults to evaluation/prediction_metadata.json beside the clean run.")
    ap.add_argument("--calibration_val_predictions", default="", help="Defaults to calibration/val_predictions.csv beside calibration.json.")
    ap.add_argument("--noise_environments", nargs="+", required=True)
    ap.add_argument("--snr_db", nargs="+", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--global_noise_seed", type=int, default=3407)
    ap.add_argument("--noise_repeat", type=int, default=0)
    ap.add_argument("--selected_channel", type=int, default=None, help="Optional guard; when set it must equal manifest selected_channel.")
    ap.add_argument("--target_coverage", type=float, default=0.95)
    ap.add_argument("--allow_overwrite", action="store_true")
    ap.add_argument("--clean_gate_tolerance", type=float, default=1e-6)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = prepare_out_dir(Path(args.out_dir), args.allow_overwrite)
    bundle_path = require_file(args.prototype_bundle, "prototype bundle")
    calibration_path = require_file(args.calibration_json, "calibration JSON")
    ckpt = require_file(args.ckpt, "hierarchical checkpoint")
    test_manifest = require_file(args.test_manifest, "test manifest")
    reference_pred_csv = require_file(args.reference_pred_csv, "reference predictions")
    noise_manifest = require_file(args.noise_manifest, "noise manifest")
    noise_provenance_json = require_file(
        args.noise_provenance_json or (noise_manifest.parent / "PROVENANCE.json"),
        "DEMAND provenance JSON",
    )
    clean_prediction_metadata_path = require_file(
        args.clean_prediction_metadata or (calibration_path.parent.parent / "evaluation" / "prediction_metadata.json"),
        "clean prediction metadata",
    )
    calibration_val_predictions_path = require_file(
        args.calibration_val_predictions or (calibration_path.parent / "val_predictions.csv"),
        "clean validation predictions",
    )

    bundle = load_prototype_bundle(bundle_path)
    bundle_meta = bundle["metadata"]
    calibration = read_json(calibration_path)
    clean_prediction_metadata = read_json(clean_prediction_metadata_path)
    noise_provenance = read_json(noise_provenance_json)
    if str(calibration.get("feature_backend", bundle_meta.get("feature_backend"))) != "training_exact":
        raise RuntimeError("Noise robustness paper debug requires feature_backend=training_exact.")
    actual_ckpt_sha = file_sha256(ckpt)
    actual_bundle_sha = file_sha256(bundle_path)
    actual_calibration_sha = file_sha256(calibration_path)
    actual_noise_manifest_sha = file_sha256(noise_manifest)
    actual_noise_provenance_sha = file_sha256(noise_provenance_json)
    expected_manifest_sha = str(noise_provenance.get("noise_source_manifest_sha256", "")).strip().lower()
    provenance_gates: dict[str, Any] = {
        "checkpoint_sha_verified": actual_ckpt_sha
        == str(bundle_meta.get("checkpoint_sha256", "")).strip().lower()
        == str(calibration.get("checkpoint_sha256", "")).strip().lower(),
        "prototype_bundle_sha_verified": actual_bundle_sha
        == str(calibration.get("prototype_bundle_sha256", "")).strip().lower(),
        "calibration_sha_verified": actual_calibration_sha
        == str(clean_prediction_metadata.get("calibration_json_sha256", "")).strip().lower(),
        "test_manifest_sha_verified": True,
        "noise_manifest_sha_verified": bool(expected_manifest_sha) and actual_noise_manifest_sha == expected_manifest_sha,
        "noise_provenance_sha_verified": bool(actual_noise_provenance_sha)
        and str(noise_provenance.get("license_status", "")) == "conflicting_metadata",
        "selected_noise_file_sha_verified": False,
        "noise_vs_pig_md5_disjoint": False,
        "selected_channel_verified": False,
        "checkpoint_sha256": actual_ckpt_sha,
        "prototype_bundle_sha256": actual_bundle_sha,
        "calibration_json_sha256": actual_calibration_sha,
        "noise_manifest_sha256": actual_noise_manifest_sha,
        "noise_provenance_json_sha256": actual_noise_provenance_sha,
    }
    if not provenance_gates["checkpoint_sha_verified"]:
        raise RuntimeError("Checkpoint SHA256 mismatch across checkpoint/prototype/calibration metadata.")
    if not provenance_gates["prototype_bundle_sha_verified"]:
        raise RuntimeError("Prototype bundle SHA256 mismatch against calibration metadata.")
    if not provenance_gates["calibration_sha_verified"]:
        raise RuntimeError("Calibration JSON SHA256 mismatch against clean prediction metadata.")
    if not provenance_gates["noise_manifest_sha_verified"]:
        raise RuntimeError("Noise manifest SHA256 mismatch against DEMAND provenance metadata.")
    if not provenance_gates["noise_provenance_sha_verified"]:
        raise RuntimeError("DEMAND provenance JSON failed license/provenance verification.")
    verify_manifest_matches_metadata(test_manifest, bundle_meta, "test")
    provenance_gates["test_manifest_sha_verified"] = True
    train_manifest = resolve_recorded_file(bundle_meta, absolute_key="train_manifest", relative_key="train_manifest", description="train manifest")
    val_manifest = resolve_recorded_file(calibration, absolute_key="val_manifest", relative_key="val_manifest", sha256_key="val_manifest_sha256", description="val manifest")
    audit = audit_manifest_disjointness({"train": train_manifest, "val": val_manifest, "test": test_manifest})
    assert_no_leakage(audit)
    write_json(out_dir / "leakage_audit.json", audit)

    config = config_from_summary(bundle_meta["model_config"])
    fold = int(bundle_meta["fold"])
    seed = int(bundle_meta["seed"])
    device = resolve_device(args.device)
    model = load_hier_model(ckpt, config, device=device)
    ds = build_hier_dataset(test_manifest, config, feature_backend="training_exact")
    meta = ds.df.copy().reset_index(drop=True)
    if ds.path_col != "path":
        meta["path"] = meta[ds.path_col]
    if ds.label_col != "label":
        meta["label"] = meta[ds.label_col]
    meta["aux_label"] = meta["__aux_label"]
    y_main = np.array([ds.main2id[str(x).strip().lower()] for x in meta["label"]], dtype=np.int64)
    y_aux = np.array([ds.aux2id[str(x).strip().lower()] for x in meta["aux_label"]], dtype=np.int64)
    manifest_paths = meta["path"].astype(str).tolist()
    val_predictions = pd.read_csv(calibration_val_predictions_path)
    method_thresholds = compute_method_rejection_thresholds(
        val_predictions=val_predictions,
        labels=config.main_labels,
        target_coverage=float(args.target_coverage),
        source_validation_predictions_path=calibration_val_predictions_path,
        methods=THRESHOLD_METHODS,
    )

    noise_df = pd.read_csv(noise_manifest)
    pig_md5_values: set[str] = set()
    for split_manifest in (train_manifest, val_manifest, test_manifest):
        split_df = pd.read_csv(split_manifest)
        if "md5" not in split_df.columns:
            raise RuntimeError(f"Manifest missing md5 column: {split_manifest}")
        pig_md5_values.update(split_df["md5"].astype(str).str.strip().str.lower().tolist())
    selected_noise: dict[str, pd.Series] = {}
    selected_channels: dict[str, int] = {}
    noise_md5_values: set[str] = set()
    selected_noise_file_sha_ok: list[bool] = []
    for environment in args.noise_environments:
        env = environment.strip().upper()
        matches = noise_df[noise_df["noise_environment"].astype(str).str.upper() == env]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one noise row for {env}, found {len(matches)}")
        row = matches.iloc[0]
        validate_noise_source_for_paper(row, paper_facing=True)
        selected_channels[env] = verify_selected_channel(args.selected_channel, row)
        selected_noise_path = require_file(row["noise_file"], f"{env} noise file")
        selected_noise_file_sha_ok.append(file_sha256(selected_noise_path) == str(row["noise_file_sha256"]).strip().lower())
        noise_md5_values.add(str(row["noise_file_md5"]).strip().lower())
        selected_noise[env] = row
    provenance_gates["selected_channel_verified"] = True
    provenance_gates["selected_noise_file_sha_verified"] = all(selected_noise_file_sha_ok)
    if not provenance_gates["selected_noise_file_sha_verified"]:
        raise RuntimeError("At least one selected DEMAND noise file SHA256 mismatches the noise manifest.")
    provenance_gates["noise_vs_pig_md5_disjoint"] = not bool(noise_md5_values & pig_md5_values)
    if not provenance_gates["noise_vs_pig_md5_disjoint"]:
        raise RuntimeError("Selected DEMAND noise MD5 overlaps with train/val/test pig audio MD5 set.")
    provenance_gates["ok"] = all(
        bool(provenance_gates[k])
        for k in (
            "checkpoint_sha_verified",
            "prototype_bundle_sha_verified",
            "calibration_sha_verified",
            "test_manifest_sha_verified",
            "noise_manifest_sha_verified",
            "noise_provenance_sha_verified",
            "selected_noise_file_sha_verified",
            "noise_vs_pig_md5_disjoint",
            "selected_channel_verified",
        )
    )
    write_json(out_dir / "provenance_gates.json", provenance_gates)

    all_predictions: list[pd.DataFrame] = []
    provenance_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    clean_macro: dict[str, float] | None = None
    clean_gate: dict[str, Any] | None = None

    conditions: list[tuple[str, float | None]] = [("clean", None)]
    for env in args.noise_environments:
        for snr in args.snr_db:
            conditions.append((env.strip().upper(), float(snr)))

    for environment, snr in conditions:
        features: list[np.ndarray] = []
        condition_provenance: list[dict[str, Any]] = []
        noise_wave: np.ndarray | None = None
        noise_row: pd.Series | None = None
        noise_sha = ""
        if environment != "clean":
            noise_row = selected_noise[environment]
            noise_path = require_file(noise_row["noise_file"], f"{environment} noise file")
            noise_sha = str(noise_row["noise_file_sha256"])
            if file_sha256(noise_path) != noise_sha:
                raise RuntimeError(f"Noise SHA256 mismatch for {noise_path}")
            noise_wave, noise_original_sr, noise_target_sr = load_noise_channel(
                noise_path,
                selected_channel=selected_channels[environment],
                target_sr=config.sr,
            )
        for _, row in meta.iterrows():
            clean_path = resolve_audio_path(row[ds.path_col])
            clean, sr, valid_start, valid_samples = load_audio_with_valid_region(
                clean_path,
                sr=config.sr,
                dur_s=config.dur_s,
            )
            clean_md5 = str(row.get("md5", "")).strip().lower()
            provenance: dict[str, Any] = {
                "fold": fold,
                "seed": seed,
                "clean_path": str(row[ds.path_col]),
                "clean_md5": clean_md5,
                "y_true": str(row["label"]),
                "aux_true": str(row["aux_label"]),
                "original_valid_samples": int(valid_samples),
                "valid_duration_ratio": float(valid_samples / clean.size),
                "noise_dataset": None if environment == "clean" else str(noise_row["dataset"]),
                "noise_environment": environment,
                "noise_file": None if environment == "clean" else str(noise_row["noise_file_project_relative"]),
                "noise_sha256": None if environment == "clean" else noise_sha,
                "environment_recording_id": None if environment == "clean" else str(noise_row["environment_recording_id"]),
                "selected_channel": None if environment == "clean" else selected_channels[environment],
                "snr_reference": "clean" if snr is None else "active_valid_region",
                "target_active_snr_db": "clean" if snr is None else float(snr),
                "simulated_noise": environment != "clean",
                "noise_protocol": "zero_shot_frozen",
                "real_farm_external_validation": False,
            }
            if environment == "clean":
                mixed = clean
                provenance.update(
                    {
                        "noise_offset": None,
                        "noise_seed": None,
                        "noise_draw_id": None,
                        "noise_repeat": None,
                        "offset_key_version": None,
                        "clean_active_rms": None,
                        "noise_active_rms_before_gain": None,
                        "gain": None,
                        "achieved_active_snr_db": None,
                        "achieved_full_window_snr_db": None,
                        "peak_before_scale": float(np.max(np.abs(clean))),
                        "final_global_scale": 1.0,
                        "clipping_detected": False,
                    }
                )
            else:
                draw_key = noise_offset_key(
                    fold=fold,
                    model_seed=seed,
                    target_active_snr_db=float(snr),
                    clean_path=normalize_identity_path(str(row[ds.path_col])),
                    clean_md5=clean_md5,
                    environment_recording_id=str(noise_row["environment_recording_id"]),
                    noise_sha256=noise_sha,
                    global_noise_seed=int(args.global_noise_seed),
                    noise_repeat=int(args.noise_repeat),
                )
                segment, offset_record = select_noise_segment(
                    noise_wave,
                    clean.size,
                    draw_key["key_parts"],
                    return_record=True,
                )
                mixed, mix_record = mix_clean_with_noise_at_snr(
                    clean,
                    segment,
                    valid_start=valid_start,
                    valid_samples=valid_samples,
                    target_snr_db=float(snr),
                )
                provenance.update(
                    {
                        "noise_draw_id": draw_key["noise_draw_id"],
                        "noise_repeat": draw_key["noise_repeat"],
                        "offset_key_version": draw_key["offset_key_version"],
                    }
                )
                provenance.update(offset_record)
                provenance.update(mix_record)
            features.append(feature_from_waveform(mixed, config))
            condition_provenance.append(provenance)
        inferred = infer_from_features(model, features, device=device, batch_size=args.batch_size)
        probs = method_probabilities(inferred, bundle, calibration)
        pred = prediction_frame(
            meta,
            y_main,
            y_aux,
            config.main_labels,
            config.aux_labels,
            {k: probs[k] for k in ["raw_softmax", "calibrated_softmax", "prototype", "hierarchical", "fused"]},
            aux_probs=inferred["aux_softmax_probs"],
            aux_proto_probs=probs["aux_prototype"],
            prototype_scores={
                "main_cosine_similarity": probs["main_cosine_similarity"],
                **prototype_scores_from_bundle(
                    inferred["embeddings"],
                    bundle,
                    temperature=calibration_float_param(
                        calibration,
                        "hierarchical",
                        "prototype_temperature",
                        fallback_keys=("prototype_temperature", "temperature"),
                        default=1.0,
                    ),
                    hier_aux_weight=calibration_float_param(
                        calibration,
                        "hierarchical",
                        "hier_aux_prob_weight",
                        fallback_keys=("hier_aux_prob_weight",),
                        default=0.5,
                    ),
                ),
            },
        )
        pred["noise_environment"] = environment
        pred["snr_db"] = "clean" if snr is None else float(snr)
        pred["condition"] = condition_label(environment, snr)
        pred["simulated_noise"] = environment != "clean"
        pred["noise_protocol"] = "zero_shot_frozen"
        pred["real_farm_external_validation"] = False
        pred["same_waveform_embedding_reused"] = True
        all_predictions.append(pred)
        provenance_rows.extend(condition_provenance)

        if environment == "clean":
            clean_gate = clean_equivalence_gate(
                reference_pred_csv=reference_pred_csv,
                manifest_paths=manifest_paths,
                clean_pred=pred,
                labels=config.main_labels,
                tolerance=args.clean_gate_tolerance,
            )
            write_json(out_dir / "clean_equivalence.json", clean_gate)
            if not clean_gate["ok"]:
                pred.to_csv(out_dir / "noise_predictions.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(provenance_rows).to_csv(out_dir / "noise_sample_provenance.csv", index=False, encoding="utf-8-sig")
                raise RuntimeError("Clean equivalence gate failed; stopping before noisy evaluation.")
            clean_macro = {
                method: multiclass_metrics(y_main, probs[method], labels=config.main_labels)["macro_f1"]
                for method in METHODS
            }
        new_metric_rows, new_confusion_rows = compute_condition_metrics(
            y_true=y_main,
            labels=config.main_labels,
            probs_by_method=probs,
            pred_frame=pred,
            calibration=calibration,
            method_thresholds=method_thresholds,
            noise_environment=environment,
            snr_db=snr,
            clean_macro_by_method=clean_macro,
        )
        metric_rows.extend(new_metric_rows)
        confusion_rows.extend(new_confusion_rows)

    predictions = pd.concat(all_predictions, ignore_index=True)
    provenance = pd.DataFrame(provenance_rows)
    metrics_table = pd.DataFrame(metric_rows)
    predictions.to_csv(out_dir / "noise_predictions.csv", index=False, encoding="utf-8-sig")
    provenance.to_csv(out_dir / "noise_sample_provenance.csv", index=False, encoding="utf-8-sig")
    metrics_table.to_csv(out_dir / "metrics_by_condition.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "confusion_by_condition_method.json", {"rows": confusion_rows})
    snr_stats_by_class = compute_snr_stats_by_class(provenance)
    pd.DataFrame(snr_stats_by_class).to_csv(out_dir / "snr_by_class.csv", index=False, encoding="utf-8-sig")
    snr_errors = []
    noisy_prov = provenance[provenance["simulated_noise"].astype(bool)]
    for _, r in noisy_prov.iterrows():
        snr_errors.append(float(r["achieved_active_snr_db"]) - float(r["target_active_snr_db"]))
    same_offset_groups = []
    if not noisy_prov.empty:
        for keys, group in noisy_prov.groupby(["clean_path", "noise_environment", "noise_repeat", "noise_draw_id"], dropna=False):
            offsets = sorted(set(int(x) for x in group["noise_offset"]))
            snrs = sorted(float(x) for x in group["target_active_snr_db"])
            same_offset_groups.append(
                {
                    "clean_path": keys[0],
                    "noise_environment": keys[1],
                    "noise_repeat": int(keys[2]),
                    "noise_draw_id": keys[3],
                    "unique_noise_offsets": offsets,
                    "target_active_snr_db": snrs,
                    "same_offset_across_snr": len(offsets) == 1,
                }
            )
    same_offset_across_snr_verified = all(x["same_offset_across_snr"] for x in same_offset_groups)
    qualification = noise_result_qualification(
        feature_backend="training_exact",
        clean_equivalence_ok=bool(clean_gate and clean_gate.get("ok", False)),
        leakage_audit_ok=bool(audit.get("ok", False)),
        manifest_sha_verified=bool(provenance_gates.get("test_manifest_sha_verified", False)),
        unsafe_allow_checkpoint_sha_mismatch=False,
        provenance_gates_ok=bool(provenance_gates.get("ok", False)),
    )
    payload = {
        "artifact_type": "prototype_noise_robustness_debug",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold": fold,
        "seed": seed,
        "lambda": bundle_meta.get("model_config", {}).get("hier_aux_weight"),
        **qualification,
        "qualification_fields": qualification,
        "simulated_noise": True,
        "noise_protocol": "zero_shot_frozen",
        "real_farm_external_validation": False,
        "noise_environments": [x.strip().upper() for x in args.noise_environments],
        "snr_db": [float(x) for x in args.snr_db],
        "target_active_snr_db": [float(x) for x in args.snr_db],
        "snr_reference": "active_valid_region",
        "noise_repeat": int(args.noise_repeat),
        "offset_key_version": "demand_noise_offset_v2",
        "selected_channels": selected_channels,
        "same_waveform_embedding_reused": True,
        "same_offset_across_snr_verified": same_offset_across_snr_verified,
        "test_parameter_selection": False,
        "clean_equivalence": clean_gate,
        "provenance_gates": provenance_gates,
        "method_specific_clean_val_thresholds": method_thresholds,
        "snr_error_abs_max": float(max(abs(x) for x in snr_errors)) if snr_errors else 0.0,
        "snr_error_mean": float(np.mean(snr_errors)) if snr_errors else 0.0,
        "leakage_audit_ok": bool(audit.get("ok", False)),
        "outputs": {
            "noise_predictions_csv": str((out_dir / "noise_predictions.csv").resolve()),
            "noise_sample_provenance_csv": str((out_dir / "noise_sample_provenance.csv").resolve()),
            "metrics_by_condition_csv": str((out_dir / "metrics_by_condition.csv").resolve()),
            "clean_equivalence_json": str((out_dir / "clean_equivalence.json").resolve()),
            "confusion_by_condition_method_json": str((out_dir / "confusion_by_condition_method.json").resolve()),
            "snr_by_class_csv": str((out_dir / "snr_by_class.csv").resolve()),
            "provenance_gates_json": str((out_dir / "provenance_gates.json").resolve()),
        },
        "metrics_by_condition": metric_rows,
        "confusion_by_condition_method": confusion_rows,
        "snr_statistics_by_class": snr_stats_by_class,
        "same_offset_across_snr_groups": same_offset_groups,
    }
    write_json(out_dir / "noise_metrics.json", payload)
    print(f"[OK] wrote noise robustness debug metrics -> {out_dir / 'noise_metrics.json'}")


if __name__ == "__main__":
    main()
