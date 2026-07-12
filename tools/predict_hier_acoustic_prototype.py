from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nomenclature import (
    build_method_metadata,
    build_model_metadata,
    reconcile_selection_protocol,
    resolve_inference_method_fields,
    validate_recorded_artifact_identity,
)
from prototype_model_adapter import (
    apply_hierarchy_confidence_penalty,
    assert_no_leakage,
    audit_manifest_disjointness,
    build_hier_dataset,
    calibration_float_param,
    config_from_summary,
    extract_embeddings,
    file_sha256,
    fuse_probabilities,
    load_audio_numpy,
    load_hier_model,
    load_prototype_bundle,
    make_numpy_logmel_feature,
    path_record,
    prediction_frame,
    project_relative_path,
    prototype_scores_from_bundle,
    read_json,
    require_file,
    result_qualification_fields,
    resolve_recorded_file,
    select_prediction_input_role,
    temperature_scale_probabilities,
    validate_fold_seed_sources,
    verify_manifest_matches_metadata,
    write_json,
)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def same_resolved_path(left: str | Path, right: str | Path) -> bool:
    return str(Path(left).resolve()).lower() == str(Path(right).resolve()).lower()


def validate_prediction_input_nomenclature(
    bundle_metadata: dict[str, Any], calibration: dict[str, Any]
) -> None:
    """Validate explicit v1 provenance while accepting unversioned artifacts."""

    try:
        config = config_from_summary(bundle_metadata["model_config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Prototype bundle is missing a valid model_config."
        ) from exc
    _, route = resolve_inference_method_fields(calibration)
    expected_stage = (
        "b3_hierarchical_prototype_top1_ablation"
        if route == "hierarchical_prototype_candidate"
        else "b2_validation_selected_hierarchical_crnn"
    )
    try:
        validated_bundle = validate_recorded_artifact_identity(
            bundle_metadata,
            expected_model_family="hierarchical_supervision_crnn",
            expected_training_stage=(
                "b2_validation_selected_hierarchical_crnn"
            ),
            expected_context_seconds=float(config.dur_s),
            context="prototype bundle",
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid prototype bundle nomenclature metadata: {exc}"
        ) from exc
    try:
        validated_calibration = validate_recorded_artifact_identity(
            calibration,
            expected_model_family="hierarchical_supervision_crnn",
            expected_training_stage=expected_stage,
            expected_context_seconds=float(config.dur_s),
            context="calibration",
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid calibration nomenclature metadata: {exc}"
        ) from exc
    if (
        validated_bundle is not None
        and validated_calibration is not None
        and "selection_protocol" in validated_bundle
        and "selection_protocol" in validated_calibration
    ):
        reconcile_selection_protocol(
            str(validated_bundle["selection_protocol"]),
            requested=str(validated_calibration["selection_protocol"]),
        )
    calibration_config = calibration.get("model_config")
    if isinstance(calibration_config, dict) and "dur_s" in calibration_config:
        calibration_duration = float(calibration_config["dur_s"])
        if not math.isclose(
            calibration_duration,
            float(config.dur_s),
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise ValueError(
                "Invalid calibration nomenclature metadata: model_config.dur_s "
                f"{calibration_duration} conflicts with prototype bundle "
                f"duration {float(config.dur_s)}."
            )


def prepare_evaluation_dir(root: Path, allow_overwrite: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    planned = [
        out_dir / "test_predictions.csv",
        out_dir / "test_predictions.json",
        out_dir / "prediction_metadata.json",
        out_dir / "leakage_audit.json",
    ]
    for path in planned:
        if path.exists() and not allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    return out_dir


def add_normalized_distance_columns(pred: pd.DataFrame, distance_stats: dict[str, Any] | None) -> pd.DataFrame:
    if not distance_stats:
        return pred

    def class_stat(section: str, label: str, stat: str) -> float | None:
        try:
            val = distance_stats[section]["classes"][label]["cosine_distance"][stat]
        except KeyError:
            return None
        return float(val) if val is not None else None

    main_norm: list[float | None] = []
    main_z: list[float | None] = []
    aux_norm: list[float | None] = []
    aux_z: list[float | None] = []

    for _, row in pred.iterrows():
        main_label = str(row.get("nearest_main_prototype", ""))
        aux_label = str(row.get("nearest_aux_prototype", ""))
        main_dist = float(row.get("nearest_main_cosine_distance", np.nan))
        aux_dist = float(row.get("nearest_aux_cosine_distance", np.nan))
        main_q95 = class_stat("main", main_label, "q95")
        main_mean = class_stat("main", main_label, "mean")
        main_std = class_stat("main", main_label, "std")
        aux_q95 = class_stat("auxiliary", aux_label, "q95")
        aux_mean = class_stat("auxiliary", aux_label, "mean")
        aux_std = class_stat("auxiliary", aux_label, "std")
        main_norm.append(main_dist / main_q95 if main_q95 and main_q95 > 0 else None)
        aux_norm.append(aux_dist / aux_q95 if aux_q95 and aux_q95 > 0 else None)
        main_z.append((main_dist - main_mean) / main_std if main_mean is not None and main_std and main_std > 0 else None)
        aux_z.append((aux_dist - aux_mean) / aux_std if aux_mean is not None and aux_std and aux_std > 0 else None)

    extra = pd.DataFrame(
        {
            "nearest_main_normalized_cosine_distance_q95": main_norm,
            "nearest_main_cosine_distance_z": main_z,
            "nearest_aux_normalized_cosine_distance_q95": aux_norm,
            "nearest_aux_cosine_distance_z": aux_z,
        }
    )
    return pd.concat([pred.reset_index(drop=True), extra], axis=1).copy()


def extract_single_audio(
    audio_path: Path,
    model,
    config,
    device: str,
    feature_backend: str,
) -> dict[str, Any]:
    import torch

    if feature_backend == "training_exact":
        feature_backend = "librosa"
    if feature_backend == "numpy_logmel":
        y, sr = load_audio_numpy(audio_path, sr=config.sr, dur_s=config.dur_s)
        x = make_numpy_logmel_feature(
            y,
            sr=sr,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            fmin=config.fmin,
            fmax=config.fmax,
        )
    elif feature_backend == "librosa":
        from train_hier_longcontext_crnn import load_audio_soundfile, make_feature

        y, sr = load_audio_soundfile(audio_path, sr=config.sr, dur_s=config.dur_s)
        x = make_feature(
            y,
            feature_mode=config.feature_mode,
            sr=sr,
            n_mels=config.n_mels,
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            fmin=config.fmin,
            fmax=config.fmax,
        )
    else:
        raise ValueError(f"Unknown feature_backend={feature_backend}")

    xb = torch.from_numpy(x).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        z = model.encode(xb)
        main_logits = model.main_head(z)
        aux_logits = model.aux_head(z)
    return {
        "metadata": pd.DataFrame(
            [
                {
                    "path": str(audio_path),
                    "label": "",
                    "subtype": "",
                    "source_id": "",
                    "md5": "",
                    "aux_label": "",
                    "input_type": "audio",
                }
            ]
        ),
        "embeddings": z.detach().cpu().numpy().astype(np.float32),
        "softmax_probs": torch.softmax(main_logits, dim=1).detach().cpu().numpy().astype(np.float32),
        "aux_softmax_probs": torch.softmax(aux_logits, dim=1).detach().cpu().numpy().astype(np.float32),
        "y_main": np.zeros(1, dtype=np.int64),
        "y_aux": np.zeros(1, dtype=np.int64),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run frozen prediction with calibrated hierarchical acoustic prototypes."
    )
    ap.add_argument("--prototype_bundle", required=True)
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--test_manifest", default="", help="Frozen test manifest. Alias: --manifest.")
    ap.add_argument("--manifest", default="", help="Manifest input for prediction.")
    ap.add_argument("--audio", default="", help="Single audio file input for prediction.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True, help="Phase root directory; evaluation/ is created inside it.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--feature_backend", choices=["auto", "training_exact", "librosa", "numpy_logmel"], default="auto")
    ap.add_argument("--allow_relocated_checkpoint", action="store_true", help="Allow checkpoint path relocation only when SHA256 matches.")
    ap.add_argument(
        "--unsafe_allow_checkpoint_sha_mismatch",
        action="store_true",
        help="Unsafe audit-only bypass for checkpoint SHA mismatch. Do not use for paper runs.",
    )
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    bundle_path = require_file(args.prototype_bundle, "prototype bundle")
    calibration_path = require_file(args.calibration_json, "calibration JSON")
    ckpt = require_file(args.ckpt, "hierarchical checkpoint")
    input_spec = select_prediction_input_role(
        test_manifest=args.test_manifest,
        manifest=args.manifest,
        audio=args.audio,
    )
    input_role = input_spec["input_role"]
    input_path = input_spec["input_path"]

    bundle = load_prototype_bundle(bundle_path)
    bundle_meta = bundle["metadata"]
    calibration = read_json(calibration_path)
    validate_prediction_input_nomenclature(bundle_meta, calibration)
    out_dir = prepare_evaluation_dir(Path(args.out_dir), args.allow_overwrite)

    expected_ckpt = calibration.get("checkpoint_path")
    ckpt_sha = file_sha256(ckpt)
    if expected_ckpt and not same_resolved_path(expected_ckpt, ckpt) and not args.allow_relocated_checkpoint:
        raise RuntimeError(
            f"checkpoint_path mismatch. Calibration expects {expected_ckpt}, but got {ckpt}. "
            "Use --allow_relocated_checkpoint only if SHA256 matches."
        )
    if calibration.get("checkpoint_sha256") and ckpt_sha != str(calibration["checkpoint_sha256"]) and not args.unsafe_allow_checkpoint_sha_mismatch:
        raise RuntimeError("Checkpoint SHA256 mismatch between calibration JSON and --ckpt.")
    if calibration.get("checkpoint_sha256") and ckpt_sha != str(calibration["checkpoint_sha256"]) and args.unsafe_allow_checkpoint_sha_mismatch:
        print("[WARN] unsafe checkpoint SHA mismatch bypass enabled; do not use this run for paper results.")
    if calibration.get("prototype_bundle_sha256") and file_sha256(bundle_path) != str(calibration["prototype_bundle_sha256"]):
        raise RuntimeError("Prototype bundle SHA256 mismatch between calibration JSON and --prototype_bundle.")

    train_manifest = resolve_recorded_file(
        bundle_meta,
        absolute_key="train_manifest",
        relative_key="train_manifest",
        description="train manifest from prototype metadata",
    )
    val_manifest = resolve_recorded_file(
        calibration,
        absolute_key="val_manifest",
        relative_key="val_manifest",
        sha256_key="val_manifest_sha256",
        description="validation manifest from calibration",
    )
    verify_file = verify_manifest_matches_metadata
    test_manifest: Path | None = None
    audio_arg = ""
    if input_role in {"frozen_test", "inference_manifest"}:
        test_manifest = require_file(
            input_path,
            "frozen test manifest" if input_role == "frozen_test" else "inference manifest",
        )
    manifest_sha_verified = False
    if input_role == "frozen_test":
        verify_file(test_manifest, bundle_meta, "test")
        manifest_sha_verified = True
        audit = audit_manifest_disjointness({"train": train_manifest, "val": val_manifest, "test": test_manifest})
    else:
        audit = audit_manifest_disjointness({"train": train_manifest, "val": val_manifest})
    assert_no_leakage(audit)
    write_json(out_dir / "leakage_audit.json", audit)

    fold = int(bundle_meta["fold"])
    seed = int(bundle_meta["seed"])
    validate_fold_seed_sources(
        fold=fold,
        seed=seed,
        sources={
            "prototype_bundle": bundle_path,
            "calibration_json": calibration_path,
            "checkpoint": ckpt,
            "manifest": test_manifest,
            "out_dir": args.out_dir,
        },
    )

    config = config_from_summary(bundle_meta["model_config"])
    selection_method, canonical_route = resolve_inference_method_fields(
        calibration, warn_on_legacy=True
    )
    selection_protocol = reconcile_selection_protocol(
        bundle_meta.get("selection_protocol"),
        requested=calibration.get("selection_protocol"),
    )
    training_stage = (
        "b3_hierarchical_prototype_top1_ablation"
        if canonical_route == "hierarchical_prototype_candidate"
        else "b2_validation_selected_hierarchical_crnn"
    )
    common_metadata = build_model_metadata(
        model_family="hierarchical_supervision_crnn",
        training_stage=training_stage,
        context_seconds=float(config.dur_s),
        selection_protocol=selection_protocol,
    )
    selected_route_metadata = (
        build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage=training_stage,
            context_seconds=float(config.dur_s),
            selection_protocol=selection_protocol,
            inference_route=canonical_route,
            legacy_method_id=selection_method,
        )
        if canonical_route is not None
        else common_metadata
    )
    feature_backend = str(calibration.get("feature_backend", bundle_meta.get("feature_backend", "librosa"))) if args.feature_backend == "auto" else args.feature_backend
    qualification = result_qualification_fields(
        feature_backend,
        fold=fold,
        seed=seed,
        input_role=input_role,
        unsafe_allow_checkpoint_sha_mismatch=bool(args.unsafe_allow_checkpoint_sha_mismatch),
        manifest_sha_verified=manifest_sha_verified,
        leakage_audit_ok=bool(audit.get("ok", False)),
    )
    device = resolve_device(args.device)
    model = load_hier_model(ckpt, config, device=device)
    if test_manifest is not None:
        ds_test = build_hier_dataset(test_manifest, config, feature_backend=feature_backend)
        extracted = extract_embeddings(
            model,
            ds_test,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        n_rows = int(len(ds_test))
    else:
        audio_arg = input_path
        audio_path = require_file(audio_arg, "audio file")
        extracted = extract_single_audio(audio_path, model, config, device=device, feature_backend=feature_backend)
        n_rows = 1

    method_best_params = calibration.get("method_best_params", {})
    raw_softmax_probs = extracted["softmax_probs"]
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
    fused = fuse_probabilities(
        calibrated_softmax_probs,
        hier_scores["hierarchical"],
        softmax_weight=fusion_alpha,
    )
    probs_by_method = {
        "raw_softmax": raw_softmax_probs,
        "calibrated_softmax": calibrated_softmax_probs,
        "prototype": proto_scores["prototype"],
        "hierarchical": hier_scores["hierarchical"],
        "fused": fused,
    }

    pred = prediction_frame(
        extracted["metadata"],
        extracted["y_main"],
        extracted["y_aux"],
        config.main_labels,
        config.aux_labels,
        probs_by_method,
        aux_probs=extracted["aux_softmax_probs"],
        aux_proto_probs=hier_scores["aux_prototype"],
        prototype_scores=hier_scores,
    )
    if input_role == "single_audio":
        pred["y_true_id"] = ""
        pred["y_true"] = ""
        pred["aux_true_id"] = ""
        pred["aux_true"] = ""

    distance_stats: dict[str, Any] | None = None
    if bundle_meta.get("distance_statistics_path"):
        distance_stats = read_json(bundle_meta["distance_statistics_path"])
    pred = add_normalized_distance_columns(pred, distance_stats)

    selected_method = selection_method
    global_threshold = float(calibration["global_rejection_threshold"])
    per_class_thresholds = {str(k): float(v) for k, v in calibration["per_class_rejection_thresholds"].items()}
    selected_probs = probs_by_method[selected_method]
    selected_conf_raw = selected_probs.max(axis=1)
    main_proto_labels = pred["prototype_pred"].astype(str).tolist()
    aux_proto_labels = pred["prototype_aux_pred"].astype(str).tolist()
    selected_conf, hierarchy_inconsistent = apply_hierarchy_confidence_penalty(
        selected_conf_raw,
        main_proto_labels,
        aux_proto_labels,
        penalty=float(calibration.get("hierarchy_confidence_penalty", 1.0)),
    )
    pred["selected_method"] = selected_method
    pred["selected_pred"] = pred[f"{selected_method}_pred"]
    pred["selected_confidence_raw"] = selected_conf_raw
    pred["selected_confidence"] = selected_conf
    pred["uncertainty"] = 1.0 - selected_conf
    pred["hierarchy_confidence_penalty"] = float(calibration.get("hierarchy_confidence_penalty", 1.0))
    pred["hierarchy_inconsistent"] = hierarchy_inconsistent
    pred["accept_global"] = pred["selected_confidence"] >= global_threshold
    pred["pred_or_uncertain_global"] = np.where(pred["accept_global"], pred["selected_pred"], "uncertain")
    pred["accept_per_class"] = [
        conf >= per_class_thresholds[pred_label]
        for pred_label, conf in zip(pred["selected_pred"], pred["selected_confidence"])
    ]
    pred["pred_or_uncertain_per_class"] = np.where(
        pred["accept_per_class"], pred["selected_pred"], "uncertain"
    )
    pred["known_state_global"] = np.where(pred["accept_global"], "known", "uncertain")
    pred["known_state_per_class"] = np.where(pred["accept_per_class"], "known", "uncertain")
    pred["reject_label"] = str(calibration.get("reject_label", "uncertain"))
    pred["unknown_detection_claim"] = False
    pred["input_role"] = input_role
    for key, value in qualification.items():
        pred[key] = value
    for key, value in selected_route_metadata.items():
        pred[key] = value

    pred_out = out_dir / "test_predictions.csv"
    pred_json = out_dir / "test_predictions.json"
    pred.to_csv(pred_out, index=False, encoding="utf-8-sig")
    pred.to_json(pred_json, orient="records", force_ascii=False, indent=2)
    prediction_csv_sha256 = file_sha256(pred_out)
    prediction_json_sha256 = file_sha256(pred_json)

    metadata = {
        "artifact_type": "hier_acoustic_prototype_frozen_predictions",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **selected_route_metadata,
        "input_kind": input_role,
        "input_role": input_role,
        "prototype_bundle": str(bundle_path.resolve()),
        "prototype_bundle_sha256": file_sha256(bundle_path),
        "calibration_json": str(calibration_path.resolve()),
        "calibration_json_sha256": file_sha256(calibration_path),
        "prediction_csv_sha256": prediction_csv_sha256,
        "prediction_json_sha256": prediction_json_sha256,
        "checkpoint_path": str(ckpt.resolve()),
        "checkpoint_sha256": ckpt_sha,
        "train_manifest": str(train_manifest.resolve()),
        "val_manifest": str(val_manifest.resolve()),
        "test_manifest": str(test_manifest.resolve()) if test_manifest is not None else None,
        "audio": str(Path(audio_arg).resolve()) if audio_arg else None,
        "project_relative_paths": {
            "prototype_bundle": project_relative_path(bundle_path),
            "calibration_json": project_relative_path(calibration_path),
            "checkpoint": project_relative_path(ckpt),
            "train_manifest": project_relative_path(train_manifest),
            "val_manifest": project_relative_path(val_manifest),
            "test_manifest": project_relative_path(test_manifest) if test_manifest is not None else None,
            "audio": project_relative_path(audio_arg) if audio_arg else None,
        },
        "paths": {
            "prototype_bundle": path_record(bundle_path),
            "calibration_json": path_record(calibration_path),
            "checkpoint": path_record(ckpt),
            "train_manifest": path_record(train_manifest),
            "val_manifest": path_record(val_manifest),
            "test_manifest": path_record(test_manifest) if test_manifest is not None else None,
            "audio": path_record(audio_arg) if audio_arg else None,
        },
        "fold": fold,
        "seed": seed,
        "selection_method": selected_method,
        "feature_backend": feature_backend,
        **qualification,
        "leakage_audit_ok": bool(audit.get("ok", False)),
        "manifest_sha_verified": bool(manifest_sha_verified),
        "allow_relocated_checkpoint": bool(args.allow_relocated_checkpoint),
        "unsafe_allow_checkpoint_sha_mismatch": bool(args.unsafe_allow_checkpoint_sha_mismatch),
        "method_best_params": method_best_params,
        "prototype_temperature": prototype_temperature,
        "hierarchical_prototype_temperature": hierarchical_temperature,
        "calibrated_softmax_temperature": calibrated_softmax_temperature,
        "softmax_weight": fusion_alpha,
        "hierarchy_confidence_penalty": float(calibration.get("hierarchy_confidence_penalty", 1.0)),
        "global_rejection_threshold": global_threshold,
        "per_class_rejection_thresholds": per_class_thresholds,
        "rows_after_filtering": n_rows,
        "test_used_for_parameter_selection": False,
        "unknown_detection_claim": False,
        "prediction_csv": str(pred_out.resolve()),
        "prediction_json": str(pred_json.resolve()),
        "leakage_audit_path": str((out_dir / "leakage_audit.json").resolve()),
    }
    write_json(out_dir / "prediction_metadata.json", metadata)

    print(
        "[INFO] selected inference method -> "
        f"{selected_route_metadata.get('display_name_en', selected_method)}"
    )
    print(f"[OK] wrote frozen predictions -> {pred_out}")
    print(f"[OK] wrote prediction metadata -> {out_dir / 'prediction_metadata.json'}")


if __name__ == "__main__":
    main()
