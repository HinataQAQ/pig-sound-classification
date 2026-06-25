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
    file_sha256,
    load_audio_with_valid_region,
    load_noise_channel,
    mix_clean_with_noise_at_snr,
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


def noise_result_qualification(
    *,
    feature_backend: str,
    clean_equivalence_ok: bool,
    leakage_audit_ok: bool,
    manifest_sha_verified: bool,
    unsafe_allow_checkpoint_sha_mismatch: bool,
) -> dict[str, Any]:
    feature_pipeline_equivalent = feature_backend == "training_exact"
    eligible = bool(
        feature_pipeline_equivalent
        and clean_equivalence_ok
        and leakage_audit_ok
        and manifest_sha_verified
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
    }


def compute_condition_metrics(
    *,
    y_true: np.ndarray,
    labels: list[str],
    probs_by_method: Mapping[str, np.ndarray],
    pred_frame: pd.DataFrame,
    calibration: Mapping[str, Any],
    noise_environment: str,
    snr_db: float | None,
    clean_macro_by_method: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        probs = probs_by_method[method]
        cls = multiclass_metrics(y_true, probs, labels=labels)
        cal = calibration_metrics(y_true, probs)
        pred = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        thresholds = [float(calibration.get("global_rejection_threshold", 0.0))]
        curve = coverage_risk_curve(y_true, pred, confidence, thresholds=thresholds)
        row: dict[str, Any] = {
            "noise_environment": noise_environment,
            "snr_db": "clean" if snr_db is None else float(snr_db),
            "condition": condition_label(noise_environment, snr_db),
            "method": method,
            **cls,
            **cal,
            "global_coverage": curve[0]["coverage"],
            "selective_risk": curve[0]["selective_risk"],
        }
        for class_name in ("feeding", "stress_vocal"):
            if class_name in labels:
                class_id = labels.index(class_name)
                support = y_true == class_id
                accepted = confidence >= thresholds[0]
                row[f"{class_name}_coverage"] = float(np.mean(accepted[support])) if np.any(support) else 0.0
                class_accepted = support & accepted
                if np.any(class_accepted):
                    row[f"{class_name}_selective_risk"] = float(np.mean(pred[class_accepted] != y_true[class_accepted]))
                else:
                    row[f"{class_name}_selective_risk"] = 0.0
        if clean_macro_by_method and method in clean_macro_by_method:
            row["macro_f1_degradation_vs_clean"] = float(clean_macro_by_method[method] - row["macro_f1"])
        rows.append(row)

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
    return rows


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
    ap.add_argument("--noise_environments", nargs="+", required=True)
    ap.add_argument("--snr_db", nargs="+", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--global_noise_seed", type=int, default=3407)
    ap.add_argument("--selected_channel", type=int, default=1)
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

    bundle = load_prototype_bundle(bundle_path)
    bundle_meta = bundle["metadata"]
    calibration = read_json(calibration_path)
    if str(calibration.get("feature_backend", bundle_meta.get("feature_backend"))) != "training_exact":
        raise RuntimeError("Noise robustness paper debug requires feature_backend=training_exact.")
    verify_manifest_matches_metadata(test_manifest, bundle_meta, "test")
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

    noise_df = pd.read_csv(noise_manifest)
    selected_noise: dict[str, pd.Series] = {}
    for environment in args.noise_environments:
        env = environment.strip().upper()
        matches = noise_df[noise_df["noise_environment"].astype(str).str.upper() == env]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one noise row for {env}, found {len(matches)}")
        row = matches.iloc[0]
        validate_noise_source_for_paper(row, paper_facing=True)
        selected_noise[env] = row

    all_predictions: list[pd.DataFrame] = []
    provenance_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
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
                selected_channel=int(noise_row["selected_channel"]),
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
                "original_valid_samples": int(valid_samples),
                "noise_dataset": None if environment == "clean" else str(noise_row["dataset"]),
                "noise_environment": environment,
                "noise_file": None if environment == "clean" else str(noise_row["noise_file_project_relative"]),
                "noise_sha256": None if environment == "clean" else noise_sha,
                "selected_channel": None if environment == "clean" else int(noise_row["selected_channel"]),
                "target_snr_db": "clean" if snr is None else float(snr),
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
                key_parts = [
                    f"fold={fold}",
                    f"seed={seed}",
                    normalize_identity_path(str(row[ds.path_col])),
                    clean_md5,
                    environment,
                    noise_sha,
                    f"snr={snr:g}",
                    f"global_noise_seed={args.global_noise_seed}",
                ]
                segment, offset_record = select_noise_segment(
                    noise_wave,
                    clean.size,
                    key_parts,
                    return_record=True,
                )
                mixed, mix_record = mix_clean_with_noise_at_snr(
                    clean,
                    segment,
                    valid_start=valid_start,
                    valid_samples=valid_samples,
                    target_snr_db=float(snr),
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
        metric_rows.extend(
            compute_condition_metrics(
                y_true=y_main,
                labels=config.main_labels,
                probs_by_method=probs,
                pred_frame=pred,
                calibration=calibration,
                noise_environment=environment,
                snr_db=snr,
                clean_macro_by_method=clean_macro,
            )
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    provenance = pd.DataFrame(provenance_rows)
    metrics_table = pd.DataFrame(metric_rows)
    predictions.to_csv(out_dir / "noise_predictions.csv", index=False, encoding="utf-8-sig")
    provenance.to_csv(out_dir / "noise_sample_provenance.csv", index=False, encoding="utf-8-sig")
    metrics_table.to_csv(out_dir / "metrics_by_condition.csv", index=False, encoding="utf-8-sig")
    snr_errors = []
    noisy_prov = provenance[provenance["simulated_noise"].astype(bool)]
    for _, r in noisy_prov.iterrows():
        snr_errors.append(float(r["achieved_active_snr_db"]) - float(r["target_snr_db"]))
    qualification = noise_result_qualification(
        feature_backend="training_exact",
        clean_equivalence_ok=bool(clean_gate and clean_gate.get("ok", False)),
        leakage_audit_ok=bool(audit.get("ok", False)),
        manifest_sha_verified=True,
        unsafe_allow_checkpoint_sha_mismatch=False,
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
        "selected_channel": int(args.selected_channel),
        "same_waveform_embedding_reused": True,
        "test_parameter_selection": False,
        "clean_equivalence": clean_gate,
        "snr_error_abs_max": float(max(abs(x) for x in snr_errors)) if snr_errors else 0.0,
        "snr_error_mean": float(np.mean(snr_errors)) if snr_errors else 0.0,
        "leakage_audit_ok": bool(audit.get("ok", False)),
        "outputs": {
            "noise_predictions_csv": str((out_dir / "noise_predictions.csv").resolve()),
            "noise_sample_provenance_csv": str((out_dir / "noise_sample_provenance.csv").resolve()),
            "metrics_by_condition_csv": str((out_dir / "metrics_by_condition.csv").resolve()),
            "clean_equivalence_json": str((out_dir / "clean_equivalence.json").resolve()),
        },
        "metrics_by_condition": metric_rows,
    }
    write_json(out_dir / "noise_metrics.json", payload)
    print(f"[OK] wrote noise robustness debug metrics -> {out_dir / 'noise_metrics.json'}")


if __name__ == "__main__":
    main()
