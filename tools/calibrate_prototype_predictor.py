from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prototype_model_adapter import (
    apply_hierarchy_confidence_penalty,
    assert_no_leakage,
    audit_manifest_disjointness,
    build_hier_dataset,
    calibration_metrics,
    config_from_summary,
    coverage_risk_curve,
    extract_embeddings,
    file_sha256,
    fusion_formula_metadata,
    fuse_probabilities,
    load_hier_model,
    load_prototype_bundle,
    method_metrics_table,
    multiclass_metrics,
    parse_csv_floats,
    prediction_frame,
    prototype_scores_from_bundle,
    read_json,
    require_file,
    select_per_class_thresholds,
    select_threshold_for_target_coverage,
    temperature_scale_probabilities,
    validate_fold_seed_sources,
    write_json,
)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def same_resolved_path(left: str | Path, right: str | Path) -> bool:
    return str(Path(left).resolve()).lower() == str(Path(right).resolve()).lower()


def prepare_stage_dir(root: Path, stage: str, allow_overwrite: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stage_dir = root / stage
    if stage_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing {stage} directory: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def finite_or_inf(value: Any) -> float:
    if value is None:
        return float("inf")
    out = float(value)
    return out if np.isfinite(out) else float("inf")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Calibrate prototype temperatures, fusion weight, hierarchy penalty, and reject thresholds on validation only."
    )
    ap.add_argument("--prototype_bundle", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True, help="Phase root directory; calibration/ is created inside it.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--feature_backend", choices=["auto", "training_exact", "librosa", "numpy_logmel"], default="auto")
    ap.add_argument("--prototype_temperature_grid", default="0.03,0.05,0.07,0.1,0.2,0.5,1.0")
    ap.add_argument("--temperature_grid", default="", help="Deprecated alias for --prototype_temperature_grid.")
    ap.add_argument("--softmax_temperature_grid", default="0.5,0.75,1.0,1.5,2.0")
    ap.add_argument("--fusion_weight_grid", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--hierarchy_penalty_grid", default="0.5,0.7,0.85,1.0")
    ap.add_argument("--hier_aux_prob_weight", type=float, default=0.5)
    ap.add_argument("--selection_method", choices=["softmax", "prototype", "hierarchical", "fused"], default="hierarchical")
    ap.add_argument("--target_coverage", type=float, default=0.95)
    ap.add_argument("--per_class_min_count", type=int, default=5)
    ap.add_argument("--ece_bins", type=int, default=10)
    ap.add_argument("--allow_ckpt_mismatch", action="store_true")
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    bundle_path = require_file(args.prototype_bundle, "prototype bundle")
    val_manifest = require_file(args.val_manifest, "validation manifest")
    ckpt = require_file(args.ckpt, "hierarchical checkpoint")
    out_root = Path(args.out_dir)
    out_dir = prepare_stage_dir(out_root, "calibration", args.allow_overwrite)

    bundle = load_prototype_bundle(bundle_path)
    metadata = bundle["metadata"]
    train_manifest = require_file(metadata["train_manifest"], "train manifest from prototype metadata")
    audit = audit_manifest_disjointness({"train": train_manifest, "val": val_manifest})
    assert_no_leakage(audit)
    write_json(out_dir / "leakage_audit.json", audit)

    expected_ckpt = metadata.get("checkpoint_path")
    if expected_ckpt and not same_resolved_path(expected_ckpt, ckpt) and not args.allow_ckpt_mismatch:
        raise RuntimeError(
            "Checkpoint mismatch. Prototype was built with "
            f"{expected_ckpt}, but --ckpt={ckpt}. Use --allow_ckpt_mismatch only for explicit audits."
        )
    expected_sha = metadata.get("checkpoint_sha256")
    if expected_sha and file_sha256(ckpt) != str(expected_sha) and not args.allow_ckpt_mismatch:
        raise RuntimeError("Checkpoint SHA256 mismatch between prototype metadata and --ckpt.")
    if metadata.get("val_manifest") and not same_resolved_path(metadata["val_manifest"], val_manifest):
        raise RuntimeError(
            f"Validation manifest mismatch. Prototype metadata expects {metadata['val_manifest']}, got {val_manifest}."
        )

    fold = int(metadata["fold"])
    seed = int(metadata["seed"])
    validate_fold_seed_sources(
        fold=fold,
        seed=seed,
        sources={
            "prototype_bundle": bundle_path,
            "val_manifest": val_manifest,
            "checkpoint": ckpt,
            "out_dir": out_root,
        },
    )

    config = config_from_summary(metadata["model_config"])
    feature_backend = str(metadata.get("feature_backend", "librosa")) if args.feature_backend == "auto" else args.feature_backend
    device = resolve_device(args.device)
    model = load_hier_model(ckpt, config, device=device)
    ds_val = build_hier_dataset(val_manifest, config, feature_backend=feature_backend)
    extracted = extract_embeddings(
        model,
        ds_val,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    proto_grid_text = args.temperature_grid if args.temperature_grid else args.prototype_temperature_grid
    prototype_temperatures = parse_csv_floats(proto_grid_text, name="prototype_temperature_grid")
    softmax_temperatures = parse_csv_floats(args.softmax_temperature_grid, name="softmax_temperature_grid")
    fusion_weights = parse_csv_floats(args.fusion_weight_grid, name="fusion_weight_grid")
    hierarchy_penalties = parse_csv_floats(args.hierarchy_penalty_grid, name="hierarchy_penalty_grid")

    grid_rows: list[dict[str, float | str | int]] = []
    cached: dict[tuple[float, float, float], dict[str, np.ndarray]] = {}

    for proto_temp in prototype_temperatures:
        proto_scores = prototype_scores_from_bundle(
            extracted["embeddings"],
            bundle,
            temperature=proto_temp,
            hier_aux_weight=args.hier_aux_prob_weight,
        )
        for softmax_temp in softmax_temperatures:
            softmax_probs = temperature_scale_probabilities(extracted["softmax_probs"], softmax_temp)
            for softmax_weight in fusion_weights:
                probs_by_method = {
                    "softmax": softmax_probs,
                    "prototype": proto_scores["prototype"],
                    "hierarchical": proto_scores["hierarchical"],
                    "fused": fuse_probabilities(
                        softmax_probs,
                        proto_scores["hierarchical"],
                        softmax_weight=softmax_weight,
                    ),
                }
                key = (float(proto_temp), float(softmax_temp), float(softmax_weight))
                cached[key] = {
                    **probs_by_method,
                    "aux_prototype": proto_scores["aux_prototype"],
                    "main_cosine_similarity": proto_scores["main_cosine_similarity"],
                    "aux_cosine_similarity": proto_scores["aux_cosine_similarity"],
                    "main_cosine_distance": proto_scores["main_cosine_distance"],
                    "aux_cosine_distance": proto_scores["aux_cosine_distance"],
                    "main_euclidean_distance": proto_scores["main_euclidean_distance"],
                    "aux_euclidean_distance": proto_scores["aux_euclidean_distance"],
                    "nearest_main_id": proto_scores["nearest_main_id"],
                    "nearest_aux_id": proto_scores["nearest_aux_id"],
                }
                for method, probs in probs_by_method.items():
                    row: dict[str, float | str | int] = {
                        "prototype_temperature": float(proto_temp),
                        "temperature": float(proto_temp),
                        "softmax_temperature": float(softmax_temp),
                        "softmax_weight": float(softmax_weight),
                        "method": method,
                    }
                    row.update(multiclass_metrics(extracted["y_main"], probs, config.main_labels))
                    row.update(calibration_metrics(extracted["y_main"], probs, n_bins=args.ece_bins))
                    grid_rows.append(row)

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_dir / "calibration_grid.csv", index=False, encoding="utf-8-sig")

    choices = grid[grid["method"] == args.selection_method].copy()
    if choices.empty:
        raise RuntimeError(f"No calibration rows for selection method={args.selection_method}")
    choices = choices.sort_values(["macro_f1", "nll", "ece"], ascending=[False, True, True])
    best = choices.iloc[0].to_dict()
    best_key = (
        float(best["prototype_temperature"]),
        float(best["softmax_temperature"]),
        float(best["softmax_weight"]),
    )
    best_probs = cached[best_key]
    selected_probs = best_probs[args.selection_method]
    selected_pred = selected_probs.argmax(axis=1)
    selected_conf_raw = selected_probs.max(axis=1)

    main_proto_pred = best_probs["prototype"].argmax(axis=1)
    aux_proto_pred = best_probs["aux_prototype"].argmax(axis=1)
    main_proto_labels = [config.main_labels[int(i)] for i in main_proto_pred]
    aux_proto_labels = [config.aux_labels[int(i)] for i in aux_proto_pred]

    penalty_rows: list[dict[str, float | int | None]] = []
    for penalty in hierarchy_penalties:
        adjusted_conf, inconsistent = apply_hierarchy_confidence_penalty(
            selected_conf_raw,
            main_proto_labels,
            aux_proto_labels,
            penalty=float(penalty),
        )
        threshold = select_threshold_for_target_coverage(adjusted_conf, args.target_coverage)
        curve = coverage_risk_curve(extracted["y_main"], selected_pred, adjusted_conf, thresholds=[threshold])[0]
        penalty_rows.append(
            {
                "hierarchy_confidence_penalty": float(penalty),
                "global_rejection_threshold": float(threshold),
                "coverage": curve["coverage"],
                "selective_risk": curve["selective_risk"],
                "selective_acc": curve["selective_acc"],
                "n_accepted": curve["n_accepted"],
                "n_total": curve["n_total"],
                "n_hierarchy_inconsistent": int(inconsistent.sum()),
            }
        )
    penalty_grid = pd.DataFrame(penalty_rows)
    penalty_grid.to_csv(out_dir / "hierarchy_penalty_grid.csv", index=False, encoding="utf-8-sig")
    penalty_grid["_risk_sort"] = penalty_grid["selective_risk"].map(finite_or_inf)
    penalty_grid = penalty_grid.sort_values(
        ["_risk_sort", "hierarchy_confidence_penalty"],
        ascending=[True, False],
    )
    best_penalty_row = penalty_grid.iloc[0].drop(labels=["_risk_sort"]).to_dict()
    best_penalty = float(best_penalty_row["hierarchy_confidence_penalty"])
    selected_conf, hierarchy_inconsistent = apply_hierarchy_confidence_penalty(
        selected_conf_raw,
        main_proto_labels,
        aux_proto_labels,
        penalty=best_penalty,
    )

    global_threshold = select_threshold_for_target_coverage(selected_conf, args.target_coverage)
    per_class_thresholds = select_per_class_thresholds(
        selected_pred,
        selected_conf,
        config.main_labels,
        target_coverage=args.target_coverage,
        min_count=args.per_class_min_count,
        fallback_threshold=global_threshold,
    )

    thresholds = sorted(set(np.linspace(0.0, 1.0, 101).tolist() + [global_threshold]))
    coverage_rows = coverage_risk_curve(
        extracted["y_main"],
        selected_pred,
        selected_conf,
        thresholds=thresholds,
    )
    pd.DataFrame(coverage_rows).to_csv(out_dir / "coverage_risk.csv", index=False, encoding="utf-8-sig")

    probs_by_method = {
        "softmax": best_probs["softmax"],
        "prototype": best_probs["prototype"],
        "hierarchical": best_probs["hierarchical"],
        "fused": best_probs["fused"],
    }
    val_pred = prediction_frame(
        extracted["metadata"],
        extracted["y_main"],
        extracted["y_aux"],
        config.main_labels,
        config.aux_labels,
        probs_by_method,
        aux_probs=extracted["aux_softmax_probs"],
        aux_proto_probs=best_probs["aux_prototype"],
        prototype_scores=best_probs,
    )
    val_pred["selected_method"] = args.selection_method
    val_pred["selected_pred"] = val_pred[f"{args.selection_method}_pred"]
    val_pred["selected_confidence_raw"] = selected_conf_raw
    val_pred["selected_confidence"] = selected_conf
    val_pred["uncertainty"] = 1.0 - selected_conf
    val_pred["hierarchy_confidence_penalty"] = best_penalty
    val_pred["hierarchy_inconsistent"] = hierarchy_inconsistent
    val_pred["accept_global"] = val_pred["selected_confidence"] >= global_threshold
    val_pred["pred_or_uncertain_global"] = np.where(
        val_pred["accept_global"], val_pred["selected_pred"], "uncertain"
    )
    val_pred["accept_per_class"] = [
        conf >= per_class_thresholds[pred]
        for pred, conf in zip(val_pred["selected_pred"], val_pred["selected_confidence"])
    ]
    val_pred["pred_or_uncertain_per_class"] = np.where(
        val_pred["accept_per_class"], val_pred["selected_pred"], "uncertain"
    )
    val_pred["known_state_global"] = np.where(val_pred["accept_global"], "known", "uncertain")
    val_pred["known_state_per_class"] = np.where(val_pred["accept_per_class"], "known", "uncertain")
    val_pred.to_csv(out_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")

    best_metrics = method_metrics_table(
        extracted["y_main"],
        probs_by_method,
        config.main_labels,
        n_bins=args.ece_bins,
    )
    best_metrics.to_csv(out_dir / "metrics_by_method_val.csv", index=False, encoding="utf-8-sig")

    calibration = {
        "artifact_type": "hier_acoustic_prototype_calibration",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prototype_bundle": str(bundle_path.resolve()),
        "prototype_bundle_sha256": file_sha256(bundle_path),
        "checkpoint_path": str(ckpt.resolve()),
        "checkpoint_sha256": file_sha256(ckpt),
        "train_manifest": str(train_manifest.resolve()),
        "train_manifest_sha256": file_sha256(train_manifest),
        "val_manifest": str(val_manifest.resolve()),
        "val_manifest_sha256": file_sha256(val_manifest),
        "fold": fold,
        "seed": seed,
        "model_config": metadata["model_config"],
        "feature_backend": feature_backend,
        "selection_method": args.selection_method,
        "prototype_temperature": float(best["prototype_temperature"]),
        "temperature": float(best["prototype_temperature"]),
        "softmax_temperature": float(best["softmax_temperature"]),
        "softmax_weight": float(best["softmax_weight"]),
        "hier_aux_prob_weight": float(args.hier_aux_prob_weight),
        **fusion_formula_metadata(),
        "hierarchy_confidence_penalty": best_penalty,
        "target_coverage": float(args.target_coverage),
        "global_rejection_threshold": float(global_threshold),
        "per_class_rejection_thresholds": per_class_thresholds,
        "reject_label": "uncertain",
        "unknown_detection_claim": False,
        "test_used_for_calibration": False,
        "best_validation_row": {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in best.items()},
        "best_hierarchy_penalty_row": {
            k: (float(v) if isinstance(v, (int, float, np.number)) else v)
            for k, v in best_penalty_row.items()
        },
        "leakage_audit_path": str((out_dir / "leakage_audit.json").resolve()),
        "calibration_grid_path": str((out_dir / "calibration_grid.csv").resolve()),
        "hierarchy_penalty_grid_path": str((out_dir / "hierarchy_penalty_grid.csv").resolve()),
        "coverage_risk_path": str((out_dir / "coverage_risk.csv").resolve()),
    }
    write_json(out_dir / "calibration.json", calibration)

    print(f"[OK] wrote calibration -> {out_dir / 'calibration.json'}")
    print(f"[OK] wrote validation predictions -> {out_dir / 'val_predictions.csv'}")
    print(f"[OK] wrote validation coverage-risk -> {out_dir / 'coverage_risk.csv'}")


if __name__ == "__main__":
    main()
