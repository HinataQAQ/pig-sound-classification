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
    calibration_relevant_parameters,
    calibration_metrics,
    config_from_summary,
    coverage_risk_curve,
    extract_embeddings,
    file_sha256,
    fusion_formula_metadata,
    fusion_mode_from_alpha,
    fuse_probabilities,
    load_hier_model,
    load_prototype_bundle,
    method_metrics_table,
    multiclass_metrics,
    parse_csv_floats,
    path_record,
    prediction_frame,
    project_relative_path,
    prototype_scores_from_bundle,
    read_json,
    require_file,
    result_qualification_fields,
    resolve_recorded_file,
    select_per_class_thresholds,
    select_threshold_for_target_coverage,
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


def json_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, np.number)):
        return float(value) if np.isfinite(float(value)) else None
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


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
    ap.add_argument(
        "--selection_method",
        choices=["raw_softmax", "calibrated_softmax", "softmax", "prototype", "hierarchical", "fused"],
        default="hierarchical",
        help="Final method selected from independently calibrated method configurations. 'softmax' aliases calibrated_softmax.",
    )
    ap.add_argument("--target_coverage", type=float, default=0.95)
    ap.add_argument("--per_class_min_count", type=int, default=5)
    ap.add_argument("--ece_bins", type=int, default=10)
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
    selection_method = "calibrated_softmax" if args.selection_method == "softmax" else args.selection_method

    bundle_path = require_file(args.prototype_bundle, "prototype bundle")
    val_manifest = require_file(args.val_manifest, "validation manifest")
    ckpt = require_file(args.ckpt, "hierarchical checkpoint")
    out_root = Path(args.out_dir)
    out_dir = prepare_stage_dir(out_root, "calibration", args.allow_overwrite)

    bundle = load_prototype_bundle(bundle_path)
    metadata = bundle["metadata"]
    train_manifest = resolve_recorded_file(
        metadata,
        absolute_key="train_manifest",
        relative_key="train_manifest",
        sha256_key=None,
        description="train manifest from prototype metadata",
    )
    verify_manifest_matches_metadata(val_manifest, metadata, "val")
    audit = audit_manifest_disjointness({"train": train_manifest, "val": val_manifest})
    assert_no_leakage(audit)
    write_json(out_dir / "leakage_audit.json", audit)

    expected_ckpt = metadata.get("checkpoint_path")
    ckpt_sha = file_sha256(ckpt)
    expected_sha = metadata.get("checkpoint_sha256")
    if expected_ckpt and not same_resolved_path(expected_ckpt, ckpt) and not args.allow_relocated_checkpoint:
        raise RuntimeError(
            "Checkpoint mismatch. Prototype was built with "
            f"{expected_ckpt}, but --ckpt={ckpt}. Use --allow_relocated_checkpoint only if SHA256 matches."
        )
    if expected_sha and ckpt_sha != str(expected_sha) and not args.unsafe_allow_checkpoint_sha_mismatch:
        raise RuntimeError("Checkpoint SHA256 mismatch between prototype metadata and --ckpt.")
    if expected_sha and ckpt_sha != str(expected_sha) and args.unsafe_allow_checkpoint_sha_mismatch:
        print("[WARN] unsafe checkpoint SHA mismatch bypass enabled; do not use this run for paper results.")
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
    qualification = result_qualification_fields(
        feature_backend,
        fold=fold,
        seed=seed,
        unsafe_allow_checkpoint_sha_mismatch=bool(args.unsafe_allow_checkpoint_sha_mismatch),
        manifest_sha_verified=True,
        leakage_audit_ok=bool(audit.get("ok", False)),
    )
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
    method_probs: dict[str, np.ndarray] = {}
    method_aux_proto_probs: dict[str, np.ndarray] = {}
    method_scores: dict[str, dict[str, np.ndarray]] = {}

    def add_metrics_row(method: str, probs: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {**params, "method": method}
        row.update(multiclass_metrics(extracted["y_main"], probs, config.main_labels))
        row.update(calibration_metrics(extracted["y_main"], probs, n_bins=args.ece_bins))
        grid_rows.append(row)
        return row

    raw_params = calibration_relevant_parameters(
        "raw_softmax",
        prototype_temperature=None,
        softmax_temperature=None,
        softmax_weight=None,
        hier_aux_prob_weight=None,
    )
    raw_softmax_probs = extracted["softmax_probs"]
    raw_row = add_metrics_row("raw_softmax", raw_softmax_probs, raw_params)
    method_probs["raw_softmax"] = raw_softmax_probs

    softmax_rows: list[dict[str, Any]] = []
    for softmax_temp in softmax_temperatures:
        softmax_probs = temperature_scale_probabilities(extracted["softmax_probs"], softmax_temp)
        params = calibration_relevant_parameters(
            "calibrated_softmax",
            prototype_temperature=None,
            softmax_temperature=softmax_temp,
            softmax_weight=None,
            hier_aux_prob_weight=None,
        )
        softmax_rows.append(add_metrics_row("calibrated_softmax", softmax_probs, params))
    best_softmax = pd.DataFrame(softmax_rows).sort_values(
        ["nll", "ece", "macro_f1"],
        ascending=[True, True, False],
    ).iloc[0].to_dict()
    calibrated_softmax_probs = temperature_scale_probabilities(
        extracted["softmax_probs"],
        float(best_softmax["softmax_temperature"]),
    )
    method_probs["calibrated_softmax"] = calibrated_softmax_probs

    prototype_rows: list[dict[str, Any]] = []
    hierarchical_rows: list[dict[str, Any]] = []
    score_cache: dict[float, dict[str, np.ndarray]] = {}

    for proto_temp in prototype_temperatures:
        proto_scores = prototype_scores_from_bundle(
            extracted["embeddings"],
            bundle,
            temperature=proto_temp,
            hier_aux_weight=args.hier_aux_prob_weight,
        )
        score_cache[float(proto_temp)] = proto_scores
        prototype_rows.append(
            add_metrics_row(
                "prototype",
                proto_scores["prototype"],
                calibration_relevant_parameters(
                    "prototype",
                    prototype_temperature=proto_temp,
                    softmax_temperature=None,
                    softmax_weight=None,
                    hier_aux_prob_weight=None,
                ),
            )
        )
        hierarchical_rows.append(
            add_metrics_row(
                "hierarchical",
                proto_scores["hierarchical"],
                calibration_relevant_parameters(
                    "hierarchical",
                    prototype_temperature=proto_temp,
                    softmax_temperature=None,
                    softmax_weight=None,
                    hier_aux_prob_weight=args.hier_aux_prob_weight,
                ),
            )
        )

    best_prototype = pd.DataFrame(prototype_rows).sort_values(
        ["macro_f1", "nll", "ece"],
        ascending=[False, True, True],
    ).iloc[0].to_dict()
    best_hierarchical = pd.DataFrame(hierarchical_rows).sort_values(
        ["macro_f1", "nll", "ece"],
        ascending=[False, True, True],
    ).iloc[0].to_dict()

    prototype_scores_best = score_cache[float(best_prototype["prototype_temperature"])]
    hierarchical_scores_best = score_cache[float(best_hierarchical["prototype_temperature"])]
    method_probs["prototype"] = prototype_scores_best["prototype"]
    method_probs["hierarchical"] = hierarchical_scores_best["hierarchical"]
    method_aux_proto_probs["prototype"] = prototype_scores_best["aux_prototype"]
    method_aux_proto_probs["hierarchical"] = hierarchical_scores_best["aux_prototype"]
    method_scores["prototype"] = prototype_scores_best
    method_scores["hierarchical"] = hierarchical_scores_best

    fused_rows: list[dict[str, Any]] = []
    for softmax_weight in fusion_weights:
        fused_probs = fuse_probabilities(
            calibrated_softmax_probs,
            hierarchical_scores_best["hierarchical"],
            softmax_weight=softmax_weight,
        )
        params = calibration_relevant_parameters(
            "fused",
            prototype_temperature=float(best_hierarchical["prototype_temperature"]),
            softmax_temperature=float(best_softmax["softmax_temperature"]),
            softmax_weight=softmax_weight,
            hier_aux_prob_weight=args.hier_aux_prob_weight,
        )
        fused_rows.append(add_metrics_row("fused", fused_probs, params))
    best_fused = pd.DataFrame(fused_rows).sort_values(
        ["macro_f1", "nll", "ece"],
        ascending=[False, True, True],
    ).iloc[0].to_dict()
    method_probs["fused"] = fuse_probabilities(
        calibrated_softmax_probs,
        hierarchical_scores_best["hierarchical"],
        softmax_weight=float(best_fused["softmax_weight"]),
    )

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_dir / "calibration_grid.csv", index=False, encoding="utf-8-sig")

    best_rows: dict[str, dict[str, Any]] = {
        "raw_softmax": raw_row,
        "calibrated_softmax": best_softmax,
        "prototype": best_prototype,
        "hierarchical": best_hierarchical,
        "fused": best_fused,
    }
    if selection_method not in best_rows:
        raise RuntimeError(f"No calibration row for selection method={selection_method}")
    best = best_rows[selection_method]
    selected_probs = method_probs[selection_method]
    selected_pred = selected_probs.argmax(axis=1)
    selected_conf_raw = selected_probs.max(axis=1)

    main_proto_pred = method_probs["prototype"].argmax(axis=1)
    aux_proto_pred = hierarchical_scores_best["aux_prototype"].argmax(axis=1)
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

    probs_by_method = dict(method_probs)
    val_pred = prediction_frame(
        extracted["metadata"],
        extracted["y_main"],
        extracted["y_aux"],
        config.main_labels,
        config.aux_labels,
        probs_by_method,
        aux_probs=extracted["aux_softmax_probs"],
        aux_proto_probs=hierarchical_scores_best["aux_prototype"],
        prototype_scores=hierarchical_scores_best,
    )
    val_pred["selected_method"] = selection_method
    val_pred["selected_pred"] = val_pred[f"{selection_method}_pred"]
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

    method_best_params = {
        method: calibration_relevant_parameters(
            method,
            prototype_temperature=row.get("prototype_temperature"),
            softmax_temperature=row.get("softmax_temperature"),
            softmax_weight=row.get("softmax_weight"),
            hier_aux_prob_weight=row.get("hier_aux_prob_weight"),
        )
        for method, row in best_rows.items()
    }
    method_best_metrics = {
        method: {
            k: json_scalar(v)
            for k, v in row.items()
            if k
            in {
                "method",
                "n",
                "top1_acc",
                "macro_f1",
                "top2_acc",
                "ece",
                "brier",
                "nll",
                "prototype_temperature",
                "softmax_temperature",
                "softmax_weight",
                "hier_aux_prob_weight",
                "fusion_kind",
            }
        }
        for method, row in best_rows.items()
    }

    calibration = {
        "artifact_type": "hier_acoustic_prototype_calibration",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prototype_bundle": str(bundle_path.resolve()),
        "prototype_bundle_sha256": file_sha256(bundle_path),
        "checkpoint_path": str(ckpt.resolve()),
        "checkpoint_sha256": ckpt_sha,
        "train_manifest": str(train_manifest.resolve()),
        "train_manifest_sha256": file_sha256(train_manifest),
        "val_manifest": str(val_manifest.resolve()),
        "val_manifest_sha256": file_sha256(val_manifest),
        "project_relative_paths": {
            "prototype_bundle": project_relative_path(bundle_path),
            "checkpoint": project_relative_path(ckpt),
            "train_manifest": project_relative_path(train_manifest),
            "val_manifest": project_relative_path(val_manifest),
        },
        "paths": {
            "prototype_bundle": path_record(bundle_path),
            "checkpoint": path_record(ckpt),
            "train_manifest": path_record(train_manifest),
            "val_manifest": path_record(val_manifest),
        },
        "fold": fold,
        "seed": seed,
        "model_config": metadata["model_config"],
        "feature_backend": feature_backend,
        **qualification,
        "allow_relocated_checkpoint": bool(args.allow_relocated_checkpoint),
        "unsafe_allow_checkpoint_sha_mismatch": bool(args.unsafe_allow_checkpoint_sha_mismatch),
        "selection_method": selection_method,
        "method_best_params": method_best_params,
        "method_best_metrics": method_best_metrics,
        "raw_softmax": method_best_params["raw_softmax"],
        "calibrated_softmax": method_best_params["calibrated_softmax"],
        "prototype": method_best_params["prototype"],
        "hierarchical": method_best_params["hierarchical"],
        "fused": method_best_params["fused"],
        "fused_is_independent_mixed_fusion": fusion_mode_from_alpha(float(best_fused["softmax_weight"])) == "mixed",
        "fusion_result_note": (
            "fused is a mixed Softmax/prototype result only when fusion_kind=mixed; "
            "alpha=0 is pure_prototype and alpha=1 is pure_softmax."
        ),
        "prototype_temperature": (
            float(best["prototype_temperature"]) if best.get("prototype_temperature") is not None and pd.notna(best.get("prototype_temperature")) else None
        ),
        "temperature": (
            float(best["prototype_temperature"]) if best.get("prototype_temperature") is not None and pd.notna(best.get("prototype_temperature")) else None
        ),
        "softmax_temperature": (
            float(best["softmax_temperature"]) if best.get("softmax_temperature") is not None and pd.notna(best.get("softmax_temperature")) else None
        ),
        "softmax_weight": (
            float(best["softmax_weight"]) if best.get("softmax_weight") is not None and pd.notna(best.get("softmax_weight")) else None
        ),
        "hier_aux_prob_weight": float(args.hier_aux_prob_weight),
        "hier_aux_prob_weight_mode": "fixed",
        **fusion_formula_metadata(),
        "hierarchy_confidence_penalty": best_penalty,
        "target_coverage": float(args.target_coverage),
        "global_rejection_threshold": float(global_threshold),
        "per_class_rejection_thresholds": per_class_thresholds,
        "reject_label": "uncertain",
        "unknown_detection_claim": False,
        "test_used_for_calibration": False,
        "best_validation_row": {k: json_scalar(v) for k, v in best.items()},
        "best_hierarchy_penalty_row": {
            k: json_scalar(v)
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
