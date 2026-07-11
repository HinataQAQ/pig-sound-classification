from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from nomenclature import (
    build_method_metadata,
    build_model_metadata,
    reconcile_selection_protocol,
    resolve_inference_method,
    resolve_inference_method_fields,
    validate_recorded_artifact_identity,
    validate_recorded_nomenclature_metadata,
)
from prototype_model_adapter import (
    calibration_metrics,
    config_from_summary,
    coverage_risk_curve,
    file_sha256,
    hierarchy_consistency_rate,
    method_metrics_table,
    multiclass_metrics,
    read_json,
    require_file,
    result_qualification_fields,
    write_json,
)


METHODS = ["raw_softmax", "calibrated_softmax", "prototype", "hierarchical", "fused"]
REQUIRED_QUALIFICATION_FIELDS = [
    "run_scope",
    "feature_pipeline_equivalent",
    "single_fold_debug",
    "eligible_for_cv_aggregation",
    "paper_main_result",
    "feature_backend",
    "leakage_audit_ok",
    "manifest_sha_verified",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate frozen test predictions from the hierarchical acoustic prototype predictor."
    )
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--calibration_json", required=True)
    ap.add_argument("--out_dir", required=True, help="Phase root directory; evaluation/ is used inside it.")
    ap.add_argument("--ece_bins", type=int, default=10)
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def prepare_evaluation_dir(root: Path, allow_overwrite: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    planned = [
        out_dir / "metrics_by_method_test.csv",
        out_dir / "coverage_risk.csv",
        out_dir / "selective_risk_summary.csv",
        out_dir / "hierarchy_consistency.json",
        out_dir / "metrics.json",
        out_dir / "confusion_matrices.json",
    ]
    for path in planned:
        if path.exists() and not allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    return out_dir


def method_probs(df: pd.DataFrame, method: str, labels: list[str]) -> np.ndarray:
    cols = [f"{method}_prob_{label}" for label in labels]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing probability columns for method={method}: {missing}")
    return df[cols].to_numpy(dtype=np.float32)


def aux_probs(df: pd.DataFrame, labels: list[str]) -> np.ndarray:
    cols = [f"prototype_aux_prob_{label}" for label in labels]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing auxiliary prototype probability columns: {missing}")
    return df[cols].to_numpy(dtype=np.float32)


def selective_summary(df: pd.DataFrame, accept_col: str) -> dict[str, float | int | str | None]:
    accepted = df[accept_col].astype(bool).to_numpy()
    n_total = int(len(df))
    n_accept = int(accepted.sum())
    if n_accept == 0:
        acc = None
        risk = None
    else:
        acc = float((df.loc[accepted, "selected_pred"] == df.loc[accepted, "y_true"]).mean())
        risk = float(1.0 - acc)
    return {
        "accept_rule": accept_col,
        "n_total": n_total,
        "n_accepted": n_accept,
        "coverage": float(n_accept / n_total) if n_total else 0.0,
        "selective_acc": acc,
        "selective_risk": risk,
    }


def jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
            out[key] = None
        else:
            out[key] = value
    return out


def _require_sha_match(path: Path, expected: Any, description: str) -> None:
    if not expected:
        raise RuntimeError(f"prediction metadata missing required {description} SHA256.")
    actual = file_sha256(path)
    if actual.lower() != str(expected).lower():
        raise RuntimeError(f"{description} SHA256 mismatch: expected {expected}, got {actual}")


def validate_evaluation_input_nomenclature(
    prediction: dict[str, Any], calibration: dict[str, Any] | None
) -> None:
    """Reconcile explicit schema identities across evaluation inputs."""

    try:
        validate_recorded_nomenclature_metadata(prediction)
    except ValueError as exc:
        raise ValueError(
            f"Invalid prediction nomenclature metadata: {exc}"
        ) from exc
    if calibration is None:
        return
    try:
        validate_recorded_nomenclature_metadata(calibration)
    except ValueError as exc:
        raise ValueError(
            f"Invalid calibration nomenclature metadata: {exc}"
        ) from exc
    try:
        config = config_from_summary(calibration["model_config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Calibration is missing a valid model_config."
        ) from exc
    calibration_method, calibration_route = resolve_inference_method_fields(
        calibration
    )
    prediction_method, prediction_route = resolve_inference_method_fields(
        prediction
    )
    if (calibration_method, calibration_route) != (
        prediction_method,
        prediction_route,
    ):
        raise ValueError(
            "Conflicting inference method provenance between calibration and "
            "prediction metadata: "
            f"{(calibration_method, calibration_route)!r} != "
            f"{(prediction_method, prediction_route)!r}."
        )
    expected_stage = (
        "b3_hierarchical_prototype_top1_ablation"
        if calibration_route == "hierarchical_prototype_candidate"
        else "b2_validation_selected_hierarchical_crnn"
    )
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
    try:
        validated_prediction = validate_recorded_artifact_identity(
            prediction,
            expected_model_family="hierarchical_supervision_crnn",
            expected_training_stage=expected_stage,
            expected_context_seconds=float(config.dur_s),
            context="prediction",
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid prediction nomenclature metadata: {exc}"
        ) from exc
    if (
        validated_calibration is not None
        and validated_prediction is not None
        and "selection_protocol" in validated_calibration
        and "selection_protocol" in validated_prediction
    ):
        reconcile_selection_protocol(
            str(validated_calibration["selection_protocol"]),
            requested=str(validated_prediction["selection_protocol"]),
        )


def load_and_validate_prediction_metadata(
    pred_csv: str | Path,
    calibration_json: str | Path,
    metadata_path: str | Path | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pred_path = require_file(pred_csv, "prediction CSV")
    cal_path = require_file(calibration_json, "calibration JSON")
    meta_path = Path(metadata_path) if metadata_path is not None else pred_path.parent / "prediction_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing prediction_metadata.json: {meta_path}")
    metadata = read_json(meta_path)
    validate_evaluation_input_nomenclature(metadata, calibration)

    input_role = str(metadata.get("input_role", ""))
    if input_role != "frozen_test":
        raise RuntimeError(f"Evaluation requires input_role=frozen_test; got input_role={input_role}.")

    _require_sha_match(pred_path, metadata.get("prediction_csv_sha256"), "prediction CSV")
    _require_sha_match(cal_path, metadata.get("calibration_json_sha256"), "calibration JSON")

    missing = [field for field in REQUIRED_QUALIFICATION_FIELDS if field not in metadata]
    if missing:
        raise RuntimeError(f"prediction metadata missing qualification fields: {missing}")
    if metadata.get("run_scope") != "fold_seed":
        raise RuntimeError(f"Evaluation requires run_scope=fold_seed; got {metadata.get('run_scope')}.")
    if bool(metadata.get("paper_main_result")):
        raise RuntimeError("Single fold-seed prediction metadata must have paper_main_result=false.")
    if not bool(metadata.get("leakage_audit_ok")):
        raise RuntimeError("Prediction metadata indicates leakage_audit_ok=false.")
    if not bool(metadata.get("manifest_sha_verified")):
        raise RuntimeError("Prediction metadata indicates manifest_sha_verified=false.")
    if bool(metadata.get("unsafe_allow_checkpoint_sha_mismatch", False)):
        raise RuntimeError("Unsafe checkpoint SHA mismatch runs cannot be evaluated as frozen test results.")

    if calibration is not None:
        for key in ("fold", "seed"):
            if metadata.get(key) != calibration.get(key):
                raise RuntimeError(
                    f"prediction metadata {key} mismatch: metadata={metadata.get(key)}, "
                    f"calibration={calibration.get(key)}"
                )
        if str(metadata.get("feature_backend")) != str(calibration.get("feature_backend")):
            raise RuntimeError(
                "prediction metadata feature_backend mismatch: "
                f"metadata={metadata.get('feature_backend')}, calibration={calibration.get('feature_backend')}"
            )
    return metadata


def main() -> None:
    args = parse_args()

    pred_csv = require_file(args.pred_csv, "prediction CSV")
    calibration_json = require_file(args.calibration_json, "calibration JSON")
    calibration = read_json(calibration_json)
    prediction_metadata = load_and_validate_prediction_metadata(
        pred_csv,
        calibration_json,
        calibration=calibration,
    )
    config = config_from_summary(calibration["model_config"])
    labels = config.main_labels
    aux_labels = config.aux_labels
    selected_method, selected_route = resolve_inference_method_fields(
        calibration, warn_on_legacy=True
    )
    prediction_method, prediction_route = resolve_inference_method_fields(
        prediction_metadata, warn_on_legacy=True
    )
    if (prediction_method, prediction_route) != (
        selected_method,
        selected_route,
    ):
        raise ValueError(
            "Conflicting inference method provenance between calibration and "
            "prediction metadata: "
            f"{(selected_method, selected_route)!r} != "
            f"{(prediction_method, prediction_route)!r}."
        )
    selection_protocol = reconcile_selection_protocol(
        prediction_metadata.get("selection_protocol"),
        requested=calibration.get("selection_protocol"),
    )

    def output_metadata_for_method(method: str) -> dict[str, object]:
        storage_id, route = resolve_inference_method(method)
        training_stage = (
            "b3_hierarchical_prototype_top1_ablation"
            if route == "hierarchical_prototype_candidate"
            else "b2_validation_selected_hierarchical_crnn"
        )
        common = build_model_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage=training_stage,
            context_seconds=float(config.dur_s),
            selection_protocol=selection_protocol,
        )
        if route is None:
            return common
        return build_method_metadata(
            model_family="hierarchical_supervision_crnn",
            training_stage=training_stage,
            context_seconds=float(config.dur_s),
            selection_protocol=selection_protocol,
            inference_route=route,
            legacy_method_id=storage_id,
        )

    selected_route_metadata = output_metadata_for_method(selected_method)

    out_dir = prepare_evaluation_dir(Path(args.out_dir), args.allow_overwrite)

    df = pd.read_csv(pred_csv)
    input_role = str(prediction_metadata["input_role"])
    if df["y_true"].isna().all() or (df["y_true"].astype(str).str.len() == 0).all():
        raise RuntimeError("Evaluation requires labeled manifest predictions, not single-audio predictions.")
    y_true = df["y_true_id"].to_numpy(dtype=np.int64)
    y_aux_true = df["aux_true_id"].to_numpy(dtype=np.int64)

    probs_by_method = {method: method_probs(df, method, labels) for method in METHODS}
    metrics = method_metrics_table(y_true, probs_by_method, labels, n_bins=args.ece_bins)
    method_metadata = pd.DataFrame(
        [
            output_metadata_for_method(str(method))
            for method in metrics["method"]
        ]
    )
    metrics = pd.concat(
        [metrics.reset_index(drop=True), method_metadata.reset_index(drop=True)],
        axis=1,
    )
    metrics.to_csv(out_dir / "metrics_by_method_test.csv", index=False, encoding="utf-8-sig")

    aux_prototype_probs = aux_probs(df, aux_labels)
    aux_metrics = multiclass_metrics(y_aux_true, aux_prototype_probs, aux_labels)
    aux_metrics.update(calibration_metrics(y_aux_true, aux_prototype_probs, n_bins=args.ece_bins))

    confusion: dict[str, Any] = {}
    for method, probs in probs_by_method.items():
        y_pred = probs.argmax(axis=1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        confusion[method] = {
            "labels": labels,
            "matrix": cm.astype(int).tolist(),
        }
    aux_pred = aux_prototype_probs.argmax(axis=1)
    confusion["auxiliary_prototype"] = {
        "labels": aux_labels,
        "matrix": confusion_matrix(y_aux_true, aux_pred, labels=list(range(len(aux_labels)))).astype(int).tolist(),
    }
    write_json(out_dir / "confusion_matrices.json", confusion)

    selected_pred_ids = np.asarray([labels.index(str(x)) for x in df["selected_pred"].astype(str)], dtype=np.int64)
    selected_conf = df["selected_confidence"].to_numpy(dtype=np.float32)
    threshold_grid = sorted(
        set(
            np.linspace(0.0, 1.0, 101).tolist()
            + [float(calibration["global_rejection_threshold"])]
        )
    )
    coverage_rows = coverage_risk_curve(y_true, selected_pred_ids, selected_conf, threshold_grid)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "coverage_risk.csv", index=False, encoding="utf-8-sig")

    selective = pd.DataFrame(
        [
            selective_summary(df, "accept_global"),
            selective_summary(df, "accept_per_class"),
        ]
    )
    selective.to_csv(out_dir / "selective_risk_summary.csv", index=False, encoding="utf-8-sig")

    consistency: dict[str, float] = {}
    if {"raw_softmax_pred", "softmax_aux_pred"}.issubset(df.columns):
        consistency["raw_softmax_main_vs_aux"] = hierarchy_consistency_rate(
            df["raw_softmax_pred"].astype(str).tolist(),
            df["softmax_aux_pred"].astype(str).tolist(),
        )
    if {"calibrated_softmax_pred", "softmax_aux_pred"}.issubset(df.columns):
        consistency["calibrated_softmax_main_vs_aux"] = hierarchy_consistency_rate(
            df["calibrated_softmax_pred"].astype(str).tolist(),
            df["softmax_aux_pred"].astype(str).tolist(),
        )
    if {"prototype_pred", "prototype_aux_pred"}.issubset(df.columns):
        consistency["prototype_main_vs_aux"] = hierarchy_consistency_rate(
            df["prototype_pred"].astype(str).tolist(),
            df["prototype_aux_pred"].astype(str).tolist(),
        )
    if {"hierarchical_pred", "prototype_aux_pred"}.issubset(df.columns):
        consistency["hierarchical_main_vs_aux"] = hierarchy_consistency_rate(
            df["hierarchical_pred"].astype(str).tolist(),
            df["prototype_aux_pred"].astype(str).tolist(),
        )
    if {"fused_pred", "prototype_aux_pred"}.issubset(df.columns):
        consistency["fused_main_vs_aux"] = hierarchy_consistency_rate(
            df["fused_pred"].astype(str).tolist(),
            df["prototype_aux_pred"].astype(str).tolist(),
        )
    consistency["prototype_main_aux_inconsistent_rate"] = float(df["hierarchy_inconsistent"].astype(bool).mean())
    write_json(out_dir / "hierarchy_consistency.json", consistency)

    selected_probs = probs_by_method[selected_method]
    selected_cal = calibration_metrics(y_true, selected_probs, n_bins=args.ece_bins)
    selected_metrics = metrics[metrics["method"] == selected_method].iloc[0].to_dict()
    metrics_records = {
        str(row["method"]): jsonable_row(row)
        for row in metrics.to_dict(orient="records")
    }
    metrics_json = {
        "artifact_type": "hier_acoustic_prototype_test_evaluation",
        **selected_route_metadata,
        "prediction_csv": str(pred_csv.resolve()),
        "calibration_json": str(calibration_json.resolve()),
        "fold": calibration.get("fold"),
        "seed": calibration.get("seed"),
        "labels": labels,
        "aux_labels": aux_labels,
        **result_qualification_fields(
            prediction_metadata.get("feature_backend", calibration.get("feature_backend", "training_exact")),
            fold=calibration.get("fold"),
            seed=calibration.get("seed"),
            input_role=input_role,
            unsafe_allow_checkpoint_sha_mismatch=bool(
                prediction_metadata.get("unsafe_allow_checkpoint_sha_mismatch", False)
            ),
            manifest_sha_verified=bool(prediction_metadata.get("manifest_sha_verified", False)),
            leakage_audit_ok=bool(prediction_metadata.get("leakage_audit_ok", False)),
        ),
        "input_role": input_role,
        "prediction_metadata_json": str((pred_csv.parent / "prediction_metadata.json").resolve()),
        "prediction_csv_sha256": prediction_metadata.get("prediction_csv_sha256"),
        "calibration_json_sha256": prediction_metadata.get("calibration_json_sha256"),
        "leakage_audit_ok": bool(prediction_metadata.get("leakage_audit_ok", False)),
        "manifest_sha_verified": bool(prediction_metadata.get("manifest_sha_verified", False)),
        "selection_method": selected_method,
        "method_best_params": calibration.get("method_best_params", {}),
        "prototype_temperature": calibration.get("prototype_temperature"),
        "softmax_temperature": calibration.get("softmax_temperature"),
        "softmax_weight": calibration.get("softmax_weight"),
        "hierarchy_confidence_penalty": float(calibration.get("hierarchy_confidence_penalty", 1.0)),
        "global_rejection_threshold": float(calibration["global_rejection_threshold"]),
        "per_class_rejection_thresholds": calibration["per_class_rejection_thresholds"],
        "unknown_detection_claim": False,
        "test_used_for_parameter_selection": False,
        "methods": metrics_records,
        "auxiliary_prototype": aux_metrics,
        "selected_method_metrics": jsonable_row(selected_metrics),
        "selected_method_calibration": selected_cal,
        "selective_risk": [jsonable_row(x) for x in selective.to_dict(orient="records")],
        "hierarchy_consistency": consistency,
        "outputs": {
            "metrics_by_method_csv": str((out_dir / "metrics_by_method_test.csv").resolve()),
            "coverage_risk_csv": str((out_dir / "coverage_risk.csv").resolve()),
            "selective_risk_csv": str((out_dir / "selective_risk_summary.csv").resolve()),
            "confusion_matrices_json": str((out_dir / "confusion_matrices.json").resolve()),
        },
    }
    write_json(out_dir / "metrics.json", metrics_json)

    print(
        "[INFO] selected inference method -> "
        f"{selected_route_metadata.get('display_name_en', selected_method)}"
    )
    print(f"[OK] wrote test metrics -> {out_dir / 'metrics.json'}")
    print(f"[OK] wrote coverage-risk -> {out_dir / 'coverage_risk.csv'}")
    print(f"[OK] wrote confusion matrices -> {out_dir / 'confusion_matrices.json'}")


if __name__ == "__main__":
    main()
