from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from prototype_model_adapter import read_json, require_file, write_json


def parse_int_csv(text: str, *, name: str) -> list[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer.")
    return values


def _canonical_combo(record: Mapping[str, Any]) -> tuple[int, int]:
    return int(record["fold"]), int(record["seed"])


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def validate_aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_folds: Sequence[int],
    expected_seeds: Sequence[int],
    expected_lambda: float | None = None,
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("No run records provided for aggregate summary.")

    expected = {(int(fold), int(seed)) for fold in expected_folds for seed in expected_seeds}
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

    duplicates = sorted(combo for combo, count in seen.items() if count > 1)
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    if duplicates:
        errors.append(f"duplicate fold-seed combinations: {duplicates}")
    if missing:
        errors.append(f"missing fold-seed combinations: {missing}")
    if extra:
        errors.append(f"unexpected fold-seed combinations: {extra}")
    if expected_lambda is not None and lambdas != {float(expected_lambda)}:
        errors.append(f"lambda mismatch: expected {expected_lambda}, got {sorted(lambdas)}")
    elif len(lambdas) != 1:
        errors.append(f"mixed lambda values: {sorted(lambdas)}")

    if errors:
        raise RuntimeError("Aggregate validation failed: " + "; ".join(errors))

    n_runs = len(records)
    screening = len(expected_folds) == 5 and len(expected_seeds) == 3 and n_runs == 15
    final_25 = len(expected_folds) == 5 and len(expected_seeds) == 5 and n_runs == 25
    return {
        "run_scope": "aggregate",
        "paper_main_result": True,
        "screening_result": bool(screening),
        "final_25_run_result": bool(final_25),
        "n_runs": int(n_runs),
        "folds": [int(x) for x in expected_folds],
        "seeds": [int(x) for x in expected_seeds],
        "lambda": float(next(iter(lambdas))),
    }


def record_from_metrics(metrics_json: str | Path) -> dict[str, Any]:
    metrics_path = require_file(metrics_json, "metrics JSON")
    metrics = read_json(metrics_path)
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
        "raw_softmax_macro_f1": metrics.get("methods", {}).get("raw_softmax", {}).get("macro_f1"),
        "hierarchical_macro_f1": metrics.get("methods", {}).get("hierarchical", {}).get("macro_f1"),
    }

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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Summarize exact hierarchical prototype CV runs and set aggregate-only paper result provenance."
    )
    ap.add_argument("--metrics_json", nargs="+", required=True, help="Per-run evaluation metrics.json files.")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--expected_folds", default="0,1,2,3,4")
    ap.add_argument("--expected_seeds", required=True)
    ap.add_argument("--expected_lambda", type=float, default=None)
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_json = Path(args.out_json)
    if out_json.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {out_json}")
    out_csv = Path(args.out_csv) if args.out_csv else None
    if out_csv is not None and out_csv.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {out_csv}")

    records = [record_from_metrics(path) for path in args.metrics_json]
    aggregate = validate_aggregate_records(
        records,
        expected_folds=parse_int_csv(args.expected_folds, name="expected_folds"),
        expected_seeds=parse_int_csv(args.expected_seeds, name="expected_seeds"),
        expected_lambda=args.expected_lambda,
    )
    payload = {
        "artifact_type": "cv5_exact_prototype_aggregate_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **aggregate,
        "records": records,
    }
    write_json(out_json, payload)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote aggregate summary -> {out_json}")


if __name__ == "__main__":
    main()
