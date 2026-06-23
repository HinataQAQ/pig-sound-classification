from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from prototype_model_adapter import (
    ROOT,
    build_hier_dataset,
    config_to_jsonable,
    extract_embeddings,
    feature_config_hash,
    file_sha256,
    load_config,
    load_hier_model,
    require_file,
    training_exact_feature_config,
    validate_expected_config,
    validate_fold_seed_sources,
    write_json,
)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_numba_cache_dir(path: str | Path | None = None) -> Path:
    cache_dir = Path(path) if path else ROOT / ".numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir.resolve())
    return cache_dir


def prepare_out_dir(root: Path, allow_overwrite: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / "softmax_reproduction"
    if out_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing softmax reproduction directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_label_csv(text: str) -> list[str]:
    return [x.strip().lower() for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Reproduce the validated hierarchical CRNN Softmax test predictions using training-exact features."
    )
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--reference_pred_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--expected_hier_aux_weight", type=float, required=True)
    ap.add_argument("--expected_macro_f1", type=float, required=True)
    ap.add_argument("--macro_f1_tolerance", type=float, default=1e-6)
    ap.add_argument("--prob_tolerance", type=float, default=1e-5)
    ap.add_argument("--expected_dur_s", type=float, default=2.0)
    ap.add_argument("--expected_feature_mode", default="logmel")
    ap.add_argument("--expected_main_labels", default="cough,calm_grunt,feeding,stress_vocal")
    ap.add_argument("--expected_aux_labels", default="dry_cough,abdominal_cough,calm_grunt,feeding,frightened_stress,anxious_stress")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--numba_cache_dir", default=str(ROOT / ".numba_cache"))
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = ensure_numba_cache_dir(args.numba_cache_dir)
    out_dir = prepare_out_dir(Path(args.out_dir), args.allow_overwrite)

    test_manifest = require_file(args.test_manifest, "test manifest")
    ckpt = require_file(args.ckpt, "hierarchical checkpoint")
    summary_json = require_file(args.summary_json, "summary JSON")
    reference_pred_csv = require_file(args.reference_pred_csv, "reference test_pred.csv")
    validate_fold_seed_sources(
        fold=args.fold,
        seed=args.seed,
        sources={
            "test_manifest": test_manifest,
            "checkpoint": ckpt,
            "summary_json": summary_json,
            "reference_pred_csv": reference_pred_csv,
            "out_dir": args.out_dir,
        },
    )

    config = load_config(summary_path=summary_json)
    validate_expected_config(
        config,
        fold=args.fold,
        seed=args.seed,
        expected_hier_aux_weight=args.expected_hier_aux_weight,
        expected_dur_s=args.expected_dur_s,
        expected_feature_mode=args.expected_feature_mode,
        expected_main_labels=parse_label_csv(args.expected_main_labels),
        expected_aux_labels=parse_label_csv(args.expected_aux_labels),
    )
    feature_config = training_exact_feature_config(config)
    feature_config["feature_config_hash"] = feature_config_hash(feature_config)

    device = resolve_device(args.device)
    start = time.perf_counter()
    model = load_hier_model(ckpt, config, device=device)
    ds_test = build_hier_dataset(test_manifest, config, feature_backend="training_exact")
    extracted = extract_embeddings(
        model,
        ds_test,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    elapsed = time.perf_counter() - start

    ref = pd.read_csv(reference_pred_csv)
    y_true = extracted["y_main"]
    probs = extracted["softmax_probs"]
    pred_ids = probs.argmax(axis=1)
    pred_labels = [config.main_labels[int(i)] for i in pred_ids]
    true_labels = [config.main_labels[int(i)] for i in y_true]
    reproduced = pd.DataFrame(
        {
            "path": ds_test.df[ds_test.path_col].astype(str).tolist(),
            "y_true": true_labels,
            "y_pred": pred_labels,
        }
    )
    for i, label in enumerate(config.main_labels):
        reproduced[f"prob_{label}"] = probs[:, i]
    reproduced.to_csv(out_dir / "softmax_reproduced_pred.csv", index=False, encoding="utf-8-sig")

    n_test = int(len(reproduced))
    macro_f1 = float(
        f1_score(
            y_true,
            pred_ids,
            labels=list(range(len(config.main_labels))),
            average="macro",
            zero_division=0,
        )
    )
    acc = float(accuracy_score(y_true, pred_ids))
    ref_preds = ref["y_pred"].astype(str).tolist() if "y_pred" in ref.columns else []
    prediction_match = [a == b for a, b in zip(pred_labels, ref_preds)]
    prediction_match_count = int(sum(prediction_match))
    prediction_all_match = prediction_match_count == n_test and len(ref_preds) == n_test

    prob_cols = [f"prob_{label}" for label in config.main_labels]
    missing_prob_cols = [c for c in prob_cols if c not in ref.columns]
    if missing_prob_cols:
        max_prob_abs_diff = None
        prob_within_tolerance = None
    else:
        ref_probs = ref[prob_cols].to_numpy(dtype=np.float32)
        max_prob_abs_diff = float(np.max(np.abs(ref_probs - probs))) if len(ref_probs) == len(probs) else None
        prob_within_tolerance = bool(max_prob_abs_diff is not None and max_prob_abs_diff <= args.prob_tolerance)

    macro_f1_diff = float(abs(macro_f1 - args.expected_macro_f1))
    macro_f1_within_tolerance = bool(macro_f1_diff <= args.macro_f1_tolerance)
    pass_status = bool(
        n_test == 168
        and prediction_all_match
        and macro_f1_within_tolerance
    )

    diff = reproduced.copy()
    if len(ref) == len(diff):
        diff["reference_y_true"] = ref["y_true"].astype(str).tolist() if "y_true" in ref.columns else ""
        diff["reference_y_pred"] = ref["y_pred"].astype(str).tolist() if "y_pred" in ref.columns else ""
        diff["prediction_match"] = prediction_match
        for col in prob_cols:
            if col in ref.columns:
                diff[f"absdiff_{col}"] = np.abs(diff[col].to_numpy(dtype=np.float32) - ref[col].to_numpy(dtype=np.float32))
    else:
        diff["prediction_match"] = False
    diff.to_csv(out_dir / "softmax_reproduction_diff.csv", index=False, encoding="utf-8-sig")

    summary = {
        "artifact_type": "hier_exact_softmax_reproduction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_backend": "training_exact",
        "softmax_reproduction_passed": pass_status,
        "reference_macro_f1": float(args.expected_macro_f1),
        "reproduced_macro_f1": macro_f1,
        "macro_f1_abs_diff": macro_f1_diff,
        "macro_f1_tolerance": float(args.macro_f1_tolerance),
        "test_rows": n_test,
        "test_rows_expected": 168,
        "test_accuracy": acc,
        "prediction_match_count": prediction_match_count,
        "prediction_all_match": prediction_all_match,
        "probability_max_abs_diff": max_prob_abs_diff,
        "probability_tolerance": float(args.prob_tolerance),
        "probability_within_tolerance": prob_within_tolerance,
        "probability_tolerance_is_advisory": True,
        "probability_advisory_note": (
            "Softmax reproduction pass/fail is gated by test row count, Macro-F1, "
            "and exact y_pred match. Probability tolerance is recorded for numeric "
            "drift audit but is not a hard gate."
        ),
        "missing_reference_probability_columns": missing_prob_cols,
        "feature_config": feature_config,
        "package_runtime": {
            "sys_executable": sys.executable,
            "python": sys.version,
            "NUMBA_CACHE_DIR": str(cache_dir.resolve()),
        },
        "paths": {
            "test_manifest": str(test_manifest.resolve()),
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_sha256": file_sha256(ckpt),
            "summary_json": str(summary_json.resolve()),
            "reference_pred_csv": str(reference_pred_csv.resolve()),
            "reproduced_pred_csv": str((out_dir / "softmax_reproduced_pred.csv").resolve()),
            "diff_csv": str((out_dir / "softmax_reproduction_diff.csv").resolve()),
        },
        "elapsed_seconds": elapsed,
    }
    write_json(out_dir / "softmax_reproduction.json", summary)
    print(f"[OK] wrote softmax reproduction -> {out_dir / 'softmax_reproduction.json'}")
    print(f"[INFO] softmax_reproduction_passed={pass_status}")
    if not pass_status:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
