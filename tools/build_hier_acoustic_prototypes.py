from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prototype_model_adapter import (
    DEFAULT_AUX_LABELS,
    DEFAULT_MAIN_LABELS,
    assert_no_leakage,
    audit_manifest_disjointness,
    build_hier_dataset,
    compute_class_prototypes,
    compute_distance_statistics,
    compute_feature_stats,
    config_to_jsonable,
    feature_config_hash,
    file_sha256,
    infer_fold_seed,
    load_config,
    load_hier_model,
    path_record,
    path_sha256_map,
    project_relative_path,
    prediction_frame,
    prototype_probabilities_from_bundle,
    prototype_sample_frame,
    require_file,
    result_qualification_fields,
    save_prototype_bundle,
    training_exact_feature_config,
    validate_expected_config,
    validate_fold_seed_sources,
    write_json,
    extract_embeddings,
)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_label_csv(text: str, *, name: str) -> list[str]:
    labels = [x.strip().lower() for x in str(text).split(",") if x.strip()]
    if not labels:
        raise ValueError(f"{name} must contain at least one label.")
    return labels


def prepare_stage_dir(root: Path, stage: str, allow_overwrite: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stage_dir = root / stage
    if stage_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing {stage} directory: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def assert_new_file(path: Path, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build train-only acoustic prototypes from a trained hierarchical CRNN embedding."
    )
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--out_dir", required=True, help="Phase root directory; artifacts/ is created inside it.")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--expected_hier_aux_weight", type=float, required=True)
    ap.add_argument("--expected_dur_s", type=float, default=2.0)
    ap.add_argument("--expected_feature_mode", default="logmel")
    ap.add_argument("--expected_main_labels", default=",".join(DEFAULT_MAIN_LABELS))
    ap.add_argument("--expected_aux_labels", default=",".join(DEFAULT_AUX_LABELS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--feature_backend", choices=["training_exact", "librosa", "numpy_logmel"], default="training_exact")
    ap.add_argument("--prototype_temperature", type=float, default=1.0)
    ap.add_argument("--hier_aux_prob_weight", type=float, default=0.5)
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    train_manifest = require_file(args.train_manifest, "train manifest")
    val_manifest = require_file(args.val_manifest, "validation manifest")
    test_manifest = require_file(args.test_manifest, "test manifest")
    ckpt = require_file(args.ckpt, "hierarchical checkpoint")
    summary_json = require_file(args.summary_json, "training summary JSON")
    out_root = Path(args.out_dir)
    artifacts_dir = prepare_stage_dir(out_root, "artifacts", args.allow_overwrite)

    leakage = audit_manifest_disjointness(
        {"train": train_manifest, "val": val_manifest, "test": test_manifest}
    )
    assert_no_leakage(leakage)
    write_json(artifacts_dir / "leakage_audit.json", leakage)

    inferred = infer_fold_seed(train_manifest, ckpt, summary_json, out_root)
    fold = int(args.fold)
    seed = int(args.seed)
    validate_fold_seed_sources(
        fold=fold,
        seed=seed,
        sources={
            "train_manifest": train_manifest,
            "val_manifest": val_manifest,
            "test_manifest": test_manifest,
            "checkpoint": ckpt,
            "summary_json": summary_json,
            "out_dir": out_root,
        },
    )

    config = load_config(summary_path=summary_json)
    expected_main = parse_label_csv(args.expected_main_labels, name="expected_main_labels")
    expected_aux = parse_label_csv(args.expected_aux_labels, name="expected_aux_labels")
    validate_expected_config(
        config,
        fold=fold,
        seed=seed,
        expected_hier_aux_weight=args.expected_hier_aux_weight,
        expected_dur_s=args.expected_dur_s,
        expected_feature_mode=args.expected_feature_mode,
        expected_main_labels=expected_main,
        expected_aux_labels=expected_aux,
    )

    device = resolve_device(args.device)
    model = load_hier_model(ckpt, config, device=device)
    ds_train = build_hier_dataset(train_manifest, config, feature_backend=args.feature_backend)
    extracted = extract_embeddings(
        model,
        ds_train,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    main_prototypes, main_counts = compute_class_prototypes(
        extracted["embeddings"],
        extracted["y_main"],
        num_classes=len(config.main_labels),
    )
    aux_prototypes, aux_counts = compute_class_prototypes(
        extracted["embeddings"],
        extracted["y_aux"],
        num_classes=len(config.aux_labels),
    )
    feature_stats = compute_feature_stats(
        ds_train,
        num_main_classes=len(config.main_labels),
        num_aux_classes=len(config.aux_labels),
    )

    main_distance_stats = compute_distance_statistics(
        extracted["embeddings"],
        extracted["y_main"],
        main_prototypes,
        class_labels=config.main_labels,
        split_name="train_main",
    )
    aux_distance_stats = compute_distance_statistics(
        extracted["embeddings"],
        extracted["y_aux"],
        aux_prototypes,
        class_labels=config.aux_labels,
        split_name="train_auxiliary",
    )
    distance_statistics = {
        "artifact_type": "prototype_train_distance_statistics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold": fold,
        "seed": seed,
        "main": main_distance_stats,
        "auxiliary": aux_distance_stats,
    }
    write_json(artifacts_dir / "distance_statistics.json", distance_statistics)

    hashes = path_sha256_map(
        {
            "checkpoint": ckpt,
            "checkpoint_summary": summary_json,
            "train_manifest": train_manifest,
            "val_manifest": val_manifest,
            "test_manifest": test_manifest,
        }
    )
    if args.feature_backend in {"training_exact", "librosa"}:
        feature_params = training_exact_feature_config(config)
        feature_params["feature_backend"] = "training_exact"
    else:
        feature_params = {
            "feature_backend": args.feature_backend,
            "feature_mode": config.feature_mode,
            "sr": config.sr,
            "dur_s": config.dur_s,
            "n_mels": config.n_mels,
            "n_mfcc": config.n_mfcc,
            "fmin": config.fmin,
            "fmax": config.fmax,
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "win_length": config.win_length,
        }
    qualification = result_qualification_fields(
        feature_params["feature_backend"],
        fold=fold,
        seed=seed,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "artifact_type": "hier_acoustic_prototype_bundle",
        "prototype_created_at_utc": created_at,
        "created_at_utc": created_at,
        "fold": fold,
        "seed": seed,
        "inferred_fold_seed": inferred,
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_path": str(ckpt.resolve()),
        "checkpoint_sha256": file_sha256(ckpt),
        "checkpoint_summary": str(summary_json.resolve()),
        "checkpoint_summary_path": str(summary_json.resolve()),
        "checkpoint_summary_sha256": file_sha256(summary_json),
        "train_manifest": str(train_manifest.resolve()),
        "val_manifest": str(val_manifest.resolve()),
        "test_manifest": str(test_manifest.resolve()),
        "manifest_sha256": {
            "train": file_sha256(train_manifest),
            "val": file_sha256(val_manifest),
            "test": file_sha256(test_manifest),
        },
        "project_relative_paths": {
            "checkpoint": project_relative_path(ckpt),
            "checkpoint_summary": project_relative_path(summary_json),
            "train_manifest": project_relative_path(train_manifest),
            "val_manifest": project_relative_path(val_manifest),
            "test_manifest": project_relative_path(test_manifest),
        },
        "paths": {
            "checkpoint": path_record(ckpt),
            "checkpoint_summary": path_record(summary_json),
            "train_manifest": path_record(train_manifest),
            "val_manifest": path_record(val_manifest),
            "test_manifest": path_record(test_manifest),
        },
        "input_hashes": hashes,
        "model_config": config_to_jsonable(config),
        "feature_backend": feature_params["feature_backend"],
        **qualification,
        "feature_params": feature_params,
        "feature_config_hash": feature_config_hash(feature_params),
        "dur_s": float(config.dur_s),
        "hier_aux_weight": config.hier_aux_weight,
        "main_labels": config.main_labels,
        "aux_labels": config.aux_labels,
        "train_rows_after_filtering": int(len(ds_train)),
        "embedding_dim": int(extracted["embeddings"].shape[1]),
        "embedding_normalization": "L2 before prototype averaging",
        "prototype_normalization": "L2 after class mean",
        "prototype_source_split": "train",
        "calibration_source_split": "validation",
        "evaluation_source_split": "test",
        "main_counts": {lab: int(count) for lab, count in zip(config.main_labels, main_counts)},
        "aux_counts": {lab: int(count) for lab, count in zip(config.aux_labels, aux_counts)},
        "leakage_audit_path": str((artifacts_dir / "leakage_audit.json").resolve()),
        "distance_statistics_path": str((artifacts_dir / "distance_statistics.json").resolve()),
        "unknown_detection_claim": False,
    }

    main_npz = artifacts_dir / "main_prototypes.npz"
    aux_npz = artifacts_dir / "aux_prototypes.npz"
    bundle_path = artifacts_dir / "prototype_bundle.npz"
    for path in [main_npz, aux_npz, bundle_path, artifacts_dir / "prototype_metadata.json"]:
        assert_new_file(path, args.allow_overwrite)

    np.savez_compressed(
        main_npz,
        main_prototypes=main_prototypes.astype(np.float32),
        main_labels=np.asarray(config.main_labels),
        main_counts=main_counts.astype(np.int64),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    np.savez_compressed(
        aux_npz,
        aux_prototypes=aux_prototypes.astype(np.float32),
        aux_labels=np.asarray(config.aux_labels),
        aux_counts=aux_counts.astype(np.int64),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    save_prototype_bundle(
        bundle_path,
        main_prototypes=main_prototypes,
        aux_prototypes=aux_prototypes,
        main_counts=main_counts,
        aux_counts=aux_counts,
        main_labels=config.main_labels,
        aux_labels=config.aux_labels,
        metadata=metadata,
        feature_stats=feature_stats,
    )
    write_json(artifacts_dir / "prototype_metadata.json", metadata)

    train_main_samples = prototype_sample_frame(
        extracted["metadata"],
        extracted["embeddings"],
        extracted["y_main"],
        main_prototypes,
        config.main_labels,
        label_prefix="main",
    )
    train_aux_samples = prototype_sample_frame(
        extracted["metadata"],
        extracted["embeddings"],
        extracted["y_aux"],
        aux_prototypes,
        config.aux_labels,
        label_prefix="aux",
    )
    train_main_samples.to_csv(artifacts_dir / "train_main_prototype_samples.csv", index=False, encoding="utf-8-sig")
    train_aux_samples.to_csv(artifacts_dir / "train_aux_prototype_samples.csv", index=False, encoding="utf-8-sig")

    bundle_like = {
        "main_prototypes": main_prototypes,
        "aux_prototypes": aux_prototypes,
        "main_labels": config.main_labels,
        "aux_labels": config.aux_labels,
    }
    proto_scores = prototype_probabilities_from_bundle(
        extracted["embeddings"],
        bundle_like,
        temperature=args.prototype_temperature,
        hier_aux_weight=args.hier_aux_prob_weight,
    )
    train_pred = prediction_frame(
        extracted["metadata"],
        extracted["y_main"],
        extracted["y_aux"],
        config.main_labels,
        config.aux_labels,
        {
            "softmax": extracted["softmax_probs"],
            "prototype": proto_scores["prototype"],
            "hierarchical": proto_scores["hierarchical"],
        },
        aux_probs=extracted["aux_softmax_probs"],
        aux_proto_probs=proto_scores["aux_prototype"],
        prototype_scores=proto_scores,
    )
    train_pred.to_csv(artifacts_dir / "train_prototype_predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"label": config.main_labels, "count": main_counts.astype(int)}).to_csv(
        artifacts_dir / "main_prototype_counts.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"label": config.aux_labels, "count": aux_counts.astype(int)}).to_csv(
        artifacts_dir / "aux_prototype_counts.csv", index=False, encoding="utf-8-sig"
    )

    run_manifest = {
        "artifact_type": "prototype_phase_run_manifest",
        "created_at_utc": created_at,
        "phase": "fold0_seed3407_debug" if (fold == 0 and seed == 3407) else "fold_seed_debug",
        "fold": fold,
        "seed": seed,
        "outputs": {
            "artifacts_dir": str(artifacts_dir.resolve()),
            "main_prototypes_npz": str(main_npz.resolve()),
            "aux_prototypes_npz": str(aux_npz.resolve()),
            "prototype_bundle_npz": str(bundle_path.resolve()),
            "prototype_metadata_json": str((artifacts_dir / "prototype_metadata.json").resolve()),
            "distance_statistics_json": str((artifacts_dir / "distance_statistics.json").resolve()),
        },
        "no_test_tuning": True,
        "unknown_detection_claim": False,
        **qualification,
    }
    run_manifest_path = out_root / "run_manifest.json"
    assert_new_file(run_manifest_path, args.allow_overwrite)
    write_json(run_manifest_path, run_manifest)

    print(f"[OK] wrote main prototypes -> {main_npz}")
    print(f"[OK] wrote aux prototypes -> {aux_npz}")
    print(f"[OK] wrote prototype bundle -> {bundle_path}")
    print(f"[OK] wrote metadata -> {artifacts_dir / 'prototype_metadata.json'}")
    print(f"[OK] wrote distance statistics -> {artifacts_dir / 'distance_statistics.json'}")


if __name__ == "__main__":
    main()
