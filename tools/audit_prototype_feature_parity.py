from __future__ import annotations

import argparse
import importlib
import json
import os
import site
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prototype_model_adapter import (
    ROOT,
    config_to_jsonable,
    load_config,
    training_exact_feature_config,
    feature_config_hash,
    build_hier_dataset,
    require_file,
    write_json,
)


PACKAGE_NAMES = ["torch", "numpy", "scipy", "librosa", "numba", "soundfile"]


def parse_split_counts(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid split count item {item!r}; expected split:n")
        split, raw_count = item.split(":", 1)
        out[split.strip()] = int(raw_count)
    if sum(out.values()) < 16:
        raise ValueError(f"At least 16 files are required, got {sum(out.values())}")
    return out


def ensure_numba_cache_dir(path: str | Path | None = None) -> Path:
    cache_dir = Path(path) if path else ROOT / ".numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir.resolve())
    return cache_dir


def timed_import(module_name: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
        elapsed = time.perf_counter() - start
        return {
            "module": module_name,
            "ok": True,
            "seconds": elapsed,
            "version": getattr(module, "__version__", None),
            "file": getattr(module, "__file__", None),
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        elapsed = time.perf_counter() - start
        return {
            "module": module_name,
            "ok": False,
            "seconds": elapsed,
            "error": repr(exc),
        }


def environment_audit(cache_dir: Path, import_reports: list[dict[str, Any]]) -> dict[str, Any]:
    user_site = site.getusersitepackages()
    packages = {row["module"]: row for row in import_reports if row["module"] in PACKAGE_NAMES}
    librosa_file = str(packages.get("librosa", {}).get("file", "") or "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sys_executable": sys.executable,
        "sys_version": sys.version,
        "sys_path": sys.path,
        "site_ENABLE_USER_SITE": site.ENABLE_USER_SITE,
        "site_user_site": user_site,
        "conda_prefix": conda_prefix,
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        "NUMBA_CACHE_DIR": str(cache_dir.resolve()),
        "packages": packages,
        "librosa_from_user_site": bool(user_site and librosa_file.lower().startswith(str(user_site).lower())),
        "librosa_from_conda_env": bool(conda_prefix and librosa_file.lower().startswith(str(conda_prefix).lower())),
        "import_reports": import_reports,
    }


def summarize_feature(x: np.ndarray, prefix: str) -> dict[str, Any]:
    arr = np.asarray(x)
    return {
        f"{prefix}_shape": "x".join(str(v) for v in arr.shape),
        f"{prefix}_dtype": str(arr.dtype),
        f"{prefix}_finite": bool(np.isfinite(arr).all()),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def compare_features(exact: np.ndarray, approx: np.ndarray) -> dict[str, Any]:
    if exact.shape != approx.shape:
        return {
            "shape_match": False,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "cosine_similarity": None,
        }
    a = exact.astype(np.float64).reshape(-1)
    b = approx.astype(np.float64).reshape(-1)
    diff = np.abs(a - b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "shape_match": True,
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": float(np.max(diff)),
        "cosine_similarity": float(np.dot(a, b) / denom) if denom > 0 else None,
    }


def selected_indices(count: int, n: int) -> list[int]:
    if n > count:
        raise ValueError(f"Requested {n} rows from split with only {count} rows.")
    return list(range(n))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Audit parity between training-exact Log-Mel features and the numpy_logmel debug backend."
    )
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_counts", default="train:6,val:5,test:5")
    ap.add_argument("--numba_cache_dir", default=str(ROOT / ".numba_cache"))
    ap.add_argument("--allow_overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing feature parity directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = ensure_numba_cache_dir(args.numba_cache_dir)
    import_reports = [timed_import(name) for name in PACKAGE_NAMES]
    train_import = timed_import("train_hier_longcontext_crnn")
    import_reports.append(train_import)
    env = environment_audit(cache_dir, import_reports)
    write_json(out_dir / "environment_audit.json", env)

    config = load_config(summary_path=require_file(args.summary_json, "summary JSON"))
    feature_config = training_exact_feature_config(config)
    feature_config["feature_config_hash"] = feature_config_hash(feature_config)

    manifests = {
        "train": require_file(args.train_manifest, "train manifest"),
        "val": require_file(args.val_manifest, "validation manifest"),
        "test": require_file(args.test_manifest, "test manifest"),
    }
    split_counts = parse_split_counts(args.split_counts)

    rows: list[dict[str, Any]] = []
    exact_times: list[float] = []
    numpy_times: list[float] = []
    for split, manifest in manifests.items():
        n = split_counts.get(split, 0)
        if n <= 0:
            continue
        exact_ds = build_hier_dataset(manifest, config, feature_backend="training_exact")
        numpy_ds = build_hier_dataset(manifest, config, feature_backend="numpy_logmel")
        for idx in selected_indices(len(exact_ds), n):
            meta = exact_ds.df.iloc[idx].to_dict()
            path_col = exact_ds.path_col

            t0 = time.perf_counter()
            exact_x, exact_y, exact_aux = exact_ds[idx]
            exact_sec = time.perf_counter() - t0
            exact_times.append(exact_sec)

            t1 = time.perf_counter()
            numpy_x, numpy_y, numpy_aux = numpy_ds[idx]
            numpy_sec = time.perf_counter() - t1
            numpy_times.append(numpy_sec)

            exact_arr = exact_x.detach().cpu().numpy()
            numpy_arr = numpy_x.detach().cpu().numpy()
            row: dict[str, Any] = {
                "split": split,
                "index": int(idx),
                "path": str(meta.get(path_col, "")),
                "label": str(meta.get(exact_ds.label_col, "")),
                "subtype": str(meta.get("__aux_label", meta.get("subtype", ""))),
                "source_id": str(meta.get("source_id", "")),
                "md5": str(meta.get("md5", "")),
                "exact_main_id": int(exact_y.item()),
                "numpy_main_id": int(numpy_y.item()),
                "exact_aux_id": int(exact_aux.item()),
                "numpy_aux_id": int(numpy_aux.item()),
                "exact_seconds": exact_sec,
                "numpy_seconds": numpy_sec,
            }
            row.update(summarize_feature(exact_arr, "exact"))
            row.update(summarize_feature(numpy_arr, "numpy"))
            row.update(compare_features(exact_arr, numpy_arr))
            rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "feature_parity_report.csv", index=False, encoding="utf-8-sig")

    timing = {
        "training_module_import_seconds": train_import.get("seconds"),
        "exact_first_file_seconds": exact_times[0] if exact_times else None,
        "exact_second_file_seconds": exact_times[1] if len(exact_times) > 1 else None,
        "exact_16_file_average_seconds": float(np.mean(exact_times[:16])) if len(exact_times) >= 16 else None,
        "exact_all_file_average_seconds": float(np.mean(exact_times)) if exact_times else None,
        "numpy_first_file_seconds": numpy_times[0] if numpy_times else None,
        "numpy_second_file_seconds": numpy_times[1] if len(numpy_times) > 1 else None,
        "numpy_16_file_average_seconds": float(np.mean(numpy_times[:16])) if len(numpy_times) >= 16 else None,
        "numpy_all_file_average_seconds": float(np.mean(numpy_times)) if numpy_times else None,
    }
    summary = {
        "artifact_type": "prototype_feature_parity_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_config": feature_config,
        "audited_parameters": feature_config,
        "n_files": int(len(report)),
        "split_counts": split_counts,
        "timing": timing,
        "mean_abs_diff_mean": float(report["mean_abs_diff"].mean()) if len(report) else None,
        "max_abs_diff_max": float(report["max_abs_diff"].max()) if len(report) else None,
        "cosine_similarity_mean": float(report["cosine_similarity"].mean()) if len(report) else None,
        "environment_audit": str((out_dir / "environment_audit.json").resolve()),
        "csv_report": str((out_dir / "feature_parity_report.csv").resolve()),
    }
    write_json(out_dir / "feature_parity_report.json", summary)

    print(f"[OK] wrote feature parity CSV -> {out_dir / 'feature_parity_report.csv'}")
    print(f"[OK] wrote feature parity JSON -> {out_dir / 'feature_parity_report.json'}")
    print(f"[OK] wrote environment audit -> {out_dir / 'environment_audit.json'}")


if __name__ == "__main__":
    main()
