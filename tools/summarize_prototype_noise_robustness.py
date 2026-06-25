from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(metrics_json: list[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for path in metrics_json:
        payload = read_json(path)
        if not payload.get("simulated_noise"):
            raise RuntimeError(f"Missing simulated_noise=true in {path}")
        if payload.get("noise_protocol") != "zero_shot_frozen":
            raise RuntimeError(f"Unexpected noise_protocol in {path}: {payload.get('noise_protocol')}")
        for row in payload.get("metrics_by_condition", []):
            if row.get("method") in {"raw_softmax", "prototype", "hierarchical", "fused"}:
                out = dict(row)
                out["fold"] = payload.get("fold")
                out["seed"] = payload.get("seed")
                out["lambda"] = payload.get("lambda")
                rows.append(out)
        provenance.append(
            {
                "metrics_json": str(path),
                "fold": payload.get("fold"),
                "seed": payload.get("seed"),
                "lambda": payload.get("lambda"),
                "feature_backend": payload.get("feature_backend"),
                "simulated_noise": payload.get("simulated_noise"),
                "noise_protocol": payload.get("noise_protocol"),
                "real_farm_external_validation": payload.get("real_farm_external_validation"),
                "leakage_audit_ok": payload.get("leakage_audit_ok"),
                "clean_equivalence_ok": payload.get("clean_equivalence", {}).get("ok"),
                "same_waveform_embedding_reused": payload.get("same_waveform_embedding_reused"),
                "snr_error_abs_max": payload.get("snr_error_abs_max"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No method metrics found in supplied noise metrics JSON files.")
    grouped = []
    for keys, group in frame.groupby(["noise_environment", "snr_db", "method"], dropna=False):
        env, snr, method = keys
        grouped.append(
            {
                "noise_environment": env,
                "snr_db": snr,
                "method": method,
                "n": int(len(group)),
                "mean_macro_f1": float(group["macro_f1"].mean()),
                "std_macro_f1": float(group["macro_f1"].std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_top1_acc": float(group["top1_acc"].mean()),
                "mean_ece": float(group["ece"].mean()),
                "mean_brier": float(group["brier"].mean()),
                "mean_nll": float(group["nll"].mean()),
                "mean_macro_f1_degradation_vs_clean": float(
                    group.get("macro_f1_degradation_vs_clean", pd.Series([np.nan] * len(group))).astype(float).mean()
                ),
            }
        )
    return pd.DataFrame(grouped), pd.DataFrame(provenance)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize zero-shot frozen prototype noise robustness metrics.")
    ap.add_argument("--metrics_json", nargs="+", required=True)
    ap.add_argument("--out_prefix", default="reports/prototype_noise_demand_w05_debug")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary, provenance = summarize(args.metrics_json)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    provenance_path = prefix.with_name(prefix.name + "_provenance.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    provenance.to_csv(provenance_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {provenance_path}")


if __name__ == "__main__":
    main()
