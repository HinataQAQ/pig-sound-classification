from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_FOLDS = {0, 1, 2, 3, 4}
EXPECTED_SEEDS = {42, 2024, 3407}
EXPECTED_ENVIRONMENTS = {"DWASHING", "TBUS", "STRAFFIC"}
EXPECTED_SNRS = {20.0, 10.0, 0.0}
EXPECTED_LAMBDA = 0.5
METHODS = ("raw_softmax", "prototype", "hierarchical", "fused")
PAIRED_COMPARISONS = (
    ("prototype", "raw_softmax"),
    ("hierarchical", "raw_softmax"),
    ("hierarchical", "prototype"),
)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_screening_protocol(payloads: list[dict[str, Any]], *, allow_smoke: bool = False) -> dict[str, Any]:
    if allow_smoke and len(payloads) == 1:
        return {
            "run_scope": "fold_seed_smoke",
            "screening_result": False,
            "paper_main_result": False,
            "n_fold_seed_runs": 1,
        }
    if len(payloads) != 15:
        raise ValueError(f"Legal screening requires exactly 15 fold-seed runs, got {len(payloads)}.")
    pairs = [(int(p["fold"]), int(p["seed"])) for p in payloads]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate fold-seed runs detected.")
    folds = {fold for fold, _ in pairs}
    seeds = {seed for _, seed in pairs}
    lambdas = {float(p.get("lambda")) for p in payloads}
    envs = {str(env).upper() for p in payloads for env in p.get("noise_environments", [])}
    snrs = {float(snr) for p in payloads for snr in p.get("snr_db", [])}
    if folds != EXPECTED_FOLDS:
        raise ValueError(f"Unexpected folds for legal screening: {sorted(folds)}")
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected seeds for legal screening: {sorted(seeds)}")
    if lambdas != {EXPECTED_LAMBDA}:
        raise ValueError(f"Unexpected or mixed lambda values: {sorted(lambdas)}")
    if envs != EXPECTED_ENVIRONMENTS:
        raise ValueError(f"Unexpected DEMAND environments: {sorted(envs)}")
    if snrs != EXPECTED_SNRS:
        raise ValueError(f"Unexpected active-event SNR grid: {sorted(snrs)}")
    for p in payloads:
        if not p.get("eligible_for_noise_aggregation", p.get("eligible_for_cv_aggregation", False)):
            raise ValueError(f"Run is not eligible for noise aggregation: fold={p.get('fold')} seed={p.get('seed')}")
    return {
        "run_scope": "aggregate",
        "screening_result": True,
        "final_25_run_result": False,
        "paper_candidate_result": True,
        "paper_main_result": False,
        "n_fold_seed_runs": 15,
        "folds": sorted(EXPECTED_FOLDS),
        "seeds": sorted(EXPECTED_SEEDS),
        "noise_environments": sorted(EXPECTED_ENVIRONMENTS),
        "target_active_snr_db": sorted(EXPECTED_SNRS, reverse=True),
        "lambda": EXPECTED_LAMBDA,
    }


def bootstrap_ci(delta: np.ndarray, *, n_boot: int = 10000, seed: int = 3407) -> tuple[float, float]:
    if len(delta) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, size=(n_boot, len(delta)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_stats(delta: np.ndarray) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    delta = np.asarray(delta, dtype=np.float64)
    ci_low, ci_high = bootstrap_ci(delta)
    if len(delta) == 0:
        p_value = 1.0
    elif np.allclose(delta, 0.0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(delta).pvalue)
    return {
        "n": int(len(delta)),
        "mean_delta": float(delta.mean()) if len(delta) else 0.0,
        "std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "wilcoxon_p": p_value,
        "wins": int((delta > 0).sum()),
        "ties": int(np.isclose(delta, 0.0).sum()),
        "losses": int((delta < 0).sum()),
    }


def paired_stats_by_condition(frame: pd.DataFrame) -> pd.DataFrame:
    noisy = frame[frame["noise_environment"].astype(str).str.lower() != "clean"].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in noisy.groupby(["noise_environment", "snr_db"], dropna=False):
        env, snr = keys
        pivot = group.pivot_table(index=["fold", "seed"], columns="method", values="macro_f1", aggfunc="first")
        for left, right in PAIRED_COMPARISONS:
            if left not in pivot.columns or right not in pivot.columns:
                raise RuntimeError(f"Missing method for paired comparison {left} - {right} at {env}/{snr}")
            stats = paired_stats((pivot[left] - pivot[right]).dropna().to_numpy())
            rows.append(
                {
                    "noise_environment": env,
                    "snr_db": snr,
                    "comparison": f"{left} - {right}",
                    **stats,
                }
            )
    overall = noisy.groupby(["fold", "seed", "method"], as_index=False)["macro_f1"].mean()
    pivot = overall.pivot_table(index=["fold", "seed"], columns="method", values="macro_f1", aggfunc="first")
    for left, right in PAIRED_COMPARISONS:
        if left in pivot.columns and right in pivot.columns:
            rows.append({"noise_environment": "ALL_NOISY", "snr_db": "ALL", "comparison": f"{left} - {right}", **paired_stats((pivot[left] - pivot[right]).dropna().to_numpy())})
    return pd.DataFrame(rows)


def aggregate_confusion(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for row in payload.get("confusion_by_condition_method", []):
            out = dict(row)
            out["fold"] = payload.get("fold")
            out["seed"] = payload.get("seed")
            rows.append(out)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    out_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["noise_environment", "snr_db", "method"], dropna=False):
        env, snr, method = keys
        labels = list(group.iloc[0]["labels"])
        cm = np.sum(np.stack(group["confusion_matrix"].map(np.asarray).to_list(), axis=0), axis=0)
        total_f1 = []
        out: dict[str, Any] = {
            "noise_environment": env,
            "snr_db": snr,
            "method": method,
            "labels": "|".join(labels),
            "confusion_matrix": json.dumps(cm.astype(int).tolist(), ensure_ascii=False),
        }
        for idx, label in enumerate(labels):
            tp = float(cm[idx, idx])
            fp = float(cm[:, idx].sum() - cm[idx, idx])
            fn = float(cm[idx, :].sum() - cm[idx, idx])
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            out[f"{label}_precision"] = precision
            out[f"{label}_recall"] = recall
            out[f"{label}_f1"] = f1
            total_f1.append(f1)
        out["macro_f1_from_cm"] = float(np.mean(total_f1)) if total_f1 else 0.0
        if "feeding" in labels and "stress_vocal" in labels:
            feeding = labels.index("feeding")
            stress = labels.index("stress_vocal")
            out["feeding_to_stress"] = int(cm[feeding, stress])
            out["stress_to_feeding"] = int(cm[stress, feeding])
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def degradation_slope(frame: pd.DataFrame) -> pd.DataFrame:
    noisy = frame[frame["noise_environment"].astype(str).str.lower() != "clean"].copy()
    rows: list[dict[str, Any]] = []
    if noisy.empty or "macro_f1_degradation_vs_clean" not in noisy.columns:
        return pd.DataFrame()
    noisy["snr_db_numeric"] = pd.to_numeric(noisy["snr_db"], errors="coerce")
    for keys, group in noisy.groupby(["noise_environment", "method"], dropna=False):
        env, method = keys
        valid = group.dropna(subset=["snr_db_numeric", "macro_f1_degradation_vs_clean"])
        if valid["snr_db_numeric"].nunique() < 2:
            slope = 0.0
        else:
            slope = float(np.polyfit(valid["snr_db_numeric"].astype(float), valid["macro_f1_degradation_vs_clean"].astype(float), deg=1)[0])
        rows.append({"noise_environment": env, "method": method, "macro_f1_degradation_slope_per_db": slope})
    return pd.DataFrame(rows)


def per_fold_mean_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    noisy = frame[frame["noise_environment"].astype(str).str.lower() != "clean"].copy()
    if noisy.empty:
        return pd.DataFrame()
    averaged = noisy.groupby(["fold", "method"], as_index=False)["macro_f1"].mean()
    pivot = averaged.pivot_table(index="fold", columns="method", values="macro_f1", aggfunc="first")
    rows: list[dict[str, Any]] = []
    for fold, row in pivot.iterrows():
        out: dict[str, Any] = {"fold": int(fold)}
        if {"prototype", "raw_softmax"}.issubset(pivot.columns):
            out["prototype_minus_raw_softmax"] = float(row["prototype"] - row["raw_softmax"])
        if {"hierarchical", "raw_softmax"}.issubset(pivot.columns):
            out["hierarchical_minus_raw_softmax"] = float(row["hierarchical"] - row["raw_softmax"])
        if {"hierarchical", "prototype"}.issubset(pivot.columns):
            out["hierarchical_minus_prototype"] = float(row["hierarchical"] - row["prototype"])
        rows.append(out)
    return pd.DataFrame(rows)


def summarize(
    metrics_json: list[str | Path],
    *,
    allow_smoke: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for path in metrics_json:
        payload = read_json(path)
        payloads.append(payload)
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
                "eligible_for_noise_aggregation": payload.get("eligible_for_noise_aggregation"),
                "paper_main_result": payload.get("paper_main_result"),
            }
        )
    protocol = validate_screening_protocol(payloads, allow_smoke=allow_smoke)
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
                "mean_top2_acc": float(group.get("top2_acc", pd.Series([np.nan] * len(group))).astype(float).mean()),
                "mean_ece": float(group["ece"].mean()),
                "mean_brier": float(group["brier"].mean()),
                "mean_nll": float(group["nll"].mean()),
                "mean_aurc": float(group.get("aurc", pd.Series([np.nan] * len(group))).astype(float).mean()),
                "mean_frozen_threshold_coverage": float(group.get("frozen_threshold_coverage", pd.Series([np.nan] * len(group))).astype(float).mean()),
                "mean_frozen_threshold_selective_risk": float(group.get("frozen_threshold_selective_risk", pd.Series([np.nan] * len(group))).astype(float).mean()),
                "mean_macro_f1_degradation_vs_clean": float(
                    group.get("macro_f1_degradation_vs_clean", pd.Series([np.nan] * len(group))).astype(float).mean()
                ),
            }
        )
    paired = paired_stats_by_condition(frame) if len(payloads) > 1 else pd.DataFrame()
    return (
        pd.DataFrame(grouped),
        pd.DataFrame(provenance),
        paired,
        aggregate_confusion(payloads),
        degradation_slope(frame),
        per_fold_mean_deltas(frame),
        protocol,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize zero-shot frozen prototype noise robustness metrics.")
    ap.add_argument("--metrics_json", nargs="+", required=True)
    ap.add_argument("--out_prefix", default="reports/prototype_noise_demand_w05_debug")
    ap.add_argument("--allow_smoke", action="store_true", help="Allow one fold-seed full-grid pre-screening smoke summary.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary, provenance, paired, confusion, slope, per_fold, protocol = summarize(args.metrics_json, allow_smoke=args.allow_smoke)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    provenance_path = prefix.with_name(prefix.name + "_provenance.csv")
    paired_path = prefix.with_name(prefix.name + "_paired_stats.csv")
    confusion_path = prefix.with_name(prefix.name + "_confusion_summary.csv")
    slope_path = prefix.with_name(prefix.name + "_degradation_slope.csv")
    per_fold_path = prefix.with_name(prefix.name + "_per_fold_deltas.csv")
    protocol_path = prefix.with_name(prefix.name + "_provenance.json")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    provenance.to_csv(provenance_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    confusion.to_csv(confusion_path, index=False, encoding="utf-8-sig")
    slope.to_csv(slope_path, index=False, encoding="utf-8-sig")
    per_fold.to_csv(per_fold_path, index=False, encoding="utf-8-sig")
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {provenance_path}")
    print(f"[OK] wrote {paired_path}")
    print(f"[OK] wrote {confusion_path}")
    print(f"[OK] wrote {slope_path}")
    print(f"[OK] wrote {per_fold_path}")
    print(f"[OK] wrote {protocol_path}")


if __name__ == "__main__":
    main()
