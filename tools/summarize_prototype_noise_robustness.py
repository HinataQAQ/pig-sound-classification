from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED_FOLDS = {0, 1, 2, 3, 4}
EXPECTED_SEEDS = {42, 2024, 3407}
EXPECTED_ENVIRONMENTS = {"DWASHING", "TBUS", "STRAFFIC"}
EXPECTED_SNRS = {20.0, 10.0, 0.0}
EXPECTED_LAMBDA = 0.5
EXPECTED_GLOBAL_NOISE_SEED = 3407
EXPECTED_NOISE_REPEAT = 0
EXPECTED_OFFSET_KEY_VERSION = "demand_noise_offset_v2"
METHODS = ("raw_softmax", "prototype", "hierarchical", "fused")
PAIRED_COMPARISONS = (
    ("prototype", "raw_softmax"),
    ("hierarchical", "raw_softmax"),
    ("hierarchical", "prototype"),
)
STRATA = {
    "ALL_NOISY": {20.0, 10.0, 0.0},
    "MODERATE_NOISE": {20.0, 10.0},
    "EXTREME_STRESS": {0.0},
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_float_set(values: Iterable[Any]) -> set[float]:
    return {float(x) for x in values}


def _condition_key(row: dict[str, Any]) -> tuple[str, Any, str]:
    env = str(row.get("noise_environment")).upper()
    method = str(row.get("method"))
    if env == "CLEAN":
        snr: Any = "clean"
    else:
        snr = float(row.get("target_active_snr_db", row.get("snr_db")))
    return env.lower() if env == "CLEAN" else env, snr, method


def validate_single_run_exact_grid(payload: dict[str, Any]) -> None:
    envs = {str(env).upper() for env in payload.get("noise_environments", [])}
    if envs != EXPECTED_ENVIRONMENTS:
        raise ValueError(f"Run environments must be exactly {sorted(EXPECTED_ENVIRONMENTS)}, got {sorted(envs)}")
    snrs = _as_float_set(payload.get("target_active_snr_db", payload.get("snr_db", [])))
    if snrs != EXPECTED_SNRS:
        raise ValueError(f"Run active-event SNR grid must be exactly {sorted(EXPECTED_SNRS, reverse=True)}, got {sorted(snrs, reverse=True)}")
    checks = {
        "noise_repeat": int(payload.get("noise_repeat", -1)) == EXPECTED_NOISE_REPEAT,
        "global_noise_seed": int(payload.get("global_noise_seed", -1)) == EXPECTED_GLOBAL_NOISE_SEED,
        "offset_key_version": payload.get("offset_key_version") == EXPECTED_OFFSET_KEY_VERSION,
        "simulated_noise": payload.get("simulated_noise") is True,
        "noise_protocol": payload.get("noise_protocol") == "zero_shot_frozen",
        "real_farm_external_validation": payload.get("real_farm_external_validation") is False,
        "clean_equivalence.ok": payload.get("clean_equivalence", {}).get("ok") is True,
        "provenance_gates.ok": payload.get("provenance_gates", {}).get("ok") is True,
        "same_waveform_embedding_reused": payload.get("same_waveform_embedding_reused") is True,
        "same_offset_across_snr_verified": payload.get("same_offset_across_snr_verified") is True,
        "test_parameter_selection": payload.get("test_parameter_selection") is False,
        "feature_backend": payload.get("feature_backend") == "training_exact",
        "eligible_for_noise_aggregation": payload.get("eligible_for_noise_aggregation") is True,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(
            f"Run failed required provenance gates for fold={payload.get('fold')} seed={payload.get('seed')}: {failed}"
        )
    if payload.get("run_stage") == "screening":
        if payload.get("single_fold_debug") is not False or payload.get("run_scope") != "fold_seed":
            raise ValueError("screening run must have single_fold_debug=false and run_scope=fold_seed")
    rows = [row for row in payload.get("metrics_by_condition", []) if row.get("method") in METHODS]
    expected = {("clean", "clean", method) for method in METHODS}
    expected.update((env, snr, method) for env in EXPECTED_ENVIRONMENTS for snr in EXPECTED_SNRS for method in METHODS)
    actual = [_condition_key(row) for row in rows]
    duplicates = sorted({key for key in actual if actual.count(key) > 1})
    missing = sorted(expected - set(actual), key=str)
    extra = sorted(set(actual) - expected, key=str)
    if duplicates or missing or extra:
        raise ValueError(
            "Run condition/method grid mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}, duplicates={duplicates[:5]}"
        )


def validate_screening_protocol(payloads: list[dict[str, Any]], *, allow_smoke: bool = False) -> dict[str, Any]:
    if allow_smoke and len(payloads) == 1:
        validate_single_run_exact_grid(payloads[0])
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
        validate_single_run_exact_grid(p)
        if p.get("run_stage") != "screening":
            raise ValueError(f"Expected run_stage=screening for legal screening, got {p.get('run_stage')}")
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
        "global_noise_seed": EXPECTED_GLOBAL_NOISE_SEED,
        "noise_repeat": EXPECTED_NOISE_REPEAT,
        "offset_key_version": EXPECTED_OFFSET_KEY_VERSION,
        "simulated_noise": True,
        "real_farm_external_validation": False,
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
            deltas = pivot[left] - pivot[right]
            if len(deltas) != 15 or deltas.isna().any():
                raise RuntimeError(f"Strict paired n must be 15 for {left} - {right} at {env}/{snr}; got paired n={deltas.dropna().shape[0]}")
            stats = paired_stats(deltas.to_numpy())
            rows.append(
                {
                    "noise_environment": env,
                    "snr_db": snr,
                    "comparison": f"{left} - {right}",
                    **stats,
                }
            )
    for stratum, snrs in STRATA.items():
        subset = noisy[pd.to_numeric(noisy["snr_db"], errors="coerce").isin(snrs)].copy()
        averaged = subset.groupby(["fold", "seed", "method"], as_index=False)["macro_f1"].mean()
        pivot = averaged.pivot_table(index=["fold", "seed"], columns="method", values="macro_f1", aggfunc="first")
        for left, right in PAIRED_COMPARISONS:
            if left not in pivot.columns or right not in pivot.columns:
                raise RuntimeError(f"Missing method for paired comparison {left} - {right} in {stratum}")
            deltas = pivot[left] - pivot[right]
            if len(deltas) != 15 or deltas.isna().any():
                raise RuntimeError(f"Strict paired n must be 15 for {left} - {right} in {stratum}; got paired n={deltas.dropna().shape[0]}")
            rows.append(
                {
                    "noise_environment": stratum,
                    "snr_db": "GROUP",
                    "condition_family": "extreme simulated-noise stress condition" if stratum == "EXTREME_STRESS" else stratum.lower(),
                    "comparison": f"{left} - {right}",
                    **paired_stats(deltas.to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def fold_cluster_paired_stats(frame: pd.DataFrame) -> pd.DataFrame:
    noisy = frame[frame["noise_environment"].astype(str).str.lower() != "clean"].copy()
    noisy["snr_numeric"] = pd.to_numeric(noisy["snr_db"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for stratum, snrs in STRATA.items():
        subset = noisy[noisy["snr_numeric"].isin(snrs)].copy()
        averaged = subset.groupby(["fold", "seed", "method"], as_index=False)["macro_f1"].mean()
        pivot_seed = averaged.pivot_table(index=["fold", "seed"], columns="method", values="macro_f1", aggfunc="first")
        for left, right in PAIRED_COMPARISONS:
            if left not in pivot_seed.columns or right not in pivot_seed.columns:
                raise RuntimeError(f"Missing method for fold-cluster comparison {left} - {right} in {stratum}")
            seed_deltas = (pivot_seed[left] - pivot_seed[right]).reset_index(name="delta")
            fold_deltas = seed_deltas.groupby("fold", as_index=False)["delta"].mean()
            if len(fold_deltas) != 5:
                raise RuntimeError(f"Fold-cluster stats require 5 folds for {left} - {right} in {stratum}; got {len(fold_deltas)}")
            delta = fold_deltas["delta"].to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_ci(delta)
            rows.append(
                {
                    "noise_environment": stratum,
                    "snr_db": "GROUP",
                    "comparison": f"{left} - {right}",
                    "n_folds": int(len(delta)),
                    "fold_level_mean_delta": float(delta.mean()),
                    "fold_level_std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                    "fold_cluster_bootstrap_ci95_low": ci_low,
                    "fold_cluster_bootstrap_ci95_high": ci_high,
                    "fold_level_wins": int((delta > 0).sum()),
                    "fold_level_ties": int(np.isclose(delta, 0.0).sum()),
                    "fold_level_losses": int((delta < 0).sum()),
                    "wilcoxon_on_5_fold_means_low_power": paired_stats(delta)["wilcoxon_p"],
                }
            )
    return pd.DataFrame(rows)


def build_cross_seed_draw_audit(provenance_csvs: list[str | Path]) -> pd.DataFrame:
    frames = []
    required = {
        "fold",
        "seed",
        "clean_md5",
        "noise_environment",
        "noise_repeat",
        "offset_key_version",
        "noise_draw_id",
        "noise_offset",
        "noise_sha256",
        "selected_channel",
        "global_noise_seed",
    }
    for path in provenance_csvs:
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Noise provenance missing required columns in {path}: {sorted(missing)}")
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    noisy = all_rows[all_rows["noise_environment"].astype(str).str.lower() != "clean"].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in noisy.groupby(["fold", "clean_md5", "noise_environment", "noise_repeat", "offset_key_version"], dropna=False):
        seeds = sorted(int(x) for x in set(group["seed"]))
        field_uniques = {
            field: sorted(set(str(x) for x in group[field]))
            for field in ("noise_draw_id", "noise_offset", "noise_sha256", "selected_channel", "global_noise_seed")
        }
        ok = seeds == sorted(EXPECTED_SEEDS) and all(len(values) == 1 for values in field_uniques.values())
        row = {
            "fold": int(keys[0]),
            "clean_md5": keys[1],
            "noise_environment": keys[2],
            "noise_repeat": int(keys[3]),
            "offset_key_version": keys[4],
            "seeds": "|".join(str(x) for x in seeds),
            "ok": bool(ok),
        }
        for field, values in field_uniques.items():
            row[f"{field}_unique_count"] = int(len(values))
            row[field] = values[0] if len(values) == 1 else "|".join(values)
        rows.append(row)
    audit = pd.DataFrame(rows)
    bad = audit[~audit["ok"]] if not audit.empty else audit
    if not bad.empty:
        raise ValueError(f"cross-seed noise draw audit failed for {len(bad)} groups")
    return audit


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

    def append_group(env: str, snr: Any, method: str, group: pd.DataFrame, extra: dict[str, Any] | None = None) -> None:
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
        if "cough" in labels:
            cough = labels.index("cough")
            for label in labels:
                if label != "cough":
                    out[f"cough_to_{label}"] = int(cm[cough, labels.index(label)])
        if extra:
            out.update(extra)
        out_rows.append(out)
    for keys, group in frame.groupby(["noise_environment", "snr_db", "method"], dropna=False):
        env, snr, method = keys
        append_group(env, snr, method, group)
    noisy = frame[frame["noise_environment"].astype(str).str.lower() != "clean"].copy()
    noisy["snr_numeric"] = pd.to_numeric(noisy["snr_db"], errors="coerce")
    for stratum, snrs in STRATA.items():
        subset = noisy[noisy["snr_numeric"].isin(snrs)]
        for method, group in subset.groupby("method", dropna=False):
            append_group(
                stratum,
                "GROUP",
                method,
                group,
                {"condition_family": "extreme simulated-noise stress condition" if stratum == "EXTREME_STRESS" else stratum.lower()},
            )
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
    rows: list[dict[str, Any]] = []
    noisy["snr_numeric"] = pd.to_numeric(noisy["snr_db"], errors="coerce")
    for stratum, snrs in STRATA.items():
        subset = noisy[noisy["snr_numeric"].isin(snrs)]
        averaged = subset.groupby(["fold", "method"], as_index=False)["macro_f1"].mean()
        pivot = averaged.pivot_table(index="fold", columns="method", values="macro_f1", aggfunc="first")
        for fold, row in pivot.iterrows():
            out: dict[str, Any] = {"noise_environment": stratum, "snr_db": "GROUP", "fold": int(fold)}
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
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
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
                "run_stage": payload.get("run_stage"),
                "run_scope": payload.get("run_scope"),
                "global_noise_seed": payload.get("global_noise_seed"),
                "noise_repeat": payload.get("noise_repeat"),
                "offset_key_version": payload.get("offset_key_version"),
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
    noisy = frame[frame["noise_environment"].astype(str).str.lower() != "clean"].copy()
    noisy["snr_numeric"] = pd.to_numeric(noisy["snr_db"], errors="coerce")
    for stratum, snrs in STRATA.items():
        subset = noisy[noisy["snr_numeric"].isin(snrs)].copy()
        averaged = subset.groupby(["fold", "seed", "method"], as_index=False).agg(
            macro_f1=("macro_f1", "mean"),
            top1_acc=("top1_acc", "mean"),
            top2_acc=("top2_acc", "mean"),
            ece=("ece", "mean"),
            brier=("brier", "mean"),
            nll=("nll", "mean"),
            aurc=("aurc", "mean"),
            frozen_threshold_coverage=("frozen_threshold_coverage", "mean"),
            frozen_threshold_selective_risk=("frozen_threshold_selective_risk", "mean"),
        )
        for method, group in averaged.groupby("method", dropna=False):
            grouped.append(
                {
                    "noise_environment": stratum,
                    "snr_db": "GROUP",
                    "condition_family": "extreme simulated-noise stress condition" if stratum == "EXTREME_STRESS" else stratum.lower(),
                    "method": method,
                    "n": int(len(group)),
                    "mean_macro_f1": float(group["macro_f1"].mean()),
                    "std_macro_f1": float(group["macro_f1"].std(ddof=1)) if len(group) > 1 else 0.0,
                    "mean_top1_acc": float(group["top1_acc"].mean()),
                    "mean_top2_acc": float(group["top2_acc"].mean()),
                    "mean_ece": float(group["ece"].mean()),
                    "mean_brier": float(group["brier"].mean()),
                    "mean_nll": float(group["nll"].mean()),
                    "mean_aurc": float(group["aurc"].mean()),
                    "mean_frozen_threshold_coverage": float(group["frozen_threshold_coverage"].mean()),
                    "mean_frozen_threshold_selective_risk": float(group["frozen_threshold_selective_risk"].mean()),
                    "mean_macro_f1_degradation_vs_clean": float("nan"),
                }
            )
    paired = paired_stats_by_condition(frame) if len(payloads) > 1 else pd.DataFrame()
    fold_cluster = fold_cluster_paired_stats(frame) if len(payloads) > 1 else pd.DataFrame()
    cross_seed = pd.DataFrame()
    if len(payloads) > 1:
        provenance_paths: list[str | Path] = []
        for payload in payloads:
            outputs = payload.get("outputs", {})
            path = outputs.get("noise_sample_provenance_csv")
            if not path:
                raise RuntimeError(f"Run missing noise_sample_provenance_csv output: fold={payload.get('fold')} seed={payload.get('seed')}")
            provenance_paths.append(path)
        cross_seed = build_cross_seed_draw_audit(provenance_paths)
        protocol["cross_seed_noise_draw_audit_ok"] = True
    return (
        pd.DataFrame(grouped),
        pd.DataFrame(provenance),
        paired,
        aggregate_confusion(payloads),
        degradation_slope(frame),
        per_fold_mean_deltas(frame),
        fold_cluster,
        cross_seed,
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
    summary, provenance, paired, confusion, slope, per_fold, fold_cluster, cross_seed, protocol = summarize(args.metrics_json, allow_smoke=args.allow_smoke)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    provenance_path = prefix.with_name(prefix.name + "_provenance.csv")
    paired_path = prefix.with_name(prefix.name + "_paired_stats.csv")
    confusion_path = prefix.with_name(prefix.name + "_confusion_summary.csv")
    slope_path = prefix.with_name(prefix.name + "_degradation_slope.csv")
    per_fold_path = prefix.with_name(prefix.name + "_per_fold_deltas.csv")
    fold_cluster_path = prefix.with_name(prefix.name + "_fold_cluster_paired_stats.csv")
    cross_seed_path = prefix.with_name(prefix.name + "_cross_seed_draw_audit.csv")
    cross_seed_json_path = prefix.with_name(prefix.name + "_cross_seed_draw_audit.json")
    protocol_path = prefix.with_name(prefix.name + "_provenance.json")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    provenance.to_csv(provenance_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    confusion.to_csv(confusion_path, index=False, encoding="utf-8-sig")
    slope.to_csv(slope_path, index=False, encoding="utf-8-sig")
    per_fold.to_csv(per_fold_path, index=False, encoding="utf-8-sig")
    fold_cluster.to_csv(fold_cluster_path, index=False, encoding="utf-8-sig")
    if not cross_seed.empty:
        cross_seed.to_csv(cross_seed_path, index=False, encoding="utf-8-sig")
        cross_seed_json_path.write_text(
            json.dumps(
                {
                    "ok": bool(cross_seed["ok"].all()),
                    "n_groups": int(len(cross_seed)),
                    "csv": str(cross_seed_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        protocol["cross_seed_noise_draw_audit_csv"] = str(cross_seed_path)
        protocol["cross_seed_noise_draw_audit_json"] = str(cross_seed_json_path)
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {provenance_path}")
    print(f"[OK] wrote {paired_path}")
    print(f"[OK] wrote {confusion_path}")
    print(f"[OK] wrote {slope_path}")
    print(f"[OK] wrote {per_fold_path}")
    print(f"[OK] wrote {fold_cluster_path}")
    if not cross_seed.empty:
        print(f"[OK] wrote {cross_seed_path}")
        print(f"[OK] wrote {cross_seed_json_path}")
    print(f"[OK] wrote {protocol_path}")


if __name__ == "__main__":
    main()
