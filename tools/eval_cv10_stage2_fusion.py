from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

LABELS_4 = ["cough", "calm_grunt", "feeding", "stress_vocal"]
LABELS_BIN = ["feeding", "stress_vocal"]


def infer_path_col(df: pd.DataFrame) -> str:
    for c in ["path", "filepath", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer path column from columns: {list(df.columns)}")


def infer_label_col(df: pd.DataFrame) -> str:
    for c in ["label", "target", "y"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer label column from columns: {list(df.columns)}")


def normalize_path(x) -> str:
    return str(x).replace("\\", "/").strip()


def normalize_label(x, labels):
    """
    兼容两种 test_pred.csv：
      y_pred = feeding / stress_vocal / cough ...
      y_pred = 0 / 1 / 2 / 3
    """
    s = str(x).strip().lower()

    if s in labels:
        return s

    # 兼容 "0", "1", "2", "3", "0.0"
    try:
        idx = int(float(s))
        if 0 <= idx < len(labels):
            return labels[idx]
    except Exception:
        pass

    # 兼容 "tensor(0)" 之类格式
    m = re.search(r"-?\d+", s)
    if m:
        try:
            idx = int(m.group(0))
            if 0 <= idx < len(labels):
                return labels[idx]
        except Exception:
            pass

    raise RuntimeError(f"Cannot normalize label value={x!r} using labels={labels}")


def read_pred_with_manifest(pred_csv: Path, manifest_csv: Path, pred_labels):
    """
    不再相信 pred_csv 里面的 y_true。
    y_true 统一从 manifest.csv 读取，避免 y_true 是数字导致报错。
    """
    if not pred_csv.exists():
        raise FileNotFoundError(pred_csv)

    if not manifest_csv.exists():
        raise FileNotFoundError(manifest_csv)

    pred = pd.read_csv(pred_csv)
    manifest = pd.read_csv(manifest_csv)

    path_col = infer_path_col(manifest)
    label_col = infer_label_col(manifest)

    if len(pred) != len(manifest):
        raise RuntimeError(
            f"Length mismatch:\n"
            f"  pred={pred_csv} len={len(pred)}\n"
            f"  manifest={manifest_csv} len={len(manifest)}"
        )

    if "y_pred" not in pred.columns:
        raise RuntimeError(f"{pred_csv} must contain y_pred column.")

    out = pd.DataFrame()
    out["path"] = manifest[path_col].map(normalize_path).values
    out["y_true"] = manifest[label_col].astype(str).str.strip().str.lower().values
    out["y_pred"] = [normalize_label(v, pred_labels) for v in pred["y_pred"].values]

    return out


def stage1_pred_path(fold: int, seed: int, stage1_variant: str) -> Path:
    """
    stage1_variant:
      baseline -> reports/cv10_fold{fold}_logmel_seed{seed}/test_pred.csv
      pretrain -> reports/cv10_pretrain_no_feeding_fold{fold}_logmel_seed{seed}/test_pred.csv
    """
    if stage1_variant == "baseline":
        return ROOT / "reports" / f"cv10_fold{fold}_logmel_seed{seed}" / "test_pred.csv"

    if stage1_variant == "pretrain":
        return ROOT / "reports" / f"cv10_pretrain_no_feeding_fold{fold}_logmel_seed{seed}" / "test_pred.csv"

    raise ValueError(f"Unknown stage1_variant={stage1_variant}")


def expert_pred_path(fold: int, seed: int, expert_model: str) -> Path:
    """
    expert_model:
      logmel
      mctafd_no_se_gated
    """
    return ROOT / "reports" / f"cv10_binary_fold{fold}_{expert_model}_seed{seed}" / "test_pred.csv"


def evaluate_one(fold: int, seed: int, expert_model: str, stage1_variant: str):
    full_manifest = (
        ROOT
        / "data"
        / "manifests_pigvocal_4class_dedup_cv10"
        / f"fold{fold}"
        / "test.csv"
    )

    binary_manifest = (
        ROOT
        / "data"
        / "manifests_pigvocal_feeding_stress_cv10"
        / f"fold{fold}"
        / "test.csv"
    )

    s1_csv = stage1_pred_path(fold, seed, stage1_variant)
    ex_csv = expert_pred_path(fold, seed, expert_model)

    stage1 = read_pred_with_manifest(
        pred_csv=s1_csv,
        manifest_csv=full_manifest,
        pred_labels=LABELS_4,
    )

    expert = read_pred_with_manifest(
        pred_csv=ex_csv,
        manifest_csv=binary_manifest,
        pred_labels=LABELS_BIN,
    )

    expert_map = {
        normalize_path(r["path"]): str(r["y_pred"]).strip().lower()
        for _, r in expert.iterrows()
    }

    y_true = []
    y_pred_stage1 = []
    y_pred_stage2 = []

    replaced = 0

    for _, r in stage1.iterrows():
        path = normalize_path(r["path"])
        true = str(r["y_true"]).strip().lower()
        pred1 = str(r["y_pred"]).strip().lower()

        pred2 = pred1

        # 保守二阶段策略：
        # 只有 Stage 1 已经判断为 feeding 或 stress_vocal，
        # 并且该样本确实在 binary test manifest 中，才调用专家修正。
        if pred1 in {"feeding", "stress_vocal"} and path in expert_map:
            pred2 = expert_map[path]
            replaced += 1

        y_true.append(true)
        y_pred_stage1.append(pred1)
        y_pred_stage2.append(pred2)

    stage1_acc = accuracy_score(y_true, y_pred_stage1)
    stage1_mf1 = f1_score(
        y_true,
        y_pred_stage1,
        labels=LABELS_4,
        average="macro",
        zero_division=0,
    )

    stage2_acc = accuracy_score(y_true, y_pred_stage2)
    stage2_mf1 = f1_score(
        y_true,
        y_pred_stage2,
        labels=LABELS_4,
        average="macro",
        zero_division=0,
    )

    cm1 = confusion_matrix(y_true, y_pred_stage1, labels=LABELS_4)
    cm2 = confusion_matrix(y_true, y_pred_stage2, labels=LABELS_4)

    return {
        "fold": fold,
        "seed": seed,
        "expert_model": expert_model,
        "stage1_variant": stage1_variant,
        "n": len(y_true),
        "replaced": replaced,
        "stage1_acc": stage1_acc,
        "stage1_macro_f1": stage1_mf1,
        "stage2_acc": stage2_acc,
        "stage2_macro_f1": stage2_mf1,
        "delta_macro_f1": stage2_mf1 - stage1_mf1,
        "cm_stage1": cm1,
        "cm_stage2": cm2,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--expert_model",
        required=True,
        choices=["logmel", "mctafd_no_se_gated"],
    )

    ap.add_argument(
        "--stage1_variant",
        default="baseline",
        choices=["baseline", "pretrain"],
        help="baseline=cv10_fold*_logmel, pretrain=cv10_pretrain_no_feeding_fold*_logmel",
    )

    ap.add_argument("--seeds", type=str, default="3407,42,2024")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4,5,6,7,8,9")

    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    rows = []
    cm1_all = []
    cm2_all = []

    missing = []

    for seed in seeds:
        for fold in folds:
            s1_csv = stage1_pred_path(fold, seed, args.stage1_variant)
            ex_csv = expert_pred_path(fold, seed, args.expert_model)

            if not s1_csv.exists():
                missing.append(str(s1_csv))

            if not ex_csv.exists():
                missing.append(str(ex_csv))

    if missing:
        print("\n[MISSING FILES]")
        for p in missing:
            print(p)
        raise SystemExit("[FAIL] Some required test_pred.csv files are missing. Train missing runs first.")

    for seed in seeds:
        for fold in folds:
            r = evaluate_one(
                fold=fold,
                seed=seed,
                expert_model=args.expert_model,
                stage1_variant=args.stage1_variant,
            )

            rows.append({k: v for k, v in r.items() if not k.startswith("cm_")})
            cm1_all.append(r["cm_stage1"])
            cm2_all.append(r["cm_stage2"])

    df = pd.DataFrame(rows)

    out_runs = (
        ROOT
        / "reports"
        / f"cv10_stage2_{args.stage1_variant}_{args.expert_model}_runs.csv"
    )

    df.to_csv(out_runs, index=False, encoding="utf-8-sig")

    cm1 = np.stack(cm1_all, axis=0).sum(axis=0)
    cm2 = np.stack(cm2_all, axis=0).sum(axis=0)

    delta = df["delta_macro_f1"].values

    rng = np.random.default_rng(3407)
    boots = []

    for _ in range(10000):
        sample = rng.choice(delta, size=len(delta), replace=True)
        boots.append(sample.mean())

    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    try:
        stat, p = wilcoxon(df["stage2_macro_f1"], df["stage1_macro_f1"])
    except ValueError:
        p = np.nan

    summary = {
        "stage1_variant": args.stage1_variant,
        "expert_model": args.expert_model,
        "n_runs": len(df),
        "stage1_mean_macro_f1": df["stage1_macro_f1"].mean(),
        "stage1_std_macro_f1": df["stage1_macro_f1"].std(),
        "stage2_mean_macro_f1": df["stage2_macro_f1"].mean(),
        "stage2_std_macro_f1": df["stage2_macro_f1"].std(),
        "mean_delta": df["delta_macro_f1"].mean(),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "wilcoxon_p": p,
        "mean_replaced": df["replaced"].mean(),
    }

    out_summary = (
        ROOT
        / "reports"
        / f"cv10_stage2_{args.stage1_variant}_{args.expert_model}_summary.csv"
    )

    pd.DataFrame([summary]).to_csv(out_summary, index=False, encoding="utf-8-sig")

    print("\n[STAGE2 SUMMARY]")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n[STAGE1 CONFUSION]")
    print(pd.DataFrame(cm1, index=LABELS_4, columns=LABELS_4).to_string())

    print("\n[STAGE2 CONFUSION]")
    print(pd.DataFrame(cm2, index=LABELS_4, columns=LABELS_4).to_string())

    print(f"\n[OK] wrote -> {out_runs}")
    print(f"[OK] wrote -> {out_summary}")


if __name__ == "__main__":
    main()