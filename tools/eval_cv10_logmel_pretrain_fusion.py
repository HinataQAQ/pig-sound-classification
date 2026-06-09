from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from train_mctafd_crnn_pretrain import (
    PigVocalFeatureDataset,
    AblationCRNN,
    feature_channels,
)


LABELS = ["cough", "calm_grunt", "feeding", "stress_vocal"]


def parse_list_int(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_float(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


@torch.no_grad()
def extract_model_state(ckpt):
    """
    兼容多种 checkpoint 保存格式：

    1. torch.save(model.state_dict(), path)
    2. torch.save({"model": model.state_dict(), ...}, path)
    3. torch.save({"state_dict": model.state_dict(), ...}, path)
    4. DataParallel 保存的 module.xxx 前缀
    """
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            # 可能它本身就是 state_dict
            state = ckpt
    else:
        state = ckpt

    # 去掉 DataParallel 的 module. 前缀
    cleaned = {}

    for k, v in state.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    return cleaned


@torch.no_grad()
def predict_probs(
    ckpt_path: Path,
    manifest_path: Path,
    device: str,
    feature_mode: str = "logmel",
    use_se: bool = True,
    n_mels: int = 64,
    fmax: float = 8000,
    batch_size: int = 16,
):
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    ds = PigVocalFeatureDataset(
        manifest=str(manifest_path),
        labels=LABELS,
        feature_mode=feature_mode,
        sr=32000,
        dur_s=1.0,
        cache=True,
        n_mels=n_mels,
        n_mfcc=20,
        n_fft=1024,
        hop_length=320,
        win_length=800,
        fmin=50,
        fmax=fmax,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = AblationCRNN(
        num_classes=len(LABELS),
        in_channels=feature_channels(feature_mode),
        use_se=use_se,
        fusion_type="direct",
        aux_gate_init=-4.0,
    )

    raw_ckpt = torch.load(ckpt_path, map_location="cpu")
    state = extract_model_state(raw_ckpt)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("\n[LOAD ERROR]")
        print(f"ckpt_path = {ckpt_path}")
        print("checkpoint keys example:")
        for i, k in enumerate(state.keys()):
            if i >= 20:
                break
            print(" ", k)
        raise e

    model = model.to(device)
    model.eval()

    y_true = []
    probs = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)

        y_true.extend(yb.numpy().tolist())
        probs.extend(prob.detach().cpu().numpy().tolist())

    return np.array(y_true, dtype=np.int64), np.array(probs, dtype=np.float32)


def metrics_from_probs(y_true, probs):
    pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, pred)
    mf1 = f1_score(
        y_true,
        pred,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(
        y_true,
        pred,
        labels=list(range(len(LABELS))),
    )

    return acc, mf1, cm, pred


def best_alpha_on_val(y_true, probs_base, probs_pre, alphas):
    best = {
        "alpha": None,
        "val_acc": None,
        "val_macro_f1": -1.0,
    }

    for alpha in alphas:
        fused = alpha * probs_pre + (1.0 - alpha) * probs_base
        acc, mf1, _, _ = metrics_from_probs(y_true, fused)

        if mf1 > best["val_macro_f1"]:
            best = {
                "alpha": float(alpha),
                "val_acc": float(acc),
                "val_macro_f1": float(mf1),
            }

    return best


def run_one(fold: int, seed: int, alphas, device: str, batch_size: int):
    manifest_root = ROOT / "data" / "manifests_pigvocal_4class_dedup_cv10"

    val_manifest = manifest_root / f"fold{fold}" / "val.csv"
    test_manifest = manifest_root / f"fold{fold}" / "test.csv"

    ckpt_base = ROOT / "checkpoints" / f"cv10_fold{fold}_logmel_seed{seed}.pt"
    ckpt_pre = ROOT / "checkpoints" / f"cv10_pretrain_no_feeding_fold{fold}_logmel_seed{seed}.pt"

    y_val_base, p_val_base = predict_probs(
        ckpt_path=ckpt_base,
        manifest_path=val_manifest,
        device=device,
        batch_size=batch_size,
    )

    y_val_pre, p_val_pre = predict_probs(
        ckpt_path=ckpt_pre,
        manifest_path=val_manifest,
        device=device,
        batch_size=batch_size,
    )

    if not np.array_equal(y_val_base, y_val_pre):
        raise RuntimeError(f"Val labels mismatch for fold={fold}, seed={seed}")

    best = best_alpha_on_val(
        y_true=y_val_base,
        probs_base=p_val_base,
        probs_pre=p_val_pre,
        alphas=alphas,
    )

    y_test_base, p_test_base = predict_probs(
        ckpt_path=ckpt_base,
        manifest_path=test_manifest,
        device=device,
        batch_size=batch_size,
    )

    y_test_pre, p_test_pre = predict_probs(
        ckpt_path=ckpt_pre,
        manifest_path=test_manifest,
        device=device,
        batch_size=batch_size,
    )

    if not np.array_equal(y_test_base, y_test_pre):
        raise RuntimeError(f"Test labels mismatch for fold={fold}, seed={seed}")

    base_acc, base_mf1, base_cm, _ = metrics_from_probs(y_test_base, p_test_base)
    pre_acc, pre_mf1, pre_cm, _ = metrics_from_probs(y_test_base, p_test_pre)

    alpha = best["alpha"]
    p_test_fused = alpha * p_test_pre + (1.0 - alpha) * p_test_base

    fused_acc, fused_mf1, fused_cm, fused_pred = metrics_from_probs(
        y_test_base,
        p_test_fused,
    )

    return {
        "fold": fold,
        "seed": seed,
        "alpha": alpha,
        "val_alpha_acc": best["val_acc"],
        "val_alpha_macro_f1": best["val_macro_f1"],

        "base_test_acc": base_acc,
        "base_test_macro_f1": base_mf1,

        "pretrain_test_acc": pre_acc,
        "pretrain_test_macro_f1": pre_mf1,

        "fusion_test_acc": fused_acc,
        "fusion_test_macro_f1": fused_mf1,

        "delta_fusion_minus_base": fused_mf1 - base_mf1,
        "delta_fusion_minus_pretrain": fused_mf1 - pre_mf1,
        "delta_pretrain_minus_base": pre_mf1 - base_mf1,

        "base_cm": base_cm,
        "pretrain_cm": pre_cm,
        "fusion_cm": fused_cm,
    }


def bootstrap_ci(delta, seed=3407, n_boot=10000):
    rng = np.random.default_rng(seed)
    boots = []

    delta = np.asarray(delta, dtype=np.float64)

    for _ in range(n_boot):
        sample = rng.choice(delta, size=len(delta), replace=True)
        boots.append(sample.mean())

    low, high = np.percentile(boots, [2.5, 97.5])

    return float(low), float(high)


def paired_test(a, b):
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        p = np.nan
    return float(p)


def summarize_pair(df, a_col, b_col, name):
    delta = df[a_col].values - df[b_col].values
    ci_low, ci_high = bootstrap_ci(delta)
    p = paired_test(df[a_col].values, df[b_col].values)

    return {
        "comparison": name,
        "n": len(df),
        "mean_a": float(df[a_col].mean()),
        "mean_b": float(df[b_col].mean()),
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "wilcoxon_p": p,
    }


def aggregate_cm(rows, key):
    cms = [r[key] for r in rows]
    return np.stack(cms, axis=0).sum(axis=0)


def cm_metrics(cm):
    f1s = {}
    recalls = {}

    for i, lab in enumerate(LABELS):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0

        recalls[lab] = rec
        f1s[lab] = f1

    return {
        "macro_f1_from_cm": float(np.mean(list(f1s.values()))),
        "feeding_f1": float(f1s["feeding"]),
        "stress_f1": float(f1s["stress_vocal"]),
        "feeding_recall": float(recalls["feeding"]),
        "stress_recall": float(recalls["stress_vocal"]),
        "feeding_to_stress": int(cm[LABELS.index("feeding"), LABELS.index("stress_vocal")]),
        "stress_to_feeding": int(cm[LABELS.index("stress_vocal"), LABELS.index("feeding")]),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seeds", type=str, default="3407,42,2024")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument(
        "--alphas",
        type=str,
        default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0",
    )
    ap.add_argument("--batch_size", type=int, default=16)

    args = ap.parse_args()

    seeds = parse_list_int(args.seeds)
    folds = parse_list_int(args.folds)
    alphas = parse_list_float(args.alphas)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] device =", device)
    print("[INFO] seeds =", seeds)
    print("[INFO] folds =", folds)
    print("[INFO] alphas =", alphas)

    rows = []

    for seed in seeds:
        for fold in folds:
            print(f"\n[RUN] fold={fold} seed={seed}")
            r = run_one(
                fold=fold,
                seed=seed,
                alphas=alphas,
                device=device,
                batch_size=args.batch_size,
            )
            rows.append(r)

            print(
                f"alpha={r['alpha']:.2f} | "
                f"base={r['base_test_macro_f1']:.6f} | "
                f"pretrain={r['pretrain_test_macro_f1']:.6f} | "
                f"fusion={r['fusion_test_macro_f1']:.6f}"
            )

    simple_rows = []

    for r in rows:
        simple_rows.append({
            k: v for k, v in r.items()
            if not k.endswith("_cm")
        })

    df = pd.DataFrame(simple_rows)

    runs_out = ROOT / "reports" / "cv10_logmel_pretrain_latefusion_runs.csv"
    df.to_csv(runs_out, index=False, encoding="utf-8-sig")

    summary_rows = [
        {
            "model": "baseline_logmel",
            "n": len(df),
            "mean_macro_f1": df["base_test_macro_f1"].mean(),
            "std_macro_f1": df["base_test_macro_f1"].std(),
            "min_macro_f1": df["base_test_macro_f1"].min(),
            "max_macro_f1": df["base_test_macro_f1"].max(),
        },
        {
            "model": "pretrain_logmel",
            "n": len(df),
            "mean_macro_f1": df["pretrain_test_macro_f1"].mean(),
            "std_macro_f1": df["pretrain_test_macro_f1"].std(),
            "min_macro_f1": df["pretrain_test_macro_f1"].min(),
            "max_macro_f1": df["pretrain_test_macro_f1"].max(),
        },
        {
            "model": "latefusion_logmel",
            "n": len(df),
            "mean_macro_f1": df["fusion_test_macro_f1"].mean(),
            "std_macro_f1": df["fusion_test_macro_f1"].std(),
            "min_macro_f1": df["fusion_test_macro_f1"].min(),
            "max_macro_f1": df["fusion_test_macro_f1"].max(),
        },
    ]

    summary = pd.DataFrame(summary_rows)

    summary_out = ROOT / "reports" / "cv10_logmel_pretrain_latefusion_summary.csv"
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    pair_rows = [
        summarize_pair(
            df,
            "fusion_test_macro_f1",
            "base_test_macro_f1",
            "latefusion_minus_baseline",
        ),
        summarize_pair(
            df,
            "fusion_test_macro_f1",
            "pretrain_test_macro_f1",
            "latefusion_minus_pretrain",
        ),
        summarize_pair(
            df,
            "pretrain_test_macro_f1",
            "base_test_macro_f1",
            "pretrain_minus_baseline",
        ),
    ]

    paired = pd.DataFrame(pair_rows)

    paired_out = ROOT / "reports" / "cv10_logmel_pretrain_latefusion_paired_stats.csv"
    paired.to_csv(paired_out, index=False, encoding="utf-8-sig")

    cm_rows = []

    for model_name, key in [
        ("baseline_logmel", "base_cm"),
        ("pretrain_logmel", "pretrain_cm"),
        ("latefusion_logmel", "fusion_cm"),
    ]:
        cm = aggregate_cm(rows, key)
        row = {"model": model_name}
        row.update(cm_metrics(cm))
        cm_rows.append(row)

        print(f"\n[CONFUSION] {model_name}")
        print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    cm_df = pd.DataFrame(cm_rows)

    cm_out = ROOT / "reports" / "cv10_logmel_pretrain_latefusion_confusion_summary.csv"
    cm_df.to_csv(cm_out, index=False, encoding="utf-8-sig")

    print("\n[LATE FUSION SUMMARY]")
    print(summary.to_string(index=False))

    print("\n[PAIRED STATS]")
    print(paired.to_string(index=False))

    print("\n[CONFUSION SUMMARY]")
    print(cm_df.to_string(index=False))

    print(f"\n[OK] wrote -> {runs_out}")
    print(f"[OK] wrote -> {summary_out}")
    print(f"[OK] wrote -> {paired_out}")
    print(f"[OK] wrote -> {cm_out}")


if __name__ == "__main__":
    main()