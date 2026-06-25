from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
NOISE_DIR = ROOT / "reports" / "prototype_noise_demand_w05_final"


def ensure_dirs() -> None:
    for name in ["manuscript", "tables", "figures", "appendix", "references", "reproducibility"]:
        (PAPER / name).mkdir(parents=True, exist_ok=True)


def read_csv(path: str | Path) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"Required source table not found: {p}")
    return pd.read_csv(p)


def fmt(x: Any, digits: int = 4) -> Any:
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), digits)
    try:
        return round(float(x), digits)
    except (TypeError, ValueError):
        return x


def write_table(name: str, frame: pd.DataFrame) -> Path:
    path = PAPER / "tables" / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(PAPER / "figures" / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figures" / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def table1() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "Clean classes", "value": "cough; calm_grunt; feeding; stress_vocal"},
            {"item": "Auxiliary subtypes", "value": "dry_cough; abdominal_cough; calm_grunt; feeding; frightened_stress; anxious_stress"},
            {"item": "Clean backbone protocol", "value": "cap3x expansion; 2-second Log-Mel; CRNN; 5 folds x 5 seeds"},
            {"item": "Prototype protocol", "value": "train-only prototypes; validation-only calibration; frozen test evaluation; no cross-fold or cross-seed mixing"},
            {"item": "Noise protocol", "value": "DEMAND DWASHING/TBUS/STRAFFIC; active-event SNR 20/10/0 dB; global_noise_seed=3407; noise_repeat=0"},
            {"item": "Leakage controls", "value": "path/source_id/MD5 disjointness; manifest SHA checks; checkpoint/prototype/calibration SHA provenance"},
            {"item": "External validation", "value": "Not performed; all noise results are simulated-noise robustness only"},
        ]
    )


def clean_ablation_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variants = read_csv("paper_results/tables/cv5_expanded_variant_summary.csv")
    hier = read_csv("paper_results/tables/hier_longcontext_summary.csv")
    table2_models = [
        "mctafd_no_se_gated",
        "logmel",
        "logmel_dur2",
        "logmel_dur2_attn",
        "logmel_specaug",
        "logmel_pcen",
        "logmel_sg",
    ]
    t2 = variants[variants["model"].isin(table2_models)].copy()
    t2 = t2[["protocol", "model", "n", "mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1"]]
    for col in ["mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1"]:
        t2[col] = t2[col].map(fmt)
    duration_map = {"logmel": "1 s", "logmel_dur2": "2 s", "logmel_dur3": "3 s"}
    t3 = variants[(variants["protocol"] == "cap3x") & (variants["model"].isin(duration_map))].copy()
    t3["duration"] = t3["model"].map(duration_map)
    t3 = t3[["duration", "model", "n", "mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1"]]
    for col in ["mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1"]:
        t3[col] = t3[col].map(fmt)
    paired = read_csv("paper_results/tables/cv5_cap3x_logmel_dur2_vs_dur1_paired_stats.csv")
    t3["paired_note"] = ""
    if not paired.empty:
        row = paired.iloc[0]
        t3.loc[t3["duration"] == "2 s", "paired_note"] = (
            f"2 s - 1 s mean delta={fmt(row.get('mean_delta'))}, "
            f"Wilcoxon p={fmt(row.get('wilcoxon_p'), 6)}"
        )
    t4 = hier[["model", "n", "mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1", "mean_aux_f1"]].copy()
    for col in ["mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1", "mean_aux_f1"]:
        t4[col] = t4[col].map(fmt)
    return t2, t3, t4


def clean_prototype_table() -> pd.DataFrame:
    frame = read_csv("reports/prototype_cv5_exact_w05_final_summary.csv")
    cols = ["method", "n", "mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1", "mean_top1_acc", "mean_top2_acc", "mean_ece", "mean_brier", "mean_nll"]
    out = frame[cols].copy()
    for col in out.columns:
        if col not in {"method", "n"}:
            out[col] = out[col].map(fmt)
    return out


def noise_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = read_csv(NOISE_DIR / "final_summary.csv")
    paired = read_csv(NOISE_DIR / "final_paired_stats.csv")
    clean_paired = read_csv("reports/prototype_cv5_exact_w05_final_paired_stats.csv")
    per_class = read_csv(NOISE_DIR / "final_per_class_summary.csv")
    confusion = read_csv(NOISE_DIR / "final_confusion_summary.csv")
    selective = read_csv(NOISE_DIR / "final_selective_summary.csv")
    aurc_augrc = read_csv(NOISE_DIR / "final_aurc_augrc_summary.csv")

    t6 = summary[summary["noise_environment"].isin(["MODERATE_NOISE", "EXTREME_STRESS", "ALL_NOISY"])].copy()
    t6 = t6[["noise_environment", "condition_family", "method", "n", "mean_macro_f1", "std_macro_f1", "mean_top1_acc", "mean_top2_acc", "mean_ece", "mean_brier", "mean_nll"]]
    for col in t6.columns:
        if col not in {"noise_environment", "condition_family", "method", "n"}:
            t6[col] = t6[col].map(fmt)

    clean_paired = clean_paired.copy()
    clean_paired["experiment"] = "clean_prototype"
    paired = paired[paired["noise_environment"].isin(["MODERATE_NOISE", "EXTREME_STRESS", "ALL_NOISY"])].copy()
    paired["experiment"] = "simulated_noise"
    t7 = pd.concat([clean_paired, paired], ignore_index=True, sort=False)
    keep7 = ["experiment", "noise_environment", "snr_db", "comparison", "n", "mean_delta", "std_delta", "bootstrap_ci95_low", "bootstrap_ci95_high", "wilcoxon_p", "wins", "ties", "losses"]
    t7 = t7[[c for c in keep7 if c in t7.columns]]
    for col in ["mean_delta", "std_delta", "bootstrap_ci95_low", "bootstrap_ci95_high", "wilcoxon_p"]:
        if col in t7.columns:
            t7[col] = t7[col].map(lambda x: fmt(x, 6))

    all_noisy_pc = per_class[(per_class["noise_environment"] == "ALL_NOISY") & (per_class["snr_db"].astype(str) == "GROUP")].copy()
    error_rows = []
    for _, row in confusion[(confusion["noise_environment"] == "ALL_NOISY") & (confusion["snr_db"].astype(str) == "GROUP")].iterrows():
        labels = str(row["labels"]).split("|")
        cm = np.array(ast.literal_eval(str(row["confusion_matrix"])), dtype=int)
        idx = {label: i for i, label in enumerate(labels)}
        error_rows.append(
            {
                "method": row["method"],
                "feeding_to_stress": int(cm[idx["feeding"], idx["stress_vocal"]]),
                "stress_to_feeding": int(cm[idx["stress_vocal"], idx["feeding"]]),
                "cough_to_calm_grunt": int(cm[idx["cough"], idx["calm_grunt"]]),
                "cough_to_feeding": int(cm[idx["cough"], idx["feeding"]]),
                "cough_to_stress_vocal": int(cm[idx["cough"], idx["stress_vocal"]]),
            }
        )
    t8 = all_noisy_pc.merge(pd.DataFrame(error_rows), on="method", how="left")
    for col in ["precision", "recall", "f1"]:
        t8[col] = t8[col].map(fmt)

    t9 = selective.merge(
        aurc_augrc,
        on=["noise_environment", "snr_db", "condition_family", "method", "n"],
        how="left",
    )
    t9 = t9[t9["noise_environment"].isin(["MODERATE_NOISE", "EXTREME_STRESS", "ALL_NOISY"])].copy()
    for col in t9.columns:
        if col not in {"noise_environment", "snr_db", "condition_family", "method", "n"}:
            t9[col] = t9[col].map(fmt)
    return t6, t7, t8, t9


def make_figures() -> None:
    variants = read_csv("paper_results/tables/cv5_expanded_variant_summary.csv")
    clean_proto = read_csv("reports/prototype_cv5_exact_w05_final_summary.csv")
    noise_summary = read_csv(NOISE_DIR / "final_summary.csv")
    per_class = read_csv(NOISE_DIR / "final_per_class_summary.csv")
    confusion = read_csv(NOISE_DIR / "final_confusion_summary.csv")
    runs = read_csv(NOISE_DIR / "final_runs.csv")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    boxes = [
        (0.02, 0.55, "Leakage-free\ntrain / val / test"),
        (0.22, 0.55, "2-s Log-Mel\nCRNN backbone"),
        (0.42, 0.55, "Hierarchical\nsubtype supervision"),
        (0.62, 0.55, "Main + subtype\nprototypes"),
        (0.82, 0.55, "Clean and DEMAND\nfrozen evaluation"),
        (0.32, 0.12, "Validation-only calibration\nthresholds / temperatures / fusion"),
        (0.66, 0.12, "Selective prediction\nAURC / corrected AUGRC"),
    ]
    for x, y, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.15, 0.2, boxstyle="round,pad=0.02", linewidth=1.2, facecolor="#eef4fb", edgecolor="#345"))
        ax.text(x + 0.075, y + 0.1, text, ha="center", va="center", fontsize=9)
    for x in [0.17, 0.37, 0.57, 0.77]:
        ax.annotate("", xy=(x + 0.04, 0.65), xytext=(x, 0.65), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.annotate("", xy=(0.45, 0.35), xytext=(0.48, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.annotate("", xy=(0.73, 0.35), xytext=(0.7, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.set_title("Overall Method Architecture")
    save_figure(fig, "figure1_overall_method_architecture")

    duration = variants[(variants["protocol"] == "cap3x") & (variants["model"].isin(["logmel", "logmel_dur2", "logmel_dur3"]))].copy()
    duration["duration"] = duration["model"].map({"logmel": "1 s", "logmel_dur2": "2 s", "logmel_dur3": "3 s"})
    duration = duration.sort_values("duration")
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar(duration["duration"], duration["mean_macro_f1"], yerr=duration["std_macro_f1"], color=["#7aa6c2", "#3b7ea1", "#9ebc7b"], capsize=4)
    ax.set_ylim(0.88, 0.97)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Duration Comparison")
    save_figure(fig, "figure2_duration_macro_f1")

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.axis("off")
    main_labels = ["cough", "calm_grunt", "feeding", "stress_vocal"]
    aux_map = {
        "cough": ["dry_cough", "abdominal_cough"],
        "calm_grunt": ["calm_grunt"],
        "feeding": ["feeding"],
        "stress_vocal": ["frightened_stress", "anxious_stress"],
    }
    xs = np.linspace(0.12, 0.88, len(main_labels))
    for x, main in zip(xs, main_labels):
        ax.add_patch(FancyBboxPatch((x - 0.08, 0.65), 0.16, 0.13, boxstyle="round,pad=0.02", facecolor="#edf7ed", edgecolor="#385"))
        ax.text(x, 0.715, main, ha="center", va="center", fontsize=9)
        ys = np.linspace(0.25, 0.47, len(aux_map[main]))
        for y, aux in zip(ys, aux_map[main]):
            ax.add_patch(FancyBboxPatch((x - 0.08, y), 0.16, 0.09, boxstyle="round,pad=0.015", facecolor="#fff7e8", edgecolor="#a75"))
            ax.text(x, y + 0.045, aux, ha="center", va="center", fontsize=8)
            ax.annotate("", xy=(x, y + 0.1), xytext=(x, 0.65), arrowprops={"arrowstyle": "-", "lw": 1.0})
    ax.set_title("Main and Subtype Prototype Structure")
    save_figure(fig, "figure3_prototype_structure")

    snr_rows = noise_summary[(noise_summary["noise_environment"].isin(["DWASHING", "TBUS", "STRAFFIC"]))].copy()
    snr_rows["snr_numeric"] = pd.to_numeric(snr_rows["snr_db"], errors="coerce")
    snr_mean = snr_rows.groupby(["snr_numeric", "method"], as_index=False)["mean_macro_f1"].mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method in ["raw_softmax", "prototype", "hierarchical", "fused"]:
        g = snr_mean[snr_mean["method"] == method].sort_values("snr_numeric")
        ax.plot(g["snr_numeric"], g["mean_macro_f1"], marker="o", label=method)
    ax.set_xlabel("Active-event SNR (dB)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 versus Active-event SNR")
    ax.legend(fontsize=8)
    save_figure(fig, "figure4_macro_f1_vs_snr")

    env_mean = snr_rows.groupby(["noise_environment", "method"], as_index=False)["mean_macro_f1"].mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    pivot = env_mean.pivot(index="noise_environment", columns="method", values="mean_macro_f1")
    pivot[["raw_softmax", "prototype", "hierarchical", "fused"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Noise Degradation by DEMAND Environment")
    ax.legend(fontsize=8)
    save_figure(fig, "figure5_noise_degradation_by_environment")

    err_rows = []
    for _, row in confusion[(confusion["noise_environment"] == "ALL_NOISY") & (confusion["snr_db"].astype(str) == "GROUP")].iterrows():
        labels = str(row["labels"]).split("|")
        cm = np.array(ast.literal_eval(str(row["confusion_matrix"])), dtype=int)
        idx = {label: i for i, label in enumerate(labels)}
        err_rows.append({"method": row["method"], "feeding_to_stress": cm[idx["feeding"], idx["stress_vocal"]], "stress_to_feeding": cm[idx["stress_vocal"], idx["feeding"]]})
    err = pd.DataFrame(err_rows).set_index("method")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    err[["feeding_to_stress", "stress_to_feeding"]].plot(kind="bar", ax=ax, color=["#c86f60", "#5f8db8"])
    ax.set_ylabel("Error count")
    ax.set_title("Feeding/Stress Confusion under ALL_NOISY")
    save_figure(fig, "figure6_feeding_stress_confusion")

    cough = per_class[(per_class["class"] == "cough") & (per_class["noise_environment"].isin(["DWASHING", "TBUS", "STRAFFIC"]))].copy()
    cough["snr_numeric"] = pd.to_numeric(cough["snr_db"], errors="coerce")
    cough_mean = cough.groupby(["snr_numeric", "method"], as_index=False)["recall"].mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method in ["raw_softmax", "prototype", "hierarchical", "fused"]:
        g = cough_mean[cough_mean["method"] == method].sort_values("snr_numeric")
        ax.plot(g["snr_numeric"], g["recall"], marker="o", label=method)
    ax.set_xlabel("Active-event SNR (dB)")
    ax.set_ylabel("Cough recall")
    ax.set_title("Cough Collapse under Active-event Noise")
    ax.legend(fontsize=8)
    save_figure(fig, "figure7_cough_collapse")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    pred_paths = [Path(str(p)) for p in runs["metrics_json"]]
    method_losses: dict[str, list[np.ndarray]] = {m: [] for m in ["raw_softmax", "prototype", "hierarchical", "fused"]}
    method_conf: dict[str, list[np.ndarray]] = {m: [] for m in ["raw_softmax", "prototype", "hierarchical", "fused"]}
    for metrics_path in pred_paths:
        payload = json.loads((ROOT / metrics_path).read_text(encoding="utf-8"))
        pred = pd.read_csv(ROOT / payload["outputs"]["noise_predictions_csv"])
        pred = pred[pred["noise_environment"].astype(str).str.lower() != "clean"]
        y_true = pred["y_true_id"].to_numpy(dtype=np.int64)
        for method in method_losses:
            y_pred = pred[f"{method}_pred_id"].to_numpy(dtype=np.int64)
            method_losses[method].append((y_true != y_pred).astype(float))
            method_conf[method].append(pred[f"{method}_confidence"].to_numpy(dtype=float))
    for method in method_losses:
        losses = np.concatenate(method_losses[method])
        conf = np.concatenate(method_conf[method])
        order = np.argsort(-conf)
        coverage = np.arange(1, len(losses) + 1, dtype=float) / float(len(losses))
        risk = np.cumsum(losses[order]) / np.arange(1, len(losses) + 1, dtype=float)
        ax.plot(coverage, risk, label=method)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("Risk-Coverage Curves under ALL_NOISY")
    ax.legend(fontsize=8)
    save_figure(fig, "figure8_risk_coverage_curves")


def write_text_outputs(tables: dict[str, pd.DataFrame]) -> None:
    noise_summary = read_csv(NOISE_DIR / "final_summary.csv")
    noise_paired = read_csv(NOISE_DIR / "final_paired_stats.csv")
    clean_summary = read_csv("reports/prototype_cv5_exact_w05_final_summary.csv")
    all_noisy = noise_summary[(noise_summary["noise_environment"] == "ALL_NOISY") & (noise_summary["snr_db"].astype(str) == "GROUP")]
    raw = all_noisy[all_noisy["method"] == "raw_softmax"].iloc[0]
    proto = all_noisy[all_noisy["method"] == "prototype"].iloc[0]
    hier = all_noisy[all_noisy["method"] == "hierarchical"].iloc[0]
    proto_delta = noise_paired[(noise_paired["noise_environment"] == "ALL_NOISY") & (noise_paired["comparison"] == "prototype - raw_softmax")].iloc[0]
    hier_delta = noise_paired[(noise_paired["noise_environment"] == "ALL_NOISY") & (noise_paired["comparison"] == "hierarchical - raw_softmax")].iloc[0]
    clean_hier = clean_summary[clean_summary["method"] == "hierarchical"].iloc[0]

    cn_abstract = (
        "摘要：面向猪场环境中的猪声识别，本文在已验证的 class-capped cap3x、2 秒 Log-Mel CRNN 主线基础上，"
        "研究时长感知层级表征、后验声学原型推理以及模拟环境噪声下的鲁棒性。干净语音闭集实验表明，2 秒上下文相对 1 秒输入带来显著的 Macro-F1 提升；"
        "层级辅助子类监督引入了细粒度声学语义并改善 feeding 与 stress_vocal 边界，但其相对 2 秒基线的增益未达到统计显著。"
        f"在 lambda=0.5 的 25 个 fold-seed clean 原型实验中，层级原型 Macro-F1 为 {fmt(clean_hier['mean_macro_f1'])}。"
        f"在 DEMAND DWASHING、TBUS 与 STRAFFIC 的零样本冻结模拟噪声测试中，main prototype 在 ALL_NOISY 条件下 Macro-F1 为 {fmt(proto['mean_macro_f1'])}，"
        f"较 raw Softmax 的 {fmt(raw['mean_macro_f1'])} 平均提升 {fmt(proto_delta['mean_delta'], 6)}，配对 Wilcoxon p={fmt(proto_delta['wilcoxon_p'], 6)}。"
        "结果说明，未经噪声重训练的主类原型推理可提升模拟环境噪声下的整体鲁棒性，但本文未进行真实猪场外部验证，也不声称 0 dB 咳嗽识别已经可靠。"
    )
    en_abstract = (
        "Abstract: This study investigates pig vocalization recognition with duration-aware hierarchical representations, "
        "post-hoc acoustic prototypes, and simulated environmental-noise robustness on top of a validated class-capped cap3x, "
        "2-s Log-Mel CRNN pipeline. Clean closed-set experiments show that 2-s temporal context significantly improves Macro-F1 over 1-s input. "
        "Hierarchical subtype supervision adds fine-grained acoustic semantics and improves the feeding/stress_vocal boundary, "
        "but its incremental gain over the 2-s baseline is not statistically significant. "
        f"In the lambda=0.5 25-run clean prototype evaluation, hierarchical prototypes reach a Macro-F1 of {fmt(clean_hier['mean_macro_f1'])}. "
        f"Under zero-shot frozen DEMAND noise from DWASHING, TBUS, and STRAFFIC, the main prototype achieves an ALL_NOISY Macro-F1 of {fmt(proto['mean_macro_f1'])}, "
        f"improving over raw Softmax ({fmt(raw['mean_macro_f1'])}) by {fmt(proto_delta['mean_delta'], 6)} on average (paired Wilcoxon p={fmt(proto_delta['wilcoxon_p'], 6)}). "
        "The results support main-class prototype inference as a simulated-noise robustness improvement without noisy retraining, "
        "while real-farm external validation and reliable 0 dB cough recognition remain open limitations."
    )

    cn_md = f"""# 面向模拟环境噪声鲁棒猪声识别的时长感知层级表征与双原型推理方法

{cn_abstract}

## 关键词

猪声识别；Log-Mel；CRNN；层级监督；原型推理；模拟环境噪声；选择性分类

## 1 引言

猪声自动识别可为规模化养殖中的健康和行为监测提供工程化感知入口。本文关注四类主声学事件：cough、calm_grunt、feeding 和 stress_vocal。与真实猪场部署相比，当前研究只覆盖干净数据和 DEMAND 环境噪声模拟，因此结论严格限定为闭集分类与模拟噪声鲁棒性。

## 2 方法

主模型采用 2 秒 Log-Mel 输入和 CRNN backbone，并在主类分类之外加入六个辅助子类监督。后验原型模块冻结 Softmax 主模型，用每个 fold-seed 的 train split 构建主类和辅助子类 embedding 原型，用 validation split 校准温度、阈值和融合权重，并在 test split 上冻结评价。

## 3 实验协议

所有实验保持 path、source_id 和 MD5 去重边界。干净模型比较采用匹配 fold-seed 配对统计。模拟噪声实验使用 DEMAND 的 DWASHING、TBUS 和 STRAFFIC，活动区 SNR 为 20、10 和 0 dB。噪声绘制、SNR、threshold 和统计协议在 final 25-run 前冻结，测试集不参与任何调参。

## 4 结果

2 秒上下文是干净数据上的主要显著增益来源。层级辅助监督的平均 Macro-F1 高于 2 秒基线，但相对增益不显著，因此只作为细粒度语义与边界改善证据。clean 原型实验显示 hierarchical prototype 的 Macro-F1 为 {fmt(clean_hier['mean_macro_f1'])}。

在模拟噪声 ALL_NOISY 聚合条件下，raw Softmax Macro-F1 为 {fmt(raw['mean_macro_f1'])}，main prototype 为 {fmt(proto['mean_macro_f1'])}，hierarchical prototype 为 {fmt(hier['mean_macro_f1'])}。main prototype 相对 raw Softmax 的平均增益为 {fmt(proto_delta['mean_delta'], 6)}，95% bootstrap CI 为 [{fmt(proto_delta['bootstrap_ci95_low'], 6)}, {fmt(proto_delta['bootstrap_ci95_high'], 6)}]，Wilcoxon p={fmt(proto_delta['wilcoxon_p'], 6)}。hierarchical prototype 相对 raw Softmax 的平均增益为 {fmt(hier_delta['mean_delta'], 6)}，未达到统计显著。

## 5 讨论与局限

本文不能声称层级原型是最鲁棒方法，也不能声称完成真实猪场外部验证。0 dB 活动区 SNR 下 cough recall 明显退化，说明强噪声下咳嗽识别仍是主要风险。AUGRC 已按 fd-shifts generalized risk 定义修正，使用已保存 prediction CSV 重新计算。

## 6 结论

结果支持三点：2 秒上下文显著改善干净分类；层级辅助监督提供细粒度语义但增益不显著；主类原型推理在模拟 DEMAND 噪声下显著提升 aggregate Macro-F1，且不需要噪声重训练。
"""
    en_md = f"""# Duration-Aware Hierarchical Representation and Dual-Prototype Inference for Pig Vocalization Recognition under Simulated Environmental Noise

{en_abstract}

## Keywords

Pig vocalization recognition; Log-Mel; CRNN; hierarchical supervision; prototype inference; simulated environmental noise; selective classification

## 1. Introduction

Automatic pig vocalization recognition can support engineering-oriented monitoring in livestock environments. This work studies four closed-set main classes: cough, calm_grunt, feeding, and stress_vocal. The evidence is limited to clean evaluation and DEMAND-based simulated noise; real-farm external validation has not been performed.

## 2. Method

The clean mainline uses 2-s Log-Mel inputs and a CRNN backbone. Hierarchical auxiliary supervision predicts six acoustic subtypes in addition to the main class. The post-hoc prototype module keeps the Softmax model frozen, builds fold-seed-specific main and subtype embedding prototypes from the train split only, calibrates temperatures and thresholds on validation only, and evaluates on the frozen test split.

## 3. Experimental Protocol

All runs preserve path, source_id, and MD5 disjointness. Clean model comparisons use matched fold-seed paired statistics. The simulated-noise protocol uses DEMAND DWASHING, TBUS, and STRAFFIC at active-event SNRs of 20, 10, and 0 dB. Noise draws, SNRs, thresholds, and statistical procedures are frozen before the final 25-run evaluation.

## 4. Results

The 2-s context is the dominant significant improvement on clean data. Hierarchical subtype supervision improves the mean and the feeding/stress_vocal boundary, but its incremental gain over the 2-s baseline is not statistically significant. The clean hierarchical prototype Macro-F1 is {fmt(clean_hier['mean_macro_f1'])}.

Under ALL_NOISY simulated noise, raw Softmax reaches a Macro-F1 of {fmt(raw['mean_macro_f1'])}, main prototype reaches {fmt(proto['mean_macro_f1'])}, and hierarchical prototype reaches {fmt(hier['mean_macro_f1'])}. Main prototype improves over raw Softmax by {fmt(proto_delta['mean_delta'], 6)} on average, with a 95% bootstrap CI of [{fmt(proto_delta['bootstrap_ci95_low'], 6)}, {fmt(proto_delta['bootstrap_ci95_high'], 6)}] and Wilcoxon p={fmt(proto_delta['wilcoxon_p'], 6)}. The hierarchical prototype improvement over raw Softmax is {fmt(hier_delta['mean_delta'], 6)} and is not statistically significant.

## 5. Discussion and Limitations

The study does not claim that hierarchical prototypes are the most noise robust, does not report real-farm external validation, and does not claim reliable cough recognition at 0 dB. Strong active-event noise collapses cough recall, making cough robustness the main unresolved deployment risk. AUGRC has been corrected according to the fd-shifts generalized-risk definition and recomputed from stored prediction CSV files.

## 6. Conclusion

The evidence supports three fixed claims: 2-s context significantly improves clean classification; hierarchical subtype supervision adds fine-grained semantics but does not provide a statistically significant incremental gain over the 2-s baseline; and post-hoc main-class prototype inference significantly improves aggregate Macro-F1 under simulated DEMAND noise without noisy retraining.
"""
    claims = f"""# Claims and Evidence

## Supported

1. 2-second context significantly improves clean pig-vocal classification.
   Evidence: `paper/tables/table3_duration_comparison.csv` and paired duration statistics.

2. Hierarchical subtype supervision introduces fine-grained acoustic semantics and improves the clean decision boundary, but its incremental gain over the 2-second baseline is not statistically significant.
   Evidence: `paper/tables/table4_hierarchical_aux_weight_ablation.csv` and paired hierarchy-vs-duration statistics.

3. Post-hoc main-class prototype inference significantly improves aggregate Macro-F1 under simulated DEMAND noise without noisy retraining.
   Evidence: ALL_NOISY prototype - raw_softmax mean delta {fmt(proto_delta['mean_delta'], 6)}, Wilcoxon p={fmt(proto_delta['wilcoxon_p'], 6)}.

## Not Claimed

- Hierarchical prototype is the most noise robust.
- Real-farm external validation has been completed.
- Universal uncertainty improvement.
- First invention of prototypical networks.
- Successful cough recognition at 0 dB.

## AUGRC Correction

AURC uses `sum(loss_accepted) / accepted_count`; corrected AUGRC uses fd-shifts generalized risk `sum(loss_accepted) / total_count`, integrated over coverage. The corrected final selective summaries were recomputed from existing prediction CSV files, not from new model inference.
"""
    refs = """@misc{demand2018,
  title = {DEMAND: a collection of multi-channel recordings of acoustic noise in diverse environments},
  author = {Thiemann, Joachim and Ito, Nobutaka and Vincent, Emmanuel},
  year = {2018},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.1227121},
  url = {https://zenodo.org/records/1227121}
}

@inproceedings{snell2017prototypical,
  title = {Prototypical Networks for Few-shot Learning},
  author = {Snell, Jake and Swersky, Kevin and Zemel, Richard S.},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2017},
  url = {https://arxiv.org/abs/1703.05175}
}

@misc{fdshifts_riskcoverage,
  title = {fd-shifts RiskCoverageStats implementation},
  author = {{IML-DKFZ}},
  year = {2024},
  url = {https://github.com/IML-DKFZ/fd-shifts/blob/main/fd_shifts/analysis/rc_stats.py},
  note = {Accessed 2026-06-25}
}

@inproceedings{choi2017crnn,
  title = {Convolutional Recurrent Neural Networks for Music Classification},
  author = {Choi, Keunwoo and Fazekas, Gyorgy and Sandler, Mark and Cho, Kyunghyun},
  booktitle = {Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing},
  year = {2017},
  url = {https://arxiv.org/abs/1609.04243}
}

@inproceedings{geifman2017selective,
  title = {Selective Classification for Deep Neural Networks},
  author = {Geifman, Yonatan and El-Yaniv, Ran},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2017},
  url = {https://arxiv.org/abs/1705.08500}
}

@inproceedings{mcfee2015librosa,
  title = {librosa: Audio and Music Signal Analysis in Python},
  author = {McFee, Brian and Raffel, Colin and Liang, Dawen and Ellis, Daniel P. W. and McVicar, Matt and Battenberg, Eric and Nieto, Oriol},
  booktitle = {Proceedings of the Python in Science Conference},
  year = {2015},
  url = {https://conference.scipy.org/proceedings/scipy2015/brian_mcfee.html}
}
"""
    (PAPER / "manuscript" / "paper_cn_draft.md").write_text(cn_md, encoding="utf-8")
    (PAPER / "manuscript" / "paper_en_draft.md").write_text(en_md, encoding="utf-8")
    (PAPER / "CLAIMS_AND_EVIDENCE.md").write_text(claims, encoding="utf-8")
    (PAPER / "references" / "references.bib").write_text(refs, encoding="utf-8")

    commands = """# Reproducibility Commands

These commands regenerate the corrected selective summaries and the paper package from existing outputs only.

```powershell
$env:PYTHONNOUSERSITE="1"
$metrics = @()
foreach ($fold in 0..4) {
  foreach ($seed in @(42,123,777,2024,3407)) {
    $stage = if ($seed -in @(123,777)) { "final" } else { "screening" }
    $metrics += "reports/prototype_noise_demand_w05_fold${fold}_seed${seed}_${stage}/noise_metrics.json"
  }
}
python tools\\summarize_prototype_noise_robustness.py --metrics_json $metrics --protocol final --out_dir reports\\prototype_noise_demand_w05_final --file_prefix final --recompute_selective_from_predictions
python tools\\create_paper_package.py
```

No model inference, retraining, new SNRs, new noise draws, or threshold changes are performed by these commands.
"""
    (PAPER / "reproducibility" / "commands.md").write_text(commands, encoding="utf-8")


def write_appendix() -> None:
    files = {
        "clean_25_run_prototype_summary.csv": read_csv("reports/prototype_cv5_exact_w05_final_summary.csv"),
        "clean_25_run_prototype_paired_stats.csv": read_csv("reports/prototype_cv5_exact_w05_final_paired_stats.csv"),
        "noise_25_run_summary.csv": read_csv(NOISE_DIR / "final_summary.csv"),
        "noise_25_run_paired_stats.csv": read_csv(NOISE_DIR / "final_paired_stats.csv"),
        "noise_25_run_per_class_summary.csv": read_csv(NOISE_DIR / "final_per_class_summary.csv"),
        "noise_25_run_selective_summary.csv": read_csv(NOISE_DIR / "final_selective_summary.csv"),
        "noise_25_run_aurc_augrc_summary.csv": read_csv(NOISE_DIR / "final_aurc_augrc_summary.csv"),
        "noise_25_run_fold_level_deltas.csv": read_csv(NOISE_DIR / "final_per_fold_deltas.csv"),
        "noise_25_run_cluster_bootstrap.csv": read_csv(NOISE_DIR / "final_cluster_bootstrap.csv"),
        "noise_25_run_cross_seed_draw_audit.csv": read_csv(NOISE_DIR / "final_cross_seed_draw_audit.csv"),
        "clean_calibration_distribution.csv": read_csv("reports/prototype_cv5_exact_w05_final_calibration_distribution.csv"),
    }
    for name, frame in files.items():
        frame.to_csv(PAPER / "appendix" / name, index=False, encoding="utf-8-sig")
    provenance = json.loads((NOISE_DIR / "final_provenance.json").read_text(encoding="utf-8"))
    (PAPER / "appendix" / "noise_25_run_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    limitations = """# Limitations

- Simulated DEMAND noise is not real-farm external validation.
- The protocol is closed-set; unknown/open-set recognition is not established.
- Cough recall collapses under 0 dB active-event simulated noise.
- Hierarchical auxiliary supervision improves mean clean performance but is not statistically significant over the 2-second baseline.
- AUGRC is corrected and reported from saved prediction CSV files; prior equal AURC/AUGRC summaries should not be used.
"""
    (PAPER / "appendix" / "limitations.md").write_text(limitations, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    t2, t3, t4 = clean_ablation_tables()
    t6, t7, t8, t9 = noise_tables()
    tables = {
        "table1_dataset_and_leakage_free_protocol.csv": table1(),
        "table2_clean_baseline_and_architecture_ablations.csv": t2,
        "table3_duration_comparison.csv": t3,
        "table4_hierarchical_aux_weight_ablation.csv": t4,
        "table5_clean_softmax_main_prototype_hierarchical_prototype.csv": clean_prototype_table(),
        "table6_simulated_noise_robustness_by_stratum.csv": t6,
        "table7_paired_statistics.csv": t7,
        "table8_per_class_noise_results_and_confusion_directions.csv": t8,
        "table9_selective_prediction_metrics_corrected_augrc.csv": t9,
    }
    for name, frame in tables.items():
        write_table(name, frame)
    make_figures()
    write_appendix()
    write_text_outputs(tables)
    manifest = {
        "paper_title_cn": "面向模拟环境噪声鲁棒猪声识别的时长感知层级表征与双原型推理方法",
        "paper_title_en": "Duration-Aware Hierarchical Representation and Dual-Prototype Inference for Pig Vocalization Recognition under Simulated Environmental Noise",
        "tables": sorted(tables),
        "figures": sorted(p.name for p in (PAPER / "figures").glob("figure*.svg")),
        "reference_count": 6,
        "real_farm_external_validation": False,
        "simulated_noise_only": True,
        "augrc_corrected": True,
    }
    (PAPER / "reproducibility" / "paper_package_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
