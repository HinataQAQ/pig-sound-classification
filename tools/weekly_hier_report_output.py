from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def title(text: str):
    print("\n" + text)
    print("=" * 92)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[MISSING] {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt_float_cols(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def print_table(df: pd.DataFrame, cols=None, sort_col=None, ascending=False):
    if df is None or len(df) == 0:
        print("[NO DATA]")
        return

    out = df.copy()

    if cols is not None:
        exist_cols = [c for c in cols if c in out.columns]
        out = out[exist_cols]

    if sort_col and sort_col in out.columns:
        out[sort_col] = pd.to_numeric(out[sort_col], errors="coerce")
        out = out.sort_values(sort_col, ascending=ascending)

    # Keep console screenshot compact and readable.
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)

    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def build_main_plus_hier_table():
    expanded_path = REPORTS / "cv5_expanded_variant_summary.csv"
    hier_path = REPORTS / "hier_longcontext_summary.csv"

    expanded = read_csv_safe(expanded_path)
    hier = read_csv_safe(hier_path)

    rows = []

    if len(expanded) > 0:
        cap3x = expanded[expanded["protocol"].astype(str).str.lower() == "cap3x"].copy()
        keep_models = [
            "logmel_dur2",
            "logmel_dur3",
            "logmel_dur2_attn",
            "logmel",
            "mctafd_no_se_gated",
            "logmel_specaug",
            "logmel_dual",
            "logmel_bilstm",
            "logmel_sg",
            "logmel_pcen",
        ]
        cap3x = cap3x[cap3x["model"].isin(keep_models)].copy()
        for _, r in cap3x.iterrows():
            rows.append({
                "protocol": "cap3x",
                "model": r.get("model"),
                "n": r.get("n"),
                "mean_macro_f1": r.get("mean_macro_f1"),
                "std_macro_f1": r.get("std_macro_f1"),
                "min_macro_f1": r.get("min_macro_f1"),
                "max_macro_f1": r.get("max_macro_f1"),
                "mean_aux_f1": None,
            })

    if len(hier) > 0:
        for _, r in hier.iterrows():
            rows.append({
                "protocol": "cap3x",
                "model": r.get("model"),
                "n": r.get("n"),
                "mean_macro_f1": r.get("mean_macro_f1"),
                "std_macro_f1": r.get("std_macro_f1"),
                "min_macro_f1": r.get("min_macro_f1"),
                "max_macro_f1": r.get("max_macro_f1"),
                "mean_aux_f1": r.get("mean_aux_f1"),
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    for c in ["n", "mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1", "mean_aux_f1"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Avoid duplicated exact rows if script is rerun after merged files.
    df = df.drop_duplicates(subset=["protocol", "model"], keep="first")
    df = df.sort_values("mean_macro_f1", ascending=False)
    return df


def screenshot_1_main_summary():
    title("SCREENSHOT 1 — cap3x 主结果总览（已合并 2s Hier w02 / w05 / w10）")
    df = build_main_plus_hier_table()
    print_table(
        df,
        cols=[
            "protocol", "model", "n", "mean_macro_f1", "std_macro_f1",
            "min_macro_f1", "max_macro_f1", "mean_aux_f1",
        ],
        sort_col="mean_macro_f1",
        ascending=False,
    )


def screenshot_2_dur2_vs_dur1():
    title("SCREENSHOT 2 — 2s vs 1s 配对统计")
    path = REPORTS / "cv5_cap3x_logmel_dur2_vs_dur1_paired_stats.csv"
    df = read_csv_safe(path)
    print_table(
        df,
        cols=["comparison", "n", "dur1_mean", "dur2_mean", "mean_delta", "ci95_low", "ci95_high", "wilcoxon_p"],
    )


def screenshot_3_dur_boundary():
    title("SCREENSHOT 3 — 2s 长上下文对 feeding / stress_vocal 边界的改善")
    path = REPORTS / "cv5_cap3x_logmel_dur_confusion_summary.csv"
    df = read_csv_safe(path)
    print_table(
        df,
        cols=["model", "n_runs", "macro_f1_from_cm", "feeding_f1", "stress_f1", "feeding_to_stress", "stress_to_feeding"],
    )


def screenshot_4_hier_weight_ablation():
    title("SCREENSHOT 4A — 层级辅助监督权重消融 w02 / w05 / w10")
    path = REPORTS / "hier_longcontext_summary.csv"
    df = read_csv_safe(path)
    print_table(
        df,
        cols=["model", "n", "mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1", "mean_aux_f1"],
        sort_col="mean_macro_f1",
        ascending=False,
    )


def screenshot_4b_hier_paired():
    title("SCREENSHOT 4B — 层级辅助监督 paired stats")
    paths = [
        REPORTS / "cv5_cap3x_hier_w02_vs_dur2_paired_stats.csv",
        REPORTS / "cv5_cap3x_hier_w05_vs_dur2_paired_stats.csv",
        REPORTS / "cv5_cap3x_hier_w10_vs_dur2_paired_stats.csv",
    ]
    frames = []
    for p in paths:
        df = read_csv_safe(p)
        if len(df) > 0:
            frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame()
    print_table(
        out,
        cols=["comparison", "n", "dur2_mean", "hier_mean", "mean_delta", "ci95_low", "ci95_high", "wilcoxon_p"],
    )


def screenshot_4c_hier_confusion():
    title("SCREENSHOT 4C — 层级辅助监督对 feeding / stress_vocal 边界的影响")
    paths = [
        REPORTS / "hier_longcontext_confusion_summary_w02.csv",
        REPORTS / "hier_longcontext_confusion_summary_w05.csv",
        REPORTS / "hier_longcontext_confusion_summary_w10.csv",
    ]
    frames = []
    for p in paths:
        df = read_csv_safe(p)
        if len(df) > 0:
            frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["model"], keep="last")
    else:
        out = pd.DataFrame()
    print_table(
        out,
        cols=["model", "n_runs", "macro_f1_from_cm", "feeding_f1", "stress_f1", "feeding_to_stress", "stress_to_feeding"],
    )


def screenshot_5_duration_only():
    title("SCREENSHOT 5A — duration-only baseline：证明 2s 不是仅靠时长作弊")
    path = REPORTS / "duration_only_baseline_summary.csv"
    df = read_csv_safe(path)
    print_table(
        df,
        cols=["model", "n", "mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1"],
        sort_col="mean_macro_f1",
        ascending=False,
    )


def screenshot_5b_conclusion():
    title("SCREENSHOT 5B — 汇报结论文字")
    lines = [
        "1. clean 主线已从 MCTAFD/Gate-MCTAFD 转为 Duration-aware 2s Long-context Log-Mel-CRNN。",
        "2. 2s 相比 1s 的主提升显著：Macro-F1 0.9226 -> 0.9472，paired p=0.000162。",
        "3. duration-only baseline 仅 0.63~0.74，说明 2s 不是单纯靠音频时长分类。",
        "4. 层级辅助监督进一步提升至约 0.951，主要继续改善 feeding / stress_vocal 边界。",
        "5. w10 均值最高，w05 更稳；论文写法应强调 2s 是主显著增益，hier 是小幅边界增强。",
        "6. SpecAugment / PCEN / 谱门控降噪在 clean 数据下效果不佳，后续等真实猪场噪声数据到位后再评估鲁棒性。",
    ]
    for line in lines:
        print(line)


def main():
    screenshot_1_main_summary()
    screenshot_2_dur2_vs_dur1()
    screenshot_3_dur_boundary()
    screenshot_4_hier_weight_ablation()
    screenshot_4b_hier_paired()
    screenshot_4c_hier_confusion()
    screenshot_5_duration_only()
    screenshot_5b_conclusion()


if __name__ == "__main__":
    main()
