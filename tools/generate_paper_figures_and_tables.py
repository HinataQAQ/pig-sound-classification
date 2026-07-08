from __future__ import annotations

import argparse
import csv
import html
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from prototype_model_adapter import (
    DEFAULT_AUX_LABELS,
    DEFAULT_AUX_TO_MAIN,
    DEFAULT_MAIN_LABELS,
)


ROOT = Path(__file__).resolve().parents[1]

METHOD_ORDER = ["raw_softmax", "prototype", "hierarchical"]
METHOD_LABELS = {
    "raw_softmax": "raw Softmax",
    "prototype": "main-class prototype",
    "hierarchical": "hierarchical prototype",
    "fused": "fused",
}
METHOD_COLORS = {
    "raw_softmax": "#4C78A8",
    "prototype": "#59A14F",
    "hierarchical": "#F28E2B",
    "fused": "#B07AA1",
}

DEMAND_ENVS = ["DWASHING", "TBUS", "STRAFFIC"]
SNR_ORDER = [20.0, 10.0, 0.0]

NOISE_NOTE = (
    "DEMAND simulated additive noise; 0 dB active-event SNR / "
    "extreme simulated-noise stress condition."
)

TABLES: list[tuple[str, str, str]] = [
    (
        "Table 1",
        "Dataset and leakage-free protocol",
        "table1_dataset_and_leakage_free_protocol.csv",
    ),
    (
        "Table 2",
        "Clean baseline and architecture ablations",
        "table2_clean_baseline_and_architecture_ablations.csv",
    ),
    ("Table 3", "Duration comparison", "table3_duration_comparison.csv"),
    (
        "Table 4",
        "Hierarchical auxiliary weight ablation",
        "table4_hierarchical_aux_weight_ablation.csv",
    ),
    (
        "Table 5",
        "Clean Softmax/prototype/hierarchical prototype results",
        "table5_clean_softmax_main_prototype_hierarchical_prototype.csv",
    ),
    (
        "Table 6",
        "Simulated-noise robustness by stratum",
        "table6_simulated_noise_robustness_by_stratum.csv",
    ),
    ("Table 7", "Paired statistics", "table7_paired_statistics.csv"),
    (
        "Table 8",
        "Per-class noise results and confusion directions",
        "table8_per_class_noise_results_and_confusion_directions.csv",
    ),
    (
        "Table 9",
        "Selective prediction metrics after AUGRC correction",
        "table9_selective_prediction_metrics_corrected_augrc.csv",
    ),
]

FIGURE_SOURCES = {
    "figure1_overall_method_architecture": [
        "tools/train_hier_longcontext_crnn.py",
        "tools/prototype_model_adapter.py",
        "paper/tables/table1_dataset_and_leakage_free_protocol.csv",
    ],
    "figure2_duration_macro_f1": [
        "paper/tables/table3_duration_comparison.csv",
    ],
    "figure3_prototype_structure": [
        "tools/prototype_model_adapter.py",
    ],
    "figure4_macro_f1_vs_snr": [
        "paper/appendix/noise_25_run_summary.csv",
    ],
    "figure5_noise_degradation_by_environment": [
        "paper/appendix/noise_25_run_summary.csv",
    ],
    "figure6_feeding_stress_confusion": [
        "paper/tables/table8_per_class_noise_results_and_confusion_directions.csv",
        "paper/appendix/noise_25_run_summary.csv",
    ],
    "figure7_cough_collapse": [
        "paper/tables/table8_per_class_noise_results_and_confusion_directions.csv",
        "paper/appendix/noise_25_run_summary.csv",
    ],
    "figure8_risk_coverage_curves": [
        "paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv",
    ],
}


@dataclass(frozen=True)
class GeneratedArtifacts:
    figures: list[Path]
    tables: list[Path]
    manifests: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready figures and Word-friendly table files from "
            "approved paper CSV summaries. This script does not train models or "
            "modify source result CSV values."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root containing paper/ and tools/ (default: repository root).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution in dots per inch (default: 300).",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Only generate Word-friendly table files and manifests.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Only generate figures and manifests.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected {description} to be a file: {path}")
    return path


def require_sources(root: Path) -> None:
    for figure, sources in FIGURE_SOURCES.items():
        for source in sources:
            require_file(root / source, f"source for {figure}")
    for _, title, filename in TABLES:
        require_file(root / "paper" / "tables" / filename, f"source for {title}")


def read_csv(root: Path, relative_path: str, description: str) -> pd.DataFrame:
    return pd.read_csv(require_file(root / relative_path, description))


def read_csv_as_text(root: Path, relative_path: str, description: str) -> pd.DataFrame:
    return pd.read_csv(require_file(root / relative_path, description), dtype=str).fillna("")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, base_path: Path, dpi: int) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".svg", ".pdf", ".png"):
        out = base_path.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = dpi
        fig.savefig(out, **kwargs)
        outputs.append(out)
    plt.close(fig)
    return outputs


def fmt_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#333333",
    fontsize: int = 8,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#333333",
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def plot_figure1(root: Path, output_dir: Path, dpi: int) -> list[Path]:
    table1 = read_csv_as_text(
        root,
        "paper/tables/table1_dataset_and_leakage_free_protocol.csv",
        "dataset/leakage protocol table",
    )
    protocol = "; ".join(table1["value"].head(4).tolist())

    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    stages = [
        ("waveform", "#E6F0FA"),
        ("2s crop/pad", "#E6F0FA"),
        ("Log-Mel", "#E6F0FA"),
        ("CNN", "#E8F5E9"),
        ("BiGRU/GRU", "#E8F5E9"),
        ("embedding z", "#E8F5E9"),
    ]
    x0 = 0.03
    width = 0.13
    gap = 0.025
    y = 0.64
    height = 0.16
    centers = []
    for i, (label, color) in enumerate(stages):
        x = x0 + i * (width + gap)
        draw_box(ax, (x, y), width, height, label, color)
        centers.append((x + width / 2, y + height / 2))
        if i > 0:
            prev = centers[i - 1]
            draw_arrow(ax, (prev[0] + width / 2 - 0.01, prev[1]), (x - 0.01, y + height / 2))

    embed_x, embed_y = centers[-1]
    head_y = 0.36
    proto_y = 0.14
    head_specs = [
        ("main head", 0.50, METHOD_COLORS["raw_softmax"]),
        ("aux head", 0.68, "#D37295"),
    ]
    proto_specs = [
        ("main-class prototype", 0.50, METHOD_COLORS["prototype"]),
        ("hierarchical prototype\n(aux subtype map)", 0.68, METHOD_COLORS["hierarchical"]),
    ]
    for label, x, color in head_specs:
        draw_box(ax, (x, head_y), 0.15, 0.13, label, "#FFF4E6", edgecolor=color)
        draw_arrow(ax, (embed_x, embed_y - 0.09), (x + 0.075, head_y + 0.13), color=color, rad=0.05)
    for label, x, color in proto_specs:
        draw_box(ax, (x, proto_y), 0.15, 0.13, label, "#FFFFFF", edgecolor=color, fontsize=7)
        draw_arrow(ax, (x + 0.075, head_y), (x + 0.075, proto_y + 0.13), color=color)

    draw_box(ax, (0.84, 0.35), 0.13, 0.16, "confidence /\nrejection", "#F4F1FA", edgecolor="#6F4E9B")
    draw_arrow(ax, (0.65, head_y + 0.09), (0.84, 0.46), color="#6F4E9B", rad=0.06)
    draw_arrow(ax, (0.83, head_y + 0.04), (0.84, 0.40), color="#6F4E9B", rad=-0.04)
    draw_arrow(ax, (0.755, proto_y + 0.13), (0.86, 0.35), color="#6F4E9B", rad=-0.16)

    ax.text(0.03, 0.93, "Overall method architecture", fontsize=13, weight="bold", ha="left")
    ax.text(
        0.03,
        0.87,
        "cap3x + 2-second Log-Mel CRNN with hierarchical subtype auxiliary supervision",
        fontsize=9,
        ha="left",
    )
    ax.text(
        0.03,
        0.05,
        "Evaluation strata: clean | MODERATE_NOISE | EXTREME_STRESS | ALL_NOISY. "
        "Prototype/calibration artifacts are train/validation split specific.",
        fontsize=8,
        ha="left",
        color="#333333",
    )
    ax.text(
        0.03,
        0.01,
        f"Protocol source summary: {protocol}",
        fontsize=6.7,
        ha="left",
        color="#555555",
    )
    return save_figure(fig, output_dir / "figure1_overall_method_architecture", dpi)


def plot_figure2(root: Path, output_dir: Path, dpi: int) -> list[Path]:
    df = read_csv(root, "paper/tables/table3_duration_comparison.csv", "duration comparison table")
    df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)
    df = df.sort_values("duration_num")

    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    colors = ["#4C78A8", "#59A14F", "#F28E2B"]
    bars = ax.bar(
        df["duration"],
        df["mean_macro_f1"],
        yerr=df["std_macro_f1"],
        capsize=4,
        color=colors,
        edgecolor="#333333",
        linewidth=0.7,
    )
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["mean_macro_f1"] + row["std_macro_f1"] + 0.004,
            f"{row['mean_macro_f1']:.4f}\nn={int(row['n'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0.86, 1.005)
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Input duration")
    ax.set_title("Duration ablation on clean cap3x Log-Mel CRNN")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.plot([0, 1], [0.991, 0.991], color="#333333", linewidth=0.8)
    ax.text(
        0.5,
        0.994,
        "2s - 1s mean delta=0.0246, p=0.000162",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.text(0.01, -0.22, "Source: paper/tables/table3_duration_comparison.csv", transform=ax.transAxes, fontsize=7)
    return save_figure(fig, output_dir / "figure2_duration_macro_f1", dpi)


def plot_figure3(output_dir: Path, dpi: int) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    main_y = {
        "cough": 0.78,
        "calm_grunt": 0.58,
        "feeding": 0.38,
        "stress_vocal": 0.18,
    }
    aux_y = {
        "dry_cough": 0.86,
        "abdominal_cough": 0.72,
        "calm_grunt": 0.58,
        "feeding": 0.38,
        "frightened_stress": 0.24,
        "anxious_stress": 0.10,
    }
    ax.text(0.05, 0.96, "Prototype hierarchy", fontsize=13, weight="bold", ha="left")
    ax.text(0.16, 0.91, "main-class prototype", fontsize=9, weight="bold", ha="center")
    ax.text(0.72, 0.91, "hierarchical subtype prototype", fontsize=9, weight="bold", ha="center")

    for label in DEFAULT_MAIN_LABELS:
        draw_box(
            ax,
            (0.07, main_y[label] - 0.045),
            0.22,
            0.09,
            label,
            "#E8F5E9",
            edgecolor=METHOD_COLORS["prototype"],
        )
    for label in DEFAULT_AUX_LABELS:
        draw_box(
            ax,
            (0.60, aux_y[label] - 0.04),
            0.26,
            0.08,
            label,
            "#FFF4E6",
            edgecolor=METHOD_COLORS["hierarchical"],
        )
        main = DEFAULT_AUX_TO_MAIN[label]
        draw_arrow(
            ax,
            (0.60, aux_y[label]),
            (0.29, main_y[main]),
            color="#555555",
            rad=0.04,
        )

    ax.text(
        0.05,
        0.015,
        "Arrows map each subtype prototype back to its parent main class; "
        "this distinguishes main-class prototype inference from hierarchical prototype inference.",
        fontsize=7.8,
        ha="left",
        color="#333333",
    )
    return save_figure(fig, output_dir / "figure3_prototype_structure", dpi)


def numeric_snr_frame(noise: pd.DataFrame) -> pd.DataFrame:
    out = noise.copy()
    out["snr_numeric"] = pd.to_numeric(out["snr_db"], errors="coerce")
    return out[out["noise_environment"].isin(DEMAND_ENVS) & out["snr_numeric"].notna()]


def plot_figure4(root: Path, output_dir: Path, dpi: int) -> list[Path]:
    noise = numeric_snr_frame(read_csv(root, "paper/appendix/noise_25_run_summary.csv", "noise summary"))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), sharey=True)
    x = np.arange(len(SNR_ORDER))
    for ax, env in zip(axes, DEMAND_ENVS):
        env_df = noise[noise["noise_environment"] == env]
        for method in METHOD_ORDER:
            series = []
            for snr in SNR_ORDER:
                row = env_df[(env_df["method"] == method) & (env_df["snr_numeric"] == snr)]
                series.append(float(row["mean_macro_f1"].iloc[0]))
            ax.plot(
                x,
                series,
                marker="o",
                linewidth=1.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(env)
        ax.set_xticks(x, ["20", "10", "0"])
        ax.set_xlabel("active-event SNR (dB)")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_ylim(0.40, 0.82)
    axes[2].legend(loc="lower left", bbox_to_anchor=(1.02, 0.02), frameon=False)
    fig.suptitle("Macro-F1 versus active-event SNR under DEMAND simulated additive noise")
    fig.subplots_adjust(bottom=0.22, right=0.82, top=0.82, wspace=0.20)
    fig.text(0.02, 0.04, NOISE_NOTE, fontsize=7.5, ha="left")
    return save_figure(fig, output_dir / "figure4_macro_f1_vs_snr", dpi)


def plot_figure5(root: Path, output_dir: Path, dpi: int) -> list[Path]:
    noise = numeric_snr_frame(read_csv(root, "paper/appendix/noise_25_run_summary.csv", "noise summary"))
    rows: list[tuple[str, float]] = []
    values: list[list[float]] = []
    for env in DEMAND_ENVS:
        for snr in SNR_ORDER:
            rows.append((env, snr))
            row_vals = []
            for method in METHOD_ORDER:
                row = noise[
                    (noise["noise_environment"] == env)
                    & (noise["snr_numeric"] == snr)
                    & (noise["method"] == method)
                ]
                row_vals.append(float(row["mean_macro_f1_degradation_vs_clean"].iloc[0]))
            values.append(row_vals)
    matrix = np.asarray(values)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.18, vmax=0.52)
    ax.set_xticks(np.arange(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=14, ha="right")
    ax.set_yticks(np.arange(len(rows)), [f"{env} {int(snr)} dB" for env, snr in rows])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, fmt_float(matrix[i, j]), ha="center", va="center", fontsize=8)

    for i, (env, snr) in enumerate(rows):
        if env == "STRAFFIC" and snr == 0.0:
            ax.add_patch(Rectangle((-0.5, i - 0.5), len(METHOD_ORDER), 1.0, fill=False, edgecolor="#111111", lw=2.0))
            ax.text(
                len(METHOD_ORDER) - 0.02,
                i,
                "largest degradation region",
                ha="left",
                va="center",
                fontsize=8,
                weight="bold",
            )
    ax.set_title("Noise degradation by DEMAND environment")
    ax.set_xlabel("Method")
    ax.set_ylabel("Environment and active-event SNR")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Macro-F1 degradation vs clean")
    fig.subplots_adjust(left=0.18, right=0.84, bottom=0.28, top=0.90)
    fig.text(0.02, 0.04, NOISE_NOTE, fontsize=7.5, ha="left")
    return save_figure(fig, output_dir / "figure5_noise_degradation_by_environment", dpi)


def confusion_direction_frame(root: Path) -> pd.DataFrame:
    table = read_csv(
        root,
        "paper/tables/table8_per_class_noise_results_and_confusion_directions.csv",
        "per-class noise/confusion table",
    )
    table = table[table["method"].isin(METHOD_ORDER)]
    return table.drop_duplicates("method").set_index("method").loc[METHOD_ORDER].reset_index()


def plot_figure6(root: Path, output_dir: Path, dpi: int) -> list[Path]:
    confusion = confusion_direction_frame(root)
    noise = read_csv(root, "paper/appendix/noise_25_run_summary.csv", "noise summary")
    all_noisy = noise[(noise["noise_environment"] == "ALL_NOISY") & (noise["method"].isin(METHOD_ORDER))]
    macro = {row["method"]: float(row["mean_macro_f1"]) for _, row in all_noisy.iterrows()}

    x = np.arange(len(METHOD_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(
        x - width / 2,
        confusion["feeding_to_stress"],
        width=width,
        label="feeding -> stress_vocal",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        confusion["stress_to_feeding"],
        width=width,
        label="stress_vocal -> feeding",
        color="#F28E2B",
    )
    ax.set_xticks(x, [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=10, ha="right")
    ax.set_ylabel("Aggregated error count")
    ax.set_title("Feeding/Stress confusion under ALL_NOISY simulated additive noise")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_ylim(0, 2650)
    fig.subplots_adjust(bottom=0.28, top=0.86)
    ax.text(
        0.01,
        -0.32,
        "Prototype reduces feeding -> stress_vocal versus raw Softmax; hierarchical reduces "
        "stress_vocal -> feeding but has lower ALL_NOISY Macro-F1 than main-class prototype.",
        transform=ax.transAxes,
        fontsize=7.5,
        ha="left",
    )
    ax.text(
        0.01,
        -0.39,
        "ALL_NOISY Macro-F1: raw Softmax=0.6173, main-class prototype=0.6230, hierarchical prototype=0.6200.",
        transform=ax.transAxes,
        fontsize=7.5,
        ha="left",
    )
    return save_figure(fig, output_dir / "figure6_feeding_stress_confusion", dpi)


def plot_figure7(root: Path, output_dir: Path, dpi: int) -> list[Path]:
    table = read_csv(
        root,
        "paper/tables/table8_per_class_noise_results_and_confusion_directions.csv",
        "per-class noise/confusion table",
    )
    cough = table[(table["class"] == "cough") & (table["method"].isin(METHOD_ORDER))]
    cough = cough.set_index("method").loc[METHOD_ORDER].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1.0, 1.25]})
    axes[0].bar(
        np.arange(len(METHOD_ORDER)),
        cough["f1"],
        color=[METHOD_COLORS[m] for m in METHOD_ORDER],
        edgecolor="#333333",
        linewidth=0.6,
    )
    axes[0].set_xticks(np.arange(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15, ha="right")
    axes[0].set_ylabel("Cough F1")
    axes[0].set_ylim(0, 0.13)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    for i, value in enumerate(cough["f1"]):
        axes[0].text(i, float(value) + 0.004, fmt_float(float(value)), ha="center", fontsize=8)

    destinations = ["cough_to_calm_grunt", "cough_to_feeding", "cough_to_stress_vocal"]
    destination_labels = ["cough -> calm_grunt", "cough -> feeding", "cough -> stress_vocal"]
    destination_colors = ["#86BCB6", "#EDC948", "#E15759"]
    bottom = np.zeros(len(METHOD_ORDER))
    for col, label, color in zip(destinations, destination_labels, destination_colors):
        axes[1].bar(
            np.arange(len(METHOD_ORDER)),
            cough[col],
            bottom=bottom,
            label=label,
            color=color,
        )
        bottom += cough[col].to_numpy(dtype=float)
    axes[1].set_xticks(np.arange(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15, ha="right")
    axes[1].set_ylabel("Aggregated cough error count")
    axes[1].legend(frameon=False, fontsize=7.5)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    for i, row in cough.iterrows():
        axes[1].text(
            i,
            float(row["cough_to_stress_vocal"]) + float(row["cough_to_calm_grunt"]) + 200,
            f"to stress={int(row['cough_to_stress_vocal'])}",
            ha="center",
            fontsize=7,
        )

    fig.suptitle("Cough collapse under simulated active-event noise")
    fig.subplots_adjust(bottom=0.24, top=0.84, wspace=0.24)
    fig.text(0.02, 0.04, "ALL_NOISY stratum; DEMAND simulated additive noise.", fontsize=7.5, ha="left")
    return save_figure(fig, output_dir / "figure7_cough_collapse", dpi)


def plot_figure8(root: Path, output_dir: Path, dpi: int) -> list[Path]:
    table = read_csv(
        root,
        "paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv",
        "selective prediction metrics table",
    )
    df = table[(table["noise_environment"] == "ALL_NOISY") & (table["method"].isin(METHOD_ORDER))]
    df = df.set_index("method").loc[METHOD_ORDER].reset_index()
    coverages = [0.80, 0.90, 0.95]
    coverage_cols = [
        "mean_risk_at_coverage_0_80_y",
        "mean_risk_at_coverage_0_90_y",
        "mean_risk_at_coverage_0_95_y",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.3), gridspec_kw={"width_ratios": [1.15, 1.0]})
    for method in METHOD_ORDER:
        row = df[df["method"] == method].iloc[0]
        risks = [float(row[col]) for col in coverage_cols]
        axes[0].plot(
            coverages,
            risks,
            marker="o",
            linewidth=1.8,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0].set_xlabel("Coverage")
    axes[0].set_ylabel("Selective risk")
    axes[0].set_title("Fixed-coverage risk")
    axes[0].set_ylim(0.285, 0.314)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].legend(frameon=False, loc="lower left")

    x = np.arange(len(METHOD_ORDER))
    width = 0.34
    axes[1].bar(x - width / 2, df["mean_aurc"], width=width, label="AURC", color="#4C78A8")
    axes[1].bar(x + width / 2, df["mean_augrc"], width=width, label="AUGRC", color="#F28E2B")
    axes[1].set_xticks(x, [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15, ha="right")
    axes[1].set_ylabel("Risk-coverage area")
    axes[1].set_title("AURC / corrected AUGRC")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    for i, row in df.iterrows():
        axes[1].text(i - width / 2, float(row["mean_aurc"]) + 0.004, fmt_float(float(row["mean_aurc"])), ha="center", fontsize=7)
        axes[1].text(i + width / 2, float(row["mean_augrc"]) + 0.004, fmt_float(float(row["mean_augrc"])), ha="center", fontsize=7)

    fig.suptitle("Risk-coverage curves under ALL_NOISY simulated additive noise")
    fig.subplots_adjust(bottom=0.28, top=0.82, wspace=0.22)
    fig.text(
        0.01,
        0.04,
        "Lower is better. Prototype AURC/AUGRC is not better than raw Softmax; fixed-coverage risk is shown alongside area metrics.",
        fontsize=7.5,
        ha="left",
    )
    return save_figure(fig, output_dir / "figure8_risk_coverage_curves", dpi)


def generate_figures(root: Path, dpi: int) -> list[Path]:
    output_dir = root / "paper" / "figures"
    configure_matplotlib()
    outputs: list[Path] = []
    outputs.extend(plot_figure1(root, output_dir, dpi))
    outputs.extend(plot_figure2(root, output_dir, dpi))
    outputs.extend(plot_figure3(output_dir, dpi))
    outputs.extend(plot_figure4(root, output_dir, dpi))
    outputs.extend(plot_figure5(root, output_dir, dpi))
    outputs.extend(plot_figure6(root, output_dir, dpi))
    outputs.extend(plot_figure7(root, output_dir, dpi))
    outputs.extend(plot_figure8(root, output_dir, dpi))
    return outputs


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def dataframe_to_html(title: str, df: pd.DataFrame) -> str:
    table = df.to_html(index=False, escape=True, border=0, na_rep="")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1 {{ font-size: 18px; margin: 0 0 12px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 10.5pt; }}
th, td {{ border: 1px solid #888; padding: 4px 6px; vertical-align: top; }}
th {{ background: #f0f0f0; font-weight: 700; }}
caption {{ caption-side: top; text-align: left; font-weight: 700; margin-bottom: 8px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
{table}
</body>
</html>
"""


def docx_text(value: object) -> str:
    return "" if value is None else str(value)


def xml_run_text(value: object) -> str:
    text = docx_text(value)
    parts = text.splitlines() or [""]
    escaped_parts = []
    for idx, part in enumerate(parts):
        if idx:
            escaped_parts.append("<w:br/>")
        escaped_parts.append(f'<w:t xml:space="preserve">{html.escape(part)}</w:t>')
    return "".join(escaped_parts)


def docx_paragraph(text: str, *, bold: bool = False, size_half_points: int = 22) -> str:
    rpr = f"<w:rPr>{'<w:b/>' if bold else ''}<w:sz w:val=\"{size_half_points}\"/></w:rPr>"
    return f"<w:p><w:r>{rpr}{xml_run_text(text)}</w:r></w:p>"


def docx_cell(text: object, *, header: bool = False) -> str:
    shading = '<w:shd w:fill="F0F0F0"/>' if header else ""
    rpr = "<w:rPr><w:b/></w:rPr>" if header else ""
    return (
        "<w:tc>"
        f"<w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/>{shading}</w:tcPr>"
        f"<w:p><w:r>{rpr}{xml_run_text(text)}</w:r></w:p>"
        "</w:tc>"
    )


def docx_table(df: pd.DataFrame) -> str:
    borders = (
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="999999"/></w:tblBorders>'
    )
    rows = ["<w:tr>" + "".join(docx_cell(col, header=True) for col in df.columns) + "</w:tr>"]
    for _, row in df.iterrows():
        rows.append("<w:tr>" + "".join(docx_cell(row[col]) for col in df.columns) + "</w:tr>")
    return (
        "<w:tbl>"
        f"<w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>{borders}</w:tblPr>"
        + "".join(rows)
        + "</w:tbl>"
    )


def write_docx(path: Path, tables: Iterable[tuple[str, pd.DataFrame]]) -> None:
    body_parts = [
        docx_paragraph("Paper main tables", bold=True, size_half_points=32),
        docx_paragraph(
            "Generated from paper/tables CSV files. Result values are copied as text from the source CSVs.",
            size_half_points=18,
        ),
    ]
    for title, df in tables:
        body_parts.append(docx_paragraph(title, bold=True, size_half_points=26))
        body_parts.append(docx_table(df))
        body_parts.append(docx_paragraph("", size_half_points=6))
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body_parts)
        + '<w:sectPr><w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/>'
        '<w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" '
        'w:header="360" w:footer="360" w:gutter="0"/></w:sectPr>'
        "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)


def generate_tables(root: Path) -> list[Path]:
    output_dir = root / "paper" / "tables" / "word_friendly"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    docx_tables: list[tuple[str, pd.DataFrame]] = []
    html_parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Paper main tables</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#222;}"
        "h1{font-size:20px;}h2{font-size:16px;margin-top:24px;}"
        "table{border-collapse:collapse;width:100%;font-size:10.5pt;margin-bottom:18px;}"
        "th,td{border:1px solid #888;padding:4px 6px;vertical-align:top;}"
        "th{background:#f0f0f0;font-weight:700;}</style></head><body>"
        "<h1>Paper main tables</h1>"
    ]

    for table_id, title, filename in TABLES:
        full_title = f"{table_id}. {title}"
        rel_path = f"paper/tables/{filename}"
        df = read_csv_as_text(root, rel_path, full_title)
        docx_tables.append((full_title, df))
        html_file = output_dir / f"{slugify(table_id + '_' + title)}.html"
        html_file.write_text(dataframe_to_html(full_title, df), encoding="utf-8")
        outputs.append(html_file)
        html_parts.append(f"<h2>{html.escape(full_title)}</h2>")
        html_parts.append(df.to_html(index=False, escape=True, border=0, na_rep=""))

    consolidated_html = output_dir / "paper_main_tables.html"
    consolidated_html.write_text("\n".join(html_parts) + "</body></html>\n", encoding="utf-8")
    outputs.append(consolidated_html)

    docx_path = output_dir / "paper_main_tables.docx"
    write_docx(docx_path, docx_tables)
    outputs.append(docx_path)
    return outputs


def write_manifest(root: Path, artifacts: GeneratedArtifacts) -> list[Path]:
    output_dir = root / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "figure_data_sources.csv"
    json_path = output_dir / "figure_data_sources.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["figure", "sources"])
        for figure, sources in FIGURE_SOURCES.items():
            writer.writerow([figure, "; ".join(sources)])
    json_path.write_text(json.dumps(FIGURE_SOURCES, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    table_manifest = root / "paper" / "tables" / "word_friendly" / "table_sources.csv"
    table_manifest.parent.mkdir(parents=True, exist_ok=True)
    with table_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["table", "title", "source_csv"])
        for table_id, title, filename in TABLES:
            writer.writerow([table_id, title, f"paper/tables/{filename}"])

    return [csv_path, json_path, table_manifest]


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    require_sources(root)
    figure_outputs: list[Path] = []
    table_outputs: list[Path] = []

    if not args.skip_figures:
        figure_outputs = generate_figures(root, args.dpi)
    if not args.skip_tables:
        table_outputs = generate_tables(root)

    artifacts = GeneratedArtifacts(figures=figure_outputs, tables=table_outputs, manifests=[])
    manifest_outputs = write_manifest(root, artifacts)

    payload = {
        "figures": [str(p.relative_to(root).as_posix()) for p in figure_outputs],
        "tables": [str(p.relative_to(root).as_posix()) for p in table_outputs],
        "manifests": [str(p.relative_to(root).as_posix()) for p in manifest_outputs],
        "dpi": args.dpi,
        "numeric_result_csvs_modified": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
