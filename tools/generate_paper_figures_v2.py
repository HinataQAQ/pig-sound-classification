"""Generate publication figures for the pig sound classification manuscript.

The script reads only paper-ready CSV sources and writes Nature-style figure
exports plus a figure-contract document. It does not read audio, checkpoints,
prediction files, JSON files, or run model inference.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


METHOD_LABELS = {
    "raw_softmax": "Raw\nsoftmax",
    "calibrated_softmax": "Calibrated\nsoftmax",
    "prototype": "Main\nprototype",
    "hierarchical": "Hierarchical\nprototype",
    "fused": "Fused",
}

METHOD_COLORS = {
    "raw_softmax": "#4E5B73",
    "calibrated_softmax": "#8C96A8",
    "prototype": "#2C6E9F",
    "hierarchical": "#58A182",
    "fused": "#9A7B49",
}

ACCENT = "#B04A3A"
TEXT = "#20242A"
GRID = "#D9DEE7"


@dataclass(frozen=True)
class FigureContract:
    figure: str
    conclusion: str
    panels: list[str]
    sources: list[str]
    not_to_claim: list[str]
    exports: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper/figures_v2 manuscript figures from locked paper CSVs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/figures_v2"),
        help="Output directory for SVG, PDF, PNG, and contracts.",
    )
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4A4F59",
            "axes.linewidth": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def resolve(root: Path, rel: str) -> Path:
    return (root / rel).resolve()


def require_file(root: Path, rel: str) -> Path:
    path = resolve(root, rel)
    if not path.exists():
        raise FileNotFoundError(f"BLOCKER: required source file is missing: {path}")
    return path


def read_csv(root: Path, rel: str, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = require_file(root, rel)
    df = pd.read_csv(path)
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"BLOCKER: {path} is missing required columns: {missing}")
    return df


def read_text(root: Path, rel: str) -> str:
    path = require_file(root, rel)
    return path.read_text(encoding="utf-8")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
    ):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, **kwargs)
        if suffix == ".svg":
            svg_text = path.read_text(encoding="utf-8")
            path.write_text(re.sub(r"[ \t]+(\r?\n)", r"\1", svg_text), encoding="utf-8")
        written.append(path.as_posix())
    plt.close(fig)
    return written


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=TEXT,
    )


def clean_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)


def metric_col(df: pd.DataFrame, base: str) -> str:
    candidates = [base, f"{base}_x", f"{base}_y"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"BLOCKER: no column found for metric {base}; tried {candidates}")


def table1_value(table1: pd.DataFrame, item: str) -> str:
    rows = table1.loc[table1["item"] == item, "value"]
    if rows.empty:
        raise ValueError(f"BLOCKER: table1 is missing item: {item}")
    return str(rows.iloc[0])


def parse_delta_p(note: str) -> tuple[str, str]:
    delta = re.search(r"delta=([0-9.]+)", note)
    pval = re.search(r"p=([0-9.]+)", note)
    if not delta or not pval:
        raise ValueError(f"BLOCKER: could not parse paired delta and p-value from: {note}")
    return delta.group(1), pval.group(1)


def contract_exports(stem: str) -> list[str]:
    return [f"paper/figures_v2/{stem}.svg", f"paper/figures_v2/{stem}.pdf", f"paper/figures_v2/{stem}.png"]


def build_contracts() -> list[FigureContract]:
    return [
        FigureContract(
            figure="Figure 1. Overall workflow",
            conclusion=(
                "The manuscript evaluates a 2 s Log-Mel CRNN with hierarchical heads and "
                "train-only post-hoc prototypes under clean and simulated DEMAND noise settings."
            ),
            panels=[
                "A. Two-second event context and Log-Mel CRNN encoder.",
                "B. Main-class and auxiliary-subtype supervision heads.",
                "C. Train-only main and subtype prototype construction followed by validation-only calibration.",
                "D. DEMAND simulated-noise evaluation boundary.",
            ],
            sources=["paper/tables/table1_dataset_and_leakage_free_protocol.csv", "paper/CLAIMS_AND_EVIDENCE.md"],
            not_to_claim=[
                "Do not claim real-farm external validation.",
                "Do not claim prototype learning was invented here.",
                "Do not claim 0 dB is a clean or no-noise setting.",
            ],
            exports=contract_exports("figure1_overall_workflow"),
        ),
        FigureContract(
            figure="Figure 2. Duration ablation",
            conclusion=(
                "A 2 s Log-Mel context is the clean-audio mainline because it improves Macro-F1 "
                "over the 1 s baseline in matched 25-run statistics."
            ),
            panels=[
                "A. Mean Macro-F1 for 1 s, 2 s, and 3 s contexts with run-to-run standard deviation.",
                "B. Annotation of the 2 s - 1 s paired delta and Wilcoxon p-value from the table note.",
            ],
            sources=["paper/tables/table3_duration_comparison.csv"],
            not_to_claim=[
                "Do not claim first use of 2 s context.",
                "Do not overinterpret 3 s against 2 s because the run counts differ.",
            ],
            exports=contract_exports("figure2_duration_ablation"),
        ),
        FigureContract(
            figure="Figure 3. Hierarchical supervision story",
            conclusion=(
                "Auxiliary subtype supervision adds fine-grained semantic structure; lambda=0.5 is "
                "the stable setting and lambda=1.0 has the highest mean Macro-F1."
            ),
            panels=[
                "A. Four main classes and six auxiliary subtypes.",
                "B. Training objective L_main + lambda L_aux.",
                "C. Lambda ablation with Macro-F1 mean and standard deviation.",
            ],
            sources=[
                "paper/tables/table1_dataset_and_leakage_free_protocol.csv",
                "paper/tables/table4_hierarchical_aux_weight_ablation.csv",
            ],
            not_to_claim=[
                "Do not claim the hierarchical gain is statistically significant over the 2 s baseline.",
                "Do not treat the fused output as the main innovation.",
            ],
            exports=contract_exports("figure3_hierarchical_supervision"),
        ),
        FigureContract(
            figure="Figure 4. Clean prototype comparison",
            conclusion=(
                "On clean test data, hierarchical prototype inference has the highest Macro-F1, "
                "whereas calibration metrics differ by method."
            ),
            panels=[
                "A. Macro-F1.",
                "B. Expected calibration error.",
                "C. Brier score.",
                "D. Negative log likelihood.",
            ],
            sources=["paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv"],
            not_to_claim=[
                "Do not claim all prototype variants improve every metric.",
                "Do not claim universal uncertainty improvement.",
            ],
            exports=contract_exports("figure4_clean_prototype_comparison"),
        ),
        FigureContract(
            figure="Figure 5. Simulated noise robustness",
            conclusion=(
                "The main-class prototype is the most stable aggregate option for MODERATE_NOISE "
                "and ALL_NOISY, while hierarchical prototypes are not noise-optimal."
            ),
            panels=[
                "A. Macro-F1 across MODERATE_NOISE, EXTREME_STRESS, and ALL_NOISY.",
                "B. Paired statistics callouts for prototype - raw softmax.",
            ],
            sources=[
                "paper/tables/table6_simulated_noise_robustness_by_stratum.csv",
                "paper/tables/table7_paired_statistics.csv",
            ],
            not_to_claim=[
                "Do not claim hierarchical prototype is best under noise.",
                "Do not claim simulated DEMAND noise is real-farm external validation.",
            ],
            exports=contract_exports("figure5_simulated_noise_robustness"),
        ),
        FigureContract(
            figure="Figure 6. Active-event SNR curve",
            conclusion=(
                "Macro-F1 degrades from clean through 20, 10, and 0 dB active-event SNR, "
                "with 0 dB explicitly treated as an extreme simulated-noise stress condition."
            ),
            panels=[
                "A. DWASHING SNR curve.",
                "B. STRAFFIC SNR curve.",
                "C. TBUS SNR curve.",
            ],
            sources=["reports/prototype_noise_demand_w05_final/final_summary.csv"],
            not_to_claim=[
                "Do not describe 0 dB as no noise.",
                "Do not generalize these simulated DEMAND conditions to external farm validation.",
            ],
            exports=contract_exports("figure6_active_event_snr_curve"),
        ),
        FigureContract(
            figure="Figure 7. Per-class failure",
            conclusion=(
                "Cough recognition collapses under noisy aggregate evaluation, and cough-to-stress "
                "plus feeding/stress directions dominate the failure profile."
            ),
            panels=[
                "A. Per-class F1 under ALL_NOISY for raw, main prototype, and hierarchical prototype.",
                "B. Cough-to-stress_vocal confusion counts.",
                "C. Feeding-to-stress and stress-to-feeding confusion directions.",
            ],
            sources=[
                "paper/tables/table8_per_class_noise_results_and_confusion_directions.csv",
                "reports/prototype_noise_demand_w05_final/final_per_class_summary.csv",
            ],
            not_to_claim=[
                "Do not claim cough is reliable at 0 dB.",
                "Do not infer unmeasured causal noise mechanisms from aggregate confusion counts.",
            ],
            exports=contract_exports("figure7_per_class_failure"),
        ),
        FigureContract(
            figure="Figure 8. Selective prediction",
            conclusion=(
                "Prototype inference does not comprehensively improve uncertainty ranking: raw "
                "softmax has lower AURC/AUGRC, while main prototypes slightly improve frozen-threshold risk."
            ),
            panels=[
                "A. AURC.",
                "B. AUGRC.",
                "C. Risk at 95% coverage.",
                "D. Frozen-threshold selective risk.",
            ],
            sources=["paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv"],
            not_to_claim=[
                "Do not claim prototype methods universally improve uncertainty.",
                "Do not omit that AURC/AUGRC rank raw softmax better.",
            ],
            exports=contract_exports("figure8_selective_prediction"),
        ),
    ]


def write_contracts(out_dir: Path, contracts: list[FigureContract]) -> Path:
    lines = ["# Figure Contracts", ""]
    for contract in contracts:
        lines.extend(
            [
                f"## {contract.figure}",
                "",
                f"- Figure conclusion: {contract.conclusion}",
                "- Panels:",
            ]
        )
        lines.extend([f"  - {panel}" for panel in contract.panels])
        lines.append("- Evidence source CSV/source files:")
        lines.extend([f"  - {source}" for source in contract.sources])
        lines.append("- What not to claim:")
        lines.extend([f"  - {claim}" for claim in contract.not_to_claim])
        lines.append("- Export files:")
        lines.extend([f"  - {export}" for export in contract.exports])
        lines.append("")
    path = out_dir / "figure_contracts.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def figure1(root: Path, out_dir: Path) -> list[str]:
    table1 = read_csv(root, "paper/tables/table1_dataset_and_leakage_free_protocol.csv", ["item", "value"])
    claims = read_text(root, "paper/CLAIMS_AND_EVIDENCE.md")
    if "simulated DEMAND noise" not in claims and "simulated-noise" not in claims:
        raise ValueError("BLOCKER: CLAIMS_AND_EVIDENCE.md lacks simulated-noise boundary text.")

    clean_classes = table1_value(table1, "Clean classes").replace("; ", "\n")
    aux_subtypes = table1_value(table1, "Auxiliary subtypes").replace("; ", "\n")
    noise_protocol = table1_value(table1, "Noise protocol")
    external_validation = table1_value(table1, "External validation")
    if "DEMAND" not in noise_protocol or "20/10/0 dB" not in noise_protocol:
        raise ValueError("BLOCKER: table1 noise protocol lacks DEMAND 20/10/0 dB boundary.")

    fig, ax = plt.subplots(figsize=(7.4, 4.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        fc: str,
        ec: str = "#556070",
        fontsize: float = 7.4,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=0.8,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=TEXT, fontsize=fontsize, linespacing=1.12)

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.85,
                color="#56616F",
                shrinkA=3,
                shrinkB=3,
            )
        )

    box(0.04, 0.72, 0.18, 0.12, "Pig-vocal event\n2 s context", "#E9EEF4", fontsize=7.8)
    box(0.29, 0.72, 0.18, 0.12, "Log-Mel\nspectrogram", "#E7F0F5", fontsize=7.8)
    box(0.54, 0.72, 0.18, 0.12, "CRNN\nencoder", "#E7F0F5", fontsize=7.8)
    arrow((0.22, 0.78), (0.29, 0.78))
    arrow((0.47, 0.78), (0.54, 0.78))

    box(0.79, 0.76, 0.18, 0.19, f"Main head\n4 classes\n{clean_classes}", "#EEF5EA", fontsize=6.6)
    box(0.79, 0.47, 0.18, 0.25, f"Aux head\n6 subtypes\n{aux_subtypes}", "#EEF5EA", fontsize=6.2)
    arrow((0.72, 0.79), (0.79, 0.86))
    arrow((0.72, 0.75), (0.79, 0.60))

    box(0.20, 0.35, 0.23, 0.14, "Train-only main\nprototypes\n4 class centers", "#F3EEE2")
    box(0.50, 0.35, 0.23, 0.14, "Train-only subtype\nprototypes\n6 subtype centers", "#F3EEE2")
    arrow((0.63, 0.72), (0.32, 0.49))
    arrow((0.64, 0.72), (0.62, 0.49))

    box(0.20, 0.11, 0.23, 0.12, "Validation-only\ncalibration", "#F7F2E8")
    box(0.50, 0.11, 0.23, 0.12, "Frozen test\ninference", "#F7F2E8")
    arrow((0.32, 0.35), (0.32, 0.23))
    arrow((0.62, 0.35), (0.62, 0.23))

    noise_text = (
        "DEMAND simulated\n"
        "noise evaluation\n"
        "DWASHING / TBUS /\n"
        "STRAFFIC\n"
        "active-event SNR:\n"
        "20, 10, 0 dB"
    )
    box(0.77, 0.15, 0.20, 0.27, noise_text, "#F6E9E5", ACCENT, fontsize=6.6)
    arrow((0.73, 0.17), (0.77, 0.26))
    ax.text(
        0.77,
        0.06,
        f"Boundary: simulated noise != real-farm external validation\n{external_validation}",
        ha="left",
        va="bottom",
        color=ACCENT,
        fontsize=6.8,
    )
    ax.text(0.02, 0.97, "Figure 1 | Overall workflow", fontweight="bold", fontsize=9.5, color=TEXT, va="top")
    return save_figure(fig, out_dir, "figure1_overall_workflow")


def figure2(root: Path, out_dir: Path) -> list[str]:
    df = read_csv(
        root,
        "paper/tables/table3_duration_comparison.csv",
        ["duration", "model", "n", "mean_macro_f1", "std_macro_f1", "paired_note"],
    )
    order = ["1 s", "2 s", "3 s"]
    plot_df = df.set_index("duration").loc[order].reset_index()
    note = plot_df.loc[plot_df["duration"] == "2 s", "paired_note"].iloc[0]
    delta, pval = parse_delta_p(str(note))

    fig, ax = plt.subplots(figsize=(4.6, 3.25))
    x = np.arange(len(plot_df))
    colors = ["#9AA4B2", METHOD_COLORS["prototype"], "#C4A36D"]
    ax.bar(
        x,
        plot_df["mean_macro_f1"],
        yerr=plot_df["std_macro_f1"],
        width=0.62,
        color=colors,
        edgecolor="#30343A",
        linewidth=0.55,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row.duration}\n(n={int(row.n)})" for row in plot_df.itertuples()])
    ax.set_ylabel("Macro-F1")
    ax.set_title("2 s context is the clean-audio mainline")
    ax.set_ylim(0.88, 0.98)
    clean_axis(ax)
    for i, row in enumerate(plot_df.itertuples()):
        ax.text(i, row.mean_macro_f1 + row.std_macro_f1 + 0.004, f"{row.mean_macro_f1:.4f}", ha="center")

    y = 0.971
    ax.plot([0, 0, 1, 1], [y - 0.004, y, y, y - 0.004], color=ACCENT, linewidth=0.8)
    ax.text(0.5, y + 0.002, f"2 s - 1 s delta={delta}; Wilcoxon p={pval}", ha="center", color=ACCENT)
    ax.text(0.02, 0.02, "No claim of first use of 2 s context.", transform=ax.transAxes, color="#5B626D")
    return save_figure(fig, out_dir, "figure2_duration_ablation")


def figure3(root: Path, out_dir: Path) -> list[str]:
    table1 = read_csv(root, "paper/tables/table1_dataset_and_leakage_free_protocol.csv", ["item", "value"])
    df = read_csv(
        root,
        "paper/tables/table4_hierarchical_aux_weight_ablation.csv",
        ["model", "n", "mean_macro_f1", "std_macro_f1", "mean_aux_f1"],
    )
    main_classes = [item.strip() for item in table1_value(table1, "Clean classes").split(";")]
    aux_subtypes = [item.strip() for item in table1_value(table1, "Auxiliary subtypes").split(";")]

    def lambda_label(model: str) -> str:
        if model.endswith("w02"):
            return "0.2"
        if model.endswith("w05"):
            return "0.5"
        if model.endswith("w10"):
            return "1.0"
        raise ValueError(f"BLOCKER: cannot parse lambda weight from {model}")

    df = df.copy()
    df["lambda"] = df["model"].map(lambda_label)
    df = df.set_index("lambda").loc[["0.2", "0.5", "1.0"]].reset_index()

    fig = plt.figure(figsize=(7.4, 4.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.85, 1.35], wspace=0.52)
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2])

    for ax in (ax_tree, ax_loss):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    add_panel_label(ax_tree, "A")
    add_panel_label(ax_loss, "B")
    add_panel_label(ax_bar, "C")
    ax_tree.set_title("Main labels bind to subtypes", loc="left")

    y_positions = np.linspace(0.82, 0.20, len(main_classes))
    subtype_map = {
        "cough": ["dry_cough", "abdominal_cough"],
        "calm_grunt": ["calm_grunt"],
        "feeding": ["feeding"],
        "stress_vocal": ["frightened_stress", "anxious_stress"],
    }
    for cls, y in zip(main_classes, y_positions):
        ax_tree.add_patch(
            FancyBboxPatch((0.03, y - 0.045), 0.34, 0.09, boxstyle="round,pad=0.012", fc="#E9EEF4", ec="#556070", lw=0.7)
        )
        ax_tree.text(0.20, y, cls, ha="center", va="center")
        sub = subtype_map.get(cls, [])
        sub_text = "\n".join(sub)
        ax_tree.add_patch(
            FancyBboxPatch((0.55, y - 0.055), 0.40, 0.11, boxstyle="round,pad=0.012", fc="#EEF5EA", ec="#556070", lw=0.7)
        )
        ax_tree.text(0.75, y, sub_text, ha="center", va="center", fontsize=6.5, linespacing=1.1)
        ax_tree.add_patch(FancyArrowPatch((0.37, y), (0.55, y), arrowstyle="-|>", mutation_scale=8, lw=0.75, color="#56616F"))
    if set(sum(subtype_map.values(), [])) != set(aux_subtypes):
        raise ValueError("BLOCKER: subtype mapping does not match table1 auxiliary subtype list.")

    ax_loss.set_title("Objective", loc="left")
    ax_loss.text(
        0.5,
        0.62,
        "Loss = L_main\n+ lambda x L_aux",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=TEXT,
    )
    ax_loss.text(
        0.5,
        0.33,
        "Shared CRNN embedding\nfour-class decision\nsix-subtype boundary cue",
        ha="center",
        va="center",
        color="#5B626D",
        linespacing=1.2,
    )

    ax_bar.bar(
        np.arange(len(df)),
        df["mean_macro_f1"],
        yerr=df["std_macro_f1"],
        width=0.6,
        color=[METHOD_COLORS["hierarchical"], METHOD_COLORS["prototype"], "#2D7F67"],
        edgecolor="#30343A",
        linewidth=0.55,
        capsize=3,
    )
    ax_bar.set_xticks(np.arange(len(df)))
    ax_bar.set_xticklabels([f"{value}\n(n={int(n)})" for value, n in zip(df["lambda"], df["n"])])
    ax_bar.set_xlabel("lambda")
    ax_bar.set_ylabel("Macro-F1", labelpad=2)
    ax_bar.set_title("Auxiliary-weight ablation", loc="left")
    ax_bar.set_ylim(0.925, 0.975)
    clean_axis(ax_bar)
    for i, row in enumerate(df.itertuples()):
        label = f"{row.mean_macro_f1:.4f}"
        ax_bar.text(i, row.mean_macro_f1 + row.std_macro_f1 + 0.002, label, ha="center")
    stable_idx = int(df.index[df["lambda"] == "0.5"][0])
    high_idx = int(df.index[df["lambda"] == "1.0"][0])
    ax_bar.annotate("stable", xy=(stable_idx, df.loc[stable_idx, "mean_macro_f1"]), xytext=(stable_idx - 0.45, 0.969),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=ACCENT), color=ACCENT)
    ax_bar.annotate("highest mean", xy=(high_idx, df.loc[high_idx, "mean_macro_f1"]), xytext=(high_idx - 0.45, 0.964),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=ACCENT), color=ACCENT)
    fig.subplots_adjust(top=0.88, bottom=0.18)
    return save_figure(fig, out_dir, "figure3_hierarchical_supervision")


def plot_metric_bars(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, ylabel: str, order: list[str]) -> None:
    values = df.set_index("method").loc[order, metric]
    x = np.arange(len(order))
    ax.bar(
        x,
        values,
        width=0.62,
        color=[METHOD_COLORS[m] for m in order],
        edgecolor="#30343A",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in order], rotation=0)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    clean_axis(ax)
    for i, val in enumerate(values):
        ax.text(i, val + (values.max() - values.min() + 0.01) * 0.04, f"{val:.4f}", ha="center", fontsize=6.8)


def figure4(root: Path, out_dir: Path) -> list[str]:
    df = read_csv(
        root,
        "paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv",
        ["method", "mean_macro_f1", "mean_ece", "mean_brier", "mean_nll"],
    )
    order = ["raw_softmax", "calibrated_softmax", "prototype", "hierarchical", "fused"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.85))
    metrics = [
        ("mean_macro_f1", "Macro-F1", "higher is better"),
        ("mean_ece", "ECE", "lower is better"),
        ("mean_brier", "Brier score", "lower is better"),
        ("mean_nll", "NLL", "lower is better"),
    ]
    for ax, (metric, title, subtitle), label in zip(axes.ravel(), metrics, ["A", "B", "C", "D"]):
        add_panel_label(ax, label)
        plot_metric_bars(ax, df, metric, f"{title} ({subtitle})", title, order)
        if metric == "mean_macro_f1":
            ax.set_ylim(0.946, 0.957)
        elif metric == "mean_ece":
            ax.set_ylim(0, 0.036)
        elif metric == "mean_brier":
            ax.set_ylim(0, 0.082)
        else:
            ax.set_ylim(0, 0.14)
    for ax in axes[0, :]:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    fig.subplots_adjust(hspace=0.48, wspace=0.22, top=0.90, bottom=0.13)
    fig.suptitle("Figure 4 | Clean prototype comparison", x=0.02, y=0.985, ha="left", fontweight="bold")
    return save_figure(fig, out_dir, "figure4_clean_prototype_comparison")


def figure5(root: Path, out_dir: Path) -> list[str]:
    df = read_csv(
        root,
        "paper/tables/table6_simulated_noise_robustness_by_stratum.csv",
        ["noise_environment", "method", "mean_macro_f1", "std_macro_f1"],
    )
    stats = read_csv(
        root,
        "paper/tables/table7_paired_statistics.csv",
        ["experiment", "noise_environment", "snr_db", "comparison", "mean_delta", "wilcoxon_p"],
    )
    strata = ["MODERATE_NOISE", "EXTREME_STRESS", "ALL_NOISY"]
    methods = ["raw_softmax", "prototype", "hierarchical"]
    plot_df = df[df["noise_environment"].isin(strata) & df["method"].isin(methods)].copy()

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(strata))
    width = 0.23
    offsets = [-width, 0, width]
    for offset, method in zip(offsets, methods):
        vals = plot_df[plot_df["method"] == method].set_index("noise_environment").loc[strata]
        ax.bar(
            x + offset,
            vals["mean_macro_f1"],
            yerr=vals["std_macro_f1"],
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#30343A",
            linewidth=0.5,
            capsize=2.5,
            label=METHOD_LABELS[method].replace("\n", " "),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Moderate\nnoise", "Extreme\nstress", "All\nnoisy"])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Main-class prototype is steadier under simulated noise")
    ax.set_ylim(0.52, 0.68)
    clean_axis(ax)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))

    for idx, stratum in enumerate(["MODERATE_NOISE", "ALL_NOISY"]):
        row = stats[
            (stats["experiment"] == "simulated_noise")
            & (stats["noise_environment"] == stratum)
            & (stats["snr_db"].astype(str) == "GROUP")
            & (stats["comparison"] == "prototype - raw_softmax")
        ]
        if row.empty:
            raise ValueError(f"BLOCKER: missing paired statistics for {stratum} prototype - raw_softmax")
        delta = row["mean_delta"].iloc[0]
        pval = row["wilcoxon_p"].iloc[0]
        xpos = 0 if stratum == "MODERATE_NOISE" else 2
        ax.text(xpos, 0.672, f"delta={delta:.4f}\np={pval:.4g}", ha="center", va="top", color=ACCENT)
    return save_figure(fig, out_dir, "figure5_simulated_noise_robustness")


def figure6(root: Path, out_dir: Path) -> list[str]:
    df = read_csv(
        root,
        "reports/prototype_noise_demand_w05_final/final_summary.csv",
        ["noise_environment", "snr_db", "method", "mean_macro_f1", "std_macro_f1"],
    )
    methods = ["raw_softmax", "prototype", "hierarchical"]
    envs = ["DWASHING", "STRAFFIC", "TBUS"]
    x = np.arange(4)
    labels = ["clean", "20 dB", "10 dB", "0 dB"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35), sharey=True)
    clean_rows = df[(df["noise_environment"] == "clean") & (df["method"].isin(methods))]
    if clean_rows.empty:
        raise ValueError("BLOCKER: final_summary.csv lacks clean rows.")

    for ax, env, panel in zip(axes, envs, ["A", "B", "C"]):
        add_panel_label(ax, panel)
        ax.set_title(env, loc="left")
        env_rows = df[(df["noise_environment"] == env) & (df["method"].isin(methods))]
        for method in methods:
            clean_value = clean_rows[clean_rows["method"] == method]["mean_macro_f1"].iloc[0]
            values = [clean_value]
            errors = [clean_rows[clean_rows["method"] == method]["std_macro_f1"].iloc[0]]
            for snr in ["20.0", "10.0", "0.0"]:
                row = env_rows[(env_rows["snr_db"].astype(str) == snr) & (env_rows["method"] == method)]
                if row.empty:
                    raise ValueError(f"BLOCKER: missing {env} {snr} {method} in final_summary.csv")
                values.append(row["mean_macro_f1"].iloc[0])
                errors.append(row["std_macro_f1"].iloc[0])
            ax.errorbar(
                x,
                values,
                yerr=errors,
                marker="o",
                markersize=3.2,
                linewidth=1.1,
                capsize=2,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method].replace("\n", " "),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.40, 0.99)
        clean_axis(ax)
        ax.axvspan(2.72, 3.28, color="#F6E9E5", zorder=0)
        ax.text(3, 0.965, "0 dB = extreme\nsimulated stress", ha="center", va="top", color=ACCENT, fontsize=6.5)
    axes[0].set_ylabel("Macro-F1")
    axes[1].set_xlabel("Active-event SNR condition")
    axes[0].legend(ncols=3, loc="upper left", bbox_to_anchor=(0.02, -0.18))
    fig.suptitle("Figure 6 | Active-event SNR degradation under DEMAND simulated noise", x=0.02, y=0.995, ha="left", fontweight="bold")
    return save_figure(fig, out_dir, "figure6_active_event_snr_curve")


def figure7(root: Path, out_dir: Path) -> list[str]:
    table8 = read_csv(
        root,
        "paper/tables/table8_per_class_noise_results_and_confusion_directions.csv",
        [
            "noise_environment",
            "method",
            "class",
            "f1",
            "feeding_to_stress",
            "stress_to_feeding",
            "cough_to_stress_vocal",
        ],
    )
    per_class = read_csv(
        root,
        "reports/prototype_noise_demand_w05_final/final_per_class_summary.csv",
        ["noise_environment", "snr_db", "method", "class", "recall"],
    )
    methods = ["raw_softmax", "prototype", "hierarchical"]
    classes = ["cough", "calm_grunt", "feeding", "stress_vocal"]
    plot_df = table8[
        (table8["noise_environment"] == "ALL_NOISY")
        & (table8["method"].isin(methods))
        & (table8["class"].isin(classes))
    ]
    if plot_df.empty:
        raise ValueError("BLOCKER: table8 lacks ALL_NOISY per-class rows.")

    confusion = plot_df.drop_duplicates("method").set_index("method").loc[methods]
    zero_cough = per_class[
        (per_class["snr_db"].astype(str) == "0.0")
        & (per_class["class"] == "cough")
        & (per_class["method"].isin(methods))
    ]
    if zero_cough.empty:
        raise ValueError("BLOCKER: final_per_class_summary.csv lacks 0 dB cough rows.")
    recall_min = zero_cough["recall"].min() * 100
    recall_max = zero_cough["recall"].max() * 100

    fig = plt.figure(figsize=(7.4, 4.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 0.95, 1.2], wspace=0.50)
    ax_f1 = fig.add_subplot(gs[0, 0])
    ax_cough = fig.add_subplot(gs[0, 1])
    ax_dirs = fig.add_subplot(gs[0, 2])
    add_panel_label(ax_f1, "A")
    add_panel_label(ax_cough, "B")
    add_panel_label(ax_dirs, "C")

    x = np.arange(len(classes))
    width = 0.23
    for idx, method in enumerate(methods):
        vals = plot_df[plot_df["method"] == method].set_index("class").loc[classes]["f1"]
        ax_f1.bar(
            x + (idx - 1) * width,
            vals,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#30343A",
            linewidth=0.5,
            label=METHOD_LABELS[method].replace("\n", " "),
        )
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(["cough", "calm\ngrunt", "feeding", "stress\nvocal"])
    ax_f1.set_ylabel("F1 under ALL_NOISY")
    ax_f1.set_title("Per-class F1", loc="left")
    ax_f1.set_ylim(0, 1.08)
    clean_axis(ax_f1)
    ax_f1.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax_f1.text(
        0,
        0.22,
        f"0 dB cough recall range\nacross DEMAND methods/envs:\n{recall_min:.1f}-{recall_max:.1f}%",
        ha="center",
        va="bottom",
        color=ACCENT,
        fontsize=6.8,
    )

    ax_cough.bar(
        np.arange(len(methods)),
        confusion["cough_to_stress_vocal"],
        width=0.62,
        color=[METHOD_COLORS[m] for m in methods],
        edgecolor="#30343A",
        linewidth=0.5,
    )
    ax_cough.set_xticks(np.arange(len(methods)))
    ax_cough.set_xticklabels(["Raw", "Main", "Hier"])
    ax_cough.set_title("cough -> stress", loc="left")
    ax_cough.set_ylabel("Confusion count")
    clean_axis(ax_cough)

    directions = ["feeding_to_stress", "stress_to_feeding"]
    x2 = np.arange(len(directions))
    for idx, method in enumerate(methods):
        vals = confusion.loc[method, directions]
        ax_dirs.bar(
            x2 + (idx - 1) * width,
            vals,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#30343A",
            linewidth=0.5,
        )
    ax_dirs.set_xticks(x2)
    ax_dirs.set_xticklabels(["feeding\n-> stress", "stress\n-> feeding"])
    ax_dirs.set_title("feeding/stress", loc="left")
    ax_dirs.set_ylabel("Confusion count")
    clean_axis(ax_dirs)
    fig.subplots_adjust(top=0.82, bottom=0.25)
    fig.suptitle("Figure 7 | Per-class failure under simulated noise", x=0.02, y=0.985, ha="left", fontweight="bold")
    return save_figure(fig, out_dir, "figure7_per_class_failure")


def figure8(root: Path, out_dir: Path) -> list[str]:
    df = read_csv(
        root,
        "paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv",
        [
            "noise_environment",
            "method",
            "mean_frozen_threshold_selective_risk",
            "mean_aurc",
            "mean_augrc",
        ],
    )
    risk95_col = metric_col(df, "mean_risk_at_coverage_0_95")
    methods = ["raw_softmax", "prototype", "hierarchical", "fused"]
    plot_df = df[(df["noise_environment"] == "ALL_NOISY") & (df["method"].isin(methods))].copy()
    if plot_df.empty:
        raise ValueError("BLOCKER: table9 lacks ALL_NOISY selective prediction rows.")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8))
    metrics = [
        ("mean_aurc", "AURC", "Raw softmax ranks best"),
        ("mean_augrc", "AUGRC", "Raw softmax ranks best"),
        (risk95_col, "Risk at 95% coverage", "Prototype slightly lower"),
        ("mean_frozen_threshold_selective_risk", "Frozen-threshold risk", "Prototype slightly lower"),
    ]
    for ax, (metric, ylabel, subtitle), label in zip(axes.ravel(), metrics, ["A", "B", "C", "D"]):
        add_panel_label(ax, label)
        plot_metric_bars(ax, plot_df, metric, ylabel, ylabel, methods)
        ax.text(0.02, 0.92, subtitle, transform=ax.transAxes, color="#5B626D", va="top", fontsize=6.8)
        ax.set_ylim(0, plot_df[metric].max() * 1.22)
    fig.subplots_adjust(hspace=0.55, wspace=0.25, top=0.90, bottom=0.13)
    fig.suptitle("Figure 8 | Selective prediction under ALL_NOISY", x=0.02, y=0.985, ha="left", fontweight="bold")
    return save_figure(fig, out_dir, "figure8_selective_prediction")


def generate_all(root: Path, out_dir: Path) -> dict[str, list[str]]:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    contracts = build_contracts()
    write_contracts(out_dir, contracts)
    return {
        "Figure 1": figure1(root, out_dir),
        "Figure 2": figure2(root, out_dir),
        "Figure 3": figure3(root, out_dir),
        "Figure 4": figure4(root, out_dir),
        "Figure 5": figure5(root, out_dir),
        "Figure 6": figure6(root, out_dir),
        "Figure 7": figure7(root, out_dir),
        "Figure 8": figure8(root, out_dir),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    outputs = generate_all(root, out_dir)
    print("Generated figure contracts and exports:")
    print((out_dir / "figure_contracts.md").as_posix())
    for figure, paths in outputs.items():
        for path in paths:
            print(f"{figure}: {path}")


if __name__ == "__main__":
    main()
