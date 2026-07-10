"""Generate the locked journal figure package without running any model.

The script reads only paper-ready CSV/Markdown evidence and writes figure
artwork, legends, source maps, and QA records to the explicitly approved paper
directories. It does not read audio, checkpoints, prediction files, or result
JSON files, and it never writes into a source-data directory.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image


MAIN_FIGURE_STEMS = (
    "figure1_overall_framework",
    "figure2_duration_clean_ablations",
    "figure3_hierarchical_clean_prototypes",
    "figure4_simulated_noise_robustness",
    "figure5_failure_selective_prediction",
)

SUPPLEMENTARY_FIGURE_STEMS = (
    "figure_s1_clean_ablations",
    "figure_s2_demand_environment_snr",
    "figure_s3_confusion_calibration_fold_deltas",
)

MAIN_LABELS = ("cough", "calm_grunt", "feeding", "stress_vocal")
AUX_TO_MAIN = {
    "dry_cough": "cough",
    "abdominal_cough": "cough",
    "calm_grunt": "calm_grunt",
    "feeding": "feeding",
    "frightened_stress": "stress_vocal",
    "anxious_stress": "stress_vocal",
}

METHOD_ORDER = ("raw_softmax", "prototype", "hierarchical")
METHOD_LABELS = {
    "raw_softmax": "Raw Softmax",
    "prototype": "Main prototype",
    "hierarchical": "Hierarchical prototype",
    "fused": "Fused",
}
METHOD_COLORS = {
    "raw_softmax": "#555D6B",
    "prototype": "#2E6F8E",
    "hierarchical": "#70865A",
    "fused": "#9A7A52",
}
METHOD_MARKERS = {
    "raw_softmax": "o",
    "prototype": "s",
    "hierarchical": "^",
    "fused": "D",
}

ENV_COLORS = {
    "DWASHING": "#2E6F8E",
    "TBUS": "#70865A",
    "STRAFFIC": "#9A6A5A",
}
ENV_MARKERS = {"DWASHING": "o", "TBUS": "s", "STRAFFIC": "^"}

TEXT = "#20242A"
MUTED = "#69717E"
GRID = "#D9DEE5"
PALE_BLUE = "#DDEAF0"
PALE_GREEN = "#E3EAD9"
PALE_RED = "#F1DEDA"
ACCENT_RED = "#A95449"
WHITE = "#FFFFFF"

FIGURE_WIDTH_MM = 183.0
MAIN_HEIGHTS_MM = {
    MAIN_FIGURE_STEMS[0]: 126.0,
    MAIN_FIGURE_STEMS[1]: 142.0,
    MAIN_FIGURE_STEMS[2]: 132.0,
    MAIN_FIGURE_STEMS[3]: 168.0,
    MAIN_FIGURE_STEMS[4]: 156.0,
}
SUPP_HEIGHTS_MM = {
    SUPPLEMENTARY_FIGURE_STEMS[0]: 146.0,
    SUPPLEMENTARY_FIGURE_STEMS[1]: 142.0,
    SUPPLEMENTARY_FIGURE_STEMS[2]: 224.0,
}


@dataclass(frozen=True)
class ExportRecord:
    figure: str
    stem: str
    paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate exactly five journal main figures and three supplementary "
            "figures from locked paper CSVs; no training or inference is run."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current working directory).",
    )
    parser.add_argument(
        "--main-out",
        type=Path,
        default=Path("paper/figures_journal"),
        help="Main-figure output directory under the project root.",
    )
    parser.add_argument(
        "--supp-out",
        type=Path,
        default=Path("paper/supplementary/figures"),
        help="Supplementary-figure output directory under the project root.",
    )
    parser.add_argument(
        "--source-out",
        type=Path,
        default=Path("paper/figure_source_data"),
        help="Figure source-map output directory under the project root.",
    )
    parser.add_argument(
        "--review-out",
        type=Path,
        default=Path("paper/review"),
        help="Audit output directory under the project root.",
    )
    parser.add_argument(
        "--skip-tiff",
        action="store_true",
        help="Skip the optional 600 dpi TIFF export.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate locked sources and output scope without writing files.",
    )
    return parser.parse_args()


def resolve_under(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def assert_output_scope(path: Path, approved_directories: Sequence[Path]) -> None:
    resolved = path.resolve()
    for directory in approved_directories:
        approved = directory.resolve()
        if resolved == approved or approved in resolved.parents:
            return
    raise ValueError(
        f"BLOCKER: output path is outside the approved output directories: {resolved}"
    )


def required_source_paths(root: Path) -> tuple[Path, ...]:
    relpaths = (
        "paper/tables/table1_dataset_and_leakage_free_protocol.csv",
        "paper/tables/table2_clean_baseline_and_architecture_ablations.csv",
        "paper/tables/table3_duration_comparison.csv",
        "paper/tables/table4_hierarchical_aux_weight_ablation.csv",
        "paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv",
        "paper/tables/table6_simulated_noise_robustness_by_stratum.csv",
        "paper/tables/table8_per_class_noise_results_and_confusion_directions.csv",
        "paper/appendix/clean_25_run_prototype_paired_stats.csv",
        "paper/appendix/clean_calibration_distribution.csv",
        "paper/appendix/noise_25_run_aurc_augrc_summary.csv",
        "paper/appendix/noise_25_run_cluster_bootstrap.csv",
        "paper/appendix/noise_25_run_fold_level_deltas.csv",
        "paper/appendix/noise_25_run_paired_stats.csv",
        "paper/appendix/noise_25_run_per_class_summary.csv",
        "paper/appendix/noise_25_run_selective_summary.csv",
        "paper/appendix/noise_25_run_summary.csv",
        "reports/prototype_noise_demand_w05_final/final_confusion_summary.csv",
        "reports/prototype_noise_demand_w05_final/final_runs.csv",
        "paper_results/tables/cv5_cap3x_logmel_dur2_vs_dur1_paired_stats.csv",
        "paper_results/tables/cv5_cap3x_hier_w02_vs_dur2_paired_stats.csv",
        "paper_results/tables/cv5_cap3x_hier_w05_vs_dur2_paired_stats.csv",
        "paper_results/tables/cv5_cap3x_hier_w10_vs_dur2_paired_stats.csv",
        "paper/CLAIMS_AND_EVIDENCE_v2.md",
        "paper/manuscript/paper_en_full_story_polished.md",
        "docs/NOISE_ROBUSTNESS_PROTOCOL.md",
        "tools/prototype_model_adapter.py",
    )
    return tuple((root / relpath).resolve() for relpath in relpaths)


def require_sources(root: Path) -> tuple[Path, ...]:
    sources = required_source_paths(root)
    missing = [path for path in sources if not path.is_file()]
    if missing:
        listing = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"BLOCKER: required locked source files are missing:\n{listing}")
    return sources


def read_csv(
    root: Path,
    relpath: str,
    required_columns: Iterable[str] = (),
) -> pd.DataFrame:
    path = (root / relpath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BLOCKER: required source file is missing: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"BLOCKER: {path} is missing required columns: {missing}")
    return frame


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_sources(paths: Iterable[Path]) -> dict[Path, str]:
    return {path.resolve(): file_sha256(path) for path in paths}


def format_p_value(value: float) -> str:
    value = float(value)
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"BLOCKER: invalid P value: {value}")
    return f"P={value:.6g}"


def parse_confusion_matrix(labels_text: str, matrix_text: str) -> np.ndarray:
    labels = [label.strip() for label in str(labels_text).split("|") if label.strip()]
    try:
        matrix = np.asarray(ast.literal_eval(str(matrix_text)), dtype=np.int64)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"BLOCKER: invalid confusion matrix: {matrix_text}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"BLOCKER: confusion matrix must be square, got {matrix.shape}")
    if matrix.shape[0] != len(labels):
        raise ValueError(
            "BLOCKER: confusion-matrix label count does not match matrix shape: "
            f"{len(labels)} labels for {matrix.shape}"
        )
    if (matrix < 0).any():
        raise ValueError("BLOCKER: confusion matrix contains negative counts")
    return matrix


def build_source_map_rows() -> list[dict[str, str]]:
    def row(
        figure: str,
        panel: str,
        source_csv: str,
        source_columns: str,
        transformation: str,
        filtering: str,
        statistic: str,
        output_file: str,
    ) -> dict[str, str]:
        return {
            "figure": figure,
            "panel": panel,
            "source_csv": source_csv,
            "source_columns": source_columns,
            "transformation": transformation,
            "filtering": filtering,
            "statistic": statistic,
            "output_file": output_file,
        }

    main = "paper/figures_journal"
    supp = "paper/supplementary/figures"
    rows = [
        row("Figure 1", "a", "paper/tables/table1_dataset_and_leakage_free_protocol.csv; paper/manuscript/paper_en_full_story_polished.md", "item; value; Methods/Data-source text", "Conceptual left-to-right reuse and 2 s preprocessing schematic; no experimental values", "Public/third-party source statement; clean classes; clean backbone protocol", "Not applicable", f"{main}/{MAIN_FIGURE_STEMS[0]}.svg"),
        row("Figure 1", "b", "paper/tables/table1_dataset_and_leakage_free_protocol.csv; tools/prototype_model_adapter.py", "item; value; DEFAULT_AUX_TO_MAIN", "Map four main labels to six auxiliary subtypes and depict parallel training heads", "Clean classes; auxiliary subtypes", "Not applicable", f"{main}/{MAIN_FIGURE_STEMS[0]}.svg"),
        row("Figure 1", "c", "paper/tables/table1_dataset_and_leakage_free_protocol.csv", "item; value", "Depict train-only main and subtype prototype construction with validation-only calibration", "Prototype protocol", "Not applicable", f"{main}/{MAIN_FIGURE_STEMS[0]}.svg"),
        row("Figure 1", "d", "paper/tables/table1_dataset_and_leakage_free_protocol.csv", "item; value", "Depict frozen clean and DEMAND simulated-additive-noise evaluation as separate evaluation conditions", "Noise protocol; external validation", "Not applicable", f"{main}/{MAIN_FIGURE_STEMS[0]}.svg"),
        row("Figure 2", "a", "paper/tables/table3_duration_comparison.csv", "duration; n; mean_macro_f1; std_macro_f1", "Order 1 s, 2 s, 3 s and plot mean with run-level SD", "All duration rows; no pairing of 3 s with 2 s", "Mean ± sample SD across fold-seed runs", f"{main}/{MAIN_FIGURE_STEMS[1]}.svg"),
        row("Figure 2", "b", "paper_results/tables/cv5_cap3x_logmel_dur2_vs_dur1_paired_stats.csv", "n; mean_delta; ci95_low; ci95_high; wilcoxon_p", "Plot matched 2 s minus 1 s mean delta and saved percentile-bootstrap CI", "Exact fold-seed matched pairs", "10,000-resample paired-delta bootstrap CI; two-sided Wilcoxon", f"{main}/{MAIN_FIGURE_STEMS[1]}.svg"),
        row("Figure 2", "c", "paper/tables/table2_clean_baseline_and_architecture_ablations.csv", "model; protocol; n; mean_macro_f1; std_macro_f1", "Select locked 2 s Log-Mel and five clean alternatives; plot mean with run-level SD", "cap3x rows: logmel_dur2, mctafd_no_se_gated, logmel_pcen, logmel_sg, logmel_specaug, logmel_dur2_attn", "Descriptive mean ± sample SD; unmatched n explicitly labelled", f"{main}/{MAIN_FIGURE_STEMS[1]}.svg"),
        row("Figure 3", "a", "paper/tables/table1_dataset_and_leakage_free_protocol.csv; tools/prototype_model_adapter.py", "item; value; DEFAULT_AUX_TO_MAIN", "Draw explicit subtype-to-main hierarchy", "Four main classes and six auxiliary subtypes", "Not applicable", f"{main}/{MAIN_FIGURE_STEMS[2]}.svg"),
        row("Figure 3", "b", "paper/tables/table4_hierarchical_aux_weight_ablation.csv; paper_results/tables/cv5_cap3x_hier_w02_vs_dur2_paired_stats.csv; paper_results/tables/cv5_cap3x_hier_w05_vs_dur2_paired_stats.csv; paper_results/tables/cv5_cap3x_hier_w10_vs_dur2_paired_stats.csv", "model; n; mean_macro_f1; std_macro_f1; ci95_low; ci95_high; wilcoxon_p", "Order lambda 0.2, 0.5, 1.0; plot mean ± SD and exact paired P values vs non-hierarchical 2 s baseline", "All three locked weights; exact 25 matched pairs", "Mean ± sample SD; two-sided Wilcoxon; paired-delta bootstrap CI in source", f"{main}/{MAIN_FIGURE_STEMS[2]}.svg"),
        row("Figure 3", "c", "paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv; paper/appendix/clean_25_run_prototype_paired_stats.csv; reports/prototype_noise_demand_w05_final/final_runs.csv", "method; n; mean_macro_f1; std_macro_f1; comparison; mean_delta; bootstrap_ci95_low; bootstrap_ci95_high; wilcoxon_p; fold; seed; lambda", "Select raw, main prototype, hierarchical prototype from the same lambda=0.5 checkpoint cohort", "method in raw_softmax, prototype, hierarchical; final_runs has 25 unique fold-seed rows and lambda=0.5", "Mean ± sample SD; paired bootstrap CI and two-sided Wilcoxon in source", f"{main}/{MAIN_FIGURE_STEMS[2]}.svg"),
        row("Figure 4", "a", "paper/tables/table6_simulated_noise_robustness_by_stratum.csv", "noise_environment; method; n; mean_macro_f1; std_macro_f1", "Plot three methods across moderate, extreme and all-noisy strata", "method in raw_softmax, prototype, hierarchical", "Mean ± sample SD across 25 fold-seed stratum means", f"{main}/{MAIN_FIGURE_STEMS[3]}.svg"),
        row("Figure 4", "b", "paper/appendix/noise_25_run_summary.csv", "noise_environment; snr_db; method; n; mean_macro_f1; std_macro_f1", "Plot clean/20/10/0 dB mean curves separately for DWASHING, TBUS and STRAFFIC", "method in raw_softmax, prototype, hierarchical; one clean row per method plus noise_environment in DWASHING, TBUS, STRAFFIC and snr_db in 20, 10, 0; exclude GROUP aggregate rows", "Mean across 25 fold-seed runs; no CI drawn", f"{main}/{MAIN_FIGURE_STEMS[3]}.svg"),
        row("Figure 4", "c", "paper/appendix/noise_25_run_paired_stats.csv; paper/appendix/noise_25_run_cluster_bootstrap.csv", "noise_environment; comparison; mean_delta; bootstrap_ci95_low; bootstrap_ci95_high; wilcoxon_p; fold_cluster_bootstrap_ci95_low; fold_cluster_bootstrap_ci95_high; wilcoxon_on_5_fold_means_low_power", "Align three comparisons by stratum and overlay run-level and fold-cluster intervals", "GROUP rows for ALL_NOISY, MODERATE_NOISE, EXTREME_STRESS", "Run-level paired bootstrap and stored unadjusted Wilcoxon P; five-fold cluster bootstrap and stored unadjusted low-power fold-mean Wilcoxon P", f"{main}/{MAIN_FIGURE_STEMS[3]}.svg"),
        row("Figure 5", "a", "paper/appendix/noise_25_run_per_class_summary.csv", "noise_environment; snr_db; method; class; recall", "Plot cough recall from clean through 20/10/0 dB for each DEMAND environment", "class=cough; method in raw_softmax, prototype, hierarchical; one clean row per method plus noise_environment in DWASHING, TBUS, STRAFFIC and snr_db in 20, 10, 0; exclude GROUP aggregate rows", "Pooled confusion-derived recall; no inferential test", f"{main}/{MAIN_FIGURE_STEMS[4]}.svg"),
        row("Figure 5", "b", "paper/tables/table8_per_class_noise_results_and_confusion_directions.csv", "method; feeding_to_stress; stress_to_feeding", "Plot pooled ALL_NOISY directional error counts", "ALL_NOISY/GROUP; method in raw_softmax, prototype, hierarchical", "Pooled decision counts, not independent-event counts", f"{main}/{MAIN_FIGURE_STEMS[4]}.svg"),
        row("Figure 5", "c", "paper/appendix/noise_25_run_aurc_augrc_summary.csv; paper/appendix/noise_25_run_selective_summary.csv", "method; n; mean_aurc; mean_augrc; mean_risk_at_coverage_0_95; mean_frozen_threshold_selective_risk", "Join authoritative ALL_NOISY ranking and selective summaries by method", "ALL_NOISY/GROUP; method in raw_softmax, prototype, hierarchical", "Descriptive 25-run means; lower is better; no paired inferential test", f"{main}/{MAIN_FIGURE_STEMS[4]}.svg"),
        row("Supplementary Figure S1", "a", "paper/tables/table2_clean_baseline_and_architecture_ablations.csv", "protocol; model; n; mean_macro_f1; std_macro_f1; min_macro_f1; max_macro_f1", "Plot all nine locked clean rows with min-max and mean ± SD", "No unlocked or archived-only rows added", "Descriptive mean ± sample SD and observed min-max", f"{supp}/{SUPPLEMENTARY_FIGURE_STEMS[0]}.svg"),
        row("Supplementary Figure S2", "a", "paper/appendix/noise_25_run_summary.csv", "noise_environment; snr_db; method; n; mean_macro_f1; std_macro_f1", "Environment-by-SNR curves", "method=raw_softmax; one clean row plus noise_environment in DWASHING, TBUS, STRAFFIC and snr_db in 20, 10, 0; exclude GROUP aggregate rows", "Mean ± sample SD across 25 runs", f"{supp}/{SUPPLEMENTARY_FIGURE_STEMS[1]}.svg"),
        row("Supplementary Figure S2", "b", "paper/appendix/noise_25_run_summary.csv", "noise_environment; snr_db; method; n; mean_macro_f1; std_macro_f1", "Environment-by-SNR curves", "method=prototype; one clean row plus noise_environment in DWASHING, TBUS, STRAFFIC and snr_db in 20, 10, 0; exclude GROUP aggregate rows", "Mean ± sample SD across 25 runs", f"{supp}/{SUPPLEMENTARY_FIGURE_STEMS[1]}.svg"),
        row("Supplementary Figure S2", "c", "paper/appendix/noise_25_run_summary.csv", "noise_environment; snr_db; method; n; mean_macro_f1; std_macro_f1", "Environment-by-SNR curves", "method=hierarchical; one clean row plus noise_environment in DWASHING, TBUS, STRAFFIC and snr_db in 20, 10, 0; exclude GROUP aggregate rows", "Mean ± sample SD across 25 runs", f"{supp}/{SUPPLEMENTARY_FIGURE_STEMS[1]}.svg"),
        row("Supplementary Figure S2", "d", "paper/appendix/noise_25_run_summary.csv", "noise_environment; snr_db; method; n; mean_macro_f1; std_macro_f1", "Environment-by-SNR curves", "method=fused; one clean row plus noise_environment in DWASHING, TBUS, STRAFFIC and snr_db in 20, 10, 0; exclude GROUP aggregate rows", "Mean ± sample SD across 25 runs", f"{supp}/{SUPPLEMENTARY_FIGURE_STEMS[1]}.svg"),
        row("Supplementary Figure S3", "a", "reports/prototype_noise_demand_w05_final/final_confusion_summary.csv", "noise_environment; method; labels; confusion_matrix", "Parse and row-normalize four complete ALL_NOISY confusion matrices; encode percentage by bubble area", "ALL_NOISY/GROUP; four methods", "Pooled counts over 25 fold-seed runs and nine simulated-noise conditions", f"{supp}/{SUPPLEMENTARY_FIGURE_STEMS[2]}.svg"),
        row("Supplementary Figure S3", "b", "paper/appendix/clean_calibration_distribution.csv", "parameter; value; count; n_runs", "Plot complete discrete calibration-selection frequencies", "Discrete parameters: softmax/main/hier temperatures, hierarchy penalty, fusion alpha/kind", "Counts across 25 validation-calibrated fold-seed runs", f"{supp}/{SUPPLEMENTARY_FIGURE_STEMS[2]}.svg"),
        row("Supplementary Figure S3", "c", "paper/appendix/clean_calibration_distribution.csv", "parameter; value; count; n_runs", "Expand count-weighted global and per-class frozen thresholds and plot distributions", "global and per-class rejection-threshold parameters", "Distribution across 25 validation-calibrated fold-seed runs", f"{supp}/{SUPPLEMENTARY_FIGURE_STEMS[2]}.svg"),
        row("Supplementary Figure S3", "d", "paper/appendix/noise_25_run_fold_level_deltas.csv", "noise_environment; fold; prototype_minus_raw_softmax; hierarchical_minus_raw_softmax; hierarchical_minus_prototype", "Plot all five fold means for all three grouped strata and comparisons", "Folds 0-4; ALL_NOISY, MODERATE_NOISE, EXTREME_STRESS", "Fold-mean paired deltas; descriptive sensitivity display", f"{supp}/{SUPPLEMENTARY_FIGURE_STEMS[2]}.svg"),
    ]
    return rows


def build_claim_rows() -> list[dict[str, str]]:
    rows = [
        ("Figure 1", "a", "The evaluated mainline computationally reuses public/third-party pig audio and applies 2 s Log-Mel input with a CRNN shared embedding.", "paper/tables/table1_dataset_and_leakage_free_protocol.csv; paper/manuscript/paper_en_full_story_polished.md", "protocol", "Do not claim a novel audio acquisition pipeline."),
        ("Figure 1", "b", "Four main classes and six auxiliary subtypes provide hierarchical supervision.", "paper/tables/table1_dataset_and_leakage_free_protocol.csv; tools/prototype_model_adapter.py", "protocol", "Do not invent an automatic routing mechanism."),
        ("Figure 1", "c", "Main and subtype prototypes are train-only artifacts with validation-only calibration.", "paper/tables/table1_dataset_and_leakage_free_protocol.csv", "strong boundary", "Do not imply test-set prototype construction or calibration."),
        ("Figure 1", "d", "DEMAND testing is simulated additive-noise evaluation.", "paper/tables/table1_dataset_and_leakage_free_protocol.csv", "strong boundary", "Do not call this real-farm external validation."),
        ("Figure 2", "a", "The available 2 s mean exceeds the 1 s and 3 s means.", "paper/tables/table3_duration_comparison.csv", "descriptive", "Do not claim 3 s is significantly worse than 2 s."),
        ("Figure 2", "b", "Matched 2 s input improves clean Macro-F1 over 1 s input.", "paper_results/tables/cv5_cap3x_logmel_dur2_vs_dur1_paired_stats.csv", "run-level statistical", "Do not generalize 2 s as universally optimal."),
        ("Figure 2", "c", "Locked complex clean variants did not replace the 2 s Log-Mel mainline.", "paper/tables/table2_clean_baseline_and_architecture_ablations.csv", "descriptive unmatched", "Do not present unmatched n=15 versus n=25 rows as paired tests."),
        ("Figure 3", "a", "Auxiliary labels refine cough and stress categories while calm-grunt and feeding map one-to-one.", "paper/tables/table1_dataset_and_leakage_free_protocol.csv; tools/prototype_model_adapter.py", "protocol", "Do not infer unmeasured biological hierarchy."),
        ("Figure 3", "b", "Lambda=1.0 has the highest mean and lambda=0.5 the lowest SD.", "paper/tables/table4_hierarchical_aux_weight_ablation.csv", "descriptive bounded", "Do not claim a significant incremental clean gain."),
        ("Figure 3", "c", "Within the same lambda=0.5 fold-seed checkpoint cohort, clean hierarchical prototypes have the highest mean among the three displayed inference modes.", "paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv; paper/appendix/clean_25_run_prototype_paired_stats.csv; reports/prototype_noise_demand_w05_final/final_runs.csv", "descriptive bounded", "Do not call the clean gain statistically significant or compare raw Softmax here with the non-hierarchical 2 s baseline."),
        ("Figure 4", "a", "Main prototypes have the highest mean in all three grouped simulated-noise strata.", "paper/tables/table6_simulated_noise_robustness_by_stratum.csv", "descriptive", "Do not claim a large or universal robustness gain."),
        ("Figure 4", "b", "Performance degrades toward 0 dB and depends on DEMAND environment.", "paper/appendix/noise_25_run_summary.csv", "descriptive", "Do not describe 0 dB as no noise."),
        ("Figure 4", "c", "Moderate-noise support is strongest; fold clustering weakens some run-level inferences.", "paper/appendix/noise_25_run_paired_stats.csv; paper/appendix/noise_25_run_cluster_bootstrap.csv", "sensitivity-aware", "Do not report run-level significance without the fold-cluster caveat or claim hierarchical prototypes are most robust."),
        ("Figure 5", "a", "Cough recall collapses under severe simulated noise.", "paper/appendix/noise_25_run_per_class_summary.csv", "strong descriptive", "Do not claim reliable cough recognition at 0 dB."),
        ("Figure 5", "b", "Feeding-to-stress and stress-to-feeding errors remain method-dependent failure directions.", "paper/tables/table8_per_class_noise_results_and_confusion_directions.csv", "descriptive pooled", "Do not treat pooled decisions as independent events or infer a causal noise mechanism."),
        ("Figure 5", "c", "Raw Softmax ranks best on AURC/AUGRC while main prototypes show only small fixed-operating-point risk advantages.", "paper/appendix/noise_25_run_aurc_augrc_summary.csv; paper/appendix/noise_25_run_selective_summary.csv", "descriptive bounded", "Do not claim universal uncertainty improvement."),
        ("Supplementary Figure S1", "a", "The complete locked clean ablation table retains the 2 s Log-Mel mean as the strongest listed mainline.", "paper/tables/table2_clean_baseline_and_architecture_ablations.csv", "descriptive unmatched", "Do not add archived-only BiLSTM or other unlocked values."),
        ("Supplementary Figure S2", "a", "Raw Softmax varies by DEMAND environment and SNR.", "paper/appendix/noise_25_run_summary.csv", "descriptive", "Do not generalize simulated noise to farm deployment."),
        ("Supplementary Figure S2", "b", "Main-prototype performance varies by DEMAND environment and SNR.", "paper/appendix/noise_25_run_summary.csv", "descriptive", "Do not claim every environment/SNR comparison is significant."),
        ("Supplementary Figure S2", "c", "Hierarchical-prototype performance varies by DEMAND environment and SNR.", "paper/appendix/noise_25_run_summary.csv", "descriptive", "Do not claim hierarchical prototypes dominate under noise."),
        ("Supplementary Figure S2", "d", "Fused inference is retained as a supplementary ablation.", "paper/appendix/noise_25_run_summary.csv", "descriptive ablation", "Do not present fused inference as the main contribution."),
        ("Supplementary Figure S3", "a", "Complete pooled confusion matrices expose class-specific failure structure.", "reports/prototype_noise_demand_w05_final/final_confusion_summary.csv", "descriptive pooled", "Do not treat pooled seed decisions as independent recordings."),
        ("Supplementary Figure S3", "b", "Validation-selected calibration settings vary across fold-seed runs.", "paper/appendix/clean_calibration_distribution.csv", "descriptive validation-only", "Do not imply test-set calibration."),
        ("Supplementary Figure S3", "c", "Frozen rejection thresholds vary by class across fold-seed runs.", "paper/appendix/clean_calibration_distribution.csv", "descriptive validation-only", "Do not claim universal calibration improvement."),
        ("Supplementary Figure S3", "d", "Fold-level deltas reveal heterogeneity hidden by run-level averaging.", "paper/appendix/noise_25_run_fold_level_deltas.csv", "sensitivity display", "Do not mix fold-level points with 25 run-level replicates."),
    ]
    return [
        {
            "figure": figure,
            "panel": panel,
            "supported_claim": claim,
            "evidence_file": evidence,
            "claim_strength": strength,
            "prohibited_claim": prohibited,
        }
        for figure, panel, claim, evidence, strength, prohibited in rows
    ]


def validate_locked_sources(root: Path) -> None:
    require_sources(root)
    duration = read_csv(
        root,
        "paper/tables/table3_duration_comparison.csv",
        ("duration", "n", "mean_macro_f1", "std_macro_f1"),
    )
    if set(duration["duration"].astype(str)) != {"1 s", "2 s", "3 s"}:
        raise ValueError("BLOCKER: duration table does not contain exactly 1 s, 2 s and 3 s")
    expected_n = dict(zip(duration["duration"], duration["n"].astype(int)))
    if expected_n != {"2 s": 25, "3 s": 15, "1 s": 25}:
        raise ValueError(f"BLOCKER: unexpected duration run counts: {expected_n}")

    table4 = read_csv(
        root,
        "paper/tables/table4_hierarchical_aux_weight_ablation.csv",
        ("model", "n", "mean_macro_f1", "std_macro_f1"),
    )
    if set(table4["model"]) != {
        "logmel_dur2_hier_w02",
        "logmel_dur2_hier_w05",
        "logmel_dur2_hier_w10",
    }:
        raise ValueError("BLOCKER: hierarchical-weight table is incomplete or changed")

    table5 = read_csv(
        root,
        "paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv",
        ("method", "n", "mean_macro_f1", "std_macro_f1"),
    )
    if not set(METHOD_ORDER).issubset(set(table5["method"])):
        raise ValueError("BLOCKER: clean prototype table is missing a required method")

    final_runs = read_csv(
        root,
        "reports/prototype_noise_demand_w05_final/final_runs.csv",
        ("fold", "seed", "lambda"),
    )
    if len(final_runs) != 25 or final_runs.duplicated(["fold", "seed"]).any():
        raise ValueError(
            "BLOCKER: lambda=0.5 prototype cohort must contain 25 unique fold-seed rows"
        )
    if set(final_runs["fold"].astype(int)) != set(range(5)):
        raise ValueError("BLOCKER: lambda=0.5 prototype cohort is missing one or more folds")
    if not np.allclose(final_runs["lambda"].astype(float).to_numpy(), 0.5):
        raise ValueError("BLOCKER: prototype cohort contains a non-0.5 auxiliary-loss weight")

    table6 = read_csv(
        root,
        "paper/tables/table6_simulated_noise_robustness_by_stratum.csv",
        ("noise_environment", "method", "n", "mean_macro_f1", "std_macro_f1"),
    )
    expected_strata = {"MODERATE_NOISE", "EXTREME_STRESS", "ALL_NOISY"}
    if not expected_strata.issubset(set(table6["noise_environment"])):
        raise ValueError("BLOCKER: simulated-noise stratum table is incomplete")

    summary = read_csv(
        root,
        "paper/appendix/noise_25_run_summary.csv",
        ("noise_environment", "snr_db", "method", "n", "mean_macro_f1"),
    )
    numeric = summary[pd.to_numeric(summary["snr_db"], errors="coerce").notna()]
    if set(numeric["noise_environment"]) != {"DWASHING", "TBUS", "STRAFFIC"}:
        raise ValueError("BLOCKER: DEMAND environment set is incomplete or changed")

    claims = (root / "paper/CLAIMS_AND_EVIDENCE_v2.md").read_text(encoding="utf-8")
    boundary_phrases = (
        "simulated additive noise",
        "not real-farm external validation",
        "0 dB active-event SNR",
        "not statistically significant",
        "do not comprehensively improve uncertainty",
    )
    missing_boundaries = [phrase for phrase in boundary_phrases if phrase.lower() not in claims.lower()]
    if missing_boundaries:
        raise ValueError(f"BLOCKER: claim ledger is missing required boundaries: {missing_boundaries}")


def configure_style() -> str:
    simsun_path = Path("C:/Windows/Fonts/simsun.ttc")
    if not simsun_path.is_file():
        raise FileNotFoundError(
            "BLOCKER: SimSun is required for this Chinese Word figure package but "
            f"was not found at {simsun_path}"
        )
    font_manager.fontManager.addfont(str(simsun_path))
    simsun_name = font_manager.FontProperties(fname=str(simsun_path)).get_name()
    if simsun_name != "SimSun":
        raise RuntimeError(f"BLOCKER: expected SimSun, resolved {simsun_name!r}")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["SimSun", "Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#4B515A",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.unicode_minus": False,
        }
    )
    return simsun_name


def figure_size(height_mm: float) -> tuple[float, float]:
    return FIGURE_WIDTH_MM / 25.4, height_mm / 25.4


def add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.11,
    y: float = 1.03,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )


def clean_axis(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.65, length=2.4, color="#4B515A")
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)


def probability_axis(ax: plt.Axes, ylabel: str = "Macro-F1") -> None:
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(ylabel)
    clean_axis(ax, grid=True)


def method_handles(methods: Sequence[str] = METHOD_ORDER) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=1.2,
            markersize=4.2,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]


def draw_rect(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = WHITE,
    edgecolor: str = "#68717D",
    fontsize: float = 6.5,
    fontweight: str = "normal",
) -> Rectangle:
    patch = Rectangle(
        xy,
        width,
        height,
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.85,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=TEXT,
        fontsize=fontsize,
        fontweight=fontweight,
        linespacing=1.15,
    )
    return patch


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#68717D",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=0.85,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1.2,
            shrinkB=1.2,
        )
    )


def annotate_value(ax: plt.Axes, x: float, y: float, value: float, *, dy: float = 0.025) -> None:
    ax.text(x, min(y + dy, 0.985), f"{value:.4f}", ha="center", va="bottom", fontsize=5.8, color=TEXT)


def text_color_for_fill(color: str) -> str:
    value = color.lstrip("#")
    if len(value) != 6:
        return TEXT
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return WHITE if luminance < 145 else TEXT


def plot_vertical_mean_sd(
    ax: plt.Axes,
    labels: Sequence[str],
    means: Sequence[float],
    sds: Sequence[float],
    ns: Sequence[int],
    colors: Sequence[str],
    *,
    ylabel: str = "Macro-F1",
    hatches: Sequence[str] | None = None,
) -> None:
    x = np.arange(len(labels), dtype=float)
    bars = ax.bar(
        x,
        means,
        width=0.58,
        color=colors,
        edgecolor="#3F4650",
        linewidth=0.55,
        yerr=sds,
        error_kw={"elinewidth": 0.75, "capsize": 2.4, "capthick": 0.75, "ecolor": "#30343A"},
        zorder=2,
    )
    if hatches:
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    probability_axis(ax, ylabel)
    for xpos, mean, n, color in zip(x, means, ns, colors):
        annotate_value(ax, xpos, mean, mean)
        ax.text(xpos, 0.035, f"n={int(n)}", ha="center", va="bottom", fontsize=5.5, color=text_color_for_fill(color))


def model_label(model: str) -> str:
    labels = {
        "logmel_dur2": "2 s Log-Mel",
        "mctafd_no_se_gated": "MCTAFD",
        "logmel_pcen": "PCEN",
        "logmel_sg": "Spectral gate",
        "logmel_specaug": "SpecAugment",
        "logmel_dur2_attn": "Attention pooling",
        "logmel": "1 s Log-Mel",
    }
    return labels.get(model, model)


def lambda_value(model: str) -> float:
    mapping = {
        "logmel_dur2_hier_w02": 0.2,
        "logmel_dur2_hier_w05": 0.5,
        "logmel_dur2_hier_w10": 1.0,
    }
    if model not in mapping:
        raise ValueError(f"BLOCKER: unknown hierarchical model tag: {model}")
    return mapping[model]


def build_figure1(root: Path) -> plt.Figure:
    table1 = read_csv(
        root,
        "paper/tables/table1_dataset_and_leakage_free_protocol.csv",
        ("item", "value"),
    )
    items = dict(zip(table1["item"].astype(str), table1["value"].astype(str)))
    if "Not performed" not in items.get("External validation", ""):
        raise ValueError("BLOCKER: external-validation boundary changed in Table 1")

    fig, ax = plt.subplots(figsize=figure_size(MAIN_HEIGHTS_MM[MAIN_FIGURE_STEMS[0]]))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.97, bottom=0.055)
    ax.set_axis_off()

    ax.text(0.012, 0.95, "a", transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=TEXT)
    boxes = [
        ((0.035, 0.73), 0.16, 0.14, "Public / third-party\npig audio", PALE_BLUE),
        ((0.235, 0.73), 0.16, 0.14, "2 s waveform", WHITE),
        ((0.435, 0.73), 0.16, 0.14, "Log-Mel feature", PALE_GREEN),
        ((0.635, 0.73), 0.16, 0.14, "CRNN shared\nembedding", "#E5E2EE"),
    ]
    for xy, width, height, label, face in boxes:
        draw_rect(ax, xy, width, height, label, facecolor=face, fontweight="bold" if "CRNN" in label else "normal")
    for start, end in [((0.195, 0.80), (0.235, 0.80)), ((0.395, 0.80), (0.435, 0.80)), ((0.595, 0.80), (0.635, 0.80))]:
        draw_arrow(ax, start, end)

    wave_x = np.linspace(0.255, 0.375, 100)
    envelope = np.exp(-((wave_x - 0.315) / 0.036) ** 2)
    wave_y = 0.748 + 0.018 * envelope * np.sin(np.linspace(0, 10 * np.pi, wave_x.size))
    ax.plot(wave_x, wave_y, transform=ax.transAxes, color=METHOD_COLORS["raw_softmax"], linewidth=0.75, clip_on=False)
    for idx, color in enumerate(("#D8E7DF", "#BFD8CE", "#9DC1B6")):
        ax.add_patch(
            Rectangle(
                (0.46 + 0.035 * idx, 0.745),
                0.028,
                0.034 + 0.012 * idx,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
            )
        )

    ax.text(0.012, 0.63, "b", transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=TEXT)
    draw_rect(ax, (0.035, 0.42), 0.25, 0.16, "Four-class main head\ncough · calm_grunt\nfeeding · stress_vocal", facecolor=PALE_BLUE)
    draw_rect(
        ax,
        (0.32, 0.39),
        0.31,
        0.22,
        "Six-subtype auxiliary head\ndry_cough · abdominal_cough\ncalm_grunt · feeding\nfrightened_stress · anxious_stress",
        facecolor=PALE_GREEN,
    )
    draw_arrow(ax, (0.715, 0.73), (0.16, 0.58), connectionstyle="arc3,rad=0.12")
    draw_arrow(ax, (0.735, 0.73), (0.48, 0.61), connectionstyle="arc3,rad=0.08")
    ax.text(0.36, 0.65, "training supervision", transform=ax.transAxes, ha="center", fontsize=5.8, color=MUTED)

    ax.text(0.655, 0.63, "c", transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=TEXT)
    draw_rect(ax, (0.68, 0.50), 0.14, 0.10, "Train-only\nmain prototypes", facecolor=PALE_BLUE, fontsize=6.0)
    draw_rect(ax, (0.68, 0.34), 0.14, 0.10, "Train-only\nsubtype prototypes", facecolor=PALE_GREEN, fontsize=6.0)
    draw_rect(ax, (0.835, 0.36), 0.15, 0.21, "Frozen outputs\nRaw Softmax (main head)\nMain prototype\nHierarchical\nprototype", facecolor=WHITE, fontsize=5.1)
    draw_arrow(ax, (0.715, 0.73), (0.75, 0.60))
    ax.plot(
        [0.715, 0.64, 0.64],
        [0.73, 0.65, 0.39],
        transform=ax.transAxes,
        color="#68717D",
        linewidth=0.85,
    )
    draw_arrow(ax, (0.64, 0.39), (0.68, 0.39))
    draw_arrow(ax, (0.82, 0.55), (0.835, 0.51))
    draw_arrow(ax, (0.82, 0.39), (0.835, 0.43))
    ax.text(0.705, 0.64, "train embeddings only", transform=ax.transAxes, fontsize=5.4, color=MUTED)
    ax.text(0.68, 0.295, "Calibration: validation only", transform=ax.transAxes, fontsize=5.6, color=MUTED)

    ax.text(0.012, 0.25, "d", transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=TEXT)
    draw_rect(ax, (0.48, 0.08), 0.20, 0.13, "Clean evaluation\nfrozen test only", facecolor="#EEF1F4", fontweight="bold")
    draw_rect(ax, (0.73, 0.08), 0.25, 0.13, "DEMAND simulated additive noise\nDWASHING · TBUS · STRAFFIC\n20 / 10 / 0 dB active-event SNR", facecolor=PALE_RED, fontsize=6.0, fontweight="bold")
    draw_arrow(ax, (0.90, 0.36), (0.61, 0.21), connectionstyle="arc3,rad=-0.08")
    draw_arrow(ax, (0.94, 0.36), (0.855, 0.21), connectionstyle="arc3,rad=0.04")
    ax.text(
        0.855,
        0.018,
        "Simulated additive noise ≠\nreal-farm external validation",
        transform=ax.transAxes,
        fontsize=5.7,
        color=ACCENT_RED,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax.text(
        0.035,
        0.12,
        "Parallel inference methods are evaluated independently;\nno automatic routing mechanism is used.",
        transform=ax.transAxes,
        fontsize=6.0,
        color=MUTED,
        va="center",
    )
    return fig


def build_figure2(root: Path) -> plt.Figure:
    duration = read_csv(
        root,
        "paper/tables/table3_duration_comparison.csv",
        ("duration", "n", "mean_macro_f1", "std_macro_f1"),
    ).set_index("duration").loc[["1 s", "2 s", "3 s"]]
    paired = read_csv(
        root,
        "paper_results/tables/cv5_cap3x_logmel_dur2_vs_dur1_paired_stats.csv",
        ("n", "mean_delta", "ci95_low", "ci95_high", "wilcoxon_p"),
    ).iloc[0]
    ablations = read_csv(
        root,
        "paper/tables/table2_clean_baseline_and_architecture_ablations.csv",
        ("protocol", "model", "n", "mean_macro_f1", "std_macro_f1"),
    )
    order = [
        "logmel_dur2",
        "mctafd_no_se_gated",
        "logmel_pcen",
        "logmel_sg",
        "logmel_specaug",
        "logmel_dur2_attn",
    ]
    selected = ablations[(ablations["protocol"] == "cap3x") & ablations["model"].isin(order)].copy()
    selected["order"] = selected["model"].map({model: idx for idx, model in enumerate(order)})
    selected = selected.sort_values("order")
    if selected["model"].tolist() != order:
        raise ValueError("BLOCKER: locked clean-ablation rows are incomplete")

    fig = plt.figure(figsize=figure_size(MAIN_HEIGHTS_MM[MAIN_FIGURE_STEMS[1]]), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.28], width_ratios=[1.1, 0.9])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    duration_means = duration["mean_macro_f1"].astype(float).to_numpy()
    plot_vertical_mean_sd(
        ax_a,
        duration.index.tolist(),
        duration_means,
        duration["std_macro_f1"].astype(float).to_numpy(),
        duration["n"].astype(int).to_numpy(),
        ["#AEB7C3", METHOD_COLORS["prototype"], "#93A68B"],
        hatches=["//", "", ".."],
    )
    ax_a.set_title("Context duration", loc="left", fontweight="bold")
    add_panel_label(ax_a, "a")

    estimate = float(paired["mean_delta"])
    low = float(paired["ci95_low"])
    high = float(paired["ci95_high"])
    ax_b.axvline(0.0, color=MUTED, linestyle="--", linewidth=0.8)
    ax_b.plot([low, high], [0, 0], color=METHOD_COLORS["prototype"], linewidth=2.0)
    ax_b.plot(estimate, 0, marker="o", color=METHOD_COLORS["prototype"], markersize=5.0)
    ax_b.set_xlim(min(-0.006, low - 0.004), high + 0.008)
    ax_b.set_ylim(-0.7, 0.7)
    ax_b.set_yticks([0])
    ax_b.set_yticklabels(["2 s - 1 s"])
    ax_b.set_xlabel("Paired Macro-F1 difference")
    ax_b.set_title("Matched duration effect", loc="left", fontweight="bold")
    ax_b.text(estimate, 0.23, f"Δ={estimate:.4f}", ha="center", fontsize=6.2, color=TEXT)
    ax_b.text(0.02, 0.08, f"95% bootstrap CI [{low:.4f}, {high:.4f}]\n{format_p_value(float(paired['wilcoxon_p']))}; n={int(paired['n'])}", transform=ax_b.transAxes, fontsize=5.8, color=MUTED, va="bottom")
    clean_axis(ax_b)
    add_panel_label(ax_b, "b")

    y = np.arange(len(selected))[::-1]
    means = selected["mean_macro_f1"].astype(float).to_numpy()
    sds = selected["std_macro_f1"].astype(float).to_numpy()
    colors = [METHOD_COLORS["prototype"]] + ["#AEB7C3"] * (len(selected) - 1)
    bars = ax_c.barh(
        y,
        means,
        xerr=sds,
        height=0.55,
        color=colors,
        edgecolor="#3F4650",
        linewidth=0.5,
        error_kw={"elinewidth": 0.7, "capsize": 2.2, "capthick": 0.7, "ecolor": "#30343A"},
    )
    for idx, bar in enumerate(bars):
        if idx > 0:
            bar.set_hatch("//" if idx % 2 else "..")
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_c.set_xlabel("Macro-F1")
    ns = selected["n"].astype(int).to_numpy()
    stds = selected["std_macro_f1"].astype(float).to_numpy()
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(
        [
            f"{model_label(model)} (n={n})"
            for model, n in zip(selected["model"], ns)
        ]
    )
    ax_c.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax_c.set_axisbelow(True)
    ax_c.set_title("Locked clean front-end / pooling comparisons", loc="left", fontweight="bold")
    for ypos, mean, std in zip(y, means, stds):
        ax_c.text(mean + std + 0.005, ypos, f"{mean:.4f}", va="center", fontsize=5.8, color=TEXT)
    ax_c.text(0.01, -0.18, "Descriptive mean ± run-level SD; n=15 rows are not treated as paired with n=25.", transform=ax_c.transAxes, fontsize=5.7, color=MUTED)
    clean_axis(ax_c)
    add_panel_label(ax_c, "c", x=-0.07)
    return fig


def build_figure3(root: Path) -> plt.Figure:
    weights = read_csv(
        root,
        "paper/tables/table4_hierarchical_aux_weight_ablation.csv",
        ("model", "n", "mean_macro_f1", "std_macro_f1"),
    ).copy()
    weights["lambda"] = weights["model"].map(lambda_value)
    weights = weights.sort_values("lambda")
    paired_paths = {
        0.2: "paper_results/tables/cv5_cap3x_hier_w02_vs_dur2_paired_stats.csv",
        0.5: "paper_results/tables/cv5_cap3x_hier_w05_vs_dur2_paired_stats.csv",
        1.0: "paper_results/tables/cv5_cap3x_hier_w10_vs_dur2_paired_stats.csv",
    }
    p_by_weight = {
        weight: float(read_csv(root, path, ("wilcoxon_p",)).iloc[0]["wilcoxon_p"])
        for weight, path in paired_paths.items()
    }
    clean = read_csv(
        root,
        "paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv",
        ("method", "n", "mean_macro_f1", "std_macro_f1"),
    ).set_index("method").loc[list(METHOD_ORDER)]
    paired_clean = read_csv(
        root,
        "paper/appendix/clean_25_run_prototype_paired_stats.csv",
        ("comparison", "wilcoxon_p"),
    )
    clean_p = dict(zip(paired_clean["comparison"].astype(str), paired_clean["wilcoxon_p"].astype(float)))

    fig = plt.figure(figsize=figure_size(MAIN_HEIGHTS_MM[MAIN_FIGURE_STEMS[2]]), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    ax_a.set_axis_off()
    add_panel_label(ax_a, "a", x=-0.03, y=1.01)
    ax_a.text(0.04, 0.96, "Main classes", transform=ax_a.transAxes, fontsize=7.0, fontweight="bold", color=TEXT)
    ax_a.text(0.58, 0.96, "Auxiliary subtypes", transform=ax_a.transAxes, fontsize=7.0, fontweight="bold", color=TEXT)
    main_y = {"cough": 0.78, "calm_grunt": 0.57, "feeding": 0.36, "stress_vocal": 0.15}
    main_face = {"cough": PALE_BLUE, "calm_grunt": "#E9E7F1", "feeding": PALE_GREEN, "stress_vocal": PALE_RED}
    subtype_y = {
        "dry_cough": 0.85,
        "abdominal_cough": 0.71,
        "calm_grunt": 0.57,
        "feeding": 0.36,
        "frightened_stress": 0.22,
        "anxious_stress": 0.08,
    }
    for label, yval in main_y.items():
        draw_rect(ax_a, (0.04, yval - 0.055), 0.31, 0.11, label, facecolor=main_face[label], fontsize=6.4, fontweight="bold")
    for subtype, yval in subtype_y.items():
        parent = AUX_TO_MAIN[subtype]
        draw_rect(ax_a, (0.60, yval - 0.047), 0.35, 0.094, subtype, facecolor=main_face[parent], fontsize=5.8)
        draw_arrow(ax_a, (0.35, main_y[parent]), (0.60, yval), color="#7A818B")
    ax_a.text(0.04, 0.005, "Auxiliary supervision is used during training; the main task remains four-class classification.", transform=ax_a.transAxes, fontsize=5.6, color=MUTED)

    labels = [
        f"λ={value:g}\nvs 2 s {format_p_value(p_by_weight[float(value)])}"
        for value in weights["lambda"]
    ]
    plot_vertical_mean_sd(
        ax_b,
        labels,
        weights["mean_macro_f1"].astype(float).to_numpy(),
        weights["std_macro_f1"].astype(float).to_numpy(),
        weights["n"].astype(int).to_numpy(),
        ["#AFC4D0", "#7899AA", METHOD_COLORS["prototype"]],
        hatches=["//", "..", ""],
    )
    ax_b.text(1, 0.69, "lowest SD\nbalanced", ha="center", fontsize=5.4, color=WHITE)
    ax_b.text(2, 0.69, "highest mean", ha="center", fontsize=5.4, color=WHITE)
    ax_b.set_title(
        "Auxiliary-loss weight\nNo significant incremental clean gain",
        loc="left",
        fontweight="bold",
    )
    add_panel_label(ax_b, "b")

    raw_main = clean_p.get("prototype - raw_softmax", clean_p.get("main_prototype - raw_softmax"))
    raw_hier = clean_p.get("hierarchical - raw_softmax")
    if raw_main is None or raw_hier is None:
        raise ValueError("BLOCKER: clean prototype paired comparisons are missing")
    plot_vertical_mean_sd(
        ax_c,
        [
            "Raw\nSoftmax",
            f"Main prototype\nvs raw {format_p_value(float(raw_main))}",
            f"Hierarchical prototype\nvs raw {format_p_value(float(raw_hier))}",
        ],
        clean["mean_macro_f1"].astype(float).to_numpy(),
        clean["std_macro_f1"].astype(float).to_numpy(),
        clean["n"].astype(int).to_numpy(),
        [METHOD_COLORS[method] for method in METHOD_ORDER],
        hatches=["//", "", ".."],
    )
    ax_c.set_title("Clean post-hoc prototype inference\nSame λ=0.5 cohort; no significant paired gain", loc="left", fontweight="bold")
    add_panel_label(ax_c, "c")
    return fig


def build_figure4(root: Path) -> plt.Figure:
    strata = read_csv(
        root,
        "paper/tables/table6_simulated_noise_robustness_by_stratum.csv",
        ("noise_environment", "method", "n", "mean_macro_f1", "std_macro_f1"),
    )
    summary = read_csv(
        root,
        "paper/appendix/noise_25_run_summary.csv",
        ("noise_environment", "snr_db", "method", "n", "mean_macro_f1", "std_macro_f1"),
    )
    paired = read_csv(
        root,
        "paper/appendix/noise_25_run_paired_stats.csv",
        ("noise_environment", "snr_db", "comparison", "mean_delta", "bootstrap_ci95_low", "bootstrap_ci95_high", "wilcoxon_p"),
    )
    clustered = read_csv(
        root,
        "paper/appendix/noise_25_run_cluster_bootstrap.csv",
        ("noise_environment", "comparison", "fold_cluster_bootstrap_ci95_low", "fold_cluster_bootstrap_ci95_high", "wilcoxon_on_5_fold_means_low_power"),
    )

    fig = plt.figure(figsize=figure_size(MAIN_HEIGHTS_MM[MAIN_FIGURE_STEMS[3]]), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.18], width_ratios=[0.95, 1.35])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    c_grid = gs[1, :].subgridspec(1, 3, wspace=0.18)
    c_axes = [fig.add_subplot(c_grid[0, idx]) for idx in range(3)]

    stratum_order = ["MODERATE_NOISE", "EXTREME_STRESS", "ALL_NOISY"]
    stratum_labels = ["Moderate", "Extreme 0 dB", "All noisy"]
    x = np.arange(len(stratum_order), dtype=float)
    width = 0.22
    for method_idx, method in enumerate(METHOD_ORDER):
        method_rows = strata[strata["method"] == method].set_index("noise_environment").loc[stratum_order]
        offset = (method_idx - 1) * width
        bars = ax_a.bar(
            x + offset,
            method_rows["mean_macro_f1"].astype(float),
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#3F4650",
            linewidth=0.45,
            yerr=method_rows["std_macro_f1"].astype(float),
            error_kw={"elinewidth": 0.65, "capsize": 1.8, "capthick": 0.65, "ecolor": "#30343A"},
            label=METHOD_LABELS[method],
        )
        for bar in bars:
            if method == "raw_softmax":
                bar.set_hatch("//")
            elif method == "hierarchical":
                bar.set_hatch("..")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(stratum_labels)
    probability_axis(ax_a)
    ax_a.set_title("Grouped simulated-noise strata (n=25)", loc="left", fontweight="bold")
    ax_a.text(
        0.98,
        0.96,
        "Main prototype highest in all strata\nModerate-noise evidence is strongest",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=MUTED,
    )
    add_panel_label(ax_a, "a")

    snr_order = ["clean", "20", "10", "0"]
    environments = ["DWASHING", "TBUS", "STRAFFIC"]
    group_starts = {environment: idx * 5 for idx, environment in enumerate(environments)}
    for environment in environments:
        snr_x = group_starts[environment] + np.arange(len(snr_order))
        for method in METHOD_ORDER:
            clean_row = summary[(summary["noise_environment"].astype(str).str.lower() == "clean") & (summary["method"] == method)]
            if len(clean_row) != 1:
                raise ValueError(f"BLOCKER: expected one clean summary row for {method}")
            values = [float(clean_row.iloc[0]["mean_macro_f1"])]
            for snr in (20.0, 10.0, 0.0):
                rows = summary[
                    (summary["noise_environment"] == environment)
                    & (pd.to_numeric(summary["snr_db"], errors="coerce") == snr)
                    & (summary["method"] == method)
                ]
                if len(rows) != 1:
                    raise ValueError(f"BLOCKER: missing {environment}/{snr}/{method} summary row")
                values.append(float(rows.iloc[0]["mean_macro_f1"]))
            ax_b.plot(
                snr_x,
                values,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=1.0,
                markersize=3.2,
            )
        ax_b.text(group_starts[environment] + 1.5, 0.94, environment, ha="center", va="top", fontsize=6.1, fontweight="bold", color=MUTED)
    for separator in (4.0, 9.0):
        ax_b.axvline(separator, color=GRID, linewidth=0.7)
    all_ticks = [group_starts[environment] + offset for environment in environments for offset in range(4)]
    ax_b.set_xticks(all_ticks)
    ax_b.set_xticklabels(snr_order * len(environments))
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_ylabel("Macro-F1")
    ax_b.set_xlabel("Active-event SNR (dB), repeated within DEMAND environment")
    ax_b.set_title("Clean-to-noise trajectories (n=25 means)", loc="left", fontweight="bold")
    clean_axis(ax_b, grid=True)
    add_panel_label(ax_b, "b")
    ax_b.legend(handles=method_handles(), loc="lower left", ncol=3, columnspacing=0.9, handletextpad=0.35, fontsize=5.3)

    comparison_order = [
        "prototype - raw_softmax",
        "hierarchical - raw_softmax",
        "hierarchical - prototype",
    ]
    comparison_labels = ["Main - raw", "Hier. - raw", "Hier. - main"]
    for ax, stratum, title in zip(c_axes, stratum_order, stratum_labels):
        run_rows = paired[
            (paired["noise_environment"] == stratum)
            & (paired["snr_db"].astype(str).str.upper() == "GROUP")
        ].set_index("comparison")
        fold_rows = clustered[clustered["noise_environment"] == stratum].set_index("comparison")
        if not set(comparison_order).issubset(run_rows.index) or not set(comparison_order).issubset(fold_rows.index):
            raise ValueError(f"BLOCKER: paired comparison rows are incomplete for {stratum}")
        y = np.arange(3)[::-1]
        for ypos, comparison in zip(y, comparison_order):
            run = run_rows.loc[comparison]
            fold = fold_rows.loc[comparison]
            ax.plot(
                [float(fold["fold_cluster_bootstrap_ci95_low"]), float(fold["fold_cluster_bootstrap_ci95_high"])],
                [ypos + 0.10, ypos + 0.10],
                color="#8A9099",
                linewidth=1.0,
            )
            ax.plot(float(run["mean_delta"]), ypos + 0.10, marker="s", markerfacecolor=WHITE, markeredgecolor="#6D747E", markersize=3.6)
            ax.plot(
                [float(run["bootstrap_ci95_low"]), float(run["bootstrap_ci95_high"])],
                [ypos - 0.10, ypos - 0.10],
                color=METHOD_COLORS["prototype"],
                linewidth=1.7,
            )
            ax.plot(float(run["mean_delta"]), ypos - 0.10, marker="o", color=METHOD_COLORS["prototype"], markersize=3.6)
            ax.text(
                0.98,
                ypos,
                f"run {format_p_value(float(run['wilcoxon_p']))}\nfold {format_p_value(float(fold['wilcoxon_on_5_fold_means_low_power']))}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=5.0,
                color=MUTED,
            )
        ax.axvline(0.0, color="#6D747E", linestyle="--", linewidth=0.75)
        ax.set_xlim(-0.026, 0.038)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks(y)
        ax.set_yticklabels(comparison_labels)
        ax.set_xlabel("Paired Macro-F1\ndifference")
        ax.set_title(title, fontsize=6.8, fontweight="bold")
        clean_axis(ax)
    add_panel_label(c_axes[0], "c", x=-0.35)
    c_axes[-1].legend(
        handles=[
            Line2D([0], [0], color=METHOD_COLORS["prototype"], marker="o", linewidth=1.7, label="25 run-level pairs: bootstrap CI"),
            Line2D([0], [0], color="#8A9099", marker="s", markerfacecolor=WHITE, linewidth=1.0, label="5-fold cluster bootstrap CI"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, -0.39),
        fontsize=5.3,
    )
    return fig


def build_figure5(root: Path) -> plt.Figure:
    per_class = read_csv(
        root,
        "paper/appendix/noise_25_run_per_class_summary.csv",
        ("noise_environment", "snr_db", "method", "class", "recall", "f1"),
    )
    directions = read_csv(
        root,
        "paper/tables/table8_per_class_noise_results_and_confusion_directions.csv",
        ("noise_environment", "snr_db", "method", "class", "feeding_to_stress", "stress_to_feeding"),
    )
    aurc = read_csv(
        root,
        "paper/appendix/noise_25_run_aurc_augrc_summary.csv",
        ("noise_environment", "snr_db", "method", "n", "mean_aurc", "mean_augrc"),
    )
    selective = read_csv(
        root,
        "paper/appendix/noise_25_run_selective_summary.csv",
        ("noise_environment", "snr_db", "method", "n", "mean_risk_at_coverage_0_95", "mean_frozen_threshold_selective_risk"),
    )

    fig = plt.figure(figsize=figure_size(MAIN_HEIGHTS_MM[MAIN_FIGURE_STEMS[4]]), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[0.9, 1.1])
    a_grid = gs[0, :].subgridspec(1, 3, wspace=0.15)
    a_axes = [fig.add_subplot(a_grid[0, idx]) for idx in range(3)]
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    environments = ["DWASHING", "TBUS", "STRAFFIC"]
    x_labels = ["clean", "20", "10", "0"]
    x = np.arange(4)
    for ax, environment in zip(a_axes, environments):
        for method in METHOD_ORDER:
            clean_rows = per_class[
                (per_class["noise_environment"].astype(str).str.lower() == "clean")
                & (per_class["class"] == "cough")
                & (per_class["method"] == method)
            ]
            if len(clean_rows) != 1:
                raise ValueError(f"BLOCKER: expected one clean cough row for {method}")
            values = [float(clean_rows.iloc[0]["recall"])]
            for snr in (20.0, 10.0, 0.0):
                rows = per_class[
                    (per_class["noise_environment"] == environment)
                    & (pd.to_numeric(per_class["snr_db"], errors="coerce") == snr)
                    & (per_class["class"] == "cough")
                    & (per_class["method"] == method)
                ]
                if len(rows) != 1:
                    raise ValueError(f"BLOCKER: missing cough row for {environment}/{snr}/{method}")
                values.append(float(rows.iloc[0]["recall"]))
            ax.plot(x, values, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], linewidth=1.05, markersize=3.4)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        if ax is a_axes[0]:
            ax.set_ylabel("Cough recall")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Active-event SNR (dB)")
        ax.set_title(environment, fontsize=6.8, fontweight="bold")
        clean_axis(ax, grid=True)
    add_panel_label(a_axes[0], "a", x=-0.24)
    a_axes[0].text(0.0, 1.16, "Cough failure across simulated-noise severity", transform=a_axes[0].transAxes, fontsize=7.5, fontweight="bold", color=TEXT)
    a_axes[-1].text(0.98, 0.10, "0 dB is an extreme\nsimulated-noise stress condition", transform=a_axes[-1].transAxes, ha="right", fontsize=5.2, color=ACCENT_RED)
    fig.legend(handles=method_handles(), loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=3, columnspacing=1.0, handletextpad=0.4)

    pooled = directions[
        (directions["noise_environment"] == "ALL_NOISY")
        & (directions["snr_db"].astype(str).str.upper() == "GROUP")
        & (directions["class"] == "cough")
        & (directions["method"].isin(METHOD_ORDER))
    ].drop_duplicates("method").set_index("method").loc[list(METHOD_ORDER)]
    categories = ["Feeding → stress", "Stress → feeding"]
    base = np.arange(2)
    width = 0.22
    for method_idx, method in enumerate(METHOD_ORDER):
        values = [float(pooled.loc[method, "feeding_to_stress"]), float(pooled.loc[method, "stress_to_feeding"])]
        positions = base + (method_idx - 1) * width
        bars = ax_b.bar(positions, values, width=width, color=METHOD_COLORS[method], edgecolor="#3F4650", linewidth=0.45)
        for bar, value in zip(bars, values):
            ax_b.text(bar.get_x() + bar.get_width() / 2, value + 35, f"{int(value)}", ha="center", va="bottom", fontsize=5.0, rotation=90)
            if method == "raw_softmax":
                bar.set_hatch("//")
            elif method == "hierarchical":
                bar.set_hatch("..")
    ax_b.set_xticks(base)
    ax_b.set_xticklabels(categories)
    ax_b.set_ylabel("Pooled decision count")
    ax_b.set_ylim(0, max(float(pooled["feeding_to_stress"].max()), float(pooled["stress_to_feeding"].max())) * 1.18)
    ax_b.set_title("ALL_NOISY boundary errors", loc="left", fontweight="bold")
    ax_b.text(0.01, 0.95, "25 fold-seed runs pooled; counts are not independent recordings", transform=ax_b.transAxes, va="top", fontsize=5.0, color=MUTED)
    clean_axis(ax_b, grid=True)
    add_panel_label(ax_b, "b")

    a_rows = aurc[
        (aurc["noise_environment"] == "ALL_NOISY")
        & (aurc["snr_db"].astype(str).str.upper() == "GROUP")
        & (aurc["method"].isin(METHOD_ORDER))
    ].set_index("method")
    s_rows = selective[
        (selective["noise_environment"] == "ALL_NOISY")
        & (selective["snr_db"].astype(str).str.upper() == "GROUP")
        & (selective["method"].isin(METHOD_ORDER))
    ].set_index("method")
    metric_specs = [
        ("AURC", "mean_aurc", a_rows),
        ("AUGRC", "mean_augrc", a_rows),
        ("Risk @ 95%", "mean_risk_at_coverage_0_95", s_rows),
        ("Frozen-threshold risk", "mean_frozen_threshold_selective_risk", s_rows),
    ]
    y = np.arange(len(metric_specs))[::-1]
    offsets = {"raw_softmax": 0.16, "prototype": 0.0, "hierarchical": -0.16}
    for method in METHOD_ORDER:
        values = [float(frame.loc[method, column]) for _, column, frame in metric_specs]
        ax_c.scatter(values, y + offsets[method], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], s=18, label=METHOD_LABELS[method], zorder=3)
        for value, ypos in zip(values, y + offsets[method]):
            ax_c.text(value + 0.006, ypos, f"{value:.4f}", va="center", fontsize=5.0, color=TEXT)
    ax_c.set_xlim(0.0, 0.36)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([label for label, _, _ in metric_specs])
    ax_c.set_xlabel("Risk metric (lower is better)")
    ax_c.set_title(
        "ALL_NOISY selective prediction (n=25)\nRaw best on AURC/AUGRC; only small fixed-point risk changes",
        loc="left",
        fontweight="bold",
    )
    clean_axis(ax_c, grid=True)
    add_panel_label(ax_c, "c")
    return fig


def build_figure_s1(root: Path) -> plt.Figure:
    ablations = read_csv(
        root,
        "paper/tables/table2_clean_baseline_and_architecture_ablations.csv",
        ("protocol", "model", "n", "mean_macro_f1", "std_macro_f1", "min_macro_f1", "max_macro_f1"),
    ).copy()
    labels = [f"{row.protocol} · {model_label(row.model)}" for row in ablations.itertuples()]
    y = np.arange(len(ablations))[::-1]

    fig, ax = plt.subplots(
        figsize=figure_size(SUPP_HEIGHTS_MM[SUPPLEMENTARY_FIGURE_STEMS[0]]),
        layout="constrained",
    )
    for ypos, row in zip(y, ablations.itertuples()):
        is_mainline = row.protocol == "cap3x" and row.model == "logmel_dur2"
        color = METHOD_COLORS["prototype"] if is_mainline else "#7E8793"
        ax.plot([float(row.min_macro_f1), float(row.max_macro_f1)], [ypos, ypos], color="#C7CCD3", linewidth=1.1, zorder=1)
        ax.plot(
            [float(row.mean_macro_f1) - float(row.std_macro_f1), float(row.mean_macro_f1) + float(row.std_macro_f1)],
            [ypos, ypos],
            color=color,
            linewidth=2.2,
            zorder=2,
        )
        ax.plot(float(row.mean_macro_f1), ypos, marker="o", color=color, markersize=4.5, zorder=3)
        ax.text(float(row.max_macro_f1) + 0.012, ypos, f"{float(row.mean_macro_f1):.4f} ± {float(row.std_macro_f1):.4f}  (n={int(row.n)})", va="center", fontsize=5.7, color=TEXT)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Macro-F1")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title("Complete locked clean architecture / front-end ablations", loc="left", fontweight="bold")
    ax.text(0.01, -0.10, "Thin line: observed min–max. Thick line: mean ± run-level SD. n=25 is 5 folds × 5 seeds; n=15 is 5 folds × 3 seeds.", transform=ax.transAxes, fontsize=5.8, color=MUTED)
    clean_axis(ax, grid=True)
    add_panel_label(ax, "a", x=-0.06)
    return fig


def build_figure_s2(root: Path) -> plt.Figure:
    summary = read_csv(
        root,
        "paper/appendix/noise_25_run_summary.csv",
        ("noise_environment", "snr_db", "method", "n", "mean_macro_f1", "std_macro_f1"),
    )
    methods = ("raw_softmax", "prototype", "hierarchical", "fused")
    environments = ("DWASHING", "TBUS", "STRAFFIC")
    x_values = np.array([20.0, 10.0, 0.0])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figure_size(SUPP_HEIGHTS_MM[SUPPLEMENTARY_FIGURE_STEMS[1]]),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for panel, ax, method in zip("abcd", axes.ravel(), methods):
        clean_rows = summary[
            (summary["noise_environment"].astype(str).str.lower() == "clean")
            & (summary["method"] == method)
        ]
        if len(clean_rows) != 1:
            raise ValueError(f"BLOCKER: expected one clean row for {method}")
        clean_mean = float(clean_rows.iloc[0]["mean_macro_f1"])
        ax.axhline(clean_mean, color="#8A9099", linestyle="--", linewidth=0.8, label="Clean reference")
        for environment in environments:
            rows = summary[
                (summary["noise_environment"] == environment)
                & (summary["method"] == method)
                & pd.to_numeric(summary["snr_db"], errors="coerce").isin(x_values)
            ].copy()
            rows["snr_numeric"] = pd.to_numeric(rows["snr_db"], errors="coerce")
            rows = rows.set_index("snr_numeric").loc[x_values]
            ax.errorbar(
                x_values,
                rows["mean_macro_f1"].astype(float),
                yerr=rows["std_macro_f1"].astype(float),
                color=ENV_COLORS[environment],
                marker=ENV_MARKERS[environment],
                linewidth=1.0,
                markersize=3.5,
                capsize=2.0,
                elinewidth=0.65,
                label=environment,
            )
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks(x_values)
        ax.set_xlim(22.5, -2.5)
        ax.set_title(METHOD_LABELS[method], loc="left", fontweight="bold")
        clean_axis(ax, grid=True)
        add_panel_label(ax, panel)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1")
    for ax in axes[1, :]:
        ax.set_xlabel("Active-event SNR (dB)")
    handles = [
        Line2D([0], [0], color=ENV_COLORS[env], marker=ENV_MARKERS[env], linewidth=1.0, label=env)
        for env in environments
    ] + [Line2D([0], [0], color="#8A9099", linestyle="--", linewidth=0.8, label="Clean reference")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, columnspacing=1.2)
    fig.text(0.5, -0.012, "All error bars are sample SD across n=25 fold-seed runs. The 0 dB condition is an extreme simulated-noise stress condition.", ha="center", fontsize=5.8, color=MUTED)
    return fig


def _plot_confusion_bubbles(ax: plt.Axes, matrix: np.ndarray, labels: Sequence[str], title: str) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals > 0) * 100.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = float(percentages[row_idx, col_idx])
            color = METHOD_COLORS["prototype"] if row_idx == col_idx else ACCENT_RED
            ax.scatter(col_idx, row_idx, s=max(5.0, value * 2.1), color=color, alpha=0.78, edgecolor=WHITE, linewidth=0.45)
            ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=5.0, color=TEXT)
    short = ["cough", "calm", "feed", "stress"]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(short)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=6.4, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def build_figure_s3(root: Path) -> plt.Figure:
    confusion = read_csv(
        root,
        "reports/prototype_noise_demand_w05_final/final_confusion_summary.csv",
        ("noise_environment", "snr_db", "method", "labels", "confusion_matrix"),
    )
    calibration = read_csv(
        root,
        "paper/appendix/clean_calibration_distribution.csv",
        ("parameter", "value", "count", "n_runs"),
    )
    fold_deltas = read_csv(
        root,
        "paper/appendix/noise_25_run_fold_level_deltas.csv",
        ("noise_environment", "fold", "prototype_minus_raw_softmax", "hierarchical_minus_raw_softmax", "hierarchical_minus_prototype"),
    )

    fig = plt.figure(figsize=figure_size(SUPP_HEIGHTS_MM[SUPPLEMENTARY_FIGURE_STEMS[2]]), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.2, 1.0], width_ratios=[1.15, 0.85])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])

    all_noisy = confusion[
        (confusion["noise_environment"] == "ALL_NOISY")
        & (confusion["snr_db"].astype(str).str.upper() == "GROUP")
    ].set_index("method")
    confusion_methods = ("raw_softmax", "prototype", "hierarchical", "fused")
    x_offsets = {method: index * 5.4 for index, method in enumerate(confusion_methods)}
    for method in confusion_methods:
        if method not in all_noisy.index:
            raise ValueError(f"BLOCKER: ALL_NOISY confusion matrix is missing for {method}")
        row = all_noisy.loc[method]
        labels = str(row["labels"]).split("|")
        matrix = parse_confusion_matrix(str(row["labels"]), str(row["confusion_matrix"]))
        row_totals = matrix.sum(axis=1, keepdims=True)
        percentages = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals > 0) * 100.0
        x0 = x_offsets[method]
        for row_idx in range(4):
            for col_idx in range(4):
                value = float(percentages[row_idx, col_idx])
                color = METHOD_COLORS["prototype"] if row_idx == col_idx else ACCENT_RED
                ax_a.scatter(x0 + col_idx, row_idx, s=max(6.0, value * 2.1), color=color, alpha=0.78, edgecolor=WHITE, linewidth=0.45)
                ax_a.text(x0 + col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=5.0, color=TEXT)
        ax_a.text(x0 + 1.5, -0.92, METHOD_LABELS[method], ha="center", va="center", fontsize=6.2, fontweight="bold", color=TEXT)
        for col_idx, label in enumerate(("cough", "calm", "feed", "stress")):
            ax_a.text(x0 + col_idx, 4.02, label, ha="right", va="top", rotation=35, fontsize=5.0, color=TEXT)
    ax_a.set_xlim(-1.0, x_offsets["fused"] + 4.0)
    ax_a.set_ylim(4.55, -1.28)
    ax_a.set_yticks(range(4))
    ax_a.set_yticklabels(["cough", "calm", "feed", "stress"])
    ax_a.set_xticks([])
    ax_a.set_ylabel("True class")
    ax_a.set_xlabel("Predicted class (repeated within method)")
    ax_a.set_title("Complete ALL_NOISY confusion matrices (row-normalized %, bubble area)", loc="left", fontweight="bold")
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    ax_a.tick_params(length=0)
    add_panel_label(ax_a, "a", x=-0.035)

    discrete_params = [
        "softmax_temperature",
        "main_prototype_temperature",
        "hierarchical_prototype_temperature",
        "hierarchy_penalty",
        "fusion_alpha",
        "fusion_kind",
    ]
    discrete_labels = ["Softmax T", "Main-proto T", "Hier.-proto T", "Hierarchy penalty", "Fusion α", "Fusion kind"]
    segment_colors = ["#DCE6EC", "#AFC4D0", "#7899AA", "#DCE5D5", "#A8B99A", "#C7C2D5"]
    y = np.arange(len(discrete_params))[::-1]
    for ypos, parameter in zip(y, discrete_params):
        rows = calibration[calibration["parameter"] == parameter].copy()
        if rows.empty:
            raise ValueError(f"BLOCKER: missing calibration distribution for {parameter}")
        rows["count"] = pd.to_numeric(rows["count"], errors="raise").astype(int)
        total = int(rows["count"].sum())
        if total != 25:
            raise ValueError(f"BLOCKER: calibration counts for {parameter} sum to {total}, expected 25")
        left = 0.0
        for idx, entry in enumerate(rows.itertuples()):
            width = float(entry.count) / total
            ax_b.barh(ypos, width, left=left, height=0.62, color=segment_colors[idx % len(segment_colors)], edgecolor=WHITE, linewidth=0.6)
            label = f"{entry.value} ({entry.count})"
            rotation = 90 if width < 0.13 else 0
            ax_b.text(left + width / 2, ypos, label, ha="center", va="center", fontsize=5.0, rotation=rotation, color=TEXT)
            left += width
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_xlabel("Fraction of 25 validation-calibrated runs")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(discrete_labels)
    ax_b.set_title("Complete discrete calibration selections", loc="left", fontweight="bold")
    clean_axis(ax_b)
    add_panel_label(ax_b, "b", x=-0.12)

    threshold_params = [
        "global_rejection_threshold",
        "per_class_rejection_threshold:cough",
        "per_class_rejection_threshold:calm_grunt",
        "per_class_rejection_threshold:feeding",
        "per_class_rejection_threshold:stress_vocal",
    ]
    threshold_labels = ["Global", "Cough", "Calm", "Feeding", "Stress"]
    threshold_values: list[np.ndarray] = []
    for parameter in threshold_params:
        rows = calibration[calibration["parameter"] == parameter]
        if rows.empty:
            raise ValueError(f"BLOCKER: missing threshold distribution for {parameter}")
        expanded = np.repeat(pd.to_numeric(rows["value"], errors="raise").to_numpy(dtype=float), pd.to_numeric(rows["count"], errors="raise").to_numpy(dtype=int))
        if expanded.size != 25:
            raise ValueError(f"BLOCKER: threshold distribution for {parameter} has {expanded.size} runs, expected 25")
        threshold_values.append(expanded)
    box = ax_c.boxplot(
        threshold_values,
        patch_artist=True,
        widths=0.52,
        medianprops={"color": TEXT, "linewidth": 0.9},
        whiskerprops={"color": "#68717D", "linewidth": 0.75},
        capprops={"color": "#68717D", "linewidth": 0.75},
        flierprops={"marker": "o", "markersize": 2.0, "markerfacecolor": WHITE, "markeredgecolor": "#68717D"},
    )
    for patch, color in zip(box["boxes"], ["#DCE6EC", "#AFC4D0", "#C7C2D5", "#DCE5D5", "#E8D8D3"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#68717D")
        patch.set_linewidth(0.75)
    ax_c.set_xticklabels(threshold_labels, rotation=25, ha="right")
    ax_c.set_ylim(0.45, 1.02)
    ax_c.set_ylabel("Frozen threshold")
    ax_c.set_title("Validation-selected rejection thresholds", loc="left", fontweight="bold")
    clean_axis(ax_c, grid=True)
    add_panel_label(ax_c, "c")

    stratum_order = ["ALL_NOISY", "MODERATE_NOISE", "EXTREME_STRESS"]
    stratum_labels = ["All noisy", "Moderate", "Extreme 0 dB"]
    delta_columns = ["prototype_minus_raw_softmax", "hierarchical_minus_raw_softmax", "hierarchical_minus_prototype"]
    delta_labels = ["Main - raw", "Hier. - raw", "Hier. - main"]
    delta_colors = [METHOD_COLORS["prototype"], METHOD_COLORS["hierarchical"], ACCENT_RED]
    group_starts = {stratum: index * 4 for index, stratum in enumerate(stratum_order)}
    xticks: list[float] = []
    xticklabels: list[str] = []
    for stratum, title in zip(stratum_order, stratum_labels):
        rows = fold_deltas[fold_deltas["noise_environment"] == stratum].sort_values("fold")
        if rows["fold"].astype(int).tolist() != [0, 1, 2, 3, 4]:
            raise ValueError(f"BLOCKER: fold-level deltas are incomplete for {stratum}")
        for xidx, (column, color) in enumerate(zip(delta_columns, delta_colors)):
            values = rows[column].astype(float).to_numpy()
            offsets = np.linspace(-0.12, 0.12, values.size)
            xpos = group_starts[stratum] + xidx
            ax_d.scatter(np.full(values.size, xpos) + offsets, values, color=color, s=14, marker="o", zorder=3)
            ax_d.plot([xpos - 0.18, xpos + 0.18], [values.mean(), values.mean()], color=TEXT, linewidth=1.2)
            xticks.append(xpos)
            xticklabels.append(delta_labels[xidx])
        ax_d.text(group_starts[stratum] + 1.0, 0.042, title, ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=TEXT)
    for separator in (3.0, 7.0):
        ax_d.axvline(separator, color=GRID, linewidth=0.7)
    ax_d.axhline(0.0, color="#6D747E", linestyle="--", linewidth=0.75)
    ax_d.set_xticks(xticks)
    ax_d.set_xticklabels(xticklabels, rotation=24, ha="right")
    ax_d.set_ylim(-0.026, 0.046)
    ax_d.set_ylabel("Fold-mean Macro-F1 delta")
    ax_d.set_xlabel("Comparison, repeated within simulated-noise stratum")
    ax_d.set_title("All five fold-level deltas (points); horizontal segment is the five-fold mean", loc="left", fontweight="bold")
    clean_axis(ax_d, grid=True)
    add_panel_label(ax_d, "d", x=-0.035)
    return fig


def build_all_figures(root: Path) -> dict[str, plt.Figure]:
    root = root.resolve()
    validate_locked_sources(root)
    configure_style()
    return {
        MAIN_FIGURE_STEMS[0]: build_figure1(root),
        MAIN_FIGURE_STEMS[1]: build_figure2(root),
        MAIN_FIGURE_STEMS[2]: build_figure3(root),
        MAIN_FIGURE_STEMS[3]: build_figure4(root),
        MAIN_FIGURE_STEMS[4]: build_figure5(root),
        SUPPLEMENTARY_FIGURE_STEMS[0]: build_figure_s1(root),
        SUPPLEMENTARY_FIGURE_STEMS[1]: build_figure_s2(root),
        SUPPLEMENTARY_FIGURE_STEMS[2]: build_figure_s3(root),
    }


def save_figure_bundle(
    figure: plt.Figure,
    out_dir: Path,
    stem: str,
    approved_directories: Sequence[Path],
    *,
    include_tiff: bool,
) -> tuple[Path, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    formats: list[tuple[str, dict[str, object]]] = [
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
    ]
    if include_tiff:
        formats.append(("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}))
    paths: list[Path] = []
    for suffix, kwargs in formats:
        path = out_dir / f"{stem}.{suffix}"
        assert_output_scope(path, approved_directories)
        figure.savefig(path, facecolor=WHITE, edgecolor="none", bbox_inches=None, **kwargs)
        if not path.is_file() or path.stat().st_size <= 1024:
            raise RuntimeError(f"BLOCKER: figure export is missing or unexpectedly small: {path}")
        paths.append(path)
    svg_path = out_dir / f"{stem}.svg"
    svg_text = svg_path.read_text(encoding="utf-8")
    if not re.search(r"<text\b", svg_text):
        raise RuntimeError(f"BLOCKER: SVG text was converted to paths: {svg_path}")
    if "SimSun" not in svg_text:
        raise RuntimeError(f"BLOCKER: SVG does not record SimSun as the primary font: {svg_path}")
    if "<image" in svg_text:
        raise RuntimeError(f"BLOCKER: SVG unexpectedly embeds a raster image: {svg_path}")
    return tuple(paths)


def _active_snr_sentence_en(used: bool) -> str:
    prefix = "Active-event SNR is computed over the valid non-padding clean-event region."
    if used:
        return prefix + " The 0 dB setting is an extreme simulated-noise stress condition, not a no-noise condition."
    return "Active-event SNR is not used in this clean/protocol figure; elsewhere, " + prefix[0].lower() + prefix[1:]


def _active_snr_sentence_cn(used: bool) -> str:
    prefix = "active-event SNR 按干净事件的有效非填充区域计算。"
    if used:
        return prefix + "0 dB 是极端模拟噪声压力条件，不是无噪声条件。"
    return "本干净/协议图不使用 active-event SNR；其他图中，" + prefix


def build_legend_texts(root: Path) -> tuple[str, str]:
    duration_pair = read_csv(
        root,
        "paper_results/tables/cv5_cap3x_logmel_dur2_vs_dur1_paired_stats.csv",
        ("n", "mean_delta", "ci95_low", "ci95_high", "wilcoxon_p"),
    ).iloc[0]
    hierarchy_pairs = [
        read_csv(root, path, ("wilcoxon_p",)).iloc[0]
        for path in (
            "paper_results/tables/cv5_cap3x_hier_w02_vs_dur2_paired_stats.csv",
            "paper_results/tables/cv5_cap3x_hier_w05_vs_dur2_paired_stats.csv",
            "paper_results/tables/cv5_cap3x_hier_w10_vs_dur2_paired_stats.csv",
        )
    ]
    clean_pairs = read_csv(
        root,
        "paper/appendix/clean_25_run_prototype_paired_stats.csv",
        ("comparison", "wilcoxon_p"),
    ).set_index("comparison")
    noise_pairs = read_csv(
        root,
        "paper/appendix/noise_25_run_paired_stats.csv",
        ("noise_environment", "snr_db", "comparison", "wilcoxon_p"),
    )
    fold_pairs = read_csv(
        root,
        "paper/appendix/noise_25_run_cluster_bootstrap.csv",
        ("noise_environment", "comparison", "wilcoxon_on_5_fold_means_low_power"),
    )

    def noise_p(stratum: str, comparison: str) -> tuple[float, float]:
        run = noise_pairs[
            (noise_pairs["noise_environment"] == stratum)
            & (noise_pairs["snr_db"].astype(str).str.upper() == "GROUP")
            & (noise_pairs["comparison"] == comparison)
        ]
        fold = fold_pairs[
            (fold_pairs["noise_environment"] == stratum)
            & (fold_pairs["comparison"] == comparison)
        ]
        if len(run) != 1 or len(fold) != 1:
            raise ValueError(f"BLOCKER: missing paired legend statistics for {stratum}/{comparison}")
        return float(run.iloc[0]["wilcoxon_p"]), float(fold.iloc[0]["wilcoxon_on_5_fold_means_low_power"])

    all_run, all_fold = noise_p("ALL_NOISY", "prototype - raw_softmax")
    mod_run, mod_fold = noise_p("MODERATE_NOISE", "prototype - raw_softmax")
    ext_run, ext_fold = noise_p("EXTREME_STRESS", "prototype - raw_softmax")
    h_ps = [float(row["wilcoxon_p"]) for row in hierarchy_pairs]

    en_sections = [
        "# Figure Legends (English)",
        "",
        "## Fig. 1 | Evidence-bounded pig-vocalization classification framework",
        "The figure asks how the locked clean and simulated-noise evaluation pipeline is organized. **a**, Public/third-party pig audio is represented as a 2 s waveform, converted to Log-Mel features and encoded by a CRNN. **b**, Parallel four-class and six-subtype heads supervise the shared embedding. **c**, main-class and subtype prototypes are built from training embeddings only; calibration uses validation data only; raw Softmax, main-prototype and hierarchical-prototype outputs are evaluated independently, without automatic routing. **d**, frozen clean evaluation and DEMAND additive-noise simulation are separated. This is a protocol schematic: n is not applicable, and no error bars, inferential test or P value applies. " + _active_snr_sentence_en(False) + " The figure does not establish real-farm external validation or a novel routing mechanism. Source data are provided in the figure source map.",
        "",
        "## Fig. 2 | Duration selection and locked clean ablations",
        f"The figure asks why 2 s Log-Mel became the clean mainline. **a**, mean Macro-F1 ± sample SD for 1 s (n=25), 2 s (n=25) and 3 s (n=15), where n is a fold-seed run (25=5 folds×5 seeds; 15=5 folds×3 seeds). **b**, matched 2 s−1 s delta across n={int(duration_pair['n'])} fold-seed pairs; point is the mean delta ({float(duration_pair['mean_delta']):.6f}), interval is a 10,000-resample paired-delta percentile-bootstrap 95% CI [{float(duration_pair['ci95_low']):.6f}, {float(duration_pair['ci95_high']):.6f}], and the exact two-sided Wilcoxon result is {format_p_value(float(duration_pair['wilcoxon_p']))}. **c**, locked clean alternatives are descriptive mean ± sample SD with n shown per row; unmatched n=15 and n=25 rows are not paired tests. " + _active_snr_sentence_en(False) + " The figure does not show that 3 s is statistically worse or that 2 s is universally optimal.",
        "",
        "## Fig. 3 | Hierarchical supervision and clean prototype inference",
        f"The figure asks what hierarchy contributes without overstating clean performance. **a**, four main classes map to six auxiliary subtypes. **b**, mean Macro-F1 ± sample SD across n=25 fold-seed runs for λ=0.2, 0.5 and 1.0; λ=0.5 has the lowest SD and λ=1.0 the highest mean. Exact two-sided Wilcoxon P values versus the matched non-hierarchical 2 s baseline are {format_p_value(h_ps[0])}, {format_p_value(h_ps[1])} and {format_p_value(h_ps[2])}, respectively; all paired bootstrap CIs cross zero. **c**, raw Softmax, main prototype and hierarchical prototype are from the same λ=0.5 checkpoint cohort (n=25); error bars are sample SD. Main-versus-raw is {format_p_value(float(clean_pairs.loc['prototype - raw_softmax', 'wilcoxon_p']))}; hierarchical-versus-raw is {format_p_value(float(clean_pairs.loc['hierarchical - raw_softmax', 'wilcoxon_p']))}; two-sided Wilcoxon, unadjusted. " + _active_snr_sentence_en(False) + " Neither hierarchical supervision nor the clean hierarchical-prototype gain is a significant incremental claim.",
        "",
        "## Fig. 4 | Main-class prototypes under simulated additive noise",
        f"The figure asks where prototype robustness is supported and how inference changes with clustering. **a**, mean Macro-F1 ± sample SD across n=25 fold-seed stratum means for moderate noise (20/10 dB), extreme stress (0 dB) and all nine noisy conditions. **b**, clean/20/10/0 dB mean trajectories for DWASHING, TBUS and STRAFFIC; no interval is drawn in this panel. **c**, paired deltas for main−raw, hierarchical−raw and hierarchical−main. Filled circles/thick intervals are 25-pair, 10,000-resample run-level bootstrap CIs; open squares/thin intervals are 10,000-resample fold-cluster CIs after averaging seeds within five folds. Exact run/fold Wilcoxon P values are printed for every row as stored, without multiplicity adjustment. For main−raw they are: moderate {format_p_value(mod_run)}/{format_p_value(mod_fold)}, all noisy {format_p_value(all_run)}/{format_p_value(all_fold)}, extreme {format_p_value(ext_run)}/{format_p_value(ext_fold)}. " + _active_snr_sentence_en(True) + " Evidence is strongest under moderate noise; run-level significance must not be presented without the fold-cluster sensitivity, and hierarchical prototypes are not the most robust noisy option.",
        "",
        "## Fig. 5 | Failure modes and selective prediction",
        "The figure asks which failures remain and whether prototypes improve uncertainty ranking. **a**, pooled confusion-derived cough recall from clean through 20/10/0 dB for each DEMAND environment. **b**, ALL_NOISY feeding→stress and stress→feeding pooled decision counts. Panels a and b pool repeated decisions from the same 25 fold-seed runs; those decisions are not independent recordings. **c**, ALL_NOISY AURC, AUGRC, risk at 95% coverage and frozen-threshold selective risk as descriptive n=25 means, where n denotes fold-seed runs; lower is better and no error bars or paired inferential P values are available. Raw Softmax is better on AURC/AUGRC; the main prototype has only small fixed-operating-point risk advantages. " + _active_snr_sentence_en(True) + " The figure does not support reliable 0 dB cough recognition, causal error mechanisms or universal uncertainty improvement.",
        "",
        "## Supplementary Fig. S1 | Complete locked clean ablations",
        "The figure asks whether any locked clean architecture/front-end variant replaces the 2 s Log-Mel mainline. The point is the run-level mean; the thick interval is mean ± sample SD; the thin interval is observed min–max. n=25 denotes 5 folds×5 seeds and n=15 denotes 5 folds×3 seeds. No matched inferential test or P value is available for these rows. " + _active_snr_sentence_en(False) + " Archived-only values are not added, and unmatched rows do not establish significance.",
        "",
        "## Supplementary Fig. S2 | DEMAND environment-by-SNR results",
        "The figure asks how each inference method behaves for every DEMAND environment×SNR combination. **a–d**, raw Softmax, main prototype, hierarchical prototype and fused ablation, respectively. Points are n=25 fold-seed means; error bars are sample SD; dashed lines are the corresponding clean mean. No multiplicity-adjusted environment-specific inference or fold-cluster P value is available. " + _active_snr_sentence_en(True) + " These controlled simulations are not real-farm external validation, and fused inference is not the main contribution.",
        "",
        "## Supplementary Fig. S3 | Confusion, calibration and fold sensitivity",
        "The figure asks whether pooled errors, validation calibration and fold-level effects support the main claims. **a**, complete ALL_NOISY 4×4 confusion matrices for four methods, row-normalized to percentages; bubble area encodes percentage and each true-class row pools 9,450 repeated decisions across 25 fold-seed runs and nine noise conditions. **b**, complete discrete validation-selected calibration frequencies across n=25 runs. **c**, global and per-class frozen rejection-threshold distributions across the same n=25 validation calibrations. **d**, five fold-mean deltas for three strata and comparisons; points are folds and horizontal segments are five-fold means. No new inferential P value is computed in this figure; exact run/fold P values remain in Fig. 4/source data. " + _active_snr_sentence_en(True) + " Calibration is validation-only, fold points are not 25 independent runs, and pooled decisions are not independent recordings.",
    ]

    cn_sections = [
        "# 图注（中文）",
        "",
        "## 图 1 | 有证据边界的猪声分类框架",
        "本图回答锁定的干净与模拟噪声评估流程如何组织。**a**，公开/第三方猪声音频表示为 2 s 波形，经 Log-Mel 特征和 CRNN 编码。**b**，四分类主头与六子类型辅助头并行监督共享表示。**c**，主类和子类型原型仅由训练嵌入构建，校准仅使用验证集；raw Softmax、main prototype 和 hierarchical prototype 独立评估，不存在自动路由。**d**，冻结的干净评估与 DEMAND 模拟加性噪声评估分开。本图为协议示意图，n 不适用，且不涉及误差线、推断检验或 P 值。" + _active_snr_sentence_cn(False) + "本图不能推出真实猪场外部验证，也不能推出新的自动路由机制。源数据对应关系见 figure source map。",
        "",
        "## 图 2 | 时长选择与锁定的干净消融",
        f"本图回答为何 2 s Log-Mel 成为干净主线。**a**，1 s（n=25）、2 s（n=25）和 3 s（n=15）的 Macro-F1 均值 ± 样本标准差；n 是 fold-seed run，25=5 folds×5 seeds，15=5 folds×3 seeds。**b**，n={int(duration_pair['n'])} 个匹配 fold-seed 对的 2 s−1 s 差值；点为平均差 {float(duration_pair['mean_delta']):.6f}，区间为 10,000 次配对差值百分位 bootstrap 95% CI [{float(duration_pair['ci95_low']):.6f}, {float(duration_pair['ci95_high']):.6f}]，双侧 Wilcoxon 精确结果为 {format_p_value(float(duration_pair['wilcoxon_p']))}。**c**，锁定的干净替代方案显示描述性均值 ± 样本标准差并逐行标 n；n=15 与 n=25 不作为配对检验。" + _active_snr_sentence_cn(False) + "本图不能推出 3 s 显著差于 2 s，也不能推出 2 s 普遍最优。",
        "",
        "## 图 3 | 层级监督与干净原型推理",
        f"本图回答层级结构在不过度声称干净性能时提供什么。**a**，四个主类对应六个辅助子类型。**b**，λ=0.2、0.5、1.0 在 n=25 个 fold-seed run 上的 Macro-F1 均值 ± 样本标准差；λ=0.5 的标准差最低，λ=1.0 的均值最高。相对匹配的非层级 2 s baseline，双侧 Wilcoxon P 值依次为 {format_p_value(h_ps[0])}、{format_p_value(h_ps[1])}、{format_p_value(h_ps[2])}，所有配对 bootstrap CI 均跨 0。**c**，raw Softmax、main prototype 与 hierarchical prototype 来自同一 λ=0.5 checkpoint cohort（n=25），误差线为样本标准差；main 对 raw 为 {format_p_value(float(clean_pairs.loc['prototype - raw_softmax', 'wilcoxon_p']))}，hierarchical 对 raw 为 {format_p_value(float(clean_pairs.loc['hierarchical - raw_softmax', 'wilcoxon_p']))}，均为未校正双侧 Wilcoxon。" + _active_snr_sentence_cn(False) + "不能声称层级监督或干净 hierarchical prototype 带来显著增量。",
        "",
        "## 图 4 | 主类原型在模拟加性噪声下的表现",
        f"本图回答原型鲁棒性在哪些条件下得到支持，以及聚类敏感性如何改变推断。**a**，moderate noise（20/10 dB）、extreme stress（0 dB）和全部 9 个噪声条件的 n=25 fold-seed 分层均值，显示 Macro-F1 均值 ± 样本标准差。**b**，DWASHING、TBUS、STRAFFIC 的 clean/20/10/0 dB 均值轨迹，本 panel 不画区间。**c**，main−raw、hierarchical−raw、hierarchical−main 的配对差值；实心圆/粗线为 25 对 run-level 的 10,000 次 bootstrap CI，空心方块/细线为先在 5 个 fold 内平均 seeds 后的 10,000 次 fold-cluster bootstrap CI。每行均标源文件中保存的 exact run/fold Wilcoxon P 值，未作多重性校正。main−raw 在 moderate、ALL_NOISY、extreme 的 run/fold P 分别为 {format_p_value(mod_run)}/{format_p_value(mod_fold)}、{format_p_value(all_run)}/{format_p_value(all_fold)}、{format_p_value(ext_run)}/{format_p_value(ext_fold)}。" + _active_snr_sentence_cn(True) + "证据在 moderate noise 最强；不得脱离 fold-cluster 敏感性单独宣称 run-level 显著，也不得声称 hierarchical prototype 在噪声下最稳健。",
        "",
        "## 图 5 | 失败模式与选择性预测",
        "本图回答剩余失败是什么，以及 prototype 是否改善不确定性排序。**a**，每个 DEMAND 环境从 clean 到 20/10/0 dB 的 pooled-confusion cough recall。**b**，ALL_NOISY 下 feeding→stress 与 stress→feeding 的 pooled decision counts。Panels a 和 b 汇总同一组 25 个 fold-seed runs 的重复决策，这些决策不是独立录音。**c**，ALL_NOISY 的 AURC、AUGRC、risk@95% coverage 和 frozen-threshold selective risk，均为描述性 n=25 均值，其中 n 表示 fold-seed runs；越低越好，没有可用误差线或配对推断 P 值。raw Softmax 的 AURC/AUGRC 更好；main prototype 仅在固定工作点风险上有小优势。" + _active_snr_sentence_cn(True) + "本图不能推出 0 dB cough 已可靠、错误具有因果机制，或 prototype 普遍改善不确定性。",
        "",
        "## 补充图 S1 | 完整锁定干净消融",
        "本图回答任何锁定的干净架构/前端变体是否替代 2 s Log-Mel 主线。点为 run-level 均值，粗线为均值 ± 样本标准差，细线为观察到的 min–max。n=25 表示 5 folds×5 seeds，n=15 表示 5 folds×3 seeds。各行没有匹配推断检验或 P 值。" + _active_snr_sentence_cn(False) + "不加入仅归档但未锁定的数值，也不把 unmatched rows 解释为显著性证据。",
        "",
        "## 补充图 S2 | DEMAND 环境 × SNR 详细结果",
        "本图回答每种推理方法在所有 DEMAND environment×SNR 组合中的表现。**a–d** 分别为 raw Softmax、main prototype、hierarchical prototype 和 fused 消融。点为 n=25 fold-seed 均值，误差线为样本标准差，虚线为对应 clean 均值。没有 multiplicity-adjusted environment-specific inference 或 fold-cluster P 值。" + _active_snr_sentence_cn(True) + "这些受控模拟不是真实猪场外部验证，fused 也不是主贡献。",
        "",
        "## 补充图 S3 | 混淆、校准与 fold 敏感性",
        "本图回答 pooled 错误、验证集校准和 fold-level 效应是否支持主张。**a**，四种方法的完整 ALL_NOISY 4×4 混淆矩阵，逐真实类别行归一化为百分比，气泡面积表示比例；每个真实类行汇总 25 个 fold-seed run 与 9 个噪声条件的 9,450 个重复决策。**b**，n=25 次验证集校准的完整离散参数选择频率。**c**，同一 n=25 验证校准中的全局与逐类冻结拒绝阈值分布。**d**，三个分层与三种比较的全部 5 个 fold 均值差；点为 fold，横线为五 fold 均值。本图不新增推断 P 值；exact run/fold P 保留在图 4/source data。" + _active_snr_sentence_cn(True) + "校准仅使用验证集，5 个 fold 点不能当作 25 个独立 run，pooled decisions 也不是独立录音。",
    ]
    return "\n".join(en_sections).strip() + "\n", "\n".join(cn_sections).strip() + "\n"


def write_text(path: Path, text: str, approved_directories: Sequence[Path]) -> None:
    assert_output_scope(path, approved_directories)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_table(path: Path, rows: list[dict[str, str]], approved_directories: Sequence[Path]) -> None:
    assert_output_scope(path, approved_directories)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def inspect_export(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
    }
    suffix = path.suffix.lower()
    if suffix == ".svg":
        text = path.read_text(encoding="utf-8")
        result.update(
            {
                "text_nodes": len(re.findall(r"<text\b", text)),
                "simsun_refs": text.count("SimSun"),
                "embedded_images": text.count("<image"),
                "gradients": text.count("Gradient"),
                "filters": text.count("<filter"),
            }
        )
    elif suffix in {".png", ".tiff", ".tif"}:
        with Image.open(path) as image:
            result["pixels"] = f"{image.width}×{image.height}"
            result["dpi"] = image.info.get("dpi")
            rgb = image.convert("RGB")
            array = np.asarray(rgb)
            nonwhite = np.any(array < 250, axis=2)
            coords = np.argwhere(nonwhite)
            if coords.size:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                result["content_margin_px"] = f"L{x0}/R{image.width - 1 - x1}/T{y0}/B{image.height - 1 - y1}"
            else:
                result["content_margin_px"] = "blank"
    return result


def build_qa_report(
    root: Path,
    exports: Sequence[ExportRecord],
    source_hashes: dict[Path, str],
    *,
    font_name: str,
    include_tiff: bool,
) -> str:
    inspected = [inspect_export(path) for record in exports for path in record.paths]
    svg_rows = [row for row in inspected if str(row["path"]).endswith(".svg")]
    main_stems = {record.stem for record in exports if record.stem in MAIN_FIGURE_STEMS}
    supp_stems = {record.stem for record in exports if record.stem in SUPPLEMENTARY_FIGURE_STEMS}
    expected_formats = 4 if include_tiff else 3
    bundle_complete = all(len(record.paths) == expected_formats for record in exports)
    svg_ok = all(
        int(row.get("text_nodes", 0)) > 0
        and int(row.get("simsun_refs", 0)) > 0
        and int(row.get("embedded_images", 0)) == 0
        and int(row.get("gradients", 0)) == 0
        and int(row.get("filters", 0)) == 0
        for row in svg_rows
    )
    lines = [
        "# Final Figure QA Report",
        "",
        "## Scope and figure contract",
        "",
        "- Main-figure count: 5 (PASS)" if len(main_stems) == 5 else f"- Main-figure count: {len(main_stems)} (FAIL)",
        "- Supplementary-figure count: 3 (PASS)" if len(supp_stems) == 3 else f"- Supplementary-figure count: {len(supp_stems)} (FAIL)",
        f"- Export bundle completeness: {'PASS' if bundle_complete else 'FAIL'} ({'SVG/PDF/300 dpi PNG/600 dpi TIFF' if include_tiff else 'SVG/PDF/300 dpi PNG'})",
        f"- Primary font: {font_name}; fallback stack: Arial, DejaVu Sans.",
        "- SVG font policy: editable text nodes (`svg.fonttype=none`); SimSun is referenced but not embedded or redistributed.",
        "- Figure width: fixed 183 mm for every main and supplementary figure; no `bbox_inches='tight'` size drift.",
        "- Background: white. No gradient, shadow, 3D, decorative icon, rounded dashboard card, or embedded table screenshot is used.",
        "- All Macro-F1 and recall axes use 0–1. Small paired effects are shown only on explicit difference axes around zero.",
        "",
        "## Evidence contracts",
        "",
        "- Figure 1 conclusion: the pipeline separates 2 s Log-Mel/CRNN representation, hierarchical training, train-only prototypes, validation-only calibration, and frozen clean/simulated-noise evaluation.",
        "- Figure 2 conclusion: the major clean gain is the matched 2 s context effect; unmatched n=15 and n=25 ablations remain descriptive.",
        "- Figure 3 conclusion: hierarchy provides semantic structure; λ=1.0 has the highest mean, λ=0.5 the lowest SD, and clean increments are nonsignificant.",
        "- Figure 4 conclusion: main prototypes show the strongest bounded evidence under moderate simulated noise, while fold clustering tempers some run-level claims.",
        "- Figure 5 conclusion: cough collapses under severe simulated noise; raw Softmax ranks uncertainty better by AURC/AUGRC, with only small prototype fixed-operating-point risk advantages.",
        "",
        "## Machine checks",
        "",
        f"- SVG editable text/font/no-raster/no-gradient/no-filter check: {'PASS' if svg_ok else 'FAIL'}.",
        f"- Non-empty export check: {'PASS' if all(int(row['bytes']) > 1024 for row in inspected) else 'FAIL'}.",
        "- Source-hash preservation is checked before and after generation; all required evidence sources were unchanged.",
        "- Figure/source and figure/claim coverage: all 25 planned panels have entries.",
        "",
        "| File | Bytes | Text nodes | SimSun refs | Pixels | DPI | Content margins (px) |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in inspected:
        rel = Path(str(row["path"])).resolve().relative_to(root.resolve()).as_posix()
        lines.append(
            f"| `{rel}` | {row['bytes']} | {row.get('text_nodes', '')} | {row.get('simsun_refs', '')} | {row.get('pixels', '')} | {row.get('dpi', '')} | {row.get('content_margin_px', '')} |"
        )
    lines.extend(
        [
            "",
            "## Source hashes",
            "",
            "The following SHA-256 values were captured before generation and rechecked after generation.",
            "",
            "| Source | SHA-256 |",
            "|---|---|",
        ]
    )
    for path, digest in sorted(source_hashes.items(), key=lambda item: item[0].as_posix()):
        rel = path.relative_to(root.resolve()).as_posix()
        lines.append(f"| `{rel}` | `{digest}` |")
    lines.extend(
        [
            "",
            "## Manual visual QA",
            "",
            "Pending post-export visual inspection in the Codex app. This section must be updated before final delivery.",
            "",
            "## Known limitations",
            "",
            "- SimSun remains editable but is not embedded in SVG; recipients without SimSun may see font substitution/reflow.",
            "- SimSun has no separate native bold face in the local font file; bold panel labels may be synthetically emboldened.",
            "- No external PDF font-inspection executable is installed; PDF validity is checked by successful Matplotlib export and non-empty file size.",
            "- Per-environment/SNR fold-cluster CIs and selective-metric paired CIs/P values do not exist in the locked source package and are not invented.",
            "- ALL_NOISY confusion counts pool repeated seed decisions; they are not independent recordings.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_package(
    root: Path,
    main_out: Path,
    supp_out: Path,
    source_out: Path,
    review_out: Path,
    *,
    include_tiff: bool,
) -> list[ExportRecord]:
    root = root.resolve()
    main_out = resolve_under(root, main_out)
    supp_out = resolve_under(root, supp_out)
    source_out = resolve_under(root, source_out)
    review_out = resolve_under(root, review_out)
    approved = (
        (root / "paper/figures_journal").resolve(),
        (root / "paper/supplementary/figures").resolve(),
        (root / "paper/figure_source_data").resolve(),
        (root / "paper/review").resolve(),
    )
    for directory in (main_out, supp_out, source_out, review_out):
        assert_output_scope(directory, approved)

    sources = require_sources(root)
    before_hashes = snapshot_sources(sources)
    figures = build_all_figures(root)
    font_name = configure_style()
    records: list[ExportRecord] = []
    try:
        for stem in MAIN_FIGURE_STEMS:
            paths = save_figure_bundle(figures[stem], main_out, stem, approved, include_tiff=include_tiff)
            records.append(ExportRecord(figure=f"Figure {MAIN_FIGURE_STEMS.index(stem) + 1}", stem=stem, paths=paths))
        for stem in SUPPLEMENTARY_FIGURE_STEMS:
            paths = save_figure_bundle(figures[stem], supp_out, stem, approved, include_tiff=include_tiff)
            records.append(ExportRecord(figure=f"Supplementary Figure S{SUPPLEMENTARY_FIGURE_STEMS.index(stem) + 1}", stem=stem, paths=paths))
    finally:
        for figure in figures.values():
            plt.close(figure)

    en_legends, cn_legends = build_legend_texts(root)
    write_text(main_out / "FIGURE_LEGENDS_EN.md", en_legends, approved)
    write_text(main_out / "FIGURE_LEGENDS_CN.md", cn_legends, approved)
    write_table(source_out / "FIGURE_SOURCE_MAP.csv", build_source_map_rows(), approved)
    write_table(source_out / "FIGURE_TO_CLAIM_MATRIX.csv", build_claim_rows(), approved)

    after_hashes = snapshot_sources(sources)
    if before_hashes != after_hashes:
        changed = [path for path in sources if before_hashes[path] != after_hashes[path]]
        raise RuntimeError(f"BLOCKER: locked source files changed during figure generation: {changed}")
    qa = build_qa_report(
        root,
        records,
        before_hashes,
        font_name=font_name,
        include_tiff=include_tiff,
    )
    write_text(main_out / "FIGURE_QA_REPORT.md", qa, approved)
    return records


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    validate_locked_sources(root)
    approved = (
        (root / "paper/figures_journal").resolve(),
        (root / "paper/supplementary/figures").resolve(),
        (root / "paper/figure_source_data").resolve(),
        (root / "paper/review").resolve(),
    )
    for path in (args.main_out, args.supp_out, args.source_out, args.review_out):
        assert_output_scope(resolve_under(root, path), approved)
    if args.validate_only:
        print("[OK] locked figure sources and approved output scope validated; no files written")
        return
    records = generate_package(
        root,
        args.main_out,
        args.supp_out,
        args.source_out,
        args.review_out,
        include_tiff=not args.skip_tiff,
    )
    print(f"[OK] generated {len(MAIN_FIGURE_STEMS)} main and {len(SUPPLEMENTARY_FIGURE_STEMS)} supplementary figures")
    for record in records:
        print(f"{record.figure}: " + ", ".join(str(path) for path in record.paths))


if __name__ == "__main__":
    main()
