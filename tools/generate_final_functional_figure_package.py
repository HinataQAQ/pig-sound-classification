#!/usr/bin/env python3
"""Build the frozen-only final functional figure package.

The module deliberately consumes only already validated summaries, prediction
tables, and train-only prototype representatives.  It never loads a checkpoint
for inference and never selects a hyperparameter from test results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image

try:  # Package import under unittest.
    from tools.generate_final_validation_audit import (
        AUX_LABELS,
        AUX_TO_MAIN,
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        CANDIDATE_LAMBDAS,
        EXPECTED_FOLDS,
        EXPECTED_KEYS,
        EXPECTED_SEEDS,
        MAIN_LABELS,
        add_disagreement_aggregates,
        add_margin_aggregates,
        aggregate_functional_metrics,
        compute_framework_summary,
        compute_run_functional_metrics,
        checkpoint_path,
        load_lambda_selection_sources,
        load_validation_selected_runs,
        manifest_path,
        paired_statistics,
        prototype_run_dir,
        raw_prediction_path,
        safe_error_auroc,
        select_lambda_by_fold,
        sha256_file,
        summary_path,
    )
except ModuleNotFoundError:  # Direct ``python tools/<script>.py`` execution.
    from generate_final_validation_audit import (  # type: ignore[no-redef]
        AUX_LABELS,
        AUX_TO_MAIN,
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        CANDIDATE_LAMBDAS,
        EXPECTED_FOLDS,
        EXPECTED_KEYS,
        EXPECTED_SEEDS,
        MAIN_LABELS,
        add_disagreement_aggregates,
        add_margin_aggregates,
        aggregate_functional_metrics,
        compute_framework_summary,
        compute_run_functional_metrics,
        checkpoint_path,
        load_lambda_selection_sources,
        load_validation_selected_runs,
        manifest_path,
        paired_statistics,
        prototype_run_dir,
        raw_prediction_path,
        safe_error_auroc,
        select_lambda_by_fold,
        sha256_file,
        summary_path,
    )


DERIVED_TABLE_FILENAMES = (
    "deployable_predicted_margin_summary.csv",
    "hierarchy_inconsistency_utility.csv",
    "prototype_functional_final_summary.csv",
)

MAIN_FIGURE_STEMS = (
    "figure1_final_framework",
    "figure2_primary_classification_evidence",
    "figure3_prototype_functional_value",
    "figure4_prototype_candidate_cases",
)

SUPPLEMENTARY_FIGURE_STEMS = (
    "figure_s1_complete_clean_ablations",
    "figure_s2_fixed_lambda_retrospective_cumulative",
    "figure_s3_lambda_selection_by_fold",
    "figure_s4_best_epoch_validation_test_gap",
    "figure_s5_demand_simulated_noise",
    "figure_s6_aurc_augrc",
    "figure_s7_label_aware_true_class_margin",
)

FIGURE_WIDTH_MM = 183.0
MAIN_FIGURE_DIR = Path("paper/figures_final")
SUPPLEMENTARY_FIGURE_DIR = MAIN_FIGURE_DIR
FINAL_VALIDATION_DIR = Path("paper/final_validation")
LOCKED_LAMBDA_SELECTION_CSV = (
    FINAL_VALIDATION_DIR / "lambda_selection_by_fold.csv"
)
SOURCE_DATA_DIR = MAIN_FIGURE_DIR
REVIEW_DIR = MAIN_FIGURE_DIR
FIGURE_EXTENSIONS = ("svg", "pdf", "png", "tiff")

TEXT = "#20242A"
MUTED = "#69717E"
GRID = "#D9DEE5"
WHITE = "#FFFFFF"
BLUE = "#477C95"
GREEN = "#70865A"
GOLD = "#B58A45"
RED = "#A95449"
PALE_BLUE = "#DDEAF0"
PALE_GREEN = "#E3EAD9"
PALE_GOLD = "#EEE6D1"
PALE_RED = "#F1DEDA"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEPLOYABLE_MARGIN_METRICS = (
    "correct_margin_mean",
    "correct_margin_q1",
    "correct_margin_median",
    "correct_margin_q3",
    "error_margin_mean",
    "error_margin_q1",
    "error_margin_median",
    "error_margin_q3",
    "error_detection_auroc",
)
_HIERARCHY_UTILITY_METRICS = (
    "inconsistency_prevalence",
    "overall_error_rate",
    "error_rate_inconsistent",
    "error_rate_consistent",
    "error_enrichment_ratio",
    "inconsistency_error_precision",
    "inconsistency_error_recall",
    "inconsistency_error_lift_vs_overall",
)


def predicted_distance_margin(distances: np.ndarray) -> np.ndarray:
    """Return the label-free gap between nearest and second-nearest prototypes."""

    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Prototype distances must be a two-dimensional matrix")
    if matrix.shape[1] < 2:
        raise ValueError("Prototype distances must contain at least two columns")
    if not np.isfinite(matrix).all():
        raise ValueError("Prototype distances contain non-finite values")
    ordered = np.sort(matrix, axis=1, kind="stable")
    return ordered[:, 1] - ordered[:, 0]


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, context: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _main_distance_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = [f"main_proto_cosine_distance_{label}" for label in MAIN_LABELS]
    _require_columns(frame, columns, context="Prediction table")
    matrix = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Prediction table contains non-finite main prototype distances")
    return matrix


def _distribution(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return (float("nan"),) * 4
    q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
    return float(values.mean()), float(q1), float(median), float(q3)


def compute_run_deployable_margin(
    predictions: pd.DataFrame, *, fold: int, seed: int
) -> pd.DataFrame:
    """Summarise a deployable shared main-prototype distance gap for one run.

    The geometric score is identical for Main Prototype and Hierarchical
    Prototype.  Only each method's correctness partition differs.  There is no
    stored hierarchical cosine-distance vector, so no mapped-subtype distance is
    constructed or implied here.
    """

    if predictions.empty:
        raise ValueError("Prediction table is empty")
    _require_columns(
        predictions,
        ("y_true", "prototype_pred", "hierarchical_pred"),
        context="Prediction table",
    )
    truth = predictions["y_true"].astype(str).to_numpy()
    margins = predicted_distance_margin(_main_distance_matrix(predictions))
    rows: list[dict[str, object]] = []
    for method, prediction_column in (
        ("main_prototype", "prototype_pred"),
        ("hierarchical_prototype", "hierarchical_pred"),
    ):
        predicted = predictions[prediction_column].astype(str).to_numpy()
        errors = predicted != truth
        correct_values = margins[~errors]
        error_values = margins[errors]
        correct_mean, correct_q1, correct_median, correct_q3 = _distribution(
            correct_values
        )
        error_mean, error_q1, error_median, error_q3 = _distribution(error_values)
        auroc, auroc_status = safe_error_auroc(errors=errors, scores=-margins)
        rows.append(
            {
                "record_type": "run",
                "fold": int(fold),
                "seed": int(seed),
                "method": method,
                "distance_level": "main",
                "margin_scope": "shared_main_prototype_geometry",
                "margin_definition": "second_nearest_minus_nearest",
                "margin_uses_true_label": False,
                "evaluation_uses_true_label": True,
                "n": int(len(margins)),
                "n_correct": int((~errors).sum()),
                "n_error": int(errors.sum()),
                "correct_margin_mean": correct_mean,
                "correct_margin_q1": correct_q1,
                "correct_margin_median": correct_median,
                "correct_margin_q3": correct_q3,
                "error_margin_mean": error_mean,
                "error_margin_q1": error_q1,
                "error_margin_median": error_median,
                "error_margin_q3": error_q3,
                "error_detection_auroc": auroc,
                "auroc_status": auroc_status,
                "error_score": "negative_predicted_distance_margin",
            }
        )
    return pd.DataFrame(rows)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or math.isclose(denominator, 0.0, abs_tol=0.0):
        return float("nan")
    return float(numerator / denominator)


def _classification_ratio_zero_division_zero(
    numerator: float, denominator: float
) -> float:
    """Return a classification ratio with the project-wide zero_division=0 rule."""

    if not np.isfinite(denominator) or math.isclose(denominator, 0.0, abs_tol=0.0):
        return 0.0
    return float(numerator / denominator)


def compute_run_hierarchy_utility(
    predictions: pd.DataFrame, *, fold: int, seed: int
) -> pd.DataFrame:
    """Evaluate hierarchy inconsistency as a method-specific error flag."""

    if predictions.empty:
        raise ValueError("Prediction table is empty")
    _require_columns(
        predictions,
        (
            "y_true",
            "raw_softmax_pred",
            "prototype_pred",
            "hierarchical_pred",
            "softmax_aux_pred",
            "prototype_aux_pred",
        ),
        context="Prediction table",
    )
    truth = predictions["y_true"].astype(str).to_numpy()
    method_columns = (
        ("raw_softmax", "raw_softmax_pred", "softmax_aux_pred"),
        ("main_prototype", "prototype_pred", "prototype_aux_pred"),
        (
            "hierarchical_prototype",
            "hierarchical_pred",
            "prototype_aux_pred",
        ),
    )
    rows: list[dict[str, object]] = []
    for method, main_column, subtype_column in method_columns:
        main_pred = predictions[main_column].astype(str).to_numpy()
        subtype_pred = predictions[subtype_column].astype(str).to_numpy()
        unknown = sorted(set(subtype_pred) - set(AUX_TO_MAIN))
        if unknown:
            raise ValueError(f"Prediction table contains unknown auxiliary labels: {unknown}")
        mapped_subtype = np.asarray(
            [AUX_TO_MAIN[value] for value in subtype_pred], dtype=object
        )
        inconsistent = main_pred != mapped_subtype
        errors = main_pred != truth
        consistent = ~inconsistent
        n_inconsistent = int(inconsistent.sum())
        n_consistent = int(consistent.sum())
        n_errors = int(errors.sum())
        flagged_errors = int((inconsistent & errors).sum())
        error_rate_inconsistent = (
            float(errors[inconsistent].mean())
            if n_inconsistent
            else float("nan")
        )
        error_rate_consistent = (
            float(errors[consistent].mean()) if n_consistent else float("nan")
        )
        overall_error_rate = float(errors.mean())
        rows.append(
            {
                "record_type": "run",
                "fold": int(fold),
                "seed": int(seed),
                "method": method,
                "n": int(len(predictions)),
                "n_inconsistent": n_inconsistent,
                "n_consistent": n_consistent,
                "n_errors": n_errors,
                "n_inconsistent_errors": flagged_errors,
                "inconsistency_prevalence": float(inconsistent.mean()),
                "overall_error_rate": overall_error_rate,
                "error_rate_inconsistent": error_rate_inconsistent,
                "error_rate_consistent": error_rate_consistent,
                "error_enrichment_ratio": _safe_ratio(
                    error_rate_inconsistent, error_rate_consistent
                ),
                "inconsistency_error_precision": _classification_ratio_zero_division_zero(
                    flagged_errors, n_inconsistent
                ),
                "inconsistency_error_recall": _classification_ratio_zero_division_zero(
                    flagged_errors, n_errors
                ),
                "inconsistency_error_lift_vs_overall": _safe_ratio(
                    error_rate_inconsistent, overall_error_rate
                ),
                "flag_definition": "main_prediction_differs_from_mapped_subtype_prediction",
                "uses_true_label_for_flag": False,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray, *, n_boot: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        return float("nan"), float("nan")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_boot, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _aggregate_run_metrics(
    run_rows: pd.DataFrame,
    *,
    metrics: Sequence[str],
    metadata: Sequence[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    _require_columns(
        run_rows,
        ("fold", "seed", "method", *metrics),
        context="Run-level aggregate input",
    )
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if run_rows.empty:
        raise ValueError("Run-level aggregate input is empty")
    if run_rows[["fold", "seed", "method"]].duplicated().any():
        raise ValueError("Run-level aggregate input contains duplicate fold-seed-method rows")
    runs = run_rows.copy()
    runs["record_type"] = "run"
    expected_keys = {
        (int(fold), int(run_seed))
        for fold in EXPECTED_FOLDS
        for run_seed in EXPECTED_SEEDS
    }
    for method, group in runs.groupby("method", sort=False):
        actual_keys = {
            (int(fold), int(run_seed))
            for fold, run_seed in zip(group["fold"], group["seed"], strict=True)
        }
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"Method {method} does not contain the exact 5-fold x 5-seed grid; "
                f"missing={missing}, extra={extra}"
            )
    fold_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    for method, group in runs.groupby("method", sort=False):
        method_metadata: dict[str, object] = {}
        for column in metadata:
            if column in group:
                values = group[column].drop_duplicates()
                if len(values) != 1:
                    raise ValueError(
                        f"Method {method} has inconsistent aggregate metadata in {column}"
                    )
                method_metadata[column] = values.iloc[0]
        fold_metric_values: dict[str, list[float]] = {metric: [] for metric in metrics}
        for fold, subset in group.groupby("fold", sort=True):
            row: dict[str, object] = {
                "record_type": "fold_mean",
                "fold": int(fold),
                "seed": np.nan,
                "method": method,
                "n": int(len(subset)),
                **method_metadata,
            }
            for metric in metrics:
                available = pd.to_numeric(
                    subset[metric], errors="raise"
                ).dropna().to_numpy(float)
                value = float(available.mean()) if available.size else float("nan")
                row[metric] = value
                row[f"{metric}_n_seeds_available"] = int(available.size)
                fold_metric_values[metric].append(value)
            fold_rows.append(row)
        overall: dict[str, object] = {
            "record_type": "overall_run_mean",
            "fold": np.nan,
            "seed": np.nan,
            "method": method,
            "n": int(len(group)),
            "n_runs": int(len(group)),
            "fold_cluster_n": int(group["fold"].nunique()),
            **method_metadata,
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="raise").dropna().to_numpy(float)
            fold_values = np.asarray(fold_metric_values[metric], dtype=float)
            available_fold_values = fold_values[np.isfinite(fold_values)]
            overall[metric] = (
                float(available_fold_values.mean())
                if available_fold_values.size
                else float("nan")
            )
            overall[f"{metric}_run_mean"] = (
                float(values.mean()) if values.size else float("nan")
            )
            overall[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if values.size > 1 else float("nan")
            )
            overall[f"{metric}_fold_sd"] = (
                float(available_fold_values.std(ddof=1))
                if available_fold_values.size > 1
                else float("nan")
            )
            overall[f"{metric}_n_runs_available"] = int(values.size)
            overall[f"{metric}_n_folds_available"] = int(
                available_fold_values.size
            )
            low, high = _bootstrap_mean_ci(
                available_fold_values, n_boot=n_boot, seed=seed
            )
            overall[f"{metric}_fold_cluster_ci95_low"] = low
            overall[f"{metric}_fold_cluster_ci95_high"] = high
        overall_rows.append(overall)
    return pd.concat(
        [runs, pd.DataFrame(fold_rows), pd.DataFrame(overall_rows)],
        ignore_index=True,
        sort=False,
    )


def aggregate_deployable_margin(
    run_rows: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Add fold means and deterministic fold-cluster CIs to margin rows."""

    return _aggregate_run_metrics(
        run_rows,
        metrics=_DEPLOYABLE_MARGIN_METRICS,
        metadata=(
            "distance_level",
            "margin_scope",
            "margin_definition",
            "margin_uses_true_label",
            "evaluation_uses_true_label",
            "error_score",
        ),
        n_boot=n_boot,
        seed=seed,
    )


def aggregate_hierarchy_utility(
    run_rows: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Add fold means and deterministic fold-cluster CIs to utility rows."""

    result = _aggregate_run_metrics(
        run_rows,
        metrics=_HIERARCHY_UTILITY_METRICS,
        metadata=("flag_definition", "uses_true_label_for_flag"),
        n_boot=n_boot,
        seed=seed,
    )
    count_columns = (
        "n_inconsistent",
        "n_consistent",
        "n_errors",
        "n_inconsistent_errors",
    )
    for method, method_rows in run_rows.groupby("method", sort=False):
        for fold in EXPECTED_FOLDS:
            source = method_rows[method_rows["fold"].astype(int) == int(fold)]
            target = (
                result["record_type"].eq("fold_mean")
                & result["method"].eq(method)
                & result["fold"].eq(float(fold))
            )
            result.loc[target, "repeated_decision_n"] = int(source["n"].sum())
            for column in count_columns:
                result.loc[target, column] = int(source[column].sum())
        target = result["record_type"].eq("overall_run_mean") & result[
            "method"
        ].eq(method)
        result.loc[target, "repeated_decision_n"] = int(method_rows["n"].sum())
        for column in count_columns:
            result.loc[target, column] = int(method_rows[column].sum())
    return result


def _read_csv(
    root: Path, relative: str | Path, required: Iterable[str]
) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen CSV source is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas supplies the details.
        raise ValueError(f"Could not read required frozen CSV source {path}: {exc}") from exc
    _require_columns(frame, required, context=f"Frozen CSV source {path}")
    return frame


def load_locked_lambda_selection(
    root: Path,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Load the approved fold-wise lambda choices as the sole cohort authority."""

    frame = _read_csv(
        root.resolve(),
        LOCKED_LAMBDA_SELECTION_CSV,
        ("fold", "selected_lambda", "test_metrics_used_for_selection"),
    ).copy()
    try:
        folds = pd.to_numeric(frame["fold"], errors="raise").to_numpy(float)
        lambdas = pd.to_numeric(
            frame["selected_lambda"], errors="raise"
        ).to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Locked lambda selection contains non-numeric values") from exc
    if not np.isfinite(folds).all() or not np.equal(folds, np.floor(folds)).all():
        raise ValueError("Locked lambda selection contains invalid fold identifiers")
    fold_ids = folds.astype(int)
    expected = set(EXPECTED_FOLDS)
    actual = set(fold_ids.tolist())
    if len(frame) != len(EXPECTED_FOLDS) or actual != expected or len(actual) != len(frame):
        raise ValueError(
            "Locked lambda selection must contain exactly one row for each fold; "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    normalized_test_use = (
        frame["test_metrics_used_for_selection"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if not normalized_test_use.isin({"false", "0"}).all():
        raise ValueError("Locked lambda selection must not use test metrics")
    if not np.isfinite(lambdas).all():
        raise ValueError("Locked lambda selection contains non-finite lambdas")
    selected: dict[int, float] = {}
    for fold, value in zip(fold_ids, lambdas, strict=True):
        candidates = [
            float(candidate)
            for candidate in CANDIDATE_LAMBDAS
            if math.isclose(
                float(value), float(candidate), rel_tol=0.0, abs_tol=1e-12
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Locked lambda selection has unsupported lambda {value} for fold {fold}"
            )
        selected[int(fold)] = candidates[0]
    frame["fold"] = fold_ids
    frame["selected_lambda"] = [selected[int(fold)] for fold in fold_ids]
    return frame.sort_values("fold", kind="stable").reset_index(drop=True), selected


def validate_locked_lambda_reproduction(
    locked: Mapping[int, float], reproduced: Mapping[int, float]
) -> None:
    """Fail if validation summaries do not reproduce the locked selection artifact."""

    expected_keys = set(EXPECTED_FOLDS)
    locked_keys = {int(fold) for fold in locked}
    reproduced_keys = {int(fold) for fold in reproduced}
    mismatches = {
        fold: (locked.get(fold), reproduced.get(fold))
        for fold in sorted(expected_keys | locked_keys | reproduced_keys)
        if fold not in locked
        or fold not in reproduced
        or not math.isclose(
            float(locked[fold]),
            float(reproduced[fold]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if locked_keys != expected_keys or reproduced_keys != expected_keys or mismatches:
        raise ValueError(
            "Validation-summary reproduction does not reproduce locked lambda selection; "
            f"mismatches={mismatches}"
        )


def _selected_lambda(
    selected_lambdas: Mapping[int, float] | pd.DataFrame, fold: int
) -> float:
    if isinstance(selected_lambdas, pd.DataFrame):
        _require_columns(
            selected_lambdas,
            ("fold", "selected_lambda"),
            context="Selected-lambda table",
        )
        rows = selected_lambdas[selected_lambdas["fold"].astype(int) == fold]
        if len(rows) != 1:
            raise ValueError(f"Selected-lambda table must contain one row for fold {fold}")
        return float(rows.iloc[0]["selected_lambda"])
    if fold not in selected_lambdas:
        raise ValueError(f"Selected-lambda mapping is missing fold {fold}")
    return float(selected_lambdas[fold])


def _top2_from_row(
    row: pd.Series, *, prefix: str, labels: Sequence[str]
) -> tuple[str, float, list[tuple[str, float]]]:
    columns = [f"{prefix}_prob_{label}" for label in labels]
    missing = [column for column in columns if column not in row.index]
    if missing:
        raise ValueError(f"Case prediction row is missing probability columns: {missing}")
    probabilities = np.asarray([float(row[column]) for column in columns])
    if not np.isfinite(probabilities).all():
        raise ValueError("Case prediction row contains non-finite probabilities")
    order = np.argsort(-probabilities, kind="stable")[:2]
    ranked = [(labels[index], float(probabilities[index])) for index in order]
    text = " | ".join(f"{label}:{probability:.6f}" for label, probability in ranked)
    return text, float(ranked[0][1] - ranked[1][1]), ranked


def _lower_median(
    frame: pd.DataFrame, columns: Sequence[str]
) -> tuple[pd.Series, int]:
    if frame.empty:
        raise ValueError("Deterministic case-selection pool is empty")
    ordered = frame.sort_values(list(columns), kind="stable").reset_index(drop=True)
    index = (len(ordered) - 1) // 2
    return ordered.iloc[index], index + 1


def _closest_representative(
    frame: pd.DataFrame,
    *,
    label_column: str,
    label: str,
    distance_column: str,
) -> pd.Series:
    _require_columns(
        frame,
        ("path", "source_id", "md5", label_column, distance_column),
        context="Train-only representative table",
    )
    candidates = frame[frame[label_column].astype(str) == label].copy()
    if candidates.empty:
        raise ValueError(f"No train-only representative is available for {label_column}={label}")
    candidates[distance_column] = pd.to_numeric(
        candidates[distance_column], errors="raise"
    )
    return candidates.sort_values(
        [distance_column, "source_id", "path"], kind="stable"
    ).iloc[0]


def select_final_case_studies(
    root: Path,
    selected_lambdas: Mapping[int, float] | pd.DataFrame,
    predictions: Mapping[tuple[int, int], pd.DataFrame],
) -> pd.DataFrame:
    """Select seven deterministic cases from validation-selected fold 0/seed 42."""

    root = root.resolve()
    fold, seed = 0, 42
    if (fold, seed) not in predictions:
        raise ValueError("Selected predictions are missing the fold 0/seed 42 case run")
    frame = predictions[(fold, seed)].copy()
    required = {
        "path",
        "source_id",
        "md5",
        "y_true",
        "raw_softmax_pred",
        "prototype_pred",
        "hierarchical_pred",
        "prototype_aux_pred",
        "prototype_aux_mapped_main",
        *(f"raw_softmax_prob_{label}" for label in MAIN_LABELS),
        *(f"prototype_prob_{label}" for label in MAIN_LABELS),
        *(f"hierarchical_prob_{label}" for label in MAIN_LABELS),
        *(f"prototype_aux_prob_{label}" for label in AUX_LABELS),
        *(f"main_proto_cosine_distance_{label}" for label in MAIN_LABELS),
        *(f"aux_proto_cosine_distance_{label}" for label in AUX_LABELS),
    }
    _require_columns(frame, required, context="Case prediction table")
    if frame.empty:
        raise ValueError("Case prediction table is empty")
    if frame["source_id"].astype(str).duplicated().any():
        raise ValueError("Case prediction table contains duplicate source_id values")
    if "input_role" in frame and set(frame["input_role"].astype(str)) != {"frozen_test"}:
        raise ValueError("Case prediction table is not exclusively frozen_test")

    raw_margins: list[float] = []
    raw_confidences: list[float] = []
    shared_margins = predicted_distance_margin(_main_distance_matrix(frame))
    for _, row in frame.iterrows():
        _, raw_margin, raw_top2 = _top2_from_row(
            row, prefix="raw_softmax", labels=MAIN_LABELS
        )
        raw_margins.append(raw_margin)
        raw_confidences.append(raw_top2[0][1])
    frame["_raw_margin"] = raw_margins
    frame["_raw_confidence"] = raw_confidences
    frame["_shared_predicted_distance_margin"] = shared_margins

    selected: list[tuple[str, pd.Series, str, int, int]] = []
    used: set[str] = set()
    for label in MAIN_LABELS:
        pool = frame[
            (frame["y_true"].astype(str) == label)
            & (frame["raw_softmax_pred"].astype(str) == label)
            & (frame["prototype_pred"].astype(str) == label)
            & (frame["hierarchical_pred"].astype(str) == label)
            & ~frame["source_id"].astype(str).isin(used)
        ]
        chosen, rank = _lower_median(pool, ("_raw_confidence", "source_id"))
        selected.append(
            (
                f"correct_{label}",
                chosen,
                "lower median raw Softmax confidence",
                rank,
                len(pool),
            )
        )
        used.add(str(chosen["source_id"]))

    boundary_labels = {"feeding", "stress_vocal"}
    for true_label, suffix in (("feeding", "feeding"), ("stress_vocal", "stress")):
        indices: list[int] = []
        for index, row in frame.iterrows():
            raw_top2 = {
                label
                for label, _ in _top2_from_row(
                    row, prefix="raw_softmax", labels=MAIN_LABELS
                )[2]
            }
            hierarchical_top2 = {
                label
                for label, _ in _top2_from_row(
                    row, prefix="hierarchical", labels=MAIN_LABELS
                )[2]
            }
            has_route_error = any(
                str(row[column]) != true_label
                for column in (
                    "raw_softmax_pred",
                    "prototype_pred",
                    "hierarchical_pred",
                )
            )
            if (
                str(row["y_true"]) == true_label
                and (raw_top2 == boundary_labels or hierarchical_top2 == boundary_labels)
                and has_route_error
                and str(row["source_id"]) not in used
            ):
                indices.append(index)
        pool = frame.loc[indices]
        chosen, rank = _lower_median(pool, ("_raw_margin", "source_id"))
        selected.append(
            (
                f"feeding_stress_boundary_{suffix}",
                chosen,
                "lower median raw Top-2 margin among boundary errors",
                rank,
                len(pool),
            )
        )
        used.add(str(chosen["source_id"]))

    remaining = frame[~frame["source_id"].astype(str).isin(used)].sort_values(
        ["_shared_predicted_distance_margin", "source_id"], kind="stable"
    )
    if remaining.empty:
        raise ValueError("No remaining frozen-test row is available for the low-margin case")
    selected.append(
        (
            "low_predicted_margin",
            remaining.iloc[0],
            "minimum remaining shared main-prototype predicted-distance margin",
            1,
            len(remaining),
        )
    )

    candidate = _selected_lambda(selected_lambdas, fold)
    run_dir = prototype_run_dir(root, fold, seed, candidate)
    main_representatives_path = (
        run_dir / "artifacts" / "train_main_prototype_samples.csv"
    )
    auxiliary_representatives_path = (
        run_dir / "artifacts" / "train_aux_prototype_samples.csv"
    )
    if not main_representatives_path.is_file():
        raise FileNotFoundError(
            f"Required train-only main representative source is missing: {main_representatives_path}"
        )
    if not auxiliary_representatives_path.is_file():
        raise FileNotFoundError(
            "Required train-only auxiliary representative source is missing: "
            f"{auxiliary_representatives_path}"
        )
    main_representatives = pd.read_csv(main_representatives_path)
    auxiliary_representatives = pd.read_csv(auxiliary_representatives_path)

    output_rows: list[dict[str, object]] = []
    prediction_path = run_dir / "evaluation" / "test_predictions.csv"
    for case_index, (case_type, row, selection_rule, rank, pool_size) in enumerate(
        selected, start=1
    ):
        raw_text, raw_margin, _ = _top2_from_row(
            row, prefix="raw_softmax", labels=MAIN_LABELS
        )
        main_text, main_probability_margin, _ = _top2_from_row(
            row, prefix="prototype", labels=MAIN_LABELS
        )
        hierarchical_text, hierarchical_probability_margin, _ = _top2_from_row(
            row, prefix="hierarchical", labels=MAIN_LABELS
        )
        subtype_text, subtype_margin, _ = _top2_from_row(
            row, prefix="prototype_aux", labels=AUX_LABELS
        )
        predicted_main = str(row["hierarchical_pred"])
        predicted_subtype = str(row["prototype_aux_pred"])
        main_representative = _closest_representative(
            main_representatives,
            label_column="main_label",
            label=predicted_main,
            distance_column="main_own_cosine_distance",
        )
        auxiliary_representative = _closest_representative(
            auxiliary_representatives,
            label_column="aux_label",
            label=predicted_subtype,
            distance_column="aux_own_cosine_distance",
        )
        shared_margin = float(row["_shared_predicted_distance_margin"])
        output: dict[str, object] = {
            "case_id": f"C{case_index}",
            "case_type": case_type,
            "selection_rule": selection_rule,
            "selection_rank": int(rank),
            "selection_pool_size": int(pool_size),
            "fold": fold,
            "seed": seed,
            "selected_lambda": candidate,
            "run_source": prediction_path.resolve().relative_to(root).as_posix(),
            "path": str(row["path"]).replace("\\", "/"),
            "source_id": str(row["source_id"]),
            "md5": str(row["md5"]),
            "true_class": str(row["y_true"]),
            "raw_softmax_pred": str(row["raw_softmax_pred"]),
            "raw_softmax_top2": raw_text,
            "raw_softmax_margin": raw_margin,
            "main_prototype_pred": str(row["prototype_pred"]),
            "main_prototype_top2": main_text,
            "main_prototype_probability_margin": main_probability_margin,
            "hierarchical_prototype_pred": predicted_main,
            "hierarchical_prototype_top2": hierarchical_text,
            "hierarchical_prototype_probability_margin": hierarchical_probability_margin,
            "subtype_pred": predicted_subtype,
            "subtype_top2": subtype_text,
            "subtype_margin": subtype_margin,
            "margin": shared_margin,
            "predicted_distance_margin": shared_margin,
            "margin_scope": "shared_main_prototype_geometry",
            "margin_definition": "second_nearest_minus_nearest",
            "hierarchy_consistency": predicted_main
            == str(row["prototype_aux_mapped_main"]),
            "prototype_aux_mapped_main": str(row["prototype_aux_mapped_main"]),
            "closest_representative_sample": str(main_representative["path"]).replace(
                "\\", "/"
            ),
            "closest_representative_source_id": str(
                main_representative["source_id"]
            ),
            "closest_representative_md5": str(main_representative["md5"]),
            "closest_representative_class": str(main_representative["main_label"]),
            "closest_representative_prototype_distance": float(
                main_representative["main_own_cosine_distance"]
            ),
            "closest_subtype_representative_sample": str(
                auxiliary_representative["path"]
            ).replace("\\", "/"),
            "closest_subtype_representative_source_id": str(
                auxiliary_representative["source_id"]
            ),
            "closest_subtype_representative_md5": str(
                auxiliary_representative["md5"]
            ),
            "closest_subtype_representative_prototype_distance": float(
                auxiliary_representative["aux_own_cosine_distance"]
            ),
            "representative_split": "train_only",
            "representative_definition": "a training sample closest to the predicted class prototype",
        }
        for label in MAIN_LABELS:
            output[f"raw_softmax_prob_{label}"] = float(
                row[f"raw_softmax_prob_{label}"]
            )
            output[f"main_prototype_prob_{label}"] = float(
                row[f"prototype_prob_{label}"]
            )
            output[f"hierarchical_prototype_prob_{label}"] = float(
                row[f"hierarchical_prob_{label}"]
            )
            output[f"main_distance_{label}"] = float(
                row[f"main_proto_cosine_distance_{label}"]
            )
        for label in AUX_LABELS:
            output[f"subtype_prob_{label}"] = float(
                row[f"prototype_aux_prob_{label}"]
            )
            output[f"subtype_distance_{label}"] = float(
                row[f"aux_proto_cosine_distance_{label}"]
            )
        output_rows.append(output)
    cases = pd.DataFrame(output_rows)
    if cases["source_id"].nunique() != len(cases):
        raise ValueError("Deterministic case roles are not unique")
    return cases


def _build_functional_final_summary(analysis: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    functional = analysis["functional"]
    for row in functional.itertuples(index=False):
        rows.append(
            {
                "evidence_family": "label_aware_functional_evaluation",
                "method": row.method,
                "metric": row.metric,
                "n_runs": row.n_runs_total,
                "mean": row.mean,
                "sd": row.sd,
                "fold_cluster_ci95_low": row.fold_cluster_ci95_low,
                "fold_cluster_ci95_high": row.fold_cluster_ci95_high,
                "deployability": "label_aware_evaluation_only",
            }
        )
    for family, table, metrics, deployability in (
        (
            "shared_main_prototype_geometry",
            analysis["deployable_margins"],
            (
                "correct_margin_median",
                "error_margin_median",
                "error_detection_auroc",
            ),
            "deployable_score_label_aware_outcome_evaluation",
        ),
        (
            "hierarchy_inconsistency_utility",
            analysis["hierarchy_utility"],
            _HIERARCHY_UTILITY_METRICS,
            "deployable_flag_label_aware_outcome_evaluation",
        ),
    ):
        overall = table[table["record_type"] == "overall_run_mean"]
        for item in overall.itertuples(index=False):
            record = item._asdict()
            for metric in metrics:
                rows.append(
                    {
                        "evidence_family": family,
                        "method": record["method"],
                        "metric": metric,
                        "n_runs": record["n_runs"],
                        "n_runs_available": record.get(
                            f"{metric}_n_runs_available", record["n_runs"]
                        ),
                        "mean": record[metric],
                        "sd": record.get(f"{metric}_sd", float("nan")),
                        "fold_cluster_ci95_low": record.get(
                            f"{metric}_fold_cluster_ci95_low", float("nan")
                        ),
                        "fold_cluster_ci95_high": record.get(
                            f"{metric}_fold_cluster_ci95_high", float("nan")
                        ),
                        "deployability": deployability,
                    }
                )
    disagreements = analysis["disagreements"]
    disagreement_runs = disagreements[disagreements["record_type"] == "run"]
    disagreement_overall = disagreements[
        disagreements["record_type"] == "overall_run_mean"
    ]
    for item in disagreement_overall.itertuples(index=False):
        record = item._asdict()
        method = str(record["prototype_method"])
        method_runs = disagreement_runs[
            disagreement_runs["prototype_method"].eq(method)
        ]
        harmed_total = int(record["repeated_run_total_raw_correct_prototype_wrong"])
        rescued_total = int(record["repeated_run_total_prototype_correct_raw_wrong"])
        both_wrong_total = int(record["repeated_run_total_both_wrong"])
        if method == "hierarchical_prototype":
            interpretation = (
                "Hierarchical Prototype harmed more repeated predictions than it "
                f"rescued in the frozen cohort ({harmed_total} harmed, "
                f"{rescued_total} rescued, {both_wrong_total} both wrong)."
            )
        else:
            interpretation = (
                "Main Prototype disagreement balance in the frozen cohort: "
                f"{harmed_total} harmed, {rescued_total} rescued, "
                f"{both_wrong_total} both wrong."
            )
        for metric in (
            "raw_correct_prototype_wrong",
            "prototype_correct_raw_wrong",
            "both_wrong",
        ):
            values = method_runs[metric].to_numpy(float)
            rows.append(
                {
                    "evidence_family": "disagreement_analysis",
                    "method": method,
                    "metric": metric,
                    "n_runs": int(len(method_runs)),
                    "n_runs_available": int(len(method_runs)),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "fold_cluster_ci95_low": record[
                        f"{metric}_fold_cluster_ci95_low"
                    ],
                    "fold_cluster_ci95_high": record[
                        f"{metric}_fold_cluster_ci95_high"
                    ],
                    "repeated_prediction_total": int(
                        record[f"repeated_run_total_{metric}"]
                    ),
                    "deployability": "label_aware_disagreement_evaluation_only",
                    "interpretation": interpretation,
                }
            )
        net_values = (
            method_runs["raw_correct_prototype_wrong"].to_numpy(float)
            - method_runs["prototype_correct_raw_wrong"].to_numpy(float)
        )
        fold_net = [
            float(
                net_values[
                    method_runs["fold"].astype(int).to_numpy() == int(fold)
                ].mean()
            )
            for fold in EXPECTED_FOLDS
        ]
        net_low, net_high = _bootstrap_mean_ci(
            fold_net,
            n_boot=int(analysis.get("n_boot", BOOTSTRAP_RESAMPLES)),
            seed=int(analysis.get("bootstrap_seed", BOOTSTRAP_SEED)),
        )
        rows.append(
            {
                "evidence_family": "disagreement_analysis",
                "method": method,
                "metric": "net_harm_minus_rescue",
                "n_runs": int(len(method_runs)),
                "n_runs_available": int(len(method_runs)),
                "mean": float(net_values.mean()),
                "sd": float(net_values.std(ddof=1)),
                "fold_cluster_ci95_low": net_low,
                "fold_cluster_ci95_high": net_high,
                "repeated_prediction_total": harmed_total - rescued_total,
                "deployability": "label_aware_disagreement_evaluation_only",
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def load_analysis(
    root: Path,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Load and derive the complete analysis without writing any artifact."""

    root = root.resolve()
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    locked_selection, selected_lambdas = load_locked_lambda_selection(root)
    selection_records, selection_hashes = load_lambda_selection_sources(root)
    selection, reproduced_lambdas = select_lambda_by_fold(selection_records)
    validate_locked_lambda_reproduction(selected_lambdas, reproduced_lambdas)
    runs, predictions, validated_source_paths = load_validation_selected_runs(
        root, selected_lambdas
    )
    framework_summary = compute_framework_summary(runs)
    primary_comparisons = (
        ("B1", "B0"),
        ("B2_valsel", "B1"),
        ("B3_valsel", "B2_valsel"),
    )
    framework_paired, framework_folds = paired_statistics(
        runs, primary_comparisons, n_boot=n_boot, seed=seed
    )
    locked_validation_paired = _read_csv(
        root,
        "paper/final_validation/validation_selected_framework_paired_stats.csv",
        ("comparison", "wilcoxon_raw_p", "holm_adjusted_p"),
    ).set_index("comparison")
    framework_paired["holm_family"] = ""
    duration_mask = framework_paired["comparison"].eq("B1 - B0")
    framework_paired.loc[duration_mask, "holm_adjusted_p"] = np.nan
    framework_paired.loc[duration_mask, "holm_family"] = (
        "not_applicable_prespecified_duration_contrast"
    )
    for comparison in ("B2_valsel - B1", "B3_valsel - B2_valsel"):
        target = framework_paired["comparison"].eq(comparison)
        if int(target.sum()) != 1 or comparison not in locked_validation_paired.index:
            raise ValueError(f"Locked primary comparison is missing: {comparison}")
        derived_p = float(framework_paired.loc[target, "wilcoxon_raw_p"].iloc[0])
        locked_p = float(locked_validation_paired.loc[comparison, "wilcoxon_raw_p"])
        if not math.isclose(derived_p, locked_p, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Derived Wilcoxon P value does not reproduce locked validation for {comparison}"
            )
        framework_paired.loc[target, "holm_adjusted_p"] = float(
            locked_validation_paired.loc[comparison, "holm_adjusted_p"]
        )
        framework_paired.loc[target, "holm_family"] = (
            "locked_final_validation_7_comparisons"
        )

    functional_runs: list[pd.DataFrame] = []
    label_margin_runs: list[pd.DataFrame] = []
    disagreement_runs: list[pd.DataFrame] = []
    deployable_runs: list[pd.DataFrame] = []
    hierarchy_runs: list[pd.DataFrame] = []
    for (fold, run_seed), frame in sorted(predictions.items()):
        functional, label_margins, disagreements = compute_run_functional_metrics(
            frame, fold=fold, seed=run_seed
        )
        functional_runs.append(functional)
        label_margin_runs.append(label_margins)
        disagreement_runs.append(disagreements)
        deployable_runs.append(
            compute_run_deployable_margin(frame, fold=fold, seed=run_seed)
        )
        hierarchy_runs.append(
            compute_run_hierarchy_utility(frame, fold=fold, seed=run_seed)
        )
    functional_run_table = pd.concat(functional_runs, ignore_index=True)
    label_margin_run_table = pd.concat(label_margin_runs, ignore_index=True)
    disagreement_run_table = pd.concat(disagreement_runs, ignore_index=True)
    deployable_run_table = pd.concat(deployable_runs, ignore_index=True)
    hierarchy_run_table = pd.concat(hierarchy_runs, ignore_index=True)
    functional = aggregate_functional_metrics(
        functional_run_table, n_boot=n_boot, seed=seed
    )
    label_aware_margins = add_margin_aggregates(
        label_margin_run_table, n_boot=n_boot, seed=seed
    )
    disagreements = add_disagreement_aggregates(
        disagreement_run_table, n_boot=n_boot, seed=seed
    )
    deployable_margins = aggregate_deployable_margin(
        deployable_run_table, n_boot=n_boot, seed=seed
    )
    hierarchy_utility = aggregate_hierarchy_utility(
        hierarchy_run_table, n_boot=n_boot, seed=seed
    )

    locked_tables = {
        "clean_ablations": _read_csv(
            root,
            "paper/tables/table2_clean_baseline_and_architecture_ablations.csv",
            ("protocol", "model", "n", "mean_macro_f1", "std_macro_f1"),
        ),
        "cumulative_summary": _read_csv(
            root,
            "paper_results/tables/cumulative_framework_summary.csv",
            ("stage", "stage_name", "n", "mean_macro_f1", "std_macro_f1"),
        ),
        "cumulative_runs": _read_csv(
            root,
            "paper_results/tables/cumulative_framework_runs.csv",
            ("fold", "seed", "B0", "B1", "B2", "B3", "lambda"),
        ),
        "cumulative_paired": _read_csv(
            root,
            "paper_results/tables/cumulative_framework_paired_stats.csv",
            (
                "comparison",
                "mean_delta",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "wilcoxon_p",
            ),
        ),
        "epochs": _read_csv(
            root,
            "paper/final_validation/convergence_best_epoch_summary.csv",
            ("stage", "fold", "seed", "best_epoch"),
        ),
        "gaps": _read_csv(
            root,
            "paper/final_validation/validation_test_gap_summary.csv",
            ("stage", "fold", "seed", "validation_minus_test_gap"),
        ),
        "noise": _read_csv(
            root,
            "paper/appendix/noise_25_run_summary.csv",
            (
                "noise_environment",
                "snr_db",
                "method",
                "n",
                "mean_macro_f1",
                "std_macro_f1",
            ),
        ),
        "selective": _read_csv(
            root,
            "paper/appendix/noise_25_run_aurc_augrc_summary.csv",
            (
                "noise_environment",
                "snr_db",
                "method",
                "n",
                "mean_aurc",
                "mean_augrc",
            ),
        ),
    }
    cases = select_final_case_studies(root, selected_lambdas, predictions)
    selected_prediction_paths = tuple(
        prototype_run_dir(root, fold, run_seed, selected_lambdas[fold])
        / "evaluation"
        / "test_predictions.csv"
        for fold in EXPECTED_FOLDS
        for run_seed in EXPECTED_SEEDS
    )
    source_paths = {
        *(Path(path).resolve() for path in validated_source_paths),
        *((root / path).resolve() for path in selection_hashes),
        *(path.resolve() for path in selected_prediction_paths),
        (root / LOCKED_LAMBDA_SELECTION_CSV).resolve(),
    }
    for relative in (
        "paper/tables/table1_dataset_and_leakage_free_protocol.csv",
        "paper/tables/table2_clean_baseline_and_architecture_ablations.csv",
        "paper_results/tables/cumulative_framework_summary.csv",
        "paper_results/tables/cumulative_framework_runs.csv",
        "paper_results/tables/cumulative_framework_paired_stats.csv",
        "paper/final_validation/convergence_best_epoch_summary.csv",
        "paper/final_validation/validation_test_gap_summary.csv",
        "paper/appendix/noise_25_run_summary.csv",
        "paper/appendix/noise_25_run_aurc_augrc_summary.csv",
    ):
        source_paths.add((root / relative).resolve())
    analysis: dict[str, Any] = {
        "selection_records": selection_records,
        "selection": selection,
        "locked_selection": locked_selection,
        "reproduced_lambdas": reproduced_lambdas,
        "selected_lambdas": selected_lambdas,
        "runs": runs,
        "predictions": predictions,
        "framework_summary": framework_summary,
        "framework_paired": framework_paired,
        "framework_folds": framework_folds,
        "functional_runs": functional_run_table,
        "functional": functional,
        "label_aware_margins": label_aware_margins,
        "disagreements": disagreements,
        "deployable_margin_runs": deployable_run_table,
        "deployable_margins": deployable_margins,
        "hierarchy_utility_runs": hierarchy_run_table,
        "hierarchy_utility": hierarchy_utility,
        "cases": cases,
        "selected_prediction_paths": selected_prediction_paths,
        "source_paths": tuple(sorted(source_paths, key=str)),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        **locked_tables,
    }
    analysis["functional_final"] = _build_functional_final_summary(analysis)
    return analysis


def configure_style() -> None:
    """Apply a compact, editable-vector publication style."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Noto Sans CJK SC",
                "Microsoft YaHei",
                "DejaVu Sans",
                "Liberation Sans",
            ],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


def _new_figure(
    rows: int,
    columns: int,
    *,
    height_mm: float,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(FIGURE_WIDTH_MM / 25.4, height_mm / 25.4),
        facecolor=WHITE,
        constrained_layout=True,
        gridspec_kw={
            **({"width_ratios": width_ratios} if width_ratios else {}),
            **({"height_ratios": height_ratios} if height_ratios else {}),
        },
    )
    return fig, np.asarray(axes, dtype=object).reshape(rows, columns)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=TEXT,
    )


def _clean_axis(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=2.5, color=MUTED)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.75, zorder=0)


def _probability_axis(ax: plt.Axes, ylabel: str = "Macro-F1") -> None:
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel(ylabel)
    _clean_axis(ax, grid=True)


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    fontsize: float = 6.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=MUTED,
        linewidth=0.7,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
    )


def _draw_arrow(
    ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.9},
    )


def build_figure1(root: Path, analysis: Mapping[str, Any]) -> plt.Figure:
    """Build the evidence-bounded final framework schematic."""

    del root, analysis
    fig, axes = _new_figure(2, 2, height_mm=118.0)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    for ax in axes.ravel():
        ax.set_axis_off()

    _draw_box(ax_a, 0.03, 0.58, 0.25, 0.22, "Train\n(no test tuning)", facecolor=PALE_GREEN)
    _draw_box(ax_a, 0.37, 0.58, 0.25, 0.22, "Validation\nselection/calibration", facecolor=PALE_GOLD)
    _draw_box(ax_a, 0.71, 0.58, 0.25, 0.22, "Frozen test\nevaluation only", facecolor=PALE_RED)
    _draw_arrow(ax_a, (0.28, 0.69), (0.37, 0.69))
    _draw_arrow(ax_a, (0.62, 0.69), (0.71, 0.69))
    ax_a.text(0.5, 0.32, "Exact path, source-ID and MD5 disjointness", transform=ax_a.transAxes, ha="center", color=MUTED)
    ax_a.set_title("Disjoint split roles", loc="left", fontweight="bold")
    _panel_label(ax_a, "a")

    _draw_box(ax_b, 0.01, 0.58, 0.17, 0.20, "2 s\nwaveform", facecolor=PALE_BLUE)
    _draw_box(ax_b, 0.24, 0.58, 0.17, 0.20, "Log-Mel", facecolor=PALE_BLUE)
    _draw_box(
        ax_b,
        0.47,
        0.55,
        0.24,
        0.26,
        "Shared hierarchical CRNN\nfrozen embedding",
        facecolor=PALE_BLUE,
        fontsize=5.0,
    )
    _draw_box(
        ax_b,
        0.78,
        0.68,
        0.20,
        0.18,
        "Primary Raw Softmax\n4-class prediction",
        facecolor=PALE_GREEN,
        fontsize=5.0,
    )
    _draw_box(
        ax_b,
        0.78,
        0.37,
        0.20,
        0.18,
        "Auxiliary 6-subtype\nsupervision",
        facecolor=PALE_GOLD,
        fontsize=5.0,
    )
    _draw_arrow(ax_b, (0.18, 0.68), (0.24, 0.68))
    _draw_arrow(ax_b, (0.41, 0.68), (0.47, 0.68))
    _draw_arrow(ax_b, (0.71, 0.70), (0.78, 0.77))
    _draw_arrow(ax_b, (0.71, 0.62), (0.78, 0.46))
    ax_b.set_title("Validated representation and auxiliary supervision", loc="left", fontweight="bold")
    _panel_label(ax_b, "b")

    _draw_box(ax_c, 0.01, 0.51, 0.17, 0.22, "Frozen\nembedding", facecolor=PALE_BLUE)
    _draw_box(
        ax_c,
        0.25,
        0.48,
        0.23,
        0.28,
        "Train-only\nmain prototypes\n+ subtype prototypes",
        facecolor=PALE_GREEN,
        fontsize=4.9,
    )
    _draw_box(
        ax_c,
        0.55,
        0.37,
        0.43,
        0.50,
        "Parallel diagnostics\nTop-k candidates\nDistances\n"
        "Predicted Top-1/Top-2 distance margin\n"
        "Hierarchy consistency\nPrototype representative",
        facecolor=PALE_RED,
        fontsize=4.5,
    )
    _draw_arrow(ax_c, (0.18, 0.62), (0.25, 0.62))
    _draw_arrow(ax_c, (0.48, 0.62), (0.55, 0.62))
    ax_c.text(
        0.5,
        0.18,
        "Validation-only calibration; frozen-test labels are evaluation only",
        transform=ax_c.transAxes,
        ha="center",
        color=MUTED,
    )
    ax_c.set_title("Parallel prototype candidate and diagnostic layer", loc="left", fontweight="bold")
    _panel_label(ax_c, "c")

    route_specs = (
        ("Primary", "Raw Softmax (B2)", PALE_BLUE),
        ("Candidate", "Main Prototype parallel candidates", PALE_GREEN),
        ("Ablation", "Hierarchical Prototype (B3 Top-1)", PALE_GOLD),
    )
    for index, (role, name, color) in enumerate(route_specs):
        y = 0.72 - index * 0.25
        _draw_box(ax_d, 0.05, y, 0.20, 0.15, role, facecolor=color)
        _draw_box(ax_d, 0.38, y, 0.50, 0.15, name, facecolor=color)
        _draw_arrow(ax_d, (0.25, y + 0.075), (0.38, y + 0.075))
    ax_d.text(0.5, 0.12, "Parallel evaluation; no automatic routing claim", transform=ax_d.transAxes, ha="center", color=MUTED)
    ax_d.set_title("Evidence-bounded deployment interpretation", loc="left", fontweight="bold")
    _panel_label(ax_d, "d")
    return fig


def build_figure2(root: Path, analysis: Mapping[str, Any]) -> plt.Figure:
    """Build the primary validation-selected classification evidence figure."""

    del root
    fig, axes = _new_figure(1, 3, height_mm=78.0, width_ratios=(1.08, 1.0, 0.86))
    ax_a, ax_b, ax_c = axes.ravel()
    summary = analysis["framework_summary"].set_index("stage")
    stages = ("B0", "B1", "B2_valsel", "B3_valsel")
    labels = ("1 s", "2 s", "B2 Raw", "B3 Hier.")
    means = summary.loc[list(stages), "mean_macro_f1"].to_numpy(float)
    sds = summary.loc[list(stages), "sd_macro_f1"].to_numpy(float)
    colors = (MUTED, BLUE, GREEN, GOLD)
    x = np.arange(len(stages))
    ax_a.bar(x, means, yerr=sds, color=colors, width=0.68, capsize=2, zorder=3)
    runs = analysis["runs"]
    offsets = np.linspace(-0.22, 0.22, len(runs))
    for stage_index, stage in enumerate(stages):
        values = runs.sort_values(["fold", "seed"], kind="stable")[stage].to_numpy(float)
        ax_a.scatter(
            stage_index + offsets,
            values,
            s=7,
            facecolor=WHITE,
            edgecolor=TEXT,
            linewidth=0.35,
            alpha=0.9,
            zorder=4,
        )
    ax_a.set_xticks(x, labels, rotation=18, ha="right")
    _probability_axis(ax_a)
    ax_a.set_title("Matched 5-fold × 5-seed clean evidence", loc="left", fontweight="bold")
    _panel_label(ax_a, "a")

    paired = analysis["framework_paired"].set_index("comparison")
    wanted = (
        "B1 - B0",
        "B2_valsel - B1",
        "B3_valsel - B2_valsel",
    )
    delta_labels = ("B1-B0", "B2-B1", "B3-B2")
    rows = paired.loc[list(wanted)]
    deltas = rows["mean_delta"].to_numpy(float)
    lows = rows["fold_cluster_bootstrap_ci95_low"].to_numpy(float)
    highs = rows["fold_cluster_bootstrap_ci95_high"].to_numpy(float)
    y = np.arange(len(wanted))[::-1]
    ax_b.axvline(0.0, color=MUTED, linewidth=0.8)
    ax_b.errorbar(
        deltas,
        y,
        xerr=np.vstack((deltas - lows, highs - deltas)),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=2,
    )
    ax_b.set_yticks(y, delta_labels)
    for ypos, row in zip(y, rows.itertuples()):
        if np.isfinite(float(row.holm_adjusted_p)):
            holm_text = (
                f"Holm {_format_p(row.holm_adjusted_p)}\nlocked 7-test family"
            )
        else:
            holm_text = "Holm n/a\nprespecified duration contrast"
        ax_b.annotate(
            f"{_format_p(row.wilcoxon_raw_p)}\n{holm_text}",
            (highs[len(y) - 1 - ypos], ypos),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=4.6,
            color=MUTED,
        )
    ax_b.set_xlabel("Mean paired Δ Macro-F1\n(fold-cluster 95% CI)")
    _clean_axis(ax_b, grid=False)
    ax_b.set_title("Incremental evidence", loc="left", fontweight="bold")
    _panel_label(ax_b, "b")

    selection = analysis["selection"].sort_values("fold")
    candidates = (0.2, 0.5, 1.0)
    selected_values = selection["selected_lambda"].to_numpy(float)
    selected_colors = [(BLUE, GREEN, GOLD)[candidates.index(value)] for value in selected_values]
    ax_c.plot(selection["fold"], selected_values, color=GRID, linewidth=0.8, zorder=1)
    ax_c.scatter(selection["fold"], selected_values, c=selected_colors, s=42, zorder=3)
    annotation_offsets = {
        0: (5, 7, "left"),
        1: (-8, 7, "right"),
        2: (0, 17, "center"),
        3: (8, 7, "left"),
        4: (0, 7, "center"),
    }
    for fold, value in zip(selection["fold"], selected_values, strict=True):
        x_offset, y_offset, alignment = annotation_offsets[int(fold)]
        ax_c.annotate(
            f"lambda={value:g}",
            (fold, value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=alignment,
            fontsize=4.7,
        )
    ax_c.set_xticks(EXPECTED_FOLDS)
    ax_c.set_xlabel("Fold")
    ax_c.set_yticks(candidates)
    ax_c.set_ylim(0.1, 1.1)
    ax_c.set_ylabel("Validation-selected lambda")
    _clean_axis(ax_c, grid=True)
    ax_c.text(
        0.03,
        0.38,
        "B1: main supported gain\n"
        "B2: exploratory/non-significant\n"
        "B3 does not improve Top-1",
        transform=ax_c.transAxes,
        fontsize=4.8,
        color=MUTED,
        va="bottom",
    )
    ax_c.set_title("Selected lambda by fold", loc="left", fontweight="bold")
    _panel_label(ax_c, "c")
    return fig


def _overall_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["record_type"] == "overall_run_mean"].copy()


def build_figure3(root: Path, analysis: Mapping[str, Any]) -> plt.Figure:
    """Build the complete functional-value evidence requested for prototypes."""

    del root
    fig, axes = _new_figure(2, 2, height_mm=120.0)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    methods = ("raw_softmax", "main_prototype", "hierarchical_prototype")
    method_labels = ("Raw", "Main", "Hier.")
    method_colors = (BLUE, GREEN, GOLD)
    functional = analysis["functional"].set_index(["method", "metric"])

    coverage_metrics = (
        "main_top1_accuracy",
        "main_top2_accuracy",
        "subtype_top1_accuracy",
        "subtype_top2_accuracy",
    )
    coverage_labels = ("Main R1", "Main R2", "Subtype R1", "Subtype R2")
    x = np.arange(len(coverage_metrics))
    width = 0.24
    for method_index, (method, label, color) in enumerate(
        zip(methods, method_labels, method_colors)
    ):
        values = [float(functional.loc[(method, metric), "mean"]) for metric in coverage_metrics]
        lows = [
            float(functional.loc[(method, metric), "fold_cluster_ci95_low"])
            for metric in coverage_metrics
        ]
        highs = [
            float(functional.loc[(method, metric), "fold_cluster_ci95_high"])
            for metric in coverage_metrics
        ]
        values_array = np.asarray(values, dtype=float)
        ax_a.bar(
            x + (method_index - 1) * width,
            values,
            width,
            color=color,
            label=label,
            yerr=np.vstack(
                (values_array - np.asarray(lows), np.asarray(highs) - values_array)
            ),
            capsize=1.8,
        )
    ax_a.set_xticks(x, coverage_labels, rotation=18, ha="right")
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_ylabel("Rank coverage")
    ax_a.legend(ncols=3, loc="lower left")
    _clean_axis(ax_a, grid=True)
    ax_a.set_title("Main/subtype Rank-1 and Rank-2 coverage", loc="left", fontweight="bold")
    _panel_label(ax_a, "a")

    boundary_metrics = (
        "feeding_stress_exact_pair_top2_coverage",
        "feeding_stress_true_class_top2_coverage",
    )
    boundary_labels = ("Exact feeding/stress pair", "True class in Top-2")
    x2 = np.arange(2)
    for method_index, (method, label, color) in enumerate(
        zip(methods, method_labels, method_colors)
    ):
        values = [float(functional.loc[(method, metric), "mean"]) for metric in boundary_metrics]
        lows = [
            float(functional.loc[(method, metric), "fold_cluster_ci95_low"])
            for metric in boundary_metrics
        ]
        highs = [
            float(functional.loc[(method, metric), "fold_cluster_ci95_high"])
            for metric in boundary_metrics
        ]
        values_array = np.asarray(values, dtype=float)
        ax_b.bar(
            x2 + (method_index - 1) * width,
            values,
            width,
            color=color,
            label=label,
            yerr=np.vstack(
                (values_array - np.asarray(lows), np.asarray(highs) - values_array)
            ),
            capsize=1.8,
        )
    ax_b.set_xticks(x2, boundary_labels, rotation=12, ha="right")
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("Top-2 coverage")
    ax_b.legend(ncols=3, loc="lower left")
    _clean_axis(ax_b, grid=True)
    ax_b.set_title("Feeding/stress boundary candidate coverage", loc="left", fontweight="bold")
    _panel_label(ax_b, "b")

    utility = _overall_rows(analysis["hierarchy_utility"]).set_index("method")
    utility_metrics = (
        "inconsistency_prevalence",
        "inconsistency_error_precision",
        "inconsistency_error_recall",
    )
    utility_labels = ("Prevalence", "Error precision", "Error recall")
    x3 = np.arange(len(utility_metrics))
    utility_folds = analysis["hierarchy_utility"]
    utility_folds = utility_folds[utility_folds["record_type"] == "fold_mean"]
    for method_index, (method, label, color) in enumerate(
        zip(methods, method_labels, method_colors)
    ):
        values = utility.loc[method, list(utility_metrics)].to_numpy(float)
        lows = utility.loc[
            method,
            [f"{metric}_fold_cluster_ci95_low" for metric in utility_metrics],
        ].to_numpy(float)
        highs = utility.loc[
            method,
            [f"{metric}_fold_cluster_ci95_high" for metric in utility_metrics],
        ].to_numpy(float)
        positions = x3 + (method_index - 1) * width
        ax_c.bar(
            positions,
            values,
            width,
            color=color,
            label=label,
            yerr=np.vstack((values - lows, highs - values)),
            capsize=1.8,
        )
        method_folds = utility_folds[utility_folds["method"].eq(method)]
        for metric_index, metric in enumerate(utility_metrics):
            fold_values = method_folds[metric].dropna().to_numpy(float)
            ax_c.scatter(
                np.full(len(fold_values), positions[metric_index]),
                fold_values,
                s=5,
                facecolor=WHITE,
                edgecolor=TEXT,
                linewidth=0.3,
                zorder=4,
            )
            ax_c.annotate(
                f"{100.0 * values[metric_index]:.1f}%",
                (positions[metric_index], values[metric_index]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=4.0,
                color=TEXT,
            )
    ax_c.set_xticks(x3, utility_labels, rotation=15, ha="right")
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_ylabel("Rate")
    ax_c.legend(ncols=3, loc="upper left")
    _clean_axis(ax_c, grid=True)
    enrichment_axis = ax_c.twinx()
    enrichment = utility.loc[list(methods), "error_enrichment_ratio"].to_numpy(float)
    enrichment_x = np.linspace(-0.18, 0.18, len(methods)) + 2.55
    for xpos, method, value, color, label in zip(
        enrichment_x, methods, enrichment, method_colors, method_labels, strict=True
    ):
        if np.isfinite(value):
            low = float(
                utility.loc[
                    method, "error_enrichment_ratio_fold_cluster_ci95_low"
                ]
            )
            high = float(
                utility.loc[
                    method, "error_enrichment_ratio_fold_cluster_ci95_high"
                ]
            )
            enrichment_axis.errorbar(
                xpos,
                value,
                yerr=[[value - low], [high - value]],
                fmt="D",
                color=color,
                capsize=1.8,
                markersize=3.5,
            )
            fold_values = utility_folds.loc[
                utility_folds["method"].eq(method), "error_enrichment_ratio"
            ].dropna().to_numpy(float)
            enrichment_axis.scatter(
                np.full(len(fold_values), xpos),
                fold_values,
                s=5,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=0.3,
                zorder=4,
            )
            n_available = int(
                utility.loc[method, "error_enrichment_ratio_n_runs_available"]
            )
            enrichment_axis.annotate(
                f"{label}\nn={n_available}",
                (xpos, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=4.2,
            )
        else:
            enrichment_axis.annotate(f"{label}: NA", (xpos, 0.05), ha="center", fontsize=4.5, color=color)
    enrichment_axis.set_ylabel("Error enrichment ratio")
    enrichment_axis.spines["top"].set_visible(False)
    ax_c.set_title("Inconsistency prevalence and detection utility", loc="left", fontweight="bold")
    _panel_label(ax_c, "c")

    run_margins = analysis["deployable_margins"]
    run_margins = run_margins[run_margins["record_type"] == "run"]
    prototype_methods = ("main_prototype", "hierarchical_prototype")
    distributions = []
    distribution_labels = []
    for method, label in zip(prototype_methods, ("Main", "Hier.")):
        subset = run_margins[run_margins["method"] == method]
        distributions.extend(
            (
                subset["correct_margin_median"].to_numpy(float),
                subset["error_margin_median"].dropna().to_numpy(float),
            )
        )
        distribution_labels.extend((f"{label}\ncorrect", f"{label}\nerror"))
    box = ax_d.boxplot(
        distributions,
        tick_labels=distribution_labels,
        showfliers=False,
        patch_artist=True,
        boxprops={"edgecolor": MUTED},
        medianprops={"color": TEXT},
    )
    for patch, color in zip(box["boxes"], (PALE_GREEN, PALE_RED, PALE_GREEN, PALE_RED)):
        patch.set_facecolor(color)
    rng = np.random.default_rng(3407)
    for position, values in enumerate(distributions, start=1):
        ax_d.scatter(
            position + rng.uniform(-0.10, 0.10, len(values)),
            values,
            s=5,
            color=MUTED,
            alpha=0.55,
        )
    ax_d.set_ylabel("Run median predicted-distance gap")
    _clean_axis(ax_d, grid=True)
    auroc_axis = ax_d.twinx()
    overall = _overall_rows(analysis["deployable_margins"]).set_index("method")
    for center, method, color in zip((1.5, 3.5), prototype_methods, (GREEN, GOLD)):
        value = float(overall.loc[method, "error_detection_auroc"])
        low = float(overall.loc[method, "error_detection_auroc_fold_cluster_ci95_low"])
        high = float(overall.loc[method, "error_detection_auroc_fold_cluster_ci95_high"])
        auroc_axis.errorbar(
            center,
            value,
            yerr=[[value - low], [high - value]],
            fmt="D",
            color=color,
            capsize=2,
            markersize=3.5,
        )
    auroc_axis.set_ylim(0.0, 1.0)
    auroc_axis.set_ylabel("Error AUROC (fold-cluster 95% CI)")
    auroc_axis.spines["top"].set_visible(False)
    ax_d.set_title("Predicted-distance distributions and AUROC", loc="left", fontweight="bold")
    _panel_label(ax_d, "d")
    return fig


def _heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    title: str,
    cmap: str = "YlGnBu_r",
    value_format: str = ".2f",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                format(float(matrix[row, column]), value_format),
                ha="center",
                va="center",
                fontsize=4.7,
                color=TEXT,
            )
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=30, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    image.set_rasterized(False)


def build_figure4(root: Path, analysis: Mapping[str, Any]) -> plt.Figure:
    """Build the seven deterministic prototype candidate cases."""

    del root
    cases = analysis["cases"]
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM / 25.4, 142.0 / 25.4),
        facecolor=WHITE,
        constrained_layout=True,
    )
    grid = fig.add_gridspec(2, 3, height_ratios=(0.92, 1.08))
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])

    table_rows = []
    for row in cases.itertuples():
        table_rows.append(
            (
                row.case_id,
                row.true_class,
                row.raw_softmax_top2,
                row.main_prototype_top2,
                row.subtype_top2,
            )
        )
    table = ax_a.table(
        cellText=table_rows,
        colLabels=("Case", "True class", "Softmax Top-2", "Main prototype Top-2", "Subtype prototype Top-2"),
        colWidths=(0.07, 0.11, 0.23, 0.23, 0.30),
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(4.5)
    table.scale(1.0, 1.25)
    for (row_index, _column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.4)
        cell.set_facecolor(PALE_BLUE if row_index == 0 else WHITE)
    ax_a.set_axis_off()
    ax_a.set_title("True class and complete Top-2 candidate evidence", loc="left", fontweight="bold")
    _panel_label(ax_a, "a")

    main_distances = cases[
        [f"main_distance_{label}" for label in MAIN_LABELS]
    ].to_numpy(float)
    _heatmap(
        ax_b,
        main_distances,
        xlabels=("cough", "calm", "feeding", "stress"),
        ylabels=cases["case_id"].tolist(),
        title="Main cosine distances",
    )
    _panel_label(ax_b, "b")

    y = np.arange(len(cases))[::-1]
    values = cases["predicted_distance_margin"].to_numpy(float)
    consistency = cases["hierarchy_consistency"].astype(bool).to_numpy()
    colors = np.where(consistency, GREEN, RED)
    ax_c.hlines(y, 0.0, values, color=colors, linewidth=2.4)
    ax_c.scatter(values, y, c=colors, s=24, edgecolor=WHITE, linewidth=0.5)
    ax_c.set_yticks(y, cases["case_id"].tolist())
    ax_c.set_xlabel("Second-nearest minus nearest main distance")
    x_max = ax_c.get_xlim()[1]
    for ypos, value, consistent in zip(y, values, consistency, strict=True):
        label_x = value - 0.025 * x_max if value > 0.65 * x_max else 0.98 * x_max
        ax_c.text(
            label_x,
            ypos,
            "consistent" if consistent else "inconsistent",
            ha="right",
            va="center",
            fontsize=4.5,
            color=GREEN if consistent else RED,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.88, "pad": 0.4},
        )
    _clean_axis(ax_c, grid=False)
    ax_c.set_title("Predicted margin and hierarchy consistency", loc="left", fontweight="bold")
    _panel_label(ax_c, "c")

    representative_distance = cases[
        "closest_representative_prototype_distance"
    ].to_numpy(float)
    ax_d.barh(y, representative_distance, color=PALE_GREEN, edgecolor=GREEN)
    ax_d.set_yticks(
        y,
        [
            f"{row.case_id} {row.closest_representative_class} | "
            f"{str(row.closest_representative_source_id)[-8:]}"
            for row in cases.itertuples()
        ],
    )
    ax_d.set_xlabel("Train representative distance")
    _clean_axis(ax_d, grid=True)
    ax_d.set_title(
        "Hierarchical-route representative",
        loc="left",
        fontweight="bold",
        fontsize=6.0,
        pad=15,
    )
    ax_d.text(
        0.0,
        1.01,
        "a training sample closest to the predicted class prototype",
        transform=ax_d.transAxes,
        fontsize=3.6,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    _panel_label(ax_d, "d")
    return fig


def build_figure_s1(root: Path, analysis: Mapping[str, Any]) -> plt.Figure:
    """Build all locked clean ablations without unmatched inference."""

    del root
    frame = analysis["clean_ablations"].copy().sort_values(
        "mean_macro_f1", ascending=True, kind="stable"
    )
    fig, axes = _new_figure(1, 1, height_mm=max(86.0, 6.0 * len(frame)))
    ax = axes[0, 0]
    y = np.arange(len(frame))
    means = frame["mean_macro_f1"].to_numpy(float)
    sds = frame["std_macro_f1"].to_numpy(float)
    ax.errorbar(means, y, xerr=sds, fmt="o", color=BLUE, ecolor=MUTED, capsize=2)
    ax.set_yticks(y, [f"{p} | {m} (n={n})" for p, m, n in frame[["protocol", "model", "n"]].itertuples(index=False, name=None)])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean Macro-F1 ± sample SD")
    _clean_axis(ax, grid=True)
    ax.set_title("Complete locked clean ablations", loc="left", fontweight="bold")
    _panel_label(ax, "a")
    return fig


def build_figure_s2(root: Path, analysis: Mapping[str, Any]) -> plt.Figure:
    """Build the fixed-λ retrospective cumulative framework evidence."""

    del root
    fig, axes = _new_figure(1, 2, height_mm=78.0)
    ax_a, ax_b = axes.ravel()
    summary = analysis["cumulative_summary"]
    cumulative_runs = analysis["cumulative_runs"].sort_values(
        ["fold", "seed"], kind="stable"
    )
    stages = tuple(summary["stage"].astype(str))
    x = np.arange(len(stages))
    for row in cumulative_runs.itertuples(index=False):
        values = [float(getattr(row, stage)) for stage in stages]
        ax_a.plot(x, values, color=GRID, linewidth=0.55, alpha=0.75, zorder=1)
        ax_a.scatter(x, values, s=4, color=MUTED, alpha=0.5, zorder=2)
    ax_a.plot(
        x,
        summary["mean_macro_f1"].to_numpy(float),
        color=BLUE,
        linewidth=2.0,
        marker="o",
        label="25-run mean",
        zorder=3,
    )
    ax_a.set_xticks(x, summary["stage"].astype(str), rotation=15)
    _probability_axis(ax_a)
    ax_a.legend(loc="lower right")
    ax_a.set_title("All 25 fixed-lambda trajectories", loc="left", fontweight="bold")
    _panel_label(ax_a, "a")

    paired = analysis["cumulative_paired"]
    y = np.arange(len(paired))[::-1]
    delta = paired["mean_delta"].to_numpy(float)
    low = paired["bootstrap_ci95_low"].to_numpy(float)
    high = paired["bootstrap_ci95_high"].to_numpy(float)
    ax_b.axvline(0.0, color=MUTED, linewidth=0.8)
    ax_b.errorbar(delta, y, xerr=np.vstack((delta - low, high - delta)), fmt="o", color=BLUE, capsize=2)
    ax_b.set_yticks(y, paired["comparison"].astype(str))
    for ypos, upper, p_value in zip(y, high, paired["wilcoxon_p"], strict=True):
        ax_b.annotate(
            _format_p(float(p_value)),
            (upper, ypos),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=4.5,
            color=MUTED,
        )
    ax_b.set_xlabel("Mean matched Δ Macro-F1 (run bootstrap 95% CI)")
    _clean_axis(ax_b)
    ax_b.set_title(
        "Retrospective paired increments\n(run bootstrap; multiplicity-unadjusted P)",
        loc="left",
        fontweight="bold",
    )
    _panel_label(ax_b, "b")
    return fig


def build_figure_s3(root: Path, analysis: Mapping[str, Any]) -> plt.Figure:
    """Build fold-wise λ selection detail."""

    del root
    fig, axes = _new_figure(1, 2, height_mm=76.0)
    ax_a, ax_b = axes.ravel()
    selection = analysis["selection"].sort_values("fold")
    candidates = (0.2, 0.5, 1.0)
    for index, candidate in enumerate(candidates):
        tag = str(candidate).replace(".", "_")
        mean = selection[f"lambda_{tag}_mean_best_val_macro_f1"].to_numpy(float)
        sd = selection[f"lambda_{tag}_sd_best_val_macro_f1"].to_numpy(float)
        ax_a.errorbar(
            selection["fold"],
            mean,
            yerr=sd,
            marker=("o", "s", "^")[index],
            color=(BLUE, GREEN, GOLD)[index],
            capsize=2,
            label=f"λ={candidate}",
        )
    ax_a.set_xticks(EXPECTED_FOLDS)
    ax_a.set_xlabel("Fold")
    ax_a.set_ylabel("Five-seed mean best validation Macro-F1 ± SD")
    ax_a.legend()
    _clean_axis(ax_a, grid=True)
    ax_a.set_title("Candidate evidence used for selection", loc="left", fontweight="bold")
    _panel_label(ax_a, "a")

    selected = selection["selected_lambda"].to_numpy(float)
    ax_b.scatter(selection["fold"], selected, c=[(BLUE, GREEN, GOLD)[candidates.index(value)] for value in selected], s=42)
    ax_b.set_xticks(EXPECTED_FOLDS)
    ax_b.set_yticks(candidates)
    ax_b.set_ylim(0.1, 1.1)
    ax_b.set_xlabel("Fold")
    ax_b.set_ylabel("Selected λ")
    _clean_axis(ax_b, grid=True)
    ax_b.set_title("Locked validation-only selections", loc="left", fontweight="bold")
    _panel_label(ax_b, "b")
    return fig


def build_figure_s4(root: Path, analysis: Mapping[str, Any]) -> plt.Figure:
    """Build best-epoch and validation-test gap diagnostics."""

    del root
    fig, axes = _new_figure(1, 2, height_mm=78.0)
    ax_a, ax_b = axes.ravel()
    epochs = analysis["epochs"]
    gaps = analysis["gaps"]
    stages = tuple(dict.fromkeys(epochs["stage"].astype(str)))
    epoch_values = [epochs.loc[epochs["stage"].astype(str) == stage, "best_epoch"].to_numpy(float) for stage in stages]
    ax_a.boxplot(epoch_values, tick_labels=stages, showfliers=False, patch_artist=True, boxprops={"facecolor": PALE_BLUE, "edgecolor": BLUE}, medianprops={"color": RED})
    ax_a.set_ylabel("Best epoch")
    _clean_axis(ax_a, grid=True)
    ax_a.set_title("Validation-selected convergence", loc="left", fontweight="bold")
    _panel_label(ax_a, "a")

    gap_values = [gaps.loc[gaps["stage"].astype(str) == stage, "validation_minus_test_gap"].to_numpy(float) for stage in stages]
    ax_b.axhline(0.0, color=MUTED, linewidth=0.8)
    ax_b.boxplot(gap_values, tick_labels=stages, showfliers=False, patch_artist=True, boxprops={"facecolor": PALE_GOLD, "edgecolor": GOLD}, medianprops={"color": RED})
    ax_b.set_ylabel("Best validation − frozen-test Macro-F1")
    _clean_axis(ax_b, grid=True)
    ax_b.set_title("Validation–test gap", loc="left", fontweight="bold")
    _panel_label(ax_b, "b")
    return fig


def build_figure_s5(root: Path, analysis: Mapping[str, Any]) -> plt.Figure:
    """Build DEMAND simulated-noise trajectories and aggregate strata."""

    del root
    fig, axes = _new_figure(2, 2, height_mm=116.0)
    noise = analysis["noise"].copy()
    noise["snr_numeric"] = pd.to_numeric(noise["snr_db"], errors="coerce")
    methods = ("raw_softmax", "prototype", "hierarchical")
    method_colors = {"raw_softmax": BLUE, "prototype": GREEN, "hierarchical": GOLD}
    for ax, environment, panel in zip(
        axes.ravel()[:3], ("DWASHING", "TBUS", "STRAFFIC"), ("a", "b", "c")
    ):
        detail = noise[
            (noise["noise_environment"] == environment)
            & noise["snr_numeric"].notna()
        ]
        for method in methods:
            subset = detail[detail["method"] == method].sort_values("snr_numeric")
            if subset.empty:
                continue
            ax.errorbar(
                subset["snr_numeric"],
                subset["mean_macro_f1"],
                yerr=subset["std_macro_f1"],
                marker="o",
                capsize=2,
                color=method_colors[method],
                label=method.replace("_", " "),
            )
        ax.set_xlabel("SNR (dB)")
        _probability_axis(ax)
        if panel == "a":
            ax.legend()
        ax.set_title(f"{environment} x SNR", loc="left", fontweight="bold")
        _panel_label(ax, panel)

    ax_d = axes.ravel()[3]
    grouped = noise[noise["noise_environment"] == "ALL_NOISY"]
    if grouped.empty:
        grouped = noise[noise["snr_numeric"].notna()].groupby("method", as_index=False).agg(mean_macro_f1=("mean_macro_f1", "mean"), std_macro_f1=("mean_macro_f1", "std"))
    grouped = grouped[grouped["method"].isin(methods)].drop_duplicates("method").set_index("method")
    present = [method for method in methods if method in grouped.index]
    x = np.arange(len(present))
    ax_d.bar(x, grouped.loc[present, "mean_macro_f1"], yerr=grouped.loc[present, "std_macro_f1"], color=[method_colors[m] for m in present], capsize=2)
    ax_d.set_xticks(x, [m.replace("_", " ") for m in present], rotation=18, ha="right")
    _probability_axis(ax_d)
    ax_d.set_title("ALL_NOISY fixed-lambda=0.5 aggregate", loc="left", fontweight="bold")
    _panel_label(ax_d, "d")
    return fig


def build_figure_s6(root: Path, analysis: Mapping[str, Any]) -> plt.Figure:
    """Build AURC/AUGRC selective-prediction evidence."""

    del root
    fig, axes = _new_figure(1, 2, height_mm=76.0)
    ax_a, ax_b = axes.ravel()
    selective = analysis["selective"].copy()
    group = selective[
        (selective["noise_environment"].astype(str) == "ALL_NOISY")
        & (selective["snr_db"].astype(str).str.upper() == "GROUP")
    ]
    if group.empty:
        group = selective.groupby("method", as_index=False).agg(
            n=("n", "max"),
            mean_aurc=("mean_aurc", "mean"),
            mean_augrc=("mean_augrc", "mean"),
        )
    methods = [method for method in ("raw_softmax", "prototype", "hierarchical") if method in set(group["method"])]
    data = group.drop_duplicates("method").set_index("method").loc[methods]
    colors = [BLUE, GREEN, GOLD][: len(methods)]
    for ax, metric, title, panel in (
        (ax_a, "mean_aurc", "AURC (lower is better)", "a"),
        (ax_b, "mean_augrc", "AUGRC (lower is better)", "b"),
    ):
        x = np.arange(len(methods))
        ax.bar(x, data[metric].to_numpy(float), color=colors)
        ax.set_xticks(x, [method.replace("_", " ") for method in methods], rotation=18, ha="right")
        ax.set_ylabel(metric.replace("mean_", "").upper())
        _clean_axis(ax, grid=True)
        ax.set_title(title, loc="left", fontweight="bold")
        _panel_label(ax, panel)
    return fig


def build_figure_s7(root: Path, analysis: Mapping[str, Any]) -> plt.Figure:
    """Build explicitly label-aware true-class prototype margin diagnostics."""

    del root
    fig, axes = _new_figure(1, 2, height_mm=78.0)
    ax_a, ax_b = axes.ravel()
    margins = _overall_rows(analysis["label_aware_margins"])
    methods = ("raw_softmax", "main_prototype", "hierarchical_prototype")
    labels = ("Raw", "Main", "Hier.")
    main = margins[(margins["level"] == "main") & margins["method"].isin(methods)].set_index("method")
    x = np.arange(len(methods))
    width = 0.34
    correct = main.loc[list(methods), "correct_margin_median"].to_numpy(float)
    correct_low = main.loc[
        list(methods), "correct_margin_median_fold_cluster_ci95_low"
    ].to_numpy(float)
    correct_high = main.loc[
        list(methods), "correct_margin_median_fold_cluster_ci95_high"
    ].to_numpy(float)
    error = main.loc[list(methods), "error_margin_median"].to_numpy(float)
    error_low = main.loc[
        list(methods), "error_margin_median_fold_cluster_ci95_low"
    ].to_numpy(float)
    error_high = main.loc[
        list(methods), "error_margin_median_fold_cluster_ci95_high"
    ].to_numpy(float)
    ax_a.bar(
        x - width / 2,
        correct,
        width,
        color=GREEN,
        label="Correct",
        yerr=np.vstack((correct - correct_low, correct_high - correct)),
        capsize=2,
    )
    ax_a.bar(
        x + width / 2,
        error,
        width,
        color=RED,
        label="Error",
        yerr=np.vstack((error - error_low, error_high - error)),
        capsize=2,
    )
    ax_a.axhline(0.0, color=MUTED, linewidth=0.7)
    ax_a.set_xticks(x, labels)
    ax_a.set_ylabel("Nearest-wrong distance − true distance")
    ax_a.legend()
    _clean_axis(ax_a, grid=True)
    ax_a.set_title("True-class main margin (label-aware)", loc="left", fontweight="bold")
    _panel_label(ax_a, "a")

    subtype = margins[(margins["level"] == "subtype") & margins["method"].isin(methods)].set_index("method")
    auroc = subtype.loc[list(methods), "error_detection_auroc"].to_numpy(float)
    auroc_low = subtype.loc[
        list(methods), "error_detection_auroc_fold_cluster_ci95_low"
    ].to_numpy(float)
    auroc_high = subtype.loc[
        list(methods), "error_detection_auroc_fold_cluster_ci95_high"
    ].to_numpy(float)
    ax_b.bar(
        x,
        auroc,
        color=(BLUE, GREEN, GOLD),
        yerr=np.vstack((auroc - auroc_low, auroc_high - auroc)),
        capsize=2,
    )
    ax_b.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.8)
    ax_b.set_xticks(x, labels)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("Subtype error-detection AUROC")
    _clean_axis(ax_b, grid=True)
    ax_b.set_title("True-subtype margin evaluation (label-aware)", loc="left", fontweight="bold")
    _panel_label(ax_b, "b")
    return fig


def build_all_figures(
    root: Path, analysis: Mapping[str, Any]
) -> dict[str, plt.Figure]:
    """Build all four main and seven supplementary matplotlib figures."""

    root = root.resolve()
    configure_style()
    builders = (
        (MAIN_FIGURE_STEMS[0], build_figure1),
        (MAIN_FIGURE_STEMS[1], build_figure2),
        (MAIN_FIGURE_STEMS[2], build_figure3),
        (MAIN_FIGURE_STEMS[3], build_figure4),
        (SUPPLEMENTARY_FIGURE_STEMS[0], build_figure_s1),
        (SUPPLEMENTARY_FIGURE_STEMS[1], build_figure_s2),
        (SUPPLEMENTARY_FIGURE_STEMS[2], build_figure_s3),
        (SUPPLEMENTARY_FIGURE_STEMS[3], build_figure_s4),
        (SUPPLEMENTARY_FIGURE_STEMS[4], build_figure_s5),
        (SUPPLEMENTARY_FIGURE_STEMS[5], build_figure_s6),
        (SUPPLEMENTARY_FIGURE_STEMS[6], build_figure_s7),
    )
    figures = {stem: builder(root, analysis) for stem, builder in builders}
    for stem, figure in figures.items():
        if not math.isclose(
            figure.get_figwidth() * 25.4, FIGURE_WIDTH_MM, abs_tol=1e-7
        ):
            raise RuntimeError(f"Figure {stem} does not have the required 183-mm width")
        if not figure.axes:
            raise RuntimeError(f"Figure {stem} has no axes")
        if tuple(figure.get_facecolor()) != (1.0, 1.0, 1.0, 1.0):
            raise RuntimeError(f"Figure {stem} does not have a white background")
    return figures


_PANEL_SPECS: tuple[
    tuple[str, str, str, tuple[str, ...], str, str, str, str], ...
] = (
    (
        "Figure 1",
        "a",
        MAIN_FIGURE_STEMS[0],
        ("paper/tables/table1_dataset_and_leakage_free_protocol.csv",),
        "Render locked train/validation/frozen-test roles and disjointness rules",
        "protocol_only",
        "The final framework keeps model fitting, validation decisions, and frozen-test evaluation separate.",
        "Does not establish external real-farm validity.",
    ),
    (
        "Figure 1",
        "b",
        MAIN_FIGURE_STEMS[0],
        ("paper/tables/table1_dataset_and_leakage_free_protocol.csv",),
        "Render the validated 2-s Log-Mel CRNN and dual supervision heads",
        "protocol_only",
        "The validated representation uses a 2-s Log-Mel CRNN with four main and six subtype labels.",
        "Does not claim a new backbone architecture.",
    ),
    (
        "Figure 1",
        "c",
        MAIN_FIGURE_STEMS[0],
        (
            "__case_sources__",
            "paper/tables/table1_dataset_and_leakage_free_protocol.csv",
        ),
        "Trace train-only prototype construction and validation-only calibration",
        "deployable_protocol",
        "Class prototypes are built from training embeddings; calibration is validation-only.",
        "Does not use test labels to construct or calibrate prototypes.",
    ),
    (
        "Figure 1",
        "d",
        MAIN_FIGURE_STEMS[0],
        ("paper/final_validation/validation_selected_framework_summary.csv",),
        "Separate B2 Raw Softmax primary evidence from parallel prototype candidates and B3 ablation",
        "deployable_primary_and_candidate_routes",
        "B2 validation-selected Raw Softmax is primary; prototype routes are parallel candidates.",
        "Does not establish an automatic routing policy.",
    ),
    (
        "Figure 2",
        "a",
        MAIN_FIGURE_STEMS[1],
        (
            "paper/final_validation/validation_selected_framework_runs.csv",
            "paper/final_validation/validation_selected_framework_summary.csv",
        ),
        "Plot run means with sample SD on a complete 0-1 Macro-F1 axis",
        "label_aware_frozen_test_evaluation",
        "Two-second context supplies the major established clean-audio gain.",
        "Does not imply that the B2 or B3 increment is significant.",
    ),
    (
        "Figure 2",
        "b",
        MAIN_FIGURE_STEMS[1],
        (
            "paper/final_validation/validation_selected_framework_paired_stats.csv",
            "paper_results/tables/cumulative_framework_paired_stats.csv",
        ),
        "Plot matched mean deltas with deterministic fold-cluster bootstrap intervals",
        "label_aware_frozen_test_evaluation",
        "The B2 and B3 increments are bounded by matched fold-seed and fold-cluster evidence.",
        "Never call the B2 or B3 increment statistically significant.",
    ),
    (
        "Figure 2",
        "c",
        MAIN_FIGURE_STEMS[1],
        ("paper/final_validation/lambda_selection_by_fold.csv",),
        "Plot only validation means and fold-wise selected lambdas",
        "validation_only_selection",
        "Lambda is selected independently within each fold from validation performance.",
        "Does not use frozen-test performance for selection.",
    ),
    (
        "Figure 3",
        "a",
        MAIN_FIGURE_STEMS[2],
        ("__selected_predictions__",),
        "Compute main/subtype Rank-1 and Rank-2 coverage separately for all three routes",
        "label_aware_candidate_coverage",
        "Main and subtype ranking coverage quantifies candidate utility.",
        "Coverage uses labels and is not a deployable score.",
    ),
    (
        "Figure 3",
        "b",
        MAIN_FIGURE_STEMS[2],
        ("__selected_predictions__",),
        "Compute exact feeding/stress pair and true-class Top-2 coverage",
        "label_aware_boundary_candidate_coverage",
        "Top-2 coverage quantifies feeding/stress boundary candidate retention.",
        "Does not establish automatic boundary resolution.",
    ),
    (
        "Figure 3",
        "c",
        MAIN_FIGURE_STEMS[2],
        ("__selected_predictions__",),
        "Recompute main-versus-mapped-subtype inconsistency separately for all three methods",
        "deployable_flag_label_aware_evaluation",
        "Hierarchy inconsistency enriches some method-specific error subsets.",
        "Does not establish a safe automatic rejector.",
    ),
    (
        "Figure 3",
        "d",
        MAIN_FIGURE_STEMS[2],
        ("__selected_predictions__",),
        "Show 25-run correct/error shared-margin distributions and error AUROC with fold-cluster CI",
        "deployable_score_label_aware_evaluation",
        "Both prototype routes share the same main-prototype geometry while correctness partitions differ.",
        "Does not invent a hierarchical or mapped-subtype cosine-distance vector.",
    ),
    (
        "Figure 4",
        "a",
        MAIN_FIGURE_STEMS[3],
        ("__case_sources__",),
        "Apply the locked fold-0/seed-42 protocol and tabulate true class plus Softmax, main-prototype, and subtype Top-2 evidence",
        "label_aware_frozen_test_case_evidence",
        "Seven unique frozen cases show complete Top-2 candidate evidence.",
        "Cases are illustrative and do not estimate population prevalence.",
    ),
    (
        "Figure 4",
        "b",
        MAIN_FIGURE_STEMS[3],
        ("__case_sources__",),
        "Show distances from every case to all four main-class prototypes",
        "deployable_case_geometry",
        "The same cases expose the complete shared main-prototype geometry.",
        "Does not use true labels to define the displayed distances.",
    ),
    (
        "Figure 4",
        "c",
        MAIN_FIGURE_STEMS[3],
        ("__case_sources__",),
        "Show the label-free shared main-prototype predicted-distance margin",
        "deployable_case_score",
        "The seventh case is the minimum remaining shared predicted-distance margin.",
        "Does not call this a hierarchical decision distance.",
    ),
    (
        "Figure 4",
        "d",
        MAIN_FIGURE_STEMS[3],
        ("__case_sources__",),
        "Attach the closest same-run train-only representative of the hierarchical-predicted main class",
        "train_only_representative_trace",
        "Every illustrative case is linked to a deterministic representative for the hierarchical-predicted main class.",
        "Does not treat a representative as a biological exemplar.",
    ),
    (
        "Supplementary Figure S1",
        "a",
        SUPPLEMENTARY_FIGURE_STEMS[0],
        ("paper/tables/table2_clean_baseline_and_architecture_ablations.csv",),
        "Plot every locked clean ablation as mean plus sample SD without unmatched inference",
        "label_aware_descriptive_evaluation",
        "No retained clean alternative replaces the 2-s Log-Mel mainline.",
        "Unmatched n=15 and n=25 rows do not establish significance.",
    ),
    (
        "Supplementary Figure S2",
        "a",
        SUPPLEMENTARY_FIGURE_STEMS[1],
        (
            "paper_results/tables/cumulative_framework_summary.csv",
            "paper_results/tables/cumulative_framework_runs.csv",
        ),
        "Plot all 25 fixed-lambda retrospective stage trajectories and their mean",
        "retrospective_label_aware_evaluation",
        "The fixed-lambda cohort provides a retrospective cumulative view.",
        "Does not supersede the validation-selected primary analysis.",
    ),
    (
        "Supplementary Figure S2",
        "b",
        SUPPLEMENTARY_FIGURE_STEMS[1],
        ("paper_results/tables/cumulative_framework_paired_stats.csv",),
        "Plot fixed-lambda matched increments and stored bootstrap intervals",
        "retrospective_label_aware_evaluation",
        "Fixed-lambda paired increments remain bounded retrospective evidence.",
        "Does not make nonsignificant increments significant.",
    ),
    (
        "Supplementary Figure S3",
        "a",
        SUPPLEMENTARY_FIGURE_STEMS[2],
        ("paper/final_validation/lambda_selection_by_fold.csv",),
        "Plot all candidate validation means and sample SD values",
        "validation_only_selection",
        "All candidate weights are retained in the fold-wise selection audit.",
        "Does not use test metrics.",
    ),
    (
        "Supplementary Figure S3",
        "b",
        SUPPLEMENTARY_FIGURE_STEMS[2],
        ("paper/final_validation/lambda_selection_by_fold.csv",),
        "Show the selected lambda for each fold",
        "validation_only_selection",
        "Selected weights vary across folds under the locked rule.",
        "Does not define one globally optimal lambda.",
    ),
    (
        "Supplementary Figure S4",
        "a",
        SUPPLEMENTARY_FIGURE_STEMS[3],
        ("paper/final_validation/convergence_best_epoch_summary.csv",),
        "Summarise best epochs across matched fold-seed runs",
        "validation_selected_training_diagnostic",
        "Best epochs document convergence of the frozen cohorts.",
        "Does not compare compute efficiency prospectively.",
    ),
    (
        "Supplementary Figure S4",
        "b",
        SUPPLEMENTARY_FIGURE_STEMS[3],
        ("paper/final_validation/validation_test_gap_summary.csv",),
        "Summarise best-validation minus frozen-test gaps",
        "label_aware_diagnostic",
        "Validation-test gaps disclose selection variability.",
        "Does not justify test-set tuning.",
    ),
    (
        "Supplementary Figure S5",
        "a",
        SUPPLEMENTARY_FIGURE_STEMS[4],
        (
            "paper/appendix/noise_25_run_summary.csv",
            "docs/NOISE_SOURCE_DECISION.md",
        ),
        "Plot DEMAND environment-by-SNR descriptive means and sample SD",
        "controlled_simulated_noise_evaluation",
        "Prototype behavior is evaluated under controlled additive noise.",
        "Does not establish real-farm external validity.",
    ),
    (
        "Supplementary Figure S5",
        "b",
        SUPPLEMENTARY_FIGURE_STEMS[4],
        (
            "paper/appendix/noise_25_run_summary.csv",
            "docs/NOISE_SOURCE_DECISION.md",
        ),
        "Plot TBUS environment-by-SNR descriptive means and sample SD",
        "controlled_simulated_noise_evaluation",
        "TBUS results retain the complete fixed-lambda SNR series.",
        "Does not establish real-farm external validity.",
    ),
    (
        "Supplementary Figure S5",
        "c",
        SUPPLEMENTARY_FIGURE_STEMS[4],
        (
            "paper/appendix/noise_25_run_summary.csv",
            "docs/NOISE_SOURCE_DECISION.md",
        ),
        "Plot STRAFFIC environment-by-SNR descriptive means and sample SD",
        "controlled_simulated_noise_evaluation",
        "STRAFFIC results retain the complete fixed-lambda SNR series.",
        "Does not establish real-farm external validity.",
    ),
    (
        "Supplementary Figure S5",
        "d",
        SUPPLEMENTARY_FIGURE_STEMS[4],
        (
            "paper/appendix/noise_25_run_summary.csv",
            "docs/NOISE_SOURCE_DECISION.md",
        ),
        "Plot the stored ALL_NOISY fixed-lambda=0.5 descriptive aggregate",
        "controlled_simulated_noise_evaluation",
        "The aggregate bounds average simulated-noise performance.",
        "Does not make repeated decisions independent recordings.",
    ),
    (
        "Supplementary Figure S6",
        "a",
        SUPPLEMENTARY_FIGURE_STEMS[5],
        (
            "paper/appendix/noise_25_run_aurc_augrc_summary.csv",
            "docs/NOISE_ROBUSTNESS_PROTOCOL.md",
        ),
        "Plot stored ALL_NOISY AURC means",
        "label_aware_selective_prediction_evaluation",
        "AURC compares uncertainty ranking descriptively.",
        "Does not show universal prototype uncertainty improvement.",
    ),
    (
        "Supplementary Figure S6",
        "b",
        SUPPLEMENTARY_FIGURE_STEMS[5],
        (
            "paper/appendix/noise_25_run_aurc_augrc_summary.csv",
            "docs/NOISE_ROBUSTNESS_PROTOCOL.md",
        ),
        "Plot stored ALL_NOISY AUGRC means",
        "label_aware_selective_prediction_evaluation",
        "AUGRC reports selective-ranking quality relative to generalized risk.",
        "Does not establish a deployment operating point.",
    ),
    (
        "Supplementary Figure S7",
        "a",
        SUPPLEMENTARY_FIGURE_STEMS[6],
        ("paper/final_validation/prototype_margin_error_detection.csv",),
        "Plot true-class favorable-distance margins by correctness",
        "explicitly_label_aware_diagnostic",
        "True-class margins diagnose prototype geometry after evaluation.",
        "This score is label-aware and is not deployable.",
    ),
    (
        "Supplementary Figure S7",
        "b",
        SUPPLEMENTARY_FIGURE_STEMS[6],
        ("paper/final_validation/prototype_margin_error_detection.csv",),
        "Plot subtype error AUROC from true-subtype favorable margins",
        "explicitly_label_aware_diagnostic",
        "Subtype true-class margins provide a bounded diagnostic comparison.",
        "This label-aware diagnostic is not a deployment claim.",
    ),
)


def _locked_selected_prediction_paths(root: Path) -> tuple[Path, ...]:
    selection = _read_csv(
        root,
        "paper/final_validation/lambda_selection_by_fold.csv",
        ("fold", "selected_lambda"),
    )
    selected = {
        int(row.fold): float(row.selected_lambda) for row in selection.itertuples()
    }
    return tuple(
        prototype_run_dir(root, fold, seed, selected[fold])
        / "evaluation"
        / "test_predictions.csv"
        for fold in EXPECTED_FOLDS
        for seed in EXPECTED_SEEDS
    )


def _case_source_paths(
    root: Path, analysis: Mapping[str, Any] | None
) -> tuple[Path, ...]:
    if analysis is not None:
        selected = analysis["selected_lambdas"]
    else:
        selection = _read_csv(
            root,
            "paper/final_validation/lambda_selection_by_fold.csv",
            ("fold", "selected_lambda"),
        )
        selected = {
            int(row.fold): float(row.selected_lambda) for row in selection.itertuples()
        }
    candidate = float(selected[0])
    run_dir = prototype_run_dir(root, 0, 42, candidate)
    return (
        run_dir / "evaluation" / "test_predictions.csv",
        run_dir / "calibration" / "val_predictions.csv",
        run_dir / "artifacts" / "train_main_prototype_samples.csv",
        run_dir / "artifacts" / "train_aux_prototype_samples.csv",
    )


def _resolve_panel_sources(
    root: Path,
    source_tokens: Sequence[str],
    analysis: Mapping[str, Any] | None,
) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for token in source_tokens:
        if token == "__selected_predictions__":
            if analysis is not None:
                resolved.extend(Path(path) for path in analysis["selected_prediction_paths"])
            else:
                resolved.extend(_locked_selected_prediction_paths(root))
        elif token == "__case_sources__":
            resolved.extend(_case_source_paths(root, analysis))
        else:
            resolved.append(root / token)
    unique = tuple(dict.fromkeys(path.resolve() for path in resolved))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required figure source files are missing: "
            + "; ".join(str(path) for path in missing)
        )
    return unique


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _figure_output_path(stem: str) -> str:
    return (MAIN_FIGURE_DIR / f"{stem}.svg").as_posix()


def build_source_map_rows(
    root: Path, analysis: Mapping[str, Any]
) -> pd.DataFrame:
    """Build one SHA-bearing provenance row for every planned panel."""

    root = root.resolve()
    rows: list[dict[str, object]] = []
    for (
        figure,
        panel,
        stem,
        source_tokens,
        transformation,
        deployability,
        _claim,
        _limitation,
    ) in _PANEL_SPECS:
        paths = _resolve_panel_sources(root, source_tokens, analysis)
        rows.append(
            {
                "figure": figure,
                "panel": panel,
                "source_path": ";".join(_relative(root, path) for path in paths),
                "source_sha256": ";".join(sha256_file(path) for path in paths),
                "deployability": deployability,
                "transformation": transformation,
                "output_file": _figure_output_path(stem),
            }
        )
    return pd.DataFrame(rows)


def build_claim_rows(root: Path | None = None) -> pd.DataFrame:
    """Build one evidence-and-limitation claim row for every planned panel."""

    repository = (root or _REPO_ROOT).resolve()
    rows: list[dict[str, object]] = []
    for (
        figure,
        panel,
        stem,
        source_tokens,
        transformation,
        deployability,
        claim,
        limitation,
    ) in _PANEL_SPECS:
        paths = _resolve_panel_sources(repository, source_tokens, None)
        rows.append(
            {
                "figure": figure,
                "panel": panel,
                "claim": claim,
                "evidence_file": ";".join(
                    _relative(repository, path) for path in paths
                ),
                "evidence_sha256": ";".join(sha256_file(path) for path in paths),
                "deployability": deployability,
                "transformation": transformation,
                "limitation": limitation,
                "output_file": _figure_output_path(stem),
            }
        )
    return pd.DataFrame(rows)


def collect_frozen_source_paths(
    root: Path, analysis: Mapping[str, Any], source_map: pd.DataFrame
) -> tuple[Path, ...]:
    """Return every frozen artifact loaded by analysis or cited by a figure panel."""

    root = root.resolve()
    paths = {Path(path).resolve() for path in analysis["source_paths"]}
    for value in source_map["source_path"]:
        paths.update((root / item).resolve() for item in str(value).split(";") if item)
    missing = sorted((path for path in paths if not path.is_file()), key=str)
    if missing:
        raise FileNotFoundError(
            "Frozen SHA-audit sources are missing: "
            + "; ".join(str(path) for path in missing)
        )
    return tuple(sorted(paths, key=str))


def _selection_source_paths(root: Path) -> tuple[Path, ...]:
    root = root.resolve()
    return tuple(
        summary_path(root, "hierarchical", fold, run_seed, candidate).resolve()
        for fold in EXPECTED_FOLDS
        for candidate in CANDIDATE_LAMBDAS
        for run_seed in EXPECTED_SEEDS
    )


def _selection_locator_paths(root: Path) -> tuple[Path, ...]:
    """Return the pre-hashed locked selection plus its validation summaries."""

    return (
        (root.resolve() / LOCKED_LAMBDA_SELECTION_CSV).resolve(),
        *_selection_source_paths(root),
    )


def discover_pre_analysis_source_paths(root: Path) -> tuple[Path, ...]:
    """Discover the complete frozen inventory before the main analysis pass.

    The approved lambda-selection CSV is the sole authority for choosing the
    already locked fold-wise run directories. The 75 validation summaries are
    read only to reproduce and verify that locked artifact. ``generate`` hashes
    all 76 locator files before calling this function, then hashes the returned
    full inventory before ``load_analysis`` consumes it.
    """

    root = root.resolve()
    _locked_selection, selected_lambdas = load_locked_lambda_selection(root)
    selection_records, _selection_hashes = load_lambda_selection_sources(root)
    _selection, reproduced_lambdas = select_lambda_by_fold(selection_records)
    validate_locked_lambda_reproduction(selected_lambdas, reproduced_lambdas)
    paths: set[Path] = set(_selection_locator_paths(root))
    prototype_relative_paths = (
        "softmax_reproduction/softmax_reproduction.json",
        "softmax_reproduction/sample_alignment_report.csv",
        "artifacts/prototype_metadata.json",
        "artifacts/prototype_bundle.npz",
        "artifacts/train_main_prototype_samples.csv",
        "artifacts/train_aux_prototype_samples.csv",
        "calibration/calibration.json",
        "calibration/val_predictions.csv",
        "evaluation/prediction_metadata.json",
        "evaluation/metrics.json",
        "evaluation/test_predictions.csv",
        "evaluation/test_predictions.json",
        "artifacts/leakage_audit.json",
        "calibration/leakage_audit.json",
        "evaluation/leakage_audit.json",
    )
    for fold, run_seed in EXPECTED_KEYS:
        paths.add(summary_path(root, "B0", fold, run_seed).resolve())
        paths.add(summary_path(root, "B1", fold, run_seed).resolve())
        for split in ("train", "val", "test"):
            paths.add(manifest_path(root, fold, split).resolve())
        for candidate in {float(selected_lambdas[fold]), 0.5}:
            paths.add(
                summary_path(
                    root, "hierarchical", fold, run_seed, candidate
                ).resolve()
            )
            paths.add(checkpoint_path(root, fold, run_seed, candidate).resolve())
            paths.add(raw_prediction_path(root, fold, run_seed, candidate).resolve())
            run_dir = prototype_run_dir(root, fold, run_seed, candidate)
            paths.update((run_dir / relative).resolve() for relative in prototype_relative_paths)
    for specification in _PANEL_SPECS:
        for token in specification[3]:
            if not token.startswith("__"):
                paths.add((root / token).resolve())
    missing = sorted((path for path in paths if not path.is_file()), key=str)
    if missing:
        raise FileNotFoundError(
            "Pre-analysis SHA inventory is missing frozen sources: "
            + "; ".join(str(path) for path in missing)
        )
    return tuple(sorted(paths, key=str))


def _format_p(value: float) -> str:
    return f"P={float(value):.9g}"


def build_legend_texts(analysis: Mapping[str, Any]) -> tuple[str, str]:
    """Return self-contained English and Chinese legends for all 11 figures."""

    paired = analysis["framework_paired"].set_index("comparison")
    b1 = paired.loc["B1 - B0"]
    b2 = paired.loc["B2_valsel - B1"]
    b3 = paired.loc["B3_valsel - B2_valsel"]
    retrospective_p = "; ".join(
        f"{row.comparison} {_format_p(row.wilcoxon_p)}"
        for row in analysis["cumulative_paired"].itertuples(index=False)
    )
    utility = _overall_rows(analysis["hierarchy_utility"]).set_index("method")
    enrichment_availability = ", ".join(
        f"{label} n={int(utility.loc[method, 'error_enrichment_ratio_n_runs_available'])}"
        for method, label in zip(
            ("raw_softmax", "main_prototype", "hierarchical_prototype"),
            ("Raw", "Main", "Hierarchical"),
            strict=True,
        )
    )
    common_en = (
        "Unless stated otherwise, n=25 denotes matched fold-seed runs; centre is "
        "the mean of five fold means and spread is the available-run sample SD. Intervals are deterministic "
        "fold-cluster bootstrap 95% CI after averaging seeds within each of five "
        "folds. Paired tests use two-sided Wilcoxon signed-rank P values from "
        "SciPy method=auto (asymptotic when zero differences preclude an exact "
        "null distribution). Numeric P values and the applicable Holm adjustment "
        "are reported without significance stars. A deployable "
        "score or flag is label-free at use time, whereas every correctness, error, "
        "AUROC, precision, recall, and true-class margin is label-aware evaluation. "
        "The evidence does not establish real-farm external validity or automatic routing."
    )
    figure_text_en = (
        (
            "Figure 1 | Final evidence-bounded framework",
            "Panels a-d separate path/source-ID/MD5-disjoint split roles, the 2-s Log-Mel shared hierarchical convolutional recurrent neural network (CRNN), and the inference roles. B2 Raw Softmax is the primary four-class route. In parallel, the frozen embedding feeds train-only main/subtype prototypes, Top-k candidates, distances, the predicted Top-1/Top-2 distance margin, hierarchy consistency and a prototype representative; calibration is validation-only and B3 remains an ablation. Here n=not applicable; no centre, SD, CI, Wilcoxon or Holm test applies. Pig-, session-, device- and farm-level grouping are not established by this schematic. Protocol elements are deployable in principle, but the schematic does not establish accuracy or external validity.",
        ),
        (
            "Figure 2 | Primary validation-selected classification evidence",
            f"Panel a shows all 25 run points plus mean ± SD Macro-F1 on a 0-1 axis. Panel b shows fold-cluster bootstrap 95% CI for exactly three matched increments. B1−B0 is {_format_p(b1['wilcoxon_raw_p'])} (Holm not applicable: prespecified duration contrast outside the locked prototype family); B2−B1 is {_format_p(b2['wilcoxon_raw_p'])} (Holm {_format_p(b2['holm_adjusted_p'])}); B3−B2 is {_format_p(b3['wilcoxon_raw_p'])} (Holm {_format_p(b3['holm_adjusted_p'])}). B2/B3 Holm values preserve the locked final-validation family of seven comparisons. B1 is the main supported gain; B2 is exploratory and non-significant; B3 does not improve Top-1. Panel c shows validation-selected lambda by fold. This label-aware test evidence does not establish a deployable performance guarantee.",
        ),
        (
            "Figure 3 | Prototype functional value",
            f"Panel a reports main/subtype Rank-1 and Rank-2 coverage and panel b reports feeding/stress Top-2 coverage. Panel c recomputes inconsistency prevalence, error precision, error recall and enrichment separately for Raw, Main and Hierarchical routes, displaying fold means and fold-cluster intervals; enrichment is undefined for zero-flag runs ({enrichment_availability}). Panel d uses the same four-main-prototype second-nearest-minus-nearest margin for Main and Hierarchical Prototype, with 25-run correct/error distributions and AUROC fold-cluster CI; no hierarchical distance vector exists. n=25 total runs; metric-specific available-run counts are shown where needed. Centre is the mean of fold means, spread is available-run SD, and fold-cluster bootstrap 95% CI is primary where shown. No new Wilcoxon test is used and Holm status is not applicable. Only the margin and inconsistency flag are deployable; outcomes and coverage are label-aware. Low prevalence and recall limit the inconsistency flag; this does not establish safe automatic rejection.",
        ),
        (
            "Figure 4 | Deterministic prototype candidate cases",
            "Seven unique frozen-test roles are selected from fold 0/seed 42: C1 cough correct, C2 calm-grunt correct, C3 feeding correct, C4 stress-vocal correct, C5 feeding-boundary, C6 stress-boundary, and C7 low predicted margin. Panel d uses the hierarchical-predicted main class; each prototype representative is a training sample closest to the predicted class prototype. Here n=7 illustrative cases; centre, SD, fold-cluster bootstrap 95% CI, Wilcoxon and Holm are not applicable. The displayed margin is deployable but case correctness is label-aware. Cases do not establish prevalence.",
        ),
        (
            "Supplementary Figure S1 | Complete clean ablations",
            "Each row reports n=15 or n=25 fold-seed runs as mean ± sample SD; no fold-cluster bootstrap 95% CI, matched Wilcoxon test or Holm comparison is available for unmatched rows. Results are label-aware and do not establish a deployable advantage for an alternative clean frontend.",
        ),
        (
            "Supplementary Figure S2 | Fixed-lambda retrospective cumulative evidence",
            f"Panel a reports all n=25 fixed-lambda=0.5 fold-seed trajectories and their mean; panel b reports matched mean deltas with stored 10,000-resample run bootstrap CI and five multiplicity-unadjusted numeric two-sided Wilcoxon P values ({retrospective_p}). Holm status is not used for this retrospective family. This run-level inference is descriptive, label-aware and does not replace the validation-selected primary result or establish a deployable gain.",
        ),
        (
            "Supplementary Figure S3 | Lambda selection by fold",
            "Each candidate has n=5 training-seed validation optima per fold; centre is mean and spread is sample SD. No test-set value, fold-cluster bootstrap 95% CI, Wilcoxon test or Holm decision enters selection. Selection is validation-deployable but does not establish one globally optimal lambda.",
        ),
        (
            "Supplementary Figure S4 | Best epoch and validation-test gap",
            "Boxplots summarise n=25 runs per stage using medians and interquartile ranges; no mean ± SD is encoded, and no fold-cluster bootstrap 95% CI, Wilcoxon or Holm test is added. Best epochs are validation-selected and gaps are label-aware frozen-test diagnostics; they do not justify test tuning.",
        ),
        (
            "Supplementary Figure S5 | DEMAND simulated noise",
            "Points and bars are n=25 fold-seed run means with sample SD under controlled additive DEMAND noise: DWASHING denotes domestic washroom/washing-machine noise, TBUS public-transit bus noise and STRAFFIC busy street-traffic noise. Panel d is the stored fixed-lambda=0.5 ALL_NOISY aggregate. No new fold-cluster bootstrap 95% CI, Wilcoxon or Holm test is added here. Performance is label-aware; noise generation is deployable as a test protocol but does not establish real-farm external validity.",
        ),
        (
            "Supplementary Figure S6 | AURC and AUGRC",
            "ALL_NOISY area under the risk-coverage curve (AURC) and area under the generalized risk-coverage curve (AUGRC) are descriptive n=25 means; lower is better. Spread, fold-cluster bootstrap 95% CI, paired Wilcoxon and Holm results are unavailable for these stored summaries. Scores are deployable uncertainty rankings, but risk outcomes are label-aware and do not establish a universal operating point.",
        ),
        (
            "Supplementary Figure S7 | Label-aware true-class margins",
            "True-main and true-subtype favorable-distance margins use n=25 runs; centre is the mean of run medians and variability is the displayed fold-cluster bootstrap 95% CI (SD is not encoded). No additional Wilcoxon or Holm test is made. These quantities explicitly use the true label, are not deployable, and do not establish a hierarchical decision distance.",
        ),
    )
    english = ["# Final Functional Figure Legends (English)", ""]
    for title, body in figure_text_en:
        english.extend((f"## {title}", "", body, ""))
    english.extend(("## Statistical and evidence conventions", "", common_en, ""))

    common_cn = (
        "除另有说明外，n=25 表示匹配的 fold-seed 运行；中心为 5 个 fold 均值的平均，离散度为可用运行的样本 SD。"
        "区间为先在 5 个 fold 内平均种子后进行的 fold-cluster bootstrap 95% CI。"
        "配对检验使用 SciPy method=auto 的双侧 Wilcoxon 符号秩检验；存在零差时使用渐近 P 值。"
        "报告数值 P 值与适用的 Holm 校正，不使用显著性星号。"
        "可部署评分或标志在使用时不依赖真实标签；正确性、错误率、AUROC、精确率、"
        "召回率与真类间隔均属于标签感知评估。证据不证明真实猪场外部有效性或自动路由。"
    )
    chinese_sections = (
        ("图 1｜最终证据边界框架", "a-d 分别给出 path/source-ID/MD5 互斥的数据角色、2 s Log-Mel 共享层级卷积循环神经网络（CRNN）与推理角色。B2 Raw Softmax 是主要四类路由；并行的冻结嵌入进入仅训练集主类/子类原型，输出 Top-k 候选、距离、预测 Top-1/Top-2 距离间隔、层级一致性与原型代表；校准仅使用验证集，B3 仍为消融。n 不适用；均值、SD、CI、Wilcoxon 与 Holm 均不适用。协议要素原则上可部署，但本图未证明猪个体、录音会话、设备或农场层面的分组独立性，也不证明准确率或外部有效性。"),
        ("图 2｜验证选择的主要分类证据", f"a 在 0-1 纵轴上显示全部 n=25 个运行点及 Macro-F1 均值±SD；b 显示恰好三个匹配增量及 fold-cluster bootstrap 95% CI。B1−B0：{_format_p(b1['wilcoxon_raw_p'])}，Holm 不适用（预先设定的时长对比）；B2−B1：{_format_p(b2['wilcoxon_raw_p'])}，Holm {_format_p(b2['holm_adjusted_p'])}；B3−B2：{_format_p(b3['wilcoxon_raw_p'])}，Holm {_format_p(b3['holm_adjusted_p'])}。B2/B3 保留锁定的七比较 Holm family。B1 是主要支持增益；B2 为探索性且不显著；B3 不改善 Top-1。c 显示逐 fold 的验证选择 λ，未使用测试集。这些标签感知检验不构成可部署性能保证。"),
        ("图 3｜原型功能价值", f"a 报告主类/子类 Rank-1 与 Rank-2 覆盖，b 报告 feeding/stress Top-2 覆盖。c 分别重算 Raw、Main、Hierarchical 的不一致率、错误精确率、错误召回率与富集比；无标志运行的富集比不可定义（{enrichment_availability}）。d 中 Main 与 Hierarchical 共享同一个四主类原型的第二近减最近距离，只按各自正确/错误划分；不存在层级余弦距离向量。总运行数 n=25，必要时显示指标特定的可用运行数；中心为 fold 均值的平均，离散为可用运行 SD，区间为 fold-cluster bootstrap 95% CI。无新增 Wilcoxon，Holm 不适用。间隔与不一致标志在使用时可部署，正确性、精确率、召回率和 AUROC 为标签感知评估。低流行率与低召回率限制了该标志，不证明安全自动拒识。"),
        ("图 4｜确定性候选案例", "fold 0/seed 42 的 7 个角色为：C1 cough 正确例、C2 calm-grunt 正确例、C3 feeding 正确例、C4 stress-vocal 正确例、C5 feeding 边界例、C6 stress 边界例及 C7 低预测间隔例。d 使用层级路由预测的主类；原型代表样本定义为 a training sample closest to the predicted class prototype。n=7；均值、SD、CI、Wilcoxon 与 Holm 不适用。预测距离间隔在使用时可部署，案例正确性为标签感知；案例不估计总体流行率。"),
        ("补充图 S1｜完整干净消融", "每行 n=15 或 n=25，以均值±样本 SD 表示；不匹配行无 fold-cluster bootstrap 95% CI、Wilcoxon 或 Holm 推断。标签感知结果不证明替代前端具有可部署优势。"),
        ("补充图 S2｜固定 λ 回顾性累积证据", f"a 显示全部 n=25 条 fixed-lambda=0.5 fold-seed 轨迹及其均值；b 为配对均值差与存储的 10,000 次 run-bootstrap CI、五个未进行多重性校正的双侧 Wilcoxon 数值 P 值（{retrospective_p}）。该回顾性族未用 Holm；这是描述性、标签感知的运行层面分析，不替代验证选择的主要分析，也不证明可部署增益。"),
        ("补充图 S3｜逐 fold 的 λ 选择", "每个候选在每个 fold 有 n=5 个种子验证最优值，显示均值±SD。选择不使用测试值，也不使用 fold-cluster CI、Wilcoxon 或 Holm；不证明全局最优 λ。"),
        ("补充图 S4｜最佳 epoch 与验证-测试差", "每阶段 n=25，以中位数与四分位距表示；未新增均值±SD、fold-cluster CI、Wilcoxon 或 Holm。差值为标签感知诊断，不允许测试集调参。"),
        ("补充图 S5｜DEMAND 模拟噪声", "受控加性 DEMAND 噪声下 n=25，显示均值±SD。DWASHING 为含前开式洗衣机运转声的家庭洗衣间录音，TBUS 为公共交通巴士录音，STRAFFIC 为繁忙街道交通路口录音；d 为存储的 fixed-lambda=0.5 ALL_NOISY 聚合。未新增 fold-cluster CI、Wilcoxon 或 Holm。标签感知性能不证明真实猪场外部有效性。"),
        ("补充图 S6｜AURC 与 AUGRC", "ALL_NOISY 的风险-覆盖曲线下面积（AURC）与广义风险-覆盖曲线下面积（AUGRC）为描述性 n=25 均值，越低越好；无可用 SD、fold-cluster CI、Wilcoxon 或 Holm。排序分数在使用时可部署，风险结局为标签感知；不证明通用工作点。"),
        ("补充图 S7｜标签感知真类间隔", "n=25；中心为运行中位数的均值，变异性由图中的 fold-cluster bootstrap 95% CI 表示（未编码 SD）；无新增 Wilcoxon 或 Holm。该量使用真实标签、不可部署，也不是层级决策距离。"),
    )
    chinese = ["# 最终功能图图注（中文）", ""]
    for title, body in chinese_sections:
        chinese.extend((f"## {title}", "", body, ""))
    chinese.extend(("## 统计与证据约定", "", common_cn, ""))
    return "\n".join(english).strip() + "\n", "\n".join(chinese).strip() + "\n"


def required_output_paths(root: Path) -> tuple[Path, ...]:
    """Return the complete, unique output inventory for no-overwrite checks."""

    root = root.resolve()
    paths: list[Path] = [
        root / FINAL_VALIDATION_DIR / filename for filename in DERIVED_TABLE_FILENAMES
    ]
    for stem in MAIN_FIGURE_STEMS + SUPPLEMENTARY_FIGURE_STEMS:
        paths.extend(
            root / MAIN_FIGURE_DIR / f"{stem}.{extension}"
            for extension in FIGURE_EXTENSIONS
        )
    paths.extend(
        root / MAIN_FIGURE_DIR / filename
        for filename in (
            "FIGURE_LEGENDS_EN.md",
            "FIGURE_LEGENDS_CN.md",
            "FIGURE_SOURCE_MAP.csv",
            "FIGURE_TO_CLAIM_MATRIX.csv",
            "FIGURE_QA_REPORT.md",
        )
    )
    return tuple(paths)


def refuse_existing(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing final-package outputs: "
            + "; ".join(str(path) for path in existing)
        )


def _atomic_replace(source: Path, target: Path) -> Path:
    """Small wrapper kept injectable for promotion rollback tests."""

    return source.replace(target)


def _promote_staged_outputs(
    *,
    root: Path,
    staged_figure_dir: Path,
    table_pairs: Sequence[tuple[Path, Path]],
) -> tuple[Path, ...]:
    """Promote a verified staging package and roll back every partial move."""

    root = root.resolve()
    final_figure_dir = (root / MAIN_FIGURE_DIR).resolve()
    staged_figure_dir = staged_figure_dir.resolve()
    if not staged_figure_dir.is_dir():
        raise FileNotFoundError(f"Staged figure directory is missing: {staged_figure_dir}")
    normalized_pairs = tuple(
        (source.resolve(), target.resolve()) for source, target in table_pairs
    )
    missing = [source for source, _target in normalized_pairs if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "Staged derived tables are missing: " + "; ".join(map(str, missing))
        )
    targets = (final_figure_dir, *(target for _source, target in normalized_pairs))
    refuse_existing(targets)
    final_figure_dir.parent.mkdir(parents=True, exist_ok=True)
    for _source, target in normalized_pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
    promoted: list[Path] = []
    try:
        _atomic_replace(staged_figure_dir, final_figure_dir)
        promoted.append(final_figure_dir)
        for source, target in normalized_pairs:
            _atomic_replace(source, target)
            promoted.append(target)
    except Exception:
        for path in reversed(promoted):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        raise
    return tuple(promoted)


def inspect_export(path: Path) -> dict[str, object]:
    """Inspect one export for size, editability, DPI, background and content."""

    if not path.is_file() or path.stat().st_size <= 1024:
        raise RuntimeError(f"Figure export is missing or unexpectedly small: {path}")
    result: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "status": "PASS",
    }
    suffix = path.suffix.lower()
    if suffix == ".svg":
        text = path.read_text(encoding="utf-8")
        text_nodes = len(re.findall(r"<text\b", text))
        width_match = re.search(r'width="([0-9.]+)pt"', text)
        if text_nodes == 0:
            raise RuntimeError(f"SVG text was converted to paths: {path}")
        if not width_match:
            raise RuntimeError(f"Could not verify SVG width: {path}")
        width_mm = float(width_match.group(1)) * 25.4 / 72.0
        if not math.isclose(width_mm, FIGURE_WIDTH_MM, abs_tol=0.05):
            raise RuntimeError(
                f"SVG width is {width_mm:.3f} mm, expected {FIGURE_WIDTH_MM} mm: {path}"
            )
        result.update({"text_nodes": text_nodes, "width_mm": width_mm})
    elif suffix in {".png", ".tiff", ".tif"}:
        expected_dpi = 300 if suffix == ".png" else 600
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            if not dpi or any(abs(float(value) - expected_dpi) > 1.0 for value in dpi):
                raise RuntimeError(
                    f"Raster DPI is {dpi}, expected {expected_dpi}: {path}"
                )
            expected_width = FIGURE_WIDTH_MM / 25.4 * expected_dpi
            if abs(image.width - expected_width) > 2:
                raise RuntimeError(
                    f"Raster width is {image.width}px, expected {expected_width:.1f}px: {path}"
                )
            array = np.asarray(image.convert("RGB"))
            if not np.any(array < 250):
                raise RuntimeError(f"Raster export is blank: {path}")
            corners = np.asarray(
                [array[0, 0], array[0, -1], array[-1, 0], array[-1, -1]]
            )
            if not np.all(corners >= 250):
                raise RuntimeError(f"Raster export does not have a white background: {path}")
            result.update(
                {
                    "pixels": f"{image.width}x{image.height}",
                    "dpi": f"{float(dpi[0]):.3f}x{float(dpi[1]):.3f}",
                }
            )
    return result


def save_figure_bundle(
    figure: plt.Figure, output_dir: Path, stem: str
) -> tuple[Path, ...]:
    """Export editable SVG/PDF, 300-dpi PNG and 600-dpi TIFF."""

    output_dir.mkdir(parents=True, exist_ok=True)
    formats: tuple[tuple[str, dict[str, object]], ...] = (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
        (
            "tiff",
            {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
        ),
    )
    paths: list[Path] = []
    for suffix, options in formats:
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(
            path,
            facecolor=WHITE,
            edgecolor="none",
            bbox_inches=None,
            **options,
        )
        if suffix == "svg":
            svg_text = path.read_text(encoding="utf-8")
            normalized_svg = re.sub(
                r"[ \t]+(?=\r?$)", "", svg_text, flags=re.MULTILINE
            )
            path.write_text(normalized_svg, encoding="utf-8")
        inspect_export(path)
        paths.append(path)
    return tuple(paths)


def _qa_report(
    exports: Sequence[Path], source_map: pd.DataFrame, *, frozen_source_count: int
) -> str:
    inspections = [inspect_export(path) for path in exports]
    lines = [
        "# Final Functional Figure Package QA",
        "",
        "## Machine checks",
        "",
        "All exports were checked for non-empty content, fixed 183-mm width, white background, editable SVG text, 300-dpi PNG, and 600-dpi TIFF.",
        "",
        "| File | Bytes | Detail | Status |",
        "|---|---:|---|---|",
    ]
    for item in inspections:
        detail = item.get("pixels", f"{item.get('text_nodes', 'vector')} text nodes")
        display_path = (MAIN_FIGURE_DIR / Path(str(item["path"])).name).as_posix()
        lines.append(
            f"| {display_path} | {item['bytes']} | {detail} | {item['status']} |"
        )
    lines.extend(
        (
            "",
            "## Provenance checks",
            "",
            f"- Panel rows: {len(source_map)}; duplicate figure/panel rows: {int(source_map[['figure', 'panel']].duplicated().sum())}.",
            "- Every panel row records source paths and freshly computed SHA-256 values.",
            f"- Full frozen-source audit: {frozen_source_count} files hashed before and after generation; no digest changed.",
            "- Shared prototype margin scope: `shared_main_prototype_geometry`.",
            "- No mapped-subtype or hierarchical cosine-distance vector was constructed.",
            "- No training, checkpoint inference, Atlas generation, or test-set tuning was performed.",
            "",
        )
    )
    return "\n".join(lines)


def generate(
    root: Path,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Validate or generate the package without modifying frozen sources."""

    root = root.resolve()
    planned = required_output_paths(root)
    if not validate_only:
        refuse_existing(planned)
    locator_paths = _selection_locator_paths(root)
    missing_locators = [path for path in locator_paths if not path.is_file()]
    if missing_locators:
        raise FileNotFoundError(
            "Lambda-selection locator sources are missing: "
            + "; ".join(map(str, missing_locators))
        )
    locator_hashes_before = {path: sha256_file(path) for path in locator_paths}
    pre_analysis_paths = discover_pre_analysis_source_paths(root)
    locator_hashes_after_discovery = {
        path: sha256_file(path) for path in locator_paths
    }
    if locator_hashes_before != locator_hashes_after_discovery:
        raise RuntimeError("Lambda-selection sources changed during source discovery")
    pre_analysis_hashes = {
        path: locator_hashes_before.get(path, sha256_file(path))
        for path in pre_analysis_paths
    }
    analysis = load_analysis(root, n_boot=n_boot, seed=seed)
    source_map = build_source_map_rows(root, analysis)
    claims = build_claim_rows(root)
    english, chinese = build_legend_texts(analysis)
    evidence_paths = collect_frozen_source_paths(root, analysis, source_map)
    undiscovered = sorted(set(evidence_paths) - set(pre_analysis_paths), key=str)
    if undiscovered:
        raise RuntimeError(
            "Analysis consumed frozen sources outside the pre-analysis SHA inventory: "
            + "; ".join(map(str, undiscovered))
        )
    source_hashes_before = {path: pre_analysis_hashes[path] for path in evidence_paths}
    source_hashes_after_analysis = {
        path: sha256_file(path) for path in evidence_paths
    }
    if source_hashes_before != source_hashes_after_analysis:
        changed = [
            _relative(root, path)
            for path, digest in source_hashes_before.items()
            if source_hashes_after_analysis.get(path) != digest
        ]
        raise RuntimeError(f"Frozen sources changed during analysis: {changed}")
    figures = build_all_figures(root, analysis)
    if validate_only:
        for figure in figures.values():
            plt.close(figure)
        source_hashes_after = {
            path: sha256_file(path) for path in evidence_paths
        }
        if source_hashes_before != source_hashes_after:
            raise RuntimeError("Frozen figure sources changed during validation")
        return {
            "validated": True,
            "written": (),
            "figure_count": len(figures),
            "panel_count": len(source_map),
            "frozen_source_count": len(evidence_paths),
            "pre_analysis_sha_locked": True,
        }

    table_frames = (
        analysis["deployable_margins"],
        analysis["hierarchy_utility"],
        analysis["functional_final"],
    )
    try:
        paper_dir = root / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".final_functional_staging_", dir=paper_dir
        ) as temporary:
            staging_root = Path(temporary)
            staged_validation = staging_root / "final_validation"
            staged_figures = staging_root / "figures_final"
            staged_validation.mkdir(parents=True)
            staged_figures.mkdir(parents=True)
            table_pairs: list[tuple[Path, Path]] = []
            for filename, frame in zip(
                DERIVED_TABLE_FILENAMES, table_frames, strict=True
            ):
                staged_path = staged_validation / filename
                frame.to_csv(staged_path, index=False, encoding="utf-8")
                table_pairs.append(
                    (staged_path, root / FINAL_VALIDATION_DIR / filename)
                )
            source_map.to_csv(
                staged_figures / "FIGURE_SOURCE_MAP.csv",
                index=False,
                encoding="utf-8",
            )
            claims.to_csv(
                staged_figures / "FIGURE_TO_CLAIM_MATRIX.csv",
                index=False,
                encoding="utf-8",
            )
            for filename, content in (
                ("FIGURE_LEGENDS_EN.md", english),
                ("FIGURE_LEGENDS_CN.md", chinese),
            ):
                (staged_figures / filename).write_text(content, encoding="utf-8")
            exports: list[Path] = []
            for stem, figure in figures.items():
                exports.extend(save_figure_bundle(figure, staged_figures, stem))
                plt.close(figure)
            source_hashes_after_export = {
                path: sha256_file(path) for path in evidence_paths
            }
            if source_hashes_before != source_hashes_after_export:
                changed = [
                    _relative(root, path)
                    for path, digest in source_hashes_before.items()
                    if source_hashes_after_export.get(path) != digest
                ]
                raise RuntimeError(
                    f"Frozen sources changed during staged generation: {changed}"
                )
            (staged_figures / "FIGURE_QA_REPORT.md").write_text(
                _qa_report(
                    exports, source_map, frozen_source_count=len(evidence_paths)
                ),
                encoding="utf-8",
            )
            expected_staged = [
                *(staged_validation / name for name in DERIVED_TABLE_FILENAMES),
                *(
                    staged_figures / path.name
                    for path in planned
                    if path.parent.resolve() == (root / MAIN_FIGURE_DIR).resolve()
                ),
            ]
            incomplete = [
                path
                for path in expected_staged
                if not path.is_file() or path.stat().st_size == 0
            ]
            if incomplete:
                raise RuntimeError(
                    "Staged final package is incomplete: "
                    + "; ".join(map(str, incomplete))
                )
            promoted = _promote_staged_outputs(
                root=root,
                staged_figure_dir=staged_figures,
                table_pairs=tuple(table_pairs),
            )
            source_hashes_after_promotion = {
                path: sha256_file(path) for path in evidence_paths
            }
            if source_hashes_before != source_hashes_after_promotion:
                for path in reversed(promoted):
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.exists():
                        path.unlink()
                raise RuntimeError("Frozen sources changed during package promotion")
    finally:
        for figure in figures.values():
            plt.close(figure)
    return {
        "validated": True,
        "written": tuple(_relative(root, path) for path in planned),
        "figure_count": len(figures),
        "panel_count": len(source_map),
        "frozen_source_count": len(evidence_paths),
        "pre_analysis_sha_locked": True,
        "failure_atomic_promotion": True,
        "source_hashes_unchanged": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen-only final functional figure package from "
            "validation-selected prediction artifacts. No training or inference is run."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
        help=f"Fold-cluster bootstrap resamples (default: {BOOTSTRAP_RESAMPLES}).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
        help=f"Deterministic bootstrap seed (default: {BOOTSTRAP_SEED}).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all sources and build figures in memory without writing outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = generate(
        args.root,
        n_boot=args.n_boot,
        seed=args.bootstrap_seed,
        validate_only=args.validate_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


# Descriptive aliases keep every public builder discoverable by its output stem.
build_figure1_final_framework = build_figure1
build_figure2_primary_classification_evidence = build_figure2
build_figure3_prototype_functional_value = build_figure3
build_figure4_prototype_candidate_cases = build_figure4
build_figure_s1_complete_clean_ablations = build_figure_s1
build_figure_s2_fixed_lambda_retrospective_cumulative = build_figure_s2
build_figure_s3_lambda_selection_by_fold = build_figure_s3
build_figure_s4_best_epoch_validation_test_gap = build_figure_s4
build_figure_s5_demand_simulated_noise = build_figure_s5
build_figure_s6_aurc_augrc = build_figure_s6
build_figure_s7_label_aware_true_class_margin = build_figure_s7


if __name__ == "__main__":
    main()
