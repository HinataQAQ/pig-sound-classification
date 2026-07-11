# Final Functional Figure Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive deployable prototype-margin and hierarchy-inconsistency statistics from frozen validation-selected predictions, then produce four main and seven supplementary publication figures without training, inference, or changes to established result CSV/JSON files.

**Architecture:** Add one frozen-only Python generator that reuses validated loaders and statistics from `tools/generate_final_validation_audit.py`. A focused unittest module defines the statistics, output inventory, case-selection, provenance, legend, and figure contracts before implementation. The generator snapshots every source artifact before and after analysis and writes only the approved new paths.

**Tech Stack:** Python, pandas, NumPy, SciPy, scikit-learn, matplotlib, Pillow, unittest; PowerShell through the `pigsound-gpu` Conda environment.

## Global Constraints

- Read existing frozen artifacts only; never train, run checkpoint inference, build Atlas artifacts, or select a new hyperparameter.
- B2 validation-selected Raw Softmax is primary; prototypes are a parallel candidate/diagnostic layer; B3 is an ablation.
- Preserve every existing validation/result CSV and JSON byte-for-byte.
- Use folds 0-4 and seeds 42, 123, 777, 2024, 3407 with matched fold-seed aggregation.
- Use 10,000 deterministic fold-cluster bootstrap resamples and seed 3407 unless overridden through argparse.
- Keep SVG text editable; export PDF, 300-dpi PNG, and 600-dpi TIFF on white with a consistent sans-serif stack.
- Report exact P values and Holm adjustment; never call the B2 increment or B3 increment significant.

---

### Task 1: Lock statistics and output contracts with failing tests

**Files:**
- Create: `tests/test_generate_final_functional_figure_package.py`

**Interfaces:**
- Consumes: frozen test-prediction-shaped pandas DataFrames.
- Produces: contracts for `predicted_distance_margin`, run/aggregate metrics, deterministic cases, outputs, figures, maps, and legends.

- [ ] **Step 1: Write failing tests.**

```python
def test_predicted_margin_sorts_distances_without_truth_labels():
    values = np.array([[0.40, 0.10, 0.25, 0.90]])
    np.testing.assert_allclose(predicted_distance_margin(values), [0.15])

def test_hierarchy_utility_reports_complete_detection_utility():
    row = compute_run_hierarchy_utility(_utility_fixture(), fold=0, seed=42)
    required = {
        "inconsistency_prevalence", "error_rate_inconsistent",
        "error_rate_consistent", "error_enrichment_ratio",
        "inconsistency_error_precision", "inconsistency_error_recall",
    }
    assert required.issubset(row.columns)
```

- [ ] **Step 2: Run RED.**

Run: `conda run -n pigsound-gpu python -m unittest tests.test_generate_final_functional_figure_package`

Expected: import failure because the generator does not exist.

- [ ] **Step 3: Add tests for the exact three CSVs, four main/seven supplementary stems, seven case roles, complete source/claim coverage, 0-1 primary Macro-F1 axes, editable SVG settings, legend minimums, no-overwrite guards, and clear missing-source errors.**

### Task 2: Implement frozen-only statistics and deterministic cases

**Files:**
- Create: `tools/generate_final_functional_figure_package.py`
- Test: `tests/test_generate_final_functional_figure_package.py`

**Interfaces:**
- Consumes: validated selected-lambda loaders, selected prediction frames, and train-only representative CSVs.
- Produces: the three requested CSVs.

- [ ] **Step 1: Implement the label-free distance margin.**

```python
def predicted_distance_margin(distances: np.ndarray) -> np.ndarray:
    matrix = np.asarray(distances, dtype=np.float64)
    ordered = np.sort(matrix, axis=1, kind="stable")
    return ordered[:, 1] - ordered[:, 0]
```

Both routes use the same four-main-prototype cosine-distance gap because the
frozen hierarchical rule mixes main and mapped-subtype probabilities and has no
method-specific cosine-distance vector. Correct/error strata use
`prototype_pred` for Main Prototype and `hierarchical_pred` for Hierarchical
Prototype. The output must call this a shared deployable main-prototype geometry
margin, not a hierarchical decision distance.

- [ ] **Step 2: Implement fold-seed, fold-mean, and overall summaries with fold-cluster CIs for correct/error medians, error AUROC, inconsistency prevalence, conditional errors, enrichment, precision, and recall.**

- [ ] **Step 3: Adapt the existing fold-0/seed-42 case protocol to the validation-selected run. Change only the seventh role to the minimum remaining shared main-prototype predicted-distance margin and retain the train-only representative definition.**

- [ ] **Step 4: Run the focused tests and verify GREEN.**

### Task 3: Implement figures, legends, maps, and QA

**Files:**
- Modify: `tools/generate_final_functional_figure_package.py`
- Test: `tests/test_generate_final_functional_figure_package.py`

**Interfaces:**
- Consumes: new statistics and locked framework, ablation, noise, selective-prediction, and label-aware tables.
- Produces: 11 figures plus bilingual legends, source/claim maps, and QA.

- [ ] **Step 1: Implement four main figure builders.**

```python
MAIN_FIGURE_STEMS = (
    "figure1_final_framework",
    "figure2_primary_classification_evidence",
    "figure3_prototype_functional_value",
    "figure4_prototype_candidate_cases",
)
```

- [ ] **Step 2: Implement seven supplementary figure builders.**

```python
SUPPLEMENTARY_FIGURE_STEMS = (
    "figure_s1_complete_clean_ablations",
    "figure_s2_fixed_lambda_retrospective_cumulative",
    "figure_s3_lambda_selection_by_fold",
    "figure_s4_best_epoch_validation_test_gap",
    "figure_s5_demand_simulated_noise",
    "figure_s6_aurc_augrc",
    "figure_s7_label_aware_true_class_margin",
)
```

- [ ] **Step 3: Add fixed 183-mm export, editable SVG text, PDF, 300-dpi PNG, 600-dpi TIFF, and machine checks for size, DPI, white background, and non-empty content.**

- [ ] **Step 4: Add self-contained bilingual legends and a SHA-bearing source/claim row for every panel. Each legend states n, centre/spread, CI, test, Holm status, deployability, and what is not established.**

- [ ] **Step 5: Run the focused tests and verify GREEN.**

### Task 4: Generate, visually audit, and independently review

**Files:**
- Create: `docs/FINAL_FUNCTIONAL_FIGURE_PACKAGE.md`
- Create: requested final-validation/figure artifacts.
- Create: `paper/review/FINAL_FUNCTIONAL_FIGURE_AUDIT.md`

- [ ] **Step 1: Run `--help` and `--validate-only`.**

Run: `conda run -n pigsound-gpu python tools/generate_final_functional_figure_package.py --help`

Run: `conda run -n pigsound-gpu python tools/generate_final_functional_figure_package.py --root . --validate-only`

- [ ] **Step 2: Generate once into the unique approved paths.**

Run: `conda run -n pigsound-gpu python tools/generate_final_functional_figure_package.py --root .`

- [ ] **Step 3: Inspect all PNGs at original resolution and record clipping, overlap, axis, and semantic checks in the QA report.**

- [ ] **Step 4: Apply the ARS reviewer workflow read-only and write `paper/review/FINAL_FUNCTIONAL_FIGURE_AUDIT.md` without modifying the outputs under review.**

- [ ] **Step 5: Run focused plus regression unittests.**

Run: `conda run -n pigsound-gpu python -m unittest tests.test_generate_final_functional_figure_package tests.test_generate_final_validation_audit tests.test_generate_journal_figure_package tests.test_generate_cumulative_framework_analysis`

### Task 5: Provenance, handoff, commit, and push

**Files:**
- Modify: `handoff/CODEX_TO_GPT.md`
- Modify: `handoff/CODEX_TO_GPT.json`
- Modify: `handoff/HISTORY.md`

- [ ] **Step 1: Verify no pre-existing result CSV/JSON changed, all exports are non-empty, SVGs retain text nodes, PNG/TIFF DPI is correct, and all source SHA-256 values match.**

- [ ] **Step 2: Update the three GPT handoff files with commands, branch/commit, changed files, metrics, leakage boundary, paper usability, blockers, and scientific questions.**

- [ ] **Step 3: Run `git diff --check` and the complete verification command immediately before commit.**

- [ ] **Step 4: Stage the explicit files, commit `feat: add final functional figure package`, push `paper/final-figures-functional-framework`, and stop.**
