# Duration-Aware Hierarchical Prototype Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the locked B0-B3 cumulative Macro-F1 analysis, deterministic prototype case studies, and publication exports without retraining or changing existing results.

**Architecture:** One standalone Python generator reads and validates the exact 25-run frozen artifacts, emits normalized CSV source data, and renders both figures from those normalized frames. Pure helper functions expose the validation, statistics, and selection logic to focused unit tests; the CLI adds only root, validation-only, and fixed-bootstrap reproducibility controls.

**Tech Stack:** Python 3, pandas, NumPy, SciPy, Matplotlib, Pillow, unittest, Windows PowerShell, Conda `pigsound-gpu`.

## Global Constraints

- No model retraining, inference rerun, checkpoint mutation, audio mutation, or existing CSV/JSON modification.
- Use exactly folds `0..4`, seeds `42, 123, 777, 2024, 3407`, and hierarchical auxiliary lambda `0.5`.
- Pair strictly by `(fold, seed)` and compute B1-B0, B2-B1, B3-B2, B3-B1, and B3-B0 directly.
- Use Macro-F1 terminology; never relabel it as accuracy.
- Use 10,000 bootstrap resamples and seed 3407 for paper outputs.
- Use Python/Matplotlib exclusively for SVG, PDF, PNG, previews, and visual QA.
- Preserve every pre-existing untracked file and stage only the scoped deliverables.

---

### Task 1: Test the locked analysis contract

**Files:**
- Create: `tests/test_generate_cumulative_framework_analysis.py`
- Create: `tools/generate_cumulative_framework_analysis.py`

**Interfaces:**
- Consumes: repository root and frozen result files defined in the design.
- Produces: `load_cumulative_runs(root)`, `compute_stage_summary(runs)`, `compute_paired_statistics(runs)`, `select_case_studies(root)`, `build_cumulative_figure(...)`, and `build_case_figure(...)`.

- [ ] **Step 1: Write the failing import and contract tests**

```python
from tools import generate_cumulative_framework_analysis as cumulative

def test_locked_runs_are_exactly_25_matched_rows(self):
    runs = cumulative.load_cumulative_runs(ROOT)
    self.assertEqual(len(runs), 25)
    self.assertEqual(set(runs.columns) >= {"fold", "seed", "B0", "B1", "B2", "B3"}, True)
    self.assertAlmostEqual(runs["B0"].mean(), 0.9226064673362835)
    self.assertAlmostEqual(runs["B3"].mean(), 0.9539862142299651)

def test_direct_comparisons_include_b3_minus_b0(self):
    paired, folds = cumulative.compute_paired_statistics(cumulative.load_cumulative_runs(ROOT))
    self.assertEqual(paired["comparison"].tolist(), ["B1 - B0", "B2 - B1", "B3 - B2", "B3 - B1", "B3 - B0"])
    self.assertEqual(folds.groupby("comparison").size().to_dict()["B3 - B0"], 5)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `$env:PYTHONNOUSERSITE='1'; conda run -n pigsound-gpu python -m unittest tests.test_generate_cumulative_framework_analysis -v`

Expected: import failure because `tools.generate_cumulative_framework_analysis` does not exist.

- [ ] **Step 3: Add the minimal module shell and constants**

```python
EXPECTED_FOLDS = tuple(range(5))
EXPECTED_SEEDS = (42, 123, 777, 2024, 3407)
STAGE_ORDER = ("B0", "B1", "B2", "B3")
COMPARISONS = (("B1", "B0"), ("B2", "B1"), ("B3", "B2"), ("B3", "B1"), ("B3", "B0"))
```

- [ ] **Step 4: Implement strict source loading and direct statistics**

Read each fold-seed source explicitly, validate stage identity and prototype provenance, cross-check B2 against B3 raw Softmax, and compute run/fold outputs with two-sided Wilcoxon and percentile bootstraps.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Task 1 unittest command. Expected: all analysis-contract tests pass.

### Task 2: Test and implement deterministic cases

**Files:**
- Modify: `tests/test_generate_cumulative_framework_analysis.py`
- Modify: `tools/generate_cumulative_framework_analysis.py`

**Interfaces:**
- Consumes: validated canonical `(fold, seed)=(0, 42)` test predictions and train-only representative CSVs.
- Produces: seven ordered case rows with requested prediction, distance, margin, consistency, and provenance fields.

- [ ] **Step 1: Add failing case-selection tests**

```python
def test_case_selection_is_canonical_complete_and_unique(self):
    cases = cumulative.select_case_studies(ROOT)
    self.assertEqual(cases["case_type"].tolist(), [
        "correct_cough", "correct_calm_grunt", "correct_feeding",
        "correct_stress_vocal", "feeding_stress_boundary_feeding",
        "feeding_stress_boundary_stress", "low_margin_uncertain",
    ])
    self.assertEqual(set(cases["fold"]), {0})
    self.assertEqual(set(cases["seed"]), {42})
    self.assertEqual(cases["source_id"].nunique(), 7)
```

- [ ] **Step 2: Run the focused case tests and verify RED**

Expected: missing `select_case_studies` behavior or missing output fields.

- [ ] **Step 3: Implement the exact rank rules and representative lookup**

Use lower-median deterministic ranks, `source_id` tie breaks, no thresholds, and minimum own-prototype distance among train-only samples for main/subtype representatives.

- [ ] **Step 4: Run the focused case tests and verify GREEN**

Expected: seven unique cases from fold 0 seed 42 with complete requested fields.

### Task 3: Test and implement tables and figures

**Files:**
- Modify: `tests/test_generate_cumulative_framework_analysis.py`
- Modify: `tools/generate_cumulative_framework_analysis.py`
- Create: `paper/tables/cumulative_framework_runs.csv`
- Create: `paper/tables/cumulative_framework_summary.csv`
- Create: `paper/tables/cumulative_framework_paired_stats.csv`
- Create: `paper/tables/cumulative_framework_fold_stats.csv`
- Create: `paper/tables/prototype_prediction_case_studies.csv`
- Create: `paper/figures_journal/figure_cumulative_framework.{svg,pdf,png}`
- Create: `paper/figures_journal/figure_prototype_prediction_cases.{svg,pdf,png}`

**Interfaces:**
- Consumes: normalized runs, summary, paired, fold, and case frames.
- Produces: exact requested CSV and fixed-geometry publication exports.

- [ ] **Step 1: Add failing schema/export tests**

Test that output paths are restricted to the requested paper directories, all CSV schemas contain the required fields, figures are 183 mm wide, SVG text remains editable, and PNGs are 300 dpi.

- [ ] **Step 2: Run export tests and verify RED**

Expected: missing generator/export behavior.

- [ ] **Step 3: Implement CSV writes and Matplotlib builders**

Use a stage trajectory panel, a dual-CI forest plot, B3-B0 fold sensitivity, a vector architecture schematic, and a case evidence matrix. Set `svg.fonttype='none'`, `pdf.fonttype=42`, white background, restrained family colors, lowercase panel labels, and exact P-value text without stars.

- [ ] **Step 4: Generate outputs**

Run: `$env:PYTHONNOUSERSITE='1'; conda run -n pigsound-gpu python tools/generate_cumulative_framework_analysis.py --root .`

Expected: five CSVs and six figure exports are written only to their new requested paths.

- [ ] **Step 5: Run tests and verify GREEN**

Run the full new unittest module. Expected: all tests pass.

### Task 4: Documentation, visual QA, integrity, and delivery

**Files:**
- Create: `docs/CUMULATIVE_FRAMEWORK_ANALYSIS.md`
- Modify: `handoff/CODEX_TO_GPT.md`
- Modify: `handoff/CODEX_TO_GPT.json`
- Modify: `handoff/HISTORY.md`

**Interfaces:**
- Consumes: final tables, figures, test output, source hashes, and git diff.
- Produces: reproducibility instructions, leakage/provenance audit, paper-use statement, commit, and push.

- [ ] **Step 1: Document command, inputs, field definitions, case ranks, and inference limits**

Record that representatives are nearest to their train-only prototypes, not claimed nearest neighbours to test embeddings, and that B2-B1/B3-B2 significance is reported exactly as computed.

- [ ] **Step 2: Run validate-only and full test suites**

Run `--help`, `--validate-only`, the new unittest module, the existing journal-figure tests, and the repository's prototype pipeline tests.

- [ ] **Step 3: Perform visual QA in Python outputs**

Inspect both PNGs at original resolution, verify dimensions/DPI, parse SVG text nodes, and confirm no clipping, overlap, raster embedding, or significance-star encoding.

- [ ] **Step 4: Verify source integrity and leakage boundaries**

Recompute SHA-256 for all 240 frozen files used by the pipeline and require exact equality to the pre-run snapshot. Confirm all 25 evaluation leakage audits report no path/source-ID/MD5 overlap and no test parameter selection.

- [ ] **Step 5: Review and stage only scoped files**

Use explicit path arguments for `git add`; review `git diff --cached --stat`, `git diff --cached --check` excluding generated SVG path whitespace, and the staged file list.

- [ ] **Step 6: Commit, update the handoff commit reference, and push**

Create the implementation commit, update the three handoff files with the actual SHA and commands, create the handoff commit if needed, push `paper/sci-draft-v1`, report final delivery HEAD, and stop.
