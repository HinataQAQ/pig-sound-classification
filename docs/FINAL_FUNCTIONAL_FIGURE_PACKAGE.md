# Final Functional Figure Package

This package implements the accepted `framework_functional_only` interpretation.
B2 validation-selected Raw Softmax remains the primary four-class classifier;
prototype outputs are a parallel candidate-prediction and diagnostic layer, and
B3 is retained as an ablation. The generator reads existing frozen tables and
prediction files only. It does not load checkpoints, run inference, train a
model, select a hyperparameter, or modify established result CSV/JSON files.

## Commands

From the repository root in PowerShell:

```powershell
$env:PYTHONNOUSERSITE="1"
conda run -n pigsound-gpu python tools/generate_final_functional_figure_package.py --help
conda run -n pigsound-gpu python tools/generate_final_functional_figure_package.py --root . --n-boot 10000 --bootstrap-seed 3407 --validate-only
conda run -n pigsound-gpu python tools/generate_final_functional_figure_package.py --root . --n-boot 10000 --bootstrap-seed 3407
```

The write command refuses an occupied output target. It stages every file,
checks source hashes again, validates all four export formats, and then promotes
the package atomically. The approved
`paper/final_validation/lambda_selection_by_fold.csv` is the sole authority for
the selected frozen cohort; the 75 validation summaries are used only to verify
that they reproduce the locked choices, with a mismatch treated as an error.

## Derived tables

- `paper/final_validation/deployable_predicted_margin_summary.csv` contains
  `run`, `fold_mean`, and `overall_run_mean` records for Main Prototype and
  Hierarchical Prototype. The score is the shared four-main-prototype distance
  gap `second_nearest - nearest`; `margin_uses_true_label=false`. Correct/error
  partitions and error AUROC are outcome evaluations, so
  `evaluation_uses_true_label=true`. Metric-specific availability, run SD,
  fold SD, and five-fold cluster-bootstrap limits are explicit columns.
- `paper/final_validation/hierarchy_inconsistency_utility.csv` contains the
  same three aggregation levels for Raw Softmax, Main Prototype, and
  Hierarchical Prototype. It reports prevalence, conditional error rates,
  enrichment, error precision, error recall, and availability counts. The flag
  compares the main prediction with the mapped subtype prediction and does not
  use the true label; its performance evaluation does.
- `paper/final_validation/prototype_functional_final_summary.csv` is a compact
  long-form summary of candidate coverage, prototype geometry, deployable
  margin evaluation, hierarchy-inconsistency utility, and disagreement totals.
  `deployability` and `interpretation` define the permitted use of every row.

All aggregate evidence uses the exact 5-fold x 5-seed cohort. Primary intervals
average seeds within each fold and bootstrap the five fold means with 10,000
deterministic resamples (seed 3407).

## Figure outputs and provenance

`paper/figures_final/` contains Figures 1-4 and Supplementary Figures S1-S7 in
editable-text SVG, PDF, 300-dpi PNG, and 600-dpi TIFF. The English and Chinese
legend files define sample size, centre/spread, interval, test, Holm status,
deployability, and limitations for every figure. `FIGURE_SOURCE_MAP.csv` and
`FIGURE_TO_CLAIM_MATRIX.csv` provide one row per panel with relative evidence
paths and SHA-256 digests. `FIGURE_QA_REPORT.md` records machine and visual QA.

## Interpretation boundary

The predicted distance margin is available at use time, but correct/error
separation and AUROC require labels for evaluation. Hierarchy inconsistency is
a rare, error-enriched diagnostic flag with low error recall; it is not a safe
automatic rejector. True-class margins in Figure S7 are label-aware and
unavailable at deployment. DEMAND noise results are simulated and
supplementary. The package establishes controlled closed-set evidence with
path/source-ID/MD5 disjointness, not pig-, session-, device-, farm-, welfare-,
or real-farm external validity.
