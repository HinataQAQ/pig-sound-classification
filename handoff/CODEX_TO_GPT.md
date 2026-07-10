# Codex to GPT Handoff - Final Journal Figure Package

## Status

Completed. The user supplied `APPROVED: true` and restricted this round to the
final paper figure package. No model, training, inference, result regeneration,
result CSV/JSON edit, or manuscript rewrite was performed.

## Branch and Commit

- Branch: `paper/sci-draft-v1`
- Base commit: `3c6bd7817095b9e2165624c1fae934db0fa39975`
- Figure-package implementation commit:
  `e6a7f91579cc2fa128ae9e2dce2b154f55ea0774`
- Handoff commit: the delivery HEAD reported in the final Codex response

## Skills and Review Route

- `nature-figure`: Python/Matplotlib journal-figure contract, evidence-first
  plotting, fixed export geometry, source mapping, and figure QA.
- `academic-research-suite`: read-only integrity verification, claim alignment,
  and methodology-reviewer figure-to-claim audit.
- `karpathy-guidelines`, test-driven development, and verification-before-
  completion were used for surgical implementation and regression checks.
- PaperSpine was not run.

## Changed Files

Implementation and tests:

- `tools/generate_journal_figure_package.py`
- `tests/test_generate_journal_figure_package.py`

Main figure package:

- `paper/figures_journal/figure1_overall_framework.{svg,pdf,png,tiff}`
- `paper/figures_journal/figure2_duration_clean_ablations.{svg,pdf,png,tiff}`
- `paper/figures_journal/figure3_hierarchical_clean_prototypes.{svg,pdf,png,tiff}`
- `paper/figures_journal/figure4_simulated_noise_robustness.{svg,pdf,png,tiff}`
- `paper/figures_journal/figure5_failure_selective_prediction.{svg,pdf,png,tiff}`
- `paper/figures_journal/FIGURE_LEGENDS_CN.md`
- `paper/figures_journal/FIGURE_LEGENDS_EN.md`
- `paper/figures_journal/FIGURE_QA_REPORT.md`

Supplementary package:

- `paper/supplementary/figures/figure_s1_clean_ablations.{svg,pdf,png,tiff}`
- `paper/supplementary/figures/figure_s2_demand_environment_snr.{svg,pdf,png,tiff}`
- `paper/supplementary/figures/figure_s3_confusion_calibration_fold_deltas.{svg,pdf,png,tiff}`

Trace and review outputs:

- `paper/figure_source_data/FIGURE_SOURCE_MAP.csv`
- `paper/figure_source_data/FIGURE_TO_CLAIM_MATRIX.csv`
- `paper/review/FINAL_FIGURE_AUDIT.md`
- `handoff/CODEX_TO_GPT.md`
- `handoff/CODEX_TO_GPT.json`
- `handoff/HISTORY.md`

No manuscript, bibliography, paper-result table, appendix result, report result,
model code, checkpoint, prediction, or audio file was changed.

## Exact Commands

Python was run only after activating the required environment:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONNOUSERSITE="1"
conda activate pigsound-gpu
cd C:\py\pigsound\pig-sound-classification
```

Generation and validation:

```powershell
python tools/generate_journal_figure_package.py --help
python tools/generate_journal_figure_package.py --root . --validate-only
python -m unittest tests.test_generate_journal_figure_package -v
python tools/generate_journal_figure_package.py --root .
```

Scope and source-integrity checks:

```powershell
git diff --name-only -- paper/manuscript paper/references paper/tables paper/appendix paper_results reports src tools/prototype_model_adapter.py
Get-ChildItem paper\appendix,paper\tables,reports\prototype_noise_demand_w05_final -Recurse -File | Get-FileHash -Algorithm SHA256
git diff --cached --check -- . ':(exclude)paper/figures_journal/*.svg' ':(exclude)paper/supplementary/figures/*.svg'
```

The generated Matplotlib SVGs contain normal trailing spaces in multiline path
data, so the whitespace check excluded SVG artwork while checking all code,
Markdown, and CSV deliverables.

Commit and delivery:

```powershell
git add -- tools/generate_journal_figure_package.py tests/test_generate_journal_figure_package.py paper/figures_journal paper/supplementary/figures paper/figure_source_data paper/review/FINAL_FIGURE_AUDIT.md
git commit -m "feat: add final journal figure package"
git add -- handoff/CODEX_TO_GPT.md handoff/CODEX_TO_GPT.json handoff/HISTORY.md
git commit -m "docs: update final figure handoff"
git push origin paper/sci-draft-v1
```

## Deliverables and QA

- Exactly 5 main figures and 3 supplementary figures.
- Each figure has editable-text SVG, PDF, 300 dpi PNG, and 600 dpi LZW TIFF.
- Every figure is fixed at 183 mm width; `bbox_inches='tight'` is not used.
- Primary font is SimSun, with Arial and DejaVu Sans fallbacks recorded.
- SVG uses text nodes and references SimSun; no SVG contains an embedded raster,
  gradient, or filter.
- White background; no shadows, 3D, dashboard cards, decorative icons, rainbow
  palette, or table screenshots.
- Macro-F1/recall axes retain 0-1 limits; only explicit delta panels use a
  centered narrow axis.
- Exact P values are printed without significance stars.
- All eight 300 dpi PNGs were inspected at original resolution. No clipping or
  legend/data overlap remains. Figure 4 right margins are PNG 10 px and TIFF
  21 px.
- Source map and claim matrix contain aligned entries for all 25 panels.
- Unit tests: 10/10 passed.
- Final programmatic package audit passed.

## Locked Metrics Used in Figures

No new metrics were produced. The package visualizes only locked evidence:

- Duration: 1 s `0.9226` (n=25), 2 s `0.9472` (n=25), 3 s `0.9434`
  (n=15).
- Matched 2 s - 1 s delta: `0.024630891`, 95% CI
  `[0.014112838, 0.035642072]`, Wilcoxon P=`0.0001623034477`.
- Hierarchical weights: lambda 0.2/0.5/1.0 means
  `0.950506/0.951050/0.951508`; all incremental clean paired CIs cross zero.
- The 25 clean raw/main/hierarchical outputs in Figure 3c are directly traced
  to one lambda=0.5 fold-seed checkpoint cohort.
- Main prototype - raw under moderate/ALL_NOISY/extreme simulation has
  run/fold P values `0.00167307/0.0625`, `0.0105108/0.3125`, and
  `0.560171/1.0`.
- Raw Softmax retains the best ALL_NOISY AURC/AUGRC; main prototype has only a
  small frozen-threshold selective-risk advantage.

## ARS Figure-to-Claim Audit

- Final verdict: `PASS WITH NOTES`.
- Blocking findings: 0.
- Serious or medium integrity findings: 0.
- Prohibited overclaims: 0.
- Coverage: 25/25 panels.
- The audit explicitly preserves these boundaries:
  - simulated additive noise is not real-farm external validation;
  - 0 dB is an extreme simulated-noise stress condition, not no-noise;
  - hierarchical clean gains are not statistically significant;
  - run-level and fold-cluster evidence are separate;
  - Figure 4 stored Wilcoxon P values are unadjusted;
  - pooled repeated decisions are not independent recordings;
  - prototype inference does not universally improve uncertainty.

## Leakage and Data Audit

- No training, inference, feature extraction, checkpoint loading, prototype
  construction, calibration, threshold fitting, or noise simulation ran.
- No test-set tuning occurred.
- Training-only prototype and validation-only calibration boundaries are shown
  but were not recomputed.
- A pre/post SHA-256 comparison passed for 39 locked result CSV/JSON files.
- The generator's own pre/post snapshot also found no changed required source.
- Train/validation/test roles, folds, seeds, predictions, and manifests are
  unchanged.

## Paper Usability

- `paper_usable`: true for the final figure package.
- `figure_package_complete`: true.
- `submission_complete`: conditional on existing manuscript/release metadata,
  not on this figure task.
- The figures are suitable for the current Chinese Word working draft and later
  journal layout, subject to target-journal production preflight.

## Remaining Notes and Scientific-Judgment Questions

- SimSun is referenced but not embedded in SVG. A system without SimSun may
  substitute/reflow text; local PDF font inspection tooling is unavailable.
- The fold-cluster sensitivity has only five folds and low power.
- No environment-specific fold-cluster CI or selective-metric paired CI/P exists;
  none was invented.
- DEMAND controlled additive noise cannot replace real-farm external validation.
- GPT Pro/editorial judgment is still needed on whether the final manuscript
  should retain the stored unadjusted nine-comparison P display or apply and
  report a prespecified multiplicity correction in a separate approved round.
- The target journal should determine whether 600 dpi TIFF is required and what
  final font-embedding preflight is needed.

## Stop Point

The approved final-figure stage is complete. Do not modify the manuscript or
start another paper/experiment stage without a new explicit `APPROVED: true`.
