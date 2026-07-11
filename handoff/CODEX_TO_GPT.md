# Codex to GPT Handoff — Duration-Aware Hierarchical Prototype Inference Framework

## Status

Completed the approved cumulative-analysis stage from the existing 25 matched
clean-audio runs. No model training, model inference, feature extraction,
prototype reconstruction, calibration, threshold fitting, lambda change,
manifest edit, checkpoint edit, audio edit, noise-result edit, or established
CSV/JSON result edit was performed.

## Branch and commits

- Current branch: `paper/prior-art-fair-benchmark`.
- Round base: `d96192d73ae32adadebd3c5bd910b31848283fd3`.
- Implementation/output commit:
  `f265f6b7bf009d46a318d2683bff5b85c879c5e1`.
- Handoff commit: reported as delivery HEAD in the final Codex response.
- The current branch is a deliberate descendant of `paper/sci-draft-v1` and
  retains the preceding approved prior-art/fair-benchmark audit.

## Exact commands

Run from `C:\py\pigsound\pig-sound-classification` with
`PYTHONNOUSERSITE=1`:

```powershell
& "C:\py\anaconda3\envs\pigsound-gpu\python.exe" tools/generate_cumulative_framework_analysis.py --help
& "C:\py\anaconda3\envs\pigsound-gpu\python.exe" tools/generate_cumulative_framework_analysis.py --root . --validate-only
& "C:\py\anaconda3\envs\pigsound-gpu\python.exe" tools/generate_cumulative_framework_analysis.py --root .
& "C:\py\anaconda3\envs\pigsound-gpu\python.exe" -m unittest discover -s tests -v
git diff --cached --check -- tools/generate_cumulative_framework_analysis.py paper_results/scripts/generate_cumulative_framework_analysis.py tests/test_generate_cumulative_framework_analysis.py docs/CUMULATIVE_FRAMEWORK_ANALYSIS.md docs/superpowers/specs/2026-07-11-duration-aware-hierarchical-prototype-inference-design.md docs/superpowers/plans/2026-07-11-duration-aware-hierarchical-prototype-inference.md paper/tables paper_results/tables
```

`conda run -n pigsound-gpu` also succeeded for generation and focused tests.
One final `conda run` validation attempt collided with another process using a
Conda temporary activation file, so the final verification used the exact
environment Python executable above. No dependency was installed.

## Changed files

Implementation/output commit:

- `tools/generate_cumulative_framework_analysis.py`
- `paper_results/scripts/generate_cumulative_framework_analysis.py`
- `tests/test_generate_cumulative_framework_analysis.py`
- `docs/CUMULATIVE_FRAMEWORK_ANALYSIS.md`
- `docs/superpowers/specs/2026-07-11-duration-aware-hierarchical-prototype-inference-design.md`
- `docs/superpowers/plans/2026-07-11-duration-aware-hierarchical-prototype-inference.md`
- `paper/tables/cumulative_framework_runs.csv`
- `paper/tables/cumulative_framework_summary.csv`
- `paper/tables/cumulative_framework_paired_stats.csv`
- `paper/tables/cumulative_framework_fold_stats.csv`
- `paper/tables/prototype_prediction_case_studies.csv`
- the five byte-identical table mirrors under `paper_results/tables/`
- `paper/figures_journal/figure_cumulative_framework.{svg,pdf,png}`
- `paper/figures_journal/figure_prototype_prediction_cases.{svg,pdf,png}`

Handoff-only commit:

- `handoff/CODEX_TO_GPT.md`
- `handoff/CODEX_TO_GPT.json`
- `handoff/HISTORY.md`

No unrelated benchmark/reference work, local experiment output, or untracked
workspace file was included.

## Locked stage summary

All values are Macro-F1 over the exact folds `0-4` and seeds
`42,123,777,2024,3407`.

| Stage | Definition | n | Mean | SD | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| B0 | 1 s Log-Mel CRNN | 25 | 0.922606467 | 0.014540638 | 0.891876 | 0.946238 |
| B1 | 2 s Log-Mel CRNN | 25 | 0.947237358 | 0.016763541 | 0.921269 | 0.982140 |
| B2 | B1 + hierarchical auxiliary supervision, lambda=0.5, Raw Softmax | 25 | 0.951049673 | 0.012393314 | 0.921720 | 0.976177 |
| B3 | B2 backbone + hierarchical acoustic prototype inference | 25 | 0.953986214 | 0.012003773 | 0.934291 | 0.976177 |

## Direct paired statistics

Every contrast was calculated directly from matched fold-seed rows. Bootstrap
intervals use 10,000 percentile resamples and seed 3407. Wilcoxon tests are
two-sided and unadjusted.

| Contrast | Baseline → final mean | Mean delta ± SD | Run bootstrap 95% CI | Wilcoxon P | W/T/L | Fold deltas (0–4) | Fold-cluster 95% CI |
|---|---|---|---|---|---|---|---|
| B1−B0 | 0.922606467 → 0.947237358 | 0.024630891 ± 0.028534613 | [0.014112838, 0.035642072] | 0.000162303448 | 21/0/4 | −0.000170332, 0.029243138, 0.009924437, 0.013381116, 0.070776095 | [0.006577865, 0.048280218] |
| B2−B1 | 0.947237358 → 0.951049673 | 0.003812315 ± 0.012940365 | [−0.001320423, 0.008726684] | 0.170241396 | 16/1/8 | 0.012167035, −0.007299930, 0.009552194, 0.009366052, −0.004723776 | [−0.003406537, 0.010560902] |
| B3−B2 | 0.951049673 → 0.953986214 | 0.002936541 ± 0.010335613 | [−0.000477565, 0.007294065] | 0.058252952 | 13/8/4 | −0.001168631, 0.013334835, 0.001204159, 0.000107454, 0.001204888 | [−0.000219515, 0.008263370] |
| B3−B1 | 0.947237358 → 0.953986214 | 0.006748856 ± 0.011611119 | [0.002388569, 0.011270120] | 0.013808562 | 18/0/7 | 0.010998404, 0.006034905, 0.010756353, 0.009473505, −0.003518888 | [0.001295329, 0.010596604] |
| B3−B0 | 0.922606467 → 0.953986214 | 0.031379747 ± 0.023102131 | [0.022821112, 0.040301488] | 2.98023224e−7 | 24/0/1 | 0.010828072, 0.035278044, 0.020680790, 0.022854622, 0.067257207 | [0.017174469, 0.049575547] |

The cumulative B3−B0 gain is `+0.031379747` Macro-F1 and was tested directly,
not inferred from intermediate tests. All five B3−B0 fold means are positive.
B2−B1 and B3−B2 are not statistically significant. Do not claim that every
incremental module is significant.

## Prototype case studies

Seven deterministic cases were selected only from fold 0 seed 42:

- four all-route-correct class cases use the lower median Raw Softmax
  confidence within each class;
- the true-feeding and true-stress boundary errors use the lower median raw
  Top-2 margin in their predefined class-specific pools;
- the uncertain case is the remaining row with minimum final hierarchical
  prototype Top-2 margin;
- ties use `source_id`; no confidence threshold was fitted.

The table includes true class, all requested Top-2 routes, all main/subtype
distances, hierarchical margin, final hierarchy consistency, path/source-ID/MD5,
and train-only representative traces. All seven final hierarchy-consistency
flags are true; that does not mean all seven classifications are correct (the
two boundary cases remain errors).

The closest representative is the same-run training sample closest to the
predicted class prototype. Query-to-training embeddings were not saved, so it
is not claimed to be the nearest training neighbour to the query.

## Leakage, provenance, and immutable-input audit

- Exactly 25 unique fold-seed rows exist for every stage; no imputation or
  fold/seed mixing was used.
- All 25 B2 values exactly equal the matched prototype cohort Raw Softmax
  Macro-F1 values.
- All 25 prototype metadata files bind the complete model/STFT/label identity
  to the expected B2 summary path and actual summary SHA-256.
- All 15 actual train/validation/test manifest SHA-256 values match metadata.
- All 25 evaluation leakage audits pass exact-path, source-ID, and MD5
  disjointness; test parameter selection is false.
- For the canonical cases, frozen predictions exactly equal the test-manifest
  identity triples, both representative CSVs exactly equal train-manifest
  identity triples, and all three splits are independently disjoint on path,
  source ID, and MD5.
- Prototype source is train, calibration source is validation, and evaluation
  role is frozen test.
- An external pre/post SHA-256 check confirmed all 240 frozen inputs unchanged.
- Checkpoint bytes were deliberately not read. The expected checkpoint path and
  declared digest are internally consistent; linked B2 summaries and manifests
  were verified against actual bytes.
- Fold 0 seed 3407 retains `single_fold_debug=true` in both metadata and metrics
  while remaining explicitly `eligible_for_cv_aggregation=true`; the complete
  predefined grid is retained and the marker is disclosed in the run table.

No required source file is missing.

## Raw Softmax reproduction nuance

Case-table Raw Softmax probabilities come from the same B3 evaluation
prediction artifact as the prototype probabilities/distances, preserving exact
sample alignment. They are same-checkpoint B2 forward reproductions rather than
copied values from the original B2 prediction CSV. Class predictions and
Macro-F1 match exactly in all 25 audits, but the advisory probability drift is
up to `9.449124e-4` across the cohort (`2.601743e-4` for fold 0 seed 42). This
can swap a near-median case rank and is disclosed in the documentation.

## Verification

- Generator `--help`: passed.
- Generator full run: passed; 240 sources unchanged.
- Generator `--validate-only`: passed; 240 sources unchanged.
- Full repository test suite: 118/118 passed.
- Task-specific tests: 22/22 passed.
- Required table row counts: 25, 4, 5, 25, and 7.
- Paper/paper-results table and script mirrors: 6/6 byte-identical.
- SVGs: editable text and no embedded raster images.
- PDFs: embedded TrueType fonts.
- PNGs: 300 dpi, 183 mm width.
- Both PNGs visually inspected; no clipping or material overlap remains.
- Non-generated code/docs/tables passed `git diff --check`. Matplotlib-generated
  SVG path lines contain generator-standard trailing spaces.
- Independent code review: PASS, no Critical/Important/Minor issues.
- Independent scientific recomputation: PASS; maximum numeric discrepancy
  `1.11e-16`; fallacy scan 11/11.

## Paper usability and scientific cautions

- `paper_usable=true` for the cumulative tables, figures, and deterministic
  case-study table.
- The five Wilcoxon P values are unadjusted and 25 runs are clustered within
  only five folds. B3−B1 does not survive illustrative Bonferroni correction;
  the five-fold Wilcoxon sensitivity for B3−B0 is `P=0.0625`.
- Treat the fold-cluster bootstrap as a five-cluster sensitivity analysis.
- Do not use causal language, do not claim every module is significant, and do
  not call Macro-F1 accuracy.
- The work does not establish real-farm external generalization.

## Blockers and questions for GPT Pro

There are no missing-source or implementation blockers for this delivery.
Scientific/editorial judgment remains necessary for:

1. whether to report all five unadjusted Wilcoxon tests in the main text or
   move incremental contrasts to supplementary material with a multiplicity
   caveat;
2. whether the eligible fold 0 seed 3407 debug marker needs main-text,
   supplementary, or provenance-only disclosure;
3. whether a future separately approved inference stage should save
   query-to-training embeddings for true nearest-neighbour exemplars;
4. whether future case ranks should retain the single aligned B3 reproduction
   artifact or be redesigned around the original B2 probability file.

## Stop point

Review and integrate the delivered cumulative evidence only after scientific
judgment. Do not start manuscript revision, new inference, new training, a new
benchmark, or the proposed exemplar extension without another explicit
`APPROVED: true` stage.
