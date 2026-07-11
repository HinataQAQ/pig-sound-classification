# Codex -> GPT Pro Handoff

## Status

`COMPLETED_FINAL_FUNCTIONAL_FIGURE_PACKAGE`

The approved final figure stage is complete under the acceptance interpretation
`framework_functional_only`. Stop after reviewing this handoff. Do not begin
manuscript regeneration, training, inference, Atlas, few-shot/open-set work,
noise experiments, or a new parameter-selection stage without a new explicit
`APPROVED: true` instruction.

## Repository state

- Source branch: `paper/final-validation-audit`
- Source/base commit: `9953cd73f0b0c86eab9ba4386295d75f5fd87cdb`
- Work branch: `paper/final-figures-functional-framework`
- Implementation/results commit:
  `ddeb89708e9c52b301b254a6382cc80890eb7ad4`
- Handoff commit: reported in the final Codex response
- Acceptance: `framework_functional_only`
- Paper usable: `true`, subject to the limitations below

## Authorization and data boundaries

- Model training: **not run**
- Checkpoint/model inference: **not run**
- Feature extraction or prototype rebuilding: **not run**
- Atlas, few-shot/open-set, or new noise experiment: **not run**
- New hyperparameter selection: **not run**
- Test-set tuning or threshold selection: **not run**
- Audio, checkpoint, manifest, or established result modification: **none**
- B2 validation-selected Raw Softmax remains the primary classifier.
- Prototypes remain a parallel candidate-prediction/diagnostic layer.
- B3 remains an ablation and is not presented as superior Top-1.
- `paper/final_validation/lambda_selection_by_fold.csv` is authoritative for
  frozen-cohort choice. The 75 validation summaries are reproduction checks
  only; a mismatch hard-fails before analysis.

## Commands actually executed

Environment and generator interface:

```powershell
$env:PYTHONNOUSERSITE="1"
conda run -n pigsound-gpu python tools/generate_final_functional_figure_package.py --help
conda run -n pigsound-gpu python tools/generate_final_functional_figure_package.py --root . --n-boot 10000 --bootstrap-seed 3407 --validate-only
```

Final generation from frozen artifacts only:

```powershell
$env:PYTHONNOUSERSITE="1"
conda run -n pigsound-gpu python tools/generate_final_functional_figure_package.py --root . --n-boot 10000 --bootstrap-seed 3407
```

Final regression and syntax verification:

```powershell
$env:PYTHONNOUSERSITE="1"
conda run -n pigsound-gpu python -m unittest tests.test_generate_final_validation_audit tests.test_generate_journal_figure_package tests.test_generate_cumulative_framework_analysis tests.test_generate_final_functional_figure_package
conda run -n pigsound-gpu python -m py_compile tools/generate_final_functional_figure_package.py tests/test_generate_final_functional_figure_package.py
git diff --cached --check
```

Integrity checks also re-imported the pre-task SHA baseline at
`C:\Users\s1205\AppData\Local\Temp\pig-final-functional-csv-json-before.xml`,
re-hashed all 3,956 pre-existing CSV/JSON files, re-hashed every semicolon-split
path in both 30-panel provenance maps, checked all 44 exports for non-empty
content, and scanned all SVG lines for trailing whitespace.

## New implementation and documentation

- `tools/generate_final_functional_figure_package.py`
- `tests/test_generate_final_functional_figure_package.py`
- `docs/FINAL_FUNCTIONAL_FIGURE_PACKAGE.md`
- `docs/superpowers/plans/2026-07-11-final-functional-figure-package.md`
- `paper/review/FINAL_FUNCTIONAL_FIGURE_AUDIT.md`

The generator enforces the exact 5-fold x 5-seed cohort, label-free use-time
margin/flag definitions, fold-mean aggregation, deterministic five-fold cluster
bootstrap, locked selection reproduction, pre-analysis SHA locking, staged
four-format QA, failure-atomic promotion/rollback, and refusal to overwrite an
occupied output target.

## New derived tables

- `paper/final_validation/deployable_predicted_margin_summary.csv`
- `paper/final_validation/hierarchy_inconsistency_utility.csv`
- `paper/final_validation/prototype_functional_final_summary.csv`

No pre-existing validation/result CSV or JSON changed.

### Deployable shared main-prototype distance margin

The score is `second_nearest - nearest` across the four main-prototype cosine
distances and does not use the true label. Correct/error partitions and AUROC
are label-aware evaluation. Values below are means of five fold means; intervals
are deterministic 10,000-resample fold-cluster 95% CIs.

| Route | Correct median margin | Error median margin | Error AUROC from negative margin |
|---|---:|---:|---:|
| Main Prototype | 1.016730 `[0.985230, 1.045626]` | 0.096617 `[0.072188, 0.121045]` | 0.946018 `[0.932777, 0.956515]` |
| Hierarchical Prototype | 1.017693 `[0.984639, 1.049550]` | 0.091626 `[0.070375, 0.113967]` | 0.946918 `[0.936142, 0.957694]` |

The Main and Hierarchical rows share one geometric margin; only their
method-specific correct/error partitions differ. No mapped-subtype or
hierarchical cosine-distance vector exists.

### Hierarchy inconsistency utility

| Route | Prevalence | Error precision | Error recall | Error enrichment | Available enrichment runs |
|---|---:|---:|---:|---:|---:|
| Raw Softmax | 0.016429 `[0.011429, 0.020952]` | 0.298667 `[0.167333, 0.424000]` | 0.116404 `[0.055455, 0.179333]` | 9.299352 `[3.588273, 17.775635]` | 24/25 |
| Main Prototype | 0.011429 `[0.007857, 0.014286]` | 0.476667 `[0.303333, 0.646667]` | 0.137645 `[0.064641, 0.207641]` | 15.600734 `[8.198683, 24.139583]` | 22/25 |
| Hierarchical Prototype | 0.008333 `[0.004524, 0.012143]` | 0.492000 `[0.304000, 0.686667]` | 0.116013 `[0.052154, 0.179492]` | 17.911560 `[11.097512, 25.873583]` | 18/25 |

Conditional inconsistent/consistent error rates are respectively
`0.322833/0.042855` for Raw, `0.535000/0.042135` for Main, and
`0.688333/0.044371` for Hierarchical when centred as fold means. The low
prevalence and recall prohibit interpretation as a safe automatic rejector.

### Frozen disagreement totals

- Main Prototype: 29 Raw-correct/prototype-wrong, 28 prototype-correct/Raw-wrong,
  0 both wrong; net repeated harm `+1`.
- Hierarchical Prototype: 37 harmed, 29 rescued, 0 both wrong; net repeated harm
  `+8`. It harmed more repeated predictions than it rescued.

## Primary classification interpretation retained

- B1-B0: mean delta `+0.024631`, fold-cluster 95% CI
  `[0.006578, 0.048280]`, Wilcoxon `P=0.000162303448`; Holm not applicable to
  this prespecified duration contrast.
- B2-B1: mean delta `+0.004944`, fold-cluster 95% CI
  `[-0.000481, 0.009461]`, raw `P=0.122846338`, locked-family Holm
  `P=0.614231688`; exploratory and nonsignificant.
- B3-B2: mean delta `-0.001916`, fold-cluster 95% CI
  `[-0.004631, 0.000799]`, raw `P=0.477536357`, locked-family Holm
  `P=0.955072713`; no Top-1 improvement.

## Figure package

Four main figures, each in editable-text SVG, PDF, 300-dpi PNG, and 600-dpi
TIFF:

- `paper/figures_final/figure1_final_framework.*`
- `paper/figures_final/figure2_primary_classification_evidence.*`
- `paper/figures_final/figure3_prototype_functional_value.*`
- `paper/figures_final/figure4_prototype_candidate_cases.*`

Seven supplementary figures in the same four formats:

- `paper/figures_final/figure_s1_complete_clean_ablations.*`
- `paper/figures_final/figure_s2_fixed_lambda_retrospective_cumulative.*`
- `paper/figures_final/figure_s3_lambda_selection_by_fold.*`
- `paper/figures_final/figure_s4_best_epoch_validation_test_gap.*`
- `paper/figures_final/figure_s5_demand_simulated_noise.*`
- `paper/figures_final/figure_s6_aurc_augrc.*`
- `paper/figures_final/figure_s7_label_aware_true_class_margin.*`

Legends, maps, and QA:

- `paper/figures_final/FIGURE_LEGENDS_EN.md`
- `paper/figures_final/FIGURE_LEGENDS_CN.md`
- `paper/figures_final/FIGURE_SOURCE_MAP.csv`
- `paper/figures_final/FIGURE_TO_CLAIM_MATRIX.csv`
- `paper/figures_final/FIGURE_QA_REPORT.md`

## Verification evidence

- Final focused/regression suite: **82/82 passed**.
- Focused final-functional suite: **27/27 passed**.
- Generator `--validate-only`: 11 figures, 30 panels, 921 frozen sources,
  pre-analysis SHA lock passed, zero writes.
- Final generation: 52 outputs; failure-atomic promotion and unchanged source
  hashes reported true.
- 44/44 figure exports non-empty; SVG text editable; PNG 300 dpi; TIFF 600 dpi;
  fixed 183-mm width and white-background checks passed.
- All 11 PNGs passed original-resolution visual QA; Figure 4's final shortened
  hierarchical-route title is unclipped.
- Source map: 30 unique panel rows and 151 verified path/SHA-256 links. Claim
  matrix: 30 aligned rows with matching evidence paths/hashes.
- Full source audit: 921 frozen artifacts unchanged before/after analysis,
  export, and promotion.
- Pre-existing result audit: 3,956 baseline CSV/JSON files, 0 missing, 0 changed.
- `py_compile`, staged `git diff --check`, and SVG trailing-whitespace scan pass.
- Independent code review: PASS after locked-selection hardening; no remaining
  P0/P1/P2 defect.
- Five-role ARS review: ACCEPT. Superseding EIC and methodology cards score
  D1-D5 all pass; no mandatory block/warn majority or D4 block fired.

## Leakage audit and research integrity

- Exact path, source-ID, and MD5 train/validation/test disjointness remains
  valid in the frozen selected runs.
- Prototype construction sources are train-only; validation remains the only
  calibration/selection role; frozen test labels are used only for evaluation.
- The deployable margin and hierarchy-inconsistency flag do not use test labels
  at use time. Correctness, AUROC, precision, recall, and true-class margins are
  explicitly label-aware evaluation.
- Pig-, session-, device-, barn-, and farm-level grouping is not established.
- No result is described as statistically significant when the paired evidence
  is nonsignificant.

## Paper usability, limitations, and blockers

`paper_usable=true` for the `framework_functional_only` narrative.

Required adjacent limitations:

- only five fold clusters underlie fold-cluster intervals;
- S2 is retrospective run-bootstrap inference with five unadjusted Wilcoxon
  values and cannot replace the validation-selected chain;
- S6 lacks stored uncertainty estimates;
- DEMAND evidence is simulated and supplementary;
- no welfare diagnosis, safe rejector, universal threshold, operational cost
  mapping, or real-farm external validity is established;
- the QA records the full 921-source hash audit result/count, while persisted
  maps contain directly cited panel evidence rather than a full digest ledger.

Blockers: none for this approved stage.

## Questions requiring GPT Pro scientific judgment

1. Whether the later manuscript should lead with the compact Figure 1
   architecture or begin directly with Figure 2's primary evidence.
2. Whether a future separately approved release should persist a complete
   921-artifact digest ledger in addition to the panel maps.
3. Whether the next scientific stage should be manuscript-only integration or
   a separately designed real-farm external-validation protocol. Neither is
   authorized by this handoff.

## Stop point

Review the committed package and this handoff, then stop. A recommended next
step is a separate, explicitly approved manuscript-integration stage that keeps
B2 Raw Softmax primary and carries every limitation above. Do not execute that
stage automatically.
