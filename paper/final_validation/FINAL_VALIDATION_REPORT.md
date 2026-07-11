# Final Validation Report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-11T09:55:41.954770+00:00
- Verification Status: VERIFIED
- Version Label: final_validation_v1
- Source: frozen clean 5-fold × 5-seed experiment artifacts
- Overall Confidence: CAUTION

## Scope and locked interpretation

This is a locked validation-only sensitivity analysis. It is neither fully prospective nor fully confirmatory. Candidate λ values were fixed at 0.2, 0.5, and 1.0; selection used only the five-seed validation mean, then validation SD within a 1e-6 mean tie, then the smallest λ. Outer-fold test metrics did not participate in selection. No backbone was retrained and no test-derived margin threshold was selected.

## Fold-wise validation selection

| Fold | Selected λ | λ=0.2 val mean | λ=0.5 val mean | λ=1.0 val mean | Reason |
|---|---|---|---|---|---|
| 0 | 1.0 | 0.985000 | 0.985000 | 0.986666 | highest_validation_mean |
| 1 | 0.2 | 0.973319 | 0.968321 | 0.969967 | highest_validation_mean |
| 2 | 0.2 | 0.981660 | 0.978332 | 0.979984 | highest_validation_mean |
| 3 | 0.2 | 0.964987 | 0.963307 | 0.961619 | highest_validation_mean |
| 4 | 0.5 | 0.964964 | 0.968321 | 0.966596 | highest_validation_mean |

Every one of the 75 source validation summaries is recorded with its SHA256 in `lambda_selection_provenance.json`.

## Validation-selected framework results

| Stage | n | Mean Macro-F1 | SD | Min | Max |
|---|---|---|---|---|---|
| B0 | 25 | 0.922606 | 0.014541 | 0.891876 | 0.946238 |
| B1 | 25 | 0.947237 | 0.016764 | 0.921269 | 0.982140 |
| B2_valsel | 25 | 0.952182 | 0.013262 | 0.927918 | 0.976177 |
| B3_valsel | 25 | 0.950266 | 0.016528 | 0.921269 | 0.976177 |
| B2_fixed_w05 | 25 | 0.951050 | 0.012393 | 0.921720 | 0.976177 |
| B3_fixed_w05 | 25 | 0.953986 | 0.012004 | 0.934291 | 0.976177 |

| Comparison | n | Mean Δ | Run bootstrap 95% CI | Wilcoxon raw P | Holm P | W/T/L | Fold-cluster 95% CI |
|---|---|---|---|---|---|---|---|
| B2_valsel - B1 | 25 | 0.004944 | [0.000131, 0.009698] | 0.122846338 | 0.614231688 | 17/1/7 | [-0.000481, 0.009461] |
| B2_valsel - B0 | 25 | 0.029575 | [0.020723, 0.038718] | 0.000001967 | 0.000013769 | 22/0/3 | [0.013832, 0.048162] |
| B2_fixed_w05 - B2_valsel | 25 | -0.001132 | [-0.004735, 0.002394] | 0.629162352 | 0.955072713 | 8/6/11 | [-0.008327, 0.004436] |
| B3_valsel - B2_valsel | 25 | -0.001916 | [-0.005378, 0.001497] | 0.477536357 | 0.955072713 | 6/8/11 | [-0.004631, 0.000799] |
| B3_valsel - B1 | 25 | 0.003028 | [-0.001626, 0.007589] | 0.290397166 | 0.871191498 | 13/1/11 | [-0.001552, 0.007609] |
| B3_valsel - B0 | 25 | 0.027659 | [0.017602, 0.037928] | 0.000026643 | 0.000159860 | 20/0/5 | [0.009256, 0.050154] |
| B3_fixed_w05 - B3_valsel | 25 | 0.003720 | [-0.001662, 0.008862] | 0.127398044 | 0.614231688 | 12/7/6 | [-0.000912, 0.008940] |

Holm adjustment is applied once across the seven requested validation-selected/fixed-sensitivity comparisons. Seeds are not treated as independent test cohorts: five fold-mean deltas and a five-fold cluster bootstrap are reported separately.

## Prototype functional value

| Method | Metric | Available runs | Mean | SD | Fold-cluster 95% CI |
|---|---|---|---|---|---|
| raw_softmax | main_top1_accuracy | 25 | 0.952381 | 0.013086 | [0.942857, 0.961667] |
| raw_softmax | main_top2_accuracy | 25 | 1.000000 | 0.000000 | [1.000000, 1.000000] |
| raw_softmax | subtype_top1_accuracy | 25 | 0.843571 | 0.036269 | [0.825000, 0.863095] |
| raw_softmax | subtype_top2_accuracy | 25 | 0.988333 | 0.006539 | [0.984286, 0.992857] |
| raw_softmax | hierarchy_consistency_rate | 25 | 0.983571 | 0.011563 | [0.979048, 0.988571] |
| raw_softmax | classification_error_rate_hierarchy_consistent | 25 | 0.042855 | 0.012861 | [0.031980, 0.053077] |
| raw_softmax | classification_error_rate_hierarchy_inconsistent | 24 | 0.311111 | 0.327841 | [0.167333, 0.472333] |
| raw_softmax | feeding_stress_exact_pair_top2_coverage | 25 | 0.987619 | 0.018340 | [0.980000, 0.995238] |
| main_prototype | main_top1_accuracy | 25 | 0.952143 | 0.013582 | [0.942619, 0.961190] |
| main_prototype | main_top2_accuracy | 25 | 1.000000 | 0.000000 | [1.000000, 1.000000] |
| main_prototype | subtype_top1_accuracy | 25 | 0.858095 | 0.028839 | [0.847143, 0.874286] |
| main_prototype | subtype_top2_accuracy | 25 | 0.986905 | 0.008241 | [0.981190, 0.992143] |
| main_prototype | hierarchy_consistency_rate | 25 | 0.988571 | 0.007669 | [0.985714, 0.992143] |
| main_prototype | classification_error_rate_hierarchy_consistent | 25 | 0.042135 | 0.014171 | [0.030928, 0.051904] |
| main_prototype | classification_error_rate_hierarchy_inconsistent | 22 | 0.541667 | 0.392346 | [0.356667, 0.701667] |
| main_prototype | feeding_stress_exact_pair_top2_coverage | 25 | 0.997143 | 0.007897 | [0.993810, 0.999524] |
| hierarchical_prototype | main_top1_accuracy | 25 | 0.950476 | 0.016276 | [0.938333, 0.961667] |
| hierarchical_prototype | main_top2_accuracy | 25 | 1.000000 | 0.000000 | [1.000000, 1.000000] |
| hierarchical_prototype | subtype_top1_accuracy | 25 | 0.858095 | 0.028839 | [0.847143, 0.874286] |
| hierarchical_prototype | subtype_top2_accuracy | 25 | 0.986905 | 0.008241 | [0.981190, 0.992143] |
| hierarchical_prototype | hierarchy_consistency_rate | 25 | 0.991667 | 0.007490 | [0.987857, 0.995476] |
| hierarchical_prototype | classification_error_rate_hierarchy_consistent | 25 | 0.044371 | 0.016618 | [0.032739, 0.057602] |
| hierarchical_prototype | classification_error_rate_hierarchy_inconsistent | 18 | 0.683333 | 0.369287 | [0.463333, 0.905000] |
| hierarchical_prototype | feeding_stress_exact_pair_top2_coverage | 25 | 0.996667 | 0.008074 | [0.992381, 0.999524] |

### Prototype ranks and margins

| Geometry metric | Available runs | Mean | SD | Fold-cluster 95% CI |
|---|---|---|---|---|
| true_main_prototype_mean_rank | 25 | 1.047857 | 0.013582 | [1.038810, 1.057381] |
| true_main_prototype_rank1_rate | 25 | 0.952143 | 0.013582 | [0.942619, 0.961190] |
| true_main_prototype_rank2_rate | 25 | 1.000000 | 0.000000 | [1.000000, 1.000000] |
| true_main_prototype_mrr | 25 | 0.976071 | 0.006791 | [0.971310, 0.980595] |
| true_subtype_prototype_mean_rank | 25 | 1.155000 | 0.031846 | [1.139286, 1.167857] |
| true_subtype_prototype_rank1_rate | 25 | 0.858095 | 0.028839 | [0.847143, 0.874286] |
| true_subtype_prototype_rank2_rate | 25 | 0.986905 | 0.008241 | [0.981190, 0.992143] |
| true_subtype_prototype_mrr | 25 | 0.926865 | 0.014811 | [0.921071, 0.934802] |

| Method | Level | Median margin if correct | Median margin if error | Error AUROC | AUROC fold-cluster 95% CI |
|---|---|---|---|---|---|
| raw_softmax | main | 1.015420 | -0.085278 | 0.995612 | [0.992041, 0.998859] |
| raw_softmax | subtype | 0.283925 | -0.016141 | 0.968437 | [0.950534, 0.978833] |
| main_prototype | main | 1.016730 | -0.096617 | 1.000000 | [1.000000, 1.000000] |
| main_prototype | subtype | 0.277586 | -0.025361 | 1.000000 | [1.000000, 1.000000] |
| hierarchical_prototype | main | 1.017693 | -0.089270 | 0.998160 | [0.995753, 0.999865] |
| hierarchical_prototype | subtype | 0.277586 | -0.025361 | 1.000000 | [1.000000, 1.000000] |

### Raw/prototype disagreements

| Prototype method | Mean disagreement rate | Repeated-run disagreements | Raw correct / prototype wrong | Prototype correct / Raw wrong | Both wrong | Rate fold-cluster 95% CI |
|---|---|---|---|---|---|---|
| main_prototype | 0.013571 | 57 | 29 | 28 | 0 | [0.007143, 0.019286] |
| hierarchical_prototype | 0.015714 | 66 | 37 | 29 | 0 | [0.009286, 0.020000] |

True-main and true-subtype prototype ranks are computed from ascending stored cosine distances with stable fixed-label tie breaking. Main and hierarchical prototype methods share the same auxiliary-prototype ordering; their subtype values are therefore not independent improvements. Favorable distance margin is `nearest wrong prototype distance - true prototype distance`; per-run error AUROC uses negative margin only as a diagnostic score and no test threshold is fitted.

Disagreement counts are reported per fold-seed before aggregation. The representative description remains: **“a training sample closest to the predicted class prototype”**. Case studies, where used elsewhere, remain deterministic and illustrative only.

## Convergence and stability

| Stage | Mean best epoch | SD | Median | Range | Provenance |
|---|---|---|---|---|---|
| B0 | 14.24 | 6.17 | 15.0 | 6-28 | direct |
| B1 | 14.48 | 7.50 | 13.0 | 3-27 | direct |
| B2 | 17.72 | 7.17 | 17.0 | 6-30 | direct_selected_checkpoint |
| B3 | 17.72 | 7.17 | 17.0 | 6-30 | inherited_from_selected_B2_checkpoint |

| Stage | Mean val-test gap | SD | Median | Range |
|---|---|---|---|---|
| B0 | 0.022564 | 0.020471 | 0.023807 | [-0.015695, 0.057361] |
| B1 | 0.024395 | 0.023920 | 0.028577 | [-0.032364, 0.069944] |
| B2 | 0.022809 | 0.019255 | 0.017846 | [-0.009510, 0.063747] |
| B3 | 0.024725 | 0.022620 | 0.020269 | [-0.009510, 0.063747] |

Complete epoch-level train-loss and validation-Macro-F1 histories were unavailable (`{'B0': '0/25', 'B1': '0/25', 'B2': '0/25', 'B3': 'not_applicable_post_hoc'}`). Median/IQR learning curves were therefore not emitted or regenerated. B3 is post-hoc inference and inherits the selected B2 checkpoint epoch and validation metric; it has no independent training convergence trajectory.

## Acceptance interpretation

**B. framework_functional_only**

- B2_valsel − B1 mean Δ: 0.004944
- B3_valsel − B2_valsel mean Δ: -0.001916
- Positive B3_valsel − B1 fold means: 3/5
- Major functional contradiction detected: True
- Measurable candidate-prediction value: True

The label is applied from the evidence literally rather than selected to favour the paper.

## Leakage, provenance, and reproducibility

- Train data: checkpoint fitting and train-only main/subtype prototypes.
- Validation data: λ selection by outer fold and the existing prototype temperature/penalty/rejection calibration protocol.
- Test data: frozen evaluation only.
- Every included prototype run passed exact-path, source-ID, and MD5 disjointness audits; exact path/source-ID/MD5/main-label/subtype membership against its train, validation, and frozen-test manifest; freshly recomputed checkpoint/summary/manifest/prototype-bundle/calibration/prediction SHA256 link verification; exact Softmax prediction/Macro-F1 reproduction; and `training_exact` feature qualification.
- Reproduced Raw Softmax probabilities had a maximum advisory absolute drift of 0.001078847 versus the canonical training output, with 0 full main-class rank-order mismatches. The compact canonical rows are first bound to frozen manifest truth, then joined one-to-one to exact path/source-ID/MD5 identities with zero truth or prediction mismatches. Functional Raw main-class ranks use those identity-aligned canonical probabilities; auxiliary Softmax probabilities use the `training_exact` reproduction because the original compact test CSV did not persist auxiliary scores.
- Frozen-source SHA256 files checked before and after analysis: 905; changed: 0.
- Missing required selected combinations after reconstruction: none.
- Existing epoch histories were searched in all 75 B0/B1/selected-B2 run directories; 0 candidate history files and 8 repository `.log` files were structurally inspected without retraining.

## Statistical fallacy scan (11/11 checked)

| Fallacy | Status | Audit finding |
|---|---|---|
| Simpson's paradox | NOTE | Aggregate and all five fold-level directions are reported; no aggregate-only inference is used. |
| Ecological fallacy | NOTE | Fold/run summaries are not converted into clip- or farm-level causal claims. |
| Berkson's paradox | CAUTION | The frozen closed-set cohort is selected and does not establish real-farm prevalence performance. |
| Collider bias | NOTE | No post-outcome covariate adjustment is performed. |
| Base-rate neglect | CAUTION | Top-k and Macro-F1 are reported as closed-set metrics, not diagnostic predictive values. |
| Regression to the mean | NOTE | All 25 matched runs are retained; no extreme run is selected. |
| Survivorship bias | NOTE | No fold-seed result is dropped; all required keys are present. |
| Look-elsewhere effect | NOTE | Seven requested contrasts are all shown and Holm-adjusted together. |
| Garden of forking paths | CAUTION | The analysis is explicitly validation-selected and retrospective, not fully confirmatory. |
| Correlation ≠ causation | NOTE | No causal biological or deployment claim is made from embedding geometry. |
| Reverse causality | NOTE | Not applicable to the fixed prediction comparison; no directional causal claim is made. |

## Reproducibility entry points

The audit branch is `paper/final-validation-audit`; the implementation and delivery commit SHAs are recorded in `handoff/CODEX_TO_GPT.md` and `handoff/CODEX_TO_GPT.json` after commit creation. From a checkout where the target output directory does not yet exist:

```powershell
cd C:/py/pigsound/pig-sound-classification
$env:PYTHONNOUSERSITE="1"
$env:NUMBA_CACHE_DIR=(Resolve-Path '.numba_cache').Path
& "C:/py/anaconda3/envs/pigsound-gpu/python.exe" tools/generate_final_validation_audit.py --root . --validate-only
& "C:/py/anaconda3/envs/pigsound-gpu/python.exe" tools/generate_final_validation_audit.py --root .
```

The one-time 20-run selected-lambda prototype reconstruction used the five existing exact-protocol commands without `--allow_overwrite`; its complete fold/seed loop, fixed grids, and output-directory patterns are recorded in the GPT handoff. No backbone was retrained.

## Output status

All required audit tables and the five preliminary SVG/PDF/PNG audit figures are under `paper/final_validation/`. These are not final manuscript figures. The manuscript was not regenerated.
