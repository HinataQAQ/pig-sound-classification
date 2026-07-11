# Codex to GPT Handoff — Final Validation Audit

## Status

Completed the approved final scientific validation pass before figure
regeneration and manuscript production. The locked fold-wise validation-only
lambda analysis, validation-selected B2/B3 evaluation, prototype functional
audit, and existing-log convergence audit are complete.

No backbone was retrained. Twenty missing selected-lambda prototype runs were
reconstructed from existing checkpoints with train-only prototypes,
validation-only calibration, and one frozen-test prediction/evaluation pass.
Five fold-4 lambda-0.5 prototype runs were reused. No test-derived parameter,
threshold, grid, case, or lambda was selected.

## Branch and commits

- Branch: `paper/final-validation-audit`.
- Base commit: `ccf3171be6c53a75299427bc1cf726377a9c1c29`.
- Audit implementation/output commit:
  `69366cbf8f60fd4ed4068039a9d78629ab7a41ef`.
- Handoff commit: reported as delivery HEAD in the final Codex response.

## Exact environment and commands

Run from `C:\py\pigsound\pig-sound-classification`:

```powershell
$env:PYTHONNOUSERSITE="1"
$env:NUMBA_CACHE_DIR=(Resolve-Path '.numba_cache').Path
$python="C:\py\anaconda3\envs\pigsound-gpu\python.exe"
```

`conda activate pigsound-gpu` was attempted first, but the local shell has a
GBK/invalid-PATH-character activation failure. The exact environment Python
above was therefore used; no dependency was installed.

The following locked loop is the exact five-command protocol used for the 20
missing selected combinations. It deliberately excludes fold 4 because its
selected lambda is 0.5 and all five exact runs already existed. It never passes
`--allow_overwrite`.

```powershell
$selected=@(
  [pscustomobject]@{Fold=0; Lambda=1.0; Code='10'},
  [pscustomobject]@{Fold=1; Lambda=0.2; Code='02'},
  [pscustomobject]@{Fold=2; Lambda=0.2; Code='02'},
  [pscustomobject]@{Fold=3; Lambda=0.2; Code='02'}
)
$seeds=42,123,777,2024,3407

foreach($item in $selected){
  foreach($seed in $seeds){
    $fold=$item.Fold
    $lambda=$item.Lambda
    $code=$item.Code
    $manifestRoot="paper_results/manifests/manifests_pigvocal_4class_expanded_train_cv5_cap3x/fold$fold"
    $sourceRoot="reports/cv5_expanded_cap3x_fold${fold}_logmel_dur2_hier_w${code}_seed${seed}"
    $out="reports/prototype_cv5_exact_valsel_w${code}_fold${fold}_seed${seed}"
    $ckpt="checkpoints/cv5_expanded_cap3x_fold${fold}_logmel_dur2_hier_w${code}_seed${seed}.pt"
    $summary="$sourceRoot/summary.json"
    $expectedMacro=(Get-Content $summary -Raw | ConvertFrom-Json).test_macro_f1

    & $python tools/eval_hier_exact_softmax_reproduction.py `
      --test_manifest "$manifestRoot/test.csv" --ckpt $ckpt `
      --summary_json $summary --reference_pred_csv "$sourceRoot/test_pred.csv" `
      --out_dir $out --fold $fold --seed $seed `
      --expected_hier_aux_weight $lambda --expected_macro_f1 $expectedMacro `
      --expected_test_rows 168 --macro_f1_tolerance 1e-6 `
      --prob_tolerance 1e-5 --device cuda --batch_size 64 --num_workers 0 `
      --numba_cache_dir $env:NUMBA_CACHE_DIR
    if($LASTEXITCODE -ne 0){ throw "Softmax reproduction failed: $out" }

    & $python tools/build_hier_acoustic_prototypes.py `
      --train_manifest "$manifestRoot/train.csv" `
      --val_manifest "$manifestRoot/val.csv" `
      --test_manifest "$manifestRoot/test.csv" --ckpt $ckpt `
      --summary_json $summary --out_dir $out --fold $fold --seed $seed `
      --expected_hier_aux_weight $lambda --feature_backend training_exact `
      --prototype_temperature 0.07 --hier_aux_prob_weight $lambda `
      --device cuda --batch_size 32 --num_workers 0
    if($LASTEXITCODE -ne 0){ throw "Prototype build failed: $out" }

    & $python tools/calibrate_prototype_predictor.py `
      --prototype_bundle "$out/artifacts/prototype_bundle.npz" `
      --val_manifest "$manifestRoot/val.csv" --ckpt $ckpt --out_dir $out `
      --feature_backend training_exact --selection_method hierarchical `
      --prototype_temperature_grid '0.03,0.05,0.07,0.1,0.2,0.5,1.0' `
      --softmax_temperature_grid '0.5,0.75,1.0,1.5,2.0' `
      --fusion_weight_grid '0.0,0.25,0.5,0.75,1.0' `
      --hierarchy_penalty_grid '0.5,0.7,0.85,1.0' `
      --hier_aux_prob_weight $lambda --target_coverage 0.95 `
      --per_class_min_count 5 --device cuda --batch_size 32 --num_workers 0
    if($LASTEXITCODE -ne 0){ throw "Calibration failed: $out" }

    & $python tools/predict_hier_acoustic_prototype.py `
      --prototype_bundle "$out/artifacts/prototype_bundle.npz" `
      --calibration_json "$out/calibration/calibration.json" `
      --test_manifest "$manifestRoot/test.csv" --ckpt $ckpt --out_dir $out `
      --feature_backend training_exact --device cuda --batch_size 32 --num_workers 0
    if($LASTEXITCODE -ne 0){ throw "Frozen prediction failed: $out" }

    & $python tools/eval_hier_acoustic_prototype.py `
      --pred_csv "$out/evaluation/test_predictions.csv" `
      --calibration_json "$out/calibration/calibration.json" --out_dir $out
    if($LASTEXITCODE -ne 0){ throw "Evaluation failed: $out" }
  }
}
```

Audit generation and verification:

```powershell
& $python tools/generate_final_validation_audit.py --help
& $python tools/eval_hier_exact_softmax_reproduction.py --help
& $python tools/build_hier_acoustic_prototypes.py --help
& $python tools/calibrate_prototype_predictor.py --help
& $python tools/predict_hier_acoustic_prototype.py --help
& $python tools/eval_hier_acoustic_prototype.py --help
& $python tools/generate_cumulative_framework_analysis.py --help
& $python tools/generate_final_validation_audit.py --root . --validate-only
& $python tools/generate_final_validation_audit.py --root .
& $python tools/generate_cumulative_framework_analysis.py --root . --validate-only
& $python -m py_compile tools/generate_final_validation_audit.py tests/test_generate_final_validation_audit.py
& $python -m unittest discover -s tests -v
git diff --cached --check
```

The generator refuses existing outputs. The final full write followed a safety
check that the disposable target was exactly
`C:\py\pigsound\pig-sound-classification\paper\final_validation`.

## Fold-wise validation-only lambda selection

Selection used the five-seed mean `best_val_macro_f1`, then lowest validation
sample SD when means differed by at most `1e-6`, then smallest lambda. No test
metric participated.

| Fold | Selected lambda | lambda 0.2 mean/SD | lambda 0.5 mean/SD | lambda 1.0 mean/SD |
|---:|---:|---:|---:|---:|
| 0 | 1.0 | 0.985000 / 0.003726 | 0.985000 / 0.003726 | 0.986666 / 0.004563 |
| 1 | 0.2 | 0.973319 / 0.006974 | 0.968321 / 0.006975 | 0.969967 / 0.004588 |
| 2 | 0.2 | 0.981660 / 0.006974 | 0.978332 / 0.007454 | 0.979984 / 0.007466 |
| 3 | 0.2 | 0.964987 / 0.006973 | 0.963307 / 0.007470 | 0.961619 / 0.007477 |
| 4 | 0.5 | 0.964964 / 0.006978 | 0.968321 / 0.003699 | 0.966596 / 0.005908 |

## Stage results

| Stage | n | Mean Macro-F1 | SD | Min | Max |
|---|---:|---:|---:|---:|---:|
| B0 | 25 | 0.922606467 | 0.014540638 | 0.891876430 | 0.946238088 |
| B1 | 25 | 0.947237358 | 0.016763541 | 0.921268926 | 0.982140326 |
| B2_valsel | 25 | 0.952181754 | 0.013261982 | 0.927917620 | 0.976176971 |
| B3_valsel | 25 | 0.950265803 | 0.016527704 | 0.921268926 | 0.976176971 |
| B2 fixed lambda 0.5 | 25 | 0.951049673 | 0.012393314 | 0.921720430 | 0.976176971 |
| B3 fixed lambda 0.5 | 25 | 0.953986214 | 0.012003773 | 0.934290997 | 0.976176971 |

## Requested paired statistics

All comparisons use the exact 25 matched `(fold, seed)` rows. Run and
five-fold cluster intervals use 10,000 percentile resamples and seed 3407.
Holm adjustment is applied once across these seven tests.

| Comparison | Mean delta | Run CI95 | Raw P | Holm P | W/T/L | Fold deltas 0–4 | Fold CI95 |
|---|---:|---|---:|---:|---|---|---|
| B2_valsel - B1 | 0.004944395 | [0.000131304, 0.009698122] | 0.122846338 | 0.614231688 | 17/1/7 | 0.004773867, 0.006991420, 0.011945648, 0.005734818, -0.004723776 | [-0.000481208, 0.009461316] |
| B2_valsel - B0 | 0.029575286 | [0.020723093, 0.038717612] | 0.000001967 | 0.000013769 | 22/0/3 | 0.004603535, 0.036234559, 0.021870085, 0.019115934, 0.066052319 | [0.013832220, 0.048161663] |
| B2 fixed - B2_valsel | -0.001132080 | [-0.004734672, 0.002393660] | 0.629162352 | 0.955072713 | 8/6/11 | 0.007393168, -0.014291350, -0.002393454, 0.003631234, 0 | [-0.008327254, 0.004435901] |
| B3_valsel - B2_valsel | -0.001915950 | [-0.005377724, 0.001496912] | 0.477536357 | 0.955072713 | 6/8/11 | -0.006279508, 0.002516646, -0.003574300, -0.003447478, 0.001204888 | [-0.004631019, 0.000799118] |
| B3_valsel - B1 | 0.003028445 | [-0.001625805, 0.007589397] | 0.290397166 | 0.871191498 | 13/1/11 | -0.001505641, 0.009508066, 0.008371347, 0.002287340, -0.003518888 | [-0.001552344, 0.007609233] |
| B3_valsel - B0 | 0.027659336 | [0.017601613, 0.037928472] | 0.000026643 | 0.000159860 | 20/0/5 | -0.001675973, 0.038751204, 0.018295785, 0.015668456, 0.067257207 | [0.009256150, 0.050153605] |
| B3 fixed - B3_valsel | 0.003720411 | [-0.001661740, 0.008861914] | 0.127398044 | 0.614231688 | 12/7/6 | 0.012504045, -0.003473160, 0.002385005, 0.007186166, 0 | [-0.000912263, 0.008939660] |

B2_valsel exceeds B1 on mean, but its raw Wilcoxon P is nonsignificant and its
five-fold cluster CI crosses zero. B3_valsel is lower than B2_valsel on mean.
The fixed lambda-0.5 B3 mean is higher than validation-selected B3, but that
sensitivity contrast is also nonsignificant. Do not claim a significant
hierarchical or prototype increment.

## Prototype functional findings

Run means and five-fold cluster CIs are in the delivered CSVs. Core values:

| Method | Main Top-1/Top-2 | Subtype Top-1/Top-2 | Hierarchy consistency | Error if consistent / inconsistent | Feeding-stress exact-pair Top-2 |
|---|---|---|---:|---|---:|
| Raw Softmax | 0.952381 / 1.000000 | 0.843571 / 0.988333 | 0.983571 | 0.042855 / 0.311111 | 0.987619 |
| Main prototype | 0.952143 / 1.000000 | 0.858095 / 0.986905 | 0.988571 | 0.042135 / 0.541667 | 0.997143 |
| Hierarchical prototype | 0.950476 / 1.000000 | 0.858095 / 0.986905 | 0.991667 | 0.044371 / 0.683333 | 0.996667 |

True prototype geometry:

- main mean rank `1.047857`, rank-1 `0.952143`, rank-2 `1.0`, MRR `0.976071`;
- subtype mean rank `1.155000`, rank-1 `0.858095`, rank-2 `0.986905`, MRR `0.926865`;
- hierarchical main favorable-margin error AUROC `0.998160`, fold CI
  `[0.995753, 0.999865]`;
- prototype subtype AUROC is `1.0`, but this is nearly tautological nearest-
  prototype geometry and must not be presented as a deployable uncertainty
  threshold;
- Raw versus main-prototype repeated-run disagreements: 57, with 29 harmed,
  28 rescued, and 0 both wrong;
- Raw versus hierarchical-prototype disagreements: 66, with 37 harmed,
  29 rescued, and 0 both wrong.

The subtype prototype ordering is shared by the main and hierarchical
prototype routes; it is not two independent subtype improvements. Case studies
remain deterministic and illustrative. The representative wording is exactly:

> a training sample closest to the predicted class prototype

## Convergence and stability

| Stage | Mean best epoch | SD | Median | Range | Mean validation-test gap | Gap SD |
|---|---:|---:|---:|---|---:|---:|
| B0 | 14.24 | 6.17 | 15 | 6–28 | 0.022564 | 0.020471 |
| B1 | 14.48 | 7.50 | 13 | 3–27 | 0.024395 | 0.023920 |
| B2 selected | 17.72 | 7.17 | 17 | 6–30 | 0.022809 | 0.019255 |
| B3 selected | 17.72 | 7.17 | 17 | 6–30 | 0.024725 | 0.022620 |

B3 has no independent training epoch and inherits the selected B2 checkpoint
epoch and validation metric. A real search of all 75 B0/B1/selected-B2 run
directories and eight repository log files found complete epoch histories in
`0/25`, `0/25`, and `0/25` runs, respectively. Median/IQR learning curves were
not invented or regenerated.

## Acceptance interpretation

**B. `framework_functional_only`**

- B2_valsel - B1 mean delta: `+0.004944395`;
- B3_valsel - B2_valsel mean delta: `-0.001915950`;
- only `3/5` B3_valsel - B1 fold means are positive;
- prototype harm exceeds rescue for the hierarchical route;
- Top-2, hierarchy-conditioned error, rank, and margin analyses nevertheless
  provide measurable candidate-prediction/interpretive value.

The label follows the evidence literally. Outcome A is not supported.

## Leakage, provenance, and immutable-source audit

- Lambda selection reads 75 validation summaries only.
- Every selected/fixed prototype run validates exact train/validation/test
  path, source-ID, MD5, main label, and subtype membership.
- Main and subtype prototypes contain exactly the train-manifest identities;
  calibration rows contain exactly validation-manifest identities; evaluation
  rows contain exactly frozen-test identities.
- At least 32 recorded checkpoint/summary/manifest/bundle/calibration/
  prediction SHA links are freshly recomputed per selected run.
- Exact Softmax predictions and Macro-F1 reproduce for all selected runs.
- Canonical Raw probabilities are manifest-truth-bound and then joined
  one-to-one by path/source-ID/MD5; main truth and prediction mismatches are
  both zero. Maximum advisory probability drift is `0.001078847`, with zero
  full rank-order mismatches.
- The final immutable boundary contains 905 source/artifact files; all pre/post
  SHA256 values are identical.
- Missing selected artifacts after reconstruction: none.
- Test data is evaluation only; no test threshold or margin cutoff was fitted.

The compact B0/B1 prediction CSVs do not persist sample identities, so their
per-clip identity cannot be independently re-proven from those files alone.
They are the established fold-specific results on this branch; all aggregate
comparisons remain strict by `(fold, seed)`. Do not convert that limitation
into a source-independence or real-farm generalization claim.

## Files changed and generated

Committed audit package:

- `tools/generate_final_validation_audit.py`;
- `tests/test_generate_final_validation_audit.py`;
- `docs/superpowers/plans/2026-07-11-final-validation-audit.md`;
- all 12 required top-level files under `paper/final_validation/`;
- all five preliminary audit figures in SVG/PDF/300-dpi PNG under
  `paper/final_validation/figures/`.

Handoff-only commit:

- `handoff/CODEX_TO_GPT.md`;
- `handoff/CODEX_TO_GPT.json`;
- `handoff/HISTORY.md`.

The 20 reconstructed prototype directories contain 660 files and 88,465,995
bytes under these patterns:

- `reports/prototype_cv5_exact_valsel_w10_fold0_seed*/`;
- `reports/prototype_cv5_exact_valsel_w02_fold{1,2,3}_seed*/`.

They are preserved locally but intentionally excluded from Git by the approved
plan, together with all pre-existing untracked results. The paper audit package
and provenance are committed; long-term portability of the 20 detailed run
directories requires a separate result-artifact archive policy.

## Verification

- New final-validation tests: 23/23 passed.
- Prototype/cumulative focused regressions: 69/69 passed.
- Full repository suite: 141/141 passed.
- New script/test `py_compile`: passed.
- Seven relevant `--help` commands: passed.
- New validate-only pass: 905 sources unchanged.
- Prior cumulative validate-only pass: 240 sources unchanged.
- Required outputs: 27/27 present and non-empty.
- CSV row contracts: 5, 25, 6, 7, 35, 67, 186, 62, 100, 100.
- Five PNGs: visually inspected at original resolution; 300 dpi.
- Five SVGs: editable text; mechanically normalized trailing whitespace.
- Five PDFs: non-empty.
- Output overwrite refusal: passed.
- `git diff --cached --check`: passed.
- Independent code/scientific review: Ready, no Critical or Important issues.
- Independent numerical recomputation: selection, grid, means/SDs, Wilcoxon,
  and Holm all agree within `1e-15`.

## Paper usability, blockers, and GPT Pro questions

The validation tables, report, and preliminary audit figures are paper-usable
evidence, but they are not final manuscript figures. The manuscript and prior
validated figures were not regenerated or edited.

There is no missing-artifact or implementation blocker. Scientific/editorial
judgment remains necessary for:

1. whether the retrospective validation-selected analysis belongs in the main
   text or supplementary material;
2. how prominently to show that fixed lambda-0.5 B3 has a higher mean than
   B3_valsel while the sensitivity contrast is nonsignificant;
3. how much to reduce performance language for B3 given its negative mean
   increment and 37 harmed versus 29 rescued repeated-run disagreements;
4. how to present prototype margin AUROC strictly as diagnostic geometry, not
   deployable uncertainty or a fitted rejection threshold;
5. whether the 20 detailed local reconstruction directories need a separate
   durable artifact archive before release.

## Stop point

Review this final validation evidence and choose the manuscript framing. Do not
regenerate the manuscript, regenerate final paper figures, start another
experiment, or implement Atlas/few-shot/open-set/noise work without a new
explicit `APPROVED: true` stage.
