# Codex to GPT Handoff

## Task

Run the approved lambda 0.5 exact hierarchical acoustic prototype screening
experiment: 5 folds x 3 seeds, 15 fold-seed runs total.

## Status

Completed. Stop here. Do not start lambda 1.0, do not start 25-run final CV,
and do not start unknown/open-set experiments from this handoff.

## Branch

- Base PR: PR #1 squash-merged into `current-mctafd-ablation`
- PR squash merge commit: `9ee233c04ef377a6acbaeaf233da7137f2add2d4`
- Screening branch: `prototype-cv5-screening-w05`
- Note: local git fetch from GitHub was unavailable after the remote squash merge,
  so the local screening branch was fast-forwarded to the already reviewed
  `prototype-exact-review` tree before running. The remote PR itself is closed
  and merged.

## Scope

- lambda: 0.5 only
- folds: 0,1,2,3,4
- seeds: 42,2024,3407
- feature backend: `training_exact`
- train split builds prototypes
- validation split calibrates temperature, thresholds, fusion alpha, and hierarchy penalty
- frozen test split is evaluation only
- no full 25-run CV
- no lambda 1.0 comparison
- no training script changes

## Commands

Environment:

```powershell
$py = "C:\py\anaconda3\envs\pigsound-gpu\python.exe"
$env:PYTHONNOUSERSITE = "1"
$env:NUMBA_CACHE_DIR = (Join-Path (Get-Location) ".numba_cache")
```

Verification before runs:

```powershell
& $py tools\eval_hier_exact_softmax_reproduction.py --help
& $py tools\build_hier_acoustic_prototypes.py --help
& $py tools\calibrate_prototype_predictor.py --help
& $py tools\predict_hier_acoustic_prototype.py --help
& $py tools\eval_hier_acoustic_prototype.py --help
& $py tools\summarize_cv5_exact_prototype.py --help
& $py -m py_compile tools\prototype_model_adapter.py tools\build_hier_acoustic_prototypes.py tools\calibrate_prototype_predictor.py tools\predict_hier_acoustic_prototype.py tools\eval_hier_acoustic_prototype.py tools\eval_hier_exact_softmax_reproduction.py tools\summarize_cv5_exact_prototype.py tests\test_prototype_pipeline_core.py
& $py -m unittest tests.test_prototype_pipeline_core -v
```

For each fold and seed, the run sequence was:

```powershell
& $py tools\eval_hier_exact_softmax_reproduction.py --test_manifest <test.csv> --ckpt <w05 checkpoint> --summary_json <summary.json> --reference_pred_csv <reference test_pred.csv> --out_dir reports\prototype_cv5_exact_w05_fold{fold}_seed{seed} --fold <fold> --seed <seed> --expected_hier_aux_weight 0.5 --expected_macro_f1 <summary.test_macro_f1> --expected_test_rows 0 --batch_size 64 --num_workers 0 --numba_cache_dir .numba_cache
& $py tools\build_hier_acoustic_prototypes.py --train_manifest <train.csv> --val_manifest <val.csv> --test_manifest <test.csv> --ckpt <w05 checkpoint> --summary_json <summary.json> --out_dir reports\prototype_cv5_exact_w05_fold{fold}_seed{seed} --fold <fold> --seed <seed> --expected_hier_aux_weight 0.5 --expected_dur_s 2.0 --expected_feature_mode logmel --feature_backend training_exact --batch_size 64 --num_workers 0
& $py tools\calibrate_prototype_predictor.py --prototype_bundle reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}\artifacts\prototype_bundle.npz --val_manifest <val.csv> --ckpt <w05 checkpoint> --out_dir reports\prototype_cv5_exact_w05_fold{fold}_seed{seed} --feature_backend auto --batch_size 64 --num_workers 0
& $py tools\predict_hier_acoustic_prototype.py --prototype_bundle reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}\artifacts\prototype_bundle.npz --calibration_json reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}\calibration\calibration.json --test_manifest <test.csv> --ckpt <w05 checkpoint> --out_dir reports\prototype_cv5_exact_w05_fold{fold}_seed{seed} --feature_backend auto --batch_size 64 --num_workers 0
& $py tools\eval_hier_acoustic_prototype.py --pred_csv reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}\evaluation\test_predictions.csv --calibration_json reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}\calibration\calibration.json --out_dir reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}
```

Aggregate:

```powershell
& $py tools\summarize_cv5_exact_prototype.py --metrics_json reports\prototype_cv5_exact_w05_fold0_seed42\evaluation\metrics.json ... reports\prototype_cv5_exact_w05_fold4_seed3407\evaluation\metrics.json --expected_folds 0,1,2,3,4 --expected_seeds 42,2024,3407 --expected_lambda 0.5 --out_prefix reports\prototype_cv5_exact_w05
```

## Preflight

- 15 checkpoints found and loadable.
- 15 matching `summary.json` files found.
- 15 matching reference `test_pred.csv` files found.
- fold0-fold4 train/val/test manifests found.
- All summaries passed `hier_aux_weight == 0.5`, `dur_s == 2.0`,
  `feature_mode == logmel`, seed, and label-order checks.
- Checkpoints are pure state_dict files without hyperparameter metadata, so the
  lambda guard was enforced via the matching summary and w05 file family.

## Aggregate Provenance

`reports/prototype_cv5_exact_w05_screening_provenance.json`:

```json
{
  "run_scope": "aggregate",
  "screening_result": true,
  "final_25_run_result": false,
  "paper_candidate_result": true,
  "paper_main_result": false,
  "n_runs": 15,
  "lambda": 0.5
}
```

## Method Summary

| method | mean Macro-F1 | std Macro-F1 | mean Top-1 | mean Top-2 | mean ECE | mean Brier | mean NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_softmax | 0.950244 | 0.013218 | 0.950397 | 1.000000 | 0.030945 | 0.076914 | 0.128189 |
| calibrated_softmax | 0.950244 | 0.013218 | 0.950397 | 1.000000 | 0.027278 | 0.076032 | 0.128291 |
| prototype | 0.952333 | 0.012754 | 0.952381 | 1.000000 | 0.025533 | 0.070229 | 0.117694 |
| hierarchical | 0.955499 | 0.010793 | 0.955556 | 1.000000 | 0.029313 | 0.071411 | 0.128270 |
| fused | 0.954289 | 0.009766 | 0.954365 | 1.000000 | 0.027336 | 0.072093 | 0.121912 |

## Paired Statistics

| comparison | n | mean delta | std delta | CI95 low | CI95 high | Wilcoxon p | wins/ties/losses |
|---|---:|---:|---:|---:|---:|---:|---|
| hierarchical - raw_softmax | 15 | 0.005256 | 0.012493 | 0.000026 | 0.012109 | 0.055808 | 9/4/2 |
| prototype - raw_softmax | 15 | 0.002089 | 0.009631 | -0.001979 | 0.007360 | 0.342829 | 6/6/3 |
| hierarchical - prototype | 15 | 0.003166 | 0.007060 | -0.000003 | 0.006741 | 0.176296 | 5/8/2 |

## Feeding / Stress Confusion

Aggregated over 15 runs:

| method | macro F1 from CM | feeding F1 | stress F1 | feeding recall | stress recall | feeding -> stress | stress -> feeding |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_softmax | 0.950390 | 0.899598 | 0.901961 | 0.888889 | 0.912698 | 70 | 55 |
| prototype | 0.952377 | 0.903846 | 0.905660 | 0.895238 | 0.914286 | 66 | 54 |
| hierarchical | 0.955537 | 0.909238 | 0.912908 | 0.890476 | 0.931746 | 69 | 43 |

## Calibration Distribution

- calibrated softmax temperature: `{1.0: 4, 0.5: 3, 0.75: 8}`
- prototype temperature: `{0.07: 7, 0.1: 3, 0.03: 2, 0.05: 3}`
- hierarchical prototype temperature: `{0.07: 6, 0.1: 2, 0.03: 2, 0.05: 5}`
- hierarchical auxiliary probability weight: `{0.5: 15}`
- fused alpha / softmax weight: `{0.25: 1, 1.0: 8, 0.0: 2, 0.5: 1, 0.75: 3}`
- fusion kind: `{mixed: 5, pure_softmax: 8, pure_prototype: 2}`
- hierarchy confidence penalty: `{1.0: 14, 0.85: 1}`

## Coverage / Selective Risk

Mean over 15 runs:

| accept rule | mean coverage | std coverage | mean selective risk | std selective risk |
|---|---:|---:|---:|---:|
| accept_global | 0.928968 | 0.045802 | 0.022369 | 0.010111 |
| accept_per_class | 0.936905 | 0.038954 | 0.030587 | 0.011742 |

## Reproduction, Leakage, and Qualification

- 15/15 exact Softmax reproduction runs passed.
- Every reproduced prediction file matched the reference `test_pred.csv` by
  normalized path, `y_true`, and `y_pred`.
- Every run had 168/168 reference prediction matches.
- 15/15 runs used `feature_backend=training_exact`.
- 15/15 runs used `input_role=frozen_test`.
- 15/15 runs had `manifest_sha_verified=true`.
- 15/15 runs had `leakage_audit_ok=true`.
- 15/15 runs had `eligible_for_cv_aggregation=true`.
- No unsafe checkpoint SHA mismatch flag was used.

## Outputs

Aggregate files:

- `reports/prototype_cv5_exact_w05_screening_runs.csv`
- `reports/prototype_cv5_exact_w05_screening_summary.csv`
- `reports/prototype_cv5_exact_w05_screening_paired_stats.csv`
- `reports/prototype_cv5_exact_w05_screening_confusion_summary.csv`
- `reports/prototype_cv5_exact_w05_screening_provenance.json`

Per-run directories:

- `reports/prototype_cv5_exact_w05_fold{fold}_seed{seed}/`

Do not commit or publish NPZ prototype bundles, checkpoints, audio, feature
caches, or embedding arrays as paper artifacts.

## Paper Usability

- `paper_usable_as_screening_evidence=true`
- `paper_candidate_result=true`
- `paper_main_result=false`
- This is a 15-run screening result, not the final 25-run paper main result.
- The hierarchical mean Macro-F1 is higher than raw Softmax, and the paired mean
  delta is positive, but the Wilcoxon p-value is `0.055808`; do not describe it
  as statistically significant.

## Recommended Next Step

Scientific decision needed: decide whether the screening improvement and
feeding/stress reduction justify running the final 5 folds x 5 seeds lambda 0.5
exact prototype aggregate. Do not run it without explicit approval.
