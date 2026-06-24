# Codex to GPT Handoff

## Task

Run the approved lambda 0.5 exact hierarchical acoustic prototype final CV completion: add the missing seeds `123` and `777` for folds `0,1,2,3,4`, then aggregate all 25 fold-seed metrics.

## Status

Completed. Stop here. Do not start lambda 1.0, noise robustness, unknown/open-set evaluation, or paper writing from this handoff.

## Branch and commits

- Branch: `prototype-cv5-final-w05`
- Created from: `prototype-cv5-screening-w05`
- Screening commit: `e930fe1cc3bae9e20e17e7f2766a12231b221fe7`
- Merged prototype pipeline commit: `9ee233c04ef377a6acbaeaf233da7137f2add2d4`
- Final commit SHA: pending at handoff write time; use pushed branch HEAD / Codex final response.

## Scope

- lambda: `0.5`
- folds: `0,1,2,3,4`
- seeds: `42,123,777,2024,3407`
- new runs this round: `10` (`123`, `777` for each fold)
- total aggregate runs: `25`
- feature backend: `training_exact`
- selection method: `hierarchical`
- target coverage: `0.95`
- existing 15 runs rerun: no
- lambda 1.0 / noise / unknown-open-set started: no

## Commands

Environment:

```powershell
$py = "C:\py\anaconda3\envs\pigsound-gpu\python.exe"
$env:PYTHONNOUSERSITE = "1"
$env:NUMBA_CACHE_DIR = (Join-Path (Get-Location) ".numba_cache")
```

For each new fold-seed, the executed sequence was:

```powershell
& $py tools\eval_hier_exact_softmax_reproduction.py --test_manifest <test.csv> --ckpt <w05 checkpoint> --summary_json <summary.json> --reference_pred_csv <reference test_pred.csv> --out_dir reports\prototype_cv5_exact_w05_fold{fold}_seed{seed} --fold <fold> --seed <seed> --expected_hier_aux_weight 0.5 --expected_macro_f1 <summary.test_macro_f1> --expected_test_rows 0 --batch_size 64 --num_workers 0 --numba_cache_dir .numba_cache
& $py tools\build_hier_acoustic_prototypes.py --train_manifest <train.csv> --val_manifest <val.csv> --test_manifest <test.csv> --ckpt <w05 checkpoint> --summary_json <summary.json> --out_dir reports\prototype_cv5_exact_w05_fold{fold}_seed{seed} --fold <fold> --seed <seed> --expected_hier_aux_weight 0.5 --expected_dur_s 2.0 --expected_feature_mode logmel --feature_backend training_exact --batch_size 64 --num_workers 0
& $py tools\calibrate_prototype_predictor.py --prototype_bundle reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}\artifacts\prototype_bundle.npz --val_manifest <val.csv> --ckpt <w05 checkpoint> --out_dir reports\prototype_cv5_exact_w05_fold{fold}_seed{seed} --feature_backend auto --selection_method hierarchical --target_coverage 0.95 --prototype_temperature_grid 0.03,0.05,0.07,0.1,0.2,0.5,1.0 --softmax_temperature_grid 0.5,0.75,1.0,1.5,2.0 --fusion_weight_grid 0.0,0.25,0.5,0.75,1.0 --hierarchy_penalty_grid 0.5,0.7,0.85,1.0 --batch_size 64 --num_workers 0
& $py tools\predict_hier_acoustic_prototype.py --prototype_bundle reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}\artifacts\prototype_bundle.npz --calibration_json reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}\calibration\calibration.json --test_manifest <test.csv> --ckpt <w05 checkpoint> --out_dir reports\prototype_cv5_exact_w05_fold{fold}_seed{seed} --feature_backend auto --batch_size 64 --num_workers 0
& $py tools\eval_hier_acoustic_prototype.py --pred_csv reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}\evaluation\test_predictions.csv --calibration_json reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}\calibration\calibration.json --out_dir reports\prototype_cv5_exact_w05_fold{fold}_seed{seed}
```

Aggregate:

```powershell
& $py tools\summarize_cv5_exact_prototype.py --metrics_json <25 metrics.json files> --expected_folds 0,1,2,3,4 --expected_seeds 42,123,777,2024,3407 --expected_lambda 0.5 --out_prefix reports\prototype_cv5_exact_w05
```

## Aggregate provenance

```json
{
  "run_scope": "aggregate",
  "screening_result": false,
  "final_25_run_result": true,
  "paper_candidate_result": true,
  "paper_main_result": true,
  "n_runs": 25,
  "folds": [
    0,
    1,
    2,
    3,
    4
  ],
  "seeds": [
    42,
    123,
    777,
    2024,
    3407
  ],
  "lambda": 0.5
}
```

## Method summary

| method | mean Macro-F1 | std Macro-F1 | mean ECE | mean Brier | mean NLL |
|---|---:|---:|---:|---:|---:|
| raw_softmax | 0.951050 | 0.012393 | 0.030497 | 0.075434 | 0.126036 |
| calibrated_softmax | 0.951050 | 0.012393 | 0.027336 | 0.075121 | 0.125565 |
| prototype | 0.951852 | 0.013104 | 0.026173 | 0.070642 | 0.117056 |
| hierarchical | 0.953986 | 0.012004 | 0.030626 | 0.072452 | 0.127334 |
| fused | 0.953486 | 0.010343 | 0.026629 | 0.072423 | 0.121151 |

## Paired statistics

- hierarchical - raw_softmax: n=25, mean delta=0.002937, std=0.010336, CI95=[-0.000478, 0.007294], Wilcoxon p=0.058253, wins/ties/losses=13/8/4
- hierarchical - prototype: n=25, mean delta=0.002134, std=0.007270, CI95=[-0.000499, 0.004996], Wilcoxon p=0.221330, wins/ties/losses=8/12/5

Do not claim statistical significance for either comparison.

## Fold-level deltas

| fold | hier - raw | hier - prototype | h/raw W/T/L | h/proto W/T/L |
|---:|---:|---:|---:|---:|
| 0 | -0.001169 | 0.000002 | 2/2/1 | 1/3/1 |
| 1 | 0.013335 | 0.009479 | 3/2/0 | 4/0/1 |
| 2 | 0.001204 | 0.001189 | 2/3/0 | 1/4/0 |
| 3 | 0.000107 | -0.001187 | 4/0/1 | 1/2/2 |
| 4 | 0.001205 | 0.001185 | 2/1/2 | 1/3/1 |

Four of five folds have positive mean `hierarchical - raw_softmax`; fold0 is slightly negative on average.

## Feeding / stress boundary

- raw Softmax: feeding F1 `0.901679`, stress F1 `0.903073`, feeding->stress `110`, stress->feeding `95`
- prototype: feeding F1 `0.902885`, stress F1 `0.904717`, feeding->stress `111`, stress->feeding `91`
- hierarchical: feeding F1 `0.906265`, stress F1 `0.909855`, feeding->stress `117`, stress->feeding `76`

## Coverage / selective risk

- accept_global: mean coverage `0.935952`, std coverage `0.041464`, mean selective risk `0.024118`, min/max selective risk `0.006757/0.048193`
- accept_per_class: mean coverage `0.939286`, std coverage `0.036002`, mean selective risk `0.031446`, min/max selective risk `0.006849/0.049383`

## Calibration distribution

- softmax temperature: `{'0.5': 4, '0.75': 12, '1.0': 9}`
- main prototype temperature: `{'0.03': 2, '0.05': 4, '0.07': 14, '0.1': 5}`
- hierarchical prototype temperature: `{'0.03': 2, '0.05': 9, '0.07': 9, '0.1': 4, '0.2': 1}`
- hierarchy penalty: `{'0.85': 1, '1.0': 24}`
- fusion alpha: `{'0.0': 3, '0.25': 1, '0.5': 1, '0.75': 6, '1.0': 14}`
- fusion_kind: `{'mixed': 8, 'pure_prototype': 3, 'pure_softmax': 14}`
- global rejection threshold stats: `{'mean': 0.768498, 'std': 0.108303, 'min': 0.585346, 'max': 0.968976}`

Full threshold distribution is in `reports/prototype_cv5_exact_w05_final_calibration_distribution.csv`.

## Leakage and qualification

- 25/25 exact Softmax reproduction runs passed.
- 25/25 reference sample paths matched exactly.
- 25/25 reference `y_true` matched exactly.
- 25/25 reference `y_pred` matched exactly.
- train/val/test path overlap: `0`.
- train/val/test source_id overlap: `0`.
- train/val/test MD5 overlap: `0`.
- 25/25 used `feature_backend=training_exact`.
- 25/25 used `input_role=frozen_test`.
- 25/25 had `manifest_sha_verified=true`.
- 25/25 had `leakage_audit_ok=true`.
- 25/25 had `eligible_for_cv_aggregation=true`.
- No unsafe checkpoint SHA mismatch flag was used.
- No test-set calibration or parameter selection was performed.

## Outputs

Aggregate files:

- `reports/prototype_cv5_exact_w05_final_runs.csv`
- `reports/prototype_cv5_exact_w05_final_summary.csv`
- `reports/prototype_cv5_exact_w05_final_paired_stats.csv`
- `reports/prototype_cv5_exact_w05_final_confusion_summary.csv`
- `reports/prototype_cv5_exact_w05_final_provenance.json`

Additional final analyses:

- `reports/prototype_cv5_exact_w05_final_by_fold.csv`
- `reports/prototype_cv5_exact_w05_final_calibration_distribution.csv`
- `reports/prototype_cv5_exact_w05_final_selective_risk_summary.csv`

Do not commit or publish checkpoints, audio, NPZ prototype bundles, embedding arrays, caches, full `test_predictions.csv`, or atlas image collections as paper artifacts.

## Runtime and notes

- Device setting: `auto`, matching the screening-stage runner style.
- CUDA was available after the run: `NVIDIA GeForce RTX 5070 Ti`.
- CPU fallback used: no.
- Feature backend switch: no.
- Initial launcher note: one background wrapper attempt invoked bare Python before any experiment tool ran; no output directory was created. Logs were preserved locally as `reports/prototype_cv5_exact_w05_final_run10_stdout.log` and stderr companion. The corrected retry completed all 10 runs.

## Paper usability

- `paper_candidate_result=true`
- `paper_main_result=true`
- `final_25_run_result=true`
- Main interpretation: hierarchical exact prototype has a positive mean Macro-F1 delta over raw Softmax, but Wilcoxon p is `0.058253`; do not describe this as statistically significant.

## Blockers

None.

## Questions requiring scientific judgment

1. How should the paper phrase the positive but nonsignificant hierarchical-prototype gain over raw Softmax?
2. Should the final paper emphasize prototype interpretability and feeding/stress boundary behavior rather than statistical superiority?
3. Which final aggregate tables should be copied into `paper_results/` after review?

## Recommended next step

Scientific review only. Do not start lambda 1.0, noise robustness, unknown/open-set work, or paper writing without a new `APPROVED: true` handoff.
