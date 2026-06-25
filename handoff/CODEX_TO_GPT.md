# Codex to GPT Handoff - DEMAND Prescreen Fixes

## Status

Completed the requested prescreen protocol fixes and ran only the approved
fold0/seed3407 full-grid smoke. The full 15-run screening remains
`APPROVED: false` and was not started.

## Branch

- Branch: `codex/demand-noise-prescreen-fix`
- Base branch: `codex/demand-noise-debug`
- Final commit SHA: pending at handoff write; see pushed branch HEAD/final Codex
  response

## Scope Completed

- Fixed DEMAND noise offset protocol so model seed and SNR are excluded from
  segment selection.
- Added `noise_draw_id`, `noise_repeat`, and `offset_key_version`.
- Added clean-validation method-specific rejection thresholds for raw Softmax,
  prototype, hierarchical, and fused.
- Added provenance gates for checkpoint/prototype/calibration/test manifest/noise
  manifest/DEMAND provenance/selected noise file/channel/noise-vs-pig MD5.
- Removed silent channel disagreement by requiring optional CLI
  `--selected_channel` to match the manifest.
- Renamed/reported active-event SNR fields:
  `snr_reference=active_valid_region`, `target_active_snr_db`,
  `achieved_active_snr_db`, `achieved_full_window_snr_db`,
  `valid_duration_ratio`.
- Added per-class metrics, complete confusion matrices, AURC, risk at coverage
  0.80/0.90/0.95, and SNR statistics by true class.
- Hardened the aggregate summarizer so legal screening requires exactly
  5 folds x 3 seeds x 3 DEMAND environments x 3 active-event SNRs.
- Updated DEMAND license metadata to conflicting metadata fields and
  do-not-redistribute policy.

## Smoke Run

Command:

```powershell
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\eval_prototype_noise_robustness.py `
  --prototype_bundle reports\prototype_cv5_exact_w05_fold0_seed3407\artifacts\prototype_bundle.npz `
  --calibration_json reports\prototype_cv5_exact_w05_fold0_seed3407\calibration\calibration.json `
  --ckpt checkpoints\cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407.pt `
  --test_manifest paper_results\manifests\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\test.csv `
  --reference_pred_csv reports\cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407\test_pred.csv `
  --noise_manifest data\noise_sources\demand\NOISE_SOURCE_MANIFEST.csv `
  --noise_environments DWASHING TBUS STRAFFIC `
  --snr_db 20 10 0 `
  --out_dir reports\prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke `
  --device auto `
  --batch_size 32 `
  --global_noise_seed 3407 `
  --noise_repeat 0
```

Summary command:

```powershell
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\summarize_prototype_noise_robustness.py `
  --metrics_json reports\prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke\noise_metrics.json `
  --out_prefix reports\prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke\summary `
  --allow_smoke
```

## Acceptance Checks

- Clean equivalence: passed, `168/168` path/y_true/raw Softmax y_pred matches.
- Clean raw Softmax Macro-F1 reproduced: `0.946360153256705`.
- Provenance gates: all true.
- Same offset across 20/10/0 dB: true for `504` clean/environment/repeat draw
  groups.
- Same offset across model seed: covered by unit test; seed is accepted only as
  audit context and excluded from `demand_noise_offset_v2`.
- Same waveform/embedding per method: `same_waveform_embedding_reused=true`.
- 0 dB finite check: no NaN/Inf in Macro-F1 or AURC.
- Simulated noise: `true`.
- Real farm external validation: `false`.
- Paper main result: `false`.

## Provenance Gate Result

All gates were true:

- `checkpoint_sha_verified`
- `prototype_bundle_sha_verified`
- `calibration_sha_verified`
- `test_manifest_sha_verified`
- `noise_manifest_sha_verified`
- `noise_provenance_sha_verified`
- `selected_noise_file_sha_verified`
- `noise_vs_pig_md5_disjoint`
- `selected_channel_verified`

## Method-Specific Clean-Val Thresholds

- raw_softmax global `0.8629439473152161`; per class:
  cough `0.999169647693634`, calm_grunt `0.9971354603767395`,
  feeding `0.7203384041786194`, stress_vocal `0.7393773794174194`
- prototype global `0.82790607213974`; per class:
  cough `0.9999868273735046`, calm_grunt `0.9999257922172546`,
  feeding `0.7864745259284973`, stress_vocal `0.6714802980422974`
- hierarchical global `0.8532170057296753`; per class:
  cough `0.9999885559082031`, calm_grunt `0.9998205304145813`,
  feeding `0.7066670656204224`, stress_vocal `0.7599010467529297`

Threshold formula: `quantile(confidence, 1 - target_coverage)` on clean
validation predictions only. Target coverage is `0.95`.

## Full-Grid Smoke Macro-F1

| condition | raw | prototype | hierarchical |
|---|---:|---:|---:|
| clean | 0.946360 | 0.946360 | 0.952273 |
| DWASHING 20 | 0.656321 | 0.665814 | 0.661102 |
| DWASHING 10 | 0.604815 | 0.614662 | 0.620090 |
| DWASHING 0 | 0.594665 | 0.605114 | 0.600000 |
| TBUS 20 | 0.614045 | 0.614045 | 0.619464 |
| TBUS 10 | 0.599470 | 0.604409 | 0.604815 |
| TBUS 0 | 0.563761 | 0.576143 | 0.564466 |
| STRAFFIC 20 | 0.594616 | 0.604815 | 0.599940 |
| STRAFFIC 10 | 0.533141 | 0.553182 | 0.540000 |
| STRAFFIC 0 | 0.377213 | 0.363202 | 0.355094 |

## 0 dB Per-Class Notes

At 0 dB active-event SNR, cough F1 collapsed to `0.0` for all methods across
the three DEMAND environments. This is a warning sign for the full screening and
paper interpretation.

Examples:

- DWASHING 0 dB hierarchical: feeding F1 `0.800000`, stress F1 `0.600000`,
  feeding_to_stress `14`, stress_to_feeding `0`
- TBUS 0 dB hierarchical: feeding F1 `0.707692`, stress F1 `0.573427`,
  feeding_to_stress `19`, stress_to_feeding `0`
- STRAFFIC 0 dB hierarchical: feeding F1 `0.046512`, stress F1 `0.550336`,
  feeding_to_stress `36`, stress_to_feeding `0`

## Active/Full-Window SNR Example

DWASHING, true cough:

- 20 dB active-event SNR: active mean `20.000000`, full-window mean `12.920643`,
  valid-duration ratio mean `0.234185`
- 0 dB active-event SNR: active mean approximately `0.000000`, full-window mean
  `-7.079357`, valid-duration ratio mean `0.234185`

This confirms why the report must use active-event SNR terminology.

## Outputs

- `reports/prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke/noise_metrics.json`
- `reports/prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke/metrics_by_condition.csv`
- `reports/prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke/confusion_by_condition_method.json`
- `reports/prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke/snr_by_class.csv`
- `reports/prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke/provenance_gates.json`
- `reports/prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke/summary_summary.csv`
- `reports/prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke/summary_confusion_summary.csv`
- `reports/prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke/summary_degradation_slope.csv`
- `reports/prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke/summary_per_fold_deltas.csv`

## Verification

Ran:

- all new scripts `--help`
- `python -m py_compile`
- `python -m unittest tests.test_noise_robustness_pipeline -v`
- `python -m unittest tests.test_prototype_pipeline_core -v`
- `git diff --check`

See final Codex response for exact counts after commit.

## Recommendation

The protocol blockers are fixed enough to start the 15-run DEMAND screening
after review approval. Scientific caution: 0 dB active-event SNR is severe,
especially for cough, so the paper should present 0 dB as a stress test rather
than expected deployment performance.
