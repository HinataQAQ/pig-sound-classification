# Codex to GPT Handoff - DEMAND Noise Robustness Debug

## Status

Completed the approved DEMAND noise data preparation, noise robustness code, and
one fold x one seed debug run.

Latest explicit approval came from the user attachment with `APPROVED: true`.
`handoff/GPT_TO_CODEX.md` still contained a stale `APPROVED: false`; this should
be reconciled before starting any next stage.

## Branch

- Branch: `codex/demand-noise-debug`
- Base commit before this work: `3f4693975ed22a2dec8eb343c3207b2c31269f01`
- Final commit SHA: pending at handoff write; see pushed branch HEAD/final Codex
  response

## Scope Completed

- DEMAND only as the public simulated noise source.
- No MUSAN, ESC-50, or UrbanSound8K download.
- No local reviewed farm noise promoted to paper-facing use.
- No audio files, ZIP files, DEMAND PDF, cache files, checkpoints, NPZ bundles,
  or embeddings committed.
- Debug only: fold `0`, seed `3407`, lambda `0.5`, DEMAND `DWASHING`, SNR
  `20` and `10` dB.
- No 0 dB run.
- No 5 folds x 3 seeds screening run.
- No lambda 1.0 run.
- No model, clean checkpoint, clean prototype, or clean calibration update.

## Selected DEMAND Noise Sources

Selected environment families for the planned protocol:

- `DWASHING`: steady indoor/mechanical, domestic washing machine.
- `TBUS`: engine/transport machinery, public transit bus.
- `STRAFFIC`: mixed environmental, busy traffic intersection.

Debug run used only `DWASHING`.

Channel policy:

- Fixed selected channel: `ch01.wav`
- Manifest field: `selected_channel=1`
- DEMAND synchronized channels are not treated as independent recordings.
- Recording identity collapses to `DEMAND:<environment>`.

License/provenance:

- DOI: `10.5281/zenodo.1227121`
- Zenodo record: `https://zenodo.org/records/1227121`
- Zenodo metadata observed: `cc-by-4.0`
- DEMAND PDF text observed: Creative Commons Attribution-ShareAlike 3.0
  Unported
- Both observations are recorded in `data/noise_sources/demand/PROVENANCE.json`.

## Commands

Environment setup:

```powershell
conda activate pigsound-gpu
cd C:\py\pigsound\pig-sound-classification
$env:PYTHONNOUSERSITE="1"
$env:NUMBA_CACHE_DIR=(Join-Path (Get-Location) ".numba_cache")
```

DEMAND provenance/materialization:

```powershell
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\audit_noise_assets.py --prepare_demand --out_dir data\noise_sources\demand
```

Debug evaluation:

```powershell
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\eval_prototype_noise_robustness.py `
  --prototype_bundle reports\prototype_cv5_exact_w05_fold0_seed3407\artifacts\prototype_bundle.npz `
  --calibration_json reports\prototype_cv5_exact_w05_fold0_seed3407\calibration\calibration.json `
  --ckpt checkpoints\cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407.pt `
  --test_manifest paper_results\manifests\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\test.csv `
  --reference_pred_csv reports\cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407\test_pred.csv `
  --noise_manifest data\noise_sources\demand\NOISE_SOURCE_MANIFEST.csv `
  --noise_environments DWASHING `
  --snr_db 20 10 `
  --out_dir reports\prototype_noise_demand_w05_fold0_seed3407_debug `
  --device auto `
  --batch_size 32 `
  --global_noise_seed 3407 `
  --allow_overwrite
```

Summary:

```powershell
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\summarize_prototype_noise_robustness.py `
  --metrics_json reports\prototype_noise_demand_w05_fold0_seed3407_debug\noise_metrics.json `
  --out_prefix reports\prototype_noise_demand_w05_fold0_seed3407_debug\summary
```

## Clean Equivalence Gate

- Passed: yes
- Sample count: `168`
- Path alignment: `168/168`
- `y_true` match: `168/168`
- raw Softmax `y_pred` match: `168/168`
- Reference Macro-F1: `0.946360153256705`
- Reproduced Macro-F1: `0.946360153256705`
- Absolute difference: `0.0`
- Tolerance: `1e-6`

## Debug Metrics

| condition | method | Macro-F1 | Top-1 acc | Top-2 acc | ECE |
|---|---|---:|---:|---:|---:|
| clean | raw_softmax | 0.946360 | 0.946429 | 1.000000 | 0.028922 |
| clean | prototype | 0.946360 | 0.946429 | 1.000000 | 0.035536 |
| clean | hierarchical | 0.952273 | 0.952381 | 1.000000 | 0.034155 |
| DWASHING_20dB | raw_softmax | 0.660317 | 0.714286 | 0.833333 | 0.246911 |
| DWASHING_20dB | prototype | 0.693259 | 0.732143 | 0.797619 | 0.227496 |
| DWASHING_20dB | hierarchical | 0.688191 | 0.732143 | 0.803571 | 0.234583 |
| DWASHING_10dB | raw_softmax | 0.599470 | 0.666667 | 0.809524 | 0.289846 |
| DWASHING_10dB | prototype | 0.618750 | 0.690476 | 0.755952 | 0.266430 |
| DWASHING_10dB | hierarchical | 0.619464 | 0.690476 | 0.755952 | 0.264579 |

Fused results equal raw Softmax in this debug because the clean calibration's
selected fusion setting is pure Softmax for this fold-seed.

## Achieved SNR Audit

- Max absolute active-region SNR error: `4.6514175444656303e-07` dB
- Mean active-region SNR error: `2.567331231204778e-09` dB
- DWASHING 20 dB: active error mean `-1.045848e-08`, max abs
  `4.561202e-07`
- DWASHING 10 dB: active error mean `1.559315e-08`, max abs
  `4.651418e-07`

Full-window SNR differs by design because clean RMS is measured over the
non-padding valid region while noise is added over the full 2-second waveform.

## Leakage and Qualification

- train/val/test path/source_id/MD5 disjointness: passed
- Leakage audit counts: train `1174`, val `120`, test `168`
- Manifest SHA verification: passed
- Feature backend: `training_exact`
- Feature pipeline equivalent: `true`
- Input role: `frozen_test`
- Run scope: `fold_seed`
- Single-fold debug: `true`
- Eligible for later aggregation: `true`
- Paper candidate result: `false`
- Paper main result: `false`
- Simulated noise: `true`
- Noise protocol: `zero_shot_frozen`
- Real farm external validation: `false`
- Test parameter selection: `false`

## Outputs

Noise source metadata:

- `data/noise_sources/demand/NOISE_SOURCE_MANIFEST.csv`
- `data/noise_sources/demand/PROVENANCE.json`
- `data/noise_sources/local_reviewed/PROVENANCE_TEMPLATE.csv`

Debug structured outputs:

- `reports/prototype_noise_demand_w05_fold0_seed3407_debug/clean_equivalence.json`
- `reports/prototype_noise_demand_w05_fold0_seed3407_debug/leakage_audit.json`
- `reports/prototype_noise_demand_w05_fold0_seed3407_debug/metrics_by_condition.csv`
- `reports/prototype_noise_demand_w05_fold0_seed3407_debug/noise_metrics.json`
- `reports/prototype_noise_demand_w05_fold0_seed3407_debug/noise_predictions.csv`
- `reports/prototype_noise_demand_w05_fold0_seed3407_debug/noise_sample_provenance.csv`
- `reports/prototype_noise_demand_w05_fold0_seed3407_debug/summary_summary.csv`
- `reports/prototype_noise_demand_w05_fold0_seed3407_debug/summary_provenance.csv`

Docs:

- `docs/NOISE_SOURCE_DECISION.md`
- `docs/NOISE_ROBUSTNESS_PROTOCOL.md`

## Verification

Ran:

- all new script `--help` commands
- `python -m py_compile` for new scripts and test
- `python -m unittest tests.test_noise_robustness_pipeline -v`
- `python -m unittest tests.test_prototype_pipeline_core -v`
- `git diff --check`

See final Codex response for exact pass counts after commit.

## Paper Usability

This debug run is usable as pipeline/debug evidence only. It is not a paper main
result and not external farm validation. The result suggests a meaningful noise
stress test is now technically ready for 5 folds x 3 seeds screening, but the
screening should only start after explicit review approval.

## Blockers

None for the 1x1 DEMAND debug. Remaining judgment items:

1. Confirm whether DEMAND license wording in the paper should cite both Zenodo
   metadata and DEMAND PDF language.
2. Confirm whether to run the planned 5 folds x 3 seeds screening over
   `DWASHING`, `TBUS`, and `STRAFFIC` at clean/20/10/0 dB.
3. Confirm whether local reviewed farm noises should remain internal until
   provenance fields are complete.
