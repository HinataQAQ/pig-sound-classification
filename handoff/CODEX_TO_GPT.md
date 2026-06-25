# Codex to GPT Handoff - DEMAND Screening w0.5

## Status

Completed. The user provided `APPROVED: true` conditional on the pre-run
provenance patch. I implemented the patch, verified it, then ran the fixed
5 folds x 3 seeds DEMAND screening for lambda 0.5 only.

No lambda 1.0, noise-aware recalibration, noise augmentation training, local
unlicensed noise, or real-farm/open-set experiment was started.

## Branch

- Branch: `codex/demand-noise-screening-w05`
- Base branch: `codex/demand-noise-prescreen-fix`
- Commit SHA: pending at handoff write; see pushed branch HEAD/final Codex response

## Scope

- Model/checkpoints/prototypes/calibration: unchanged
- Feature backend: `training_exact`
- Lambda: `0.5`
- Folds: `0,1,2,3,4`
- Seeds: `42,2024,3407`
- DEMAND environments: `DWASHING`, `TBUS`, `STRAFFIC`
- Active-event SNR grid: `20,10,0` dB
- Global noise seed: `3407`
- Noise repeat: `0`
- Offset key version: `demand_noise_offset_v2`
- Run stage: `screening`
- Protocol: zero-shot frozen simulated noise
- Real-farm external validation: false

## Code Changes

- Added `--run_stage prescreen_smoke|screening` to
  `tools/eval_prototype_noise_robustness.py`.
- `prescreen_smoke` emits `run_scope=fold_seed_smoke`,
  `single_fold_debug=true`, `paper_main_result=false`.
- `screening` emits `run_scope=fold_seed`, `single_fold_debug=false`,
  `paper_main_result=false`.
- `noise_metrics.json` now records `run_stage`, `global_noise_seed`,
  `noise_repeat`, and `offset_key_version`.
- Renamed provenance gate from `noise_provenance_sha_verified` to
  `noise_provenance_policy_verified`, while retaining
  `noise_provenance_json_sha256`.
- Hardened `tools/summarize_prototype_noise_robustness.py`:
  exact per-run grid validation, strict paired `n=15`, fixed screening protocol,
  MODERATE_NOISE / EXTREME_STRESS / ALL_NOISY strata, fold-cluster paired
  statistics, and cross-seed noise-draw audit.
- Added unit tests for run stage fields, exact-grid failures, strict paired n,
  seed/repeat mismatch, cross-seed draw mismatch, strata grouping, and
  fold-cluster bootstrap reproducibility.

## Commands

Verification:

```powershell
$env:PYTHONNOUSERSITE="1"
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\audit_noise_assets.py --help
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\mix_audio_at_snr.py --help
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\eval_prototype_noise_robustness.py --help
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\summarize_prototype_noise_robustness.py --help
& C:\py\anaconda3\envs\pigsound-gpu\python.exe -m py_compile tools\audit_noise_assets.py tools\mix_audio_at_snr.py tools\eval_prototype_noise_robustness.py tools\summarize_prototype_noise_robustness.py tests\test_noise_robustness_pipeline.py
& C:\py\anaconda3\envs\pigsound-gpu\python.exe -m unittest tests.test_noise_robustness_pipeline -v
& C:\py\anaconda3\envs\pigsound-gpu\python.exe -m unittest tests.test_prototype_pipeline_core -v
git diff --check
```

Screening:

```powershell
$env:PYTHONNOUSERSITE="1"
$env:NUMBA_CACHE_DIR=(Join-Path (Get-Location) ".numba_cache")
foreach ($fold in 0,1,2,3,4) {
  foreach ($seed in 42,2024,3407) {
    & C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\eval_prototype_noise_robustness.py `
      --prototype_bundle reports\prototype_cv5_exact_w05_fold${fold}_seed${seed}\artifacts\prototype_bundle.npz `
      --calibration_json reports\prototype_cv5_exact_w05_fold${fold}_seed${seed}\calibration\calibration.json `
      --ckpt checkpoints\cv5_expanded_cap3x_fold${fold}_logmel_dur2_hier_w05_seed${seed}.pt `
      --test_manifest paper_results\manifests\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold${fold}\test.csv `
      --reference_pred_csv reports\cv5_expanded_cap3x_fold${fold}_logmel_dur2_hier_w05_seed${seed}\test_pred.csv `
      --noise_manifest data\noise_sources\demand\NOISE_SOURCE_MANIFEST.csv `
      --noise_environments DWASHING TBUS STRAFFIC `
      --snr_db 20 10 0 `
      --out_dir reports\prototype_noise_demand_w05_fold${fold}_seed${seed}_screening `
      --device auto `
      --batch_size 32 `
      --global_noise_seed 3407 `
      --noise_repeat 0 `
      --run_stage screening
  }
}
```

Aggregate:

```powershell
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\summarize_prototype_noise_robustness.py `
  --metrics_json reports\prototype_noise_demand_w05_fold0_seed42_screening\noise_metrics.json ... `
  --out_prefix reports\prototype_noise_demand_w05_screening
```

## Outputs

- Per-run directories:
  `reports/prototype_noise_demand_w05_fold{fold}_seed{seed}_screening/`
- Aggregate:
  - `reports/prototype_noise_demand_w05_screening_summary.csv`
  - `reports/prototype_noise_demand_w05_screening_paired_stats.csv`
  - `reports/prototype_noise_demand_w05_screening_confusion_summary.csv`
  - `reports/prototype_noise_demand_w05_screening_per_fold_deltas.csv`
  - `reports/prototype_noise_demand_w05_screening_fold_cluster_paired_stats.csv`
  - `reports/prototype_noise_demand_w05_screening_cross_seed_draw_audit.csv`
  - `reports/prototype_noise_demand_w05_screening_cross_seed_draw_audit.json`
  - `reports/prototype_noise_demand_w05_screening_provenance.json`

## Aggregate Provenance

- `run_scope=aggregate`
- `screening_result=true`
- `final_25_run_result=false`
- `paper_candidate_result=true`
- `paper_main_result=false`
- `n_fold_seed_runs=15`
- `simulated_noise=true`
- `real_farm_external_validation=false`
- `cross_seed_noise_draw_audit_ok=true`
- Cross-seed draw audit groups: `2520`, all OK

## Main Metrics

Mean/std Macro-F1:

| Stratum | raw_softmax | prototype | hierarchical | fused |
| --- | ---: | ---: | ---: | ---: |
| MODERATE_NOISE | 0.645442 / 0.027306 | 0.654077 / 0.024862 | 0.652557 / 0.027055 | 0.649658 / 0.025827 |
| EXTREME_STRESS | 0.556940 / 0.029653 | 0.565833 / 0.025803 | 0.557158 / 0.026776 | 0.561422 / 0.026679 |
| ALL_NOISY | 0.615941 / 0.025081 | 0.624662 / 0.022745 | 0.620757 / 0.023449 | 0.620246 / 0.022982 |

Paired statistics:

- MODERATE_NOISE:
  - prototype - raw: mean delta `0.008635`, p `0.002329`, wins/ties/losses `13/1/1`
  - hierarchical - raw: mean delta `0.007115`, p `0.012451`, wins/ties/losses `12/0/3`
  - hierarchical - prototype: mean delta `-0.001520`, p `0.151428`, wins/ties/losses `5/0/10`
- EXTREME_STRESS:
  - prototype - raw: mean delta `0.008893`, p `0.302795`, wins/ties/losses `7/0/8`
  - hierarchical - raw: mean delta `0.000218`, p `0.599487`, wins/ties/losses `6/0/9`
  - hierarchical - prototype: mean delta `-0.008675`, p `0.001526`, wins/ties/losses `2/0/13`
- ALL_NOISY:
  - prototype - raw: mean delta `0.008721`, p `0.002625`, wins/ties/losses `13/0/2`
  - hierarchical - raw: mean delta `0.004816`, p `0.229309`, wins/ties/losses `9/0/6`
  - hierarchical - prototype: mean delta `-0.003905`, p `0.000122`, wins/ties/losses `1/0/14`

Fold-level ALL_NOISY deltas:

- fold0: prototype-raw `0.008267`, hierarchical-raw `0.004322`, hierarchical-prototype `-0.003945`
- fold1: prototype-raw `0.031723`, hierarchical-raw `0.025341`, hierarchical-prototype `-0.006382`
- fold2: prototype-raw `0.002096`, hierarchical-raw `-0.000492`, hierarchical-prototype `-0.002588`
- fold3: prototype-raw `0.002265`, hierarchical-raw `0.001145`, hierarchical-prototype `-0.001120`
- fold4: prototype-raw `-0.000745`, hierarchical-raw `-0.006237`, hierarchical-prototype `-0.005491`

ALL_NOISY class F1 from aggregate confusion:

- raw_softmax: cough `0.087690`, calm `0.992294`, feeding `0.825939`, stress `0.598141`
- prototype: cough `0.098592`, calm `0.985739`, feeding `0.843309`, stress `0.606508`
- hierarchical: cough `0.096357`, calm `0.987542`, feeding `0.831763`, stress `0.605886`

ALL_NOISY errors:

- raw_softmax: feeding->stress `1381`, stress->feeding `350`, cough->stress `5390`, cough->calm `20`, cough->feeding `0`
- prototype: feeding->stress `1211`, stress->feeding `350`, cough->stress `5301`, cough->calm `75`, cough->feeding `0`
- hierarchical: feeding->stress `1371`, stress->feeding `281`, cough->stress `5320`, cough->calm `63`, cough->feeding `0`

AURC / frozen threshold:

- ALL_NOISY raw: AURC `0.179827`, frozen coverage/risk `0.917196 / 0.304687`
- ALL_NOISY prototype: AURC `0.208029`, frozen coverage/risk `0.921869 / 0.299186`
- ALL_NOISY hierarchical: AURC `0.210261`, frozen coverage/risk `0.917901 / 0.304017`

## Data Boundaries

- Train only: original backbone and prototype construction, already completed in exact prototype artifacts.
- Validation only: clean calibration and frozen thresholds, reused unchanged.
- Test only: clean equivalence and simulated-noise final evaluation.
- No noisy validation, no noisy recalibration, no test tuning.
- Leakage audit and SHA gates passed per run.

## Interpretation

The DEMAND screening is paper-candidate simulated-noise evidence, not a paper
main final result and not real-farm external validation. Prototype-only is more
robust than raw Softmax on ALL_NOISY and MODERATE_NOISE. Hierarchical prototype
does not outperform prototype under simulated noise and is worse than prototype
on ALL_NOISY and EXTREME_STRESS.

0 dB is explicitly an extreme simulated-noise stress condition. Cough remains
the weakest class under DEMAND noise, with most cough errors going to
`stress_vocal`.

## Failures / Warnings

- No run failed.
- The model emitted existing pandas fragmentation warnings from
  `prototype_model_adapter.py`; they did not change results or exit status.
- No local unlicensed noise, lambda 1.0, noise-aware training, or real-farm
  validation was run.

## GPT Pro Questions

1. Is the ALL_NOISY prototype gain over raw Softmax sufficient for an SCI 4
   robustness section, given the cough weakness?
2. Should the paper report hierarchical prototype as secondary/diagnostic under
   noise, since prototype-only is stronger in this screening?
3. Should 0 dB be kept in the main table as severe stress testing or moved to an
   appendix?

## Suggested Next Step

Do not start another experiment automatically. Recommended next decision:
whether to prepare paper-facing DEMAND screening tables/figures or first run a
targeted analysis of cough failure modes under simulated noise.
