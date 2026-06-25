# Codex to GPT Handoff - DEMAND Final w0.5

## Status

Completed. The user provided `APPROVED: true`; `handoff/GPT_TO_CODEX.md` was
updated from `APPROVED: false` to `APPROVED: true` before execution.

I created `codex/demand-noise-final-w05` from
`codex/demand-noise-screening-w05` at
`45af0dc4ecb40182461f923d5206baead580af1d`, added `run_stage=final`, ran only
the missing seeds `123` and `777` for all five folds, then aggregated the
existing 15 screening runs plus 10 final runs into the fixed 25-run simulated
DEMAND final result.

No lambda 1.0, noise-aware recalibration, noise augmentation training, denoising
training, local reviewed noise, open-set experiment, new noise environment, new
SNR, or new noise repeat was started.

## Branch

- Branch: `codex/demand-noise-final-w05`
- Base branch/commit: `codex/demand-noise-screening-w05` /
  `45af0dc4ecb40182461f923d5206baead580af1d`
- Commit SHA: pending at handoff write; see pushed branch HEAD/final Codex response

## Scope

- Lambda: `0.5`
- Folds: `0,1,2,3,4`
- Final seeds: `42,123,777,2024,3407`
- Newly run seeds: `123,777`
- Existing reused seeds: `42,2024,3407`
- DEMAND environments: `DWASHING`, `TBUS`, `STRAFFIC`
- Active-event SNR grid: `20,10,0` dB
- Global noise seed: `3407`
- Noise repeat: `0`
- Selected channel: `1`
- Feature backend: `training_exact`
- Noise protocol: `zero_shot_frozen`
- Run stage for new runs: `final`
- Real-farm external validation: false

## Code Changes

- `tools/eval_prototype_noise_robustness.py`
  - Added `run_stage=final`.
  - Single final fold-seed runs remain `run_scope=fold_seed`,
    `single_fold_debug=false`, and `paper_main_result=false`.
  - Added `generalized_risk_coverage_auc`; current AUGRC uses 0/1 loss, so it
    is equivalent to AURC for these classification reports.
- `tools/summarize_prototype_noise_robustness.py`
  - Added fixed `screening` and `final` protocols.
  - `screening`: exactly 5 folds x 3 seeds, `paper_main_result=false`.
  - `final`: exactly 5 folds x 5 seeds, `paper_main_result=true`.
  - Paired statistics now use strict protocol-specific paired n.
  - Cross-seed draw audit accepts the protocol-specific seed set.
  - Added directory output mode for `final_*.csv/json` files.
  - Added final per-class, selective, AURC/AUGRC, per-fold, cluster bootstrap,
    cross-seed audit, and provenance outputs.
  - Missing historical `augrc` rows are backfilled from `aurc` row by row,
    consistent with the current 0/1 generalized loss definition.
- `tests/test_noise_robustness_pipeline.py`
  - Added final-stage qualification, fixed 25-run protocol, final paired n,
    five-seed cross-seed audit, AUGRC, and mixed historical/new AUGRC backfill
    tests.
- `docs/NOISE_ROBUSTNESS_PROTOCOL.md`
  - Documented final 5 folds x 5 seeds protocol, single-run qualification, and
    AUGRC definition.

## Commands

Verification and preflight:

```powershell
$env:PYTHONNOUSERSITE="1"
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\audit_noise_assets.py --help
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\eval_prototype_noise_robustness.py --help
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\summarize_prototype_noise_robustness.py --help
& C:\py\anaconda3\envs\pigsound-gpu\python.exe -m py_compile tools\eval_prototype_noise_robustness.py tools\summarize_prototype_noise_robustness.py tests\test_noise_robustness_pipeline.py
& C:\py\anaconda3\envs\pigsound-gpu\python.exe -m unittest tests.test_noise_robustness_pipeline -v
& C:\py\anaconda3\envs\pigsound-gpu\python.exe -m unittest tests.test_prototype_pipeline_core -v
git diff --check
```

Final runs:

```powershell
$env:PYTHONNOUSERSITE="1"
$env:NUMBA_CACHE_DIR=(Resolve-Path ".numba_cache").Path
foreach ($fold in 0,1,2,3,4) {
  foreach ($seed in 123,777) {
    & C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\eval_prototype_noise_robustness.py `
      --prototype_bundle reports\prototype_cv5_exact_w05_fold${fold}_seed${seed}\artifacts\prototype_bundle.npz `
      --calibration_json reports\prototype_cv5_exact_w05_fold${fold}_seed${seed}\calibration\calibration.json `
      --ckpt checkpoints\cv5_expanded_cap3x_fold${fold}_logmel_dur2_hier_w05_seed${seed}.pt `
      --test_manifest data\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold${fold}\test.csv `
      --reference_pred_csv reports\cv5_expanded_cap3x_fold${fold}_logmel_dur2_hier_w05_seed${seed}\test_pred.csv `
      --noise_manifest data\noise_sources\demand\NOISE_SOURCE_MANIFEST.csv `
      --noise_provenance_json data\noise_sources\demand\PROVENANCE.json `
      --clean_prediction_metadata reports\prototype_cv5_exact_w05_fold${fold}_seed${seed}\evaluation\prediction_metadata.json `
      --calibration_val_predictions reports\prototype_cv5_exact_w05_fold${fold}_seed${seed}\calibration\val_predictions.csv `
      --noise_environments DWASHING TBUS STRAFFIC `
      --snr_db 20 10 0 `
      --out_dir reports\prototype_noise_demand_w05_fold${fold}_seed${seed}_final `
      --device auto `
      --batch_size 32 `
      --global_noise_seed 3407 `
      --noise_repeat 0 `
      --run_stage final `
      --selected_channel 1
  }
}
```

Final aggregate:

```powershell
& C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\summarize_prototype_noise_robustness.py `
  --protocol final `
  --metrics_json <15 screening noise_metrics.json + 10 final noise_metrics.json> `
  --out_dir reports\prototype_noise_demand_w05_final `
  --file_prefix final
```

## Outputs

New final fold-seed directories:

- `reports/prototype_noise_demand_w05_fold{fold}_seed123_final/`
- `reports/prototype_noise_demand_w05_fold{fold}_seed777_final/`

Aggregate directory:

- `reports/prototype_noise_demand_w05_final/final_summary.csv`
- `reports/prototype_noise_demand_w05_final/final_paired_stats.csv`
- `reports/prototype_noise_demand_w05_final/final_confusion_summary.csv`
- `reports/prototype_noise_demand_w05_final/final_per_class_summary.csv`
- `reports/prototype_noise_demand_w05_final/final_selective_summary.csv`
- `reports/prototype_noise_demand_w05_final/final_aurc_augrc_summary.csv`
- `reports/prototype_noise_demand_w05_final/final_per_fold_deltas.csv`
- `reports/prototype_noise_demand_w05_final/final_cluster_bootstrap.csv`
- `reports/prototype_noise_demand_w05_final/final_cross_seed_draw_audit.csv`
- `reports/prototype_noise_demand_w05_final/final_provenance.json`

## Aggregate Provenance

- `run_scope=aggregate`
- `screening_result=false`
- `final_25_run_result=true`
- `paper_candidate_result=true`
- `paper_main_result=true`
- `simulated_noise_main_result=true`
- `n_fold_seed_runs=25`
- `simulated_noise=true`
- `real_farm_external_validation=false`
- `cross_seed_noise_draw_audit_ok=true`
- Cross-seed draw audit groups: `2520`, all OK
- Cross-seed set: `42|123|777|2024|3407`

## Main Metrics

Mean/std Macro-F1:

| Stratum | raw_softmax | prototype | hierarchical | fused |
| --- | ---: | ---: | ---: | ---: |
| MODERATE_NOISE | 0.647225 / 0.025426 | 0.653354 / 0.023320 | 0.652175 / 0.025686 | 0.649915 / 0.024672 |
| EXTREME_STRESS | 0.557393 / 0.029311 | 0.562400 / 0.028306 | 0.555623 / 0.028425 | 0.560369 / 0.027277 |
| ALL_NOISY | 0.617281 / 0.022973 | 0.623036 / 0.021252 | 0.619991 / 0.022378 | 0.620066 / 0.021477 |

ALL_NOISY paired statistics:

- prototype - raw: mean delta `0.005755`, CI95 `[0.001606, 0.010894]`,
  Wilcoxon p `0.010511`, wins/ties/losses `18/0/7`.
- hierarchical - raw: mean delta `0.002710`, CI95 `[-0.000999, 0.007085]`,
  Wilcoxon p `0.312333`, wins/ties/losses `13/0/12`.
- hierarchical - prototype: mean delta `-0.003045`, CI95
  `[-0.004581, -0.001639]`, Wilcoxon p `0.000376`,
  wins/ties/losses `5/0/20`.

Fold-cluster ALL_NOISY:

- prototype - raw: mean delta `0.005755`, fold CI95 `[-0.000534, 0.014622]`,
  fold wins/ties/losses `4/0/1`.
- hierarchical - raw: mean delta `0.002710`, fold CI95
  `[-0.003510, 0.010632]`, fold wins/ties/losses `3/0/2`.
- hierarchical - prototype: mean delta `-0.003045`, fold CI95
  `[-0.004607, -0.001483]`, fold wins/ties/losses `0/0/5`.

ALL_NOISY fold-level deltas:

- fold0: prototype-raw `0.005685`, hierarchical-raw `0.002786`,
  hierarchical-prototype `-0.002899`.
- fold1: prototype-raw `0.023345`, hierarchical-raw `0.017687`,
  hierarchical-prototype `-0.005658`.
- fold2: prototype-raw `0.000196`, hierarchical-raw `-0.000624`,
  hierarchical-prototype `-0.000821`.
- fold3: prototype-raw `0.002159`, hierarchical-raw `0.000722`,
  hierarchical-prototype `-0.001437`.
- fold4: prototype-raw `-0.002610`, hierarchical-raw `-0.007020`,
  hierarchical-prototype `-0.004410`.

ALL_NOISY class F1 from aggregate confusion:

- raw: cough `0.086657`, calm `0.988748`, feeding `0.832270`,
  stress `0.600279`.
- prototype: cough `0.097424`, calm `0.979829`, feeding `0.841560`,
  stress `0.607251`.
- hierarchical: cough `0.095699`, calm `0.981712`, feeding `0.831944`,
  stress `0.607110`.
- fused: cough `0.088961`, calm `0.987355`, feeding `0.838345`,
  stress `0.602811`.

ALL_NOISY errors:

- raw: feeding->stress `2168`, stress->feeding `616`, cough->calm `86`,
  cough->feeding `1`, cough->stress `8935`.
- prototype: feeding->stress `2006`, stress->feeding `606`,
  cough->calm `217`, cough->feeding `1`, cough->stress `8748`.
- hierarchical: feeding->stress `2240`, stress->feeding `495`,
  cough->calm `194`, cough->feeding `1`, cough->stress `8780`.
- fused: feeding->stress `2075`, stress->feeding `613`, cough->calm `107`,
  cough->feeding `1`, cough->stress `8902`.

ALL_NOISY AURC/AUGRC and risk:

- raw: AURC/AUGRC `0.177458/0.177458`, risk@80/90/95
  `0.304296/0.304035/0.308000`, frozen coverage/risk
  `0.920000/0.302386`.
- prototype: AURC/AUGRC `0.210937/0.210937`, risk@80/90/95
  `0.305679/0.299737/0.303611`, frozen coverage/risk
  `0.921190/0.298765`.
- hierarchical: AURC/AUGRC `0.212212/0.212212`, risk@80/90/95
  `0.309465/0.303626/0.307000`, frozen coverage/risk
  `0.921852/0.303299`.
- fused: AURC/AUGRC `0.184808/0.184808`, risk@80/90/95
  `0.304395/0.302193/0.305556`, frozen coverage/risk
  `0.919418/0.301143`.

Calibration distribution across 25 clean runs:

- calibrated_softmax_temperature: min `0.5`, max `1.0`, mean `0.8`
- prototype_temperature: min `0.03`, max `0.1`, mean `0.0696`
- hierarchical_temperature: min `0.03`, max `0.2`, mean `0.0696`
- hier_aux_prob_weight: min/max/mean `0.5`
- fusion_alpha: min `0.0`, max `1.0`, mean `0.77`
- fusion_kind counts: pure_softmax `14`, mixed `8`, pure_prototype `3`

## Data Boundaries

- Train only: original clean backbone/prototypes, reused unchanged.
- Validation only: clean calibration and frozen thresholds, reused unchanged.
- Test only: clean equivalence and simulated DEMAND final evaluation.
- No noisy validation, no noisy recalibration, no test tuning.
- 25/25 runs passed `eligible_for_noise_aggregation`, clean equivalence, and
  leakage audit gates.

## Interpretation

Prototype-only is the strongest method under this fixed simulated DEMAND final
protocol. Hierarchical prototype is not better than prototype-only under noise;
on ALL_NOISY it is slightly above raw Softmax but nonsignificant by the paired
run test and below prototype.

Cough remains the weakest class. Most cough errors go to `stress_vocal`.

This is a simulated-noise paper-main candidate aggregate, not real-farm external
validation.

## Failures / Warnings

- No run failed.
- Existing pandas DataFrame fragmentation warnings from
  `prototype_model_adapter.py` appeared during prediction construction; they did
  not affect exit status or recorded metrics.
- `git diff --check` passed with line-ending warnings only.

## GPT Pro Questions

1. Should the DEMAND final paper table make prototype-only the primary
   robustness finding, with hierarchical prototype as secondary?
2. Should cough failure analysis be added before writing the SCI 4 submission?
3. Should the paper phrase the result as simulated-noise robustness only, with
   real-farm external validation reserved for future work?

## Suggested Next Step

Stop. Recommended next decision: paper-facing DEMAND final tables/figures versus
a targeted cough failure-mode analysis.
