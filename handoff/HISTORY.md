# Handoff History

## 2026-06-23 - Handoff protocol initialized

- Status: completed
- Task: Created Codex/GPT handoff templates and added the GPT Pro handoff protocol to `AGENTS.md`.
- Research outputs: none; documentation/setup only.
- Metrics: not applicable.
- Leakage audit: not applicable; no data, manifests, checkpoints, or predictions were modified.

## 2026-06-23 - Exact prototype final review fixes

- Status: completed
- Branch: `prototype-exact-review`
- Task: Fixed final PR review blockers for manifest content SHA enforcement,
  prediction input roles, result qualification fields, zero-valued calibration
  parameters, path normalization, and leakage audit errors.
- Metrics: fold0/seed3407 exact Softmax Macro-F1
  `0.946360153256705`; hierarchical prototype Macro-F1
  `0.9522727272727273`.
- Leakage audit: train/val/test path, source_id, and md5 disjointness remained
  valid; validation and frozen test manifest SHA mismatch negative tests passed.
- Paper usability: exact outputs are `feature_pipeline_equivalent=true` but
  `single_fold_debug=true`, so `paper_main_result=false`.

## 2026-06-23 - Exact prototype provenance qualification fix

- Status: completed
- Branch: `prototype-exact-review`
- Task: Restricted paper-main provenance to aggregate summaries only, required
  prediction metadata during evaluation, added prediction/calibration SHA
  verification, and added the exact CV aggregate gate script.
- Metrics: fold0/seed3407 exact metrics unchanged; hierarchical prototype
  Macro-F1 remained `0.9522727272727273`.
- Tests: 37 core unit tests passed, including metadata-missing, inference-role,
  edited prediction CSV SHA mismatch, fold1/seed42 non-paper-main, and aggregate
  paper-main gate tests.
- Scope: no full CV and no lambda=1.0 run was started.

## 2026-06-23 - Exact prototype aggregate summary protocol fix

- Branch: `prototype-exact-review`
- Task: Tightened `tools/summarize_cv5_exact_prototype.py` so aggregate paper
  provenance is allowed only for fixed 5 folds x 3 seeds screening and 5 folds
  x 5 seeds final protocols.
- Result qualification: 5x3 screening now sets `paper_candidate_result=true`
  and `paper_main_result=false`; 5x5 final sets both
  `paper_candidate_result=true` and `paper_main_result=true`.
- Statistics: aggregate summary now emits method means/std/min/max,
  fold-seed-paired deltas with bootstrap CI and Wilcoxon p-value, and aggregated
  confusion summaries for raw Softmax, prototype, and hierarchical methods.
- Tests: 47 core unit tests passed, including 1x1, 4x5, duplicate,
  unexpected-fold, mixed-lambda, paired-statistics, fixed-bootstrap, and
  confusion-aggregation cases.
- Scope: no full CV, no fold/seed expansion, and no model, feature,
  calibration, prediction, or evaluation core logic changes.

## 2026-06-24 - Lambda 0.5 exact prototype 5x3 screening

- Status: completed
- Branch: `prototype-cv5-screening-w05`
- Base integration: PR #1 was squash-merged into `current-mctafd-ablation`
  with merge commit `9ee233c04ef377a6acbaeaf233da7137f2add2d4`.
- Task: Ran the approved exact hierarchical acoustic prototype screening
  experiment for folds `0,1,2,3,4` and seeds `42,2024,3407`.
- Preflight: 15 checkpoints, 15 summaries, 15 reference `test_pred.csv` files,
  and all fold train/val/test manifests were present. Summaries passed
  `hier_aux_weight=0.5`, `dur_s=2.0`, `feature_mode=logmel`, seed, and label
  order checks. Checkpoints were loadable state_dict files.
- Screening status: 15/15 fold-seed runs completed. Each run performed exact
  Softmax reproduction, train-only prototype build, validation-only calibration,
  frozen test prediction, evaluation, leakage audit, and qualification gate.
- Aggregate provenance: `run_scope=aggregate`, `screening_result=true`,
  `final_25_run_result=false`, `paper_candidate_result=true`,
  `paper_main_result=false`, `n_runs=15`.
- Method Macro-F1 mean/std: raw Softmax `0.950244/0.013218`, calibrated
  Softmax `0.950244/0.013218`, prototype `0.952333/0.012754`,
  hierarchical `0.955499/0.010793`, fused `0.954289/0.009766`.
- Paired statistics: hierarchical - raw Softmax mean delta `0.005256`,
  bootstrap CI95 `[0.000026, 0.012109]`, Wilcoxon p `0.055808`,
  wins/ties/losses `9/4/2`; hierarchical - prototype mean delta `0.003166`,
  CI95 `[-0.000003, 0.006741]`, Wilcoxon p `0.176296`,
  wins/ties/losses `5/8/2`.
- Feeding/stress confusion: hierarchical reduced stress-to-feeding errors from
  `55` under raw Softmax to `43`; feeding-to-stress was `69` versus raw
  Softmax `70`.
- Leakage and qualification: all 15 runs used `training_exact`, `frozen_test`,
  manifest SHA verification, leakage audit ok, and
  `eligible_for_cv_aggregation=true`; no unsafe checkpoint SHA mismatch was
  used.
- Paper usability: usable as screening evidence and a paper candidate aggregate,
  not as final paper-main result. Do not claim statistical significance because
  Wilcoxon p is `0.055808`.
- Scope guard: no lambda 1.0 comparison, no 25-run final CV, and no
  unknown/open-set experiment was started.


## 2026-06-25 - Lambda 0.5 exact prototype 5x5 final aggregate

- Status: completed
- Branch: `prototype-cv5-final-w05`
- Task: Added the approved missing exact prototype runs for seeds `123` and `777` across folds `0,1,2,3,4`, then aggregated all 25 lambda 0.5 fold-seed runs.
- Scope: `feature_backend=training_exact`, `selection_method=hierarchical`, `target_coverage=0.95`; no lambda 1.0, noise robustness, unknown/open-set, or paper writing was started.
- Final aggregate provenance: `run_scope=aggregate`, `screening_result=false`, `final_25_run_result=true`, `paper_candidate_result=true`, `paper_main_result=true`, `n_runs=25`.
- Method Macro-F1 mean/std: raw Softmax `0.951050/0.012393`, calibrated Softmax `0.951050/0.012393`, prototype `0.951852/0.013104`, hierarchical `0.953986/0.012004`, fused `0.953486/0.010343`.
- Paired statistics: hierarchical - raw Softmax mean delta `0.002937`, CI95 `[-0.000478, 0.007294]`, Wilcoxon p `0.058253`, wins/ties/losses `13/8/4`; hierarchical - prototype mean delta `0.002134`, CI95 `[-0.000499, 0.004996]`, Wilcoxon p `0.221330`, wins/ties/losses `8/12/5`.
- Fold deltas: four of five folds had positive mean hierarchical - raw Softmax; fold0 was slightly negative.
- Feeding/stress confusion: hierarchical feeding->stress `117`, stress->feeding `76`; raw Softmax feeding->stress `110`, stress->feeding `95`.
- Leakage and qualification: 25/25 runs passed exact Softmax reproduction, reference path/y_true/y_pred matching, train/val/test path/source_id/MD5 disjointness, manifest SHA verification, leakage audit, and CV aggregation eligibility. No unsafe checkpoint SHA mismatch was used.
- Outputs: final aggregate CSV/provenance files plus by-fold, calibration-distribution, and selective-risk summaries under `reports/prototype_cv5_exact_w05_final_*`.
- Paper usability: final paper-main candidate aggregate, but the hierarchical gain over raw Softmax remains nonsignificant; do not claim significance.

## 2026-06-25 - DEMAND simulated noise robustness 1x1 debug

- Status: completed
- Branch: `codex/demand-noise-debug`
- Task: Prepared DEMAND noise provenance, implemented deterministic SNR mixing and
  frozen exact prototype noise robustness evaluation, then ran the approved
  fold0/seed3407 lambda 0.5 debug on `DWASHING` at 20 dB and 10 dB.
- Scope: DEMAND only; no MUSAN, ESC-50, UrbanSound8K, lambda 1.0, 0 dB,
  15-run screening, or real-farm external validation was started.
- Noise source decision: selected `DWASHING`, `TBUS`, and `STRAFFIC` for the
  planned protocol; debug used only `DWASHING`. Fixed channel policy is
  `ch01.wav` / `selected_channel=1`, and synchronized DEMAND channels are not
  treated as independent recordings.
- Provenance: DOI `10.5281/zenodo.1227121`; Zenodo metadata observed
  `cc-by-4.0`; DEMAND PDF text observed Creative Commons
  Attribution-ShareAlike 3.0 Unported. Both are recorded.
- Clean equivalence: passed with 168/168 path matches, 168/168 `y_true` matches,
  168/168 raw Softmax `y_pred` matches, and Macro-F1
  `0.946360153256705` exactly reproduced.
- Debug Macro-F1: clean raw/prototype/hierarchical
  `0.946360/0.946360/0.952273`; DWASHING 20 dB
  `0.660317/0.693259/0.688191`; DWASHING 10 dB
  `0.599470/0.618750/0.619464`.
- SNR audit: max active-region SNR absolute error
  `4.6514175444656303e-07` dB; full-window SNR differs by design because clean
  RMS uses the valid non-padding region.
- Leakage and qualification: train/val/test path, source_id, and md5
  disjointness passed; `feature_backend=training_exact`,
  `noise_protocol=zero_shot_frozen`, `real_farm_external_validation=false`,
  `run_scope=fold_seed`, `paper_main_result=false`.
- Outputs: DEMAND manifest/provenance, local reviewed provenance template,
  two noise protocol docs, new noise robustness scripts/tests, and structured
  debug CSV/JSON under `reports/prototype_noise_demand_w05_fold0_seed3407_debug`.
  Downloaded ZIP/WAV/PDF/cache/audio assets were not committed.
- Paper usability: pipeline/debug evidence only. Recommend 5 folds x 3 seeds
  DEMAND screening after explicit review approval.

## 2026-06-25 - DEMAND prescreen protocol fixes and 1x1 full-grid smoke

- Status: completed
- Branch: `codex/demand-noise-prescreen-fix`
- Approval state: full 15-run screening remained `APPROVED: false`; no 15-run
  screening was started.
- Task: Fixed prescreen blockers for DEMAND simulated noise robustness, including
  seed/SNR-independent noise offsets, method-specific clean-validation
  thresholds, strengthened SHA/provenance gates, selected-channel consistency,
  active-event SNR reporting, full per-class/confusion outputs, and hardened
  aggregate screening protocol checks.
- Smoke run: fold0/seed3407, lambda 0.5, `DWASHING`, `TBUS`, `STRAFFIC`, active
  SNR `20/10/0` dB, `noise_repeat=0`, zero-shot frozen.
- Clean equivalence: passed with 168/168 path, y_true, and raw Softmax y_pred
  matches; Macro-F1 exactly reproduced at `0.946360153256705`.
- Provenance gates: checkpoint, prototype bundle, calibration JSON, test
  manifest, noise manifest, DEMAND provenance JSON, selected noise file SHA,
  noise-vs-pig MD5 disjointness, and selected channel all verified true.
- Same offset proof: `same_offset_across_snr_verified=true` over 504
  clean/environment/repeat draw groups; unit tests verify same offset across SNR
  and model seed, with different environment/repeat changing the offset.
- Method-specific clean-validation thresholds: raw Softmax global `0.862944`,
  prototype `0.827906`, hierarchical `0.853217`.
- Smoke Macro-F1: clean raw/prototype/hierarchical
  `0.946360/0.946360/0.952273`; DWASHING 20
  `0.656321/0.665814/0.661102`; DWASHING 10
  `0.604815/0.614662/0.620090`; DWASHING 0
  `0.594665/0.605114/0.600000`; TBUS 20
  `0.614045/0.614045/0.619464`; TBUS 10
  `0.599470/0.604409/0.604815`; TBUS 0
  `0.563761/0.576143/0.564466`; STRAFFIC 20
  `0.594616/0.604815/0.599940`; STRAFFIC 10
  `0.533141/0.553182/0.540000`; STRAFFIC 0
  `0.377213/0.363202/0.355094`.
- Per-class warning: cough F1 was `0.0` for all methods across all three 0 dB
  active-event SNR conditions, so 0 dB should be framed as severe stress testing
  rather than expected deployment performance.
- SNR audit: max active-region SNR error `4.918568325962269e-07` dB; DWASHING
  cough 20 dB active mean `20.000000` while full-window mean was `12.920643`
  because cough valid-duration ratio averaged `0.234185`.
- Outputs: updated DEMAND manifest/provenance, noise protocol docs, smoke
  metrics/confusion/SNR/provenance CSV/JSON, and hardened smoke/aggregate
  summaries under `reports/prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke`.
- Recommendation: protocol is ready for review before a fixed 15-run DEMAND
  screening, but do not start it without explicit `APPROVED: true`.
