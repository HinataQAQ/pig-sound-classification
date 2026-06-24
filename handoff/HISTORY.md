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
