# Codex to GPT Handoff

## Task
Review the exact hierarchical acoustic prototype predictor for the pig sound
classification project.

## Status
Completed first-phase exact fold0/seed3407 review artifacts. Do not run full CV,
lambda=1.0, or unknown/open-set experiments from this handoff.

## Branch
prototype-exact-review

## Scope
- lambda 0.5 hierarchical checkpoint only
- fold0 / seed3407 only
- train split builds prototypes
- validation split calibrates temperatures, thresholds, fusion weight, and hierarchy penalty
- test split is frozen evaluation only
- no training scripts were modified

## Key Results
- exact Softmax reproduction passed
- test rows: 168
- reference Macro-F1: 0.946360153256705
- reproduced Macro-F1: 0.946360153256705
- y_pred match count: 168 / 168
- sample alignment by normalized path: passed
- main prototype Macro-F1: 0.946360153256705
- hierarchical prototype Macro-F1: 0.9522727272727273
- auxiliary prototype Macro-F1: 0.8355042016806723
- selected ECE / Brier / NLL: 0.034153742094834655 / 0.07724517583847046 / 0.12035306543111801

## Follow-up Fixes
- Calibration is split by method. `raw_softmax` is uncalibrated,
  `calibrated_softmax` selects only softmax temperature by validation NLL,
  prototype methods select prototype-side parameters, and fused alpha is selected
  only after both sides are fixed.
- Fused alpha endpoints are labelled: alpha 0 is `pure_prototype`, alpha 1 is
  `pure_softmax`; only `fusion_kind=mixed` is an independent fusion result.
- Manifest leakage audit now requires path/source_id/md5, normalizes identities,
  rejects empty values, and validates MD5 format.
- Softmax reproduction aligns reference and reproduced predictions by normalized
  path and emits `sample_alignment_report.csv`.
- Prototype construction fails when any main or auxiliary class has zero count.
- Checkpoint relocation and unsafe SHA mismatch bypass are separate flags.
- Atlas heatmaps are labelled as normalized Log-Mel maps.
- Calibration now verifies supplied validation manifest SHA256 against prototype
  metadata, so same-path content edits fail.
- Frozen test prediction now verifies supplied test manifest SHA256 against
  prototype metadata.
- `--test_manifest`, `--manifest`, and `--audio` are distinct input roles.
  Only `input_role=frozen_test` may enter formal evaluation.
- Build, calibration, prediction, Softmax reproduction, and evaluation outputs
  now generate qualification fields: `feature_pipeline_equivalent`,
  `single_fold_debug`, `eligible_for_cv_aggregation`, `paper_main_result`, and
  `feature_backend`.
- Prediction parameter restoration now uses explicit `is None` checks so
  legitimate `0.0` values are preserved.
- Path identity normalization now folds `.`, `..`, repeated slashes, backslash
  variants, and Windows case.
- Single fold-seed outputs now always carry `run_scope=fold_seed` and
  `paper_main_result=false`.
- Prediction metadata now records prediction CSV SHA, calibration JSON SHA,
  prototype bundle SHA, leakage audit status, manifest SHA verification status,
  and run scope.
- Evaluation now requires `prediction_metadata.json` and verifies input role,
  prediction CSV SHA, calibration JSON SHA, fold/seed/backend provenance, and
  qualification fields before computing metrics.
- Added `tools/summarize_cv5_exact_prototype.py`; only a complete aggregate may
  set `run_scope=aggregate` and `paper_main_result=true`.

## Data Separation Audit
- prototype source: fold0 train manifest only
- calibration source: fold0 validation manifest only
- evaluation source: fold0 test manifest only
- path/source_id/MD5 overlap: none

## Important Notes
- The older numpy_logmel debug directory is non-paper-equivalent.
- The paper-equivalent path uses the training-exact feature backend.
- Probability max absolute drift versus reference test_pred.csv is recorded as
  advisory: 0.0009449124336242676.
- CPU was used for the final one-fold debug rerun because `--device auto`
  selected a CUDA path that stayed CPU-bound and did not complete in 35 minutes
  on this workstation. The feature backend remained `training_exact`.
- Negative tests passed: modified validation manifest content failed
  calibration by SHA; modified frozen test manifest content failed prediction by
  SHA; `numpy_logmel` qualification flags are non-paper-equivalent and not
  eligible for CV aggregation.
- Final provenance-only review tests passed: metadata missing fails evaluation,
  inference-manifest predictions fail evaluation, edited prediction CSV fails
  evaluation by SHA mismatch, fold1/seed42 single run remains non-paper-main,
  and only complete aggregate validation may produce `paper_main_result=true`.
- No checkpoint, audio, NPZ, embedding, or cache files should be included in the
  review commit.

## Review Outputs
- reports/prototype_feature_parity_fold0_seed3407/environment_audit.json
- reports/prototype_feature_parity_fold0_seed3407/feature_parity_report.csv
- reports/prototype_cv5_fold0_seed3407_exact/softmax_reproduction/softmax_reproduction.json
- reports/prototype_cv5_fold0_seed3407_exact/softmax_reproduction/sample_alignment_report.csv
- reports/prototype_cv5_fold0_seed3407_exact/calibration/calibration.json
- reports/prototype_cv5_fold0_seed3407_exact/evaluation/metrics.json
- reports/prototype_cv5_fold0_seed3407_exact/evaluation/coverage_risk.csv
- reports/prototype_cv5_fold0_seed3407_exact/artifacts/leakage_audit.json
- reports/prototype_cv5_fold0_seed3407_exact/run_manifest.json

## Recommended Next Step
Proceed to lambda 0.5, 5 folds x 3 seeds only after review approval. Keep
lambda 1.0 as a later matched ablation and do not mix embeddings or prototypes.
