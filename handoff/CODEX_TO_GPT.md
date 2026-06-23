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
- selected ECE / Brier / NLL: 0.03415388188191824 / 0.07725942134857178 / 0.1203778013586998

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
