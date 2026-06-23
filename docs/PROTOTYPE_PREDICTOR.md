# Hierarchical Acoustic Prototype Atlas Predictor

This workflow builds a post-hoc prototype predictor from a trained 2-second
hierarchical Log-Mel CRNN checkpoint. It does not retrain the backbone, change
the softmax heads, modify audio, or modify the validated training scripts.

## Phase 1 Scope

Phase 1 is limited to one matched fold-seed run:

- checkpoint: `checkpoints\cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407.pt`
- fold: `0`
- seed: `3407`
- hierarchy weight: recovered from the matching `summary.json` and checked with
  `--expected_hier_aux_weight 0.5`
- prototype source: train split only
- calibration source: validation split only
- final evaluation source: test split only

Lambda is not hard-coded in the scripts. The selected checkpoint and summary
determine the model configuration, and the expected value is supplied as a
command-line validation guard. Lambda 1.0 must be run later as a separate
ablation with separate embeddings, prototypes, calibration, and outputs.

## Data Flow

For each fold-seed:

1. Load `HierCRNN` from `tools\train_hier_longcontext_crnn.py`.
2. Restore the matching checkpoint state dict.
3. Reuse `HierCRNN.encode(x)` to extract shared embeddings.
4. L2-normalize embeddings.
5. Build main and auxiliary prototypes from train embeddings only.
6. Use validation only to choose method-specific calibration parameters:
   calibrated Softmax temperature by NLL, prototype temperature for prototype
   methods, fusion alpha after both sides are fixed, hierarchy confidence
   penalty, and reject thresholds.
7. Verify manifest content SHA256 before calibration or frozen test prediction.
8. Freeze all calibration parameters before running test prediction.
9. Evaluate test predictions without selecting any parameter from test metrics.

The output label for rejection is `uncertain`. Without a real unknown validation
set, these results are not formal unknown detection, open-set recognition, or
open-set accuracy.

## Distance Definition

Embeddings are L2-normalized before prototype construction. A prototype is the
mean of same-class normalized train embeddings, then L2-normalized again.

Primary ranking uses cosine distance:

```text
cosine_distance = 1 - cosine_similarity
```

For normalized vectors, Euclidean distance is determined by cosine similarity:

```text
euclidean_distance = sqrt(2 - 2 * cosine_similarity)
```

Both are saved for audit, but they should not be interpreted as independent
evidence in Phase 1.

## Manifest Identity

Build metadata records train, validation, and test manifest SHA256 values. During
calibration, the supplied validation manifest must match the recorded validation
SHA256. During frozen test prediction, the supplied test manifest must match the
recorded test SHA256. A path match alone is not sufficient.

Prediction inputs have distinct roles:

- `--test_manifest`: `input_role=frozen_test`; checked against the recorded test
  SHA256 and eligible for formal fold evaluation when other qualification fields
  allow it.
- `--manifest`: `input_role=inference_manifest`; used for prediction only and
  not treated as the recorded CV test split.
- `--audio`: `input_role=single_audio`; used for prediction only.

Every structured build, calibration, prediction, and evaluation output records
`feature_pipeline_equivalent`, `single_fold_debug`,
`eligible_for_cv_aggregation`, `paper_main_result`, and `feature_backend`.
`training_exact` sets `feature_pipeline_equivalent=true`; `numpy_logmel` sets it
to false. The fold0/seed3407 Phase 1 run is marked `single_fold_debug=true` and
`paper_main_result=false`.

## Phase 1 Commands

PowerShell setup:

```powershell
conda activate pigsound-gpu
cd C:\py\pigsound\pig-sound-classification
$env:PYTHONNOUSERSITE="1"
$ROOT="reports\prototype_cv5_fold0_seed3407"
$TRAIN="paper_results\manifests\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\train.csv"
$VAL="paper_results\manifests\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\val.csv"
$TEST="paper_results\manifests\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\test.csv"
$CKPT="checkpoints\cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407.pt"
$SUMMARY="reports\cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407\summary.json"
```

Build train-only prototypes:

```powershell
python tools\build_hier_acoustic_prototypes.py `
  --train_manifest $TRAIN `
  --val_manifest $VAL `
  --test_manifest $TEST `
  --ckpt $CKPT `
  --summary_json $SUMMARY `
  --out_dir $ROOT `
  --fold 0 `
  --seed 3407 `
  --expected_hier_aux_weight 0.5 `
  --expected_dur_s 2.0 `
  --expected_feature_mode logmel `
  --feature_backend training_exact
```

`numpy_logmel` is retained only for fast pipeline debugging. Results produced
with that backend must be marked `non_paper_equivalent` and must not be used as
paper-equivalent prototype metrics.

Calibrate on validation only:

```powershell
python tools\calibrate_prototype_predictor.py `
  --prototype_bundle $ROOT\artifacts\prototype_bundle.npz `
  --val_manifest $VAL `
  --ckpt $CKPT `
  --out_dir $ROOT `
  --feature_backend auto `
  --selection_method hierarchical
```

Run frozen test prediction:

```powershell
python tools\predict_hier_acoustic_prototype.py `
  --prototype_bundle $ROOT\artifacts\prototype_bundle.npz `
  --calibration_json $ROOT\calibration\calibration.json `
  --test_manifest $TEST `
  --ckpt $CKPT `
  --out_dir $ROOT `
  --feature_backend auto
```

Evaluate frozen test predictions:

```powershell
python tools\eval_hier_acoustic_prototype.py `
  --pred_csv $ROOT\evaluation\test_predictions.csv `
  --calibration_json $ROOT\calibration\calibration.json `
  --out_dir $ROOT
```

Build the atlas:

```powershell
python tools\plot_acoustic_prototype_atlas.py `
  --prototype_bundle $ROOT\artifacts\prototype_bundle.npz `
  --out_dir $ROOT
```

## Outputs

`artifacts\` contains:

- `main_prototypes.npz`
- `aux_prototypes.npz`
- `prototype_bundle.npz`
- `prototype_metadata.json`
- `distance_statistics.json`
- `train_main_prototype_samples.csv`
- `train_aux_prototype_samples.csv`
- `leakage_audit.json`

`calibration\` contains:

- `calibration.json`
- `calibration_grid.csv`
- `hierarchy_penalty_grid.csv`
- `coverage_risk.csv`
- `val_predictions.csv`
- `metrics_by_method_val.csv`
- `leakage_audit.json`

`calibration.json` stores independent best parameters for `raw_softmax`,
`calibrated_softmax`, `prototype`, `hierarchical`, and `fused`. `raw_softmax`
has no temperature scaling. `calibrated_softmax` selects only
`softmax_temperature`, primarily by validation NLL. Prototype methods select
only prototype-side parameters. Fused predictions use:

```text
fused = alpha * softmax + (1 - alpha) * prototype
```

where `alpha=0` is `pure_prototype`, `alpha=1` is `pure_softmax`, and only
`fusion_kind=mixed` should be interpreted as a genuine fusion result.

`evaluation\` contains:

- `test_predictions.csv`
- `test_predictions.json`
- `metrics.json`
- `metrics_by_method_test.csv`
- `coverage_risk.csv`
- `selective_risk_summary.csv`
- `confusion_matrices.json`
- `hierarchy_consistency.json`
- `prediction_metadata.json`
- `leakage_audit.json`

`atlas\` contains main and auxiliary normalized Log-Mel mean/std heatmaps,
prototype norm CSVs, distance-statistic CSVs, representative/farthest/ambiguous
sample lists, prototype cosine-similarity matrices, and feeding/stress focused
analysis. These heatmaps are titled `Normalized Log-Mel mean` and `Normalized
Log-Mel standard deviation` because they use the training feature tensors after
the training pipeline's per-sample z-score normalization. They support relative
pattern inspection and do not represent absolute acoustic energy.

Sample paths are written to CSV/JSON for audit. Audio files are not copied.

## Prediction Fields

`test_predictions.csv` includes:

- true labels when a manifest is used
- `input_role` and paper-qualification fields
- `raw_softmax` top-k from the uncalibrated model head
- `calibrated_softmax` top-k after validation-selected softmax temperature
- main-prototype, hierarchical-prototype, and fused top-k
- auxiliary prototype top-k
- class probabilities
- nearest main and auxiliary prototypes
- cosine similarity, cosine distance, and Euclidean distance
- normalized distance audit fields based on train distance statistics
- `hierarchy_inconsistent`
- `selected_confidence_raw`
- validation-selected hierarchy-penalized `selected_confidence`
- `uncertainty = 1 - selected_confidence`
- `known_state_global` and `known_state_per_class`
- `pred_or_uncertain_global` and `pred_or_uncertain_per_class`

## Safety Boundary

This predictor is an engineering tool for acoustic classification and model
interpretability. `uncertain` means the sample is far from or weakly supported by
the known-class prototype geometry under validation-calibrated rules. It must
not be treated as a medical diagnosis, welfare diagnosis, or confirmed unknown
sound type.
