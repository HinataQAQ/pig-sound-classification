# Prototype Noise Robustness Protocol

This protocol evaluates the exact hierarchical acoustic prototype pipeline under
simulated DEMAND noise without changing the clean model, prototypes, calibration,
or rejection thresholds.

## Scope

Current screening scope:

- folds: 0, 1, 2, 3, 4
- seeds: 42, 2024, 3407
- checkpoint family: hierarchical, lambda 0.5
- feature backend: `training_exact`
- noise source: DEMAND
- environments: `DWASHING`, `TBUS`, `STRAFFIC`
- active-event SNRs: 20 dB, 10 dB, 0 dB
- global noise seed: 3407
- noise repeat: 0
- protocol: zero-shot frozen

The 5 folds x 3 seeds screening protocol is a paper-candidate simulated-noise
screening result. It is not a 25-run final result, not a paper main result, and
not real-farm external validation.

The final simulated-noise protocol extends the same frozen protocol to exactly
5 folds x 5 seeds:

- folds: 0, 1, 2, 3, 4
- seeds: 42, 123, 777, 2024, 3407
- no new environments, SNRs, noise repeats, or recalibration

Only the complete, unique, fully qualified 25-run aggregate may set
`run_scope=aggregate`, `final_25_run_result=true`, `paper_candidate_result=true`,
`paper_main_result=true`, and `simulated_noise_main_result=true`. Every single
fold-seed run, including `run_stage=final`, remains `run_scope=fold_seed`,
`single_fold_debug=false`, and `paper_main_result=false`.

## Frozen Zero-Shot Rule

The noisy evaluation reuses the clean artifacts:

- clean checkpoint
- train-only clean prototypes
- validation-only clean calibration
- frozen rejection thresholds
- frozen fusion parameters

No noisy validation recalibration is allowed. Test results are evaluation only.

Method-specific rejection thresholds are derived from clean validation
predictions only. For `raw_softmax`, `prototype`, and `hierarchical`, the
evaluator records:

- global threshold at target coverage 0.95
- per-class thresholds
- source validation predictions SHA256
- target coverage
- threshold selection formula

No noisy validation or noisy test predictions may select these thresholds.

All noise robustness outputs must record:

- `simulated_noise=true`
- `noise_protocol=zero_shot_frozen`
- `real_farm_external_validation=false`
- `feature_backend=training_exact`
- `run_stage=prescreen_smoke`, `run_stage=screening`, or `run_stage=final`
- `run_scope=fold_seed` for a single run
- `paper_main_result=false` for a single run

## Clean Equivalence Gate

Before adding noise, the evaluator must reproduce the clean exact Softmax test
result for the same fold-seed:

- same sample count
- normalized path alignment
- `y_true` match
- raw Softmax `y_pred` match
- raw Softmax Macro-F1 absolute difference <= 1e-6

If this gate fails, noisy evaluation must stop.

## Mixing Definition

For every test sample:

1. Load the clean waveform with the same center crop / zero padding policy used
   by the training feature path.
2. Compute clean RMS over the original non-padding valid region only.
3. Select a deterministic DEMAND noise segment.
4. Scale noise to the target active-region SNR.
5. Add scaled noise across the full 2-second waveform.
6. Apply one global scale only if needed to avoid clipping.
7. Extract the exact training Log-Mel feature from the mixed waveform.

The same mixed waveform, feature, embedding, and model forward pass are reused
for raw Softmax, main prototype, hierarchical prototype, and fused predictions.
The fused result is supplementary.

## Deterministic Noise Selection

Noise offset selection uses SHA256, not Python's process-randomized `hash`.
The active protocol key version is `demand_noise_offset_v2`.

The key includes:

- fold
- normalized clean path
- clean MD5
- DEMAND environment recording ID
- noise file SHA256
- global noise seed
- noise repeat

The key deliberately excludes model seed and active-event SNR. For a fixed clean
sample, DEMAND environment, and repeat, all model seeds and SNR values reuse the
same noise segment; only the gain changes across SNR.

## Per-Sample Provenance

Each generated condition records:

- `clean_path`
- `clean_md5`
- `original_valid_samples`
- `noise_dataset`
- `noise_environment`
- `noise_file`
- `noise_sha256`
- `selected_channel`
- `noise_offset`
- `noise_seed`
- `noise_draw_id`
- `noise_repeat`
- `offset_key_version`
- `snr_reference=active_valid_region`
- `target_active_snr_db`
- `clean_active_rms`
- `noise_active_rms_before_gain`
- `gain`
- `achieved_active_snr_db`
- `achieved_full_window_snr_db`
- `valid_duration_ratio`
- `peak_before_scale`
- `final_global_scale`
- `clipping_detected`

Samples fail if the clean or noise active-region RMS is below epsilon.

## Primary SNR Grid

The planned screening grid is:

- clean
- 20 dB
- 10 dB
- 0 dB

SNR values refer to active-event SNR over the clean valid non-padding region. Use
phrases such as `0 dB active-event SNR`, not `0 dB clip SNR`.

The completed screening uses all three selected DEMAND environments at 20, 10,
and 0 dB active-event SNR for all 5 folds x 3 seeds. The 0 dB condition must be
described as an extreme simulated-noise stress condition.

## Reporting

Report raw Softmax, main prototype, hierarchical prototype, and fused metrics by
condition:

- Top-1 accuracy
- Macro-F1
- Top-2 accuracy
- ECE
- Brier score
- NLL
- coverage and selective risk
- method-specific frozen-threshold coverage and selective risk
- AURC
- AUGRC, computed from the generalized risk-coverage curve used by the official
  fd-shifts `RiskCoverageStats` implementation. For accepted prefix size `k` among `n`
  samples, AURC uses `sum(loss_accepted) / k`, while generalized risk uses
  `sum(loss_accepted) / n`. In this repository AUGRC is stored in unscaled
  0..1 area units, so it must not be copied from AURC even when the loss is
  the binary 0/1 classification error.
- risk at coverage 0.80, 0.90, and 0.95
- degradation versus clean Macro-F1
- complete confusion matrix
- precision, recall, and F1 for all four main classes
- active/full-window SNR statistics by true class

Paper language must call these results simulated noise robustness. They are not
real-farm external validation.
