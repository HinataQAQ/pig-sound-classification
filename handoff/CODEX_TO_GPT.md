# Codex to GPT Handoff - SCI Draft v1 Paper Package

## Status

Completed. The user provided `APPROVED: true` for the post-experiment paper
stage. No new model training, inference, SNR, noise draw, prototype,
checkpoint, calibration, or statistical protocol change was started.

This round only corrected AUGRC, regenerated selective-classification
statistics from existing prediction CSV files, created a paper-ready package,
and drafted bilingual manuscript files.

## Branch

- Branch: `paper/sci-draft-v1`
- Base commit: `c6de8d1541643da3d21d62032c4544f5d39cffae`
- Commit SHA: see final Codex response after commit/push

## Code Changes

- `tools/eval_prototype_noise_robustness.py`
  - Corrected `generalized_risk_coverage_auc`.
  - AURC still uses `sum(loss_accepted) / accepted_count`.
  - AUGRC now uses generalized risk
    `sum(loss_accepted) / total_sample_count`, integrated over coverage.
  - Stored scale remains project-native 0..1, not fd-shifts display scale x1000.
- `tools/summarize_prototype_noise_robustness.py`
  - Added `--recompute_selective_from_predictions`.
  - Recomputes AURC, corrected AUGRC, risk@80/90/95, and frozen-threshold
    coverage/risk from existing `noise_predictions.csv`.
  - Removed automatic AUGRC backfill from AURC.
- `tests/test_noise_robustness_pipeline.py`
  - Added deterministic test where AURC and AUGRC differ.
  - Added summary test proving AUGRC is recomputed from prediction CSV rather
    than copied from JSON/AURC.
- `tools/create_paper_package.py`
  - Added reproducible paper package generator.
- `docs/NOISE_ROBUSTNESS_PROTOCOL.md`
  - Documented corrected AUGRC formula and source.

## Commands

```powershell
$env:PYTHONNOUSERSITE="1"
git switch -c paper/sci-draft-v1

$metrics = @()
foreach ($fold in 0..4) {
  foreach ($seed in @(42,123,777,2024,3407)) {
    $stage = if ($seed -in @(123,777)) { "final" } else { "screening" }
    $metrics += "reports/prototype_noise_demand_w05_fold${fold}_seed${seed}_${stage}/noise_metrics.json"
  }
}
C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\summarize_prototype_noise_robustness.py --metrics_json $metrics --protocol final --out_dir reports\prototype_noise_demand_w05_final --file_prefix final --recompute_selective_from_predictions
C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\create_paper_package.py
C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\summarize_prototype_noise_robustness.py --help
C:\py\anaconda3\envs\pigsound-gpu\python.exe -m py_compile tools\eval_prototype_noise_robustness.py tools\summarize_prototype_noise_robustness.py tools\create_paper_package.py tests\test_noise_robustness_pipeline.py
C:\py\anaconda3\envs\pigsound-gpu\python.exe -m unittest tests.test_noise_robustness_pipeline -v
git diff --check
```

## Corrected Selective Metrics

The corrected `reports/prototype_noise_demand_w05_final/final_aurc_augrc_summary.csv`
now has distinct AURC and AUGRC values. For `ALL_NOISY`:

| Method | AURC | AUGRC | Frozen coverage | Frozen selective risk |
| --- | ---: | ---: | ---: | ---: |
| raw_softmax | 0.177458 | 0.123881 | 0.920000 | 0.302386 |
| prototype | 0.210937 | 0.139362 | 0.921190 | 0.298765 |
| hierarchical | 0.212211 | 0.140298 | 0.921852 | 0.303299 |
| fused | 0.184808 | 0.127355 | 0.919418 | 0.301143 |

Aggregate Macro-F1 is unchanged because no model inference was rerun:

- `ALL_NOISY` raw/prototype/hierarchical/fused:
  `0.617281`, `0.623036`, `0.619991`, `0.620066`.
- `MODERATE_NOISE` raw/prototype/hierarchical/fused:
  `0.647225`, `0.653354`, `0.652175`, `0.649915`.
- `EXTREME_STRESS` raw/prototype/hierarchical/fused:
  `0.557393`, `0.562400`, `0.555623`, `0.560369`.

## Paper Package

Created:

- `paper/manuscript/paper_cn_draft.md`
- `paper/manuscript/paper_en_draft.md`
- `paper/references/references.bib`
- `paper/CLAIMS_AND_EVIDENCE.md`
- `paper/tables/table1_dataset_and_leakage_free_protocol.csv`
- `paper/tables/table2_clean_baseline_and_architecture_ablations.csv`
- `paper/tables/table3_duration_comparison.csv`
- `paper/tables/table4_hierarchical_aux_weight_ablation.csv`
- `paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv`
- `paper/tables/table6_simulated_noise_robustness_by_stratum.csv`
- `paper/tables/table7_paired_statistics.csv`
- `paper/tables/table8_per_class_noise_results_and_confusion_directions.csv`
- `paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv`
- `paper/figures/figure1_overall_method_architecture.{svg,pdf,png}`
- `paper/figures/figure2_duration_macro_f1.{svg,pdf,png}`
- `paper/figures/figure3_prototype_structure.{svg,pdf,png}`
- `paper/figures/figure4_macro_f1_vs_snr.{svg,pdf,png}`
- `paper/figures/figure5_noise_degradation_by_environment.{svg,pdf,png}`
- `paper/figures/figure6_feeding_stress_confusion.{svg,pdf,png}`
- `paper/figures/figure7_cough_collapse.{svg,pdf,png}`
- `paper/figures/figure8_risk_coverage_curves.{svg,pdf,png}`
- `paper/appendix/*`
- `paper/reproducibility/*`

Reference count: `6`.

## Data Boundaries

- No training data, validation data, or test data role was changed.
- No test-set tuning was performed.
- Selective metrics were recomputed from saved frozen-test prediction CSV files.
- Full prediction CSV files were not copied into `paper/`.
- No audio, DEMAND archives, checkpoints, embeddings, NPZ files, or caches were
  added for this paper package.

## Paper Usability

- `paper_usable_as_single_run_evidence`: false
- `paper_main_result`: true only for the existing final 25-run aggregate.
- `real_farm_external_validation`: false
- `simulated_noise_only`: true

## Interpretation Guardrails

Supported:

1. 2-second context significantly improves clean pig-vocal classification.
2. Hierarchical subtype supervision adds fine-grained semantics and improves
   the clean boundary, but its gain over the 2-second baseline is not
   statistically significant.
3. Main-class prototype inference significantly improves aggregate Macro-F1
   under simulated DEMAND noise without noisy retraining.

Not supported:

- Hierarchical prototype is most noise robust.
- Real-farm external validation.
- Universal uncertainty improvement.
- First invention of prototypical networks.
- Successful cough recognition at 0 dB.

## Blockers / Questions

- The draft is structurally complete but still needs human scientific editing
  before journal submission.
- Reference metadata is intentionally conservative; verify journal-specific
  bibliography style before submission.
- Decide whether to add a dedicated cough-failure analysis subsection or leave
  it as a limitation.
