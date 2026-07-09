# Figure Contracts

## Figure 1. Overall workflow

- Figure conclusion: The manuscript evaluates a 2 s Log-Mel CRNN with hierarchical heads and train-only post-hoc prototypes under clean and simulated DEMAND noise settings.
- Panels:
  - A. Two-second event context and Log-Mel CRNN encoder.
  - B. Main-class and auxiliary-subtype supervision heads.
  - C. Train-only main and subtype prototype construction followed by validation-only calibration.
  - D. DEMAND simulated-noise evaluation boundary.
- Evidence source CSV/source files:
  - paper/tables/table1_dataset_and_leakage_free_protocol.csv
  - paper/CLAIMS_AND_EVIDENCE.md
- What not to claim:
  - Do not claim real-farm external validation.
  - Do not claim prototype learning was invented here.
  - Do not claim 0 dB is a clean or no-noise setting.
- Export files:
  - paper/figures_v2/figure1_overall_workflow.svg
  - paper/figures_v2/figure1_overall_workflow.pdf
  - paper/figures_v2/figure1_overall_workflow.png

## Figure 2. Duration ablation

- Figure conclusion: A 2 s Log-Mel context is the clean-audio mainline because it improves Macro-F1 over the 1 s baseline in matched 25-run statistics.
- Panels:
  - A. Mean Macro-F1 for 1 s, 2 s, and 3 s contexts with run-to-run standard deviation.
  - B. Annotation of the 2 s - 1 s paired delta and Wilcoxon p-value from the table note.
- Evidence source CSV/source files:
  - paper/tables/table3_duration_comparison.csv
- What not to claim:
  - Do not claim first use of 2 s context.
  - Do not overinterpret 3 s against 2 s because the run counts differ.
- Export files:
  - paper/figures_v2/figure2_duration_ablation.svg
  - paper/figures_v2/figure2_duration_ablation.pdf
  - paper/figures_v2/figure2_duration_ablation.png

## Figure 3. Hierarchical supervision story

- Figure conclusion: Auxiliary subtype supervision adds fine-grained semantic structure; lambda=0.5 is the stable setting and lambda=1.0 has the highest mean Macro-F1.
- Panels:
  - A. Four main classes and six auxiliary subtypes.
  - B. Training objective L_main + lambda L_aux.
  - C. Lambda ablation with Macro-F1 mean and standard deviation.
- Evidence source CSV/source files:
  - paper/tables/table1_dataset_and_leakage_free_protocol.csv
  - paper/tables/table4_hierarchical_aux_weight_ablation.csv
- What not to claim:
  - Do not claim the hierarchical gain is statistically significant over the 2 s baseline.
  - Do not treat the fused output as the main innovation.
- Export files:
  - paper/figures_v2/figure3_hierarchical_supervision.svg
  - paper/figures_v2/figure3_hierarchical_supervision.pdf
  - paper/figures_v2/figure3_hierarchical_supervision.png

## Figure 4. Clean prototype comparison

- Figure conclusion: On clean test data, hierarchical prototype inference has the highest Macro-F1, whereas calibration metrics differ by method.
- Panels:
  - A. Macro-F1.
  - B. Expected calibration error.
  - C. Brier score.
  - D. Negative log likelihood.
- Evidence source CSV/source files:
  - paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv
- What not to claim:
  - Do not claim all prototype variants improve every metric.
  - Do not claim universal uncertainty improvement.
- Export files:
  - paper/figures_v2/figure4_clean_prototype_comparison.svg
  - paper/figures_v2/figure4_clean_prototype_comparison.pdf
  - paper/figures_v2/figure4_clean_prototype_comparison.png

## Figure 5. Simulated noise robustness

- Figure conclusion: The main-class prototype is the most stable aggregate option for MODERATE_NOISE and ALL_NOISY, while hierarchical prototypes are not noise-optimal.
- Panels:
  - A. Macro-F1 across MODERATE_NOISE, EXTREME_STRESS, and ALL_NOISY.
  - B. Paired statistics callouts for prototype - raw softmax.
- Evidence source CSV/source files:
  - paper/tables/table6_simulated_noise_robustness_by_stratum.csv
  - paper/tables/table7_paired_statistics.csv
- What not to claim:
  - Do not claim hierarchical prototype is best under noise.
  - Do not claim simulated DEMAND noise is real-farm external validation.
- Export files:
  - paper/figures_v2/figure5_simulated_noise_robustness.svg
  - paper/figures_v2/figure5_simulated_noise_robustness.pdf
  - paper/figures_v2/figure5_simulated_noise_robustness.png

## Figure 6. Active-event SNR curve

- Figure conclusion: Macro-F1 degrades from clean through 20, 10, and 0 dB active-event SNR, with 0 dB explicitly treated as an extreme simulated-noise stress condition.
- Panels:
  - A. DWASHING SNR curve.
  - B. STRAFFIC SNR curve.
  - C. TBUS SNR curve.
- Evidence source CSV/source files:
  - reports/prototype_noise_demand_w05_final/final_summary.csv
- What not to claim:
  - Do not describe 0 dB as no noise.
  - Do not generalize these simulated DEMAND conditions to external farm validation.
- Export files:
  - paper/figures_v2/figure6_active_event_snr_curve.svg
  - paper/figures_v2/figure6_active_event_snr_curve.pdf
  - paper/figures_v2/figure6_active_event_snr_curve.png

## Figure 7. Per-class failure

- Figure conclusion: Cough recognition collapses under noisy aggregate evaluation, and cough-to-stress plus feeding/stress directions dominate the failure profile.
- Panels:
  - A. Per-class F1 under ALL_NOISY for raw, main prototype, and hierarchical prototype.
  - B. Cough-to-stress_vocal confusion counts.
  - C. Feeding-to-stress and stress-to-feeding confusion directions.
- Evidence source CSV/source files:
  - paper/tables/table8_per_class_noise_results_and_confusion_directions.csv
  - reports/prototype_noise_demand_w05_final/final_per_class_summary.csv
- What not to claim:
  - Do not claim cough is reliable at 0 dB.
  - Do not infer unmeasured causal noise mechanisms from aggregate confusion counts.
- Export files:
  - paper/figures_v2/figure7_per_class_failure.svg
  - paper/figures_v2/figure7_per_class_failure.pdf
  - paper/figures_v2/figure7_per_class_failure.png

## Figure 8. Selective prediction

- Figure conclusion: Prototype inference does not comprehensively improve uncertainty ranking: raw softmax has lower AURC/AUGRC, while main prototypes slightly improve frozen-threshold risk.
- Panels:
  - A. AURC.
  - B. AUGRC.
  - C. Risk at 95% coverage.
  - D. Frozen-threshold selective risk.
- Evidence source CSV/source files:
  - paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv
- What not to claim:
  - Do not claim prototype methods universally improve uncertainty.
  - Do not omit that AURC/AUGRC rank raw softmax better.
- Export files:
  - paper/figures_v2/figure8_selective_prediction.svg
  - paper/figures_v2/figure8_selective_prediction.pdf
  - paper/figures_v2/figure8_selective_prediction.png
