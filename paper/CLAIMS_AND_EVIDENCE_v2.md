# Claims and Evidence v2

This ledger supports `paper/manuscript/paper_cn_full_story.md` and
`paper/manuscript/paper_en_full_story.md`. It is designed to prevent unsupported
novelty, significance and deployment claims.

## Core argument

The paper shows that closed-set pig vocalization recognition is best framed as
an evidence-controlled sequence: first select a suitable 2 s event context for
the clean Log-Mel CRNN mainline, then add hierarchical subtype supervision for
fine-grained semantic structure, and finally use post-hoc main-class prototype
inference for modest simulated DEMAND noise robustness.

## Supported claims

| Claim | Evidence source | Exact support | Allowed wording | Forbidden wording |
|---|---|---|---|---|
| 2 s context is the decisive clean mainline gain. | `paper/tables/table3_duration_comparison.csv` | 1 s Log-Mel Macro-F1 `0.9226`; 2 s Log-Mel Macro-F1 `0.9472`; 2 s - 1 s mean delta `0.0246`; Wilcoxon p `0.000162`. | "2 s context significantly improved clean closed-set Macro-F1 over 1 s input." | "First use of 2 s"; "2 s is universally optimal"; "3 s can never help." |
| Feature stacking and complex clean variants did not replace the 2 s Log-Mel CRNN mainline. | `paper/tables/table2_clean_baseline_and_architecture_ablations.csv`; project guardrails for completed BiLSTM direction | 2 s Log-Mel CRNN `0.9472`; cap3x MCTAFD no-SE gated `0.9208`; PCEN `0.8916`; spectral-gate `0.9093`; SpecAugment `0.9198`; attention pooling `0.9431`. | "The locked clean package did not support returning to feature stacking as the primary direction." | "All complex methods are useless"; "BiLSTM value was X" unless a locked table is added. |
| 3 s did not outperform 2 s in the available duration table. | `paper/tables/table3_duration_comparison.csv` | 3 s Log-Mel Macro-F1 `0.9434` over `n=15`; 2 s Log-Mel Macro-F1 `0.9472` over `n=25`. | "The available 3 s comparison did not surpass the 2 s mean." | "3 s is statistically worse" unless paired stats are added. |
| Hierarchical auxiliary supervision adds semantic granularity but is not a significant main-performance claim over the 2 s baseline. | `paper/tables/table4_hierarchical_aux_weight_ablation.csv`; existing paired hierarchy-vs-duration summary referenced by project state | lambda=0.2 `0.9505`; lambda=0.5 `0.9510`; lambda=1.0 `0.9515`; hierarchical gain over 2 s baseline is not statistically significant per project guardrails. | "Hierarchical supervision provides fine-grained semantic and boundary evidence." | "Hierarchical supervision significantly improves over the 2 s baseline." |
| lambda=1.0 has the highest mean and lambda=0.5 is the balanced setting. | `paper/tables/table4_hierarchical_aux_weight_ablation.csv` | lambda=1.0 mean Macro-F1 `0.9515`; lambda=0.5 std `0.0124`, lowest among listed weights. | "lambda=1.0 has the highest mean; lambda=0.5 has the lowest standard deviation and is the balanced setting." | "lambda=0.5 is the best by every criterion." |
| Clean hierarchical prototype has the highest clean Macro-F1 among prototype variants. | `paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv`; `paper/appendix/clean_25_run_prototype_summary.csv` | hierarchical `0.9540`; fused `0.9535`; prototype `0.9519`; raw Softmax `0.9510`. | "Clean hierarchical prototype reached the highest mean Macro-F1." | "Hierarchical prototype is statistically superior to raw Softmax." |
| Clean hierarchical prototype gain over raw Softmax is not statistically significant. | `paper/appendix/clean_25_run_prototype_paired_stats.csv`; `paper/tables/table7_paired_statistics.csv` | hierarchical - raw mean delta `0.002937`; CI `[-0.000478, 0.007294]`; Wilcoxon p `0.058253`. | "The clean gain is positive in mean but did not meet the significance threshold." | "Clean hierarchical prototype significantly improves Macro-F1." |
| Main prototype is the most stable aggregate simulated-noise inference method. | `paper/tables/table6_simulated_noise_robustness_by_stratum.csv`; `paper/tables/table7_paired_statistics.csv` | ALL_NOISY raw/prototype/hierarchical Macro-F1 `0.6173/0.6230/0.6200`; prototype - raw delta `0.005755`; p `0.010511`. | "Main prototype provides a modest but statistically supported ALL_NOISY Macro-F1 gain." | "Prototype solves noise robustness"; "real-farm robust." |
| Moderate simulated noise supports the main prototype direction. | `paper/tables/table6_simulated_noise_robustness_by_stratum.csv`; `paper/tables/table7_paired_statistics.csv` | MODERATE_NOISE raw/prototype/hierarchical Macro-F1 `0.6472/0.6534/0.6522`; prototype - raw p `0.001673`. | "The main prototype gain is clearest under moderate simulated noise." | "All SNR strata show significant gains." |
| Extreme stress does not support a significant main-prototype gain. | `paper/tables/table6_simulated_noise_robustness_by_stratum.csv`; `paper/tables/table7_paired_statistics.csv` | EXTREME_STRESS raw/prototype/hierarchical Macro-F1 `0.5574/0.5624/0.5556`; prototype - raw p `0.560171`. | "In extreme stress, main prototype remains numerically higher but nonsignificant." | "Main prototype significantly improves 0 dB/extreme stress." |
| Hierarchical prototype is weaker than main prototype under simulated noise. | `paper/tables/table7_paired_statistics.csv` | ALL_NOISY hierarchical - prototype delta `-0.003045`, p `0.000376`; EXTREME_STRESS delta `-0.006777`, p `0.000162`. | "Hierarchical prototype is useful for clean semantics but weaker than main prototype under noise." | "Hierarchical prototype dominates the noisy setting." |
| DEMAND noise is simulated additive noise, not real-farm external validation. | `docs/NOISE_ROBUSTNESS_PROTOCOL.md`; `docs/NOISE_SOURCE_DECISION.md`; `paper/tables/table1_dataset_and_leakage_free_protocol.csv` | External validation field says not performed; protocol says simulated-noise robustness only. | "DEMAND simulated additive noise." | "Real pig-farm validation"; "field validation." |
| 0 dB active-event SNR is an extreme stress condition and cough is unreliable there. | `docs/NOISE_ROBUSTNESS_PROTOCOL.md`; `paper/appendix/noise_25_run_per_class_summary.csv`; `paper/tables/table8_per_class_noise_results_and_confusion_directions.csv` | 0 dB cough recall is zero for STRAFFIC and TBUS across all methods; DWASHING 0 dB cough recall is below `0.009`; ALL_NOISY cough F1 raw/prototype/hierarchical `0.0867/0.0974/0.0957`. | "0 dB active-event SNR is an extreme stress condition; cough is unreliable." | "0 dB equals a clean/no-noise setting"; "cough is robust at 0 dB." |
| Prototypes do not comprehensively improve uncertainty or risk ranking. | `paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv`; `paper/appendix/noise_25_run_aurc_augrc_summary.csv` | ALL_NOISY AURC/AUGRC raw `0.1775/0.1239`, prototype `0.2109/0.1394`, hierarchical `0.2122/0.1403`; lower is better. | "Raw Softmax ranks better by AURC/AUGRC; prototype only slightly improves frozen-threshold selective risk." | "Prototype improves uncertainty"; "prototype has better AURC/AUGRC." |
| Fused is an ablation, not the main innovation. | `paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv`; `paper/tables/table6_simulated_noise_robustness_by_stratum.csv` | Fused rows exist but are not the strongest scientific mechanism under noise. | "Fused is retained as an ablation." | "Fused is the proposed main method" unless a new explicit contribution is approved. |

## Required boundary statements

- DEMAND is simulated additive noise and is not real-farm external validation.
- 0 dB active-event SNR is an extreme stress condition, not no noise.
- Cough recognition is unreliable at 0 dB.
- Hierarchical supervision and clean hierarchical prototype gains are not statistically significant over their key comparators.
- Prototype inference does not comprehensively improve uncertainty.
- AURC/AUGRC rank raw Softmax better than prototype variants in the aggregate noise summaries.
- Prototype only slightly improves frozen-threshold selective risk.
- Hierarchical prototype is weaker than main prototype under noise.
- Fused is an ablation, not the main innovation.

## Citation gaps

The following claims require verified literature before submission:

- Pig vocalization recognition and livestock acoustic monitoring context.
- Prior work on cough, stress and feeding sound recognition in pigs.
- Environmental-noise robustness in animal-acoustic or farm-acoustic monitoring.
- Any claim comparing this pipeline to external published pig-sound systems.

Do not fabricate DOI, author names, venue names or dataset claims to fill these gaps.

## Paper usability

- `paper_usable`: true as a full-story manuscript draft grounded in existing paper result files.
- `new_model_results_created`: false.
- `source_csv_json_modified`: false.
- `real_farm_external_validation`: false.
- `simulated_noise_only`: true.
