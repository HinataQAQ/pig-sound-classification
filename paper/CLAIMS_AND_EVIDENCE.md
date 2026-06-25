# Claims and Evidence

## Supported

1. 2-second context significantly improves clean pig-vocal classification.
   Evidence: `paper/tables/table3_duration_comparison.csv` and paired duration statistics.

2. Hierarchical subtype supervision introduces fine-grained acoustic semantics and improves the clean decision boundary, but its incremental gain over the 2-second baseline is not statistically significant.
   Evidence: `paper/tables/table4_hierarchical_aux_weight_ablation.csv` and paired hierarchy-vs-duration statistics.

3. Post-hoc main-class prototype inference significantly improves aggregate Macro-F1 under simulated DEMAND noise without noisy retraining.
   Evidence: ALL_NOISY prototype - raw_softmax mean delta 0.005755, Wilcoxon p=0.010511.

## Not Claimed

- Hierarchical prototype is the most noise robust.
- Real-farm external validation has been completed.
- Universal uncertainty improvement.
- First invention of prototypical networks.
- Successful cough recognition at 0 dB.

## AUGRC Correction

AURC uses `sum(loss_accepted) / accepted_count`; corrected AUGRC uses fd-shifts generalized risk `sum(loss_accepted) / total_count`, integrated over coverage. The corrected final selective summaries were recomputed from existing prediction CSV files, not from new model inference.
