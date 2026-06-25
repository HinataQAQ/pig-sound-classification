# Duration-Aware Hierarchical Representation and Dual-Prototype Inference for Pig Vocalization Recognition under Simulated Environmental Noise

Abstract: This study investigates pig vocalization recognition with duration-aware hierarchical representations, post-hoc acoustic prototypes, and simulated environmental-noise robustness on top of a validated class-capped cap3x, 2-s Log-Mel CRNN pipeline. Clean closed-set experiments show that 2-s temporal context significantly improves Macro-F1 over 1-s input. Hierarchical subtype supervision adds fine-grained acoustic semantics and improves the feeding/stress_vocal boundary, but its incremental gain over the 2-s baseline is not statistically significant. In the lambda=0.5 25-run clean prototype evaluation, hierarchical prototypes reach a Macro-F1 of 0.954. Under zero-shot frozen DEMAND noise from DWASHING, TBUS, and STRAFFIC, the main prototype achieves an ALL_NOISY Macro-F1 of 0.623, improving over raw Softmax (0.6173) by 0.005755 on average (paired Wilcoxon p=0.010511). The results support main-class prototype inference as a simulated-noise robustness improvement without noisy retraining, while real-farm external validation and reliable 0 dB cough recognition remain open limitations.

## Keywords

Pig vocalization recognition; Log-Mel; CRNN; hierarchical supervision; prototype inference; simulated environmental noise; selective classification

## 1. Introduction

Automatic pig vocalization recognition can support engineering-oriented monitoring in livestock environments. This work studies four closed-set main classes: cough, calm_grunt, feeding, and stress_vocal. The evidence is limited to clean evaluation and DEMAND-based simulated noise; real-farm external validation has not been performed.

## 2. Method

The clean mainline uses 2-s Log-Mel inputs and a CRNN backbone. Hierarchical auxiliary supervision predicts six acoustic subtypes in addition to the main class. The post-hoc prototype module keeps the Softmax model frozen, builds fold-seed-specific main and subtype embedding prototypes from the train split only, calibrates temperatures and thresholds on validation only, and evaluates on the frozen test split.

## 3. Experimental Protocol

All runs preserve path, source_id, and MD5 disjointness. Clean model comparisons use matched fold-seed paired statistics. The simulated-noise protocol uses DEMAND DWASHING, TBUS, and STRAFFIC at active-event SNRs of 20, 10, and 0 dB. Noise draws, SNRs, thresholds, and statistical procedures are frozen before the final 25-run evaluation.

## 4. Results

The 2-s context is the dominant significant improvement on clean data. Hierarchical subtype supervision improves the mean and the feeding/stress_vocal boundary, but its incremental gain over the 2-s baseline is not statistically significant. The clean hierarchical prototype Macro-F1 is 0.954.

Under ALL_NOISY simulated noise, raw Softmax reaches a Macro-F1 of 0.6173, main prototype reaches 0.623, and hierarchical prototype reaches 0.62. Main prototype improves over raw Softmax by 0.005755 on average, with a 95% bootstrap CI of [0.001606, 0.010894] and Wilcoxon p=0.010511. The hierarchical prototype improvement over raw Softmax is 0.00271 and is not statistically significant.

## 5. Discussion and Limitations

The study does not claim that hierarchical prototypes are the most noise robust, does not report real-farm external validation, and does not claim reliable cough recognition at 0 dB. Strong active-event noise collapses cough recall, making cough robustness the main unresolved deployment risk. AUGRC has been corrected according to the fd-shifts generalized-risk definition and recomputed from stored prediction CSV files.

## 6. Conclusion

The evidence supports three fixed claims: 2-s context significantly improves clean classification; hierarchical subtype supervision adds fine-grained semantics but does not provide a statistically significant incremental gain over the 2-s baseline; and post-hoc main-class prototype inference significantly improves aggregate Macro-F1 under simulated DEMAND noise without noisy retraining.
