# Section Plan for Full-Story SCI Application Manuscript

## Nature-writing route used

- Skill: `nature-writing`
- Paper type: research
- Sections: title, abstract, introduction, related work, materials and methods, experimental protocol, results, discussion, limitations, conclusion, availability, ethics, references placeholder
- Language: zh-to-en
- Journal fragment: generic

## One-sentence argument

In closed-set pig vocalization recognition, we show that the strongest evidence-supported path is not feature stacking but a 2 s Log-Mel CRNN mainline, bounded hierarchical subtype supervision for semantic structure, and post-hoc main-class prototype inference for modest simulated DEMAND noise robustness, with no claim of real-farm validation or universal uncertainty improvement.

## Terminology ledger

| Canonical term | First-use definition | Variants to avoid or constrain | Decision |
|---|---|---|---|
| pig vocalization recognition | closed-set four-class pig sound classification task | pig sound classification, pig voice recognition | Use in title/abstract; "pig sound" only in informal notes. |
| 2 s Log-Mel CRNN | 2-second Log-Mel input with CRNN backbone | 2-second baseline, long-context model | Define once, then use consistently. |
| class-capped cap3x | training expansion protocol used by the clean mainline | cap3x expansion | Use exact project term. |
| hierarchical auxiliary supervision | main-class loss plus subtype auxiliary loss | hierarchical supervision, subtype supervision | Use full term on first use; later "hierarchical supervision" is acceptable. |
| lambda | auxiliary loss weight in `L = L_main + lambda * L_aux` | λ | Use ASCII `lambda` for manuscript consistency. |
| main prototype | post-hoc prototype using four main-class centers | prototype-only, main-class prototype | Use "main prototype" in tables and "main-class prototype" in prose when clarity helps. |
| hierarchical prototype | post-hoc prototype using main-class and subtype centers | dual prototype, subtype prototype | Do not call it the most robust method. |
| raw Softmax | uncalibrated Softmax inference from the frozen classifier | softmax, baseline Softmax | Use "raw Softmax" for the primary comparator. |
| calibrated Softmax | temperature-calibrated Softmax variant | calibrated model | Keep distinct from raw Softmax. |
| fused | fusion ablation variant | dual-prototype fusion | State that it is an ablation, not a main innovation. |
| DEMAND simulated additive noise | additive environmental-noise simulation using DEMAND recordings | real-farm noise, external validation | Never describe as real-farm validation. |
| 0 dB active-event SNR | extreme stress condition measured over the valid active event region | no noise, 0 dB clip SNR | Always include "active-event" and "extreme stress condition". |
| AURC/AUGRC | risk-coverage and generalized risk-coverage areas | uncertainty improvement | Use only with corrected values from saved prediction CSVs. |

## Full-paper evidence chain

1. Field-scale need: non-contact pig vocalization recognition for livestock monitoring.
2. Unresolved bottleneck: clean performance and noise robustness depend on event context, label granularity and inference stability, not just on a larger feature stack.
3. Proposed move: select 2 s context, add bounded subtype supervision, and evaluate frozen post-hoc prototypes.
4. Decisive evidence: Table 3 paired duration statistics; Table 4 hierarchy weights; Table 5 clean prototype summary; Tables 6-9 noise, paired statistics, class failures and selective prediction.
5. Broader implication: a reproducible engineering workflow for clean mainline selection and simulated-noise inference.
6. Boundary: no real-farm external validation, no open-set claim, no reliable 0 dB cough recognition, no universal uncertainty improvement.

## Paragraph map

### Title

Job: name the object, application condition and two technical mechanisms without unsupported novelty words.

### Abstract

1. Context: pig vocalization recognition requires event context and robustness.
2. Gap: feature stacking and raw Softmax do not resolve all clean/noise issues.
3. Approach: 2 s Log-Mel CRNN, hierarchical auxiliary supervision, post-hoc prototypes.
4. Key clean result: 2 s vs 1 s paired gain.
5. Key hierarchy result: semantic/boundary contribution, nonsignificant main gain.
6. Key noise result: main prototype improves ALL_NOISY Macro-F1.
7. Boundary: simulated noise only, 0 dB stress condition, cough and uncertainty limits.

### Introduction

1. Field stake and closed-set pig vocalization task.
2. Story 1 setup: pig events are not instantaneous; 1 s can truncate context; 3 s can add background.
3. Failed route: MCTAFD, PCEN, spectral-gate denoising, SpecAugment, BiLSTM direction and attention pooling did not become the clean mainline.
4. Story 2 setup: four labels are coarse; subtype semantics explain cough and stress_vocal internal structure.
5. Story 3 setup: Softmax boundaries can drift under simulated additive noise; prototypes test embedding-center inference.
6. Contribution paragraph with explicit boundaries.

### Related Work

1. Pig vocalization and livestock acoustic monitoring literature: placeholder until verified citations are added.
2. CRNN and Log-Mel audio classification: cite CRNN and librosa entries.
3. Prototypical inference: cite prototypical networks while stating this is post-hoc inference, not new prototype learning.
4. Selective prediction and risk-coverage: cite selective classification and fd-shifts implementation.

### Materials and Methods

1. Task definition and label hierarchy.
2. Clean mainline model and duration rationale.
3. Hierarchical auxiliary supervision objective.
4. Post-hoc prototype construction and calibration boundaries.
5. DEMAND simulated additive-noise mixing and frozen zero-shot rule.

### Experimental Protocol

1. Fold-seed protocol and matched comparisons.
2. Train/validation/test roles.
3. Leakage and SHA provenance.
4. Table map and appendix map.

### Results

1. Story 1: problem, attempts, failure evidence, convergence to 2 s, final mechanism, boundary.
2. Story 2: problem, auxiliary design, lambda results, nonsignificant gain, semantic/boundary interpretation.
3. Story 3: Softmax drift problem, main vs hierarchical prototypes, clean/noise contrast, 0 dB cough failure, selective prediction boundary.

### Discussion

1. Interpret the three narrowing decisions.
2. Explain why hierarchical prototype helps clean semantics but main prototype helps noise robustness.
3. Bound the uncertainty claim using AURC/AUGRC.
4. Re-state simulated-noise-only scope.

### Limitations

1. DEMAND simulated additive noise is not real-farm validation.
2. 0 dB active-event SNR is extreme stress, not no noise.
3. Cough is unreliable at 0 dB.
4. Prototype does not comprehensively improve uncertainty.
5. Hierarchical prototype is weaker than main prototype under noise.
6. Literature, ethics and release details remain incomplete.

### Conclusion

1. Contribution: evidence-supported clean mainline and bounded noise inference.
2. Decisive evidence: 2 s paired gain and main prototype ALL_NOISY gain.
3. Implication: practical path for reproducible pig sound classification.
4. Boundary and future work: real-farm validation, open-set recognition, cough robustness.

## Table and appendix use

| Evidence block | Use in manuscript | Allowed claim |
|---|---|---|
| `table2_clean_baseline_and_architecture_ablations.csv` | Introduction and Results Story 1 | Complex clean-audio variants did not replace 2 s Log-Mel CRNN as mainline. |
| `table3_duration_comparison.csv` | Abstract, Results Story 1, Conclusion | 2 s significantly improves over 1 s in matched 25-run comparison. |
| `table4_hierarchical_aux_weight_ablation.csv` | Results Story 2 | lambda=1.0 highest mean; lambda=0.5 lowest standard deviation and balanced setting. |
| `clean_25_run_prototype_summary.csv` | Results Story 2 and Story 3 | Clean hierarchical prototype has highest mean Macro-F1. |
| `clean_25_run_prototype_paired_stats.csv` | Results Story 2 | Clean hierarchical gain over raw Softmax is not statistically significant. |
| `table5_clean_softmax_main_prototype_hierarchical_prototype.csv` | Results Story 3 | Clean prototype comparison across raw, calibrated, main, hierarchical and fused variants. |
| `table6_simulated_noise_robustness_by_stratum.csv` | Results Story 3 | Main prototype is strongest aggregate simulated-noise method in Macro-F1. |
| `table7_paired_statistics.csv` | Abstract and Results Story 3 | ALL_NOISY main prototype gain is significant; hierarchical prototype is weaker than main prototype under noise. |
| `table8_per_class_noise_results_and_confusion_directions.csv` | Results Story 3 and Limitations | Cough remains weak; feeding/stress_vocal confusion remains a boundary. |
| `table9_selective_prediction_metrics_corrected_augrc.csv` | Discussion and Limitations | Raw Softmax ranks better by AURC/AUGRC; prototype only slightly helps frozen-threshold risk. |
| `noise_25_run_cross_seed_draw_audit.csv` | Experimental Protocol | Deterministic cross-seed noise draw audit passed. |
| `noise_25_run_per_class_summary.csv` | Limitations | 0 dB cough recall is zero for STRAFFIC/TBUS across methods and very low for DWASHING. |

## Missing submission information

- Target journal and exact formatting rules.
- Structured vs unstructured abstract requirement.
- Word limits for abstract, main text and references.
- Complete verified pig-vocalization and livestock-acoustic monitoring citations.
- Ethics approval authority and approval number or exemption statement.
- Original pig-audio data availability and redistribution permissions.
- Code release URL, version tag, archive DOI and checkpoint release policy.
- Whether figures/tables should be adapted to a specific journal template.

## Reviewer-risk audit

| Risk | Current status | Required handling |
|---|---|---|
| Overclaiming hierarchy | Controlled | Manuscript states nonsignificant gain and semantic/boundary role. |
| Treating DEMAND as real farm | Controlled | Manuscript says simulated additive noise only. |
| Misreading 0 dB | Controlled | Manuscript says 0 dB active-event SNR is an extreme stress condition. |
| Prototype uncertainty claim | Controlled | Manuscript states raw Softmax has better AURC/AUGRC. |
| Cough robustness | Controlled | Manuscript states cough is unreliable at 0 dB. |
| Missing livestock citations | Open | Add verified references before submission. |
| Ethics/data availability | Open | Author must supply original recording approvals and data-sharing status. |
