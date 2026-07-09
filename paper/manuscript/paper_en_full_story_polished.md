# Pig Vocalisation Recognition under Simulated Environmental Noise Using Duration-Aware Hierarchical Representations and Post-Hoc Prototype Inference

## Abstract

Automatic pig vocalisation recognition is a practical sensing problem for livestock monitoring. Its reliability depends on more than choosing a stronger acoustic front end or a larger classifier. Pig vocal events can include short build-up, sustained acoustic texture, and post-event context. Short clips may truncate useful cues, whereas long clips may admit irrelevant background.

This study restructures a closed-set pig-sound recognition pipeline around three evidence-led decisions. These are duration-aware context selection, hierarchical auxiliary supervision, and post-hoc prototype inference. The task covers four main classes: cough, calm_grunt, feeding, and stress_vocal. Six auxiliary subtypes are used only for hierarchical supervision.

Clean experiments used class-capped cap3x training expansion, Log-Mel features, and a convolutional recurrent neural network (CRNN) backbone. The evaluation used 5 folds x 5 seeds. The main clean-data gain came from 2 s context. Macro-F1 increased from 0.9226 for 1 s Log-Mel to 0.9472 for 2 s Log-Mel. The paired mean delta was 0.0246, with Wilcoxon p=0.000162.

More complex clean-audio modifications did not provide a more stable mainline in the locked result package. These included MCTAFD variants, PCEN, spectral-gate denoising, SpecAugment, and attention pooling. Hierarchical auxiliary supervision increased the clean mean Macro-F1 to 0.9515 at lambda=1.0. Lambda=0.5 gave the lowest cross-run standard deviation among the tested hierarchical settings.

This hierarchical gain over the 2 s baseline was not statistically significant. It is therefore interpreted as a fine-grained semantic and boundary-analysis contribution, not as a strong performance claim. Post-hoc prototypes were then evaluated without noisy retraining. Clean hierarchical prototypes reached the highest Macro-F1 among the prototype variants (0.9540).

Under DEMAND simulated additive noise, the main-class prototype was the most stable method. Under ALL_NOISY conditions, it achieved Macro-F1 0.6230, compared with 0.6173 for raw Softmax. The mean delta was 0.005755, with 95% bootstrap CI [0.001606, 0.010894] and Wilcoxon p=0.010511.

These results support main-class prototype inference as a modest simulated-noise robustness mechanism under the tested protocol. The study remains bounded. DEMAND is simulated additive noise, not real-farm external validation. The 0 dB active-event SNR setting is an extreme stress condition. Cough recognition at 0 dB is unreliable. Prototypes do not provide a universal uncertainty improvement. Hierarchical prototypes are weaker than main prototypes under noise.

## Keywords

Pig vocalisation recognition; Log-Mel; CRNN; hierarchical supervision; acoustic prototype; simulated additive noise; DEMAND; selective prediction

## Introduction

Acoustic monitoring offers a non-contact route for observing animal health and behaviour in intensive pig production. Pig sounds are not always isolated impulses. Cough, calm grunting, feeding-related sounds, and stress vocalisations can include onset transients, repeated texture, and short contextual patterns.

A system that treats each event as an instantaneous frame may discard useful temporal evidence. A system that extends context indiscriminately may absorb irrelevant environmental background. The engineering question is therefore not only which model is expressive enough. It is also which event window gives the model a fair representation of the vocalisation.

The current result package shows why this question matters. A 1 s Log-Mel CRNN baseline achieved Macro-F1 0.9226. The 2 s Log-Mel CRNN achieved 0.9472 across 25 fold-seed runs. The paired duration comparison gave a mean Macro-F1 delta of 0.0246 and Wilcoxon p=0.000162.

A 3 s Log-Mel setting reached 0.9434 in the available 15-run comparison. It did not surpass the 2 s mainline. The same ablation package did not support feature stacking as the primary direction. Tabled MCTAFD, PCEN, spectral-gate denoising, SpecAugment, and attention-pooling variants all remained below the locked 2 s Log-Mel CRNN mean.

The project record also retains BiLSTM replacement as a completed negative direction. It is not the validated clean-audio mainline. These observations narrow the initial contribution. The clean-data advance is an evidence-based shift from adding components to identifying a suitable event context.

The next challenge is semantic granularity. The four main labels are useful for deployment-facing reporting, but they compress acoustically meaningful distinctions. Cough contains dry_cough and abdominal_cough. Stress_vocal contains frightened_stress and anxious_stress. Calm_grunt and feeding also define subtype targets.

The feeding/stress_vocal boundary remained an important confusion direction in the result package. Hierarchical auxiliary supervision addresses this issue by linking the four-class main task to a six-subtype auxiliary task. The loss has the form main loss + lambda x auxiliary loss. This design asks the representation to preserve subtype-level acoustic structure without replacing the main task.

The locked results require careful wording. Lambda=1.0 achieved the highest mean Macro-F1 among the tested auxiliary weights. Lambda=0.5 gave the lowest standard deviation and is the balanced setting. The incremental hierarchical gain over the 2 s baseline was not statistically significant. The contribution is therefore interpretability and boundary shaping, not decisive performance improvement.

Environmental interference is a further challenge. Raw Softmax inference reflects discriminative decision boundaries learned under clean training conditions. Additive noise may shift embeddings towards regions where those boundaries are less reliable. Post-hoc prototype inference provides a different frozen-test decision rule.

In this rule, train-split embeddings define class centres. Validation data calibrate temperatures and thresholds. Test data are used only for evaluation. This study evaluates two prototype forms. The main prototype uses only the four main-class centres. The hierarchical prototype also incorporates the six subtype prototypes.

Clean results favour the hierarchical prototype, consistent with its semantic role. Simulated DEMAND noise favours the simpler main-class prototype. This difference is central to the paper's story. Fine-grained prototypes are useful for clean semantic explanation. Main-class prototypes are better suited to the medium-noise robustness setting evaluated here.

This paper therefore makes three bounded contributions. It identifies 2 s Log-Mel CRNN as the clean closed-set mainline through matched fold-seed evidence. It uses hierarchical auxiliary supervision to inject subtype semantics into the representation. It also avoids a significance claim that the data do not support.

The study then evaluates post-hoc main and hierarchical prototypes under a frozen DEMAND simulated-noise protocol. The results indicate that main-class prototypes provide a small but statistically supported aggregate Macro-F1 gain under simulated additive noise. The scope is deliberately limited. The study does not report real-farm external validation, unknown-class rejection, noise-aware retraining, or reliable cough recognition at 0 dB active-event SNR.

## Related Work

Automatic animal-sound recognition combines bioacoustic sensing with supervised audio classification. Prior work has used pig-vocalisation analysis to characterise call valence and production context, to support animal-welfare monitoring, and to study pig-cough or sick-cough recognition [@briefer2022pigcalls; @mcloughlin2019automated; @coutant2024scoping; @ferrari2008cough; @exadaktylos2008cough; @yin2021pigcoughcnn; @shen2022pigcoughfusion]. Recent smart-farm studies also address abnormal pig vocalisations and noisy farm-environment classification [@xie2024abnormalpig; @chung2025pvmc]. The present manuscript should not claim originality by ignoring that literature. Its distinction is instead framed around an evidence-controlled engineering sequence. This sequence covers duration selection, hierarchical subtype supervision, and frozen prototype inference under simulated additive noise.

CRNN architectures have been widely used for audio classification. Convolutional layers can extract local time-frequency patterns, while recurrent layers summarise temporal dependencies [@choi2017crnn]. In this work, the CRNN is not presented as a new architecture. It provides a stable backbone for testing which input duration and supervision structure are supported by the pig-vocalisation evidence.

Log-Mel features were extracted using standard audio-analysis tooling [@mcfee2015librosa]. The clean ablations showed that increasing front-end complexity did not outperform the 2 s Log-Mel mainline. This framing keeps the method claim conservative. The paper tests a controlled engineering sequence rather than presenting the backbone or feature extraction as new.

Prototype-based inference is related to prototypical networks. In that framework, class representations are expressed as centres in an embedding space [@snell2017prototypical]. This paper does not claim to introduce prototype learning. The prototypes here are post-hoc, fold-seed-specific artefacts.

These artefacts are built from frozen clean train embeddings. They are calibrated on validation data and evaluated on held-out test predictions. This distinction matters. The contribution is not a new trainable prototype architecture. It is a leakage-controlled inference layer that tests whether clean embedding centres can stabilise decisions under simulated additive noise.

Selective prediction and risk-coverage analysis evaluate whether confidence or distance scores can identify unreliable predictions [@geifman2017selective]. The corrected AUGRC calculations in this paper follow the generalised-risk convention used by the fd-shifts implementation [@fdshifts_riskcoverage]. The results do not support a broad claim that prototypes improve uncertainty ranking.

Under ALL_NOISY conditions, raw Softmax had lower AURC/AUGRC than the prototype variants. Prototype benefits are therefore limited to aggregate Macro-F1 and frozen-threshold selective-risk behaviour. They should not be described as universal uncertainty calibration.

## Materials and Methods

### Task definition and label structure

The task is closed-set classification of pig vocal events into four main classes: cough, calm_grunt, feeding, and stress_vocal. A six-subtype auxiliary label space was used for hierarchical supervision. The subtypes were dry_cough, abdominal_cough, calm_grunt, feeding, frightened_stress, and anxious_stress.

The auxiliary task does not change the deployment-facing main label set. Instead, it provides finer acoustic constraints during representation learning. It also enables hierarchical prototype analysis after training.

### Clean mainline model

The clean mainline uses class-capped cap3x training expansion, 2 s Log-Mel input features, and a CRNN backbone. The duration choice was treated as an experimental variable rather than a fixed assumption. The 1 s condition tests whether a compact event slice is sufficient. The 2 s condition tests whether short context improves recognition. The 3 s condition tests whether further context continues to help or begins to add redundant background.

All clean comparisons were interpreted through matched fold-seed evidence where available. This constraint keeps the duration conclusion tied to the validated result package. It also avoids treating unmatched runs as direct evidence for small performance differences.

### Hierarchical auxiliary supervision

Hierarchical supervision adds a subtype prediction head to the main-class classifier. The training objective is:

`L = L_main + lambda * L_aux`.

The main loss optimises the four-class decision task. The auxiliary loss encourages the embedding to preserve subtype structure. Three auxiliary weights were evaluated in the locked table: lambda=0.2, lambda=0.5, and lambda=1.0.

The manuscript uses lambda=0.5 as the balanced setting because it has the lowest standard deviation across runs. It also reports that lambda=1.0 has the highest mean Macro-F1. This separation prevents a small nonsignificant mean improvement from becoming an overstated performance claim.

### Post-hoc prototype inference

Prototype inference was applied after training. For each fold-seed run, train-split embeddings were used to compute class centres. The main prototype method uses four centres, one per main class. The hierarchical prototype method additionally uses six subtype centres.

Validation predictions were used to calibrate temperatures, thresholds, and fusion parameters where relevant. The frozen test split was used only for evaluation. No prototype, threshold, or calibration artefact was constructed by mixing folds or seeds.

The raw Softmax decision rule and the prototype decision rule answer different questions. Raw Softmax evaluates the discriminative boundary learned by the classifier. Prototype inference evaluates whether a test embedding remains close to class-level acoustic centres learned from the training split.

Under clean conditions, subtype prototypes can support semantic structure. Under additive noise, subtype-level centres may be more fragile than coarser main-class centres. This hypothesis is tested under a frozen noisy-evaluation protocol.

### Simulated additive-noise protocol

Noise robustness was evaluated with DEMAND environmental recordings [@demand2018]. The selected environments were DWASHING, TBUS, and STRAFFIC. The protocol mixed deterministic DEMAND segments into the 2 s pig-vocalisation waveform. Mixing used active-event SNRs of 20, 10, and 0 dB.

The 0 dB condition is described throughout as an extreme active-event SNR stress condition. It is not a no-noise setting. It is also not presented as an ordinary deployment scenario.

The noisy evaluation was zero-shot and frozen. It reused the clean checkpoint, train-only clean prototypes, validation-only clean calibration, frozen rejection thresholds, and frozen fusion parameters. No noisy validation recalibration or noisy test tuning was allowed.

The deterministic noise selection key used SHA256. It included fold, normalised clean path, clean MD5, DEMAND environment identity, noise-file SHA256, global noise seed, and repeat. It deliberately excluded model seed and SNR. A fixed clean sample and environment therefore reused the same noise segment across seeds and SNR levels. Only gain changed across SNR.

## Experimental Protocol

The clean and simulated-noise result package uses 5 folds x 5 seeds for final paper-facing summaries. Fold-seed pairing is preserved for model comparisons. The locked protocol enforces exact-path, source_id, and MD5 disjointness. Manifest and artefact SHA checks are part of the provenance record.

The clean closed-set manifests combine third-party or public pig-audio sources rather than newly collected animal recordings. The `dry_cough` and `abdominal_cough` subtypes come from the Smart Farm Korea / data.go.kr pig-cough voice dataset [@smartfarmkorea_pig_cough_voice]. The `calm_grunt`, `feeding`, `frightened_stress`, and `anxious_stress` subtypes derive from the figshare Sow Call Dataset [@sow_call_dataset_2021], with local labels mapped from source filename suffixes. The unresolved Kaggle-like scream/cough source is not used in the current cap3x mainline manifests.

Training data may build model parameters and prototypes. Validation data may calibrate thresholds, temperatures, and fusion weights. Test data are evaluation only. This separation is central to the interpretation of prototype and selective-prediction results.

The principal clean tables are organised as follows. Table 1 defines the class set, subtype set, backbone protocol, prototype protocol, noise protocol, leakage controls, and absence of external validation. Table 2 compares the clean baseline with architecture and front-end ablations. Table 3 isolates the duration comparison. Table 4 reports hierarchical auxiliary weights. Table 5 compares clean raw Softmax, calibrated Softmax, main prototype, hierarchical prototype, and fused variants.

The simulated-noise tables separate aggregate robustness, paired statistics, per-class failure modes, and selective prediction. Table 6 reports Macro-F1 and calibration metrics by noise stratum. Table 7 reports paired statistics. Table 8 reports per-class noise results and confusion directions. Table 9 reports corrected selective prediction metrics, including AURC and AUGRC.

Appendix CSVs provide additional provenance and failure-mode detail. They include per-environment/SNR summaries, fold-level deltas, per-class 0 dB behaviour, cross-seed draw audits, and calibration-distribution records. These files support reproducibility without changing the train, validation, or test roles.

## Results

### Story 1: 2 s context, not feature stacking, defines the clean mainline

The initial question was whether pig vocal events could be represented reliably by a 1 s input. The 1 s Log-Mel CRNN reached mean Macro-F1 0.9226 across 25 runs. This was a reasonable baseline, but it exposed a practical limitation.

Feeding and stress_vocal events are not always captured by a single isolated instant. Cough-related context can also be short but temporally structured. The next attempt was to enrich the front end and architecture. The locked ablation table includes MCTAFD variants, PCEN, spectral-gate denoising, SpecAugment, and attention pooling.

None provided a more stable clean mainline than 2 s Log-Mel CRNN. Cap3x logmel_dur2 reached 0.9472. In the available locked comparisons, cap3x mctafd_no_se_gated reached 0.9208, logmel_pcen reached 0.8916, logmel_sg reached 0.9093, logmel_specaug reached 0.9198, and logmel_dur2_attn reached 0.9431.

This failure evidence changed the design logic. The project did not continue to stack more handcrafted or augmented features as the primary clean-audio path. Instead, it treated event duration as the key design variable.

The 2 s Log-Mel CRNN improved over the 1 s Log-Mel baseline by a paired mean Macro-F1 delta of 0.0246. The Wilcoxon p-value was 0.000162. A 3 s Log-Mel model reached 0.9434 in the available 15-run comparison, below the 2 s mean.

The final clean mechanism is therefore duration-aware but restrained. The model receives enough local acoustic context to represent pig vocal events. It does not expand to a longer window that may admit redundant background.

The boundary of this result is also clear. The data support 2 s as the validated clean mainline for this closed-set protocol. They do not establish 2 s as a universal duration for all pig-acoustic tasks. The 3 s comparison has fewer runs than the paired 1 s and 2 s comparison. It should therefore be interpreted as support for the present package, not as evidence that longer windows can never help.

### Story 2: hierarchical supervision adds subtype semantics, not a statistically significant main-performance claim

The next question was whether four main labels were too coarse for representation learning. The main label cough groups dry_cough and abdominal_cough. Stress_vocal groups frightened_stress and anxious_stress. Feeding and stress_vocal also form a difficult boundary in the confusion summaries.

A pure four-class objective can learn a deployment-facing classifier. It does not explicitly encode these finer acoustic distinctions. Hierarchical auxiliary supervision was therefore tested as a representation constraint.

The main task remained the four-class classification problem. A subtype head predicted six auxiliary labels. The loss combined the two objectives as main loss + lambda x auxiliary loss.

In the locked auxiliary-weight table, lambda=0.2, lambda=0.5, and lambda=1.0 gave mean Macro-F1 values of 0.9505, 0.9510, and 0.9515. Lambda=1.0 therefore had the highest mean. Lambda=0.5 had the lowest cross-run standard deviation, 0.0124. It is the balanced setting used for the prototype and noise studies.

The failure evidence is important. The hierarchical gain over the 2 s baseline did not reach statistical significance. The paper should therefore not describe hierarchical supervision as a significant clean-performance improvement. Its role is more specific. It introduces fine-grained acoustic semantics and supports boundary analysis.

The clean prototype summary is consistent with this interpretation. Raw Softmax at lambda=0.5 achieved Macro-F1 0.9510. The main prototype achieved 0.9519, and the hierarchical prototype achieved 0.9540. However, hierarchical prototype minus raw Softmax had mean delta 0.002937. The 95% CI was [-0.000478, 0.007294], with Wilcoxon p=0.058253. This result is close but remains nonsignificant under the locked criterion.

The final mechanism is therefore not that hierarchy wins everywhere. It is a representation and interpretation device. Hierarchical labels help organise the acoustic space and support clean semantic analysis. The manuscript must keep its performance claim bounded.

### Story 3: post-hoc prototypes separate clean semantic explanation from simulated-noise robustness

The third question was how the clean model behaves when environmental noise shifts the input distribution. Raw Softmax is tied to the discriminative boundary learned during clean training. Under simulated additive noise, this boundary can drift. A noisy event embedding may move in directions not seen during training.

Prototype inference tests a complementary decision rule. It asks whether an embedding remains near train-derived class centres. Two prototype mechanisms were evaluated. The main prototype uses four main-class centres. The hierarchical prototype adds six subtype centres.

Clean results favoured the hierarchical prototype. It achieved mean Macro-F1 0.9540, compared with 0.9519 for main prototype and 0.9510 for raw Softmax. This supports the semantic role of subtype prototypes under clean conditions.

The noise results reversed the practical preference. Under ALL_NOISY DEMAND simulated additive noise, raw Softmax achieved Macro-F1 0.6173. The main prototype achieved 0.6230, and the hierarchical prototype achieved 0.6200.

Main prototype minus raw Softmax had mean delta 0.005755. The 95% bootstrap CI was [0.001606, 0.010894], with Wilcoxon p=0.010511. Hierarchical prototype minus raw Softmax had mean delta 0.002710 with p=0.312333. Hierarchical prototype was also significantly below main prototype under ALL_NOISY conditions, with mean delta -0.003045 and p=0.000376.

This evidence defines the final mechanism. The main prototype is more robust under the evaluated simulated noise. It uses coarser class centres that are less sensitive to subtype-level perturbations. The hierarchical prototype remains useful for clean semantic explanation. Under noise, it can over-fragment the embedding space.

The moderate-noise stratum supports the same direction. Main prototype achieved Macro-F1 0.6534, compared with 0.6472 for raw Softmax and 0.6522 for hierarchical prototype. Prototype minus raw Softmax had p=0.001673.

In the extreme-stress stratum, main prototype remained numerically above raw Softmax, 0.5624 versus 0.5574. The paired p-value was 0.560171. This condition should therefore not be described as a significant robustness gain.

Per-class results expose the main boundary. Under ALL_NOISY conditions, cough F1 remained very low for all methods. The values were 0.0867 for raw Softmax, 0.0974 for main prototype, and 0.0957 for hierarchical prototype.

Under 0 dB active-event SNR, cough recall was zero for STRAFFIC and TBUS across all methods. It was below 0.009 in DWASHING. Thus, the 0 dB condition is an extreme stress test. Cough recognition in that condition is unreliable.

Feeding and stress_vocal also remained entangled. Main prototype improved several aggregate boundary metrics relative to raw Softmax under ALL_NOISY conditions. This improvement should be interpreted at the aggregate level, not as a complete per-class solution.

Uncertainty results place another boundary on the claim. Prototypes did not comprehensively improve uncertainty or ranking. Under ALL_NOISY conditions, raw Softmax had lower AURC/AUGRC, 0.1775/0.1239, than main prototype, 0.2109/0.1394. It also outperformed hierarchical prototype, 0.2122/0.1403, on these ranking metrics.

Prototype inference showed a slight frozen-threshold selective-risk improvement under ALL_NOISY conditions. Mean frozen-threshold selective risk was 0.2988 for main prototype and 0.3024 for raw Softmax. This result should not be generalised into an uncertainty-calibration claim. The fused variant is retained as an ablation and sanity check, not as the main innovation.

## Discussion

The main lesson is that practical pig-vocalisation recognition benefits from a sequence of narrowing decisions. The initial narrowing decision is temporal. Before adding complex acoustic front ends, the system needs an event window that captures the relevant vocal context. In this result package, 2 s provided that window.

The second narrowing decision is semantic. Subtype labels can enrich the representation, but their contribution must be framed carefully. The main-performance gain over the 2 s baseline was not statistically significant. Hierarchical supervision is therefore best described as semantic and boundary-oriented.

The third narrowing decision is inferential. Clean subtype prototypes support semantic explanation. Main-class prototypes are better suited to simulated-noise robustness under the tested DEMAND protocol.

This interpretation also explains why the noise results differ from the clean prototype results. Hierarchical prototypes introduce more detailed acoustic centres. Under clean conditions, this detail can help organise the embedding space. Under additive noise, the same detail may become brittle. Noise can move samples away from subtype-specific centres even when the main-class identity remains recoverable.

The main prototype acts as a coarser attractor. It therefore provides the most stable aggregate noise gain in the evaluated DEMAND protocol. This interpretation is mechanistic but remains bounded by the available experiments.

The uncertainty findings prevent a stronger claim. If prototypes had improved Macro-F1, calibration metrics, and risk-coverage ranking together, the method could be framed as a broader reliability improvement. The locked tables do not support that framing.

Raw Softmax ranked better by AURC/AUGRC under the ALL_NOISY, MODERATE_NOISE, and EXTREME_STRESS aggregate rows. Prototype gains appeared mainly in Macro-F1 and frozen-threshold selective risk. The manuscript should therefore describe prototype inference as a robustness-oriented decision rule, not as a universal uncertainty solution.

The simulated-noise protocol is useful but incomplete. DEMAND provides public environmental recordings and deterministic provenance. However, additive DEMAND mixing is not real pig-farm external validation. Real barns include reverberation, overlapping animals, sensor-placement changes, husbandry equipment, and label-distribution shifts. These factors are not captured by the present protocol.

The 0 dB active-event SNR setting is particularly severe and should be interpreted as stress testing. The cough collapse under 0 dB makes this limitation central rather than incidental. Real-farm external validation was not performed. The current results should therefore be treated as controlled simulated-noise evidence.

## Limitations

One limitation is that the noise evaluation uses simulated additive DEMAND noise only. It does not establish external validity in real pig farms. Another is that the task is closed-set and does not address unknown sounds or open-set rejection.

Third, cough recognition under 0 dB active-event SNR is unreliable. The appendix summary reports zero cough recall in STRAFFIC and TBUS for all methods. Fourth, hierarchical auxiliary supervision improves the clean mean but does not significantly outperform the 2 s baseline.

Clean hierarchical prototype improvement over raw Softmax also remains nonsignificant at p=0.058253. Fifth, prototype inference does not comprehensively improve uncertainty. Raw Softmax has better AURC/AUGRC ranking in the aggregate noise summaries.

Sixth, the fused method is an ablation and should not be promoted as a core contribution. Finally, target-journal metadata, full source-side ethics provenance, and data-sharing permissions for original pig audio remain submission dependencies rather than empirical results.

## Conclusion

This study restructures pig-vocalisation recognition around three bounded findings. A 2 s Log-Mel CRNN is the best supported clean closed-set mainline in the locked result package. It significantly improves over 1 s input in the matched paired comparison.

Hierarchical auxiliary supervision adds subtype semantics and supports boundary analysis. Its incremental clean gain is not statistically significant and should not be overstated. Post-hoc prototype inference separates two roles. Hierarchical prototypes are most useful for clean semantic explanation. Main-class prototypes provide the most reliable aggregate simulated-noise robustness under the frozen DEMAND protocol.

The work offers a reproducible path towards noise-aware pig-vocalisation inference under controlled simulated conditions. It leaves real-farm validation, robust cough recognition under severe noise, and open-set deployment as necessary next steps.

## Data Availability

Processed aggregate tables and figure source data supporting the results are available in the project repository and/or Supplementary Data associated with this article. These materials include the paper-ready summary tables, paired statistics, figure source-data mappings, command records, cross-validation manifests, and hash-based leakage and provenance audits. Raw third-party pig audio is not redistributed. Smart Farm Korea / data.go.kr pig-cough recordings should be obtained from the original provider [@smartfarmkorea_pig_cough_voice]. Sow Call Dataset recordings should be obtained from figshare via DOI 10.6084/m9.figshare.16940389 [@sow_call_dataset_2021]. DEMAND environmental-noise recordings should be obtained from Zenodo via DOI 10.5281/zenodo.1227121 [@demand2018]. The unresolved Kaggle-like scream/cough source is excluded from the current main results and is not part of the submitted data description.

## Code Availability

The code and reproducibility package will be made available at [GitHub release URL]. The release should include the scripts, command records, aggregate tables, figure source-data mappings, manifests, and leakage/provenance audit files needed to reproduce the reported analyses from the permitted inputs. No code DOI is claimed here. Model checkpoints, prototype artifacts, and raw third-party audio are not included unless released separately with provider-compatible terms, file hashes, and artifact-to-result mapping.

## Ethics Statement

No new animal experiment, animal intervention, or prospective farm recording was conducted for this manuscript. The study is a computational reuse of public or third-party pig-audio recordings and public DEMAND environmental noise. Original source-side permissions, repository terms, and any animal-ethics approvals remain with the original data providers. Source-specific access and redistribution restrictions are described in the Data Availability statement.

## References
