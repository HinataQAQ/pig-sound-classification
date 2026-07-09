# Pig Vocalisation Recognition under Simulated Environmental Noise Using Duration-Aware Hierarchical Representations and Post-hoc Prototype Inference

## Abstract

Automatic pig-vocalisation recognition is a practical sensing problem for precision livestock monitoring. Its robustness depends not only on the classifier, but also on whether the input window captures the acoustic event without admitting excessive background. This study examines a leakage-controlled pig-sound recognition pipeline using duration-aware Log-Mel CRNN modelling, hierarchical subtype supervision, and post-hoc prototype inference.

The closed-set task includes four main classes: cough, calm_grunt, feeding, and stress_vocal. Six auxiliary subtypes are used to encourage a more structured representation: dry_cough, abdominal_cough, calm_grunt, feeding, frightened_stress, and anxious_stress. Clean experiments used class-capped cap3x training expansion, 5 folds, and 5 random seeds. The main clean-data gain came from context duration. A 2 s Log-Mel CRNN improved Macro-F1 from 0.9226 for 1 s input to 0.9472, with a paired mean delta of 0.0246 and Wilcoxon p=0.000162. A 3 s condition and several more complex front-end or pooling variants did not provide a more stable clean mainline.

Hierarchical auxiliary supervision increased the clean mean Macro-F1 to 0.9515 at lambda=1.0. The lambda=0.5 setting gave the lowest cross-run standard deviation among the tested hierarchical settings and was used for the prototype experiments. The gain over the 2 s baseline was not statistically significant. We therefore interpret hierarchical supervision as a fine-grained representation and boundary-analysis mechanism, rather than a decisive clean-performance improvement.

Post-hoc prototypes were evaluated without retraining the backbone. Clean hierarchical prototypes reached the highest prototype-variant Macro-F1, 0.9540. Under zero-shot DEMAND simulated additive noise, the main-class prototype was more robust than both raw Softmax and hierarchical prototypes. Under ALL_NOISY conditions, the main prototype achieved Macro-F1 0.6230, compared with 0.6173 for raw Softmax. The run-level paired mean delta was 0.005755, with 95% bootstrap CI [0.001606, 0.010894] and Wilcoxon p=0.010511. Fold-level evidence was strongest under moderate 20 and 10 dB active-event SNR.

These results support main-class prototype inference as a modest zero-shot robustness mechanism under simulated additive noise. The conclusions remain bounded. DEMAND mixing is not real-farm external validation. The 0 dB active-event SNR condition is an extreme stress test, and cough recognition is unreliable under that setting. Prototypes also do not provide a universal improvement in uncertainty ranking.

## Keywords

Pig vocalisation recognition; Log-Mel; CRNN; hierarchical supervision; acoustic prototype; simulated additive noise; DEMAND; selective prediction

## 1. Introduction

Acoustic monitoring offers a non-contact route for observing animal health and behaviour in intensive pig production. Pig sounds can indicate respiratory problems, affective state, feeding activity, or stress-related behaviour. These signals are attractive for engineering deployment because microphones can monitor animals without direct attachment.

However, pig vocalisations are not always isolated impulses. Cough, calm grunting, feeding-related sounds, and stress vocalisations can include onset transients, repeated acoustic texture, and short contextual patterns. A system that treats each event as a very short frame may discard useful temporal evidence. A system that extends context indiscriminately may instead absorb irrelevant environmental background. The first engineering question is therefore not only which model is more expressive. It is also which event window gives the model a representative acoustic context.

The present result package shows that this question matters. A 1 s Log-Mel CRNN baseline achieved Macro-F1 0.9226. A 2 s Log-Mel CRNN reached 0.9472 across 25 fold-seed runs. The paired duration comparison yielded a mean Macro-F1 delta of 0.0246 and Wilcoxon p=0.000162. A 3 s Log-Mel setting reached 0.9434 in the available 15-run comparison and did not exceed the 2 s mainline. The ablation package also did not support feature stacking as the primary direction. Tested MCTAFD, PCEN, spectral-gate denoising, SpecAugment, BiLSTM, and attention-pooling variants did not provide a more stable clean mainline than the 2 s Log-Mel CRNN.

The second challenge is semantic granularity. The four deployment-facing main labels are useful, but they compress acoustically meaningful distinctions. Cough includes dry_cough and abdominal_cough. Stress_vocal includes frightened_stress and anxious_stress. Feeding and stress_vocal also remain an important confusion direction. Hierarchical auxiliary supervision addresses this issue by linking the four-class main task to a six-subtype auxiliary task. The objective preserves the deployment-facing labels while encouraging the shared representation to retain subtype-level structure.

The third challenge is environmental interference. Raw Softmax inference reflects a discriminative boundary learned under clean training conditions. Additive noise may shift embeddings towards less reliable regions of that boundary. Post-hoc prototype inference evaluates a complementary rule: whether a test embedding remains close to train-derived acoustic class centres. This study compares raw Softmax, calibrated Softmax, main-class prototypes, hierarchical prototypes, and validation-selected fusion under a strict train/validation/test separation.

The clean results favour hierarchical prototypes, consistent with their semantic role. The simulated-noise results favour the simpler main-class prototype. This difference is central to the paper's argument. Fine-grained prototypes can help organise the clean embedding space, whereas coarser main-class prototypes are less brittle under the tested additive noise.

This paper makes three bounded contributions. First, it identifies 2 s Log-Mel CRNN modelling as the best supported clean closed-set mainline in the locked result package. Second, it introduces hierarchical subtype supervision as a fine-grained representation constraint while avoiding a significance claim that the data do not support. Third, it evaluates post-hoc main and hierarchical prototypes under a frozen DEMAND simulated-noise protocol, showing that main-class prototypes provide a small but statistically supported run-level Macro-F1 gain under simulated additive noise. The study does not claim real-farm external validation, unknown-class rejection, noise-aware retraining, or reliable cough recognition at 0 dB active-event SNR.

## 2. Related Work

Automatic animal-sound recognition combines bioacoustic sensing with supervised audio classification. Prior pig-vocalisation studies have characterised call valence and production context, investigated welfare-related vocal patterns, and developed cough or abnormal-vocalisation recognition methods [@briefer2022pigcalls; @tallet2013piglets; @illmann2013signalneed; @weary1995calling; @ferrari2008cough; @exadaktylos2008cough; @yin2021pigcoughcnn; @shen2022pigcoughfusion]. Recent smart-farm studies also address abnormal pig vocalisations and noisy farm-environment classification [@xie2024abnormalpig; @chung2025pvmc]. The present work does not claim to introduce pig-vocalisation recognition itself. It focuses on a leakage-controlled engineering sequence for duration selection, hierarchical supervision, and frozen prototype inference.

CRNN architectures are widely used for audio classification because convolutional layers can extract local time-frequency patterns and recurrent layers can summarise temporal dependencies [@choi2017crnn; @cakir2017crnn_sed]. In this study, the CRNN is used as a stable backbone rather than presented as a new architecture. The central question is how context duration, label hierarchy, and post-hoc inference affect a fixed pig-sound recognition protocol.

Log-Mel and cepstral features are common representations for animal and environmental audio [@mcfee2015librosa; @acevedo2009animalcalls; @huang2014anuran; @kahl2021birdnet]. The clean ablations in this study showed that increasing front-end complexity did not outperform the 2 s Log-Mel mainline. This result motivates a restrained method claim. The paper tests a controlled engineering sequence, not a new feature extractor.

Prototype-based inference is related to prototypical networks, where classes are represented by centres in an embedding space [@snell2017prototypical]. Here, the prototypes are post-hoc fold-seed-specific artefacts built from frozen train embeddings. They are not trained as a separate neural prototype architecture. This distinction is important for interpreting the method as an inference layer rather than a new backbone.

Selective prediction and risk-coverage analysis evaluate whether confidence scores can identify unreliable predictions [@geifman2017selective]. The corrected AUGRC calculations follow a generalised-risk formulation. In this study, the selective-prediction results are mixed. Raw Softmax has better aggregate AURC/AUGRC under simulated noise, while main-class prototypes slightly reduce selective risk at a frozen clean-validation threshold. The manuscript therefore avoids a broad uncertainty-improvement claim.

## 3. Materials and Methods

### 3.1 Task definition and label structure

The task is closed-set classification of pig vocal events into four main classes: cough, calm_grunt, feeding, and stress_vocal. A six-subtype auxiliary label space was used for hierarchical supervision. The subtypes were dry_cough, abdominal_cough, calm_grunt, feeding, frightened_stress, and anxious_stress.

The auxiliary task does not change the deployment-facing main label set. It provides finer acoustic constraints during representation learning and enables subtype-level prototype analysis after training.

### 3.2 Data sources and split integrity

The clean closed-set manifests combine third-party or public pig-audio sources. The dry_cough and abdominal_cough recordings were obtained from the Smart Farm Korea / data.go.kr pig-cough voice dataset. The calm_grunt, feeding, frightened_stress, and anxious_stress subtypes were derived from the figshare Sow Call Dataset. Local labels were mapped from source filename suffixes. The unresolved Kaggle-like scream/cough source was excluded from the current cap3x mainline manifests.

All splits preserve exact-path, source_id, and MD5 disjointness. Manifest and artefact SHA checks are part of the provenance record. Training data may build model parameters and prototypes. Validation data may calibrate temperatures, thresholds, and fusion weights. Test data are used only for evaluation.

### 3.3 Clean mainline model

The clean mainline uses class-capped cap3x training expansion, 2 s Log-Mel input features, and a CRNN backbone. The duration choice was treated as an experimental variable. The 1 s condition tests whether a compact event slice is sufficient. The 2 s condition tests whether short context improves recognition. The 3 s condition tests whether further context continues to help or introduces redundant background.

All clean comparisons were interpreted through matched fold-seed evidence where available. This constraint keeps duration and model comparisons tied to the validated result package.

### 3.4 Hierarchical auxiliary supervision

Hierarchical supervision adds a subtype prediction head to the main-class classifier. The training objective is:

\[
L = L_\mathrm{main} + \lambda L_\mathrm{aux}.
\]

The main loss optimises the four-class decision task. The auxiliary loss encourages the embedding to preserve subtype structure. Three auxiliary weights were evaluated: \(\lambda=0.2\), \(\lambda=0.5\), and \(\lambda=1.0\).

The manuscript uses \(\lambda=0.5\) as the balanced setting because it has the lowest standard deviation across runs. It also reports that \(\lambda=1.0\) has the highest mean Macro-F1. This separation prevents a small nonsignificant mean improvement from becoming an overstated performance claim.

### 3.5 Post-hoc prototype inference

Prototype inference was applied after training. For each fold-seed run, train-split embeddings were used to compute class centres. The main prototype method uses four centres, one per main class. The hierarchical prototype method additionally uses six subtype centres.

Embeddings were L2-normalised before prototype construction. Each prototype was the L2-normalised mean of the corresponding L2-normalised training embeddings. Classification used cosine similarity with validation-selected temperature scaling. Validation predictions were used to calibrate temperatures, thresholds, and fusion parameters where relevant. The frozen test split was used only for evaluation.

The raw Softmax rule and the prototype rule answer different questions. Raw Softmax evaluates the discriminative boundary learned by the classifier. Prototype inference evaluates whether a test embedding remains close to class-level acoustic centres learned from the training split.

### 3.6 Simulated additive-noise protocol

Noise robustness was evaluated with DEMAND environmental recordings [@demand2018]. The selected environments were DWASHING, TBUS, and STRAFFIC. The protocol mixed deterministic DEMAND segments into each 2 s pig-vocalisation waveform. Mixing used active-event SNRs of 20, 10, and 0 dB.

The 0 dB condition is described throughout as an extreme active-event SNR stress condition. It is not a no-noise setting and is not presented as an ordinary deployment condition.

The noisy evaluation was zero-shot and frozen. It reused the clean checkpoint, train-only clean prototypes, validation-only clean calibration, frozen rejection thresholds, and frozen fusion parameters. No noisy validation recalibration or noisy test tuning was allowed.

The deterministic noise-selection key used SHA256. It included fold, normalised clean path, clean MD5, DEMAND environment identity, noise-file SHA256, global noise seed, and repeat. It deliberately excluded model seed and SNR. A fixed clean sample and environment therefore reused the same noise segment across seeds and SNR levels. Only the noise gain changed across SNR.

## 4. Experimental Protocol

The clean and simulated-noise result package uses 5 folds and 5 seeds for final paper-facing summaries. Fold-seed pairing is preserved for model comparisons. The principal clean tables define the class set, subtype set, backbone protocol, duration comparison, hierarchical auxiliary weights, and clean prototype variants. The simulated-noise tables separate aggregate robustness, paired statistics, per-class failure modes, and selective-prediction metrics.

Appendix CSVs provide additional provenance and failure-mode detail. They include per-environment/SNR summaries, fold-level deltas, per-class 0 dB behaviour, cross-seed draw audits, and calibration-distribution records. These files support reproducibility without changing the roles of training, validation, or test data.

## 5. Results

### 5.1 Two-second context, rather than feature stacking, defines the clean mainline

The initial question was whether pig vocal events could be represented reliably by a 1 s input. The 1 s Log-Mel CRNN reached mean Macro-F1 0.9226 across 25 runs. This was a reasonable baseline, but it exposed a practical limitation.

The next attempt was to enrich the front end and architecture. The locked ablation table includes MCTAFD variants, PCEN, spectral-gate denoising, SpecAugment, BiLSTM, and attention pooling. None provided a more stable clean mainline than the 2 s Log-Mel CRNN. Cap3x logmel_dur2 reached 0.9472. In the available locked comparisons, cap3x mctafd_no_se_gated reached 0.9208, logmel_pcen reached 0.8916, logmel_sg reached 0.9093, logmel_specaug reached 0.9198, and logmel_dur2_attn reached 0.9431.

This failure evidence changed the design logic. The project did not continue to stack handcrafted or augmented features as the primary clean-audio path. Instead, it treated event duration as the key design variable.

The 2 s Log-Mel CRNN improved over the 1 s Log-Mel baseline by a paired mean Macro-F1 delta of 0.0246. The Wilcoxon p-value was 0.000162. A 3 s Log-Mel model reached 0.9434 in the available 15-run comparison, below the 2 s mean.

The final clean mechanism is therefore duration-aware but restrained. The model receives enough acoustic context to represent pig vocal events. It does not expand to a longer window that may admit redundant background. This conclusion applies to the present closed-set protocol and does not establish a universal duration for all pig-acoustic tasks.

### 5.2 Hierarchical supervision adds subtype semantics but not a significant clean-performance gain

The next question was whether four main labels were too coarse for representation learning. The main label cough groups dry_cough and abdominal_cough. Stress_vocal groups frightened_stress and anxious_stress. Feeding and stress_vocal also form a difficult boundary in the confusion summaries.

A pure four-class objective can learn a deployment-facing classifier, but it does not explicitly encode these finer distinctions. Hierarchical auxiliary supervision was therefore tested as a representation constraint. The main task remained the four-class problem, while a subtype head predicted six auxiliary labels.

In the locked auxiliary-weight table, \(\lambda=0.2\), \(\lambda=0.5\), and \(\lambda=1.0\) gave mean Macro-F1 values of 0.9505, 0.9510, and 0.9515. Lambda=1.0 had the highest mean. Lambda=0.5 had the lowest cross-run standard deviation, 0.0124, and was used for the prototype and noise studies.

The failure evidence is important. The hierarchical gain over the 2 s baseline did not reach statistical significance. The paper should therefore not describe hierarchical supervision as a significant clean-performance improvement. Its role is more specific: it introduces fine-grained acoustic semantics and supports boundary analysis.

The clean prototype summary is consistent with this interpretation. Raw Softmax at lambda=0.5 achieved Macro-F1 0.9510. The main prototype achieved 0.9519, and the hierarchical prototype achieved 0.9540. However, hierarchical prototype minus raw Softmax had mean delta 0.002937. The 95% CI was [-0.000478, 0.007294], with Wilcoxon p=0.058253. This result is close but remains nonsignificant under the locked criterion.

### 5.3 Main-class prototypes provide the strongest simulated-noise robustness

The third question was how the clean model behaves when environmental noise shifts the input distribution. Raw Softmax is tied to the discriminative boundary learned during clean training. Under simulated additive noise, this boundary can drift. A noisy event embedding may move in directions not seen during training.

Prototype inference tests a complementary decision rule. It asks whether an embedding remains near train-derived class centres. Clean results favoured the hierarchical prototype. It achieved mean Macro-F1 0.9540, compared with 0.9519 for the main prototype and 0.9510 for raw Softmax.

The noise results reversed the practical preference. Under ALL_NOISY DEMAND simulated additive noise, raw Softmax achieved Macro-F1 0.6173. The main prototype achieved 0.6230, and the hierarchical prototype achieved 0.6200.

Main prototype minus raw Softmax had a run-level mean delta of 0.005755. The 95% bootstrap CI was [0.001606, 0.010894], with Wilcoxon p=0.010511. Hierarchical prototype minus raw Softmax had mean delta 0.002710 and p=0.312333. Hierarchical prototype was also below the main prototype under ALL_NOISY conditions, with mean delta -0.003045 and p=0.000376.

Moderate noise provides the strongest support. Under 20 and 10 dB active-event SNR, the main prototype achieved Macro-F1 0.6534, compared with 0.6472 for raw Softmax and 0.6522 for the hierarchical prototype. Prototype minus raw Softmax had p=0.001673. Fold-level cluster analysis also supported this direction under MODERATE_NOISE, with all five folds showing positive main-prototype deltas.

In the extreme-stress stratum, the main prototype remained numerically above raw Softmax, 0.5624 versus 0.5574, but the paired p-value was 0.560171. This condition should therefore not be described as a significant robustness gain.

### 5.4 Per-class failure modes identify cough as the main extreme-noise risk

Per-class results expose the main boundary of the current system. Under ALL_NOISY conditions, cough F1 remained very low for all methods: 0.0867 for raw Softmax, 0.0974 for the main prototype, and 0.0957 for the hierarchical prototype.

Under 0 dB active-event SNR, cough recall was zero for STRAFFIC and TBUS across all methods. It was below 0.009 in DWASHING. Thus, the 0 dB condition is an extreme stress test. Cough recognition in that condition is unreliable.

Feeding and stress_vocal also remained entangled. Main prototypes improved several aggregate boundary metrics relative to raw Softmax under ALL_NOISY conditions. This improvement should be interpreted as an aggregate tendency, not as a complete per-class solution.

### 5.5 Selective-prediction metrics bound the reliability claim

Uncertainty results place another boundary on the claim. Prototypes did not comprehensively improve uncertainty or confidence ranking. Under ALL_NOISY conditions, raw Softmax had lower AURC/AUGRC, 0.1775/0.1239, than the main prototype, 0.2109/0.1394. It also outperformed the hierarchical prototype, 0.2122/0.1403, on these ranking metrics.

Prototype inference showed a slight frozen-threshold selective-risk improvement under ALL_NOISY conditions. Mean frozen-threshold selective risk was 0.2988 for the main prototype and 0.3024 for raw Softmax. This result should not be generalised into an uncertainty-calibration claim. The fused variant is retained as an ablation and sanity check, not as the main innovation.

## 6. Discussion

The main lesson is that practical pig-vocalisation recognition benefits from a sequence of narrowing decisions. The initial narrowing decision is temporal. Before adding complex acoustic front ends, the system needs an event window that captures relevant vocal context. In this result package, 2 s provided that window.

The second narrowing decision is semantic. Subtype labels can enrich the representation, but their contribution must be framed carefully. The main-performance gain over the 2 s baseline was not statistically significant. Hierarchical supervision is therefore best described as semantic and boundary-oriented.

The third narrowing decision is inferential. Clean subtype prototypes support semantic explanation, whereas main-class prototypes are better suited to simulated-noise robustness under the tested DEMAND protocol.

This interpretation explains why the noise results differ from the clean prototype results. Hierarchical prototypes introduce more detailed acoustic centres. Under clean conditions, this detail can help organise the embedding space. Under additive noise, the same detail may become brittle. Noise can move samples away from subtype-specific centres even when the main-class identity remains recoverable.

The main prototype acts as a coarser attractor. It therefore provides the most stable aggregate noise gain in the evaluated DEMAND protocol. This interpretation remains bounded by the available experiments.

The uncertainty findings prevent a stronger claim. If prototypes had improved Macro-F1, calibration metrics, and risk-coverage ranking together, the method could be framed as a broader reliability improvement. The locked tables do not support that framing. Prototype gains appeared mainly in Macro-F1 and frozen-threshold selective risk.

The simulated-noise protocol is useful but incomplete. DEMAND provides public environmental recordings and deterministic provenance. However, additive DEMAND mixing is not real pig-farm external validation. Real barns include reverberation, overlapping animals, sensor-placement changes, husbandry equipment, and label-distribution shifts. These factors are not captured by the present protocol.

The 0 dB active-event SNR setting is particularly severe and should be interpreted as stress testing. The cough collapse under 0 dB makes this limitation central rather than incidental. The current results should therefore be treated as controlled simulated-noise evidence.

## 7. Limitations

This study has several limitations. First, the noise evaluation uses simulated additive DEMAND noise only. It does not establish external validity in real pig farms. Second, the task is closed-set and does not address unknown sounds or open-set rejection.

Third, cough recognition under 0 dB active-event SNR is unreliable. Fourth, hierarchical auxiliary supervision improves the clean mean but does not significantly outperform the 2 s baseline. Clean hierarchical prototype improvement over raw Softmax also remains nonsignificant at p=0.058253.

Fifth, prototype inference does not comprehensively improve uncertainty. Raw Softmax has better AURC/AUGRC ranking in the aggregate noise summaries. Sixth, the fused method is an ablation and should not be promoted as a core contribution.

Finally, data-sharing permissions for third-party pig audio remain source-specific. Raw third-party audio is not redistributed with the repository. Access to the original recordings should follow the original provider terms.

## 8. Conclusion

This study restructures pig-vocalisation recognition around three bounded findings. A 2 s Log-Mel CRNN is the best supported clean closed-set mainline in the locked result package. It significantly improves over 1 s input in the matched paired comparison.

Hierarchical auxiliary supervision adds subtype semantics and supports boundary analysis. Its incremental clean gain is not statistically significant and should not be overstated. Post-hoc prototype inference separates two roles. Hierarchical prototypes are most useful for clean semantic explanation. Main-class prototypes provide the most reliable aggregate simulated-noise robustness under the frozen DEMAND protocol, especially in the 20 and 10 dB moderate-noise stratum.

The work offers a reproducible path towards noise-aware pig-vocalisation inference under controlled simulated conditions. It leaves real-farm validation, robust cough recognition under severe noise, and open-set deployment as necessary next steps.

## Data Availability

Processed aggregate tables and figure source data supporting the results are available in the project repository and/or Supplementary Data associated with this article. These materials include paper-ready summary tables, paired statistics, figure source-data mappings, command records, cross-validation manifests, and hash-based leakage and provenance audits. Raw third-party pig audio is not redistributed. Smart Farm Korea / data.go.kr pig-cough recordings should be obtained from the original provider [@smartfarmkorea_pig_cough_voice]. Sow Call Dataset recordings should be obtained from figshare via DOI 10.6084/m9.figshare.16940389 [@sow_call_dataset_2021]. DEMAND environmental-noise recordings should be obtained from Zenodo via DOI 10.5281/zenodo.1227121 [@demand2018]. The unresolved Kaggle-like scream/cough source is excluded from the current main results and is not part of the submitted data description.

## Code Availability

The code and reproducibility package are available at https://github.com/HinataQAQ/pig-sound-classification/releases/tag/v0.1-paper-draft-r1. The package includes scripts, command records, aggregate tables, figure source-data mappings, manifests where permitted, and leakage/provenance audit files needed to reproduce the reported analyses from the permitted inputs. No code DOI is claimed. Model checkpoints, prototype artefacts, and raw third-party audio are not included unless released separately with provider-compatible terms, file hashes, and artifact-to-result mapping.

## Ethics Statement

No new animal experiment, animal intervention, or prospective farm recording was conducted for this manuscript. The study is a computational reuse of public or third-party pig-audio recordings and public DEMAND environmental noise. Original source-side permissions, repository terms, and any animal-ethics approvals remain with the original data providers. Source-specific access and redistribution restrictions are described in the Data Availability statement.

## References
