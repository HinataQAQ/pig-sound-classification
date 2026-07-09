# Pig Vocalisation Recognition under Simulated Environmental Noise Using Duration-Aware Hierarchical Representations and Post-hoc Prototype Inference

## Abstract

Pig vocalisation recognition can support non-contact livestock monitoring, but robust performance depends on event duration, label granularity, and inference behaviour under noise. We studied a closed-set four-class task covering cough, calm_grunt, feeding, and stress_vocal. Six auxiliary subtypes were used for hierarchical supervision. In leakage-controlled 5-fold cross-validation with five random seeds, a 2 s Log-Mel CRNN improved clean Macro-F1 from 0.9226 with 1 s input to 0.9472, with paired mean delta 0.0246 and Wilcoxon p=0.000162. More complex clean-audio variants did not form a stronger mainline. Hierarchical auxiliary supervision improved the mean clean score but did not significantly exceed the 2 s baseline, so we interpret it as a semantic representation mechanism. We then froze the clean model, train-set prototypes, and validation-set calibration, and evaluated zero-shot additive DEMAND noise. Main-class prototypes achieved the most stable simulated-noise result. Under ALL_NOISY conditions, Macro-F1 increased from 0.6173 for raw Softmax to 0.6230, with paired Wilcoxon p=0.010511. The gain was strongest under 20 and 10 dB active-event SNR. The 0 dB setting caused severe cough failure. These results support post-hoc main-class prototype inference as a modest simulated-noise robustness mechanism, while real-farm external validation remains necessary.

## Keywords

Pig vocalisation recognition; Log-Mel; CRNN; hierarchical supervision; acoustic prototype; simulated additive noise; DEMAND; selective prediction

## Introduction

Acoustic monitoring is a practical route for observing animal health and behaviour without physical contact. Pig vocalisations can indicate respiratory events, affective state, feeding activity, and stress-related contexts [@briefer2022pigcalls; @tallet2013piglets; @illmann2013signalneed; @weary1995calling]. Previous work has shown that pig sounds can support cough detection and welfare-oriented monitoring [@ferrari2008cough; @exadaktylos2008cough; @yin2021pigcoughcnn; @shen2022pigcoughfusion]. Recent smart-farm systems further show the importance of noisy farm acoustics and multi-stage recognition pipelines [@xie2024abnormalpig; @chung2025pvmc]. The present study does not claim that pig vocalisation recognition itself is new. Instead, it asks how a reproducible recognition pipeline should be assembled when event duration, subtype semantics, and noise robustness are all considered.

A pig sound is not always a single isolated impulse. Cough events can be short but structured. Feeding and stress vocalisations can contain repeated texture and short temporal context. A short clip may truncate useful acoustic evidence. A longer clip may include irrelevant background. This creates a design problem before model complexity is considered: the system needs an event window that represents the vocalisation without adding excessive background.

The first part of this work therefore treats duration as an experimental variable. We compare 1 s, 2 s, and 3 s Log-Mel inputs under the same closed-set framework. The comparison is not framed as a claim that 2 s is universally optimal. It is a leakage-controlled test of which duration is best supported by the available pig-vocalisation data.

The second part addresses label granularity. The deployment-facing task has four main classes: cough, calm_grunt, feeding, and stress_vocal. However, cough contains dry_cough and abdominal_cough. Stress_vocal contains frightened_stress and anxious_stress. A four-class objective can support deployment reporting, but it may not preserve these finer acoustic distinctions. We therefore evaluate hierarchical auxiliary supervision as a representation constraint rather than as a replacement task.

The third part studies inference under simulated environmental noise. Raw Softmax uses the discriminative boundary learned by the clean classifier. Post-hoc prototypes instead ask whether a test embedding remains close to class centres computed from clean training embeddings. We compare main-class prototypes and hierarchical prototypes under a frozen protocol. This separation is central to the study. Hierarchical prototypes are expected to support clean semantic organisation. Coarser main-class prototypes may be less brittle under noise.

The contributions are deliberately bounded. First, we identify 2 s Log-Mel CRNN as the clean mainline through matched fold-seed evidence. Second, we show that hierarchical auxiliary supervision adds fine-grained semantic structure, while avoiding a significance claim that the data do not support. Third, we show that main-class post-hoc prototypes improve aggregate simulated-noise Macro-F1 without noisy retraining. The study does not report real-farm external validation, unknown-class rejection, or reliable cough recognition under extreme 0 dB active-event SNR.

## Related work

Animal-sound recognition commonly uses time-frequency features and supervised classifiers. Log-Mel and cepstral features are widely used because they represent spectral energy in a form suited to convolutional models [@mcfee2015librosa; @acevedo2009animalcalls; @huang2014anuran; @kahl2021birdnet]. CRNN architectures combine convolutional feature extraction with recurrent temporal modelling and have been used broadly in audio classification and sound-event detection [@choi2017crnn; @cakir2017crnn_sed]. In this paper, the CRNN is not treated as a new architecture. It is a stable backbone for testing duration, hierarchy, and inference choices.

Pig-specific acoustic studies motivate the label structure. Work on pig calls has linked vocal features to context, affective state, need signalling, and welfare [@briefer2022pigcalls; @tallet2013piglets; @illmann2013signalneed; @weary1995calling]. Cough recognition studies show the practical relevance of respiratory acoustics [@ferrari2008cough; @exadaktylos2008cough; @yin2021pigcoughcnn; @shen2022pigcoughfusion]. These studies motivate the use of cough and stress-related categories, but they also warn against overclaiming from clean closed-set data alone.

Prototype-based inference is related to prototypical networks, where each class is represented by a centre in an embedding space [@snell2017prototypical]. The prototypes in this study are not trainable model components. They are post-hoc artefacts built separately for every fold and seed from train-split embeddings. This design tests whether clean class centres provide a stable frozen inference rule under simulated noise.

Selective prediction evaluates how confidence scores rank reliable and unreliable predictions [@geifman2017selective]. We report risk-coverage metrics, AURC, and corrected AUGRC following the generalised-risk convention used by the fd-shifts implementation [@fdshifts_riskcoverage]. These metrics are used conservatively. The present results do not support a broad claim that prototype inference improves uncertainty ranking.

## Materials and methods

### Data sources and task definition

The task is closed-set pig-vocalisation classification. The main classes are cough, calm_grunt, feeding, and stress_vocal. The auxiliary subtypes are dry_cough, abdominal_cough, calm_grunt, feeding, frightened_stress, and anxious_stress. The dry_cough and abdominal_cough recordings come from the Smart Farm Korea / data.go.kr pig-cough voice dataset [@smartfarmkorea_pig_cough_voice]. The remaining subtypes derive from the figshare Sow Call Dataset [@sow_call_dataset_2021]. Local subtype mappings were derived from the source labels and filename suffixes. The unresolved Kaggle-like scream/cough source is excluded from the current mainline manifests.

The environmental-noise evaluation uses DEMAND recordings [@demand2018]. The selected DEMAND environments are DWASHING, TBUS, and STRAFFIC. These noises are mixed additively into existing pig vocalisations. The resulting protocol is simulated noise robustness, not real-farm external validation.

### Feature extraction and clean CRNN backbone

Audio was represented with 2 s Log-Mel features in the final clean mainline. The locked configuration used 32 kHz audio, 64 Mel bands, 1024-point FFT, 800-sample analysis windows, 320-sample hops, and a 50–8000 Hz frequency range. Short recordings were padded to the target window, while longer recordings were centre-cropped.

The clean backbone is a convolutional recurrent neural network. A 1×1 convolutional front end maps the input channels to a compact feature representation. Three convolutional blocks then extract local time-frequency patterns. A bidirectional recurrent layer summarises temporal dependencies, and mean pooling converts the recurrent sequence into a fixed embedding. The main head predicts the four deployment-facing classes. The auxiliary head predicts six subtypes when hierarchical supervision is enabled.

### Hierarchical auxiliary supervision

Hierarchical supervision adds a subtype prediction head. The training objective is

\[
L = L_{main} + \lambda L_{aux}.
\]

The main loss optimises the four-class task. The auxiliary loss encourages the embedding to preserve subtype-level acoustic structure. We evaluated \(\lambda = 0.2\), \(0.5\), and \(1.0\). The manuscript uses \(\lambda=0.5\) for prototype and noise studies because it had the lowest cross-run standard deviation. The \(\lambda=1.0\) setting is reported because it had the highest mean clean Macro-F1.

### Post-hoc prototype inference

Prototype inference was applied after training. For each fold-seed run, embeddings from the training split were L2-normalised. Each prototype was the L2-normalised mean of the normalised embeddings for that class. The main prototype method uses four prototypes. The hierarchical prototype method also uses six subtype prototypes and maps subtype probabilities back to the four main classes.

For a test embedding \(z\) and class prototype \(c_k\), cosine similarity \(z^\top c_k\) is converted into probabilities with a validation-selected temperature. Validation data calibrate temperatures, rejection thresholds, and fusion weights. Test data are used only for frozen evaluation. No prototype, calibration parameter, or threshold is built by mixing folds or seeds.

### Simulated additive-noise protocol

The zero-shot noisy evaluation reuses the clean checkpoint, clean train-derived prototypes, and clean validation calibration. No noisy validation recalibration is performed. No noisy test result is used to choose thresholds or parameters.

Noise is added at active-event SNR values of 20, 10, and 0 dB. The SNR is computed over the valid non-padding portion of the clean event, while the selected noise segment covers the complete 2 s waveform. The 0 dB condition is an extreme active-event stress test. It is not a no-noise condition and is not presented as an ordinary deployment scenario.

Noise draws are deterministic. The key includes fold, normalised clean path, clean MD5, DEMAND environment identifier, noise-file SHA256, global noise seed, and repeat. It excludes model seed and SNR. Therefore, a fixed sample and environment use the same noise segment across seeds and SNR levels; only the gain changes across SNR.

### Evaluation and statistics

Final clean and simulated-noise summaries use 5 folds × 5 seeds. Pairing is preserved by fold and seed. Clean comparisons report mean, standard deviation, paired deltas, bootstrap confidence intervals, Wilcoxon signed-rank tests, and wins/ties/losses where appropriate. Simulated-noise analyses report MODERATE_NOISE, EXTREME_STRESS, and ALL_NOISY strata. MODERATE_NOISE contains 20 and 10 dB conditions. EXTREME_STRESS contains the 0 dB condition. ALL_NOISY contains all nine environment/SNR combinations.

We also report fold-cluster statistics. These average seed-level deltas within each fold before estimating fold-level means and confidence intervals. This is important because different seeds within a fold share the same test split. Run-level p-values are therefore not treated as the only evidence for final interpretation.

## Results

### Duration selection identifies the clean mainline

The 1 s Log-Mel CRNN achieved mean Macro-F1 0.9226 across 25 runs. The 2 s Log-Mel CRNN achieved 0.9472. The paired mean delta was 0.0246, with Wilcoxon p=0.000162. The 3 s Log-Mel model reached 0.9434 in the available 15-run comparison. It did not surpass the 2 s setting.

More complex clean-audio variants did not form a stronger mainline. The tested alternatives included MCTAFD variants, PCEN, spectral-gate denoising, SpecAugment, and attention pooling. In the locked comparison table, these alternatives remained below the 2 s Log-Mel CRNN mean. This negative evidence changed the design logic from adding front-end complexity to identifying a suitable event window.

The resulting claim is bounded. The data support 2 s as the validated event context for this closed-set protocol. They do not establish 2 s as a universal duration for all pig-acoustic tasks.

### Hierarchical supervision adds semantic structure

The auxiliary-weight table showed mean Macro-F1 values of 0.9505, 0.9510, and 0.9515 for \(\lambda=0.2\), \(0.5\), and \(1.0\). The highest mean came from \(\lambda=1.0\). The lowest standard deviation, 0.0124, came from \(\lambda=0.5\). We therefore used \(\lambda=0.5\) as the balanced setting for prototype and noise studies.

The gain over the 2 s baseline was not statistically significant. This matters for interpretation. Hierarchical supervision should not be described as a decisive clean-performance improvement. Its role is to impose subtype-aware structure on the representation and support boundary analysis.

The clean prototype comparison follows the same pattern. Raw Softmax at \(\lambda=0.5\) achieved Macro-F1 0.9510. The main prototype reached 0.9519. The hierarchical prototype reached 0.9540. However, hierarchical prototype minus raw Softmax had mean delta 0.002937, 95% CI [-0.000478, 0.007294], and Wilcoxon p=0.058253. The result is positive but not significant under the locked criterion.

### Main-class prototypes improve simulated-noise robustness

Under ALL_NOISY simulated DEMAND conditions, raw Softmax achieved mean Macro-F1 0.6173. The main prototype achieved 0.6230. The hierarchical prototype achieved 0.6200. The main prototype therefore had the highest aggregate noisy Macro-F1.

The run-level paired comparison for main prototype minus raw Softmax gave mean delta 0.005755, 95% bootstrap CI [0.001606, 0.010894], Wilcoxon p=0.010511, and wins/ties/losses of 18/0/7. The corresponding hierarchical prototype minus raw Softmax delta was 0.002710, with p=0.312333. Hierarchical prototype was significantly below main prototype under ALL_NOISY conditions, with mean delta -0.003045 and p=0.000376.

Fold-cluster statistics refine this conclusion. For ALL_NOISY, the main prototype fold-cluster confidence interval crossed zero. For MODERATE_NOISE, the fold-cluster interval was positive, with 5/5 folds favouring main prototype over raw Softmax. We therefore treat the medium-noise result as the strongest robustness evidence.

The EXTREME_STRESS stratum does not support a significant robustness claim. Main prototype remained numerically above raw Softmax, 0.5624 versus 0.5574, but the paired p-value was 0.560171. The 0 dB condition should be interpreted as stress testing rather than ordinary operating noise.

### Per-class results identify cough as the main failure mode

ALL_NOISY per-class results show that cough is the main unresolved class. Cough F1 was 0.0867 for raw Softmax, 0.0974 for main prototype, and 0.0957 for hierarchical prototype. Most cough errors were redirected to stress_vocal. Under 0 dB active-event SNR, cough recall collapsed in the STRAFFIC and TBUS conditions for all methods.

Feeding and stress_vocal remained partially entangled. Under ALL_NOISY conditions, main prototype reduced feeding_to_stress errors relative to raw Softmax. Stress_to_feeding errors changed less. These observations support the robustness interpretation at the aggregate level but do not imply that per-class failure has been solved.

### Selective-prediction results do not support a broad uncertainty claim

Prototype inference did not comprehensively improve uncertainty ranking. Under ALL_NOISY conditions, raw Softmax had lower AURC/AUGRC, 0.1775/0.1239, than main prototype, 0.2109/0.1394, and hierarchical prototype, 0.2122/0.1403. Lower values indicate better risk-coverage ranking.

The main prototype showed a small advantage at the frozen-threshold operating point. Mean frozen-threshold selective risk was 0.2988 for main prototype and 0.3024 for raw Softmax. This is a narrow operating-point result. It should not be generalised into a claim that prototypes improve uncertainty calibration.

## Discussion

The main finding is that practical pig-vocalisation recognition benefits from sequential design choices. The first choice is temporal. The system should represent a complete event before adding more feature channels or a more complex temporal module. In the locked result package, the 2 s window provided that balance.

The second choice is semantic. Subtype labels add structure to the embedding. They do not, by themselves, provide a statistically significant performance improvement over the 2 s baseline. This distinction prevents the hierarchical module from being oversold.

The third choice is inferential. The clean classifier and the prototype rule answer different questions. Raw Softmax uses a learned discriminative boundary. Main-class prototypes use distance to train-derived class centres. The latter was more stable under the tested simulated noise. The hierarchical prototype was better for clean semantic organisation but became less robust than the main prototype under additive noise.

This contrast is scientifically useful. It suggests that subtype detail can sharpen clean representations but may be fragile when noise perturbs embeddings. Coarser class centres can act as more stable attractors under moderate simulated environmental interference.

The uncertainty results are also informative. A method that improves Macro-F1 does not necessarily improve confidence ranking. Here, raw Softmax performed better on AURC/AUGRC, while main prototype slightly reduced selective risk at a frozen validation-chosen threshold. The manuscript should therefore frame prototype inference as a robustness-oriented decision rule, not as a universal uncertainty solution.

The DEMAND protocol is useful because it is deterministic, public, and reproducible. It is incomplete because it is additive. Real barns include reverberation, overlapping animals, non-stationary equipment, sensor-placement variation, and label-distribution shift. These factors are outside the present evidence. The results should therefore be described as controlled simulated-noise evidence, not as external deployment validation.

## Limitations

The first limitation is the use of simulated additive DEMAND noise rather than real pig-farm external validation. The second is that the task is closed-set and does not evaluate unknown sounds. The third is that 0 dB active-event SNR causes severe cough failure. This failure should be treated as a deployment risk, not as a solved problem.

Fourth, hierarchical auxiliary supervision and clean hierarchical prototypes are positive but not statistically significant over their corresponding baselines. Fifth, prototype inference does not comprehensively improve uncertainty ranking. Sixth, the fused method is an ablation and should not be promoted as a core contribution.

Finally, the pig-audio sources are third-party datasets. The manuscript does not redistribute raw third-party audio. The Smart Farm Korea / data.go.kr source remains subject to provider terms for raw audio redistribution. This restriction is reflected in the Data Availability statement.

## Conclusion

This study supports three bounded conclusions. First, 2 s Log-Mel input is the best-supported clean event context in the locked pig-vocalisation pipeline. Second, hierarchical auxiliary supervision adds subtype-aware semantic structure but should not be described as a statistically significant clean-performance gain. Third, post-hoc main-class prototype inference provides the clearest simulated-noise robustness benefit, especially under 20 and 10 dB active-event DEMAND noise.

The work provides a reproducible framework for duration-aware and prototype-based pig-vocalisation inference. It leaves real-farm validation, robust cough recognition under severe noise, and open-set deployment as necessary next steps.

## Data Availability

Processed aggregate tables and figure source data supporting the results are available in the project repository and/or Supplementary Data associated with this article. These materials include paper-ready summary tables, paired statistics, figure source-data mappings, command records, cross-validation manifests where permitted, and hash-based leakage and provenance audits. Raw third-party pig audio is not redistributed. Smart Farm Korea / data.go.kr pig-cough recordings should be obtained from the original provider [@smartfarmkorea_pig_cough_voice]. Sow Call Dataset recordings should be obtained from figshare via DOI 10.6084/m9.figshare.16940389 [@sow_call_dataset_2021]. DEMAND environmental-noise recordings should be obtained from Zenodo via DOI 10.5281/zenodo.1227121 [@demand2018]. The unresolved Kaggle-like scream/cough source is excluded from the current main results and is not part of the submitted data description.

## Code Availability

The code and reproducibility package are available at https://github.com/HinataQAQ/pig-sound-classification/releases/tag/v0.1-paper-draft-r1. The release includes scripts, command records, aggregate tables, figure source-data mappings, manifests where permitted, and leakage/provenance audit files needed to reproduce the reported analyses from the permitted inputs. No software DOI is claimed. Model checkpoints, prototype artefacts, DEMAND audio archives, and raw third-party pig audio are not included unless released separately under provider-compatible terms with file hashes and artifact-to-result mapping.

## Ethics Statement

No new animal experiment, animal intervention, or prospective farm recording was conducted for this manuscript. The study is a computational reuse of public or third-party pig-audio recordings and public DEMAND environmental noise. Original source-side permissions, repository terms, and any animal-ethics approvals remain with the original data providers. Source-specific access and redistribution restrictions are described in the Data Availability statement.

## References
