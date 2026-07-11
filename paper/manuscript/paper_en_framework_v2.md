# A Duration-Aware Hierarchical Prototype Inference Framework for Interpretable Pig Vocalisation Recognition

## Abstract

Pig vocalisation recognition requires complete acoustic events, fine-grained health/behaviour structure, and inspectable predictions. We organise these requirements as a four-stage closed-set framework: a 1-s Log-Mel CRNN (B0), a 2-s duration-aware CRNN (B1), shared supervision over four main classes and six subtypes (B2), and fold-seed-specific post-hoc hierarchical prototype inference (B3). All comparisons use the same 5-fold × 5-seed cohort with exact-path, source-ID, and MD5 disjointness. B1 produced the strongest independently supported increment over B0 (mean Macro-F1 delta 0.024631; raw Wilcoxon P=0.000162; Holm-adjusted P=0.000649). The B2-B1 and B3-B2 increments were positive in mean but not statistically significant after Holm adjustment. The λ=0.5 B2/B3 cohort lacks a documented pre-test or validation-only selection rule, so its comparisons are retrospective and exploratory. The complete B3 framework improved mean Macro-F1 from 0.922606 to 0.953986 relative to B0 (delta 0.031380; 95% bootstrap CI [0.022821, 0.040301]; raw P=2.98×10^-7; Holm-adjusted P=1.49×10^-6; wins/ties/losses 24/0/1), and all five fold-mean differences were positive. B3 converts a single class decision into ranked main- and subtype-level candidates, prototype distances, decision margins, hierarchy-consistency indicators, uncertainty fields, and training-set representatives. Deterministic examples illustrate these output semantics but are not aggregate evidence. The cumulative B3-B0 benefit does not establish statistical interaction or independently significant gains for every stage. The present evidence is limited to clean closed-set recognition; simulated-noise analyses are supplementary, and real-farm external validation remains necessary.

## Keywords

Pig vocalisation recognition; duration-aware audio classification; hierarchical supervision; prototype inference; acoustic candidates; interpretable prediction; CRNN

## 1. Introduction

Acoustic monitoring offers a non-contact route to pig health and behaviour assessment. Vocalisations can carry information about respiratory events, feeding, affective state, and stress-related contexts [@briefer2022pigcalls; @tallet2013piglets; @illmann2013signalneed; @weary1995calling]. Pig-cough studies further show why respiratory sounds are relevant to livestock monitoring [@ferrari2008cough; @exadaktylos2008cough; @yin2021pigcoughcnn; @shen2022pigcoughfusion]. Recognition performance alone, however, is not sufficient for a deployment-facing system. The model must receive an informative event window, retain distinctions that are coarser than the available biological or behavioural labels, and expose why a predicted class is acoustically plausible.

These requirements motivate the organising statement of this study: **“See the complete event, learn the fine-grained structure, and predict with interpretable acoustic candidates.”** “See” refers to the duration study that establishes the event context. “Learn” refers to shared four-main/six-subtype supervision in one embedding space. “Predict” refers to a post-hoc prototype layer that returns ranked candidates and traceable training representatives. The statement is a framework description, not a claim that the stages interact synergistically.

Duration is the first design variable because a short window may truncate repeated feeding texture, stress-vocal context, or the temporal structure of a cough. A longer window can add irrelevant background. The question is therefore not whether a more complicated front end can be stacked onto a baseline, but whether the classifier sees a sufficiently complete event. We test 1-s and 2-s Log-Mel inputs under matched folds and seeds and retain the wider set of completed clean-audio alternatives as negative design evidence.

Label granularity is the second variable. The deployment-facing task contains cough, calm_grunt, feeding, and stress_vocal. The available labels also distinguish dry_cough from abdominal_cough and frightened_stress from anxious_stress. A four-class objective alone does not require the embedding to retain those distinctions. We therefore train the four-class and six-subtype heads jointly, while keeping the main prediction task unchanged.

Interpretability is the third variable. Raw Softmax returns a probability vector but does not directly expose the embedding's relation to class centres or subtype structure. After training, we freeze the backbone and build main and subtype prototypes separately for every fold and seed from its training split. Within each already fixed run, validation data calibrate temperatures, weights, and uncertainty criteria, and test data supply the frozen evaluation. This within-run split contract is distinct from the retrospective λ=0.5 cohort-selection limitation disclosed below. The prototype layer yields main and subtype Top-k candidates, distances, margins, hierarchy consistency, uncertainty fields, and class-prototype representatives.

The contributions are bounded as follows.

1. A matched duration study shows that 2-s context substantially improves the clean closed-set task and is more effective than the tested complex front-end alternatives.
2. A shared four-main/six-subtype supervision scheme introduces fine-grained health and behavioural semantics into one acoustic embedding space.
3. A fold-seed-specific post-hoc prototype inference layer converts a single Softmax output into main/subtype Top-k candidates, prototype distances, decision margins, hierarchy consistency, uncertainty flags, and prototype representatives.

The matched inferential support in contribution 1 is the B1-B0 duration contrast; the wider completed clean-front-end comparison is descriptive because run counts and pairings differ (Supplementary Fig. S1 and Table S1). The third contribution concerns prediction semantics and interpretability. The prototype increment alone is not statistically significant.

## 2. Related work

### 2.1 Pig-vocalisation duration and coarse-to-fine recognition

Fixed 2-s pig audio is not new. The PLOS pig-speech study used 2-s clips with combined spectral and temporal features [@wu2022pigspeech], and TransformerCNN used 2-s pig calls with a parallel Transformer and convolutional architecture [@liao2023transformercnn]. These studies establish prior use of the duration and overlapping behavioural labels. Their published scores are not direct benchmarks for this work because the locally available corpus, label mapping, split membership, grouping, class balance, and reported metrics do not all match.

Coarse-to-fine pig classification also has precedent. PVMC separates vocal activity, coarse vocal type, and contextual classes across multiple trained stages [@chung2025pvmc]. Our B2 stage instead uses a shared embedding and simultaneous main/subtype supervision, and B3 constructs prototypes after the backbone has been trained. The distinction is architectural and procedural; it is not a claim that coarse-to-fine pig recognition originated here.

Briefer et al. mapped pig calls in an acoustic representation and discussed structure within the call repertoire that may support finer groupings or subtype discovery [@briefer2022pigcalls]. This prior work motivates the biological relevance of fine-grained structure while also warning against equating a learned cluster with a validated clinical subtype.

### 2.2 Hierarchical and interpretable acoustic prototypes

Prototypical networks represent a class by an embedding-space centre and infer labels from distances to those centres [@snell2017prototypical]. Hierarchical prototypes also have prior use. HiSSNet combines hierarchy-level losses with prototypical inference for non-pig sound-event and speaker-identification tasks [@shashaank2023hissnet]. Its prototypes are integral to episodic learning; they are not the fold-seed-specific post-hoc construction used here.

AudioProtoPNet establishes prior art for interpretable acoustic prototypes and retrieval of training examples in bird-sound classification [@heinrich2025audioprotopnet]. It uses a trainable prototype classifier that replaces the ordinary classification layer. It must not be described as a post-hoc prototype method. Our B3 stage instead freezes an already trained CRNN and constructs train-only centroids without replacing or retraining the classification layer.

The targeted audit identified prior work for each ingredient but did not locate an exact pig-specific match for the complete route of fixed 2-s input, shared four-main/six-subtype supervision, fold-seed-specific post-hoc train-only prototypes, ranked main/subtype candidates, and class-prototype representatives. This is a search-bounded result through 11 July 2026, not an absolute first claim. The published-score comparability table is therefore used only to document mismatched datasets, labels, protocols, and metrics; it is not a leaderboard.

## 3. Materials and methods

### 3.1 Task, data, and leakage controls

The closed-set main task contains four classes: cough, calm_grunt, feeding, and stress_vocal. The auxiliary hierarchy contains six subtypes: dry_cough, abdominal_cough, calm_grunt, feeding, frightened_stress, and anxious_stress. Dry and abdominal cough map to cough; frightened and anxious stress map to stress_vocal; calm_grunt and feeding each map to the corresponding main class.

The clean manifests combine third-party pig-audio sources. The cough recordings derive from the Smart Farm Korea/data.go.kr pig-cough source [@smartfarmkorea_pig_cough_voice]. Calm-grunt, feeding, frightened-stress, and anxious-stress recordings derive from the Sow Call Dataset [@sow_call_dataset_2021], with local subtype labels mapped from filename suffixes. The unresolved Kaggle-like scream/cough archive is excluded from the cap3x mainline.

The evaluation grid contains five folds and five model seeds (42, 123, 777, 2024, and 3407). Within every fold, training, validation, and test splits are disjoint by normalised exact path, source ID, and MD5. For each fixed run, training data determine model parameters and prototypes, validation data calibrate temperatures, fusion weights, thresholds, and uncertainty criteria, and test data provide evaluation. Prototypes, calibration artifacts, and predictions are never mixed across folds or seeds. This run-level isolation does not remove the separately disclosed, test-informed choice of the λ=0.5 cohort.

### 3.2 Cumulative framework

The four stages are evaluated cumulatively.

- **B0:** 1-s Log-Mel CRNN.
- **B1:** 2-s duration-aware Log-Mel CRNN.
- **B2:** B1 plus shared four-main/six-subtype auxiliary supervision at λ=0.5, evaluated with the main Raw Softmax output.
- **B3:** B2 plus post-hoc hierarchical prototype candidate inference.

The sequence separates three questions: how much event context the backbone receives, what label structure the embedding learns, and what information the inference layer returns. It is not a factorial experiment, so it cannot estimate statistical interaction among duration, hierarchy, and prototypes.

### 3.3 Log-Mel CRNN and duration study

Audio is resampled and converted to a 64-bin Log-Mel representation using the same feature settings within each matched comparison [@mcfee2015librosa]. Recordings shorter than the target duration are padded; longer recordings are centre-cropped. The CRNN uses convolutional blocks to extract local time-frequency patterns, a bidirectional recurrent layer to model temporal dependencies, and temporal pooling to produce a fixed embedding for the main classifier. B0 and B1 differ in input duration rather than class definition or evaluation protocol.

The duration claim is comparative, not universal. B1 is evaluated against the matched B0 runs. Completed 3-s and complex-front-end alternatives are retained as contextual ablations but are not folded into the four-stage inferential family.

### 3.4 Shared hierarchical auxiliary supervision

B2 adds a six-subtype head to the shared embedding. The training objective is

\[
L = L_{main} + \lambda L_{aux}.
\]

The main loss optimises the four-class output; the auxiliary loss encourages subtype structure. The completed λ study found the highest test-set mean at λ=1.0 and the lowest cross-run test-set standard deviation at λ=0.5. B2 and B3 use the already frozen λ=0.5 prototype cohort; the present cumulative analysis did not change that cohort. However, the archived record does not document a pre-test or validation-only rule for selecting λ=0.5. We therefore treat this choice as retrospective and test-informed, regard comparisons involving B2 and B3 as exploratory, and do not claim that λ=0.5 is an independently selected optimum.

### 3.5 Post-hoc hierarchical prototype candidate inference

For each fold-seed run, B3 freezes the trained B2 backbone. Training embeddings are L2-normalised, averaged within each main class and subtype, and normalised again to form four main prototypes and six subtype prototypes. For an embedding \(z\) and prototype \(c_k\), cosine similarity \(z^\top c_k\) is transformed into candidate probabilities with validation-calibrated temperatures. Main and subtype evidence is combined with validation-selected weights and mapped through the known hierarchy. No test prediction selects a temperature, weight, threshold, or prototype.

B3 returns:

- Raw Softmax Top-k main candidates;
- main-prototype Top-k candidates and four main distances;
- hierarchical main Top-k candidates;
- subtype Top-k candidates and six subtype distances;
- Top-1-minus-Top-2 decision margins;
- hierarchy consistency between the hierarchical main prediction and the main class mapped from the subtype prediction;
- uncertainty fields based on frozen validation criteria; and
- a same-run training representative for the predicted class prototype.

The representative is a class-centre explanation. It does not establish which training item most resembles the test query.

### 3.6 Evaluation and multiplicity control

Macro-F1 is the primary metric, computed with `zero_division=0`. Stage summaries report mean, sample standard deviation, minimum, and maximum across the 25 matched runs. Direct contrasts report the paired mean delta, sample standard deviation of paired deltas, percentile-bootstrap 95% confidence interval, two-sided Wilcoxon signed-rank P value, and wins/ties/losses. A fold-cluster sensitivity first averages the five seeds within each fold and bootstraps the five fold means.

The five planned P values are B1-B0, B2-B1, B3-B2, B3-B1, and B3-B0. They are adjusted as one family with the step-down Holm procedure. The adjusted results are derived only from the stored raw P values; predictions and model outputs are unchanged.

### 3.7 Deterministic prototype cases

The case-study source is the pre-specified first numeric fold-seed key in the canonical expected-key list, fold 0/seed 42, after its artifact checks pass; it is not chosen by filename order. Four correct cases require Raw Softmax, main prototypes, and hierarchical prototypes all to predict the true main class; the lower-median Raw Softmax-confidence row is selected within each class. Each feeding/stress boundary pool fixes the true boundary class, requires `{feeding, stress_vocal}` to be the Raw or hierarchical Top-2 set, and requires at least one Raw, main-prototype, or hierarchical decision to be wrong; its lower-median Raw Top-2-margin row is selected. The final case is the remaining row with the smallest hierarchical Top-2 margin. Ties are resolved by source ID. This is a deterministic post-hoc display rule applied to frozen test outputs, not parameter tuning or a substitute for aggregate evaluation.

## 4. Results

### 4.1 Cumulative framework evidence

Table 1 replaces the fragmented clean-result tables with the cumulative B0-B3 framework. For B2, the cumulative difference from B0 is the arithmetic difference between stage means; B2-B0 was not one of the five planned Wilcoxon contrasts, so no direct P value is attached to that descriptive number. The B3-B1 and B3-B0 rows expose the two planned cumulative contrasts.

**Table 1. Cumulative framework performance and paired inference.**

| Stage (inference contrast) | Macro-F1, mean ± SD | Incremental delta | Cumulative delta (reference) | Bootstrap 95% CI for listed contrast | Raw P | Holm P | Holm 0.05 | W/T/L | Fold-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B0 (reference) | 0.922606 ± 0.014541 | — | 0 | — | — | — | — | — | — |
| B1 (B1-B0) | 0.947237 ± 0.016764 | +0.024631 | +0.024631 vs B0 | [0.014113, 0.035642] | 0.000162303 | 0.000649214 | yes | 21/0/4 | [0.006578, 0.048280] |
| B2 (B2-B1) | 0.951050 ± 0.012393 | +0.003812 | +0.028443 vs B0, descriptive | [-0.001320, 0.008727] | 0.170241396 | 0.170241396 | no | 16/1/8 | [-0.003407, 0.010561] |
| B3 (B3-B2) | 0.953986 ± 0.012004 | +0.002937 | +0.031380 vs B0 | [-0.000478, 0.007294] | 0.058252952 | 0.116505904 | no | 13/8/4 | [-0.000220, 0.008263] |
| B3 (B3-B1 cumulative) | 0.953986 ± 0.012004 | — | +0.006749 vs B1 | [0.002389, 0.011270] | 0.013808562 | 0.041425685 | yes | 18/0/7 | [0.001295, 0.010597] |
| B3 (B3-B0 cumulative) | 0.953986 ± 0.012004 | — | +0.031380 vs B0 | [0.022821, 0.040301] | 2.98023×10^-7 | 1.49012×10^-6 | yes | 24/0/1 | [0.017174, 0.049576] |

![Cumulative performance from B0 to B3, direct paired contrasts, fold-level B3-B0 differences, and the four-stage framework.](../figures_journal/figure_cumulative_framework.png)

**Figure 1. Cumulative duration-aware hierarchical prototype framework.** B0 is the 1-s reference; B1 changes the event duration; B2 adds shared subtype supervision; B3 adds post-hoc prototype candidate inference. The paired panel contains both adjacent and cumulative contrasts. Error bars and intervals must be read with the reported contrast labels.

B1 provides the strongest independently supported gain. Its mean increment is 0.024631, and both the run-level bootstrap interval and fold-cluster interval are above zero, although the fold 0 mean is slightly negative. B2 adds 0.003812 in mean but has a confidence interval spanning zero and Holm P=0.170241. B3 adds 0.002937 over B2, with Holm P=0.116506. Neither adjacent increment is statistically significant.

The planned B3-B1 cumulative contrast survives Holm adjustment (delta 0.006749; Holm P=0.041426). This result covers two added stages and does not identify an independently significant contribution from B2 or B3. Relative to B0, the complete B3 framework yields a mean delta of 0.031380, 24 wins and one loss, and five positive fold-mean deltas (0.010828, 0.035278, 0.020681, 0.022855, and 0.067257). These contrasts involving B2/B3 remain retrospective and exploratory because the archived λ=0.5 selection rule is not pre-test or validation-only; Holm adjustment does not correct that model-selection limitation. The cumulative framework benefit is statistically supported within the frozen cohort, but it is not a factorial interaction test and is not evidence of synergy.

### 4.2 Prototype prediction cases illustrate output semantics

Figure 2 and Table 2 show the seven deterministic cases, including one low-margin uncertain illustration selected by rank rather than by a fitted rejection threshold. Abbreviations inside the distance column are C=cough, G=calm_grunt, F=feeding, S=stress_vocal, DC=dry_cough, AC=abdominal_cough, FS=frightened_stress, and AS=anxious_stress. Distances are cosine distances. The reported margin is the hierarchical-prototype Top-1 probability minus the Top-2 probability.

![Raw, main-prototype, hierarchical-prototype, and subtype candidates for seven deterministic cases, with distances, margins, hierarchy consistency, and training representatives.](../figures_journal/figure_prototype_prediction_cases.png)

**Figure 2. Prototype prediction cases.** The examples show one case from each main class, two feeding/stress boundary errors, and one low-margin case. `[C]` denotes hierarchy consistency, not prediction correctness. In this figure, a representative sample is **a training sample closest to the predicted class prototype** within the matching fold-seed training split. The cases illustrate output semantics and are not aggregate evidence.

**Table 2. Deterministic prototype output traces.**

| Case; true class | Raw Softmax Top-2 | Main-prototype Top-2 | Hierarchical Top-2 | Subtype Top-2 | Main distances; subtype distances | Margin; hierarchy consistency | Prototype representative |
|---|---|---|---|---|---|---|---|
| C1; cough | cough 0.999388<br>calm_grunt 0.000375 | cough 1.000000<br>calm_grunt 0.000000 | cough 1.000000<br>calm_grunt 0.000000 | abdominal_cough 0.502888<br>dry_cough 0.497112 | Main: C 0.001481; G 1.210585; F 1.619128; S 1.555124.<br>Subtype: DC 0.002865; AC 0.002056; G 1.210585; F 1.619128; FS 1.583670; AS 1.412413. | 1.000000; yes | `data/external/korea_raw/abdominal_cough/train_abdominal_02014.wav` (cough; prototype distance 0.001087) |
| C2; calm_grunt | calm_grunt 0.999843<br>cough 0.000059 | calm_grunt 1.000000<br>feeding 0.000000 | calm_grunt 1.000000<br>stress_vocal 0.000000 | calm_grunt 1.000000<br>frightened_stress 0.000000 | Main: C 1.179611; G 0.010405; F 1.146829; S 1.211714.<br>Subtype: DC 1.211066; AC 1.165473; G 0.010405; F 1.146829; FS 1.056907; AS 1.400100. | 1.000000; yes | `data/raw/sow_call_dataset_labeled/calm_grunt/264-0.wav` (calm_grunt; prototype distance 0.001744) |
| C3; feeding | feeding 0.965497<br>stress_vocal 0.034205 | feeding 0.981413<br>stress_vocal 0.018587 | feeding 0.978416<br>stress_vocal 0.021584 | feeding 0.975418<br>frightened_stress 0.022237 | Main: C 1.619153; G 1.187110; F 0.031840; S 0.309496.<br>Subtype: DC 1.610973; AC 1.622037; G 1.187110; F 0.031840; FS 0.296517; AS 0.453995. | 0.956831; yes | `data/raw/sow_call_dataset_labeled/feeding/371-1.wav` (feeding; prototype distance 0.004213) |
| C4; stress_vocal | stress_vocal 0.989793<br>feeding 0.009866 | stress_vocal 0.996073<br>feeding 0.003927 | stress_vocal 0.997813<br>feeding 0.002187 | anxious_stress 0.993573<br>frightened_stress 0.005979 | Main: C 1.425363; G 1.376879; F 0.561672; S 0.174148.<br>Subtype: DC 1.403346; AC 1.434603; G 1.376879; F 0.561672; FS 0.380191; AS 0.022279. | 0.995626; yes | `data/raw/sow_call_dataset_labeled/frightened_stress/646-2.wav` (stress_vocal; prototype distance 0.011431) |
| C5; feeding boundary error | stress_vocal 0.732443<br>feeding 0.265639 | stress_vocal 0.627354<br>feeding 0.372646 | stress_vocal 0.723336<br>feeding 0.276663 | frightened_stress 0.812211<br>feeding 0.180681 | Main: C 1.619579; G 1.056284; F 0.143859; S 0.107397.<br>Subtype: DC 1.625772; AC 1.616100; G 1.056284; F 0.143859; FS 0.038647; AS 0.370351. | 0.446673; yes | `data/raw/sow_call_dataset_labeled/frightened_stress/646-2.wav` (stress_vocal; prototype distance 0.011431) |
| C6; stress_vocal boundary error | feeding 0.654527<br>stress_vocal 0.344831 | feeding 0.691019<br>stress_vocal 0.308981 | feeding 0.693089<br>stress_vocal 0.306911 | feeding 0.695159<br>anxious_stress 0.209725 | Main: C 1.630480; G 1.321099; F 0.123649; S 0.179991.<br>Subtype: DC 1.613967; AC 1.637040; G 1.321099; F 0.123649; FS 0.262882; AS 0.207533. | 0.386178; yes | `data/raw/sow_call_dataset_labeled/feeding/371-1.wav` (feeding; prototype distance 0.004213) |
| C7; feeding, low-margin uncertain | feeding 0.516230<br>stress_vocal 0.482679 | feeding 0.644388<br>stress_vocal 0.355612 | feeding 0.619619<br>stress_vocal 0.380381 | feeding 0.594849<br>frightened_stress 0.377886 | Main: C 1.683404; G 1.157619; F 0.079341; S 0.120954.<br>Subtype: DC 1.679713; AC 1.684224; G 1.157619; F 0.079341; FS 0.111101; AS 0.295133. | 0.239238; yes | `data/raw/sow_call_dataset_labeled/feeding/371-1.wav` (feeding; prototype distance 0.004213) |

The four class examples demonstrate the additional candidate fields available when the prediction is correct. C5 and C6 are more informative about the boundary: both outputs are hierarchy-consistent yet wrong, showing that hierarchy consistency measures internal agreement rather than correctness. C7 retains the correct feeding label but exposes a much smaller margin and a plausible stress-vocal alternative. These examples make the output contract visible; their selected nature precludes using them as evidence for population-level accuracy or explanation quality.

### 4.3 Deployment boundary

The existing DEMAND analysis uses simulated additive noise, frozen clean checkpoints, train-only prototypes, and validation-only within-run calibration [@demand2018]. It is retained in the Supplementary Materials as a stress test. The calibration description does not cure the separate λ-selection limitation. The analysis does not demonstrate external real-farm robustness, and the severe 0-dB active-event condition reveals unresolved cough failures. The main conclusions therefore concern clean closed-set recognition and interpretable candidate outputs.

## 5. Discussion

The cumulative evidence supports a clear ordering of claims. Seeing a complete 2-s event is the main independently supported performance change. The B1-B0 effect is much larger than either later adjacent increment and remains significant after multiplicity adjustment. This result also explains why the tested feature-stacking and front-end alternatives did not become the mainline: in this cohort, event context mattered more than additional clean-audio complexity.

The semantic role of B2 is different. Joint subtype supervision places dry and abdominal cough, calm grunt, feeding, and the two stress subtypes in one shared embedding objective. Its small positive mean increment does not justify a claim of independent statistical improvement. The evidence supports a representation design that can expose finer candidates and help analyse feeding/stress boundaries.

B3 changes what the system returns. A main-class Softmax label becomes a structured candidate record with main and subtype rankings, distances, margins, consistency, uncertainty fields, and a class-prototype representative. The case studies show how these fields distinguish a confident correct example, a consistent but wrong boundary decision, and a low-margin correct decision. Their value is semantic and diagnostic. A user study would be needed to establish whether these explanations improve farm decisions.

Three inferential statements must remain separate. First, B2-B1 is not significant. Second, B3-B2 is not significant. Third, B3-B0 is significant and B3-B1 survives Holm adjustment. The latter two are cumulative comparisons; they cannot allocate the difference to a single intermediate module. Because the study is sequential rather than factorial, it cannot test an interaction term. “Cumulative benefit” is therefore appropriate; “synergy” is not.

The prior-art audit also narrows the novelty position. Two-second pig clips, coarse-to-fine pig pipelines, hierarchical prototypes, and interpretable acoustic prototype classifiers all exist in prior work. AudioProtoPNet is specifically a trainable prototype classifier, not a post-hoc method. The contribution here is the evaluated pig-specific route and its output contract under fold-seed-specific train-only prototype construction and validation-only within-run calibration; this wording does not extend to λ selection. The targeted search did not identify an exact route match, but the search cannot prove that none exists.

Published results from TransformerCNN, PVMC, AudioProtoPNet, HiSSNet, and other studies answer different questions. Numerical scores cannot be ranked when the data, classes, grouping, sample units, conditions, and metrics differ. The comparability table records those differences and prevents a mismatched superiority claim.

## 6. Limitations

First, the task is closed-set and based on existing third-party audio. No prospective external pig-farm cohort was evaluated. The supplementary DEMAND experiments add environmental recordings to clean events but do not reproduce the source geometry, reverberation, animals, sensors, and operational conditions of a new farm.

Second, exact-path, source-ID, and MD5 disjointness prevents direct identity overlap within each fold, but the Sow Call source lacks complete animal, session, farm, and parent-recording lineage. Latent correlated-source leakage cannot be excluded. Future evaluation should use independently collected, source-grouped cohorts with recoverable lineage.

Third, the six-subtype structure derives partly from source filename mappings. It is suitable for supervised representation and output analysis, but the learned geometry should not be interpreted as a validated disease ontology. Independent annotation and biological validation are needed.

Fourth, the 25 matched rows are five seeds nested in five test folds. Run-paired statistics are reported together with fold-cluster sensitivity, but five clusters provide limited precision. The borderline B3-B1 Holm result should be interpreted alongside its five-fold sensitivity rather than as 25 independent replications.

Fifth, B2-B1 and B3-B2 are not significant after Holm adjustment. The sequential design does not estimate a duration-by-hierarchy-by-prototype interaction and cannot support synergy. The complete B3-B0 contrast establishes a framework-level difference only.

Sixth, the archived record does not establish a pre-test or validation-only rule for choosing the λ=0.5 cohort used by B2 and B3. Their comparisons are retrospective/test-informed and exploratory; multiplicity adjustment does not remove this limitation.

Seventh, the deterministic case studies illustrate output semantics. They do not estimate explanation fidelity, calibration, diagnostic utility, or human decision benefit. A representative is selected relative to a class prototype and does not establish a causal acoustic feature.

Finally, this revision did not train, evaluate, or recalibrate any model. Its numerical conclusions inherit the scope and assumptions of the frozen 25-run artifacts.

## 7. Conclusion

The framework follows one practical sequence: see the complete event, learn the fine-grained structure, and predict with interpretable acoustic candidates. The 2-s duration change provides the strongest independently supported gain. Shared subtype supervision adds semantic organisation without a significant adjacent performance claim. Post-hoc prototypes expose candidate and distance information without a significant B3-B2 increment. Within the frozen, retrospectively selected λ=0.5 cohort, B3 significantly exceeds B0, with positive mean differences in all five folds; this framework-level contrast remains exploratory because the λ-selection record is test-informed.

This cumulative result supports the complete closed-set framework but does not establish module-wise significance, statistical interaction, or real-farm robustness. External source-grouped validation and prospective evaluation of explanation utility are the next requirements.

## Supplementary Materials

Detailed DEMAND results and the former main noise figures are assigned to the Supplementary Materials. For framework v2, Supplementary Figures S1-S5 map respectively to `figure_s1_clean_ablations`, `figure_s2_demand_environment_snr`, `figure_s3_confusion_calibration_fold_deltas`, `figure4_simulated_noise_robustness`, and `figure5_failure_selective_prediction`. Supplementary Tables S1-S5 map respectively to the clean-ablation table (`table2_clean_baseline_and_architecture_ablations.csv`); grouped and full DEMAND summaries (`table6_simulated_noise_robustness_by_stratum.csv`, `noise_25_run_summary.csv`); paired run/fold analyses (`table7_paired_statistics.csv`, `noise_25_run_cluster_bootstrap.csv`); per-class failures (`table8_per_class_noise_results_and_confusion_directions.csv`, `noise_25_run_per_class_summary.csv`); and selective-prediction results (`table9_selective_prediction_metrics_corrected_augrc.csv`, `noise_25_run_selective_summary.csv`, `noise_25_run_aurc_augrc_summary.csv`). This v2 map supersedes the legacy main-figure numbering for Figures 4/5. These materials define simulated deployment boundaries and are not evidence of real-farm validation.

## Data Availability

Processed aggregate tables and figure source data are available in the project repository and/or Supplementary Data associated with this article. They include cumulative summaries, paired statistics, figure source mappings, fold manifests where permitted, and hash-based leakage and provenance audits. Raw third-party pig audio is not redistributed. Smart Farm Korea/data.go.kr cough recordings should be obtained from the original provider [@smartfarmkorea_pig_cough_voice]. Sow Call Dataset recordings should be obtained from figshare [@sow_call_dataset_2021]. DEMAND environmental recordings should be obtained from Zenodo [@demand2018]. The unresolved Kaggle-like source is excluded from the main results.

## Code Availability

The historical `v0.1-paper-draft-r1` reproducibility package is available at https://github.com/HinataQAQ/pig-sound-classification/releases/tag/v0.1-paper-draft-r1, but it predates the framework-v2 cumulative and case-study artifacts. A versioned framework-v2 release is therefore required before submission. No software DOI is claimed. Model checkpoints, prototype artifacts, DEMAND audio, and raw third-party pig audio are not included unless released separately under compatible terms.

## Ethics Statement

No new animal experiment, animal intervention, or prospective farm recording was conducted for this manuscript. The study computationally reuses public or third-party pig-audio recordings and public DEMAND environmental recordings. Source-side permissions, repository terms, and animal-ethics approvals remain with the original data providers.

## References

Citation metadata are maintained in `paper/references/references.bib`.
