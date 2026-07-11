# Prototype Prediction-Value Plan

Date: 2026-07-11
Scope: frozen-output analysis design. No model was retrained and no test-derived parameter is authorized.

## Frozen cumulative checkpoint

All four stages contain exactly the same 25 fold-seed keys: folds `0-4` × seeds `42, 123, 777, 2024, 3407`. There are no missing, extra, or duplicate pairs and no value was imputed.

| Stage | Locked definition | n | Mean Macro-F1 | Sample SD |
|---|---|---:|---:|---:|
| B0 | cap3x, 1 s Log-Mel CRNN | 25 | 0.922606467 | 0.014540638 |
| B1 | cap3x, 2 s Log-Mel CRNN | 25 | 0.947237358 | 0.016763541 |
| B2 | 2 s + hierarchical supervision, lambda=0.5, Raw Softmax | 25 | 0.951049673 | 0.012393314 |
| B3 | B2 checkpoint + `hierarchical` prototype inference | 25 | 0.953986214 | 0.012003773 |

B2 in `hier_longcontext_runs.csv` equals `raw_softmax_macro_f1` in the exact prototype cohort for every key (maximum absolute difference `0`). B3 is specifically `hierarchical_macro_f1`; it must not be substituted with main prototype, fused, calibrated Softmax, or a per-run best method.

Statistics below use the mean of the 25 paired deltas, sample SD (`ddof=1`), 10,000 percentile bootstrap resamples with `numpy.default_rng(3407)`, a two-sided Wilcoxon signed-rank test, and a separate bootstrap of the five fold-mean deltas. Fold values are listed in order 0→4.

| Contrast | n | Mean delta | SD delta | Run bootstrap 95% CI | Wilcoxon p | W/T/L | Fold-level mean deltas | Fold-cluster 95% CI |
|---|---:|---:|---:|---|---:|---:|---|---|
| B1−B0 | 25 | 0.024630891 | 0.028534613 | [0.014112838, 0.035642072] | 0.000162303 | 21/0/4 | −0.000170; 0.029243; 0.009924; 0.013381; 0.070776 | [0.006577865, 0.048280218] |
| B2−B1 | 25 | 0.003812315 | 0.012940365 | [−0.001320423, 0.008726684] | 0.170241396 | 16/1/8 | 0.012167; −0.007300; 0.009552; 0.009366; −0.004724 | [−0.003406537, 0.010560902] |
| B3−B2 | 25 | 0.002936541 | 0.010335613 | [−0.000477565, 0.007294065] | 0.058252952 | 13/8/4 | −0.001169; 0.013335; 0.001204; 0.000107; 0.001205 | [−0.000219515, 0.008263370] |
| B3−B1 | 25 | 0.006748856 | 0.011611119 | [0.002388569, 0.011270120] | 0.013808562 | 18/0/7 | 0.010998; 0.006035; 0.010756; 0.009474; −0.003519 | [0.001295329, 0.010596604] |
| B3−B0 | 25 | 0.031379747 | 0.023102131 | [0.022821112, 0.040301488] | 0.000000298 | 24/0/1 | 0.010828; 0.035278; 0.020681; 0.022855; 0.067257 | [0.017174469, 0.049575547] |

Interpretation guard: B2−B1 and B3−B2 are nonsignificant and both run and fold intervals include zero. B3−B1 is an unadjusted cumulative contrast; with Bonferroni correction for these five contrasts (`alpha=0.01`) it does not pass. Seeds within a fold reuse test clips, so the five-fold sensitivity is essential and remains low-powered.

## Analysis cohort and unit

Use the 25 exact lambda=0.5 prototype runs. Perform every calculation within one fold-seed first; only then aggregate across matched runs. Never concatenate repeated-seed test predictions as if they were independent clips. Verify `feature_backend=training_exact`, `input_role=frozen_test`, manifest hashes, Softmax reproduction, and leakage-audit success before including a run.

The primary unit is a test prediction. Uncertainty intervals and paired method comparisons are summarized over fold-seed runs, with five-fold sensitivity. Where repeated clips across seeds are analyzed, cluster or summarize by canonical `source_id` and fold rather than treating repetitions as new audio.

## 1. Main Top-2 value

For each test clip, rank all four main classes using the stored frozen method scores. Report:

- Top-1 and Top-2 accuracy;
- mean reciprocal rank and distribution of correct-class ranks 1-4;
- Top-2 inclusion by main class and by correct/incorrect Top-1 status;
- paired Raw Softmax versus main-prototype versus hierarchical-prototype deltas.

Top-2 is intrinsically easier in a four-class problem; disclose the random-ranking ceiling and do not treat a high Top-2 result as a replacement for Top-1/Macro-F1.

## 2. Subtype Top-2 value

Rank the six subtype candidates from frozen auxiliary/prototype scores. Report Top-1, Top-2, reciprocal rank, and per-subtype support. If complete subtype scores are absent from an output, mark the run/metric unavailable or perform a separately authorized inference-only export; never reconstruct ranks from hard labels.

## 3. Hierarchy consistency

Define consistency as agreement between the hierarchical main prediction and the main label obtained by mapping the predicted subtype through the fixed subtype-to-main map. Report:

- overall consistency;
- consistency stratified by true main class, true subtype, and correctness;
- the four states: consistent-correct, consistent-wrong, inconsistent-correct-main, inconsistent-wrong;
- Raw Softmax→hierarchical rescued and harmed transitions.

Consistency is not correctness. A confidently wrong main/subtype pair can be perfectly consistent.

## 4. Correct-class prototype rank

Rank main prototypes (four) and subtype prototypes (six) by ascending cosine distance. For the true class/subtype report rank-1, rank-2, median rank, mean reciprocal rank, and per-class distributions. Preserve ties with a deterministic class-index tie break and report the tie count.

## 5. Representative-sample retrieval purity

Build each retrieval bank only from the matching run's training embeddings. For each prototype, retrieve the `K` nearest training examples under the frozen distance; predeclare `K` (for example 5 and 10) and report:

- same-label purity at the main and subtype levels;
- unique-`source_id` purity and duplicate rate;
- per-prototype and macro averages;
- coverage: prototypes/classes with fewer than K eligible unique sources;
- exact path/source/MD5 exclusion checks against the test case.

Validation or test labels must never enter the bank. Representative retrieval is explanatory evidence, not an additional training or calibration step.

## 6. Distance margins

For each example define the favorable main margin as:

`nearest_wrong_prototype_distance - true_class_prototype_distance`.

Define the subtype margin analogously. Positive values mean the true prototype is closer. Compare correct and incorrect predictions using within-run median/IQR, effect size, and paired fold-seed summaries. Do not choose a threshold on test data. Any operational threshold is selected on validation and applied once to test.

## 7. Feeding/stress boundary

Predeclare the boundary subset using ground truth and predicted labels, not visual appeal. Report:

- feeding→stress and stress→feeding directional counts/rates;
- frightened versus anxious subtype strata;
- signed feeding-versus-stress distance margin;
- whether both boundary classes appear in Top-2;
- Raw Softmax→prototype rescued, harmed, unchanged-correct, and unchanged-wrong transitions;
- representative retrieval purity for the confused pair.

Do not infer biological mechanism from embedding distance alone.

## 8. Confidence and distance calibration

Compare Raw Softmax, validation-calibrated Softmax, main prototype, and hierarchical prototype using NLL, multiclass Brier score, ECE with a predeclared binning rule, classwise calibration, and reliability curves. Temperatures, fusion weights, rejection thresholds, and distance-to-confidence mappings are learned on validation only. Test calibration is evaluation only.

Report calibration separately from discrimination; improved Macro-F1 does not imply improved calibration or selective-risk ranking.

## 9. Deterministic examples

Use a predeclared canonical run, fold 0/seed 42. Select examples by a deterministic table rule:

1. for each main class, among examples correctly classified by all compared methods, choose the lower median Raw Softmax confidence;
2. for feeding/stress boundary errors, choose the lower median absolute Raw Top-2 margin for each direction;
3. add the remaining example with minimum favorable prototype margin;
4. break all ties by `source_id`, then path;
5. retrieve representatives only from the matching training split.

Publish the full eligible table and selection flag so that chosen examples are reproducible. Do not substitute “better-looking” spectrograms after inspection.

## Output contract for a future authorized analysis

Produce new, uniquely named run-level and aggregate files containing metric definitions, cohort keys, manifest/result hashes, missing-field status, and leakage checks. At minimum include a run table, fold table, paired-statistics table, calibration table, boundary table, retrieval-purity table, and deterministic-case table. Source result CSVs and existing checkpoints remain read-only.
