# Novelty Claim Guard

Status: literature-audit guard, 2026-07-11. This document does not change the locked manuscript conclusions.

## Search-bounded result

The required publisher/preprint audit covered Wu et al., TransformerCNN, Briefer et al., PVMC, HiSSNet, AudioProtoPNet, Prototypical Networks, and the DCASE 2022 few-shot bioacoustic overview. Targeted exact-route searches also checked recent pig-classification and subject-grouped pig studies.

No exact pig-specific match was located for the complete route:

> fixed 2 s pig-vocalisation input + shared main/subtype supervision + fold-seed-specific post-hoc train-only hierarchical prototypes + ranked main/subtype candidates + train-example retrieval.

This means only **no exact match was located in the targeted search**, not that no such work exists anywhere. The safe qualifier is “to the best of our targeted search,” accompanied by the search scope and date. Never convert this result into an absolute “first” claim.

The closest precedents are complementary rather than identical:

- Wu and Liao: pig-specific fixed 2 s classification and overlapping behavioral labels.
- PVMC: pig-specific coarse-to-fine pipeline, but separately trained stages rather than shared auxiliary supervision.
- HiSSNet: joint hierarchy-level losses and prototype inference, but non-pig, 1 s, and episodic/end-to-end rather than post-hoc.
- AudioProtoPNet: a trainable prototype classifier replacing the ordinary classification layer, with retrieval of similar training examples, but flat bird-species classification; it is not post-hoc.
- ProtoNet and DCASE: few-shot candidate/prototype inference, not post-hoc closed-set pig inference.

## Not allowed

The following claims are unsupported and must not appear:

- “first use of 2 s pig audio”;
- “first hierarchical sound classifier”;
- “first prototypical audio classifier”;
- “first interpretable bioacoustic prototype”;
- “state-of-the-art pig vocalization accuracy” without a matched benchmark;
- any unqualified “first,” “first ever,” “novel for the first time,” or exhaustive-prior-art statement;
- “our method outperforms TransformerCNN/PVMC” when dataset, labels, split, metric, sample unit, acoustic condition, or class balance differ;
- “hierarchical supervision significantly improves the 2 s baseline”;
- “hierarchical prototypes significantly improve Raw Softmax” on the locked clean cohort;
- “source-independent Sow Call validation” when grouping uses the current singleton filename-derived `source_id`;
- “open-set recognition,” “unknown-class rejection,” or real-farm external validation from the current closed-set and simulated-noise evidence.

## Potentially defensible

Subject to precise evidence citations and the qualifier above, the following route-level framing is potentially defensible:

- leakage-controlled cumulative evaluation of 2 s context + shared four-main/six-subtype supervision + post-hoc hierarchical prototype candidate inference;
- train-only, fold-seed-specific prototype construction;
- validation-only temperature, weight, threshold, and fusion calibration;
- Top-k main/subtype candidates, distance margin, and hierarchy consistency for pig-vocalisation recognition;
- training-example retrieval as an analysis layer only when the retrieval bank is restricted to the matching run's training split;
- the combination of elements as the contribution, rather than any one element in isolation.

Safe example:

> To the best of our targeted search through 11 July 2026, prior work separately covers fixed-duration pig-vocalisation classification, coarse-to-fine pig pipelines, hierarchical prototypical learning, and prototype-based training-example retrieval; we did not locate a study combining these elements under fold-seed-specific train-only prototype construction and validation-only calibration.

Here, validation-only calibration refers to prototype temperatures, weights,
thresholds, and uncertainty criteria within each frozen run. It does not apply
to the archived choice of `lambda=0.5`, for which no pre-test or validation-only
selection rule is documented; comparisons involving that cohort are
retrospective/test-informed and exploratory.

## Statistical guard

The frozen 25-pair analysis gives the following claim boundaries:

- B1−B0 (2 s versus 1 s): mean Macro-F1 delta `0.024631`, Wilcoxon `p=0.000162`; this is the strongest supported incremental result.
- B2−B1 (lambda=0.5 hierarchical supervision versus 2 s): delta `0.003812`, `p=0.170241`; not statistically significant.
- B3−B2 (hierarchical prototype inference versus Raw Softmax): delta `0.002937`, `p=0.058253`; not statistically significant.
- B3−B1: cumulative delta `0.006749`, unadjusted `p=0.013809`, Holm-adjusted `p=0.041426`; it survives Holm but does not prove that either hierarchy or prototype increment is independently significant, and it does not survive Bonferroni correction across the five planned contrasts at `alpha=0.01`.
- B3−B0: cumulative delta `0.031380`, but this must not be presented as causal evidence for each intermediate component.

The 25 runs reuse five test folds across seeds. Report run-paired results together with five-fold sensitivity; do not describe 25 seeds/folds as 25 independent datasets.

## Score-language guard

The external numbers in `PUBLISHED_SCORE_COMPARABILITY.csv` may be described as numerically higher or lower only with the metric and protocol attached. “Outperforms” is reserved for a direct comparison in which dataset, labels, split, metric, sample unit, clean/noise condition, and class balance all align. The present audit found no such external comparison.
