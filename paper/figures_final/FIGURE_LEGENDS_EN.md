# Final Functional Figure Legends (English)

## Figure 1 | Final evidence-bounded framework

Panels a-d separate path/source-ID/MD5-disjoint split roles, the 2-s Log-Mel shared hierarchical convolutional recurrent neural network (CRNN), and the inference roles. B2 Raw Softmax is the primary four-class route. In parallel, the frozen embedding feeds train-only main/subtype prototypes, Top-k candidates, distances, the predicted Top-1/Top-2 distance margin, hierarchy consistency and a prototype representative; calibration is validation-only and B3 remains an ablation. Here n=not applicable; no centre, SD, CI, Wilcoxon or Holm test applies. Pig-, session-, device- and farm-level grouping are not established by this schematic. Protocol elements are deployable in principle, but the schematic does not establish accuracy or external validity.

## Figure 2 | Primary validation-selected classification evidence

Panel a shows all 25 run points plus mean ± SD Macro-F1 on a 0-1 axis. Panel b shows fold-cluster bootstrap 95% CI for exactly three matched increments. B1−B0 is P=0.000162303448 (Holm not applicable: prespecified duration contrast outside the locked prototype family); B2−B1 is P=0.122846338 (Holm P=0.614231688); B3−B2 is P=0.477536357 (Holm P=0.955072713). B2/B3 Holm values preserve the locked final-validation family of seven comparisons. B1 is the main supported gain; B2 is exploratory and non-significant; B3 does not improve Top-1. Panel c shows validation-selected lambda by fold. This label-aware test evidence does not establish a deployable performance guarantee.

## Figure 3 | Prototype functional value

Panel a reports main/subtype Rank-1 and Rank-2 coverage and panel b reports feeding/stress Top-2 coverage. Panel c recomputes inconsistency prevalence, error precision, error recall and enrichment separately for Raw, Main and Hierarchical routes, displaying fold means and fold-cluster intervals; enrichment is undefined for zero-flag runs (Raw n=24, Main n=22, Hierarchical n=18). Panel d uses the same four-main-prototype second-nearest-minus-nearest margin for Main and Hierarchical Prototype, with 25-run correct/error distributions and AUROC fold-cluster CI; no hierarchical distance vector exists. n=25 total runs; metric-specific available-run counts are shown where needed. Centre is the mean of fold means, spread is available-run SD, and fold-cluster bootstrap 95% CI is primary where shown. No new Wilcoxon test is used and Holm status is not applicable. Only the margin and inconsistency flag are deployable; outcomes and coverage are label-aware. Low prevalence and recall limit the inconsistency flag; this does not establish safe automatic rejection.

## Figure 4 | Deterministic prototype candidate cases

Seven unique frozen-test roles are selected from fold 0/seed 42: C1 cough correct, C2 calm-grunt correct, C3 feeding correct, C4 stress-vocal correct, C5 feeding-boundary, C6 stress-boundary, and C7 low predicted margin. Panel d uses the hierarchical-predicted main class; each prototype representative is a training sample closest to the predicted class prototype. Here n=7 illustrative cases; centre, SD, fold-cluster bootstrap 95% CI, Wilcoxon and Holm are not applicable. The displayed margin is deployable but case correctness is label-aware. Cases do not establish prevalence.

## Supplementary Figure S1 | Complete clean ablations

Each row reports n=15 or n=25 fold-seed runs as mean ± sample SD; no fold-cluster bootstrap 95% CI, matched Wilcoxon test or Holm comparison is available for unmatched rows. Results are label-aware and do not establish a deployable advantage for an alternative clean frontend.

## Supplementary Figure S2 | Fixed-lambda retrospective cumulative evidence

Panel a reports all n=25 fixed-lambda=0.5 fold-seed trajectories and their mean; panel b reports matched mean deltas with stored 10,000-resample run bootstrap CI and five multiplicity-unadjusted numeric two-sided Wilcoxon P values (B1 - B0 P=0.000162303448; B2 - B1 P=0.170241396; B3 - B2 P=0.0582529521; B3 - B1 P=0.0138085615; B3 - B0 P=2.98023224e-07). Holm status is not used for this retrospective family. This run-level inference is descriptive, label-aware and does not replace the validation-selected primary result or establish a deployable gain.

## Supplementary Figure S3 | Lambda selection by fold

Each candidate has n=5 training-seed validation optima per fold; centre is mean and spread is sample SD. No test-set value, fold-cluster bootstrap 95% CI, Wilcoxon test or Holm decision enters selection. Selection is validation-deployable but does not establish one globally optimal lambda.

## Supplementary Figure S4 | Best epoch and validation-test gap

Boxplots summarise n=25 runs per stage using medians and interquartile ranges; no mean ± SD is encoded, and no fold-cluster bootstrap 95% CI, Wilcoxon or Holm test is added. Best epochs are validation-selected and gaps are label-aware frozen-test diagnostics; they do not justify test tuning.

## Supplementary Figure S5 | DEMAND simulated noise

Points and bars are n=25 fold-seed run means with sample SD under controlled additive DEMAND noise: DWASHING denotes domestic washroom/washing-machine noise, TBUS public-transit bus noise and STRAFFIC busy street-traffic noise. Panel d is the stored fixed-lambda=0.5 ALL_NOISY aggregate. No new fold-cluster bootstrap 95% CI, Wilcoxon or Holm test is added here. Performance is label-aware; noise generation is deployable as a test protocol but does not establish real-farm external validity.

## Supplementary Figure S6 | AURC and AUGRC

ALL_NOISY area under the risk-coverage curve (AURC) and area under the generalized risk-coverage curve (AUGRC) are descriptive n=25 means; lower is better. Spread, fold-cluster bootstrap 95% CI, paired Wilcoxon and Holm results are unavailable for these stored summaries. Scores are deployable uncertainty rankings, but risk outcomes are label-aware and do not establish a universal operating point.

## Supplementary Figure S7 | Label-aware true-class margins

True-main and true-subtype favorable-distance margins use n=25 runs; centre is the mean of run medians and variability is the displayed fold-cluster bootstrap 95% CI (SD is not encoded). No additional Wilcoxon or Holm test is made. These quantities explicitly use the true label, are not deployable, and do not establish a hierarchical decision distance.

## Statistical and evidence conventions

Unless stated otherwise, n=25 denotes matched fold-seed runs; centre is the mean of five fold means and spread is the available-run sample SD. Intervals are deterministic fold-cluster bootstrap 95% CI after averaging seeds within each of five folds. Paired tests use two-sided Wilcoxon signed-rank P values from SciPy method=auto (asymptotic when zero differences preclude an exact null distribution). Numeric P values and the applicable Holm adjustment are reported without significance stars. A deployable score or flag is label-free at use time, whereas every correctness, error, AUROC, precision, recall, and true-class margin is label-aware evaluation. The evidence does not establish real-farm external validity or automatic routing.
