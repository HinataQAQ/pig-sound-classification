# Final Functional Figure Audit

**Date:** 2026-07-11

**Workflow:** Academic Research Suite `ars-reviewer`, five-role panel, read-only review

**Accepted interpretation under review:** `framework_functional_only`

**Editorial conclusion:** **ACCEPT under the frozen contract, with bounded reporting limitations**

## Scope and precommitment

The panel reviewed the four main and seven supplementary figures, both legend
files, the 30-panel source map, the 30-panel claim matrix, the QA report, the
three newly derived CSVs, and their frozen source artifacts. Reviewers did not
edit the package. Author-side corrections prompted by the first review wave
were regenerated and rechecked before this synthesis.

The scoring dimensions were D1 methodology rigor, D2 domain accuracy, D3
argumentative coherence, D4 cross-disciplinary relevance, and D5 writing and
structure. D1-D3 were mandatory. The precommitted failure conditions were:

- F1: any mandatory dimension at `block`;
- F2: at least three of five reviewers each scoring two or more mandatory
  dimensions `warn` or worse;
- F3: any high-priority D4 score at `block`;
- F0: acceptance-grade mandatory scores when F1-F3 do not fire.

## Five-role panel

| Role | D1 | D2 | D3 | D4 | D5 | Decision | Disposition |
|---|---|---|---|---|---|---|---|
| Editor-in-chief, superseding final card | pass | pass | pass | pass | pass | accept | No remaining contract-level issue. |
| Methodology reviewer, superseding final card | pass | pass | pass | pass | pass | accept | Earlier scope/statistics warnings were verified as closed. |
| Domain reviewer, first-wave card | pass | pass | pass | pass | warn | accept | Figure 4 route and Chinese-legend details were corrected after the card. |
| Perspective reviewer, final corrected package | pass | pass | pass | pass | pass | accept | Independently reproduced 1,225 run-level fields to within `2.22e-16`. |
| Devil's advocate, final corrected package before the last parity patch | pass | pass | pass | warn | warn | accept | D4 is a retained scope warning; its two D5 legend omissions were then corrected. |

F1 did not fire: no reviewer blocked D1-D3. F2 did not fire: no final reviewer
card had two mandatory warnings, far below the three-reviewer threshold. F3 did
not fire: D4 was never blocked. The panel therefore reaches F0/accept. The
devil's-advocate D4 warning is retained as a limitation, not converted into an
operational-readiness claim.

## Required scientific checks

| Review question | Result | Evidence-based conclusion |
|---|---|---|
| Does any figure imply that B3 improves Top-1? | PASS | Figure 2 reports B3-B2 mean delta `-0.001916`, fold-cluster 95% CI `[-0.004631, 0.000799]`, raw Wilcoxon `P=0.477536357`, and locked-family Holm `P=0.955072713`. B3 is labeled an ablation and non-superior. |
| Is predicted margin separated from true-label margin? | PASS | The main-prototype second-nearest-minus-nearest gap is label-free at use time. Correct/error partitions and AUROC are explicitly label-aware evaluation. True-class geometry is confined to S7 and labeled non-deployable. No hierarchical cosine-distance vector is invented. |
| Are hierarchy inconsistency prevalence and recall visible? | PASS | Figure 3 shows prevalence, precision, recall, enrichment, fold-cluster intervals, and metric-specific availability. Mean prevalence is only `1.64%`, `1.14%`, and `0.83%`; error recall is only `11.64%`, `13.76%`, and `11.60%` for Raw, Main, and Hierarchical routes. The flag is not presented as a safe rejector. |
| Are fixed-lambda results retrospective? | PASS | S2 is explicitly fixed-lambda `0.5`, retrospective, run-bootstrap, descriptive, and multiplicity-unadjusted. It does not replace the validation-selected primary chain. |
| Are noise results supplementary? | PASS | DEMAND simulated-noise and AURC/AUGRC results remain in S5-S6, are labeled simulated/fixed-lambda evidence, and do not claim real-farm validation or a universal operating point. |
| Are all figure claims source-backed? | PASS | Source and claim maps contain one unique row for each of 30 panels. All 151 source-map path/SHA-256 pairs matched; the claim matrix mirrors them, yielding 302 verified link occurrences across both maps. The full 921-source before/after audit also passed. |

## Corrections closed during review

- Narrowed the disjointness statement to exact path/source-ID/MD5 identity and
  disclosed that pig-, session-, device-, and farm-level grouping is not
  established.
- Made S2's run-level bootstrap and unadjusted multiplicity status visible.
- Added fixed-lambda `0.5` to S5 and expanded the DEMAND environment labels.
- Expanded CRNN, AURC, and AUGRC; S6 now states that uncertainty summaries are
  unavailable in the frozen source.
- Labeled Figure 4's representative as following the hierarchical predicted
  main-class route and retained the exact training-representative definition.
- Brought the Chinese legends into decision-relevant semantic parity, including
  Figure 1 deployability scope and Figure 2 point, axis, increment, and
  fold-wise-selection details.
- Made the approved `lambda_selection_by_fold.csv` authoritative for cohort
  choice. Recalculation from 75 validation summaries is now reproduction-only
  and hard-fails on mismatch; the locked CSV plus all summaries are hashed
  before discovery.

## Accepted conclusions

1. B2 validation-selected Raw Softmax remains the primary four-class
   classifier. The supported major gain is the 2-second context contrast.
2. Prototype outputs support a parallel candidate-prediction and diagnostic
   function. Their distance margin is strongly associated with errors in
   label-aware evaluation, but no operating threshold or deployment guarantee
   is selected here.
3. Hierarchy inconsistency is rare and error-enriched, with low error recall.
   It is a diagnostic flag, not an automatic rejection or routing policy.
4. Hierarchical Prototype harmed more repeated predictions than it rescued in
   the frozen cohort (`37` harmed versus `29` rescued; `0` both wrong in the
   disagreement subset). It must not be described as the superior Top-1 route.

## Bounded limitations that remain

- Disjointness is demonstrated for path, source ID, and MD5 identity; animal,
  recording-session, device, barn, and farm grouping are not established.
- Fold-cluster intervals resample only five fold means and are necessarily
  coarse. S2 instead preserves its stored retrospective run-bootstrap intervals
  and five unadjusted Wilcoxon tests.
- S6 is descriptive because the frozen AURC/AUGRC summary does not contain
  uncertainty estimates.
- DEMAND additive noise is a controlled simulation, not real pig-farm evidence.
- No welfare-state inference, automatic rejector, operational cost mapping,
  universal threshold, or real-farm external validity is established.
- The QA report records the complete 921-source hash count and pass/fail result;
  the persisted panel maps contain the directly cited evidence hashes rather
  than a separate full-inventory digest ledger.

## Final decision

The package passes all six requested review questions and the precommitted
failure logic. It is paper-usable for the `framework_functional_only` narrative
provided the limitations above remain adjacent to the claims. No new training,
inference, hyperparameter selection, noise experiment, or manuscript expansion
is justified by this audit.
