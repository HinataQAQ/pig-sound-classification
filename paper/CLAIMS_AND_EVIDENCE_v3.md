# Claims and Evidence Ledger v3

## Scope and status

This ledger governs `paper_en_framework_v2.md` and `paper_cn_framework_v2.md`.
It is a manuscript-only revision based on frozen, pre-existing result files. No
model was trained or evaluated, no prediction was regenerated, and no clean or
noise result CSV/JSON was changed in this revision.

## Central organising statement

> See the complete event, learn the fine-grained structure, and predict with
> interpretable acoustic candidates.

The statement describes a sequence of framework functions. It is not a claim
of factorial interaction, statistical synergy, or independent significance for
every stage.

## Framework stages

| Stage | Locked definition | Paper role |
|---|---|---|
| B0 | 1-s Log-Mel CRNN | Short-context reference |
| B1 | 2-s duration-aware Log-Mel CRNN | Best-supported independent increment |
| B2 | B1 plus shared four-main/six-subtype auxiliary supervision at `lambda=0.5`, evaluated by Raw Softmax | Fine-grained semantic structure |
| B3 | B2 plus fold-seed-specific post-hoc hierarchical prototype candidate inference | Ranked candidates and interpretable inference fields |

B2 and B3 use the pre-existing frozen `lambda=0.5` prototype cohort; this
manuscript revision did not select or change that cohort. The archived record
reports that `lambda=0.5` had the lowest cross-run test Macro-F1 standard
deviation, but it does not document a pre-test or validation-only selection
rule. The choice is therefore treated as retrospective and test-informed.
Comparisons involving B2 and B3 are exploratory and do not establish an
independently selected optimum. Holm adjustment of the five planned contrasts
does not correct this model-selection limitation.

## Primary empirical claims

| ID | Claim | Evidence | Permitted wording | Boundary |
|---|---|---|---|---|
| C1 | B1 improves substantially over B0. | Mean delta `0.02463089099039724`; bootstrap 95% CI `[0.014112837971563999, 0.03564207235269266]`; Wilcoxon raw `P=0.00016230344772338867`; Holm `P=0.0006492137908935547`; W/T/L `21/0/4`; fold-cluster CI `[0.006577865394051571, 0.048280218219115614]`. | “B1 provides the strongest independently supported performance gain.” | Fold 0 mean delta is slightly negative; do not claim universal per-fold improvement. |
| C2 | B2 introduces subtype-aware supervision and has a small positive mean increment over B1. | Mean delta `0.0038123148824640517`; bootstrap CI `[-0.0013204227591915722, 0.008726684136810842]`; raw `P=0.17024139590961385`; Holm `P=0.17024139590961385`. | “B2 adds fine-grained semantic structure.” | The B2-B1 increment is not statistically significant. |
| C3 | B3 exposes main/subtype candidates, distances, margins, hierarchy consistency, uncertainty fields, and training representatives. | Frozen prototype artifacts and `prototype_prediction_case_studies.csv`; B3 mean Macro-F1 `0.9539862142299651`. | “B3 adds candidate prediction and interpretability.” | The B3-B2 increment is not statistically significant: raw `P=0.05825295212511276`, Holm `P=0.11650590425022552`. |
| C4 | The complete B3 framework outperforms B0 on the matched 25-run cohort. | Mean delta `0.031379746893681656`; bootstrap CI `[0.02282111227854716, 0.04030148837924802]`; raw `P=2.980232238769531e-07`; Holm `P=1.4901161193847656e-06`; W/T/L `24/0/1`; five positive fold means; fold-cluster CI `[0.01717446912102002, 0.049575547432752186]`. | “The complete B3 framework significantly outperforms B0.” | A cumulative comparison does not prove that each intermediate component is independently significant. |
| C5 | B3 differs from B1 across the two-stage cumulative addition. | Delta `0.006748855903284414`; raw `P=0.013808561544817114`; Holm `P=0.04142568463445134`; W/T/L `18/0/7`; fold-cluster CI `[0.0012953292136365802, 0.01059660372696845]`. | “The planned B3-B1 cumulative contrast survives Holm adjustment.” | It does not identify whether B2, B3, or their interaction caused the cumulative difference, and it does not establish synergy. |

All performance values above are Macro-F1. The 25 matched rows are five folds
by five model seeds, not 25 independent datasets. Fold-cluster intervals are
sensitivity analyses with only five clusters.

## Holm family and conclusions

The five pre-specified Wilcoxon P values were adjusted together with the
standard step-down Holm procedure. No model outputs or predictions were
recomputed.

| Holm rank | Contrast | Raw P | Multiplier | Holm-adjusted P | Significant at 0.05 |
|---:|---|---:|---:|---:|---|
| 1 | B3-B0 | `2.980232238769531e-07` | 5 | `1.4901161193847656e-06` | yes |
| 2 | B1-B0 | `0.00016230344772338867` | 4 | `0.0006492137908935547` | yes |
| 3 | B3-B1 | `0.013808561544817114` | 3 | `0.04142568463445134` | yes |
| 4 | B3-B2 | `0.05825295212511276` | 2 | `0.11650590425022552` | no |
| 5 | B2-B1 | `0.17024139590961385` | 1 | `0.17024139590961385` | no |

The adjusted values are already monotone in sorted order, so the cumulative
maximum step does not change them.

## Contribution claims

1. A matched duration study shows that 2-s context substantially improves the
   clean closed-set task and is more effective than the tested complex
   front-end alternatives. The matched inferential evidence applies to 2 s
   versus 1 s; the wider clean-ablation comparison is descriptive because its
   run counts and pairings differ.
2. A shared four-main/six-subtype supervision scheme introduces fine-grained
   health and behavioural semantics into one acoustic embedding space.
3. A fold-seed-specific post-hoc prototype inference layer converts a single
   Softmax output into main/subtype Top-k candidates, prototype distances,
   decision margins, hierarchy consistency, uncertainty flags, and prototype
   representatives.

Contribution 3 is an output-semantics and interpretability contribution. The
prototype increment alone must not be described as statistically significant.

## Prior-art positioning

| Topic | Required position | Citation key |
|---|---|---|
| Fixed 2-s pig audio | Prior use exists in the PLOS pig-speech study and TransformerCNN. | `wu2022pigspeech`; `liao2023transformercnn` |
| Coarse-to-fine pig classification | Prior use exists in PVMC, implemented as separately trained stages. | `chung2025pvmc` |
| Hierarchical prototypes | Prior use exists in HiSSNet for non-pig episodic hierarchical prototypical learning. | `shashaank2023hissnet` |
| Interpretable acoustic prototypes | AudioProtoPNet uses a trainable prototype classifier that replaces the ordinary classification layer; it is not a post-hoc prototype method. | `heinrich2025audioprotopnet` |
| Embedding vocabulary and possible subtype discovery | Briefer et al. provide pig-call embedding/vocabulary evidence relevant to finer acoustic organisation. | `briefer2022pigcalls` |
| Route-level gap | No exact pig-specific route match was identified in the targeted search through 11 July 2026. | `PRIOR_ART_OVERLAP_MATRIX_v2.csv`; `NOVELTY_CLAIM_GUARD.md` |

The route-level gap is search-bounded. It is not an absolute first claim.
Published scores are not ranked because datasets, labels, split membership,
grouping, sample units, acoustic conditions, class balance, and metrics differ.
`PUBLISHED_SCORE_COMPARABILITY.csv` is explanatory, not a leaderboard.

## Prototype case-study interpretation

The seven deterministic cases comprise one cough, one calm grunt, one feeding,
one stress vocalisation, two feeding/stress boundary errors, and one low-margin
uncertain case. Every case reports true class, Raw Softmax Top-2, main-prototype
Top-2, hierarchical-prototype Top-2, subtype Top-2, main/subtype distances,
margin, hierarchy consistency, and a prototype representative.

The representative definition used in both figure legends is:

> a training sample closest to the predicted class prototype

The representative is drawn from the matching run's training split. It is not
defined by query-to-training distance. The selected cases illustrate output
semantics and failure interpretation; they are not aggregate performance
evidence. Hierarchy consistency denotes agreement between the hierarchical
main output and the main class mapped from the predicted subtype. It does not
imply that the prediction is correct.

The source run is the pre-specified first numeric key in the canonical key list
(`fold=0`, `seed=42`) after artifact validation, not the first filename in
lexical order. Correct cases use the lower-median Raw Softmax confidence within
each jointly correct class pool. Each boundary case uses the lower-median Raw
Top-2 margin within its eligible feeding/stress error pool. The uncertain case
uses the smallest remaining hierarchical Top-2 margin. These are post-hoc
display rules applied to frozen test outputs, not parameter selection.

## Noise and deployment boundary

Detailed DEMAND results, per-class noise results, and selective-prediction
figures remain supplementary evidence. The main abstract contains no
`ALL_NOISY` Macro-F1. The main text may state only that simulated additive-noise
results define a deployment boundary and do not establish real-farm robustness.
No open-set, few-shot, or unknown-class claim is supported by this manuscript.

## Prohibited claims

- state of the art or superiority over published studies without a matched
  benchmark;
- every module is independently significant;
- B2-B1 or B3-B2 is statistically significant;
- a synergistic interaction without a factorial interaction test;
- the B3-B0 cumulative benefit proves causality for each intermediate stage;
- AudioProtoPNet is post-hoc;
- the prototype representative is selected by query-to-training proximity;
- selected cases are aggregate evidence;
- real-farm robustness, open-set recognition, few-shot recognition, or
  unknown-class rejection;
- 25 fold-seed rows are 25 independent datasets.

## Evidence-file map

| Evidence | Canonical file |
|---|---|
| Stage summaries | `paper/tables/cumulative_framework_summary.csv` |
| Wider locked clean-ablation context | `paper/tables/table2_clean_baseline_and_architecture_ablations.csv`; `paper/supplementary/figures/figure_s1_clean_ablations.{svg,pdf,png,tiff}` |
| Five raw paired contrasts | `paper/tables/cumulative_framework_paired_stats.csv` |
| Holm-adjusted paired contrasts | `paper/tables/cumulative_framework_paired_stats_holm.csv` |
| Fold sensitivity | `paper/tables/cumulative_framework_fold_stats.csv` |
| Run provenance | `paper/tables/cumulative_framework_runs.csv` |
| Case studies | `paper/tables/prototype_prediction_case_studies.csv` |
| Framework figure | `paper/figures_journal/figure_cumulative_framework.{svg,pdf,png}` |
| Case-study figure | `paper/figures_journal/figure_prototype_prediction_cases.{svg,pdf,png}` |
| Corrected prior-art matrix | `paper/references/PRIOR_ART_OVERLAP_MATRIX_v2.csv` |
| Published-score comparability | `paper/references/PUBLISHED_SCORE_COMPARABILITY.csv` |
| Novelty guard | `paper/references/NOVELTY_CLAIM_GUARD.md` |

## Remaining scientific limitations

1. The study is closed-set and uses existing third-party pig audio; it has no
   prospective external real-farm validation.
2. Exact-path, source-ID, and MD5 disjointness is verified, but animal, session,
   farm, and parent-recording lineage is incomplete for part of the Sow Call
   source, so latent correlated-source leakage cannot be excluded.
3. The six-subtype hierarchy is useful for supervision and interpretation but
   requires external biological validation beyond filename-derived source
   labels.
4. The 25 matched rows reuse five test folds; the fold-cluster sensitivity has
   only five clusters.
5. B2-B1 and B3-B2 are not significant after Holm adjustment. The design is not
   factorial and cannot estimate statistical interaction or synergy.
6. Deterministic case studies are illustrative. They do not replace aggregate
   validation, calibration analysis, or user studies of explanation utility.
7. Prototype representatives summarise a predicted class centre; they do not
   establish causal acoustic features or clinical diagnoses.
8. The archived provenance does not establish a pre-test or validation-only
   rule for selecting the `lambda=0.5` cohort used by B2 and B3. Those
   comparisons are therefore retrospective/test-informed and exploratory;
   Holm adjustment does not remove this limitation.
