# Framework v2 Claim Audit

## Verdict

**PASS WITH MATERIAL LIMITATION for transparent manuscript-only scientific
framing, subject to the remaining submission-packaging items listed below.**
The v2 manuscripts use the frozen cumulative results without training,
evaluation, inference, recalibration, or modification of established
clean/noise result CSV/JSON files. The material limitation is that the archived
record does not establish a pre-test or validation-only selection rule for the
`lambda=0.5` cohort used by B2 and B3. The manuscripts now label those
comparisons retrospective/test-informed and exploratory.

## Audited files

- `paper/manuscript/paper_en_framework_v2.md`
- `paper/manuscript/paper_cn_framework_v2.md`
- `paper/CLAIMS_AND_EVIDENCE_v3.md`
- `paper/tables/cumulative_framework_paired_stats_holm.csv`
- `paper/references/PRIOR_ART_OVERLAP_MATRIX_v2.csv`
- `paper/references/references.bib`
- the 14 approved artifacts imported from `paper/prior-art-fair-benchmark`

Trusted branch tips at import time were:

- manuscript base `paper/sci-draft-v1`:
  `b52a0f63539d544f27c57e29760384c0782e7f61`;
- artifact source `paper/prior-art-fair-benchmark`:
  `ccf3171be6c53a75299427bc1cf726377a9c1c29`.

Only the approved four cumulative CSVs, one prototype-case CSV, six figure
files, and three prior-art guard files were imported. Generator code, tests,
experiment changes, plans, benchmark feasibility files, and prior handoff
changes were not imported. After import, the stale AudioProtoPNet phrase in the
approved novelty guard was corrected within the requested prior-art scope. The
two generated SVGs received trailing-whitespace-only normalization so the full
Git whitespace check passes; their plotted content and companion PDF/PNG
renders are unchanged. The remaining 11 imports are blob-identical to the
source commit, and no cumulative/result CSV was altered.

The approval manifest was supplied in the user's attached request and is not a
tracked repository artifact. Git independently verifies the 14-file inventory
and source commit; repository history alone does not authenticate that external
approval record.

## Central-claim audit

| Required statement | Status | Evidence/wording check |
|---|---|---|
| B1 is the strongest independently supported gain. | pass | Both manuscripts report B1-B0 delta `0.024631`, raw `P=0.000162303`, and Holm `P=0.000649214`. |
| B2 adds fine-grained semantic structure. | pass | B2 is described as shared four-main/six-subtype supervision, not as an independently significant gain. |
| B3 adds candidate prediction and interpretability. | pass | Both manuscripts enumerate main/subtype Top-k, distances, margins, consistency, uncertainty fields, and representatives. |
| B2-B1 is not significant. | pass | Raw and Holm P are both `0.170241396`; wording is explicitly nonsignificant. |
| B3-B2 is not significant. | pass | Raw `P=0.058252952`; Holm `P=0.116505904`; wording is explicitly nonsignificant. |
| Complete B3 exceeds B0. | pass with material limitation | Delta `0.031380`, CI `[0.022821, 0.040301]`, raw `P=2.98e-7`, Holm `P=1.49e-6`, W/T/L `24/0/1`, and five positive fold means are reported, but the contrast is explicitly retrospective/exploratory because `lambda=0.5` lacks a documented pre-test selection rule. |
| No interaction/synergy claim. | pass | Both manuscripts state that the sequential design is not factorial and cannot test interaction or synergy. |

The B3-B1 cumulative contrast survives Holm correction (`P=0.041425685`) but
is consistently described as a two-stage cumulative contrast. It is not used
to infer an independently significant B2 or B3 increment.

## Holm audit

The standard five-test Holm step-down calculation was independently repeated
from the stored raw Wilcoxon P values. The scaled values were already monotone.

| Contrast | Raw P | Holm P | Significant at 0.05 | Interpretation |
|---|---:|---:|---|---|
| B1-B0 | `0.00016230344772338867` | `0.0006492137908935547` | true | Supported adjacent duration gain |
| B2-B1 | `0.17024139590961385` | `0.17024139590961385` | false | No independent hierarchy-performance claim |
| B3-B2 | `0.05825295212511276` | `0.11650590425022552` | false | No independent prototype-performance claim |
| B3-B1 | `0.013808561544817114` | `0.04142568463445134` | true | Cumulative two-stage contrast only |
| B3-B0 | `2.980232238769531e-07` | `1.4901161193847656e-06` | true | Complete-framework contrast |

The adjusted table was created from existing P values only. Models and
predictions were not rerun.

## Prior-art audit

The Related Work sections now state that:

- fixed 2-s pig audio appears in the PLOS pig-speech study and TransformerCNN;
- PVMC previously used a pig-specific coarse-to-fine pipeline;
- HiSSNet previously used hierarchical prototypical inference;
- AudioProtoPNet previously used interpretable acoustic prototypes;
- Briefer et al. provide an embedding/vocabulary precedent relevant to finer
  pig-call structure;
- the targeted audit found no exact pig-specific route match, but this is not
  an absolute first claim; and
- published scores are not directly ranked.

`PRIOR_ART_OVERLAP_MATRIX_v2.csv` corrects the AudioProtoPNet row to
`prototype_inference=true`, `post_hoc_prototype=false`, with explanation
“trainable prototype classifier replacing the ordinary classification layer.”
The comparison row for this study remains `prototype_inference=true` and
`post_hoc_prototype=true`, with a frozen trained backbone, train-only prototype
construction, and validation-only calibration. The imported novelty guard now
uses the same AudioProtoPNet classification. Its validation-only statement is
explicitly limited to within-run prototype calibration and does not cover the
test-informed `lambda=0.5` cohort choice.

Three missing bibliography entries were added after metadata-only checks
against the PLOS publisher page, the Springer article page, and the official
HiSSNet arXiv record: `wu2022pigspeech`, `liao2023transformercnn`, and
`shashaank2023hissnet`.

## Prototype-case audit

The two manuscripts report all seven deterministic fold-0/seed-42 cases: one
cough, one calm grunt, one feeding, one stress vocalisation, two feeding/stress
boundary errors, and one low-margin case. Each table includes the true class,
four Top-2 routes, main/subtype distances, final margin, hierarchy consistency,
and prototype representative.

The figure legends use the required definition: **“a training sample closest
to the predicted class prototype.”** C5 and C6 are explicitly described as
hierarchy-consistent but incorrect, so consistency is not conflated with
correctness. The selected cases are described as illustrations of output
semantics, not aggregate evidence.

The case source is now stated exactly as the pre-specified first numeric
expected key (`fold=0`, `seed=42`) after artifact validation, not lexical
filename order. Correct cases use lower-median Raw confidence; boundary cases
use the lower-median Raw Top-2 margin within eligible feeding/stress error
pools; the last case uses the minimum remaining hierarchical Top-2 margin.

## Noise-placement audit

Detailed DEMAND findings, former main noise figures, per-class failure tables,
and selective-prediction results are assigned to Supplementary Materials. The
main abstracts do not contain the `ALL_NOISY` Macro-F1 value. Main-text noise
wording is restricted to a simulated deployment boundary and explicitly avoids
real-farm validation claims.

The framework-v2 supplement map is explicit: existing Supplementary Figures
S1-S3 retain clean-ablation, environment-by-SNR, and
confusion/calibration/fold-sensitivity roles; former main noise Figures 4/5
become Supplementary Figures S4/S5. Supplementary Tables S1-S5 are mapped to
named clean-ablation, noise-summary, paired-noise, per-class, and
selective-prediction source files. This v2 map supersedes the legacy Figure 4/5
identifiers for this manuscript.

## Automated checks

| Check | Result |
|---|---|
| Citation-key existence | English: 18 distinct keys, 0 missing; Chinese: 18 distinct keys, 0 missing |
| English/Chinese citation parity | pass; key sets are identical |
| Cumulative numerical comparison | English and Chinese each contain the six expected table rows; all stage means/SDs, deltas, CIs, raw/Holm P values, W/T/L, fold-cluster CIs, B2-B0 descriptive delta, and five B3-B0 fold means match the CSVs; 0 failures |
| Prototype-case numerical comparison | C1-C7 true classes, four Top-2 fields, distances, margins, hierarchy consistency, representative paths/classes, and representative distances match `prototype_prediction_case_studies.csv`; 0 failures |
| Prior-art schema check | 13 rows load successfully; AudioProtoPNet and current-study boolean fields match the approved correction |
| Abstract length | English 220 whitespace-delimited tokens; Chinese 552 non-whitespace characters |
| Imported-figure visual check | pass; cumulative and case PNGs render completely and legibly |
| Overclaim grep | all matches are explicit negations, limitations, prior-art descriptions, or the permitted within-cohort B3-B0 comparison; no positive prohibited claim found |
| Protected result-file SHA check | 1,910 tracked `reports/` and `paper_results/` CSV/JSON files; ordered `<file SHA-256><two spaces><path>` aggregate SHA-256 `a27801827f770bc9604f6c434aba6b1f2e419a8110d504450a4ec941029ce5d6`; protected Git diff against `paper/sci-draft-v1` is empty |
| Protected result Git diff | 0 changed files under `reports/` and `paper_results/` |
| `git diff --check` | pass before handoff update |

## Leakage and test-use audit

This revision did not train, recalibrate, regenerate predictions, or select a
new configuration. It copied paper-facing aggregates that already bind every
B3 artifact to one fold and one seed. Within each frozen run, train builds model
parameters and prototypes; validation calibrates temperatures, weights,
thresholds, and uncertainty criteria; test supplies the stored evaluation.
Existing leakage audits report zero within-fold exact-path, source-ID, and MD5
overlap.

The configuration-level provenance is weaker. The completed comparison reports
the highest test mean at `lambda=1.0` and the lowest cross-run test standard
deviation at `lambda=0.5`, while the archive does not document a pre-test or
validation-only rule that selected `lambda=0.5` for the prototype cohort. It
would therefore be unsupported to describe that choice as free of test-set
selection. The manuscripts explicitly call it retrospective/test-informed,
treat B2/B3 comparisons as exploratory, and do not claim an independently
selected optimum. Holm multiplicity adjustment does not cure this selection
limitation.

The qualification remains important: Sow Call `source_id` values are
filename-derived singleton identities rather than animal/session/parent-recording
lineage. Exact-identity disjointness is supported; source-independent validation
is not.

## Remaining limitations and submission items

1. The cumulative and case figures are used with inline bilingual legends, but
   have not yet been added to a framework-v2-specific figure-source map,
   figure-to-claim matrix, TIFF export set, or Word-friendly table package. The
   older map remains a legacy-v1 artifact rather than the v2 numbering authority.
2. The Markdown manuscripts retain citekeys and a bibliography pointer; a
   journal-specific citeproc render remains a submission-stage task.
3. Author Contributions, Funding, Competing Interests, acknowledgements, and
   venue-specific AI-use disclosure require author-supplied information and
   were not fabricated.
4. The Sow Call citation/version and local download provenance should be
   reconciled in the final data statement. DEMAND's conflicting upstream
   licence text (CC BY-SA 3.0 versus CC BY 4.0 rights metadata) is retained in
   the bibliography and still requires provider/legal clarification.
5. External source-grouped real-farm validation, prospective explanation-utility
   evaluation, and independent subtype validation remain scientific gaps.
6. The `lambda=0.5` B2/B3 cohort lacks a documented pre-test or validation-only
   selection rule. Its comparisons are retrospective/test-informed and should
   remain explicitly exploratory in any submission.
7. The cited historical `v0.1-paper-draft-r1` release predates the new
   cumulative and case-study artifacts; a versioned framework-v2 package must
   be released before submission.

These items do not change the locked numerical conclusions, but they prevent
the current Markdown package from being treated as a final submission bundle.
