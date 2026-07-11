# Codex to GPT Handoff - Framework Manuscript v2

## Status

Completed the approved manuscript-only revision. No model training, inference,
feature extraction, calibration, clean/noise evaluation, prediction
regeneration, checkpoint loading, Atlas work, few-shot work, or open-set work
was performed.

The package is scientifically ready as a transparent exploratory manuscript,
with one material limitation: the archive does not document a pre-test or
validation-only rule for selecting the `lambda=0.5` B2/B3 cohort. Both
manuscripts now label comparisons involving B2/B3 retrospective/test-informed
and exploratory. Holm adjustment does not correct that model-selection issue.

## Branch and Commits

- Branch: `paper/framework-manuscript-v2`
- Manuscript base: `paper/sci-draft-v1` at
  `b52a0f63539d544f27c57e29760384c0782e7f61`
- Artifact source: `paper/prior-art-fair-benchmark` at
  `ccf3171be6c53a75299427bc1cf726377a9c1c29`
- Paper-package commit:
  `afb4d8cb63a219e2b3cde850daa13f31e4bc4baf`
- Handoff commit: the delivery HEAD reported in the final Codex response

## Changed Files

New framework-v2 outputs:

- `paper/manuscript/paper_en_framework_v2.md`
- `paper/manuscript/paper_cn_framework_v2.md`
- `paper/CLAIMS_AND_EVIDENCE_v3.md`
- `paper/tables/cumulative_framework_paired_stats_holm.csv`
- `paper/references/PRIOR_ART_OVERLAP_MATRIX_v2.csv`
- `paper/review/framework_v2_claim_audit.md`

Approved imported evidence:

- four `paper/tables/cumulative_framework_*.csv` source tables;
- `paper/tables/prototype_prediction_case_studies.csv`;
- cumulative framework PDF/PNG/SVG;
- prototype prediction case PDF/PNG/SVG;
- `paper/references/PRIOR_ART_OVERLAP_MATRIX.csv`;
- `paper/references/PUBLISHED_SCORE_COMPARABILITY.csv`;
- `paper/references/NOVELTY_CLAIM_GUARD.md`.

Additional manuscript support:

- `paper/references/references.bib` adds verified Wu, TransformerCNN, and
  HiSSNet entries and preserves the DEMAND upstream licence-metadata conflict.
- `NOVELTY_CLAIM_GUARD.md` corrects the stale AudioProtoPNet description.
- The two imported SVGs received trailing-whitespace-only normalization; their
  PDF/PNG companions and plotted content are unchanged.
- `handoff/CODEX_TO_GPT.md`, `handoff/CODEX_TO_GPT.json`, and
  `handoff/HISTORY.md` are updated by the delivery commit.

## Manuscript Framing

Working title:

> A Duration-Aware Hierarchical Prototype Inference Framework for
> Interpretable Pig Vocalisation Recognition

Chinese title:

> 面向可解释猪声识别的时长感知层级原型推理框架

Organising statement:

> See the complete event, learn the fine-grained structure, and predict with
> interpretable acoustic candidates.

The formal stages are B0 1-s Log-Mel CRNN, B1 2-s duration-aware CRNN, B2
shared four-main/six-subtype auxiliary supervision at the archived
`lambda=0.5` cohort evaluated by Raw Softmax, and B3 fold-seed-specific post-hoc
hierarchical prototype candidate inference.

## Locked Metrics and Holm Conclusions

All values are pre-existing read-only Macro-F1 results on 25 matched fold-seed
rows:

- B0: `0.9226064673 +/- 0.0145406384`
- B1: `0.9472373583 +/- 0.0167635408`
- B2: `0.9510496732 +/- 0.0123933139`
- B3: `0.9539862142 +/- 0.0120037735`

Five-test Holm family:

- B1-B0: raw `0.00016230344772338867`, Holm
  `0.0006492137908935547`, significant.
- B2-B1: raw/Holm `0.17024139590961385`, nonsignificant.
- B3-B2: raw `0.05825295212511276`, Holm
  `0.11650590425022552`, nonsignificant.
- B3-B1: raw `0.013808561544817114`, Holm
  `0.04142568463445134`, significant only as a cumulative two-stage contrast.
- B3-B0: raw `2.980232238769531e-07`, Holm
  `1.4901161193847656e-06`, significant as a complete-framework contrast.

Primary B3-B0 result: mean delta `0.0313797469`, bootstrap 95% CI
`[0.0228211123, 0.0403014884]`, W/T/L `24/0/1`, five positive fold means, and
fold-cluster CI `[0.0171744691, 0.0495755474]`.

B1-B0 is the strongest independently supported increment. B2 adds semantic
structure and B3 adds candidate/interpretability fields, but neither adjacent
increment is independently significant. The sequential design does not test
interaction or synergy. Because `lambda=0.5` was not selected by a documented
pre-test or validation-only rule, B2/B3 and B3-B0 inference remains exploratory
despite the reported P values.

## Prototype Case Contract

Seven deterministic fold0/seed42 cases are reported: one per main class, two
feeding/stress boundary errors, and one minimum-remaining-margin case. Correct
cases use lower-median Raw confidence; boundary cases use lower-median Raw
Top-2 margin in eligible error pools. The source run is the first numeric key
in the canonical expected-key list, not a filename-lexical choice.

The representative is defined exactly as:

> a training sample closest to the predicted class prototype

It comes only from the matching run's training split. C5/C6 are
hierarchy-consistent but wrong; consistency is not correctness. The selected
cases illustrate output semantics and are not aggregate performance evidence.

## Prior-Art Correction

AudioProtoPNet is now consistently classified as:

- `prototype_inference=true`
- `post_hoc_prototype=false`
- `trainable prototype classifier replacing the ordinary classification layer`

This study remains post-hoc, with a frozen trained backbone, train-only
fold-seed prototypes, and validation-only within-run prototype calibration.
The latter does not imply validation-only selection of `lambda=0.5`.

## Leakage and Integrity Audit

- No model, prediction, checkpoint, audio, `reports/`, or `paper_results/`
  result was changed.
- Git diff against `paper/sci-draft-v1` under `reports/` and `paper_results/`
  is empty.
- Protected set: 1,910 tracked CSV/JSON files.
- Ordered `<file SHA-256><two spaces><path>` aggregate SHA-256:
  `a27801827f770bc9604f6c434aba6b1f2e419a8110d504450a4ec941029ce5d6`.
- Existing within-fold leakage audits retain zero normalized exact-path,
  source-ID, and MD5 overlap.
- Prototypes and calibration artifacts remain fold-seed-specific; train builds
  prototypes and validation calibrates prototype parameters within each fixed
  run.
- Limitation: Sow Call animal/session/farm/parent-recording lineage is
  incomplete, so latent correlated-source leakage cannot be excluded.
- Limitation: the archived `lambda=0.5` cohort choice is test-informed; no
  confirmatory no-test-tuning claim is made for B2/B3.

## Verification

- Independent scientific review: ready to commit; no scientific blocker.
- Citation check: 18 unique keys in each language, zero missing, exact parity.
- Abstract check: English 220 whitespace tokens/244 regex words; Chinese 552
  non-whitespace code points/281 Han characters; no `0.623` in either abstract.
- Holm recomputation: exact match for all five rows and significance flags.
- Cumulative numerical check: all means/SDs, deltas, CIs, raw/Holm P values,
  W/T/L, fold-cluster CIs, and B3-B0 fold means match; zero failures.
- Case check: C1-C7 rankings, distances, margins, consistency, representatives,
  and selection rules match; zero failures.
- Prior-art CSV: 13 rows parse; AudioProtoPNet/current-study fields pass.
- Main figure links resolve; both SVGs parse as XML; PNGs were visually checked.
- Overclaim grep: matches only explicit prohibitions, negations, limitations, or
  permitted within-cohort cumulative wording.
- Full staged `git diff --check`: pass after whitespace-only SVG normalization.

No one-fold debug run was performed because this round explicitly prohibited
model evaluation and changed no executable model code.

## Commands Actually Executed

Key commands were:

```powershell
git worktree add -b paper/framework-manuscript-v2 C:\py\pigsound\pig-sound-classification-framework-v2 paper/sci-draft-v1
git restore --source paper/prior-art-fair-benchmark -- <the 14 approved artifact paths>
Import-Csv paper/tables/cumulative_framework_paired_stats.csv
Import-Csv paper/tables/cumulative_framework_paired_stats_holm.csv
Import-Csv paper/tables/prototype_prediction_case_studies.csv
Import-Csv paper/references/PRIOR_ART_OVERLAP_MATRIX_v2.csv
rg -n <citation, overclaim, selection, and evidence patterns> paper
git diff --name-only paper/sci-draft-v1 -- reports paper_results
git diff --cached --check
git commit -m "docs: revise cumulative framework manuscript"
```

Read-only inline PowerShell/Python checks recomputed Holm values, citation
parity, abstracts, cumulative tables, case rows, hashes, and protected scope.
No training or evaluation command ran.

## Paper Usability, Blockers, and Questions

- `paper_usable=true_with_material_limitation`
- `submission_bundle_complete=false`
- Detailed DEMAND results remain supplementary and do not establish real-farm
  robustness; the abstract contains no ALL_NOISY Macro-F1.

Scientific judgment required:

1. For confirmatory B2/B3 claims, should a future separately approved study
   pre-specify `lambda` from validation-only evidence or evaluate an untouched
   external cohort? The present paper must retain exploratory wording unless
   this is resolved.
2. Is the six-subtype hierarchy acceptable as an engineering supervision
   taxonomy pending independent biological annotation?
3. Before submission, provide author/funding/conflict/AI-disclosure metadata,
   reconcile source terms, publish a framework-v2 code/data package, and add
   v2-specific TIFF/source-map/Word packaging.

## Stop Point

Stop after this handoff. Do not start Atlas, few-shot, open-set, real-farm,
noise, retraining, or a confirmatory lambda stage. Before any next stage, read
`handoff/GPT_TO_CODEX.md` and proceed only if it explicitly says
`APPROVED: true`.
