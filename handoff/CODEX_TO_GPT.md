# Codex to GPT Handoff — Prior-Art Overlap Audit and Fair-Comparison Design

## Status

Completed the approved research-audit stage. No model training, model inference, feature extraction, calibration, test-set tuning, manuscript rewrite, audio edit, checkpoint edit, or established result-file edit was performed.

The audit found no exact published pig-specific route match for fixed 2 s input + shared main/subtype supervision + post-hoc train-only hierarchical prototypes + ranked candidates + representative-example retrieval. This is a targeted-search result, not an absolute first claim.

## Branch and commits

- Base branch: `paper/sci-draft-v1`
- Base commit: `b52a0f63539d544f27c57e29760384c0782e7f61`
- Current branch: `paper/prior-art-fair-benchmark`
- Audit/design artifact commit: `36f950f020ed769a4fe297d48e460a10829e08fa`
- Handoff commit: delivery HEAD reported in the final Codex response

## Changed files

Audit/design commit:

- `paper/references/PRIOR_ART_OVERLAP_MATRIX.csv`
- `paper/references/PUBLISHED_SCORE_COMPARABILITY.csv`
- `paper/references/NOVELTY_CLAIM_GUARD.md`
- `paper/benchmark/SOW_CALL_BENCHMARK_FEASIBILITY.md`
- `paper/benchmark/SOW_CALL_SOURCE_AUDIT.csv`
- `paper/benchmark/SOW_CALL_FAIR_BENCHMARK_PLAN.md`
- `paper/benchmark/PROTOTYPE_PREDICTION_VALUE_PLAN.md`

Handoff-only files:

- `handoff/CODEX_TO_GPT.md`
- `handoff/CODEX_TO_GPT.json`
- `handoff/HISTORY.md`

Concurrent staged/untracked user artifacts were left intact and excluded by explicit commit pathspec.

## Prior-art result

All eight required works were verified from publisher pages, official proceedings, or official preprints. Briefer et al. is verified at DOI `10.1038/s41598-022-07174-8`.

Closest precedents split across separate routes:

- Wu and Liao: pig-specific fixed 2 s context and calm/feeding/frightened/anxious labels.
- PVMC: pig-specific modular coarse-to-fine classification.
- HiSSNet: closest methodological overlap for hierarchy-level supervision plus prototype inference.
- AudioProtoPNet: closest overlap for frozen-representation prototype fitting and training-example retrieval.
- ProtoNet and DCASE: few-shot prototype/candidate precedent.

No verified work combines all requested elements. `NOVELTY_CLAIM_GUARD.md` prohibits absolute first claims and unmatched state-of-the-art claims.

No external score is directly comparable after checking dataset, labels, split, metric, sample unit, clean/noise condition, and class balance. Numbers above the clean B3 Macro-F1 `0.953986` exist, including TransformerCNN accuracy `0.9605`, PVMC segmentation accuracy `0.9580` and coarse accuracy `0.9888`, HiSSNet SID-L2 accuracy, AudioProtoPNet regional AUROC, and several recent pig-study accuracies. They remain numerical context only.

## Sow Call audit

Canonical local source:

| Class | Raw WAV | Unique MD5 | Labeled-tree WAV |
|---|---:|---:|---:|
| calm | 280 | 280 | 560 |
| feeding | 210 | 210 | 420 |
| frightened | 750 | 749 | 1500 |
| anxious | 506 | 504 | 1012 |
| total | 1746 | 1743 | 3492 |

All filenames follow the suffix map `0=calm`, `1=feeding`, `2=frightened`, `3=anxious`; the index has 1,746 complete rows and zero unknown labels. `data/raw/wu_pig_speech` is an exact duplicate root and is not independent.

Manifest audit:

- all 15 cap3x manifests: 7,310 rows, 3,812 unique path/source-ID/MD5 identities;
- Sow subset: 4,880 rows, 1,637 unique identities;
- missing paths, blank IDs/hashes, invalid hashes, and rehash mismatches: all zero;
- within-fold train/validation/test path, source-ID, and MD5 overlaps: all zero;
- pairwise test-fold overlaps: all zero;
- test union: 840 clips, including 630 Sow clips (`210/210/105/105`).

The current `source_id` is `parent_folder::filename_stem`; it is a singleton file identity, not animal/session/parent-recording lineage. A source-ID-grouped splitter would therefore be equivalent to a clip split. Exact duplicate leakage is preventable, but latent recording/session leakage is untestable.

Exact published 3,200/800 recovery is blocked: local source has 1,746 rather than 4,000 files, and no official item assignment, augmentation map, or split seed is available. The current main task is also not directly comparable because it adds Korean cough and merges frightened/anxious into `stress_vocal`.

The six-subtype hard predictions can be audited on the four Sow labels, but existing files do not store auxiliary probabilities/logits; current mixed-task models and 630-Sow test coverage do not constitute a fair Sow-only benchmark.

## Fair benchmark decision

Protocol A requires the complete 4,000-item corpus and exact official 3,200/800 membership. It is currently blocked.

Protocol B requires meaningful animal/session/parent-recording lineage for grouped 5-fold CV. It is currently blocked. An MD5-disjoint clip-CV fallback can be run only under a downgraded label and cannot support source-generalization claims.

TransformerCNN is marked `not_reproducible_without_assumptions` because no official code, exact split membership/seed, complete optimizer/checkpoint contract, or fully versioned preprocessing contract was located.

Recommendation: `manuscript_only_revision` unless full data plus authoritative split/lineage metadata are recovered.

## Existing cumulative 25-pair analysis

All B0/B1/B2/B3 stages contain exact folds `0-4` × seeds `42,123,777,2024,3407`; no missing/extra/duplicate keys and no imputation. B2 values equal the prototype cohort's Raw Softmax values exactly. B3 is specifically `hierarchical_macro_f1`.

| Stage | Mean Macro-F1 | SD |
|---|---:|---:|
| B0 1 s Log-Mel CRNN | 0.922606467 | 0.014540638 |
| B1 2 s Log-Mel CRNN | 0.947237358 | 0.016763541 |
| B2 B1 + hierarchy lambda=0.5, Raw Softmax | 0.951049673 | 0.012393314 |
| B3 B2 + hierarchical prototype inference | 0.953986214 | 0.012003773 |

| Contrast | Mean delta | 95% paired bootstrap CI | Wilcoxon p | W/T/L | Fold-cluster CI |
|---|---:|---|---:|---:|---|
| B1−B0 | 0.024630891 | [0.014112838, 0.035642072] | 0.000162303 | 21/0/4 | [0.006577865, 0.048280218] |
| B2−B1 | 0.003812315 | [−0.001320423, 0.008726684] | 0.170241396 | 16/1/8 | [−0.003406537, 0.010560902] |
| B3−B2 | 0.002936541 | [−0.000477565, 0.007294065] | 0.058252952 | 13/8/4 | [−0.000219515, 0.008263370] |
| B3−B1 | 0.006748856 | [0.002388569, 0.011270120] | 0.013808562 | 18/0/7 | [0.001295329, 0.010596604] |
| B3−B0 | 0.031379747 | [0.022821112, 0.040301488] | 0.000000298 | 24/0/1 | [0.017174469, 0.049575547] |

B2−B1 and B3−B2 remain nonsignificant. B3−B1 is an unadjusted cumulative contrast and does not pass Bonferroni correction over five planned contrasts at `alpha=0.01`.

## Verification and leakage audit

- Seven requested files exist and parse: 12 prior-art rows, 40 external-score rows, 12 Sow audit rows.
- Required prior-art matrix columns match the user contract exactly.
- Comparability values use only `direct`, `partially_comparable`, `contextual_only`, and `not_comparable`; direct rows: zero.
- Fresh cumulative validator passed, preserved 200 frozen source files, and reproduced all stage and paired values above.
- Independent pre/post SHA-256 comparison checked 50 required manuscript/result/script/manifest inputs: missing `0`, mismatches `0`.
- Scoped `git diff --cached --check` passed for the seven audit/design files.
- No training, inference, audio modification, or test-set parameter selection occurred.

Conda activation failed because the shell PATH contains an invalid replacement character that triggers a GBK `UnicodeEncodeError`. The frozen validator was therefore run with the exact `pigsound-gpu` environment Python executable after the failed activation attempt.

## Commands actually executed

Representative exact local commands:

```powershell
git switch -c paper/prior-art-fair-benchmark
Get-Content -Raw -Encoding UTF8 paper/manuscript/paper_en_full_story_polished.md
Get-Content -Raw -Encoding UTF8 paper/CLAIMS_AND_EVIDENCE_v2.md
Get-Content -Raw -Encoding UTF8 paper/references/references.bib
Get-Content -Raw tools/train_hier_longcontext_crnn.py
Get-Content -Raw tools/prototype_model_adapter.py
Import-Csv reports/cv5_expanded_variant_runs.csv
Import-Csv reports/hier_longcontext_runs.csv
Get-ChildItem data/manifests_pigvocal_4class_expanded_train_cv5_cap3x -Recurse -Filter *.csv
Get-FileHash -Algorithm MD5 <all unique referenced Sow manifest paths>
Get-FileHash -Algorithm SHA256 <50 frozen required source files>
Invoke-RestMethod https://api.figshare.com/v2/articles/16940389/versions/1
firecrawl --status
conda activate pigsound-gpu
$env:PYTHONNOUSERSITE='1'
& 'C:\py\anaconda3\envs\pigsound-gpu\python.exe' tools\generate_cumulative_framework_analysis.py --root . --bootstrap-resamples 10000 --bootstrap-seed 3407 --validate-only
git diff --cached --check -- <seven requested artifact paths>
git commit --only -m 'docs: audit prior art and fair benchmark' -- <seven requested artifact paths>
```

`firecrawl --status` failed because the CLI is unavailable; the literature audit used publisher/official-preprint web pages. No global CLI or dependency was installed.

## Paper usability and blockers

- `prior_art_guard_usable=true`
- `fair_comparison_design_usable=true`
- `new_benchmark_result_created=false`
- `manuscript_modified=false`
- `paper_claim_change_authorized=false`

Blockers requiring scientific or data-governance judgment:

1. Recover the full official 4,000-item Sow corpus and exact published split membership if Protocol A is desired.
2. Recover animal/session/parent-recording lineage if Protocol B source-grouped CV is desired.
3. Decide whether a clearly labeled MD5-disjoint clip-CV diagnostic is still worth running.
4. Decide whether TransformerCNN assumptions are acceptable for a description-based reimplementation.
5. Decide which targeted-search additions belong in the manuscript's Related Work; no manuscript edit was made here.

## Stop point

Review the committed audit and design only. Do not start the fair benchmark, TransformerCNN implementation, inference-only auxiliary export, manuscript revision, or any other experiment without a new explicit approved stage.
