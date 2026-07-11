# Codex to GPT Handoff - Hierarchical Acoustic Prototype Atlas Audit and Debug Plan

## Status

Completed as a planning-only research audit. The user supplied `APPROVED: true`
and restricted this round to the current prototype-pipeline audit and a
fold0/seed3407 Atlas debug plan. No Atlas implementation, model training,
inference, feature extraction, calibration, clean/noise evaluation, DEMAND run,
or full CV was started.

## Branch and Commits

- Current branch: `paper/sci-draft-v1`
- Base commit: `a1a7a552e225522e021c9d14ef7a5136a6505e4c`
- Planning commit: `83050e8d3e1bebdc7e67bd4ced3df779abb4d5d0`
- Handoff commit: the delivery HEAD reported in the final Codex response
- Repository guidance names `current-mctafd-ablation` as the primary branch;
  implementation should not begin until GPT Pro/user confirms whether to remain
  on the current paper branch or move the approved implementation to that branch.

## Changed Files

- `docs/superpowers/plans/2026-07-11-hierarchical-acoustic-prototype-atlas.md`
- `handoff/CODEX_TO_GPT.md`
- `handoff/CODEX_TO_GPT.json`
- `handoff/HISTORY.md`

No training script, predictor/evaluator implementation, manuscript, checkpoint,
audio, established prototype result, clean result, or noise result was changed.

## Current Pipeline Audit

The existing pipeline is a partial or "semi-Atlas" implementation:

- `build_hier_acoustic_prototypes.py` constructs train-only main-class and
  subtype centroids, q95/mean/std cosine dispersion, normalized Log-Mel
  mean/std, representative samples, provenance, and leakage reports.
- `calibrate_prototype_predictor.py` performs validation-only temperature,
  fusion, and rejection calibration.
- `predict_hier_acoustic_prototype.py` already emits main/subtype top-2 labels,
  distances, margins, hierarchy consistency, uncertainty, and representative
  links.
- `eval_hier_acoustic_prototype.py` and
  `plot_acoustic_prototype_atlas.py` provide clean evaluation and geometry/
  acoustic-map views.

Missing for the requested Atlas are a unified schema-versioned ten-node graph,
explicit stable node IDs/parents, explicit medoids, unified cross-level nearest
nodes, SHA-bound per-exemplar embeddings, a single exact query schema,
`unknown_candidate`/ordered `rejection_reason`, a five-baseline comparison,
few-shot node insertion, and a rights-gated open-set workflow.

Risk found in the existing evaluator: it writes multiple test metrics and a
test threshold sweep. Those artifacts are useful diagnostics but must never be
used to select thresholds or claims. The planned Atlas evaluator has no test
grid or parameter-selection path.

## Exact Debug Cohort Locked by the Plan

- Setting: fold0, seed3407, lambda=0.5, `training_exact`, clean test only.
- Checkpoint:
  `checkpoints/cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407.pt`
  - SHA-256: `d6ec8dcdd6b16a6b3212a732e1d235428ba4c7c1f5702b5b1330ca9a3c449347`
- Training summary:
  `reports/cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407/summary.json`
  - SHA-256: `b0e92e1a86081559e46eabea4719322f259b83ca58834106256f247fcacde4ec`
- Canonical prior prototype reference:
  `reports/prototype_cv5_exact_w05_fold0_seed3407/`
  - Bundle SHA-256: `e2ad7e29735dcd88ca1e197c6a8ebdb0b4b75106594feeea96ee36617b2d18ee`
  - Calibration SHA-256: `d27e0ea53487e8162cb6e7f7c1e5e8489ca13f53dab79d150cdaa54e0ccd279e`
  - Test prediction SHA-256: `e672c334dec2c187f318ab357d7c948e0d8a2df06550a207d61127ee05a0508c`
- Locked fold0 manifest counts and SHA-256:
  - train: 1174,
    `33337bf683c9a07b8bdf9dc5a0020e53e6883716924c6deafc06af85720b005b`
  - validation: 120,
    `ed86bb19512f4b36d3ed4630251e5e15b9fe38d8d2d42119eb90058e6689dd79`
  - test: 168,
    `4c2eedb0766725c84da2a41b9ec0793b874f9a3d10a80a0a79f4ada7342c5e68`

Existing leakage reports for this cohort have `ok=true` and zero cross-split
exact-path, source-ID, and MD5 overlap.

## Planned Files and Architecture

The implementation plan adds, without altering the nine audited legacy files:

- `tools/hier_acoustic_atlas_core.py`
- `tools/build_hier_acoustic_atlas.py`
- `tools/calibrate_hier_acoustic_atlas.py`
- `tools/predict_hier_acoustic_atlas.py`
- `tools/eval_hier_acoustic_atlas.py`
- `tools/eval_atlas_fewshot_extension.py`
- `tools/eval_atlas_open_set.py`
- `tools/plot_hier_acoustic_atlas.py`
- `tests/test_acoustic_atlas_pipeline.py`
- `docs/HIERARCHICAL_ACOUSTIC_ATLAS.md`

The added calibrator is necessary to enforce validation-only Atlas path-weight,
kNN-k, known-rejection, and future unknown-rejection selection. The exact
implementation tasks, interfaces, commands, outputs, and acceptance gates are
in the planning document committed above.

## Few-Shot Protocol

- Eligible paired-parent subtypes: `dry_cough`, `abdominal_cough`,
  `frightened_stress`, and `anxious_stress`.
- `calm_grunt` and `feeding` are not eligible for leave-one-subtype-node-out
  insertion because each is its parent's sole subtype.
- Remove all target-subtype train exemplars from the base Atlas, select nested
  support sets from train only for k=1,3,5,10, insert the node, and evaluate the
  frozen full clean-test query set. Test data never selects k, thresholds, or
  support samples.
- Future complete protocol: 20 deterministic support-selection repeats per
  eligible subtype. This debug round is limited to `dry_cough`, repeat 0, with
  the four nested k values.
- Report mean/std Top-1, Top-2, and Macro-F1 and compare Raw Softmax, kNN, and
  Atlas insertion. Raw auxiliary Softmax is a seen-head upper bound, not a true
  unseen-subtype baseline, because the frozen backbone/head has already seen all
  six subtype labels.

## Open-Set Rights Gate

No open-set evaluation may start before source-level rights approval. Candidate
sources audited are:

- SoundWel (Zenodo 8252482): 6888 calls reported, CC BY 4.0. Pig calls are
  external-known/domain-shift data, not automatically unknown.
- aSwine: about 15 h, CC BY-NC 4.0. Exclusive non-target farm-background labels
  may be candidates, but paper/commercial reuse needs review.
- SmartEars poultry farm: 6000 clips including 2000 `None`, CC BY 4.0; strong
  far-OOD farm-background candidate.
- Poultry Vocalization Dataset: 346 WAV including 86 noise, CC BY 4.0; smaller
  far-OOD candidate.
- FSD50K/Freesound: only per-file CC0/CC BY clips may be used; preserve source
  IDs and attribution and de-duplicate by Freesound ID.
- ESC-50: 40 pig clips, CC BY-NC; pig is external-known, not unknown.
- Local Kaggle-like scream/cough archive: licence unresolved; blocked from
  evaluation and paper use.

Local exact-file MD5 checks found zero fold0 overlap for SoundWel raw (6887
local WAV), aSwine raw (154), the unresolved Kaggle archive (400), and the
reviewed SoundWel subset (300). This proves only zero exact-file overlap; it does
not exclude shared long-recording segments, source-study overlap, or acoustic
duplicates. Non-local candidates have not yet been overlap-audited.

## Commands Actually Executed

All commands were read-only except the plan/handoff documentation commits:

```powershell
git branch --show-current
git rev-parse HEAD
git status --short
Get-Content -Raw tools/prototype_model_adapter.py
Get-Content -Raw tools/build_hier_acoustic_prototypes.py
Get-Content -Raw tools/calibrate_prototype_predictor.py
Get-Content -Raw tools/predict_hier_acoustic_prototype.py
Get-Content -Raw tools/eval_hier_acoustic_prototype.py
Get-Content -Raw tools/plot_acoustic_prototype_atlas.py
Get-Content -Raw docs/PROTOTYPE_PREDICTOR.md
Get-Content -Raw tests/test_prototype_pipeline_core.py
Get-Content -Raw paper/manuscript/paper_en_full_story_polished.md
Get-FileHash -Algorithm SHA256 <locked checkpoint/summary/manifests/prototype artifacts>
firecrawl --status
rg -n "TBD|TODO|implement later|fill in details" docs/superpowers/plans/2026-07-11-hierarchical-acoustic-prototype-atlas.md
git diff --check -- docs/superpowers/plans/2026-07-11-hierarchical-acoustic-prototype-atlas.md
git add -- docs/superpowers/plans/2026-07-11-hierarchical-acoustic-prototype-atlas.md
git commit -m "docs: plan hierarchical acoustic prototype atlas"
```

`firecrawl --status` failed because the CLI is not installed; primary official
web pages were checked through the available web browser instead. No Python,
Conda, training, prediction, evaluator, unit-test, DEMAND, or CV command ran.

## Metrics

No new metrics were produced. For cohort identification only, the existing
canonical fold0/seed3407 clean file reports:

- Raw Softmax Macro-F1: `0.9463601533`
- Hierarchical prototype main Macro-F1: `0.9522727273`
- Auxiliary subtype prototype Macro-F1: `0.8355042017`

These are pre-existing read-only metrics, not results of this Atlas round.

## Paper Usability and Blockers

- `planning_artifact_usable=true`
- `new_atlas_evidence_created=false`
- `paper_main_result_usable=false` until the approved implementation and debug
  gates pass, followed by a separately approved full matched evaluation.
- DEMAND remains available only for supplementary/deployment-boundary analysis;
  no noise result was regenerated or rewritten here.

Blockers requiring scientific or governance judgment:

1. Confirm the implementation branch.
2. Approve the additional pure core and validation-only calibrator files.
3. Approve the far-OOD versus external-known ontology and exact source list.
4. Decide whether the pseudo-new subtype experiment is acceptable given that
   the frozen representation has seen every subtype during backbone training.
5. Resolve local SoundWel 6887-versus-official-6888 count discrepancy.
6. Resolve rights for every per-file source; keep the Kaggle-like archive blocked.

## Questions for GPT Pro

1. Approve the ten-file implementation map, including
   `hier_acoustic_atlas_core.py` and `calibrate_hier_acoustic_atlas.py`?
2. Should implementation occur on `paper/sci-draft-v1` or be moved to the
   repository-primary `current-mctafd-ablation` branch?
3. Approve SmartEars/Poultry-noise plus CC0/CC-BY Freesound/FSD50K clips as the
   far-OOD source pool, with pig grunt/oink/scream kept out of true-unknown claims?
4. Is the few-shot result to be framed only as frozen-representation node
   insertion, not genuinely unseen-backbone class learning?
5. Should manuscript reframing wait until Atlas evidence exists? Codex
   recommends yes.

## Stop Point

Review the committed audit and implementation plan. Do not begin implementation,
the fold0 debug, open-set evaluation, full CV, manuscript changes, or any noise
experiment without a new explicit approved stage.
