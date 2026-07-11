# Codex -> GPT Pro Handoff

## Status

`COMPLETED_CANONICAL_NOMENCLATURE_V1`

The approved non-destructive nomenclature refactor is complete. Review this
handoff and stop; do not begin another stage without a new explicit
`APPROVED: true` instruction.

## Repository state

- Base commit: `3b336996dba15c6227d652b276c9882503714659`
- Branch: `refactor/canonical-nomenclature-v1`
- Implementation commit: `1cd4afe1a716a05481a9ee14094825ed7700d70a`
- Handoff commit: reported in the final Codex response
- Paper usable: `true` for nomenclature, metadata, and regenerated-output
  labeling only; this task adds no scientific evidence.

## Scope and authorization

- Model training: **not run**
- Checkpoint inference or feature extraction: **not run**
- Audio/manifests/checkpoints/historical results modified: **none**
- Test-set parameter selection or tuning: **none**
- Historical schemas, machine columns, checkpoint names, `state_dict` keys,
  `main_head`, and `aux_head`: **unchanged**
- Existing `reports/`, `paper/`, `paper_results/`, `checkpoints/`, and `data/`
  tracked artifacts: **unchanged**
- New final-validation, final-functional, and cumulative outputs are versioned
  and refuse overwrite; historical unversioned packages remain read-only.

## Canonical contract implemented

- Schema: `pig_sound_nomenclature.v1`
- Model families: `logmel_crnn`, `hierarchical_supervision_crnn`
- Training stages: B0/B1/B2/B3 canonical IDs from the approved task
- Inference routes: `primary_softmax`,
  `main_class_prototype_candidate`,
  `hierarchical_prototype_candidate`
- Selection protocols: `none`,
  `foldwise_validation_selected_lambda`,
  `fixed_lambda_0_5_retrospective`
- Required legacy aliases remain accepted and canonicalized without changing
  stored legacy machine keys.
- Fixed lambda=0.5 and fold-wise validation-selected provenance are never
  implicitly mapped to each other.
- Explicit schema-v1 blocks are strictly validated. Unversioned legacy records
  remain readable, but populated legacy provenance is preserved and conflicts
  fail rather than being overwritten.
- Cross-artifact model family, stage, context, route, and protocol are
  reconciled. The only legal B2-to-B3 transition is the hierarchical-prototype
  Top-1 decision route.

## Files changed

New:

- `tools/nomenclature.py`
- `docs/NOMENCLATURE.md`
- `tests/test_nomenclature.py`
- `docs/superpowers/plans/2026-07-12-canonical-nomenclature-v1.md`

Updated active code/tests/docs:

- `tools/build_hier_acoustic_prototypes.py`
- `tools/calibrate_prototype_predictor.py`
- `tools/predict_hier_acoustic_prototype.py`
- `tools/eval_hier_acoustic_prototype.py`
- `tools/summarize_cv5_exact_prototype.py`
- `tools/generate_final_validation_audit.py`
- `tools/generate_final_functional_figure_package.py`
- `tools/generate_cumulative_framework_analysis.py`
- `tests/test_generate_final_validation_audit.py`
- `tests/test_generate_final_functional_figure_package.py`
- `tests/test_generate_cumulative_framework_analysis.py`
- `docs/FINAL_FUNCTIONAL_FIGURE_PACKAGE.md`
- `docs/CUMULATIVE_FRAMEWORK_ANALYSIS.md`

## Commands actually executed

Environment:

```powershell
$env:PYTHONUTF8='1'
conda activate pigsound-gpu
cd C:\py\pigsound\pig-sound-classification
$env:PYTHONNOUSERSITE='1'
```

Compilation and full regression suite:

```powershell
python -m py_compile tools/nomenclature.py tools/build_hier_acoustic_prototypes.py tools/calibrate_prototype_predictor.py tools/eval_hier_acoustic_prototype.py tools/predict_hier_acoustic_prototype.py tools/summarize_cv5_exact_prototype.py tools/generate_final_validation_audit.py tools/generate_final_functional_figure_package.py tools/generate_cumulative_framework_analysis.py tests/test_nomenclature.py tests/test_generate_final_validation_audit.py tests/test_generate_final_functional_figure_package.py tests/test_generate_cumulative_framework_analysis.py
$env:PIGSOUND_NOMENCLATURE_HISTORICAL_SHA256_BASELINE=(Resolve-Path .git\codex_nomenclature_historical_sha256.tsv)
$env:PIGSOUND_NOMENCLATURE_CHECKPOINT_BASELINE=(Resolve-Path .git\codex_nomenclature_checkpoint_baseline.json)
python -m unittest -q tests.test_nomenclature tests.test_prototype_pipeline_core tests.test_generate_final_validation_audit tests.test_generate_final_functional_figure_package tests.test_generate_cumulative_framework_analysis tests.test_generate_journal_figure_package
```

Compatibility and static checks:

```powershell
python -m unittest -v tests.test_nomenclature.ImmutableArtifactCompatibilityTests
python tools/nomenclature.py --help
python tools/build_hier_acoustic_prototypes.py --help
python tools/calibrate_prototype_predictor.py --help
python tools/predict_hier_acoustic_prototype.py --help
python tools/eval_hier_acoustic_prototype.py --help
python tools/summarize_cv5_exact_prototype.py --help
python tools/generate_final_validation_audit.py --help
python tools/generate_final_functional_figure_package.py --help
python tools/generate_cumulative_framework_analysis.py --help
git diff --check
git diff --name-only -- reports checkpoints data paper paper_results
```

No generator, training command, or checkpoint forward pass was executed.

## Verification results

- Full suite: **184/184 passed** with both integrity tests enabled.
- `py_compile`: passed for all 13 changed/new Python files.
- CLI help: **9/9 passed**.
- Independent review: no Critical, Important, or Minor findings; ready to
  commit.
- Rendering regression: canonical figure text stays inside the 183-mm canvas;
  no constrained/tight-layout collapse warning.
- Historical tracked artifact SHA audit: **2,624/2,624 unchanged**.
- Checkpoint compatibility sample:
  `checkpoints/cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed42.pt`
  remains SHA-256
  `8728fb7afc98f7c4ea113ede3ebc2e83aa2a39a583f1928e5354b45f6f87da6c`;
  all 40 keys and tensor signatures match, and strict `HierCRNN.load_state_dict`
  passes without inference.
- `git diff --check`: passed.
- Protected tracked-path diff: empty.

## Metrics and paper interpretation

No metrics were recomputed or changed. Published/validated paper values remain
unchanged, including the approximately `0.947237` 2-s Log-Mel baseline and the
hierarchical means approximately `0.950506`, `0.951050`, and `0.951508` for
lambda 0.2/0.5/1.0. This refactor makes no new significance claim: the major
supported gain remains 2-s context, while the hierarchical increment over the
2-s baseline remains non-significant.

## Leakage audit

No train/validation/test membership or selection logic changed. Exact-path,
source-ID, and MD5 disjointness code remains intact; no test artifact was used
for parameter selection. New readers fail on conflicting model/protocol
provenance rather than silently relabeling it. The frozen 2,624-file SHA audit
found zero missing and zero changed files.

## Blockers and GPT Pro questions

- Blockers: none.
- Scientific-judgment questions: none required for this refactor.
- Suggested next step: review the canonical labels and metadata contract only.
  Do not regenerate results, revise the manuscript, retrain, or start another
  experiment without a new approved stage.
