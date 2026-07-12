# Canonical Nomenclature v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one versioned canonical nomenclature registry and use it at active read/write/display boundaries without changing historical artifacts, checkpoint structure, inference algorithms, or legacy schemas.

**Architecture:** `tools/nomenclature.py` is a dependency-light registry for canonical IDs, bilingual labels, legacy aliases, validation, and metadata construction. Active scripts retain legacy method IDs internally wherever existing probability columns and result schemas depend on them; they canonicalize at input boundaries, append canonical metadata to newly generated outputs, and obtain human-facing labels from the registry.

**Tech Stack:** Python 3.10, standard-library `argparse`/`json`/`warnings`, pandas at existing script boundaries, `unittest`, Git/SHA-256 audits.

## Global Constraints

- Do not train models or run checkpoint inference.
- Do not rename checkpoint files, checkpoint `state_dict` keys, `main_head`, or `aux_head`.
- Do not modify existing reports, paper result values, manifests, CSV/JSON artifacts, schemas, tags, or historical commits.
- Preserve legacy CLI values and persisted columns such as `raw_softmax`, `prototype`, and `hierarchical`.
- Never infer `foldwise_validation_selected_lambda` merely because lambda equals `0.5`.
- New outputs use unique paths and may append canonical metadata; existing column names remain unchanged.
- All unknown canonical IDs fail with a message naming the namespace, rejected value, and allowed IDs.

---

### Task 1: Registry contract and compatibility tests

**Files:**
- Create: `tests/test_nomenclature.py`
- Create: `tools/nomenclature.py`

**Interfaces:**
- Produces: canonical ID tuples/mappings, `NOMENCLATURE_SCHEMA_VERSION`, `validate_id`, namespace-specific canonicalizers, `display_label`, `preferred_legacy_method_id`, `build_model_metadata`, `build_method_metadata`, `metadata_from_legacy_record`, and `validate_method_metadata`.
- Consumes: no project model code and no new dependency.

- [ ] **Step 1: Write failing registry tests.** Cover all canonical values, every required legacy alias, unknown values, fixed-versus-selected separation, an existing legacy paper table row, JSON round-trip validation, an environment-gated local checkpoint load/key digest, and an environment-gated historical SHA manifest.
- [ ] **Step 2: Verify RED.** Run `python -m unittest tests.test_nomenclature -v`; expect import failure because `tools.nomenclature` does not exist.
- [ ] **Step 3: Implement the minimal registry.** Keep aliases namespace-specific and construct method IDs as `<training_stage>::<selection_protocol>::<inference_route>` so fixed and validation-selected evidence cannot collide.
- [ ] **Step 4: Verify GREEN.** Re-run `python -m unittest tests.test_nomenclature -v`; expect all tests to pass, with checkpoint/SHA tests executed when their baseline environment variables are supplied.

### Task 2: Prototype pipeline boundary integration

**Files:**
- Modify: `tools/build_hier_acoustic_prototypes.py`
- Modify: `tools/calibrate_prototype_predictor.py`
- Modify: `tools/predict_hier_acoustic_prototype.py`
- Modify: `tools/eval_hier_acoustic_prototype.py`
- Modify: `tools/summarize_cv5_exact_prototype.py`
- Test: `tests/test_nomenclature.py`
- Test: `tests/test_prototype_pipeline_core.py`

**Interfaces:**
- Consumes: registry functions from Task 1.
- Produces: legacy-compatible CLI/storage behavior plus canonical metadata in newly written JSON/CSV outputs.

- [ ] **Step 1: Add failing integration tests.** Assert canonical and legacy selection values resolve to the same legacy storage columns; old result rows remain readable; noncanonical legacy-only `calibrated_softmax` and `fused` remain accepted but are not mislabeled as canonical routes.
- [ ] **Step 2: Verify RED.** Run the new focused tests and confirm the expected missing integration behavior.
- [ ] **Step 3: Integrate at boundaries.** Preserve all probability/prediction column names, propagate an explicit selection protocol (default `none` when old artifacts do not record one), append canonical fields to new metadata, and log the canonical English display label.
- [ ] **Step 4: Verify GREEN.** Run `tests.test_nomenclature` and `tests.test_prototype_pipeline_core`.

### Task 3: Final-validation and figure/table integration

**Files:**
- Modify: `tools/generate_final_validation_audit.py`
- Modify: `tools/generate_final_functional_figure_package.py`
- Modify: `tools/generate_cumulative_framework_analysis.py`
- Modify: `docs/FINAL_FUNCTIONAL_FIGURE_PACKAGE.md`
- Modify: `docs/CUMULATIVE_FRAMEWORK_ANALYSIS.md`
- Test: `tests/test_nomenclature.py`
- Test: `tests/test_generate_final_validation_audit.py`
- Test: `tests/test_generate_final_functional_figure_package.py`
- Test: `tests/test_generate_cumulative_framework_analysis.py`

**Interfaces:**
- Consumes: registry display labels and metadata builders.
- Produces: in-memory canonical routes for legacy source rows, canonical labels in new plots/logs, foldwise-validation metadata for B2/B3 primary analyses, and fixed-0.5 retrospective metadata only for explicitly retrospective evidence.

- [ ] **Step 1: Add failing display/metadata tests.** Assert the three route labels come from the registry and emitted method rows contain all ten required metadata fields without replacing legacy `method` columns.
- [ ] **Step 2: Verify RED.** Run focused final-validation/figure tests and confirm missing canonical metadata or label failures.
- [ ] **Step 3: Add boundary adapters.** Canonicalize legacy source values after reading, retain legacy keys for joins and calculations, attach row-appropriate B0/B1/B2/B3 metadata to new tables, and add schema information to new provenance/QA payloads.
- [ ] **Step 4: Verify GREEN.** Run all focused final-validation and figure suites without generating paper outputs.

### Task 4: Documentation, integrity verification, and handoff

**Files:**
- Create: `docs/NOMENCLATURE.md`
- Modify: `handoff/CODEX_TO_GPT.md`
- Modify: `handoff/CODEX_TO_GPT.json`
- Modify: `handoff/HISTORY.md`

**Interfaces:**
- Consumes: verified registry and Git/SHA audit evidence.
- Produces: bilingual mapping documentation, exact commands/results, leakage statement, paper usability statement, and final commit/push record.

- [ ] **Step 1: Write nomenclature documentation.** Cover architecture versus route, B0-B3, canonical IDs, bilingual labels, aliases, fixed-versus-selected rules, historical artifact rules, and metadata examples.
- [ ] **Step 2: Compile and run tests.** Use the `pigsound-gpu` interpreter with `PYTHONNOUSERSITE=1` and `PYTHONUTF8=1`; run `py_compile`, nomenclature, prototype, final-validation, functional-figure, and cumulative-framework tests.
- [ ] **Step 3: Audit immutable state.** Compare all 2,624 pre-task tracked historical SHA-256 entries, reload the chosen local checkpoint, compare its file SHA and state-dict-key digest, run `git diff --check`, and inspect the diff for leakage/test-tuning changes.
- [ ] **Step 4: Update the required GPT handoff.** Record commands, branch/commit, files, metrics, path/source-ID/MD5 boundaries, paper usability, blockers, and scientific questions.
- [ ] **Step 5: Commit and push.** Commit only intended tracked source/docs/tests/handoff files on `refactor/canonical-nomenclature-v1`, push the branch, do not merge, and stop.

---

## PR #2 Review-Fix Addendum

### Task 5: Bind canonical metadata to artifact roles

**Files:**
- Modify: `tools/nomenclature.py`
- Modify: `tools/generate_cumulative_framework_analysis.py`
- Test: `tests/test_nomenclature.py`
- Test: `tests/test_generate_cumulative_framework_analysis.py`

**Interfaces:**
- Produces: `validate_expected_artifact_role(record, *, expected_model_family, expected_training_stage, expected_context_seconds, expected_selection_protocol=None, expected_inference_route=None, route_fields=("selection_method", "inference_route", "legacy_method_id"), context)`.
- Consumes: `validate_recorded_artifact_identity`, `canonicalize_selection_protocol`, and `resolve_inference_method` without changing legacy storage IDs.

- [ ] **Step 1: Add failing role tests.** Assert that B2 metadata on a B3 evaluation, `primary_softmax` versus legacy `hierarchical`, legacy `prototype` versus canonical `hierarchical_prototype_candidate`, and validation-selected metadata on a fixed-lambda artifact all raise `ValueError`; assert a consistent legacy-only artifact remains readable.
- [ ] **Step 2: Verify RED.** Run the new `CanonicalArtifactRoleTests`; expect import failure because `validate_expected_artifact_role` is absent.
- [ ] **Step 3: Implement the helper and wire readers.** Require exact family/stage/context/protocol/route matches for B0, B1, fixed-lambda B2, route-independent prototype bundles, and B3 hierarchical evaluation inputs; reject populated route metadata on the bundle.
- [ ] **Step 4: Verify GREEN.** Run the focused role tests and cumulative-framework tests; expect all to pass without generating paper outputs.

### Task 6: Emit complete aggregate schema-v1 metadata

**Files:**
- Modify: `tools/summarize_cv5_exact_prototype.py`
- Test: `tests/test_nomenclature.py`

**Interfaces:**
- Produces: complete top-level model metadata from `resolve_aggregate_nomenclature` and strictly validated run/provenance/method records.
- Consumes: `build_model_metadata`, `validate_model_metadata`, and `validate_method_metadata`.

- [ ] **Step 1: Add failing writer-contract tests.** Assert incomplete schema-v1 run and provenance blocks fail, generated run/provenance model blocks validate, every `canonical_methods` item validates, B2 routes retain B2 stage metadata, B3 hierarchical routes retain B3 stage metadata, and JSON/CSV round trips preserve fields.
- [ ] **Step 2: Verify RED.** Run the focused aggregate tests; expect missing `training_stage` or strict-validation failures.
- [ ] **Step 3: Build and validate complete metadata.** Return `build_model_metadata(...)` from `resolve_aggregate_nomenclature`, validate each run-row metadata block and provenance block before writing, and validate every canonical method item.
- [ ] **Step 4: Verify GREEN.** Re-run nomenclature and prototype aggregate tests; expect all to pass.

### Task 7: Version protected-artifact provenance and repair documentation

**Files:**
- Create: `tests/data/protected_artifacts_nomenclature_v1.sha256.json`
- Modify: `tests/test_nomenclature.py`
- Modify: `docs/CUMULATIVE_FRAMEWORK_ANALYSIS.md`
- Modify: `handoff/CODEX_TO_GPT.md`
- Modify: `handoff/CODEX_TO_GPT.json`
- Modify: `handoff/HISTORY.md`

**Interfaces:**
- Produces: a tracked manifest bound to base commit `3b336996dba15c6227d652b276c9882503714659`, source tree `417344e8a3c5fcdad715cfcd30e73525e340c311`, exactly 2,624 protected paths, and their SHA-256 values.
- Consumes: the previously captured 2,624-entry baseline; it reads but never rewrites historical artifacts.

- [ ] **Step 1: Add failing manifest assertions.** Require exact base commit, tree SHA, entry count, unique path set, and byte hashes; expect failure while the tracked manifest is absent.
- [ ] **Step 2: Create the manifest from the pre-existing baseline.** Convert the saved path/hash entries without reading or rewriting output content and verify the resulting set has exactly 2,624 unique paths.
- [ ] **Step 3: Correct documentation and handoff.** Replace only stale cumulative `_nomenclature_v1` output references; after staging every explicitly intended PR path, set `changed_files` from `git diff --cached --name-only 3b336996dba15c6227d652b276c9882503714659 --`, then recheck it against `git diff --name-only 3b336996dba15c6227d652b276c9882503714659 HEAD --` after commit.
- [ ] **Step 4: Verify full scope.** Run all requested tests, CLI help, strict checkpoint load, SHA/path audit, `git diff --check`, and a no-commit isolated merge simulation against `3b336996dba15c6227d652b276c9882503714659`.
- [ ] **Step 5: Commit, push, and update PR #2.** Push only the existing branch, keep the PR draft, update its body with fixes and exact counts, obtain a zero-Critical/zero-Important read-only re-review, and do not merge.
