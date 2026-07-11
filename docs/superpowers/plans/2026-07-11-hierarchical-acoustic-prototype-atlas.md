# Hierarchical Acoustic Prototype Atlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task, but only after a new explicit `APPROVED: true` round and after re-reading `handoff/GPT_TO_CODEX.md`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the frozen fold-seed prototype inference layer into a schema-versioned Hierarchical Acoustic Prototype Atlas with main/subtype nodes, geometry-aware query output, validation-only rejection calibration, deterministic few-shot insertion, and a rights-gated open-set protocol, without retraining or changing the CRNN backbone.

**Architecture:** Keep every validated training and prototype script unchanged. Add one pure Atlas contract/core module, then thin build, calibration, prediction, evaluation, few-shot, open-set, and plotting CLIs around the existing `prototype_model_adapter.py` model/feature/leakage functions. Train embeddings build nodes and exemplars, validation chooses all free parameters, and frozen clean test data are evaluation-only.

**Tech Stack:** Python 3, NumPy, pandas, PyTorch, scikit-learn, Matplotlib, existing `pigsound-gpu` Conda environment; Windows PowerShell commands only; no new dependency.

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-11
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Global Constraints

- This document is a plan. No Atlas build, prediction, calibration, evaluation, few-shot run, open-set run, DEMAND run, or CV run was executed while producing it.
- Preserve all current clean/noise checkpoints, summaries, predictions, tables, and report directories byte-for-byte.
- Do not modify `tools/train_mctafd_crnn_ablation.py` or `tools/train_hier_longcontext_crnn.py`; do not change the backbone, loss, checkpoint, or audio.
- The only future debug cohort authorized by this plan is fold `0`, seed `3407`, hierarchy weight `0.5`, duration `2.0`, feature mode `logmel`, feature backend `training_exact`, and clean test only.
- Never use test labels, test metrics, or open-test samples to select weights, temperatures, kNN k, rejection thresholds, source labels, support samples, or open-set operating points.
- Prototypes, radii, dispersion, medoids, Log-Mel summaries, and exemplars come from train only. Known-class calibration comes from clean validation only. Test is evaluation only.
- Use exact-path, normalized `source_id`, and MD5 disjointness; record manifest/checkpoint/artifact SHA-256 values; fail closed on mismatch.
- Write only to the new debug root `reports/hier_acoustic_atlas_fold0_seed3407_w05_debug_v1/`; never pass `--allow_overwrite` in the debug command.
- One fold-seed remains `single_fold_debug=true`, `paper_main_result=false`, and is not scientific evidence for a paper claim.
- Do not run open-set evaluation until every included source has confirmed content rights, a frozen source manifest, a source-group split, and zero audited overlap with clean train/validation/test.
- All classification metrics use `zero_division=0`.

## Locked Debug Inputs

| Input | Path | SHA-256 / rows |
|---|---|---|
| checkpoint | `checkpoints/cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407.pt` | `d6ec8dcdd6b16a6b3212a732e1d235428ba4c7c1f5702b5b1330ca9a3c449347` |
| training summary | `reports/cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407/summary.json` | `b0e92e1a86081559e46eabea4719322f259b83ca58834106256f247fcacde4ec` |
| train manifest | `paper_results/manifests/manifests_pigvocal_4class_expanded_train_cv5_cap3x/fold0/train.csv` | `33337bf683c9a07b8bdf9dc5a0020e53e6883716924c6deafc06af85720b005b`; 1174 rows |
| validation manifest | `paper_results/manifests/manifests_pigvocal_4class_expanded_train_cv5_cap3x/fold0/val.csv` | `ed86bb19512f4b36d3ed4630251e5e15b9fe38d8d2d42119eb90058e6689dd79`; 120 rows |
| clean test manifest | `paper_results/manifests/manifests_pigvocal_4class_expanded_train_cv5_cap3x/fold0/test.csv` | `4c2eedb0766725c84da2a41b9ec0793b874f9a3d10a80a0a79f4ada7342c5e68`; 168 rows |
| canonical prior prototype cohort | `reports/prototype_cv5_exact_w05_fold0_seed3407/` | reference-only; do not overwrite |

Do not use `reports/prototype_cv5_fold0_seed3407/` because it is the non-paper-equivalent NumPy feature lineage. Do not use the older `reports/prototype_cv5_fold0_seed3407_exact/` lineage as the canonical comparison.

## Frozen Semantics

### Exact Atlas node contract

Every node record must contain these 18 required fields, with no implicit aliases:

- Identity and hierarchy: `node_id`, `main_class`, `subtype`, `parent_node`.
- Geometry and support: `prototype_embedding`, `sample_count`, `cosine_radius_q95`, `mean_cosine_distance`, `std_cosine_distance`.
- Representatives and acoustic maps: `medoid_sample`, `top_representative_samples`, `mean_normalized_logmel`, `std_normalized_logmel`, `nearest_prototype_nodes`.
- Provenance: `source_fold`, `source_seed`, `train_manifest_sha`, `checkpoint_sha`.

### Node identity

- Main node ID: `main::<main_class>`; `subtype=null`; `parent_node=null`.
- Subtype node ID: `subtype::<subtype>`; `parent_node=main::<main_class>`.
- Retain distinct main/subtype nodes for `calm_grunt` and `feeding` even though their train memberships and prototype vectors are identical. This preserves the required hierarchy and must be marked by `semantic_alias=true` as an extra audit field.
- `medoid_sample` means the train sample minimizing total within-node cosine distance. With L2-normalized samples and the normalized class mean, this is exactly the rank-1 closest-to-prototype sample; ties resolve by normalized path, then source ID, then MD5.
- `nearest_prototype_nodes` is the five closest nodes from the unified 10-node cosine-distance matrix, excluding self and sorting ties by `node_id`.

### Query fields

- Every query record must contain these 15 required fields: `main_top1`, `main_top2`, `subtype_top1`, `subtype_top2`, `main_top1_distance`, `main_top2_distance`, `subtype_top1_distance`, `subtype_top2_distance`, `main_margin`, `subtype_margin`, `hierarchy_consistent`, `closest_representative_sample`, `uncertain`, `unknown_candidate`, and `rejection_reason`.
- `main_top1/main_top2` and `subtype_top1/subtype_top2` are Atlas path-ranking outputs, not Softmax outputs.
- Each `*_distance` is raw cosine distance from the query embedding to the corresponding node prototype.
- `main_margin = main_top2_distance - main_top1_distance`; `subtype_margin` is defined analogously. Both are non-negative.
- `hierarchy_consistent = (main_top1 == parent(subtype_top1))`.
- `closest_representative_sample` is the minimum-cosine-distance sample among the stored top representatives of the winning main and subtype nodes; record its path, source ID, MD5, node ID, and query distance.
- `uncertain` is the single known-class rejection boolean chosen by validation. Do not expose two competing global/per-class booleans as the canonical decision.
- In known-only debug mode, `unknown_candidate=null` and `rejection_reason` may contain known-class geometry reasons. Formal `unknown_candidate` becomes boolean only when a rights-approved unknown-validation calibration artifact is supplied.

### Atlas ranking and rejection

For a subtype node `s` with parent main node `m`:

```text
normalized_main_distance = d_cos(query, prototype_m) / max(radius_q95_m, 1e-8)
normalized_subtype_distance = d_cos(query, prototype_s) / max(radius_q95_s, 1e-8)
path_distance(m, s) = w_main * normalized_main_distance
                    + (1 - w_main) * normalized_subtype_distance
```

Validation selects `w_main` from the frozen grid `0,0.25,0.5,0.75,1`. Rank subtype paths by ascending path distance; rank each main by its best child path. Select by validation main Macro-F1, then subtype Macro-F1, then distance to the balanced value `0.5`.

The v1 known-class rejection score is:

```text
radius_ratio = max(normalized_main_distance_top1,
                   normalized_subtype_distance_top1)
```

Validation selects one global `radius_ratio` threshold at target known coverage `0.95`, breaking ties by lower selective risk and then the smaller threshold. Main/subtype margins and hierarchy consistency are reported and appear in `rejection_reason`, but are not extra tunable gates in v1. This keeps the rejection model auditable and avoids fitting a second classifier.

## File Map

- Create `tools/hier_acoustic_atlas_core.py`: schema, geometry, ranking, rejection, deterministic support selection, insertion, and artifact validation.
- Create `tools/build_hier_acoustic_atlas.py`: train-only node/exemplar/Log-Mel builder with provenance and leakage gates.
- Create `tools/calibrate_hier_acoustic_atlas.py`: validation-only Atlas weight, kNN k, temperature, and known rejection calibration.
- Create `tools/predict_hier_acoustic_atlas.py`: frozen manifest/single-audio query contract and five baseline outputs.
- Create `tools/eval_hier_acoustic_atlas.py`: frozen clean-test metrics only; no grid or parameter selection.
- Create `tools/eval_atlas_fewshot_extension.py`: deterministic leave-one-subtype-node-out insertion protocol.
- Create `tools/eval_atlas_open_set.py`: rights gate, source-group split validation, unknown-validation calibration, frozen open-test evaluation.
- Create `tools/plot_hier_acoustic_atlas.py`: node graph, unified distance matrix, radii, normalized Log-Mel, and representative views.
- Create `tests/test_acoustic_atlas_pipeline.py`: pure-function, schema, provenance, leakage, determinism, and CLI-negative tests.
- Create `docs/HIERARCHICAL_ACOUSTIC_ATLAS.md`: user commands, schemas, outputs, claim boundaries, and troubleshooting.
- Do not modify the nine existing files audited for this plan. Reuse them by import or treat their outputs as read-only references.

### Task 1: Atlas contract and pure geometry

**Files:**
- Create: `tools/hier_acoustic_atlas_core.py`
- Create: `tests/test_acoustic_atlas_pipeline.py`

**Interfaces:**
- Produces `ATLAS_SCHEMA_VERSION = "hier_acoustic_atlas.v1"`.
- Produces frozen dataclasses `AtlasNode`, `AtlasCalibration`, and `AtlasQueryResult` with the exact required fields.
- Produces `build_node_records(...)`, `rank_atlas_paths(...)`, `calibrate_known_rejection(...)`, `apply_rejection(...)`, `deterministic_support_order(...)`, and `insert_subtype_node(...)`.

- [ ] **Step 1: Write schema and ID tests.** Assert 4 main plus 6 subtype records, exact required keys, `main::`/`subtype::` IDs, and valid parent references.
- [ ] **Step 2: Run the focused tests and verify failure because the module does not exist.**

```powershell
conda activate pigsound-gpu
cd C:\py\pigsound\pig-sound-classification
$env:PYTHONNOUSERSITE="1"
python -m unittest tests.test_acoustic_atlas_pipeline.AtlasContractTests -v
```

- [ ] **Step 3: Implement the dataclasses and strict validators.** Reject missing/extra required fields, non-finite arrays, non-unit prototype vectors, invalid parent nodes, duplicate IDs, and inconsistent fold/seed/SHA values.
- [ ] **Step 4: Write and pass geometry tests.** Cover q95/mean/std, 10-node neighbor sorting, the cosine-medoid equivalence, top-2 distance alignment, margins, hierarchy consistency, and `radius_ratio` threshold selection.
- [ ] **Step 5: Commit the independently testable contract.**

```powershell
git add -- tools/hier_acoustic_atlas_core.py tests/test_acoustic_atlas_pipeline.py
git commit -m "feat: define hierarchical acoustic atlas contract"
```

### Task 2: Train-only Atlas builder

**Files:**
- Create: `tools/build_hier_acoustic_atlas.py`
- Modify: `tests/test_acoustic_atlas_pipeline.py`

**Interfaces:**
- Consumes the locked checkpoint, summary, and train/validation/test manifests.
- Reuses `load_config`, `validate_expected_config`, `load_hier_model`, `build_hier_dataset`, `extract_embeddings`, `compute_class_prototypes`, `compute_feature_stats`, and `audit_manifest_disjointness` from `tools/prototype_model_adapter.py`.
- Produces `atlas/atlas_nodes.json`, `atlas/atlas_bundle.npz`, `atlas/train_exemplars.csv`, `atlas/train_exemplar_embeddings.npz`, `atlas/node_distance_matrix.csv`, `atlas/atlas_metadata.json`, `atlas/leakage_audit.json`, and root `run_manifest.json`.

- [ ] **Step 1: Write builder negative tests.** Missing checkpoint, wrong lambda, wrong fold/seed, non-`training_exact` paper qualification, duplicate within-train path/source/MD5, cross-split overlap, and existing output root must fail clearly.
- [ ] **Step 2: Implement CLI parsing with all hyperparameters configurable and a working `--help`.** Require the same locked-value guards as the current prototype builder and default `--top_representatives 5`.
- [ ] **Step 3: Extract train embeddings/features exactly once.** Cache row-aligned normalized embeddings with path/source/MD5 and bind both CSV and NPZ by SHA-256 in `atlas_metadata.json`.
- [ ] **Step 4: Build and validate all ten nodes.** Embed required fields directly in `atlas_nodes.json`; store the same dense arrays in NPZ for efficient querying.
- [ ] **Step 5: Run unit tests and a schema-only fixture build.** The fixture uses synthetic arrays, not project audio.
- [ ] **Step 6: Commit the builder.**

```powershell
git add -- tools/build_hier_acoustic_atlas.py tests/test_acoustic_atlas_pipeline.py
git commit -m "feat: build train-only acoustic atlas nodes"
```

### Task 3: Validation-only Atlas calibration

**Files:**
- Create: `tools/calibrate_hier_acoustic_atlas.py`
- Modify: `tests/test_acoustic_atlas_pipeline.py`

**Interfaces:**
- Consumes `atlas_bundle.npz`, `atlas_metadata.json`, validation manifest, and matching checkpoint.
- Produces `calibration/atlas_calibration.json`, `calibration/atlas_weight_grid.csv`, `calibration/knn_k_grid.csv`, `calibration/val_predictions.csv`, `calibration/coverage_risk.csv`, and `calibration/leakage_audit.json`.
- Selects kNN `k` from `1,3,5,7,9` using validation main Macro-F1, then smaller k. kNN is unweighted cosine majority vote; ties resolve by summed cosine similarity, then fixed label order.

- [ ] **Step 1: Write tests proving that only validation labels reach selection functions.** A metadata role other than `validation` must fail.
- [ ] **Step 2: Implement validation extraction, Atlas grid scoring, kNN selection, and the 95%-coverage `radius_ratio` threshold.**
- [ ] **Step 3: Bind checkpoint, manifest, Atlas bundle, nodes, exemplars, and calibration tables by SHA-256.** Any relocated file is accepted only when its hash matches.
- [ ] **Step 4: Test deterministic tie-breaking and zero test-path access.**
- [ ] **Step 5: Commit the calibrator.**

```powershell
git add -- tools/calibrate_hier_acoustic_atlas.py tests/test_acoustic_atlas_pipeline.py
git commit -m "feat: calibrate atlas on validation only"
```

### Task 4: Frozen Atlas predictor and exact query contract

**Files:**
- Create: `tools/predict_hier_acoustic_atlas.py`
- Modify: `tests/test_acoustic_atlas_pipeline.py`

**Interfaces:**
- Accepts exactly one of `--test_manifest`, `--manifest`, or `--audio`.
- Produces `evaluation/test_pred.csv`, `evaluation/test_pred.json`, `evaluation/prediction_metadata.json`, and `evaluation/leakage_audit.json`.
- Emits the required query fields plus auditable baseline probability/distance columns.

- [ ] **Step 1: Write query-schema tests with two main and three subtype fixture nodes.** Verify exact top1/top2 labels, paired distances, non-negative margins, consistency, nearest representative, `uncertain`, nullable `unknown_candidate`, and deterministic ordered `rejection_reason` codes.
- [ ] **Step 2: Implement artifact/hash validation before checkpoint loading.** Include the inference manifest itself in the train/validation/query path/source/MD5 audit when it contains those columns.
- [ ] **Step 3: Implement five method outputs.** Methods are `raw_softmax`, `main_single_centroid`, `hierarchical_single_centroid`, `knn_exemplar`, and `hierarchical_acoustic_atlas`.
- [ ] **Step 4: Make known-only behavior explicit.** Without a rights-approved open-set calibration block, write `unknown_candidate` as JSON null and blank CSV, `unknown_detection_claim=false`, and never alias `uncertain` to unknown.
- [ ] **Step 5: Pass schema/hash/input-role tests and commit.**

```powershell
git add -- tools/predict_hier_acoustic_atlas.py tests/test_acoustic_atlas_pipeline.py
git commit -m "feat: add frozen acoustic atlas prediction contract"
```

### Task 5: Frozen clean evaluator and baseline parity

**Files:**
- Create: `tools/eval_hier_acoustic_atlas.py`
- Modify: `tests/test_acoustic_atlas_pipeline.py`

**Interfaces:**
- Consumes only frozen `input_role=frozen_test` predictions plus calibration/provenance artifacts.
- Produces root `summary.json`, `evaluation/metrics_by_method.csv`, `evaluation/subtype_metrics_by_method.csv`, `evaluation/confusion_matrices.json`, `evaluation/selective_risk.csv`, and `evaluation/hierarchy_consistency.json`.

- [ ] **Step 1: Write evaluator rejection tests.** Reject inference manifests, missing true labels, changed prediction SHA, wrong calibration SHA, unsafe checkpoint bypass, or `paper_main_result=true` on a single fold.
- [ ] **Step 2: Implement Top-1, Top-2, Macro-F1, calibration, per-class, consistency, and frozen selective-risk metrics with `zero_division=0`.**
- [ ] **Step 3: Do not emit a test threshold sweep.** Evaluate only the pre-frozen validation operating point, preventing post-hoc test threshold selection.
- [ ] **Step 4: Add parity assertions.** For the first three baselines, row order and metrics must reproduce the canonical prior cohort within `1e-6`; any mismatch fails the debug.
- [ ] **Step 5: Commit the evaluator.**

```powershell
git add -- tools/eval_hier_acoustic_atlas.py tests/test_acoustic_atlas_pipeline.py
git commit -m "feat: evaluate atlas against frozen baselines"
```

### Task 6: Deterministic few-shot subtype insertion

**Files:**
- Create: `tools/eval_atlas_fewshot_extension.py`
- Modify: `tests/test_acoustic_atlas_pipeline.py`

**Interfaces:**
- Future full protocol evaluates pseudo-new subtypes `dry_cough`, `abdominal_cough`, `frightened_stress`, and `anxious_stress`; `calm_grunt` and `feeding` are ineligible because removing their only subtype removes the whole main class.
- For each target, remove all target-subtype train exemplars from the base Atlas, insert a node from k train support samples, and evaluate the full frozen clean test query set.
- Reports k=`1,3,5,10`; future screening uses 20 deterministic support repetitions. The debug uses only target `dry_cough`, one repetition (`repeat=0`), and the four nested k values.

- [ ] **Step 1: Write deterministic support tests.** Hash ordering uses `SHA256("atlas-fewshot-v1|fold|model_seed|subtype|repeat|source_id|md5|normalized_path")`; each k takes the prefix of one permutation, so k sets are nested. Support must be train-only and unique by path/source/MD5.
- [ ] **Step 2: Implement leave-one-subtype-node-out reconstruction.** Recompute the affected parent from remaining train exemplars plus k supports; never use non-support target train samples as Atlas exemplars.
- [ ] **Step 3: Handle few-shot dispersion without query leakage.** Use the sibling subtype radius as a prior and fixed shrinkage `w=k/(k+5)`: `radius=w*empirical_q95+(1-w)*sibling_q95`. Record the prior source and formula in every inserted node.
- [ ] **Step 4: Implement matched baselines.** Raw auxiliary Softmax is labeled `raw_softmax_seen_head_upper_bound` because the backbone/head already saw all six subtypes during training; kNN and Atlas receive exactly the same support draw.
- [ ] **Step 5: Report per-repeat and aggregate Top-1, Top-2, Macro-F1 mean/std/min/max.** Support repetitions are sensitivity replicates, not independent model runs; no Wilcoxon claim is allowed within one fold-seed.
- [ ] **Step 6: Pass no-support/query-overlap and no-test-selection tests, then commit.**

```powershell
git add -- tools/eval_atlas_fewshot_extension.py tests/test_acoustic_atlas_pipeline.py
git commit -m "feat: add deterministic atlas few-shot insertion"
```

### Task 7: Rights-gated open-set evaluation

**Files:**
- Create: `tools/eval_atlas_open_set.py`
- Modify: `tests/test_acoustic_atlas_pipeline.py`

**Interfaces:**
- Requires `--source_rights_csv`, `--unknown_val_manifest`, and `--unknown_test_manifest` in addition to known validation/test inputs.
- Each rights row must contain `source`, `source_url`, `content_license`, `label`, `file_count`, `rights_status`, `paper_use_permission`, `source_group_key`, `manifest_sha256`, and overlap-audit SHA.
- `rights_status` and `paper_use_permission` must both equal `confirmed`; otherwise the script exits before loading audio or a checkpoint.

- [ ] **Step 1: Write the hard rights-gate tests.** Unknown, missing, noncommercial-with-commercial-target, mixed-license, or unverified rows must fail before inference.
- [ ] **Step 2: Implement source-group and identity audits.** Unknown validation and unknown test must be disjoint by source recording/uploader group, path, source ID, MD5, and source archive; both must be disjoint from clean train/validation/test.
- [ ] **Step 3: Implement validation-only open threshold selection.** Use `radius_ratio`; require known-validation coverage at least 0.95, then maximize validation balanced accuracy for known-vs-unknown and break ties by the smaller threshold.
- [ ] **Step 4: Freeze and evaluate.** Report AUROC, AUPR-unknown, FPR at 95% known TPR, unknown recall/precision/F1, known acceptance, OSCR, and Macro-F1 including unknown, stratified by source and by `far_ood` versus `near_ood`.
- [ ] **Step 5: Prevent semantic mislabeling.** Pig grunt/oink is near-OOD relative to `calm_grunt`; scream/squeal may overlap `stress_vocal`; background pig mixtures are deployment-boundary samples. They are not formal unknown positives unless the frozen ontology audit explicitly approves that mapping.
- [ ] **Step 6: Commit the gated evaluator without running it.**

```powershell
git add -- tools/eval_atlas_open_set.py tests/test_acoustic_atlas_pipeline.py
git commit -m "feat: add rights-gated atlas open-set protocol"
```

### Task 8: Plotting and documentation

**Files:**
- Create: `tools/plot_hier_acoustic_atlas.py`
- Create: `docs/HIERARCHICAL_ACOUSTIC_ATLAS.md`
- Modify: `tests/test_acoustic_atlas_pipeline.py`

- [ ] **Step 1: Plot the 10-node hierarchy and unified cosine-distance matrix.** Distinguish main/subtype nodes and parent edges; retain the zero-distance alias nodes visibly.
- [ ] **Step 2: Plot per-node q95 radius/dispersion and normalized Log-Mel mean/std.** State that Log-Mel maps are per-sample z-scored and do not show absolute energy.
- [ ] **Step 3: Export representative/medoid tables and query examples without copying audio.**
- [ ] **Step 4: Document all exact CLI commands, field definitions, output files, rights gates, baseline meanings, and scientific limitations.**
- [ ] **Step 5: Test `--help`, missing-file errors, non-empty plot outputs, and commit.**

```powershell
git add -- tools/plot_hier_acoustic_atlas.py docs/HIERARCHICAL_ACOUSTIC_ATLAS.md tests/test_acoustic_atlas_pipeline.py
git commit -m "docs: document and visualize acoustic atlas"
```

### Task 9: Authorized fold0/seed3407 clean debug only

**Files:**
- Create outputs only under `reports/hier_acoustic_atlas_fold0_seed3407_w05_debug_v1/`.
- Do not stage report artifacts unless a later explicit instruction requests it.

- [ ] **Step 1: Run all `--help` checks and unit tests.**
- [ ] **Step 2: Snapshot SHA-256 values for locked clean/noise result files before the debug.**
- [ ] **Step 3: Build the train-only Atlas.**
- [ ] **Step 4: Calibrate on clean validation only.**
- [ ] **Step 5: Predict and evaluate the frozen clean test once.**
- [ ] **Step 6: Run one deterministic few-shot target debug: `dry_cough`, repeat 0, k=`1,3,5,10`.**
- [ ] **Step 7: Plot the Atlas.**
- [ ] **Step 8: Recheck result SHA snapshots, leakage reports, output schema, and canonical baseline parity.**

Planned PowerShell commands, explicitly **NOT RUN in the planning round**:

```powershell
conda activate pigsound-gpu
cd C:\py\pigsound\pig-sound-classification
$env:PYTHONNOUSERSITE="1"

$ROOT="reports\hier_acoustic_atlas_fold0_seed3407_w05_debug_v1"
$TRAIN="paper_results\manifests\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\train.csv"
$VAL="paper_results\manifests\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\val.csv"
$TEST="paper_results\manifests\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\test.csv"
$CKPT="checkpoints\cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407.pt"
$SUMMARY="reports\cv5_expanded_cap3x_fold0_logmel_dur2_hier_w05_seed3407\summary.json"

python -m unittest tests.test_acoustic_atlas_pipeline -v

python tools\build_hier_acoustic_atlas.py `
  --train_manifest $TRAIN --val_manifest $VAL --test_manifest $TEST `
  --ckpt $CKPT --summary_json $SUMMARY --out_dir $ROOT `
  --fold 0 --seed 3407 --expected_hier_aux_weight 0.5 `
  --expected_dur_s 2.0 --expected_feature_mode logmel `
  --feature_backend training_exact --top_representatives 5

python tools\calibrate_hier_acoustic_atlas.py `
  --atlas_bundle $ROOT\atlas\atlas_bundle.npz `
  --atlas_metadata $ROOT\atlas\atlas_metadata.json `
  --val_manifest $VAL --ckpt $CKPT --out_dir $ROOT `
  --main_weight_grid 0,0.25,0.5,0.75,1 `
  --knn_k_grid 1,3,5,7,9 --target_coverage 0.95

python tools\predict_hier_acoustic_atlas.py `
  --atlas_bundle $ROOT\atlas\atlas_bundle.npz `
  --atlas_metadata $ROOT\atlas\atlas_metadata.json `
  --calibration_json $ROOT\calibration\atlas_calibration.json `
  --test_manifest $TEST --ckpt $CKPT --out_dir $ROOT

python tools\eval_hier_acoustic_atlas.py `
  --pred_csv $ROOT\evaluation\test_pred.csv `
  --calibration_json $ROOT\calibration\atlas_calibration.json `
  --out_dir $ROOT

python tools\eval_atlas_fewshot_extension.py `
  --atlas_bundle $ROOT\atlas\atlas_bundle.npz `
  --atlas_metadata $ROOT\atlas\atlas_metadata.json `
  --train_manifest $TRAIN --val_manifest $VAL --test_manifest $TEST `
  --ckpt $CKPT --out_dir $ROOT `
  --target_subtypes dry_cough --k_values 1,3,5,10 `
  --repeats 1 --support_seed 3407 --feature_backend training_exact

python tools\plot_hier_acoustic_atlas.py `
  --atlas_bundle $ROOT\atlas\atlas_bundle.npz `
  --atlas_metadata $ROOT\atlas\atlas_metadata.json `
  --out_dir $ROOT
```

No command for `eval_atlas_open_set.py` belongs in this debug because source rights and frozen unknown manifests are not yet approved.

### Task 10: Verification, paper boundary, and handoff

**Files:**
- Modify: `handoff/CODEX_TO_GPT.md`
- Modify: `handoff/CODEX_TO_GPT.json`
- Modify: `handoff/HISTORY.md`

- [ ] **Step 1: Run the complete Atlas unit suite and existing prototype core suite.**
- [ ] **Step 2: Review the diff for accidental training changes, test-set selection, output overwrite, and unknown overclaim.**
- [ ] **Step 3: Record exact commands, branch/commit, changed files, debug metrics, all SHA/leakage results, paper usability, blockers, and scientific questions.**
- [ ] **Step 4: Mark the debug `paper_usable=false` until at least the required 5-fold x 3-seed screening and rights-approved open-set protocol exist.**
- [ ] **Step 5: Commit and push only the approved code/docs/handoff files, then stop.**

Do not edit `paper/manuscript/paper_en_full_story_polished.md` during Atlas engineering. It currently makes DEMAND a main contribution and explicitly states that open-set rejection is absent. A later separately approved manuscript round may move DEMAND to supplementary/deployment-boundary analysis and promote Atlas only after matched screening/final evidence supports that narrative.

## Acceptance Criteria

1. Every new CLI supports `--help`, validates missing files clearly, defaults to no overwrite, and records a schema version.
2. Exactly ten Atlas nodes are built for the locked label hierarchy, and every required node field is present, finite, typed, and provenance-bound.
3. Query CSV/JSON contains every required field with the frozen semantics above; top2 labels match top2 distances; margins are non-negative.
4. Train builds geometry; validation selects all free parameters; clean test is touched once for frozen evaluation and never produces a selectable threshold grid.
5. The first three baselines reproduce the canonical fold0/seed3407 clean cohort within `1e-6`; kNN and Atlas are additional outputs.
6. All train/validation/test/support/query identities remain disjoint by normalized path, source ID, and MD5; all relevant artifact SHAs match.
7. The few-shot debug is limited to `dry_cough`, repeat 0, k=`1,3,5,10`; the report calls it pseudo-new subtype insertion under a subtype-seen backbone, not true unseen-class learning.
8. Open-set code fails closed on unconfirmed rights and is not run in the debug; `unknown_candidate` remains unsupported/null without an approved unknown calibration artifact.
9. No training script, checkpoint, audio, clean result, noise result, or established prototype result is modified.
10. `summary.json`, `test_pred.csv`, documentation, tests, and all required handoff records exist for the debug, and the round stops without launching full CV or the next stage.

## Self-Review

- Spec coverage: all 18 node fields, all 15 query fields, five baselines, few-shot k values, rights-gated open-set handling, expected outputs, and the restricted debug cohort map to explicit tasks.
- Placeholder scan: no implementation step depends on an unspecified dataset, threshold, grid, label mapping, output root, or metric rule.
- Type consistency: node IDs, main/subtype parent mapping, query distance/margin meanings, calibration artifacts, and output paths are defined once and reused across tasks.
- Scientific boundary: current Atlas debug cannot establish novelty, statistical superiority, unknown detection, or true unseen-subtype generalization.
