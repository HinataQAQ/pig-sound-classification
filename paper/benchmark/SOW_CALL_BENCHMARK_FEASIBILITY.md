# Sow Call Benchmark Feasibility

Date: 2026-07-11
Decision: **an exact literature-protocol replication is not currently feasible; a true source-grouped benchmark is also blocked by missing lineage metadata.** No benchmark was run.

## Evidence audited

The audit covered the canonical raw Sow Call root, the filename-derived label index, the labeled copy tree, the duplicate `wu_pig_speech` root, all 15 cap3x manifest files, all referenced local files and MD5 values, the two manifest-building scripts, and the 75 existing hierarchical `test_pred.csv` files. Counts are recorded in `SOW_CALL_SOURCE_AUDIT.csv`.

The publisher papers report 4,000 balanced 2 s clips (1,000 calm, feeding, frightened, and anxious). TransformerCNN reports 800 train and 200 test clips per class. The official figshare history is more complicated:

- version 1 exposes `Dataset.zip`, file ID `31337746`, 279,778,027 bytes, MD5 `07ebf72b599718bb90dbc7cf84d9f827`;
- version 2 is currently a metadata-only record with no attached files;
- the local provenance audit links the available roots to the version-1 download, but this task did not redownload or modify audio.

## Exact local inventory

| Original class | Filename suffix | Raw files | Unique MD5 | Labeled-tree files | Current mapping |
|---|---:|---:|---:|---:|---|
| calm | `-0` | 280 | 280 | 560 | `calm_grunt` → `calm_grunt` |
| feeding | `-1` | 210 | 210 | 420 | `feeding` → `feeding` |
| frightened | `-2` | 750 | 749 | 1,500 | `frightened_stress` → `stress_vocal` |
| anxious | `-3` | 506 | 504 | 1,012 | `anxious_stress` → `stress_vocal` |
| **Total** | — | **1,746** | **1,743** | **3,492** | — |

All 1,746 canonical filenames satisfy the numeric-suffix rule, the index has 1,746 complete rows, and the unknown-label index is empty. Original filenames and the local suffix-to-label mapping are therefore recoverable.

There are two within-source duplicate-content groups:

- `605-2.wav` and `606-2.wav` share MD5 `62e1743fb1673742d39c5b9491fc3792`;
- `1267-3.wav`, `1278-3.wav`, and `1289-3.wav` share MD5 `d6a5186c8890715b6cc373c26fde0936`.

The labeled tree contains two physical copies per indexed raw file. `data/raw/wu_pig_speech` is another exact 1,746-filename/content copy and must never be counted as an independent dataset.

## Manifest integrity

Each cap3x fold contains 1,174 training, 120 validation, and 168 test rows. The Sow subset contributes 760/90/126 rows. Across all 15 manifests there are 7,310 rows and 3,812 unique paths, `source_id`s, and MD5 values; the Sow subset has 4,880 rows and 1,637 unique values.

Audit results:

- missing referenced files: `0`;
- blank or invalid `source_id`: `0`;
- blank or invalid MD5: `0`;
- 1,637 referenced Sow files rehashed, mismatches: `0`;
- within-fold train/validation/test overlap at path, `source_id`, and MD5: `0` for every pair in every fold;
- pairwise test-fold overlap at all three identity levels: `0`;
- five-fold test union: 840 clips total, including 630 Sow clips: 210 calm, 210 feeding, 105 frightened, and 105 anxious.

Cross-fold reuse in another fold's training or validation split is expected for cross-validation. Model-specific prototypes, predictions, and calibration artifacts must remain fold-seed specific.

## Can the published 3,200/800 split be recovered?

**No.** Four independent blockers remain:

1. the local canonical corpus has 1,746 files rather than the reported 4,000;
2. no 4,000-item augmentation/selection map is present;
3. neither paper provides the 3,200/800 filename assignment or split seed;
4. the repository contains no historical manifest that reconstructs that assignment.

An arbitrary stratified 80/20 split must not be called the published split. At most, after acquiring the full corpus, it could be labeled a **published-ratio approximation** unless official membership or a deterministic official split rule becomes available.

## Leakage and grouping assessment

The current `source_id` is generated as `parent_folder::filename_stem`. It is complete and useful for exact canonical-file identity, but it has no animal, farm, pen, session, recorder, or parent-recording lineage. As a result:

- grouping by the current `source_id` is mechanically possible but produces singleton groups and is equivalent to a clip split;
- an MD5-deduplicated random split can prevent exact content duplication;
- latent correlation from the same sow, recording session, or parent recording cannot be tested and remains possible;
- the current split must not be described as source-, session-, or animal-independent;
- meaningful grouped 5-fold CV requires new authoritative lineage metadata.

This is an uncertainty statement, not evidence that latent leakage occurred.

## Direct task comparability

The present four-main-class task is **not directly comparable** with the published Sow four-state task:

- the current task adds Korean dry/abdominal cough data;
- calm and feeding remain separate main classes;
- frightened and anxious are merged into `stress_vocal` at the main level;
- evaluation uses five folds × five seeds and Macro-F1, rather than the reported single balanced clip split and headline accuracy.

The local Sow clips can support a separately declared four-state benchmark, but its scores would not be the current main-task scores.

## Six-subtype auxiliary head

The four Sow classes correspond exactly to four of the current six auxiliary labels. All 75 hierarchical prediction files contain aligned `aux_true` and `aux_pred` values, with no row-order mismatch. Therefore a hard-prediction diagnostic is feasible now if cough-subtype predictions are counted as errors.

Limitations:

- the stored files do not contain auxiliary logits or probabilities;
- a masked/renormalized four-way auxiliary decision would require an inference-only rerun;
- the existing 630-Sow test union is imbalanced at 210/210/105/105 and is not the published 800-item test;
- existing models were trained on the mixed cough/Sow task, so this diagnostic is not a fair Sow-only benchmark.

For a new fair benchmark, use the original four states as the evaluation head. Any hierarchical parent head must be declared as a Sow-specific adaptation, not silently presented as the unchanged six-subtype architecture.

## Feasibility verdict

| Question | Verdict |
|---|---|
| Exact four original labels recoverable locally? | Yes, by validated filename suffix. |
| Exact 1,000 clips per class available? | No. |
| Exact published 3,200/800 membership recoverable? | No. |
| Exact path and MD5 disjointness enforceable? | Yes. |
| True animal/session/source grouping possible? | No, not from current metadata. |
| Random split provably source-leakage free? | No; only exact identity/content leakage can be ruled out. |
| Current main task directly comparable? | No. |
| Existing auxiliary hard predictions auditable on four states? | Yes, as a bounded diagnostic only. |

The appropriate next-stage recommendation is `manuscript_only_revision` unless the complete 4,000-item corpus plus authoritative split/lineage metadata becomes available. A clip-level local benchmark could still be run later as a transparently labeled secondary experiment, but it would not resolve the requested fair-comparison question.
