# Sow Call Fair Benchmark Plan

Status: design only. No training, inference, or test-set calibration is authorized by this document.

## Objective and hard boundary

The benchmark question is whether candidate models distinguish the original Sow Call states `calm`, `feeding`, `frightened`, and `anxious` under an explicitly defined split. It is separate from the current cough/calm/feeding/stress main task.

Two protocols are planned and must never be pooled or described with the same comparison language.

## Protocol A — literature-like

Purpose: reproduce the Liao/TransformerCNN task as closely as the source evidence permits.

Required data contract:

- exactly 4,000 2 s clips;
- exactly 1,000 clips for each of the four original labels;
- authoritative original filenames and labels;
- the published 800-train/200-test membership for each class, or an official deterministic split rule and seed;
- MD5 inventory and a provenance record connecting every file to the official corpus.

Evaluation contract:

- retain the official 3,200/800 assignment unchanged;
- derive validation only from the training portion, with fixed deterministic rules; never tune on the 800 test clips;
- report accuracy and Macro-F1, plus per-class precision/recall/F1 and confusion counts;
- report exactly which metric can be compared with each published number;
- label results `literature-protocol replication` only when the data and split contract is satisfied.

Current gate: **blocked**. The local corpus has only 1,746 files, and exact published split membership is unavailable. A newly generated 80/20 split would be labeled `published-ratio approximation`, not Protocol A replication, and would not justify an “outperforms 96.05%” statement.

## Protocol B — leakage-controlled

Purpose: estimate generalization under source-lineage separation rather than mimic the published clip split.

Required data contract:

- original four labels and 2 s clips;
- a stable grouping variable tied to animal, farm/pen, recording session, or parent recording;
- exact path, source-lineage ID, and MD5 inventory;
- all members of a lineage group assigned to one fold only;
- duplicate MD5 and derivative lineage kept in the same fold.

Split and run contract:

- grouped, stratified 5-fold outer CV;
- seeds `42, 123, 777, 2024, 3407` for model initialization/training only; the outer fold membership remains fixed across models and seeds;
- a group-disjoint validation subset derived only from each outer training fold;
- preflight assertions for path, lineage, and MD5 disjointness across train/validation/test;
- store a manifest hash and split audit with each run;
- report accuracy, Macro-F1, per-class F1, confusion counts, mean, sample SD, min, max, and bootstrap CI;
- use exact fold-seed pairs for every model contrast, with Wilcoxon and fold-cluster sensitivity.

Current gate: **blocked for true source grouping**. The present `source_id` is a singleton filename identity, not lineage. Grouping by it would be a clip-level split. Before Protocol B can run, recover authoritative parent-recording/animal/session groups or explicitly downgrade the experiment to `MD5-disjoint clip CV`. The downgraded protocol may be useful, but it cannot support source-generalization language.

## Common preprocessing and label contract

- Preserve the source 2 s waveform; do not edit audio files.
- Use a fixed sample-rate/resampling and Log-Mel definition declared before test evaluation.
- Fit normalization on training data only and apply frozen transforms to validation/test.
- Evaluate the four original labels without merging frightened and anxious.
- Preserve natural class prevalence for Protocol B evaluation. Any sampler or class weighting is a training-only choice and must be fixed before test access.
- Use `zero_division=0` for all classification metrics.

For a Sow-specific hierarchy, the evaluated fine head remains the four original classes. An optional coarse auxiliary head may use the fixed three-parent mapping `calm→calm`, `feeding→feeding`, and `{frightened, anxious}→stress`. This is a benchmark adaptation; it is not the unchanged current four-main/six-subtype schema. The auxiliary weight is fixed from training/validation evidence only.

## Candidate baselines

| ID | Candidate | Frozen comparison contract | Status |
|---|---|---|---|
| S0 | Log-Mel CRNN | Four-class 2 s classifier; identical preprocessing/splits across all neural candidates | reproducible after protocol gate |
| S1 | 2 s hierarchical CRNN without prototype | Same encoder and fine four-class head as S0, plus predeclared coarse auxiliary loss; report fine-head scores | reproducible after schema implementation and protocol gate |
| S2 | kNN on frozen embeddings | Use S1 training embeddings only; select `k`, metric, and weighting on validation; test once | reproducible after protocol gate |
| S3 | main centroid prototype | Four fine-class centroids from same-run training embeddings only; normalize and choose temperature on validation only | reproducible after protocol gate |
| S4 | hierarchical centroid prototype | Training-only fine and coarse centroids; mapping and any fusion/penalty weight frozen from validation | reproducible after protocol gate |
| S5 | TransformerCNN reimplementation | MLMC 145×55; parallel two-layer/five-head Transformer and four-layer CNN as described by Liao et al. | `not_reproducible_without_assumptions` |

TransformerCNN is marked `not_reproducible_without_assumptions` because the publisher article does not provide an official implementation, exact 3,200/800 filenames or random seed, learning rate/epoch/checkpoint-selection contract, and enough versioned preprocessing detail to guarantee an exact numerical reproduction. Architecture tables constrain many shapes, but reproducing the score would still require undocumented choices. Any future implementation must publish an assumption ledger and be labeled a description-based reimplementation, not the original model.

## Hyperparameter and model-selection firewall

1. Write the candidate grid before any test inference.
2. Fit encoders, kNN banks, and centroids on training data only.
3. Use validation data only for early stopping, auxiliary weight, `k`, distance metric, temperature, fusion weights, thresholds, and checkpoint choice.
4. Freeze one configuration per fold-seed before opening its test predictions.
5. Test data are evaluation only; never pick a model, example, threshold, or prototype rule from test performance.
6. Preserve all run outputs in new experiment directories; do not overwrite established results.

## Reporting and fair-comparison language

Protocol A and B results receive separate tables. The external 96.05% TransformerCNN accuracy can be placed in the Protocol A table only after exact alignment of dataset, labels, split, metric, sample unit, clean/noise condition, and class balance. Until then, the only permitted wording is:

> TransformerCNN reported 96.05% accuracy under its published 3,200/800 clip split; the present local/source-controlled protocol is not directly comparable.

Protocol B is scientifically stronger for leakage control but, by design, is not a replication of the published split. A lower or higher number cannot establish underperformance or outperformance across protocols.

## Required outputs for a future authorized run

Each fold-seed directory must contain at least:

- `summary.json` with configuration, hashes, metrics, and calibration provenance;
- `test_pred.csv` with `y_true`, `y_pred`, four probabilities, uncertainty/distance fields, and top-k output where applicable;
- split and leakage audit covering path, lineage, and MD5;
- training-only prototype/retrieval inventory for prototype methods;
- validation-only calibration record;
- failure with a clear message if any required file or lineage field is missing.

Aggregate paper-facing outputs should include run-level, fold-level, paired-statistics, confusion, and per-class tables. No command is supplied yet because no benchmark implementation or protocol-qualified manifests exist; inventing a runnable command now would conceal the data blockers.

## Go/no-go rule

- Run Protocol A only after the full official corpus and exact official split are recovered.
- Run Protocol B only after meaningful lineage groups are recovered.
- If neither is recovered, restrict the next stage to manuscript/comparability revision or a transparently named MD5-disjoint clip-CV diagnostic.

Current recommendation: `manuscript_only_revision`.
