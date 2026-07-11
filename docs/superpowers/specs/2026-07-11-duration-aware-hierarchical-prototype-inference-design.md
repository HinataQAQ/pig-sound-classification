# Duration-Aware Hierarchical Prototype Inference Design

## Approved scope

Build a reproducible, read-only analysis of the locked 5-fold x 5-seed clean
results. The pipeline creates cumulative B0-B3 Macro-F1 tables, five direct
paired comparisons, deterministic prototype case studies, and two publication
figures. It must not train or rerun a model, select a parameter on test data,
change lambda=0.5, create a new threshold, or modify any existing result file.

## Frozen stage identity

Each stage is resolved independently for the same exact `(fold, seed)` grid:

- B0: `reports/cv5_expanded_cap3x_fold{fold}_logmel_seed{seed}/summary.json`,
  `test_macro_f1`, representing 1 s Log-Mel CRNN.
- B1: `reports/cv5_expanded_cap3x_fold{fold}_logmel_dur2_seed{seed}/summary.json`,
  `test_macro_f1`, representing 2 s Log-Mel CRNN.
- B2: `reports/cv5_expanded_cap3x_fold{fold}_logmel_dur2_hier_w05_seed{seed}/summary.json`,
  `test_macro_f1`, representing the raw Softmax output of the lambda=0.5
  hierarchical backbone.
- B3: `reports/prototype_cv5_exact_w05_fold{fold}_seed{seed}/evaluation/metrics.json`,
  `methods.hierarchical.macro_f1`, representing post-hoc hierarchical acoustic
  prototype prediction from the same lambda=0.5 backbone.

The legal grid is folds `0..4` and seeds `42, 123, 777, 2024, 3407`. Every
stage must contain exactly those 25 unique keys. B2 is cross-checked against
B3's `raw_softmax` metric for every key. Prototype runs must say
`input_role=frozen_test`, `selection_method=hierarchical`,
`test_used_for_parameter_selection=false`, `feature_backend=training_exact`,
and pass their leakage/manifest gates.

Every B3 prototype metadata file must bind to the exact expected B2 summary
path and SHA-256 and confirm 2 s input, lambda=0.5, and train/validation/test
source roles. All 15 referenced fold manifests are hash-verified. For the
canonical case run, prediction identities must equal the test manifest and
both representative identity sets must equal the train manifest by path,
source ID, and MD5; the three splits are independently disjoint on all three
keys.
Checkpoint bytes remain outside this read-only analysis. Only the expected
checkpoint path and internal agreement of its declared digest are checked;
the linked B2 summary and all manifests are verified against their actual
bytes.

## Statistics

Compute the five requested comparisons directly from the 25 matched rows:

- B1 - B0
- B2 - B1
- B3 - B2
- B3 - B1
- B3 - B0

For each comparison report both stage means, mean and sample SD of paired
deltas, a 10,000-resample percentile bootstrap 95% CI using seed 3407, a
two-sided Wilcoxon signed-rank P value, and wins/ties/losses. Ties use
`numpy.isclose`; an all-zero contrast has P=1.0. No P value is inferred by
chaining intermediate comparisons.

Fold sensitivity first averages the five seed deltas inside each fold. The
five fold means are reported explicitly. The fold-cluster interval is the
same 10,000-resample percentile bootstrap applied to those five means. This
matches the repository's existing fold-cluster convention and is presented as
a low-cluster-count sensitivity analysis, not as 25 independent folds.

## Deterministic case studies

Use one model-specific artifact only: the lexicographically first validated
prototype run, `(fold, seed)=(0, 42)`. This prevents fold/seed mixing and makes
the choice independent of performance.

Selection rules, always tie-broken by `source_id`, are:

1. For each main class, require raw Softmax, main prototype, and hierarchical
   prototype to all predict the true class. Sort by raw Softmax confidence and
   take the lower median rank, excluding any prior selection.
2. For the feeding boundary case and stress-vocal boundary case, require the
   true label to match the named side, require `{feeding, stress_vocal}` to be
   the Top-2 set for raw Softmax or hierarchical prototype, and require at
   least one of the three main prediction routes to be incorrect. Sort by raw
   Top-2 probability margin and take the lower median rank, excluding prior
   selections.
3. For the uncertain case, sort all remaining rows by the final hierarchical
   prototype Top-2 probability margin and take the minimum. This is a rank
   rule, not a fitted threshold.

Each row exposes all requested Top-2 predictions, cosine distances to all four
main and six subtype prototypes, the hierarchical Top-2 probability margin,
hierarchy consistency, and trace identifiers. The closest representative is
defined conservatively as the train-only sample with minimum own-prototype
cosine distance for the hierarchical predicted main class in that same
fold-seed artifact. It is a prototype representative, not a claimed nearest
neighbour to the test embedding. A subtype representative is also included.
Raw Softmax values are taken from the same B3 evaluation prediction artifact
as the prototype values to preserve sample alignment; they are the audited
same-checkpoint reproduction rather than copied probabilities from the original
B2 prediction CSV. The probability-drift advisory is disclosed separately and
does not alter the B2 Macro-F1 stage value.

## Figure contract

Core conclusion: the cumulative B3 framework combines the statistically strong
2 s context gain with smaller, non-uniform hierarchical and prototype
increments, yielding a directly tested net B3-B0 improvement.

- Archetype: schematic-led quantitative composite.
- Backend: Python/Matplotlib exclusively.
- Export: fixed 183 mm width, editable-text SVG, PDF with TrueType text, and
  300 dpi PNG.
- Figure `figure_cumulative_framework`:
  - a: all 25 run points plus stage means across B0-B3;
  - b: five direct paired increments with run-level and fold-cluster CIs;
  - c: five B3-B0 fold-mean deltas;
  - d: 2 s context -> hierarchical embedding -> prototype prediction.
- Figure `figure_prototype_prediction_cases`: compact case matrix showing the
  three main prediction routes, subtype evidence, margins, consistency, and
  representative trace without playing or relabelling audio.

Reviewer risks addressed explicitly: Macro-F1 is never called accuracy;
incremental nonsignificant results are not marked significant; run and fold
intervals are distinguished; n is defined as fold-seed runs; the case study is
rank-selected from one locked run; and representatives are train-only.

## Implementation boundary

A new standalone generator under `tools/` owns validation, analysis, case
selection, table writes, and rendering. A focused unit test module covers the
strict grid, direct paired statistics, deterministic selection, output schema,
and export contract. A short user document records the exact command and field
definitions. Existing baseline scripts and result files remain unchanged.
