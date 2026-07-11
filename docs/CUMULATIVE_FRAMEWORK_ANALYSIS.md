# Duration-Aware Hierarchical Prototype Inference Framework

This analysis combines the four locked clean-audio stages without retraining or
rerunning inference:

- B0: 1 s Log-Mel CRNN.
- B1: 2 s Log-Mel CRNN.
- B2: B1 plus hierarchical auxiliary supervision at lambda=0.5, evaluated with
  its raw Softmax main output.
- B3: the matched B2 backbone evaluated with post-hoc hierarchical acoustic
  prototype prediction.

## Reproduce

Run from the project root in the required environment:

```powershell
conda activate pigsound-gpu
cd C:\py\pigsound\pig-sound-classification
$env:PYTHONNOUSERSITE="1"
python tools/generate_cumulative_framework_analysis.py --help
python tools/generate_cumulative_framework_analysis.py --root . --validate-only
python tools/generate_cumulative_framework_analysis.py --root .
```

The paper run uses 10,000 percentile-bootstrap resamples and seed 3407. The
script validates that each stage has exactly folds 0-4 and seeds
42/123/777/2024/3407, that B2 exactly equals the raw Softmax metric stored in
the prototype cohort, and that every prototype run is a frozen-test,
training-exact, leakage-audited run with no test-set parameter selection.
Each of the 25 prototype metadata files must also bind to the expected B2
summary path and SHA-256, 2 s duration, lambda=0.5, and train/validation/test
role contract. The recorded hashes of all 15 fold manifests are checked
against the actual manifest files.
Checkpoint bytes are deliberately not read in this analysis; the validator
checks the expected checkpoint path and internal agreement of its declared
digest, while the linked B2 summary itself is byte-hash verified.

## Direct paired results

All differences are computed directly from the 25 matched fold-seed rows. The
Wilcoxon test is two-sided; SD is the sample SD of paired deltas. The
fold-cluster interval first averages the five seeds within each fold and then
bootstraps the five fold means.

| Comparison | Baseline mean | Final mean | Mean delta | Run bootstrap 95% CI | Wilcoxon P | W/T/L | Fold-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 - B0 | 0.922606 | 0.947237 | +0.024631 | [0.014113, 0.035642] | 0.000162303 | 21/0/4 | [0.006578, 0.048280] |
| B2 - B1 | 0.947237 | 0.951050 | +0.003812 | [-0.001320, 0.008727] | 0.170241 | 16/1/8 | [-0.003407, 0.010561] |
| B3 - B2 | 0.951050 | 0.953986 | +0.002937 | [-0.000478, 0.007294] | 0.058253 | 13/8/4 | [-0.000220, 0.008263] |
| B3 - B1 | 0.947237 | 0.953986 | +0.006749 | [0.002389, 0.011270] | 0.013809 | 18/0/7 | [0.001295, 0.010597] |
| B3 - B0 | 0.922606 | 0.953986 | +0.031380 | [0.022821, 0.040301] | 2.98023e-07 | 24/0/1 | [0.017174, 0.049576] |

B2-B1 and B3-B2 are not statistically significant. The B3-B0 result above is
a direct comparison; it is not inferred from intermediate P values.

The five B3-B0 fold-mean deltas are +0.010828, +0.035278, +0.020681,
+0.022855, and +0.067257. All five folds are positive, while the cluster count
is only five and should be described as a sensitivity analysis.

## Tables

- `paper/tables/cumulative_framework_runs.csv`: one row per matched fold-seed
  run, B0-B3 Macro-F1 values, lambda, and exact source paths.
- `paper/tables/cumulative_framework_summary.csv`: n, mean, sample SD, minimum,
  and maximum Macro-F1 for each stage.
- `paper/tables/cumulative_framework_paired_stats.csv`: all five direct
  contrasts, both stage means, delta statistics, run-level bootstrap CI,
  Wilcoxon P, wins/ties/losses, all five fold means, and fold-cluster CI.
- `paper/tables/cumulative_framework_fold_stats.csv`: one row for every
  comparison-fold pair, with five-seed stage means and paired deltas.
- `paper/tables/prototype_prediction_case_studies.csv`: seven deterministic
  prediction cases and their complete trace fields.

Byte-identical paper-ready mirrors are archived under `paper_results/tables/`,
and the generator is mirrored at
`paper_results/scripts/generate_cumulative_framework_analysis.py`.

All reported performance values are Macro-F1, not accuracy.

## Deterministic case studies

Cases come only from the lexicographically first validated model artifact,
fold 0 seed 42. No cases or prototype artifacts are mixed across folds or
seeds.

- Four correct-class cases require raw Softmax, main prototype, and
  hierarchical prototype all to be correct. The lower median raw-confidence
  row is selected per class.
- One true-feeding and one true-stress boundary error require feeding and
  stress-vocal as the relevant Top-2 pair and at least one incorrect main
  prediction route. The lower median raw Top-2 margin is selected in each
  class-specific pool.
- The uncertain case is the remaining row with the smallest final hierarchical
  prototype Top-2 margin. This is a deterministic rank rule, not a fitted
  threshold.

Ties are resolved by source ID. The table includes the true class, raw/main/
hierarchical main Top-2, prototype-subtype Top-2, all main and subtype cosine
distances, hierarchical Top-1-minus-Top-2 probability margin, final hierarchy
consistency, and path/source-ID/MD5 trace fields.

The case-table Raw Softmax columns come from the same B3 evaluation
`test_predictions.csv` as the prototype columns, so the four prediction routes
remain exactly sample-aligned. Those values are a same-checkpoint B2 forward
reproduction, not a copy of the original B2 `test_pred.csv`. The existing
25-run reproduction audits report exact class-prediction matches and Macro-F1
matches; their probability comparison is advisory and records maximum absolute
drift up to 0.000944912 across the cohort (0.000260174 for the canonical fold 0,
seed 42 case artifact). This does not change any B2 stage Macro-F1, but it is
recorded because a near-median confidence rank can differ between the two
probability files.

`hierarchy_consistency` means that the final hierarchical main prediction
equals the main class mapped from the auxiliary-prototype prediction. It is not
the older stored inconsistency flag, which compares the plain main prototype
with the mapped auxiliary prototype.

The `closest_representative_*` fields identify the same-run training sample
with minimum own-prototype cosine distance for the predicted class. This is a
train-only closest-to-prototype representative. It is not claimed to be the
nearest training neighbour to the test query because query-to-training
embeddings were not saved, and recomputing them would violate this round's
existing-data-only constraint.

## Figures

- `paper/figures_journal/figure_cumulative_framework.{svg,pdf,png}`
- `paper/figures_journal/figure_prototype_prediction_cases.{svg,pdf,png}`

Both figures are 183 mm wide. SVG text remains editable, PDF uses TrueType
font embedding, and PNG is exported at 300 dpi. In the case figure, `[C]` and
`[I]` denote hierarchy-consistent and hierarchy-inconsistent final
predictions. Full representative identifiers remain in the case-study table;
only display labels are abbreviated.

## Integrity boundary

The generator reads existing summary, frozen-test prediction, prototype
representative, and leakage-audit files. It does not read audio or checkpoints,
does not train a model, does not create prototypes, does not calibrate a
parameter, and does not introduce a rejection threshold. A pre/post SHA-256
snapshot of the 240 frozen source and cross-check files is required to remain
identical. This boundary comprises 200 metric/prediction/representative inputs,
25 prototype metadata files, and all 15 train/validation/test manifests. The
canonical predictions must equal the test-manifest identity triples, both
representative files must equal the train-manifest identity triples, and all
three canonical splits must be disjoint independently by path, source ID, and
MD5.

One locked cohort member, fold 0 seed 3407, retains
`single_fold_debug=true` in both its prototype metadata and evaluation metrics,
while both also mark it `eligible_for_cv_aggregation=true`. The complete
predefined 25-run grid is retained and this marker is exposed in
`cumulative_framework_runs.csv`; it is not silently treated as an exclusion.
