# Codex to GPT Handoff - Full-Story SCI Manuscript Rewrite

## Status

Completed. The task was approved by `handoff/GPT_TO_CODEX.md`
(`APPROVED: true`). This round used the installed `nature-writing` skill router
and fragments before drafting. It generated bilingual full-story manuscript
drafts and an updated claim-evidence ledger from existing paper result files.

No model training, model inference, checkpoint/prototype/calibration
construction, SNR/noise draw change, threshold tuning, raw audio edit, or
result CSV/JSON modification was performed.

## Branch

- Branch: `paper/sci-draft-v1`
- Base commit: `01853bf`
- Commit SHA: see final Codex response after commit/push

## Changed Files

- Added `paper/manuscript/paper_cn_full_story.md`
- Added `paper/manuscript/paper_en_full_story.md`
- Added `paper/manuscript/section_plan.md`
- Added `paper/CLAIMS_AND_EVIDENCE_v2.md`
- Updated `handoff/CODEX_TO_GPT.md`
- Updated `handoff/CODEX_TO_GPT.json`
- Updated `handoff/HISTORY.md`

## Commands

```powershell
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/SKILL.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/manifest.yaml'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/../_shared/core/reader-workflow.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/../_shared/core/paper-type-taxonomy.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/../_shared/core/ethics.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/../_shared/core/terminology-ledger.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/static/core/stance.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/static/core/workflow.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/static/core/output-format.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/static/fragments/paper_type/research.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/static/fragments/section/{title,abstract,intro,related-work,method,experiments,discussion,conclusion}.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/static/fragments/language/zh-to-en.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/static/fragments/journal/generic.md'
Get-Content -Raw 'C:/Users/s1205/.codex/skills/nature-writing/references/{article-architecture,experiments,method,related-work,conclusion,chinese-author-workflow,paper-review}.md'
git branch --show-current
git status --short --branch
Get-Content -Raw handoff/GPT_TO_CODEX.md
Get-Content -Raw paper/manuscript/paper_cn_draft.md
Get-Content -Raw paper/manuscript/paper_en_draft.md
Get-Content -Raw paper/CLAIMS_AND_EVIDENCE.md
Get-Content -Raw paper/references/references.bib
Get-Content -Raw docs/NOISE_ROBUSTNESS_PROTOCOL.md
Get-Content -Raw docs/NOISE_SOURCE_DECISION.md
Get-Content -Raw paper/tables/table1_dataset_and_leakage_free_protocol.csv
Get-Content -Raw paper/tables/table2_clean_baseline_and_architecture_ablations.csv
Get-Content -Raw paper/tables/table3_duration_comparison.csv
Get-Content -Raw paper/tables/table4_hierarchical_aux_weight_ablation.csv
Get-Content -Raw paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv
Get-Content -Raw paper/tables/table6_simulated_noise_robustness_by_stratum.csv
Get-Content -Raw paper/tables/table7_paired_statistics.csv
Get-Content -Raw paper/tables/table8_per_class_noise_results_and_confusion_directions.csv
Get-Content -Raw paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv
Get-Content -Raw paper/appendix/clean_25_run_prototype_summary.csv
Get-Content -Raw paper/appendix/clean_25_run_prototype_paired_stats.csv
Get-Content -Raw paper/appendix/noise_25_run_summary.csv
Get-Content -Raw paper/appendix/noise_25_run_paired_stats.csv
Get-Content -Raw paper/appendix/noise_25_run_aurc_augrc_summary.csv
Get-Content -Raw paper/appendix/noise_25_run_selective_summary.csv
Get-Content -Raw paper/appendix/noise_25_run_cluster_bootstrap.csv
Import-Csv paper/appendix/noise_25_run_per_class_summary.csv
Import-Csv paper/appendix/noise_25_run_cross_seed_draw_audit.csv
```

## Core Metrics Used

No metric values were recomputed or modified. The manuscript drafts cite only
existing paper CSV values.

- Duration:
  - 1 s Log-Mel Macro-F1 `0.9226`
  - 2 s Log-Mel Macro-F1 `0.9472`
  - 3 s Log-Mel Macro-F1 `0.9434`
  - 2 s - 1 s mean delta `0.0246`, Wilcoxon p `0.000162`
- Clean architecture/front-end ablations:
  - cap3x MCTAFD no-SE gated `0.9208`
  - PCEN `0.8916`
  - spectral-gate `0.9093`
  - SpecAugment `0.9198`
  - attention pooling `0.9431`
- Hierarchical auxiliary weights:
  - lambda=0.2 `0.9505`
  - lambda=0.5 `0.9510`, std `0.0124`
  - lambda=1.0 `0.9515`
- Clean prototype summary:
  - raw Softmax `0.9510`
  - main prototype `0.9519`
  - hierarchical prototype `0.9540`
  - hierarchical - raw Softmax p `0.058253`, nonsignificant
- Simulated DEMAND noise:
  - ALL_NOISY raw/prototype/hierarchical Macro-F1:
    `0.6173`, `0.6230`, `0.6200`
  - ALL_NOISY prototype - raw delta `0.005755`, CI95
    `[0.001606, 0.010894]`, Wilcoxon p `0.010511`
  - ALL_NOISY hierarchical - prototype delta `-0.003045`,
    Wilcoxon p `0.000376`
  - MODERATE_NOISE raw/prototype/hierarchical Macro-F1:
    `0.6472`, `0.6534`, `0.6522`
  - EXTREME_STRESS raw/prototype/hierarchical Macro-F1:
    `0.5574`, `0.5624`, `0.5556`
- Cough/noise boundary:
  - ALL_NOISY cough F1 raw/prototype/hierarchical:
    `0.0867`, `0.0974`, `0.0957`
  - 0 dB active-event SNR cough recall is zero for STRAFFIC and TBUS across
    all methods; DWASHING cough recall is below `0.009`
- Selective prediction:
  - ALL_NOISY raw Softmax AURC/AUGRC `0.1775/0.1239`
  - ALL_NOISY prototype AURC/AUGRC `0.2109/0.1394`
  - ALL_NOISY hierarchical AURC/AUGRC `0.2122/0.1403`
  - ALL_NOISY frozen-threshold selective risk raw/prototype:
    `0.3024/0.2988`

## Data Boundaries and Leakage Audit

- No training, validation, or test role was changed.
- No test-set tuning was performed.
- No model, prototype, calibration, threshold, SNR, noise draw, prediction, or
  checkpoint artifact was regenerated.
- No raw audio or DEMAND audio was modified or added.
- Source result CSV/JSON files under `paper/tables/` and `paper/appendix/`
  were read only.
- The manuscript repeats the locked boundary that train data build models and
  prototypes, validation data calibrate, and test data are evaluation only.
- Path/source_id/MD5 disjointness is unchanged because this task only writes
  manuscript Markdown and handoff documents.

## Paper Usability

- `paper_usable`: true for a full-story SCI application manuscript draft.
- `paper_main_result_changed`: false.
- `simulated_noise_only`: true.
- `real_farm_external_validation`: false.
- `new_results_created`: false.
- `source_csv_json_values_modified`: false.

## Blockers / Questions for GPT Pro

- Target journal and exact section/word-limit requirements are still missing.
- Complete pig-vocalization and livestock-acoustic references must be verified
  and added before submission.
- Original pig-audio ethics approval and data-sharing permissions are not
  present in the reviewed files.
- Code release URL, tag, archive DOI, checkpoint policy, and raw-data release
  policy remain to be decided.
- GPT Pro should decide how much of the bilingual draft should be condensed for
  the target journal once the venue is selected.

## Recommended Next Step

Review the full-story CN/EN drafts and fill the missing target-journal,
references, ethics, data-availability, and code-availability information. Do not
start new experiments, model runs, or paper stages without a new explicit
`APPROVED: true`.
