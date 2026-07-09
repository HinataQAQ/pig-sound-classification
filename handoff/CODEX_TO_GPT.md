# Codex to GPT Handoff - Nature Figure V2 Generation

## Status

Completed. The task was approved by `handoff/GPT_TO_CODEX.md`
(`APPROVED: true`). This round used the installed `nature-figure` skill,
loaded the required always-load files and Python/matplotlib backend fragment,
and generated a manuscript figure package under `paper/figures_v2/`.

No model training, model inference, checkpoint/prototype/calibration
construction, SNR/noise draw change, threshold tuning, raw audio edit, or
source result CSV/JSON modification was performed.

## Branch

- Branch: `paper/sci-draft-v1`
- Base commit: `d481d98`
- Commit SHA: see final Codex response after commit/push

## Changed Files

- Added `tools/generate_paper_figures_v2.py`
- Added `paper/figures_v2/figure_contracts.md`
- Added `paper/figures_v2/figure1_overall_workflow.{svg,pdf,png}`
- Added `paper/figures_v2/figure2_duration_ablation.{svg,pdf,png}`
- Added `paper/figures_v2/figure3_hierarchical_supervision.{svg,pdf,png}`
- Added `paper/figures_v2/figure4_clean_prototype_comparison.{svg,pdf,png}`
- Added `paper/figures_v2/figure5_simulated_noise_robustness.{svg,pdf,png}`
- Added `paper/figures_v2/figure6_active_event_snr_curve.{svg,pdf,png}`
- Added `paper/figures_v2/figure7_per_class_failure.{svg,pdf,png}`
- Added `paper/figures_v2/figure8_selective_prediction.{svg,pdf,png}`
- Updated `handoff/CODEX_TO_GPT.md`
- Updated `handoff/CODEX_TO_GPT.json`
- Updated `handoff/HISTORY.md`

## Commands

```powershell
Get-Content -Raw C:/Users/s1205/.codex/skills/nature-figure/SKILL.md
Get-Content -Raw C:/Users/s1205/.codex/skills/nature-figure/manifest.yaml
Get-Content -Raw C:/Users/s1205/.codex/skills/nature-figure/static/core/contract.md
Get-Content -Raw C:/Users/s1205/.codex/skills/nature-figure/static/core/stance.md
python C:/Users/s1205/.codex/skills/nature-figure/scripts/nature_figure_backend.py get
python C:/Users/s1205/.codex/skills/nature-figure/scripts/nature_figure_backend.py set python
Get-Content -Raw C:/Users/s1205/.codex/skills/nature-figure/static/fragments/backend/python.md
Get-Content -Raw handoff/GPT_TO_CODEX.md
rg --files paper/tables paper/appendix reports/prototype_noise_demand_w05_final
Get-Content -Raw paper/CLAIMS_AND_EVIDENCE.md
Get-Content -TotalCount 20 paper/tables/table1_dataset_and_leakage_free_protocol.csv
Get-Content -TotalCount 20 paper/tables/table2_clean_baseline_and_architecture_ablations.csv
Get-Content -TotalCount 20 paper/tables/table3_duration_comparison.csv
Get-Content -TotalCount 20 paper/tables/table4_hierarchical_aux_weight_ablation.csv
Get-Content -TotalCount 20 paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv
Get-Content -TotalCount 20 paper/tables/table6_simulated_noise_robustness_by_stratum.csv
Get-Content -TotalCount 20 paper/tables/table7_paired_statistics.csv
Get-Content -TotalCount 20 paper/tables/table8_per_class_noise_results_and_confusion_directions.csv
Get-Content -TotalCount 30 paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv
Get-Content -TotalCount 40 paper/appendix/noise_25_run_summary.csv
Get-Content -TotalCount 40 paper/appendix/noise_25_run_per_class_summary.csv
Get-Content -TotalCount 20 reports/prototype_noise_demand_w05_final/final_summary.csv
Get-Content -TotalCount 20 reports/prototype_noise_demand_w05_final/final_selective_summary.csv
Get-Content -TotalCount 20 reports/prototype_noise_demand_w05_final/final_per_class_summary.csv
Get-Content -TotalCount 20 reports/prototype_noise_demand_w05_final/final_paired_stats.csv
Get-Content -TotalCount 20 reports/prototype_noise_demand_w05_final/final_aurc_augrc_summary.csv
conda run -n pigsound-gpu python tools/generate_paper_figures_v2.py --help
conda run -n pigsound-gpu python tools/generate_paper_figures_v2.py --root . --out-dir paper/figures_v2
conda run -n pigsound-gpu python -c "from pathlib import Path; from PIL import Image; root=Path('paper/figures_v2'); [print(p.name+': size='+str(Image.open(p).size)+', extrema='+str(Image.open(p).convert('RGB').getextrema())) for p in sorted(root.glob('figure*.png'))]"
```

## Figure Contracts

The detailed per-figure contracts are written to
`paper/figures_v2/figure_contracts.md`.

- Figure 1: workflow and evidence boundary; simulated DEMAND noise is not
  real-farm external validation.
- Figure 2: duration ablation; 2 s is the clean mainline, not a first-use
  claim.
- Figure 3: hierarchical auxiliary supervision; lambda=0.5 is stable and
  lambda=1.0 has the highest mean, without claiming significant hierarchy gain.
- Figure 4: clean prototype comparison; hierarchical prototype has the highest
  clean Macro-F1, but prototypes do not improve every calibration metric.
- Figure 5: simulated-noise robustness; main prototype is more stable in
  MODERATE_NOISE and ALL_NOISY, without claiming hierarchical noise optimality.
- Figure 6: active-event SNR curve; 0 dB is an extreme simulated-noise stress
  condition, not no noise.
- Figure 7: per-class failure; cough is the main noisy-condition risk and
  feeding/stress confusion directions remain visible.
- Figure 8: selective prediction; prototypes do not comprehensively improve
  uncertainty, and raw Softmax ranks better by AURC/AUGRC.

## Source CSVs Used by the Generator

- `paper/tables/table1_dataset_and_leakage_free_protocol.csv`
- `paper/tables/table3_duration_comparison.csv`
- `paper/tables/table4_hierarchical_aux_weight_ablation.csv`
- `paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv`
- `paper/tables/table6_simulated_noise_robustness_by_stratum.csv`
- `paper/tables/table7_paired_statistics.csv`
- `paper/tables/table8_per_class_noise_results_and_confusion_directions.csv`
- `paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv`
- `reports/prototype_noise_demand_w05_final/final_summary.csv`
- `reports/prototype_noise_demand_w05_final/final_per_class_summary.csv`

The script also reads `paper/CLAIMS_AND_EVIDENCE.md` for non-quantitative
claim-boundary wording.

Additional allowed CSVs were inspected during planning but not used in final
plots: `paper/tables/table2_clean_baseline_and_architecture_ablations.csv`,
`paper/appendix/noise_25_run_summary.csv`,
`paper/appendix/noise_25_run_per_class_summary.csv`,
`reports/prototype_noise_demand_w05_final/final_selective_summary.csv`,
`reports/prototype_noise_demand_w05_final/final_paired_stats.csv`, and
`reports/prototype_noise_demand_w05_final/final_aurc_augrc_summary.csv`.

## Core Metrics Represented

- Duration: 1 s/2 s/3 s Macro-F1 `0.9226/0.9472/0.9434`; 2 s - 1 s
  mean delta `0.0246`, Wilcoxon p `0.000162`.
- Hierarchical auxiliary weights: lambda=0.2/0.5/1.0 Macro-F1
  `0.9505/0.9510/0.9515`; lambda=0.5 std `0.0124`.
- Clean prototype comparison: raw/calibrated/main-prototype/hierarchical/fused
  Macro-F1 `0.9510/0.9510/0.9519/0.9540/0.9535`; ECE, Brier, and NLL are
  shown from Table 5 without recomputation.
- Simulated-noise strata: MODERATE_NOISE raw/prototype/hierarchical Macro-F1
  `0.6472/0.6534/0.6522`; EXTREME_STRESS `0.5574/0.5624/0.5556`; ALL_NOISY
  `0.6173/0.6230/0.6200`.
- Paired noise statistics: prototype - raw delta `0.006129`, p `0.001673` for
  MODERATE_NOISE; prototype - raw delta `0.005755`, p `0.010511` for ALL_NOISY.
- Per-class failure: ALL_NOISY cough F1 raw/prototype/hierarchical
  `0.0867/0.0974/0.0957`; 0 dB cough recall range across DEMAND
  environments/methods `0.0-0.9%`.
- Selective prediction: ALL_NOISY AURC/AUGRC raw `0.1775/0.1239`,
  prototype `0.2109/0.1394`, hierarchical `0.2122/0.1403`; frozen-threshold
  selective risk raw/prototype `0.3024/0.2988`.

## Data Boundaries and Leakage Audit

- No training, validation, or test role was changed.
- No test-set tuning was performed.
- No model, prototype, calibration, threshold, SNR, noise draw, prediction, or
  checkpoint artifact was regenerated.
- No raw audio or DEMAND audio was read, modified, or added.
- No source result CSV/JSON file was modified.
- Path/source_id/MD5 disjointness is unchanged because this task writes only
  derived figure files, a figure-generation script, contracts, and handoff
  documents.

## Verification

- `tools/generate_paper_figures_v2.py --help` passed.
- Full generation passed and wrote SVG, PDF, and 300 dpi PNG for Figures 1-8.
- PNG dimension and nonblank pixel-extrema checks passed for all eight figures.
- PNG previews for all eight figures were visually inspected and layout issues
  found during QA were corrected before final export.
- The only runtime warning was pandas' future optional pyarrow dependency
  warning; it did not affect generation.

## Paper Usability

- `paper_usable`: true for a manuscript figure package grounded in existing
  locked result files.
- `new_results_created`: false.
- `source_csv_json_values_modified`: false.
- `simulated_noise_only`: true.
- `real_farm_external_validation`: false.
- `figures_generated`: 8.
- `blockers`: none for the requested figures.

## Questions for GPT Pro

- Decide whether the current figure typography and panel density should be
  tuned for a specific journal template after target venue selection.
- Decide whether final captions should repeat every boundary warning or defer
  some boundaries to the Results/Limitations text.
- Decide whether a separate source-data archive should be prepared for
  submission from the same locked CSV inputs.

## Recommended Next Step

Review the exported figure package and integrate the figure captions into the
manuscript. Do not start new experiments, model runs, or another paper stage
without a new explicit `APPROVED: true`.
