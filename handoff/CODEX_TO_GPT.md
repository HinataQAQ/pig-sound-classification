# Codex to GPT Handoff - Paper Figures and Tables V2

## Status

Completed. The task was approved by `handoff/GPT_TO_CODEX.md`
(`APPROVED: true`) and by the external task file
`C:\Users\s1205\OneDrive\Desktop\论文\codex_paper_figures_and_tables_tasks.md`.

This round generated reproducible paper figures and Word-friendly table files
from existing paper summary CSVs. No new model training, model inference,
checkpoint/prototype/calibration construction, SNR/noise draw change, or
statistical protocol change was started.

## Branch

- Branch: `paper/sci-draft-v1`
- Base commit: `a015ac2`
- Commit SHA: see final Codex response after commit/push

## Changed Files

- Added `tools/generate_paper_figures_and_tables.py`
  - Reproducible argparse generator.
  - Validates required source files.
  - Writes all eight figures as SVG, PDF, and 300-dpi PNG.
  - Writes Word-friendly table outputs without adding new dependencies.
- Added `paper/README_figures_tables.md`
  - Documents the generation command and output fields.
- Regenerated:
  - `paper/figures/figure1_overall_method_architecture.{svg,pdf,png}`
  - `paper/figures/figure2_duration_macro_f1.{svg,pdf,png}`
  - `paper/figures/figure3_prototype_structure.{svg,pdf,png}`
  - `paper/figures/figure4_macro_f1_vs_snr.{svg,pdf,png}`
  - `paper/figures/figure5_noise_degradation_by_environment.{svg,pdf,png}`
  - `paper/figures/figure6_feeding_stress_confusion.{svg,pdf,png}`
  - `paper/figures/figure7_cough_collapse.{svg,pdf,png}`
  - `paper/figures/figure8_risk_coverage_curves.{svg,pdf,png}`
- Added source manifests:
  - `paper/figures/figure_data_sources.csv`
  - `paper/figures/figure_data_sources.json`
- Added Word-friendly tables:
  - `paper/tables/word_friendly/paper_main_tables.docx`
  - `paper/tables/word_friendly/paper_main_tables.html`
  - `paper/tables/word_friendly/table_1_dataset_and_leakage_free_protocol.html`
  - `paper/tables/word_friendly/table_2_clean_baseline_and_architecture_ablations.html`
  - `paper/tables/word_friendly/table_3_duration_comparison.html`
  - `paper/tables/word_friendly/table_4_hierarchical_auxiliary_weight_ablation.html`
  - `paper/tables/word_friendly/table_5_clean_softmax_prototype_hierarchical_prototype_results.html`
  - `paper/tables/word_friendly/table_6_simulated_noise_robustness_by_stratum.html`
  - `paper/tables/word_friendly/table_7_paired_statistics.html`
  - `paper/tables/word_friendly/table_8_per_class_noise_results_and_confusion_directions.html`
  - `paper/tables/word_friendly/table_9_selective_prediction_metrics_after_augrc_correction.html`
  - `paper/tables/word_friendly/table_sources.csv`

## Commands

```powershell
Get-Content -Raw -Encoding UTF8 handoff\GPT_TO_CODEX.md
Get-Content -Raw -Encoding UTF8 'C:\Users\s1205\OneDrive\Desktop\论文\codex_paper_figures_and_tables_tasks.md'
git branch --show-current
git status --short --branch
rg --files paper tools handoff docs

$env:PYTHONNOUSERSITE="1"
$env:PYTHONIOENCODING="utf-8"
conda run -n pigsound-gpu python tools\generate_paper_figures_and_tables.py --help
conda run -n pigsound-gpu python -m py_compile tools\generate_paper_figures_and_tables.py
conda run -n pigsound-gpu python tools\generate_paper_figures_and_tables.py --dpi 300

python -c "import zipfile, xml.etree.ElementTree as ET; p='paper/tables/word_friendly/paper_main_tables.docx'; z=zipfile.ZipFile(p); print('\n'.join(z.namelist())); ET.fromstring(z.read('word/document.xml')); print('document.xml parse ok')"
rg -n "0 dB active-event SNR|simulated additive noise|main-class prototype|hierarchical prototype|Cough collapse under simulated active-event noise|Prototype AURC/AUGRC is not better|ALL_NOISY|MODERATE_NOISE|EXTREME_STRESS" paper\figures paper\tables\word_friendly
git diff -- paper\tables\*.csv paper\appendix\*.csv
Get-ChildItem paper\figures -File
Get-ChildItem paper\tables\word_friendly -File
```

Visual spot checks were performed on all eight generated PNG figures in Codex.

Note: PowerShell `conda activate pigsound-gpu` hit a local Windows
encoding/temp-file issue in this shell. `conda run -n pigsound-gpu` succeeded
and was used for the verified commands.

## Figure Data Sources

- Figure 1: `tools/train_hier_longcontext_crnn.py`;
  `tools/prototype_model_adapter.py`;
  `paper/tables/table1_dataset_and_leakage_free_protocol.csv`
- Figure 2: `paper/tables/table3_duration_comparison.csv`
- Figure 3: `tools/prototype_model_adapter.py`
- Figure 4: `paper/appendix/noise_25_run_summary.csv`
- Figure 5: `paper/appendix/noise_25_run_summary.csv`
- Figure 6: `paper/tables/table8_per_class_noise_results_and_confusion_directions.csv`;
  `paper/appendix/noise_25_run_summary.csv`
- Figure 7: `paper/tables/table8_per_class_noise_results_and_confusion_directions.csv`;
  `paper/appendix/noise_25_run_summary.csv`
- Figure 8: `paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv`

## Core Metrics Represented

No metric values were recomputed or modified. The generated plots and tables use
the existing paper CSV values.

- Duration comparison:
  - 1 s Macro-F1 `0.9226`
  - 2 s Macro-F1 `0.9472`
  - 3 s Macro-F1 `0.9434`
  - 2 s - 1 s mean delta `0.0246`, Wilcoxon p `0.000162`
- Clean hierarchy/prototype:
  - lambda 1.0 clean hierarchy mean Macro-F1 `0.9515`
  - lambda 0.5 clean hierarchy mean Macro-F1 `0.9510`
  - clean hierarchical prototype Macro-F1 `0.9540`
- Simulated DEMAND noise:
  - ALL_NOISY raw/prototype/hierarchical Macro-F1:
    `0.6173`, `0.6230`, `0.6200`
  - MODERATE_NOISE raw/prototype/hierarchical Macro-F1:
    `0.6472`, `0.6534`, `0.6522`
  - EXTREME_STRESS raw/prototype/hierarchical Macro-F1:
    `0.5574`, `0.5624`, `0.5556`
- Corrected selective prediction summary for ALL_NOISY:
  - raw_softmax AURC/AUGRC `0.1775 / 0.1239`
  - prototype AURC/AUGRC `0.2109 / 0.1394`
  - hierarchical AURC/AUGRC `0.2122 / 0.1403`
- Cough collapse under ALL_NOISY:
  - cough F1 raw/prototype/hierarchical: `0.0867`, `0.0974`, `0.0957`
  - cough -> stress_vocal counts raw/prototype/hierarchical:
    `8935`, `8748`, `8780`
- Feeding/stress confusion under ALL_NOISY:
  - feeding -> stress_vocal raw/prototype/hierarchical:
    `2168`, `2006`, `2240`
  - stress_vocal -> feeding raw/prototype/hierarchical:
    `616`, `606`, `495`

## Data Boundaries and Leakage Audit

- No training data, validation data, or test data role was changed.
- No test-set tuning was performed.
- No model training, model inference, calibration, prototype construction, or
  threshold selection was run.
- Source result CSVs under `paper/tables/*.csv` and `paper/appendix/*.csv`
  have no diff after generation.
- Path/source_id/MD5 disjointness is unchanged because this task reads only
  existing aggregate paper files and source-code mappings.
- No audio, DEMAND archives, checkpoints, embeddings, NPZ files, caches, or
  prediction CSV files were added.
- Word-friendly table values are copied as text from the source CSV files.

## Verification

- `--help` passed for the new generator.
- `py_compile` passed for `tools/generate_paper_figures_and_tables.py`.
- Full generation completed with `--dpi 300`.
- DOCX package opened as a zip and `word/document.xml` parsed successfully.
- Required labels were found in generated SVG/HTML outputs:
  - `simulated additive noise`
  - `0 dB active-event SNR / extreme simulated-noise stress condition`
  - `main-class prototype`
  - `hierarchical prototype`
  - `Cough collapse under simulated active-event noise`
  - `Prototype AURC/AUGRC is not better than raw Softmax`
- Visual inspection of all eight PNGs found no blocking label overlaps after
  layout fixes.

## Paper Usability

- `paper_usable`: true for figure/table package.
- `paper_main_result`: unchanged; this task does not create new scientific
  results.
- `real_farm_external_validation`: false.
- `simulated_noise_only`: true.
- Values modified: false.

## Blockers / Questions for GPT Pro

- No code or data blocker remains for figure/table generation.
- GPT Pro should decide whether the target journal prefers the DOCX table file,
  the HTML table files, or manual journal-template table formatting.
- GPT Pro should review final captions and decide whether Figure 1 needs a
  journal-specific visual style pass.

## Recommended Next Step

Stop here and review the generated paper figures/tables and captions. Do not
start a new experiment or paper stage without a new explicit `APPROVED: true`.
