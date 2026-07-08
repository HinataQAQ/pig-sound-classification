# Paper Figures and Tables

This folder contains reproducible paper figure and Word-friendly table outputs
generated from the approved paper summary CSV files. The generator does not run
model training, model inference, calibration, threshold tuning, or statistical
recomputation.

## Command

Run from the repository root:

```powershell
$env:PYTHONNOUSERSITE="1"
$env:PYTHONIOENCODING="utf-8"
conda run -n pigsound-gpu python tools\generate_paper_figures_and_tables.py --dpi 300
```

Use `--help` for options.

## Figure Outputs

Each figure is written to `paper/figures/` in three formats:

- `.svg`
- `.pdf`
- `.png` at 300 dpi by default

`paper/figures/figure_data_sources.csv` and
`paper/figures/figure_data_sources.json` list the source files used for each
figure.

## Word-Friendly Table Outputs

`paper/tables/word_friendly/` contains:

- one consolidated `paper_main_tables.docx`;
- one consolidated `paper_main_tables.html`;
- one HTML file per main table;
- `table_sources.csv`, mapping each exported table to its source CSV.

The Word-friendly table values are copied as text from the source CSV files in
`paper/tables/`; the source result CSV values are not modified.
