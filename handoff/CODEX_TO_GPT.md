# Codex to GPT Handoff - Nature Data Availability Package

## Status

Completed. The task was approved by `handoff/GPT_TO_CODEX.md`
(`APPROVED: true`). This round used the installed `nature-data` skill, loaded
`SKILL.md`, `manifest.yaml`, all `always_load` files, and the relevant
on-demand policy, repository, statement-pattern, FAIR, Chinese-author, and
source-basis references before drafting.

No model training, model inference, checkpoint/prototype/calibration
construction, SNR/noise draw change, threshold tuning, raw audio edit, or source
result CSV/JSON modification was performed.

## Branch

- Branch: `paper/sci-draft-v1`
- Base commit: `855c9d721d639e80bde3c2ad2bb6cbb57a8195d4`
- Commit SHA: see final Codex response after commit/push

## Changed Files

- Added `paper/reproducibility/DATA_AVAILABILITY.md`
- Added `paper/reproducibility/CODE_AVAILABILITY.md`
- Added `paper/reproducibility/ETHICS_STATEMENT.md`
- Added `paper/reproducibility/REPRODUCIBILITY_CHECKLIST.md`
- Added `paper/reproducibility/DATA_SOURCE_AUDIT.csv`
- Updated `handoff/CODEX_TO_GPT.md`
- Updated `handoff/CODEX_TO_GPT.json`
- Updated `handoff/HISTORY.md`

## Commands

```powershell
Get-Content -LiteralPath "$env:USERPROFILE\.codex\skills\nature-data\SKILL.md" -Raw
Get-Content -LiteralPath "$env:USERPROFILE\.codex\skills\nature-data\manifest.yaml" -Raw
Get-Content -LiteralPath "$env:USERPROFILE\.codex\skills\nature-data\static\core\stance.md" -Raw
Get-Content -LiteralPath "$env:USERPROFILE\.codex\skills\nature-data\static\core\chinese-mode.md" -Raw
Get-Content -LiteralPath "$env:USERPROFILE\.codex\skills\nature-data\static\core\workflow.md" -Raw
Get-Content -LiteralPath "$env:USERPROFILE\.codex\skills\nature-data\references\policy-principles.md" -Raw
Get-Content -LiteralPath "$env:USERPROFILE\.codex\skills\nature-data\references\repository-and-identifiers.md" -Raw
Get-Content -LiteralPath "$env:USERPROFILE\.codex\skills\nature-data\references\statement-patterns.md" -Raw
Get-Content -LiteralPath "$env:USERPROFILE\.codex\skills\nature-data\references\fair-metadata-checklist.md" -Raw
Get-Content -LiteralPath "$env:USERPROFILE\.codex\skills\nature-data\references\chinese-author-alignment.md" -Raw
Get-Content -LiteralPath "$env:USERPROFILE\.codex\skills\nature-data\references\source-basis.md" -Raw
Get-Content -LiteralPath "handoff\GPT_TO_CODEX.md" -Raw
git status --short --branch
rg --files paper/manuscript paper/tables paper/appendix paper/reproducibility data/noise_sources/demand docs handoff
Get-Content -LiteralPath "paper\manuscript\paper_en_full_story.md" -Raw
Get-Content -LiteralPath "data\noise_sources\demand\PROVENANCE.json" -Raw
Get-Content -LiteralPath "data\noise_sources\demand\NOISE_SOURCE_MANIFEST.csv" -Raw
Get-Content -LiteralPath "docs\NOISE_SOURCE_DECISION.md" -Raw
Get-Content -LiteralPath "paper\reproducibility\commands.md" -Raw
Get-Content -LiteralPath "paper\reproducibility\paper_package_manifest.json" -Raw
Get-ChildItem -LiteralPath "paper\tables" -Filter "*.csv"
Get-ChildItem -LiteralPath "paper\appendix" -Filter "*.csv"
Get-ChildItem -LiteralPath "paper\tables" -Filter "*.csv" | Sort-Object Name | ForEach-Object { "FILE: $($_.Name)"; Get-Content -LiteralPath $_.FullName; "" }
Get-ChildItem -LiteralPath "paper\appendix" -Filter "*.csv" | Sort-Object Name | ForEach-Object { $rows = (Import-Csv -LiteralPath $_.FullName).Count; $head = Get-Content -LiteralPath $_.FullName -TotalCount 3; "FILE: $($_.Name) ROWS: $rows"; $head; "" }
Get-Content -LiteralPath "paper\figures\figure_data_sources.csv" -Raw
Get-Content -LiteralPath "paper\figures\figure_data_sources.json" -Raw
Get-Content -LiteralPath "paper\references\references.bib" -Raw
Get-Content -LiteralPath "paper\appendix\noise_25_run_provenance.json" -Raw
Get-Content -LiteralPath "data\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\train.csv" -TotalCount 5
Get-ChildItem -LiteralPath "data\manifests_pigvocal_4class_expanded_train_cv5_cap3x" -Recurse -Filter "*.csv"
git branch --show-current
git remote -v
git rev-parse HEAD
Import-Csv -LiteralPath "paper\reproducibility\DATA_SOURCE_AUDIT.csv" | Measure-Object
rg -n "UNRESOLVED|DEMAND|10\.5281/zenodo\.1227121|real-farm external validation|This manuscript uses previously collected or public audio data|model checkpoints|prototype|SHA|md5" paper/reproducibility
$env:PYTHONNOUSERSITE='1'; conda run -n pigsound-gpu python tools\summarize_prototype_noise_robustness.py --help
$env:PYTHONNOUSERSITE='1'; conda run -n pigsound-gpu python tools\create_paper_package.py --help
$env:PYTHONNOUSERSITE='1'; $env:PYTHONIOENCODING='utf-8'; & 'C:\py\anaconda3\envs\pigsound-gpu\python.exe' tools\create_paper_package.py --help
$env:PYTHONNOUSERSITE='1'; $env:PYTHONIOENCODING='utf-8'; & 'C:\py\anaconda3\envs\pigsound-gpu\python.exe' tools\summarize_prototype_noise_robustness.py --help
git restore -- paper/figures/figure1_overall_method_architecture.pdf paper/figures/figure1_overall_method_architecture.png paper/figures/figure1_overall_method_architecture.svg paper/figures/figure2_duration_macro_f1.pdf paper/figures/figure2_duration_macro_f1.png paper/figures/figure2_duration_macro_f1.svg paper/figures/figure3_prototype_structure.pdf paper/figures/figure3_prototype_structure.png paper/figures/figure3_prototype_structure.svg paper/figures/figure4_macro_f1_vs_snr.pdf paper/figures/figure4_macro_f1_vs_snr.png paper/figures/figure4_macro_f1_vs_snr.svg paper/figures/figure5_noise_degradation_by_environment.pdf paper/figures/figure5_noise_degradation_by_environment.png paper/figures/figure5_noise_degradation_by_environment.svg paper/figures/figure6_feeding_stress_confusion.pdf paper/figures/figure6_feeding_stress_confusion.png paper/figures/figure6_feeding_stress_confusion.svg paper/figures/figure7_cough_collapse.pdf paper/figures/figure7_cough_collapse.png paper/figures/figure7_cough_collapse.svg paper/figures/figure8_risk_coverage_curves.pdf paper/figures/figure8_risk_coverage_curves.png paper/figures/figure8_risk_coverage_curves.svg
git status --short -- paper/reproducibility paper/figures
```

Notes:

- The first parallel `conda run` attempt for `create_paper_package.py --help`
  hit a conda temporary-file collision. A later conda attempt hit a conda GBK
  `UnicodeEncodeError`.
- Direct `pigsound-gpu` Python invocation succeeded. `create_paper_package.py`
  does not implement conventional argparse help and executed package generation
  instead; it briefly rewrote tracked `paper/figures/*` outputs. Those
  accidental tracked figure changes were restored to HEAD. Final status shows no
  remaining figure, table, appendix, command, or package-manifest diff.

## Data Source Audit Results

`paper/reproducibility/DATA_SOURCE_AUDIT.csv` contains 13 data-source rows:

- `clean_pig_vocalization_audio`
- `auxiliary_subtype_labels`
- `cv_manifests_cap3x`
- `model_checkpoints`
- `prototype_artifacts`
- `demand_noise_audio`
- `demand_noise_metadata`
- `generated_tables`
- `generated_appendix_csv`
- `generated_figures`
- `code_repository`
- `non_redistributed_audio_policy`
- `sha_provenance`

## Officially Citable Sources Now

- DEMAND environmental noise dataset:
  DOI `10.5281/zenodo.1227121`, Zenodo record
  `https://zenodo.org/records/1227121`.
- Code repository URL:
  `https://github.com/HinataQAQ/pig-sound-classification`.

The code repository is citable as a URL in draft form, but a release tag and
archival software DOI are still unresolved.

## Unresolved DOI / License Fields

- Clean pig-vocalization dataset DOI or stable URL.
- Clean pig-vocalization dataset license and redistribution rights.
- Clean pig-vocalization data owner and access-review route.
- Auxiliary subtype label ownership and metadata license.
- Processed source-data archive DOI and license.
- Code release tag, archived software DOI, and software license confirmation.
- Model checkpoint archive DOI and license, if checkpoints are released.
- Prototype artifact archive DOI and license, if NPZ bundles are released.

## Ethics and Authorization Blockers

The ethics statement includes the exact required fallback sentence:

`This manuscript uses previously collected or public audio data; the original collection ethics and permissions must be confirmed before submission.`

Fields that must be completed before submission:

- Original clean audio source name and data owner.
- Whether the clean pig-vocalization recordings were public, previously
  collected by the authors, collected by collaborators, or licensed from a third
  party.
- Animal ethics approval authority and approval number, if applicable.
- Farm, institution, or data-owner permission statement, if applicable.
- Confirmation that reuse of clean audio and auxiliary subtype labels is
  permitted for this manuscript.
- Confirmation of whether raw clean audio, metadata, and subtype labels may be
  redistributed, deposited under controlled access, or described only as
  restricted third-party material.
- Any data-use agreement, confidentiality term, farm privacy condition, or
  commercial restriction that affects audio redistribution.

## Metrics and Scientific Boundaries Preserved

- 2 s Log-Mel clean mean Macro-F1 `0.9472`; 1 s Log-Mel `0.9226`; paired delta
  `0.0246`; Wilcoxon p `0.000162`.
- Hierarchical lambda=1.0 highest mean Macro-F1 `0.9515`; lambda=0.5 balanced
  setting with lowest std `0.0124`.
- Hierarchical gain over the 2 s baseline is not claimed as statistically
  significant.
- Clean hierarchical prototype mean Macro-F1 `0.9540`, but clean
  hierarchical - raw Softmax p `0.058253` remains nonsignificant.
- ALL_NOISY raw/prototype/hierarchical Macro-F1 `0.6173/0.6230/0.6200`;
  prototype - raw delta `0.005755`, CI95 `[0.001606, 0.010894]`, p
  `0.010511`.
- DEMAND is simulated additive noise only, not real-farm external validation.
- 0 dB active-event SNR is an extreme stress condition, not no noise.
- Prototype inference is not described as a universal uncertainty improvement.

## Data Boundaries and Leakage Audit

- No training, validation, or test role was changed.
- No test-set tuning was performed.
- No new model, prototype, calibration, threshold, SNR, noise draw, prediction,
  or checkpoint artifact was created.
- No raw clean audio or DEMAND audio was added or modified.
- Final diff does not modify `paper/tables/*.csv`, `paper/appendix/*.csv`,
  `paper/figures/*`, `paper/reproducibility/commands.md`, or
  `paper/reproducibility/paper_package_manifest.json`.
- The new statements explicitly keep raw clean pig audio and DEMAND audio out of
  GitHub until rights are confirmed.
- Path/source_id/MD5 disjointness is unchanged because this task only writes
  paper documentation and an audit CSV.

## Verification

- `DATA_SOURCE_AUDIT.csv` parses with PowerShell `Import-Csv` and contains 13
  data-source rows.
- Keyword audit found the required unresolved fields, DEMAND DOI, real-farm
  external-validation warning, exact ethics fallback sentence, checkpoint,
  prototype, SHA, and MD5 language.
- All five generated files have zero non-ASCII characters.
- `tools/summarize_prototype_noise_robustness.py --help` passed under
  `pigsound-gpu`.
- `tools/create_paper_package.py --help` is not a true help command; direct
  execution produced the package manifest summary and confirmed
  `real_farm_external_validation=false`, `simulated_noise_only=true`, and
  `augrc_corrected=true`.
- Accidental figure regeneration from the package script was restored; final
  status shows only the intended reproducibility and handoff files as changes.

## Paper Usability

- `paper_usable`: true for the requested availability, code, ethics,
  reproducibility, and data-source audit package.
- `new_results_created`: false.
- `source_csv_json_values_modified`: false.
- `simulated_noise_only`: true.
- `real_farm_external_validation`: false.

## Blockers

- Clean pig audio DOI/URL/license/owner/access route unresolved.
- Animal ethics and original collection permission unresolved.
- Code release tag/software DOI/license unresolved.
- Checkpoint/prototype artifact release plan unresolved.
- DEMAND license metadata conflict should be retained in the final statement.

## Questions for GPT Pro

- Decide the final data-access route for clean pig audio: public repository,
  controlled access, third-party restricted, or institutional request route.
- Provide the exact animal ethics approval authority/number or confirm that the
  target journal accepts a previously collected/public-data statement.
- Decide whether to archive checkpoints and prototype NPZ artifacts or make code
  regeneration the official route.
- Decide the final repository for processed source data, figures, manifests, and
  code release DOI.

## Recommended Next Step

Resolve clean-audio rights, ethics, and code/source-data archival DOI fields.
Do not start new experiments or another paper stage without a new explicit
`APPROVED: true`.
