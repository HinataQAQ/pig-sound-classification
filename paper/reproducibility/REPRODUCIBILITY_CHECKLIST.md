# Reproducibility Checklist

This checklist summarizes the reproducibility state of the current paper
package. It is based on the reviewed manuscript, paper tables, appendix files,
DEMAND provenance records, and reproduction commands.

## Environment

- Operating system: Windows.
- Shell: PowerShell.
- Conda environment: `pigsound-gpu`.
- Required session setup:

```powershell
conda activate pigsound-gpu
cd C:\py\pigsound\pig-sound-classification
$env:PYTHONNOUSERSITE="1"
```

## Paper Package Commands

The documented package commands are in `paper/reproducibility/commands.md`.
They regenerate the corrected selective summaries and paper package from
existing outputs only:

```powershell
$env:PYTHONNOUSERSITE="1"
$metrics = @()
foreach ($fold in 0..4) {
  foreach ($seed in @(42,123,777,2024,3407)) {
    $stage = if ($seed -in @(123,777)) { "final" } else { "screening" }
    $metrics += "reports/prototype_noise_demand_w05_fold${fold}_seed${seed}_${stage}/noise_metrics.json"
  }
}
python tools\summarize_prototype_noise_robustness.py --metrics_json $metrics --protocol final --out_dir reports\prototype_noise_demand_w05_final --file_prefix final --recompute_selective_from_predictions
python tools\create_paper_package.py
```

These commands do not retrain models, rerun inference, introduce new SNRs,
change noise draws, or tune thresholds.

## Data and Split Controls

- Main classes: `cough`, `calm_grunt`, `feeding`, `stress_vocal`.
- Auxiliary subtypes: `dry_cough`, `abdominal_cough`, `calm_grunt`, `feeding`,
  `frightened_stress`, `anxious_stress`.
- Main clean protocol: cap3x training expansion, 2-second Log-Mel input, CRNN,
  5 folds x 5 seeds.
- CV manifests: `data/manifests_pigvocal_4class_expanded_train_cv5_cap3x/`.
- Manifest columns include `path`, `label`, `subtype`, `source_id`, and `md5`.
- Fold row counts in the reviewed local manifests: each fold has 1174 train,
  120 validation, and 168 test rows.
- Train data may build model parameters and prototypes.
- Validation data may calibrate thresholds, temperatures, fusion weights, and
  rejection criteria.
- Test data are evaluation only.
- No artifact should mix folds or seeds.
- Comparisons should use matched fold-seed pairs.
- Metrics should use `zero_division=0` for classification metrics.

## Leakage and Provenance Controls

- Required split disjointness: exact path, `source_id`, and MD5.
- Table 1 records path/source_id/MD5 disjointness, manifest SHA checks, and
  checkpoint/prototype/calibration SHA provenance.
- Noise segment selection uses deterministic SHA256-based keys.
- The noise key includes fold, normalized clean path, clean MD5, DEMAND
  environment identity, noise-file SHA256, global noise seed, and repeat.
- The noise key excludes model seed and SNR, so the same clean sample and
  DEMAND environment reuse the same noise segment across seeds and SNR levels,
  with only gain changing across SNR.
- `paper/appendix/noise_25_run_cross_seed_draw_audit.csv` contains 2520 audit
  rows and records `ok=True` in the sampled reviewed rows.
- `paper/appendix/noise_25_run_provenance.json` records
  `cross_seed_noise_draw_audit_ok=true`,
  `selective_metrics_recomputed_from_prediction_csv=true`, and
  `real_farm_external_validation=false`.

## DEMAND Noise Data

- Public dataset: DEMAND.
- DOI: `10.5281/zenodo.1227121`.
- Zenodo record: `https://zenodo.org/records/1227121`.
- Local download date recorded in provenance: `2026-06-25`.
- Selected environments: `DWASHING`, `TBUS`, `STRAFFIC`.
- Selected channel: channel 1.
- Active-event SNR levels: 20, 10, and 0 dB.
- Global noise seed: 3407.
- Noise repeat: 0.
- DEMAND audio should not be redistributed in GitHub.
- License metadata are conflicting in local provenance:
  `description_license=CC-BY-SA-3.0` and `zenodo_rights_license=cc-by-4.0`.
- Simulated DEMAND noise is not real-farm external validation.

## Paper Tables, Appendix Files, and Figures

- Main tables: `paper/tables/table1_dataset_and_leakage_free_protocol.csv`
  through `paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv`.
- Appendix CSVs: `paper/appendix/clean_25_run_prototype_summary.csv`,
  `paper/appendix/clean_25_run_prototype_paired_stats.csv`,
  `paper/appendix/clean_calibration_distribution.csv`, and
  `paper/appendix/noise_25_run_*.csv`.
- Figure source mapping:
  `paper/figures/figure_data_sources.csv` and
  `paper/figures/figure_data_sources.json`.
- Paper package manifest:
  `paper/reproducibility/paper_package_manifest.json`.

## Current Core Metrics to Preserve

- 1 s Log-Mel mean Macro-F1: 0.9226.
- 2 s Log-Mel mean Macro-F1: 0.9472.
- 2 s minus 1 s paired mean delta: 0.0246.
- 2 s versus 1 s Wilcoxon p-value: 0.000162.
- Hierarchical lambda=1.0 mean Macro-F1: 0.9515.
- Hierarchical lambda=0.5 mean Macro-F1: 0.9510.
- Hierarchical lambda=0.5 standard deviation: 0.0124.
- Clean hierarchical prototype mean Macro-F1: 0.9540.
- Clean hierarchical prototype minus raw Softmax Wilcoxon p-value: 0.058253,
  not statistically significant.
- ALL_NOISY raw Softmax, main prototype, hierarchical prototype Macro-F1:
  0.6173, 0.6230, 0.6200.
- ALL_NOISY main prototype minus raw Softmax mean delta: 0.005755.
- ALL_NOISY main prototype minus raw Softmax 95% bootstrap CI:
  [0.001606, 0.010894].
- ALL_NOISY main prototype minus raw Softmax Wilcoxon p-value: 0.010511.
- ALL_NOISY raw Softmax has lower AURC/AUGRC than prototype variants; prototype
  inference should not be described as a universal uncertainty improvement.
- 0 dB active-event SNR is an extreme simulated-noise stress condition, and
  cough recognition is unreliable there.

## Submission Blockers

- Clean pig-vocalization DOI or source URL is UNRESOLVED.
- Clean pig-vocalization license and redistribution permission are UNRESOLVED.
- Animal ethics approval authority and approval number are UNRESOLVED.
- Farm/data-owner permission and data-use conditions are UNRESOLVED.
- Code release tag, archived software DOI, and software license confirmation
  are UNRESOLVED.
- Checkpoint and prototype artifact release plan is UNRESOLVED.
- The manuscript still needs final target-journal formatting and complete
  livestock-acoustics citations.

## Final Pre-submission Checks

- Confirm that all public identifiers resolve.
- Confirm that raw audio redistribution is either permitted or replaced by a
  clear restricted-access route.
- Confirm that no test-set tuning, noisy validation recalibration, or fold/seed
  mixing has been introduced.
- Confirm that DEMAND audio is cited but not redistributed in GitHub.
- Confirm that every table and figure source file in the manuscript has a
  corresponding source-data entry.
- Confirm that the text states simulated DEMAND additive noise is not real-farm
  external validation.
