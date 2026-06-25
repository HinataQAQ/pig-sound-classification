# Reproducibility Commands

These commands regenerate the corrected selective summaries and the paper package from existing outputs only.

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

No model inference, retraining, new SNRs, new noise draws, or threshold changes are performed by these commands.
