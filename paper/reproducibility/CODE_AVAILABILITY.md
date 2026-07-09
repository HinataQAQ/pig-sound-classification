# Code Availability

This code availability statement was prepared from the current repository
metadata, the manuscript draft, and the reproducibility command package.

## Submission Draft

The analysis code for this manuscript is maintained in the GitHub repository
`HinataQAQ/pig-sound-classification`:

https://github.com/HinataQAQ/pig-sound-classification

The current working branch used to prepare this reproducibility package is
`paper/sci-draft-v1`. A final archival release tag and software DOI are
currently UNRESOLVED and should be created before submission if the target
journal requires a citable software record.

The main scripts associated with the reported clean and simulated-noise results
include:

- `tools/train_mctafd_crnn_ablation.py`
- `tools/train_hier_longcontext_crnn.py`
- `tools/summarize_cv5_expanded_variants.py`
- `tools/summarize_hier_longcontext.py`
- `tools/summarize_hier_longcontext_confusion_by_weight.py`
- `tools/paired_cv5_cap3x_logmel_dur2_vs_dur1.py`
- `tools/paired_cv5_cap3x_hier_vs_dur2_by_weight.py`
- `tools/summarize_prototype_noise_robustness.py`
- `tools/create_paper_package.py`

The paper-facing reproduction commands are documented in
`paper/reproducibility/commands.md`. Those commands rebuild the corrected
selective summaries and paper package from existing outputs only. They do not
perform model retraining, new inference, new SNR generation, new noise draws, or
threshold changes.

The code should be run on Windows PowerShell in the `pigsound-gpu` conda
environment with user-site packages disabled:

```powershell
conda activate pigsound-gpu
cd C:\py\pigsound\pig-sound-classification
$env:PYTHONNOUSERSITE="1"
```

Model checkpoints and prototype artifacts are derived computational artifacts.
The reviewed paper package includes local prototype NPZ artifacts under
`reports/prototype_cv5_exact_w05_fold*/artifacts/`, but no final public
checkpoint/prototype archive DOI is currently available. If checkpoints or
prototype bundles are released, the archive should include fold, seed, model
configuration, SHA256 checksums, and the exact relation to the corresponding
summary and prediction files.

## Repository Actions Before Submission

- Create a release tag for the exact submission version.
- Archive the release in Zenodo, Figshare, OSF, or another durable repository if
  the journal requires a DOI.
- Add a software license file if not already present.
- Add a README section mapping scripts to paper tables, figures, and appendix
  files.
- Decide whether checkpoints and prototype artifacts will be archived or
  regenerated from released code and manifests.
- Keep raw clean pig audio and DEMAND audio out of the code repository unless
  redistribution rights are explicitly confirmed.

## Unresolved Code Fields

- UNRESOLVED: final code release tag.
- UNRESOLVED: archived software DOI.
- UNRESOLVED: software license confirmation.
- UNRESOLVED: checkpoint archive DOI and license, if checkpoints are released.
- UNRESOLVED: prototype artifact archive DOI and license, if NPZ artifacts are
  released.
