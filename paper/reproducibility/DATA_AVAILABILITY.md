# Data Availability

This statement was prepared from the manuscript draft, the paper table and
appendix CSV files, the DEMAND provenance records, the noise-source decision
note, and the reproducibility package manifest. It is suitable as a submission
draft. Clean pig-audio source citations have been updated from the 2026-07-09
source-recovery audit; release DOIs and raw-audio redistribution decisions still
need final author confirmation before submission.

## Submission Draft

The processed data supporting the aggregate results in this manuscript are
provided in the project repository under `paper/tables/` and `paper/appendix/`.
These files include the clean 25-run summaries, paired statistics, calibration
distribution summaries, simulated DEMAND noise summaries, per-class summaries,
selective prediction summaries, AURC/AUGRC summaries, cluster-bootstrap
summaries, fold-level deltas, and the cross-seed noise-draw audit. Source-data
mapping for generated figures is provided in
`paper/figures/figure_data_sources.csv` and
`paper/figures/figure_data_sources.json`. The package manifest is provided in
`paper/reproducibility/paper_package_manifest.json`.

The cross-validation manifests used for the clean closed-set experiments are
stored under
`data/manifests_pigvocal_4class_expanded_train_cv5_cap3x/`. These manifests
contain `path`, `label`, `subtype`, `source_id`, and `md5` fields and define the
four main classes (`cough`, `calm_grunt`, `feeding`, `stress_vocal`) and the six
auxiliary subtypes (`dry_cough`, `abdominal_cough`, `calm_grunt`, `feeding`,
`frightened_stress`, `anxious_stress`). They are the metadata needed to
reconstruct the fold-specific train, validation, and test splits, subject to the
availability of the underlying audio.

The original clean pig-vocalization recordings are third-party or public-source
audio and are not redistributed in this repository. The current clean cap3x
manifests draw from two recovered source families. The `dry_cough` and
`abdominal_cough` files map to the Smart Farm Korea / data.go.kr pig-cough voice
dataset, with the data.go.kr item at
https://www.data.go.kr/data/15117368/fileData.do and the Smart Farm Korea source
page at
https://www.smartfarmkorea.net/unstructuredDataset/pigCoughSound.do?menuId=M11040204.
The local source-recovery audit records the data.go.kr usage-permission range
as no restriction, but raw Korean cough audio should still be obtained from the
original provider portal rather than redistributed with this repository unless
the authors complete a final provider-terms check. The `calm_grunt`, `feeding`,
`frightened_stress`, and `anxious_stress` files derive from the Sow Call Dataset
on figshare, DOI `10.6084/m9.figshare.16940389`, under CC BY 4.0. Local subtype
labels for the Sow Call Dataset are documented as filename-suffix mappings in
the source-recovery audit.

Raw third-party pig audio is therefore outside the public repository package.
Shareable materials include code, commands, aggregate tables, figures, source
data for figures, cross-validation manifests, hash/source-id leakage audits,
and provenance records. Manifests, extracted features, and derived labels may be
shared only as permitted by the corresponding source licence or terms and with
source attribution. Restricted or provider-controlled raw audio should be
obtained from the original providers. The unresolved Kaggle-like
`data/raw/scream_cough_small` source is excluded from the main results and must
not be used as final submission evidence until its licence and owner rights are
resolved.

The environmental noise data used for simulated additive-noise evaluation came
from DEMAND: a collection of multi-channel recordings of acoustic noise in
diverse environments. DEMAND is available from Zenodo at
https://zenodo.org/records/1227121 with DOI `10.5281/zenodo.1227121`. This study
used the selected DEMAND environments `DWASHING`, `TBUS`, and `STRAFFIC`, fixed
to channel 1, with local provenance recorded in
`data/noise_sources/demand/PROVENANCE.json` and
`data/noise_sources/demand/NOISE_SOURCE_MANIFEST.csv`. The repository should
cite the DEMAND DOI and Zenodo record and should not redistribute DEMAND audio.
The local provenance records conflicting license metadata:
`description_license=CC-BY-SA-3.0` and `zenodo_rights_license=cc-by-4.0`.
Because of this conflict, strict redistribution should use links, checksums, and
provenance rather than bundling DEMAND audio in GitHub.

Model checkpoints are model artifacts rather than source data. The reviewed
paper package does not provide a final public checkpoint archive or checkpoint
DOI. If the target journal or reviewers require checkpoint-level inspection, the
fold-seed checkpoints should be archived in a durable repository with file
hashes and clear mapping to the corresponding fold, seed, model configuration,
and `test_pred.csv` output. Until such an archive is created, checkpoint
availability remains UNRESOLVED.

Post-hoc prototype artifacts are derived model artifacts built from train-split
embeddings and validation-only calibration. Local prototype bundles and
prototype metadata are stored under fold-seed-specific directories such as
`reports/prototype_cv5_exact_w05_fold*/artifacts/`, including
`main_prototypes.npz`, `aux_prototypes.npz`, and `prototype_bundle.npz`.
Publication of these derived artifacts should use an archived release with
hashes and a README explaining that prototypes were built from the training
split only, calibrated on validation data only, and evaluated on the frozen test
split. Public archival DOI and license for these artifacts are currently
UNRESOLVED.

The simulated-noise results are not real-farm external validation. DEMAND mixing
uses deterministic additive environmental-noise segments at active-event SNRs of
20, 10, and 0 dB. The 0 dB setting is an extreme active-event SNR stress
condition and should not be described as no noise or as a normal deployment
environment. No real-farm external validation dataset is reported in the current
paper package.

## Formal Dataset Citations

- Thiemann, J., Ito, N., and Vincent, E. (2018). DEMAND: a collection of
  multi-channel recordings of acoustic noise in diverse environments. Zenodo.
  https://doi.org/10.5281/zenodo.1227121
- Sow call dataset. (2021). figshare.
  https://doi.org/10.6084/m9.figshare.16940389
- Smart Farm Korea and data.go.kr. Smart Farm Korea pig-cough voice dataset.
  https://www.data.go.kr/data/15117368/fileData.do and
  https://www.smartfarmkorea.net/unstructuredDataset/pigCoughSound.do?menuId=M11040204

## Repository and Citation Actions Before Submission

- Cite DEMAND, the Sow Call Dataset, and the Smart Farm Korea / data.go.kr
  pig-cough voice dataset in the manuscript and reference list.
- Do not redistribute raw third-party pig audio or DEMAND audio in GitHub; use
  source links, checksums, manifests, and provenance instead.
- Before public raw-audio archiving, complete the final Smart Farm Korea /
  data.go.kr provider-terms check and record the decision.
- Confirm whether auxiliary subtype labels and extracted features will be
  redistributed as metadata and under which source-specific licence or data-use
  condition.
- Deposit shareable processed results, figure source data, CV manifests, and
  derived artifacts in a durable repository with DOI, version, license, README,
  and checksums.
- Decide whether fold-seed model checkpoints and prototype NPZ artifacts will be
  archived for reviewers or regenerated from code.
- Keep DEMAND audio out of GitHub and cite the Zenodo DOI plus local checksums.
- Ensure the final manuscript and any archive do not imply real-farm external
  validation from simulated DEMAND noise.

## Missing Information and Risk Flags

- CONFIRMED FOR CITATION: Sow Call Dataset DOI
  `10.6084/m9.figshare.16940389`, figshare URL, and CC BY 4.0 licence.
- CONFIRMED FOR CITATION WITH CAUTION: Smart Farm Korea / data.go.kr pig-cough
  voice dataset exact URLs and data.go.kr no-restriction usage-permission range;
  raw-audio redistribution is not claimed here and still needs a final
  provider-terms check before any public raw-audio archive.
- UNRESOLVED: licence, owner rights, and ethics/permission status for the
  Kaggle-like `data/raw/scream_cough_small` source; it is excluded from current
  main results.
- UNRESOLVED: animal ethics approval authority, approval number, and permission
  conditions for any original source collection, if the target journal requires
  those details for third-party animal audio reuse.
- UNRESOLVED: archival DOI or release tag for the code and processed source-data
  package.
- UNRESOLVED: archival DOI and license for model checkpoints and prototype
  artifacts, if these are to be released.
- CONFIRMED: DEMAND DOI `10.5281/zenodo.1227121` and Zenodo record
  `https://zenodo.org/records/1227121`.
- CONFIRMED WITH CAUTION: DEMAND local provenance and checksums are recorded,
  but license metadata are conflicting and DEMAND audio should not be
  redistributed in GitHub.
