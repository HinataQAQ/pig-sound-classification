# Data Availability

This statement was prepared from the manuscript draft, the paper table and
appendix CSV files, the DEMAND provenance records, the noise-source decision
note, and the reproducibility package manifest. It is suitable as a submission
draft. Clean pig-audio source citations have been updated from the 2026-07-09
source-recovery audit. The current package does not redistribute raw audio; any
future archival DOI or provider-authorized redistribution would require separate
confirmation before submission.

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
`abdominal_cough` files map to the Ministry of Agriculture, Food and Rural
Affairs data.go.kr item titled
`농림축산식품부_양돈 기침 데이터 셋_20230727`, available at
https://www.data.go.kr/data/15117368/fileData.do. The item has no DOI. Provider
terms apply, and raw Korean cough audio should be obtained from the original
provider rather than redistributed with this repository. The `calm_grunt`,
`feeding`, `frightened_stress`, and `anxious_stress` files derive from Jie
Liao's Sow Call Dataset, version 2 (2022-10-28), on figshare, DOI
`10.6084/m9.figshare.16940389`, under CC BY 4.0. Local subtype labels for the Sow
Call Dataset are documented as filename-suffix mappings in the source-recovery
audit.

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
diverse environments. DEMAND version 1.0 (2013-06-09) is available from Zenodo
at https://zenodo.org/records/1227121 with DOI
`10.5281/zenodo.1227121`; the record licence text is CC BY-SA 3.0 Unported. This
study used the selected DEMAND environments `DWASHING`, `TBUS`, and `STRAFFIC`,
fixed to channel 1, with local provenance recorded in
`data/noise_sources/demand/PROVENANCE.json` and
`data/noise_sources/demand/NOISE_SOURCE_MANIFEST.csv`. The repository should
cite the DEMAND DOI and Zenodo record and should not redistribute DEMAND audio;
readers should obtain it directly from Zenodo under the record licence.

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

- Thiemann, J., Ito, N., and Vincent, E. (2013). DEMAND: a collection of
  multi-channel recordings of acoustic noise in diverse environments, version
  1.0. Zenodo. https://doi.org/10.5281/zenodo.1227121
- Liao, J. (2022). Sow call dataset, version 2. figshare.
  https://doi.org/10.6084/m9.figshare.16940389
- Ministry of Agriculture, Food and Rural Affairs. (2023).
  농림축산식품부_양돈 기침 데이터 셋_20230727. data.go.kr.
  https://www.data.go.kr/data/15117368/fileData.do

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

- CONFIRMED FOR CITATION: Sow Call Dataset author Jie Liao, version 2 date
  2022-10-28, DOI `10.6084/m9.figshare.16940389`, figshare URL, and CC BY 4.0
  licence.
- CONFIRMED FOR CITATION: Smart Farm Korea / data.go.kr pig-cough dataset exact
  title, institutional provider, stable item URL, WAV format, and dry/abdominal
  cough description. No DOI is listed. Provider terms apply, and raw-audio
  redistribution is not claimed here.
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
- CONFIRMED: DEMAND authors, version 1.0 date 2013-06-09, DOI
  `10.5281/zenodo.1227121`, Zenodo record
  `https://zenodo.org/records/1227121`, and record licence text CC BY-SA 3.0
  Unported. DEMAND audio is not redistributed in GitHub.
