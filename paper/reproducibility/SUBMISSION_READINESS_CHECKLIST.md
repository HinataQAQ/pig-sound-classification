# Submission Readiness Checklist

This checklist records submission metadata and availability decisions that still
need author confirmation before journal upload. It does not change the locked
experimental results or the manuscript's scientific conclusions.

## Remaining Required Items

- Target journal: select the journal and confirm its article-type, data-sharing,
  ethics, code-availability, reference-format, and supplementary-file rules.
- Code release URL/tag: create the public repository release or archived package
  URL to replace the manuscript code-release placeholder. Do not claim a code DOI
  unless one is actually minted.
- Korea provider terms final check: retain Smart Farm Korea / data.go.kr as
  `citation_partial` until the final provider terms are checked and recorded.
  Do not claim that Korean raw pig-cough audio can be redistributed.
- Corresponding author and affiliations: confirm author order, corresponding
  author details, institutional affiliations, and submission-system metadata.
- Funding: add grant numbers, funder names, and any required funder-role
  statement.
- Conflict of interest: add the final competing-interests declaration required
  by the target journal.
- Checkpoint/prototype artifact archiving: decide whether model checkpoints,
  prototype bundles, and derived artifact metadata will be archived for review
  or regenerated from code. If archived, record hashes, version, licence, and
  artifact-to-result mapping.

## Current Data-Source Status

- DEMAND: citation complete by DOI `10.5281/zenodo.1227121`; obtain raw noise
  recordings from Zenodo rather than redistributing them in the repository.
- Sow Call Dataset: citation complete by DOI `10.6084/m9.figshare.16940389`;
  obtain raw recordings from figshare.
- Smart Farm Korea / data.go.kr pig-cough recordings: citation partial; exact
  public pages are recorded, but raw-audio redistribution is not claimed.
- Kaggle-like scream/cough source: unresolved and excluded from the current main
  results and submitted data description.

## Text-Integrity Checks Required Before Upload

- No literal citation placeholders remain in the manuscript.
- The manuscript does not claim real-farm external validation.
- The manuscript does not claim that all raw pig audio is openly redistributed.
- The manuscript does not claim a code DOI or checkpoint archive until those
  resources exist.
