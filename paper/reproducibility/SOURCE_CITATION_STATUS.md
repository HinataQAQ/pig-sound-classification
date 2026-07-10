# Source Citation Status

Date: 2026-07-10

Scope: citation and statement synchronization from
`paper/reproducibility/PIG_AUDIO_SOURCE_RECOVERY_REPORT.md` and
`paper/reproducibility/PIG_AUDIO_SOURCE_AUDIT.csv` into the paper submission
package. This file does not change model results or paper conclusions.

## Citation Status Definitions

- `citation_complete`: exact DOI or stable source URL, source identity, and
  licence/terms are sufficient for citation in the current manuscript package.
- `citation_partial`: source is identified and citable, but at least one
  submission-relevant rights detail still needs final author/provider
  confirmation.
- `unresolved_do_not_submit_until_fixed`: source must not support final paper
  claims until provenance, licence, and permission blockers are resolved.
- `excluded_from_main_results`: source is locally present or audited, but is not
  part of the current cap3x clean mainline or final result claims.

## Redistribution Status Definitions

- `raw_audio_not_redistributed_source_terms_apply`: the source has a citable
  repository record, but this project package does not redistribute its raw
  audio; readers obtain it from the original repository under the source terms.
- `raw_audio_not_redistributed_provider_terms_apply`: the source is citable,
  but raw-audio access and reuse remain governed by the original provider; this
  project package does not redistribute the recordings.
- `do_not_redistribute_rights_unresolved`: source rights are unresolved, so no
  raw audio or source-derived artifacts should be redistributed.
- `not_in_current_submission_package`: the audited source is outside the current
  main results and no source files are included in the submission package.
- `no_local_audio_present`: the audited path contains no local audio to share.

## Source Status Table

| Source | Local roots | Current use | Citation status | Redistribution status | Citation key | Citation/terms basis | Required handling |
|---|---|---|---|---|---|---|---|
| DEMAND environmental noise | `data/noise_sources/demand/*` | Simulated additive-noise evaluation | `citation_complete` | `raw_audio_not_redistributed_source_terms_apply` | `demand2018` | Zenodo DOI `10.5281/zenodo.1227121`; version 1.0, 2013-06-09; record licence text CC BY-SA 3.0 Unported | Cite DOI; obtain audio from Zenodo; retain source links, checksums, and provenance |
| Sow Call Dataset | `data/raw/sow_call_dataset`; `data/raw/sow_call_dataset_labeled/*`; duplicate `data/raw/wu_pig_speech` | Current clean cap3x manifests for `calm_grunt`, `feeding`, `frightened_stress`, `anxious_stress` | `citation_complete` | `raw_audio_not_redistributed_source_terms_apply` | `sow_call_dataset_2021` | Jie Liao; figshare DOI `10.6084/m9.figshare.16940389`; version 2, 2022-10-28; CC BY 4.0 | Cite DOI; document local filename-suffix label mapping; obtain raw audio from figshare |
| Smart Farm Korea / data.go.kr pig-cough voice dataset | `data/external/korea_raw/dry_cough`; `data/external/korea_raw/abdominal_cough` | Current clean cap3x manifests for `dry_cough` and `abdominal_cough` | `citation_complete` | `raw_audio_not_redistributed_provider_terms_apply` | `smartfarmkorea_pig_cough_voice` | Exact data.go.kr item, Korean title, institutional provider, WAV format, and dry/abdominal cough description verified; no DOI; provider terms apply | Cite the stable data.go.kr URL; do not redistribute raw Korean cough audio; readers obtain it from the original provider |
| Kaggle-like porcine scream/cough dataset | `data/raw/scream_cough_small` | Not used in current cap3x manifests | `unresolved_do_not_submit_until_fixed` | `do_not_redistribute_rights_unresolved` | none | Suspected Kaggle URL recovered, but licence is `Unknown` and owner rights/ethics were not recovered | Exclude from final claims; do not cite as verified; do not share raw audio, features, or derived labels until resolved |
| SoundWel | `data/raw/soundwel`; `data/processed/soundwel`; `data/external_pig_other_reviewed` | Not used in current clean cap3x mainline | `excluded_from_main_results` | `not_in_current_submission_package` | none added in this task | Zenodo DOI `10.5281/zenodo.8252482`; CC BY 4.0 in audit | Cite and reassess redistribution only if SoundWel-derived results are promoted later |
| aSwine | `data/external/aswine`; `data/processed/cough_binary*` | Not used in current clean cap3x mainline | `excluded_from_main_results` | `not_in_current_submission_package` | none added in this task | GitHub source and paper DOI recovered; CC BY-NC 4.0 in audit | Cite only if aSwine-derived ablations or results are promoted later; preserve noncommercial restriction |
| Empty placeholder roots | `data/external/kaggle_porcine`; `data/external/sow_call`; `data/korea_subtype/raw/*` | No local audio in audit | `excluded_from_main_results` | `no_local_audio_present` | none | No audio files present | Reaudit before use if repopulated |

## Manuscript Submission Interpretation

The main clean sources are named and citable, while raw third-party pig audio is
not claimed as redistributed. Citation completeness does not imply raw-audio
redistribution permission. The Korea cough citation is complete at the metadata
level, but provider terms still govern access and reuse, and no raw Korean cough
audio is included in this project package. The Kaggle-like source remains
unresolved and excluded from all current main results. Landing-page metadata is
recorded in `paper/reproducibility/source_snapshots/SOURCE_SNAPSHOT_INDEX.md`.
