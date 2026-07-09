# Source Citation Status

Date: 2026-07-09

Scope: citation and statement synchronization from
`paper/reproducibility/PIG_AUDIO_SOURCE_RECOVERY_REPORT.md` and
`paper/reproducibility/PIG_AUDIO_SOURCE_AUDIT.csv` into the paper submission
package. This file does not change model results or paper conclusions.

## Status Definitions

- `citation_complete`: exact DOI or stable source URL, source identity, and
  licence/terms are sufficient for citation in the current manuscript package.
- `citation_partial`: source is identified and citable, but at least one
  submission-relevant rights detail still needs final author/provider
  confirmation.
- `unresolved_do_not_submit_until_fixed`: source must not support final paper
  claims until provenance, licence, and permission blockers are resolved.
- `excluded_from_main_results`: source is locally present or audited, but is not
  part of the current cap3x clean mainline or final result claims.

## Source Status Table

| Source | Local roots | Current use | Status | Citation key | Rights/terms basis | Required handling |
|---|---|---|---|---|---|---|
| DEMAND environmental noise | `data/noise_sources/demand/*` | Simulated additive-noise evaluation | `citation_complete` | `demand2018` | Zenodo DOI `10.5281/zenodo.1227121`; local audit notes conflicting licence metadata, so raw audio is not redistributed | Cite DOI; use source links, checksums, and provenance rather than bundling audio |
| Sow Call Dataset | `data/raw/sow_call_dataset`; `data/raw/sow_call_dataset_labeled/*`; duplicate `data/raw/wu_pig_speech` | Current clean cap3x manifests for `calm_grunt`, `feeding`, `frightened_stress`, `anxious_stress` | `citation_complete` | `sow_call_dataset_2021` | figshare DOI `10.6084/m9.figshare.16940389`; CC BY 4.0 in audit | Cite DOI; document local filename-suffix label mapping; raw audio may be obtained from figshare |
| Smart Farm Korea / data.go.kr pig-cough voice dataset | `data/external/korea_raw/dry_cough`; `data/external/korea_raw/abdominal_cough` | Current clean cap3x manifests for `dry_cough` and `abdominal_cough` | `citation_partial` | `smartfarmkorea_pig_cough_voice` | Exact data.go.kr item and Smart Farm Korea source URL recovered; audit records data.go.kr usage-permission range as no restriction | Cite URL; do not redistribute raw Korean cough audio in this package; complete final provider-terms check before any raw-audio archive |
| Kaggle-like porcine scream/cough dataset | `data/raw/scream_cough_small` | Not used in current cap3x manifests | `unresolved_do_not_submit_until_fixed` | none | Suspected Kaggle URL recovered, but licence is `Unknown` and owner rights/ethics were not recovered | Exclude from final claims; do not cite as verified; do not share raw audio, features, or derived labels until resolved |
| SoundWel | `data/raw/soundwel`; `data/processed/soundwel`; `data/external_pig_other_reviewed` | Not used in current clean cap3x mainline | `excluded_from_main_results` | none added in this task | Zenodo DOI `10.5281/zenodo.8252482`; CC BY 4.0 in audit | Cite only if SoundWel-derived results are promoted later |
| aSwine | `data/external/aswine`; `data/processed/cough_binary*` | Not used in current clean cap3x mainline | `excluded_from_main_results` | none added in this task | GitHub source and paper DOI recovered; CC BY-NC 4.0 in audit | Cite only if aSwine-derived ablations or results are promoted later; preserve noncommercial restriction |
| Empty placeholder roots | `data/external/kaggle_porcine`; `data/external/sow_call`; `data/korea_subtype/raw/*` | No local audio in audit | `excluded_from_main_results` | none | No audio files present | Reaudit before use if repopulated |

## Manuscript Submission Interpretation

The current manuscript package is safer than the previous draft because the
main clean sources are now named and cited, and raw third-party pig audio is not
claimed as redistributed. The package should still avoid saying that all clean
raw audio can be openly redistributed. Korea cough data should be cited as a
public source with a final provider-terms check pending for any raw-audio
archive. The Kaggle-like source remains unresolved and excluded from all current
main results.
