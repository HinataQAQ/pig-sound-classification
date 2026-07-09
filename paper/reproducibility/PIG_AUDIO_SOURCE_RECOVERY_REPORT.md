# Pig Audio Source Recovery Report

Date: 2026-07-09
Repository: `C:\py\pigsound\pig-sound-classification`
Branch observed: `paper/sci-draft-v1`

This audit only recovers clean pig-audio source provenance. It did not run model training, modify training/evaluation scripts, change paper results, or delete files.

## Executive Decision

Final recommendation: **safe_only_after_source_citation_added**.

Rationale: the two source families used by the current cap3x clean mainline were recovered with high confidence:

- `data/external/korea_raw/*` maps to the Korean Smart Farm Korea / data.go.kr pig-cough public dataset.
- `data/raw/sow_call_dataset_labeled/*` maps to the figshare **Sow call dataset** under CC BY 4.0.

Before submission, add formal source citations, source URLs/DOIs, license wording, and a clear third-party-data/ethics statement. Do not promote unresolved non-mainline local sources, especially the Kaggle scream/cough set, into final claims.

## Current Mainline Manifest Coverage

Current audited manifest root: `data/manifests_pigvocal_4class_expanded_train_cv5_cap3x/fold*/{train,val,test}.csv`.

- Total manifest rows across 5 folds and 3 splits: 7,310.
- Unique audio paths across those rows: 3,812.
- Per fold: train 1,174; validation 120; test 168.
- MD5 blank rows: 0.
- source_id blank rows: 0.
- Unique MD5 values: 3,812.
- Unique source_id values: 3,812.

Rows by local root across all fold/split CSVs:

| Local root | Rows |
|---|---:|
| `data/external/korea_raw/dry_cough` | 821 |
| `data/external/korea_raw/abdominal_cough` | 1,609 |
| `data/raw/sow_call_dataset_labeled/calm_grunt` | 1,400 |
| `data/raw/sow_call_dataset_labeled/feeding` | 1,050 |
| `data/raw/sow_call_dataset_labeled/frightened_stress` | 1,414 |
| `data/raw/sow_call_dataset_labeled/anxious_stress` | 1,016 |

Fold0 leakage check, read-only:

- `tools/check_manifest_hash_overlap.py`: no cross-split MD5 duplicate leakage.
- `tools/check_manifest_source_overlap.py`: train/val/test exact-path overlap 0 and source_id overlap 0.

## Verified Source Recoveries

### 1. Korea Pig Cough Source

Classification: **A. verified_public_reusable**, with final terms sanity check recommended before raw-audio redistribution.

Recovered source:

- Title/page: Smart Farm Korea pig cough sound unstructured dataset.
- URLs:
  - https://www.smartfarmkorea.net/unstructuredDataset/pigCoughSound.do?menuId=M11040204
  - https://www.data.go.kr/data/15117368/fileData.do
  - https://data.mafra.go.kr/opendata/data/indexOpenDataDetail.do?data_id=20230727000000002375
- Publisher/institution: Korean public agricultural/smart-farm data portals.
- License/terms found: data.go.kr lists `이용허락범위 제한 없음` (no usage-permission restriction). Smart Farm Korea terms still need a final check for redistribution workflow.
- Local evidence: `data/external/korea_raw/dry_cough/train_dry_*.wav` and `data/external/korea_raw/abdominal_cough/train_abdominal_*.wav`; source page lists dry and abdominal monthly ZIP files for pig cough WAV from six farms.
- Metadata match: 44.1 kHz mono WAV PCM16; short cough clips; dry/abdominal subtypes.
- Confidence: high.

Rights interpretation:

- Research use: yes.
- Publication: yes with formal citation.
- Raw audio redistribution: likely yes under data.go.kr no-restriction listing, but verify provider-specific portal terms before archiving raw audio.
- Manifest sharing: likely yes.
- Extracted feature sharing: likely yes.
- Derived label sharing: likely yes.
- Ethics/permission: no animal-ethics statement was found on the public source pages; describe as third-party public farm audio and ask the journal/editor whether an ethics-exemption statement is expected.

### 2. Sow Call Dataset

Classification: **A. verified_public_reusable**.

Recovered source:

- Title: Sow call dataset.
- URL: https://figshare.com/articles/dataset/Sow_call_dataset/16940389
- DOI: https://doi.org/10.6084/m9.figshare.16940389
- Publisher: figshare.
- License: CC BY 4.0.
- Local evidence: Edge download history records `Dataset.zip` / `Dataset (1).zip` from figshare file 31337746; local root `data/raw/sow_call_dataset` has 1,746 files named like `0-0.wav`, `1000-2.wav`; `sow_labeled_index.csv` maps suffix classes into local subtype folders.
- Metadata match: 48 kHz mono WAV DOUBLE; about 2 s.
- Confidence: high.

Local label mapping from `sow_labeled_index.csv`:

| Source suffix | Local subtype | Main class |
|---|---|---|
| `-0.wav` | `calm_grunt` | `calm_grunt` |
| `-1.wav` | `feeding` | `feeding` |
| `-2.wav` | `frightened_stress` | `stress_vocal` |
| `-3.wav` | `anxious_stress` | `stress_vocal` |

Rights interpretation:

- Research use: yes with attribution.
- Publication: yes with citation.
- Raw audio redistribution: yes under CC BY 4.0 with attribution and license notice.
- Manifest sharing: yes.
- Extracted feature sharing: yes.
- Derived label sharing: yes, if local label mapping is documented.
- Ethics/permission: no separate ethics statement found during this audit; cite the original public dataset and do not present it as internally collected data.

### 3. SoundWel

Classification: **A. verified_public_reusable**.

Recovered source:

- Title: The Soundwel Database: a labeled pig vocalization repository.
- URL: https://zenodo.org/records/8252482
- DOI: https://doi.org/10.5281/zenodo.8252482
- License: CC BY 4.0.
- Local evidence: Edge download history records `Soundwel Dataset - Audio and Spectrograms.zip` and `SoundwelDatasetKey.xlsx` from Zenodo 8252482; local files include SoundWel-style ETHZ/NMBU/IASP filenames and spectrogram sidecars.
- Local roots: `data/raw/soundwel`, `data/processed/soundwel`, `data/external_pig_other_reviewed`.
- Current mainline use: not used in current cap3x manifests.
- Confidence: high.

### 4. aSwine

Classification: **A. verified_public_reusable**, noncommercial restriction.

Recovered source:

- Title: aSwine Audio Dataset / Annotated Swine dataset.
- URL: https://github.com/andremsouza/aswine
- Paper DOI: https://doi.org/10.1007/s10489-025-06555-6
- License: CC BY-NC 4.0.
- Local evidence: `data/external/aswine/README.md`, `LICENSE`, `data/audio/*.wav`, and `meta/*/*.csv` match upstream repository structure.
- Current mainline use: not used in current cap3x manifests.
- Confidence: high.

## Unresolved Or High-Risk Sources

### Kaggle Porcine Scream/Cough

Classification: **E. unresolved_do_not_submit_as_final**.

- Local root: `data/raw/scream_cough_small`.
- Suspected source: Kaggle `titpigrecognition/porcine-scream-sounds-and-cough-sounds`.
- URL: https://www.kaggle.com/datasets/titpigrecognition/porcine-scream-sounds-and-cough-sounds
- Evidence: Edge download history records `archive.zip` from that Kaggle URL; local tree has `trainingdata/testdata` with `cough` and `screams` folders.
- Problem: Kaggle page JSON-LD reports license name `Unknown`; no owner permission or ethics statement was recovered.
- Action: exclude from final claims and do not redistribute, share features, or share derived labels publicly until license and owner rights are resolved.

### Empty Placeholder Roots

Classification: **E. unresolved_do_not_submit_as_final** if repopulated later; currently no local audio.

- `data/external/kaggle_porcine`
- `data/external/sow_call`
- `data/korea_subtype/raw/dry_cough`
- `data/korea_subtype/raw/abdominal_cough`

## Candidate Non-Matches

- AI Hub / aihub: searched locally and on the web; no high-confidence match to local `train_dry_*.wav` / `train_abdominal_*.wav` files was found.
- Generic pig-cough ScienceDirect articles: browser history contains literature pages, but these are papers, not recovered dataset download sources for the local files.

## Highest-Risk Dataset

- Highest risk overall: `data/raw/scream_cough_small` / Kaggle porcine scream/cough, because the source page license is `Unknown` and rights cannot be confirmed.
- Highest risk among current cap3x manifest sources: Korea pig cough, not because it appears closed, but because the current final model depends on it and raw-audio redistribution should be checked one more time against Smart Farm Korea/data.go.kr terms before archiving raw audio.

## Korea Cough Source Identification

Korea cough source identified: **yes, high confidence**.

Recovered source is the Smart Farm Korea / data.go.kr pig cough sound unstructured dataset. The source page lists dry and abdominal pig-cough WAV ZIP files by month; this matches the local `dry_cough` and `abdominal_cough` folders, filename prefixes, audio format, and subtype usage in current manifests.

## Files Produced

- `paper/reproducibility/PIG_AUDIO_SOURCE_RECOVERY_REPORT.md`
- `paper/reproducibility/PIG_AUDIO_SOURCE_AUDIT.csv`
- `paper/reproducibility/UNRESOLVED_DATA_SOURCES.md`

## Commands And Evidence Summary

Local evidence searched:

- Repository text and filenames with `rg` for Korea/Korean/AI Hub/pig cough/dry cough/wet cough/abdominal cough/respiratory/pig audio/sow call/pig vocal/figshare/zenodo/kaggle/openslr/github/aihub/Korean terms/license/terms/DOI/download/source/provenance.
- Raw and extracted audio directories under `data/raw`, `data/external`, `data/processed`, `data/external_pig_other_*`, and current cap3x manifests.
- Existing source audit file `paper/reproducibility/DATA_SOURCE_AUDIT.csv`.
- Manifest generation scripts, especially `tools/build_pigvocal_4class_expanded_train_cv5.py`, `tools/build_pigvocal_4class_dedup_cv5.py`, and `tools/build_no_feeding_pretrain_manifests.py`.
- PowerShell history.
- Downloads folder.
- Chrome/Edge history databases using keyword-only queries for relevant data-source URLs.
- Git history with source/provenance keywords and `-S` probes.

Web evidence searched:

- Smart Farm Korea / data.go.kr pig cough sound pages.
- figshare Sow call dataset page and DOI.
- Zenodo SoundWel record.
- Kaggle porcine scream/cough page metadata.
- GitHub aSwine repository.
- AI Hub candidate searches, with no high-confidence local match.
