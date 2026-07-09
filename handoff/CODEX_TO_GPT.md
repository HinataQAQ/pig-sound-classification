# Codex to GPT Handoff - Pig Audio Source Recovery Audit

## Status

Completed. The task was directly requested by the user and `handoff/GPT_TO_CODEX.md` contained `APPROVED: true`.

Scope was limited to clean pig-audio data-source provenance recovery. No model experiment, model inference, checkpoint/prototype/calibration construction, threshold tuning, paper claim edit, paper-result edit, raw-audio edit, or file deletion was performed.

Firecrawl was requested by the relevant skill but the `firecrawl` CLI was not installed in PATH and the referenced `firecrawl-search` skill directory was unavailable, so source recovery used local evidence plus built-in web search and direct page metadata reads.

## Branch

- Branch: `paper/sci-draft-v1`
- Commit SHA: see final Codex response after commit/push

## Changed Files

- Added `paper/reproducibility/PIG_AUDIO_SOURCE_RECOVERY_REPORT.md`
- Added `paper/reproducibility/PIG_AUDIO_SOURCE_AUDIT.csv`
- Added `paper/reproducibility/UNRESOLVED_DATA_SOURCES.md`
- Updated `handoff/CODEX_TO_GPT.md`
- Updated `handoff/CODEX_TO_GPT.json`
- Updated `handoff/HISTORY.md`

## Commands and Searches

```powershell
Get-Content -Raw C:\Users\s1205\.codex\plugins\cache\openai-curated-remote\superpowers\6.1.1\skills\using-superpowers\SKILL.md
Get-Content -Raw C:\Users\s1205\.codex\skills\firecrawl\SKILL.md
Get-Content -Raw handoff\GPT_TO_CODEX.md
firecrawl --status
Get-ChildItem -Force data
Get-ChildItem -Recurse -Directory data | Select-Object -First 200 FullName
rg --files data | rg -i "manifest|csv|json|readme|license|source|note|download|zip|wav|mp3|flac"
rg -n -i --hidden -g '!**/.git/**' -g '!**/*.wav' -g '!reports/**' "korea|korean|AI Hub|aihub|pig cough|dry cough|wet cough|abdominal cough|respiratory|pig audio|sow call|pig vocal|figshare|zenodo|kaggle|openslr|github|??|??|??|??|??|???|license|terms|doi|download|source|provenance" .
Get-ChildItem -Recurse -File data -Include *.zip,*.tar,*.tar.gz,*.tgz,*.rar,*.7z,*.gz
Select-String -Path (Get-PSReadLineOption).HistorySavePath -Pattern "korea|AI Hub|pig cough|dry cough|abdominal cough|sow call|figshare|zenodo|kaggle|openslr|wget|curl|download" -CaseSensitive:$false
Get-ChildItem -Path $env:USERPROFILE\Downloads -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "korea|aihub|pig|sow|cough|swine|porcine|figshare|zenodo|kaggle|openslr" }
git log --all --name-only --pretty=format:'COMMIT %H %ad %s' --date=short -- data tools docs handoff paper README* LICENSE*
Get-Content -Raw paper\reproducibility\DATA_SOURCE_AUDIT.csv
Get-Content -Raw tools\build_pigvocal_4class_expanded_train_cv5.py
Get-Content -Raw tools\build_pigvocal_4class_dedup_cv5.py
Get-Content -Raw data\external\aswine\README.md
Get-Content -TotalCount 80 data\external\aswine\LICENSE
$env:PYTHONNOUSERSITE='1'; @'<audio metadata summary script using soundfile.info>'@ | C:\py\anaconda3\envs\pigsound-gpu\python.exe -
$env:PYTHONNOUSERSITE='1'; @'<Edge/Chrome history targeted source-query script>'@ | C:\py\anaconda3\envs\pigsound-gpu\python.exe -
$env:PYTHONNOUSERSITE='1'; C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\check_manifest_hash_overlap.py --train data\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\train.csv --val data\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\val.csv --test data\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\test.csv
$env:PYTHONNOUSERSITE='1'; C:\py\anaconda3\envs\pigsound-gpu\python.exe tools\check_manifest_source_overlap.py --train data\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\train.csv --val data\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\val.csv --test data\manifests_pigvocal_4class_expanded_train_cv5_cap3x\fold0\test.csv
```

Web/source pages checked:

- Smart Farm Korea pig cough page: `https://www.smartfarmkorea.net/unstructuredDataset/pigCoughSound.do?menuId=M11040204`
- data.go.kr pig cough listing: `https://www.data.go.kr/data/15117368/fileData.do`
- MAFRA data listing: `https://data.mafra.go.kr/opendata/data/indexOpenDataDetail.do?data_id=20230727000000002375`
- figshare Sow call dataset: `https://figshare.com/articles/dataset/Sow_call_dataset/16940389`
- DOI: `https://doi.org/10.6084/m9.figshare.16940389`
- Zenodo SoundWel: `https://zenodo.org/records/8252482`
- DOI: `https://doi.org/10.5281/zenodo.8252482`
- Kaggle porcine scream/cough: `https://www.kaggle.com/datasets/titpigrecognition/porcine-scream-sounds-and-cough-sounds`
- aSwine GitHub: `https://github.com/andremsouza/aswine`

## Source Recovery Results

Verified current-mainline sources:

- `data/external/korea_raw/dry_cough`: Smart Farm Korea / data.go.kr pig cough sound, high confidence, classification `A. verified_public_reusable` pending final terms sanity check before raw-audio redistribution.
- `data/external/korea_raw/abdominal_cough`: same Korean source, high confidence, classification `A. verified_public_reusable` pending final terms sanity check.
- `data/raw/sow_call_dataset_labeled/calm_grunt`, `feeding`, `frightened_stress`, `anxious_stress`: derived labeled copies of figshare Sow call dataset, high confidence, CC BY 4.0, classification `A. verified_public_reusable`.

Verified non-current or ablation sources:

- `data/raw/sow_call_dataset` and `data/raw/wu_pig_speech`: same figshare Sow call dataset; `wu_pig_speech` is an exact local duplicate of `sow_call_dataset` by 1,746 filenames and contents.
- `data/raw/soundwel`, `data/processed/soundwel`, and `data/external_pig_other_reviewed`: SoundWel Zenodo record 8252482, CC BY 4.0, high confidence.
- `data/external/aswine` and aSwine-derived processed cough roots: aSwine GitHub dataset, CC BY-NC 4.0, high confidence.

Unresolved/high-risk:

- `data/raw/scream_cough_small`: likely Kaggle `titpigrecognition/porcine-scream-sounds-and-cough-sounds`; Edge download record supports the match, but Kaggle JSON-LD says license `Unknown`. Classification `E. unresolved_do_not_submit_as_final`.
- Empty placeholders: `data/external/kaggle_porcine`, `data/external/sow_call`, `data/korea_subtype/raw/*` contained no audio in this audit.

Korea cough source was identified: **yes, high confidence**.

## Metrics and Metadata

No scientific model metrics were produced or changed.

Audit metadata:

- Current cap3x manifests across five folds: 7,310 total rows; 3,812 unique paths; 0 blank md5; 0 blank source_id.
- Per fold: 1,174 train, 120 validation, 168 test rows.
- Korea dry: 5,440 files; 5,437 unique MD5; 44.1 kHz mono PCM16; median duration 0.468 s.
- Korea abdominal: 12,397 files; 12,377 unique MD5; 44.1 kHz mono PCM16; median duration 0.419 s.
- Sow calm: 560 files; 280 unique MD5; 48 kHz mono DOUBLE; median duration 2.016 s.
- Sow feeding: 420 files; 210 unique MD5; 48 kHz mono DOUBLE; median duration 2.000 s.
- Sow frightened: 1,500 files; 749 unique MD5; 48 kHz mono DOUBLE; median duration 2.000 s.
- Sow anxious: 1,012 files; 504 unique MD5; 48 kHz mono DOUBLE; median duration 2.000 s.

## Data Boundaries and Leakage Audit

- No training, validation, or test role was changed.
- No test-set tuning was performed.
- No model, prototype, calibration, threshold, fusion, SNR, noise draw, prediction, or checkpoint artifact was created.
- No raw audio was modified or deleted.
- No training/evaluation script was modified.
- No `paper_results/` file was modified.
- No paper claims or Data Availability text were changed.
- Fold0 read-only leakage checks passed: cross-split MD5 duplicate groups 0; exact path overlap 0; source_id overlap 0.

## Paper Usability

- `paper_usable`: true for source-provenance audit.
- `submission_complete`: false.
- `new_results_created`: false.
- `paper_claims_changed`: false.
- `data_availability_changed`: false.
- Required final recommendation: `safe_only_after_source_citation_added`.

## Blockers

- Add formal dataset citations and license wording for Korea and Sow call before submission.
- Decide whether raw Korea audio will be redistributed or only source-linked with manifests/checksums; data.go.kr says no usage restriction, but Smart Farm Korea terms should be checked once more.
- Do not use Kaggle scream/cough in final claims until license/owner rights are resolved.
- Add a third-party public-data ethics/exemption statement or obtain journal-specific guidance.

## Questions for GPT Pro

- Should the final manuscript cite both Smart Farm Korea and data.go.kr for the Korea cough source, or cite data.go.kr as the formal public-data catalog and Smart Farm Korea as download page?
- Should the Data Availability statement redistribute raw audio for CC BY/public sources, or avoid bundling audio and provide source URLs plus manifests/checksums?
- Is a separate animal ethics exemption statement needed for reuse of public third-party pig-farm audio?
- Should non-current sources SoundWel, aSwine, and Kaggle be omitted entirely from final paper data availability to keep the final clean-data story narrow?

## Recommended Next Step

Review the new source recovery files and add approved source citations/Data Availability wording. Do not start another stage without a new explicit `APPROVED: true`.
