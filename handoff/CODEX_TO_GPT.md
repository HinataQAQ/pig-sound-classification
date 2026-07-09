# Codex to GPT Handoff - Source Citation and Statement Sync

## Status

Completed. The user provided `APPROVED: true` and requested only data-source
citation completion and statement updates from the clean pig-audio provenance
audit.

Scope was limited to the paper submission package. No model experiment,
training, inference, checkpoint/prototype/calibration construction, threshold
tuning, evaluation-code edit, training-code edit, experimental result CSV/JSON
edit, raw-audio edit, or paper core-claim change was performed.

## Branch

- Branch: `paper/sci-draft-v1`
- Commit SHA: see final Codex response after commit/push

## Changed Files

- Updated `paper/references/references.bib`
- Updated `paper/reproducibility/DATA_AVAILABILITY.md`
- Updated `paper/reproducibility/ETHICS_STATEMENT.md`
- Added `paper/reproducibility/SOURCE_CITATION_STATUS.md`
- Updated `paper/manuscript/paper_en_full_story_polished.md`
- Updated `paper/manuscript/paper_cn_full_story.md`
- Updated `handoff/CODEX_TO_GPT.md`
- Updated `handoff/CODEX_TO_GPT.json`
- Updated `handoff/HISTORY.md`

## Commands and Checks

```powershell
Get-Content -Raw C:\Users\s1205\.codex\plugins\cache\openai-curated-remote\superpowers\6.1.1\skills\using-superpowers\SKILL.md
Get-Content -Raw C:\Users\s1205\.codex\skills\nature-data\SKILL.md
Get-Content -Raw C:\Users\s1205\.codex\skills\nature-data\manifest.yaml
Get-Content -Raw C:\Users\s1205\.codex\skills\nature-data\static\core\stance.md
Get-Content -Raw C:\Users\s1205\.codex\skills\nature-data\static\core\chinese-mode.md
Get-Content -Raw C:\Users\s1205\.codex\skills\nature-data\static\core\workflow.md
Get-Content -Raw handoff\GPT_TO_CODEX.md
Get-Content -Raw paper\reproducibility\PIG_AUDIO_SOURCE_RECOVERY_REPORT.md
Get-Content -Raw paper\reproducibility\PIG_AUDIO_SOURCE_AUDIT.csv
Get-Content -Raw paper\reproducibility\UNRESOLVED_DATA_SOURCES.md
Get-Content -Raw paper\reproducibility\DATA_AVAILABILITY.md
Get-Content -Raw paper\reproducibility\ETHICS_STATEMENT.md
Get-Content -Raw paper\references\references.bib
rg -n "Data Availability|Ethics|dataset|Sow|DEMAND|Korea|Korean|pig-cough|pig cough|clean pig" paper\manuscript\paper_en_full_story_polished.md
rg -n "数据可用|伦理|数据集|母猪|韩国|猪咳嗽|DEMAND|Sow|Korea|pig" paper\manuscript\paper_cn_full_story.md
git diff --name-only -- reports/**/*.csv reports/**/*.json paper/tables/**/*.csv paper/tables/**/*.json paper/appendix/**/*.csv paper/appendix/**/*.json paper/figures/**/*.csv paper/figures/**/*.json
git diff --name-only -- *.py tools/** src/**
<Python brace/key parser for paper/references/references.bib>
<Python guard asserting DATA_AVAILABILITY DOI/URL values are limited to audited DEMAND, Sow Call, and Korea source URLs/DOIs>
rg -n "Smart Farm Korea.*citation_complete|citation_complete.*Smart Farm|Kaggle.*verified|verified.*Kaggle|Kaggle-like.*citation_complete|unresolved_do_not_submit_until_fixed" paper\reproducibility paper\manuscript
git diff --check
```

## Citation Results

- DEMAND: `citation_complete`; existing BibTeX key `demand2018`; DOI
  `10.5281/zenodo.1227121`.
- Sow Call Dataset: `citation_complete`; added BibTeX key
  `sow_call_dataset_2021`; DOI `10.6084/m9.figshare.16940389`; CC BY 4.0.
- Smart Farm Korea / data.go.kr pig-cough voice dataset: `citation_partial`;
  added BibTeX key `smartfarmkorea_pig_cough_voice`; exact data.go.kr and Smart
  Farm Korea URLs came from the source audit; no DOI was fabricated; raw-audio
  redistribution is not claimed and still needs a final provider-terms check
  before any raw-audio archive.
- Kaggle-like porcine scream/cough dataset: `unresolved_do_not_submit_until_fixed`;
  no BibTeX entry was added; it remains excluded from final claims.
- SoundWel, aSwine, and empty placeholder roots: `excluded_from_main_results`
  for this paper-mainline sync; no new citations were added because they are not
  part of the current clean cap3x mainline.

## Metrics and Scientific Results

No model metrics were produced or changed. No abstract numbers, table values,
figure values, result CSV/JSON files, or core conclusions were edited.

Relevant locked metrics remain unchanged:

- 2 s Log-Mel CRNN mean Macro-F1 approximately `0.947237`.
- Hierarchical lambda means remain approximately `0.950506`, `0.951050`, and
  `0.951508` for lambda `0.2`, `0.5`, and `1.0`.
- Hierarchical gain over the 2 s baseline remains nonsignificant.
- DEMAND simulated-noise conclusions remain bounded and unchanged.

## Data Boundaries and Leakage Audit

- Train/validation/test roles were not changed.
- No test-set tuning was performed.
- No fold or seed artifact was constructed.
- No raw audio was modified, copied, redistributed, or deleted.
- No training, evaluation, or model code was modified.
- No experiment result CSV/JSON was modified.
- Existing path/source_id/MD5 leakage-audit statements were preserved; no new
  leakage-relevant data artifact was generated.

## Verification

- `references.bib` parsed successfully with 32 entries and required dataset
  keys present.
- Data Availability DOI/URL guard passed; no fabricated DOI/URL was introduced.
- Diff checks for result CSV/JSON paths returned no changed files.
- Diff checks for Python/training/evaluation code returned no changed files.
- Guardrail search confirmed Korea was not written as `citation_complete`, and
  the Kaggle-like source remains unresolved/excluded.
- `git diff --check` reported only line-ending warnings, no whitespace errors.

## Paper Usability

- `paper_usable`: true for citation and statement synchronization.
- `submission_complete`: conditional.
- The manuscript is safer for submission than before because the current
  clean-source citations and third-party raw-audio non-redistribution statement
  are now present.
- Remaining condition: do not claim raw Korean cough audio redistribution until
  the final Smart Farm Korea / data.go.kr provider-terms check is recorded.

## Blockers

- Final provider-terms screenshot or written confirmation for Korean raw-audio
  redistribution, if the authors plan to archive raw audio.
- Final repository release tag/DOI and licence for shareable code/results
  package.
- Target-journal decision on whether a no-new-animal-experiment statement is
  sufficient for computational reuse of third-party public pig audio.
- Kaggle-like scream/cough source remains unresolved and excluded.

## Questions for GPT Pro

- Should Korea be left as `citation_partial` in the final submission package, or
  upgraded only after a saved provider-terms screenshot/check?
- Should the final Data Availability statement mention only current mainline
  sources, or include excluded non-current sources in an appendix-only note?
- Is an explicit ethics-exemption sentence needed for public third-party pig
  audio reuse in the target journal?

## Recommended Next Step

Review the synchronized citation/status wording. Do not start another stage
without a new explicit `APPROVED: true`.
