# Codex to GPT Handoff - Pre-Submission Statement Placeholder Fix

## Status

Completed. The user provided `APPROVED: true` and requested only
pre-submission statement and citation-placeholder repair.

Scope was limited to manuscript/reproducibility text. No model experiment,
training, inference, checkpoint/prototype/calibration construction, threshold
tuning, evaluation-code edit, training-code edit, experimental result CSV/JSON
edit, raw-audio edit, figure edit, or core scientific-conclusion change was
performed.

## Branch

- Branch: `paper/sci-draft-v1`
- Commit SHA: see final Codex response after commit

## Changed Files

- Updated `paper/manuscript/paper_en_full_story_polished.md`
- Updated `paper/manuscript/paper_en_full_story.md`
- Updated `paper/manuscript/paper_cn_full_story.md`
- Updated `paper/references/citation_audit.csv`
- Added `paper/reproducibility/SUBMISSION_READINESS_CHECKLIST.md`
- Updated `handoff/CODEX_TO_GPT.md`
- Updated `handoff/CODEX_TO_GPT.json`
- Updated `handoff/HISTORY.md`

Inspected but not changed in this round:

- `paper/references/references.bib`
- `paper/reproducibility/DATA_AVAILABILITY.md`
- `paper/reproducibility/ETHICS_STATEMENT.md`
- `paper/reproducibility/SOURCE_CITATION_STATUS.md`
- `paper/reproducibility/PIG_AUDIO_SOURCE_RECOVERY_REPORT.md`
- `paper/CLAIMS_AND_EVIDENCE_v2.md`

## Commands and Checks

```powershell
Get-Content -Raw C:\Users\s1205\.codex\plugins\cache\openai-curated-remote\superpowers\6.1.1\skills\using-superpowers\SKILL.md
Get-Content -Raw C:\Users\s1205\.codex\skills\nature-data\SKILL.md
Get-Content -Raw C:\Users\s1205\.codex\skills\nature-data\manifest.yaml
Get-Content -Raw C:\Users\s1205\.codex\skills\nature-data\static\core\stance.md
Get-Content -Raw C:\Users\s1205\.codex\skills\nature-data\static\core\chinese-mode.md
Get-Content -Raw C:\Users\s1205\.codex\skills\nature-data\static\core\workflow.md
Get-Content -Raw C:\Users\s1205\.codex\plugins\cache\openai-curated-remote\superpowers\6.1.1\skills\verification-before-completion\SKILL.md
Get-Content -Raw handoff\GPT_TO_CODEX.md
rg -n --fixed-strings <citation-needed placeholder phrase> <target files>
rg -n --fixed-strings <references-placeholder phrase> <target files>
rg -n --fixed-strings <not-finalized phrase> <target files>
rg -n --fixed-strings <author-completion phrase> <target files>
rg -n --fixed-strings <positive-real-farm-validation phrase> <target files>
rg -n --fixed-strings <open-raw-audio-redistribution phrase> <target files>
rg -n "GitHub release URL|code DOI|Smart Farm Korea|citation_complete|citation_partial|Kaggle-like|real-farm external validation|raw third-party pig audio|redistributed" paper\manuscript\paper_en_full_story_polished.md paper\reproducibility\*.md
git diff --name-only -- reports/**/*.csv reports/**/*.json paper/tables/**/*.csv paper/tables/**/*.json paper/appendix/**/*.csv paper/appendix/**/*.json paper/figures/**/*.csv paper/figures/**/*.json
git diff --name-only -- *.py tools/** src/**
```

## Text Updates

- Replaced the Related Work citation placeholder with verified citation keys
  already present in `references.bib` and `citation_audit.csv`: pig-call
  valence/context and welfare-monitoring references, pig-cough recognition
  references, and noisy smart-farm pig-vocalisation references.
- Applied the same placeholder cleanup to the older English manuscript draft,
  and removed the remaining references-placeholder heading from the Chinese
  manuscript draft.
- Removed a non-result audit-note wording in `citation_audit.csv` that would
  otherwise trigger the requested placeholder grep; citation keys and audit
  conclusions were not changed.
- Converted manuscript Data Availability wording to submission style:
  processed aggregate tables and figure source data are available through the
  repository/supplementary package; raw third-party pig audio is not
  redistributed; Smart Farm Korea / data.go.kr, Sow Call Dataset, and DEMAND
  are accessed through original providers; the unresolved Kaggle-like source is
  excluded.
- Converted Code Availability wording to submission style with a single
  code-release URL placeholder; no code DOI is claimed; checkpoints,
  prototype artifacts, and raw audio are excluded unless released separately.
- Converted Ethics wording to submission style: no new animal experiment,
  intervention, or prospective farm recording; computational reuse of public or
  third-party audio; source-side permissions remain with providers; restrictions
  are in Data Availability.
- Added `SUBMISSION_READINESS_CHECKLIST.md` with remaining required items:
  target journal, code release URL/tag, Korea provider terms final check,
  corresponding author/affiliations, funding, conflict of interest, and whether
  checkpoint/prototype artifacts will be archived.

## Metrics and Scientific Results

No model metrics were produced or changed. No numerical result, abstract value,
paper table, figure, source result CSV/JSON, or core conclusion was edited.

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

- Placeholder grep checks returned no matches for the four requested placeholder
  phrases.
- Boundary grep checks returned no matches for the two requested positive-claim
  phrases.
- The code-release URL placeholder is present only once, in the manuscript Code
  Availability section.
- Smart Farm Korea remains `citation_partial`; it was not upgraded to
  `citation_complete`.
- The Kaggle-like source remains unresolved/excluded and is not included in the
  main data description.
- Diff checks for result CSV/JSON paths returned no changed files.
- Diff checks for Python/training/evaluation code returned no changed files.

## Paper Usability

- `paper_usable`: true for pre-submission placeholder and statement repair.
- `submission_complete`: conditional.
- The manuscript is closer to submission, but still needs a final code release
  URL/tag, target-journal metadata, corresponding-author/affiliation/funding/COI
  metadata, and the Korean provider-terms check before any raw-audio archive.

## Blockers

- Target journal and article-type requirements.
- Final public code release URL/tag; no code DOI is currently claimed.
- Final Smart Farm Korea / data.go.kr provider-terms check; Korea remains
  `citation_partial`, and raw Korean audio redistribution is not claimed.
- Corresponding author, affiliations, funding, and conflict-of-interest
  declarations.
- Decision on checkpoint/prototype artifact archival.
- Kaggle-like scream/cough source remains unresolved and excluded.

## Questions for GPT Pro

- Should the final manuscript keep the code-release URL placeholder until the release is
  minted, or should the authors provide the actual repository tag now?
- Does the target journal require a formal exemption sentence for computational
  reuse of third-party animal audio?
- Should checkpoints/prototype NPZ artifacts be archived for review, or should
  the paper rely on code regeneration plus aggregate result files?

## Recommended Next Step

Replace the single code-release placeholder after the final repository release
exists. Do not start another stage without a new explicit `APPROVED: true`.

---

# Previous Handoff - Source Citation and Statement Sync

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
