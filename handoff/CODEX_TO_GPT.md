# Codex to GPT Handoff - Nature Citation Audit

## Status

Completed. The task was directly requested by the user and
`handoff/GPT_TO_CODEX.md` contained `APPROVED: true`.

This round used the installed `nature-citation` skill. I loaded `SKILL.md`,
`manifest.yaml`, all manifest `always_load` files, and the relevant on-demand
references for search strategy and long-article/script usage before auditing.

No model training, model inference, checkpoint/prototype/calibration
construction, threshold tuning, SNR/noise draw change, raw audio edit, or source
result CSV/JSON modification was performed.

## Branch

- Branch: `paper/sci-draft-v1`
- Base commit: `1808464`
- Commit SHA: see final Codex response after commit/push

## Changed Files

- Updated `paper/references/references.bib`
- Added `paper/references/citation_audit.csv`
- Added `paper/references/missing_citations.md`
- Updated `handoff/CODEX_TO_GPT.md`
- Updated `handoff/CODEX_TO_GPT.json`
- Updated `handoff/HISTORY.md`

## Commands and Searches

```powershell
Get-Content -Raw -LiteralPath "$env:USERPROFILE\.codex\skills\nature-citation\SKILL.md"
Get-Content -Raw -LiteralPath "$env:USERPROFILE\.codex\skills\nature-citation\manifest.yaml"
Get-Content -Raw -LiteralPath "$env:USERPROFILE\.codex\skills\nature-citation\static\core\principles.md"
Get-Content -Raw -LiteralPath "$env:USERPROFILE\.codex\skills\nature-citation\static\core\chinese-mode.md"
Get-Content -Raw -LiteralPath "$env:USERPROFILE\.codex\skills\nature-citation\static\core\workflow.md"
Get-Content -Raw -LiteralPath "$env:USERPROFILE\.codex\skills\nature-citation\references\search-strategy.md"
Get-Content -Raw -LiteralPath "$env:USERPROFILE\.codex\skills\nature-citation\references\script-usage.md"
Get-Content -Raw -LiteralPath "handoff\GPT_TO_CODEX.md"
git status --short --branch
Get-ChildItem -LiteralPath paper\manuscript, paper\references, paper | Select-Object FullName, Length, LastWriteTime
rg -n "^@" paper\references\references.bib
rg -n "^#|^##|cite|@|citation|reference|claim|Claim|CLAIM" paper\manuscript\paper_en_full_story.md paper\CLAIMS_AND_EVIDENCE_v2.md
Get-Content -Raw -LiteralPath paper\references\references.bib
Get-Content -Raw -LiteralPath paper\CLAIMS_AND_EVIDENCE_v2.md
Get-Content -Raw -LiteralPath paper\manuscript\paper_en_full_story.md
Get-Content -Raw -LiteralPath paper\manuscript\paper_cn_full_story.md
rg -n "\[@|Citation needed|References placeholder|DEMAND|Log-Mel|MFCC|SpecAugment|PCEN|prototype|hierarchical|selective|risk-coverage|pig|vocalization|noise" paper\manuscript\paper_en_full_story.md
$env:PYTHONNOUSERSITE="1"; $env:PYTHONIOENCODING="utf-8"; @'<Crossref DOI metadata script>'@ | & C:\py\anaconda3\envs\pigsound-gpu\python.exe -
(rg -n "^@" paper\references\references.bib | Measure-Object).Count
Import-Csv -LiteralPath paper\references\citation_audit.csv
rg -n "[^\x00-\x7F]" paper\references\references.bib paper\references\citation_audit.csv paper\references\missing_citations.md
git diff -- paper\references\references.bib paper\references\citation_audit.csv paper\references\missing_citations.md
```

Web/metadata searches used English query families for:

- pig vocalization classification, emotional valence and production context
- pig cough sound recognition and respiratory/sick-cough detection
- noisy animal-farm pig vocalization recognition
- Log-Mel/MFCC and animal-sound classification
- CRNN audio and sound-event classification
- prototypical networks and audio prototype interpretability
- hierarchical sound labels/ontology
- selective classification/risk-coverage
- DEMAND environmental-noise dataset
- PCEN and SpecAugment method provenance

Notes:

- `conda activate pigsound-gpu` failed once in this PowerShell session with a
  conda GBK `UnicodeEncodeError`. DOI metadata checks were then run with
  `C:\py\anaconda3\envs\pigsound-gpu\python.exe` and
  `PYTHONNOUSERSITE=1`.
- The `nature_citation.py` script was not used as the final search engine
  because the user explicitly allowed broader academic search beyond the
  default Nature/CNS scope. The skill workflow was used for segmentation,
  conservative support grading, and audit structure.

## Citation Expansion Results

- Existing bibliography entries before audit: `6`.
- New bibliography entries added: `24`.
- Total bibliography entries after audit: `30`.
- Claim-audit rows: `32`.
- New files contain zero non-ASCII characters.
- `citation_audit.csv` parsed successfully with PowerShell `Import-Csv`.

Added reference groups:

- Pig call valence/context: Briefer 2022; Tallet 2013; Illmann 2013; Weary
  1995; Manteuffel 2004.
- Pig cough and pig abnormal-vocalization recognition: Ferrari 2008;
  Exadaktylos 2008; Yin 2021; Shen 2022; Xie 2024.
- Noisy pig-vocalization classification: Chung 2025 Scientific Reports.
- Bioacoustic/livestock acoustic monitoring background: McLoughlin 2019;
  Coutant 2024; Stowell 2022.
- Animal/audio feature and classifier background: Acevedo 2009; Huang 2014;
  Kahl 2021; Cakir 2017; Hershey 2017; Gemmeke 2017.
- PCEN and SpecAugment: Lostanlen 2019; Park 2019.
- Hierarchical/prototype interpretation background: Silla 2011; Heinrich 2025.

## Core Claim Support

- Pig call valence/context: strongest external support is
  `@briefer2022pigcalls`, with additional primary support from
  `@tallet2013piglets`, `@illmann2013signalneed`, and `@weary1995calling`.
- Pig cough recognition: strongest external support is
  `@ferrari2008cough`, `@exadaktylos2008cough`, `@yin2021pigcoughcnn`, and
  `@shen2022pigcoughfusion`.
- Noisy pig-vocalization/farm setting context: strongest external support is
  `@chung2025pvmc` and `@xie2024abnormalpig`; these must not be used to claim
  this paper has real-farm external validation.
- 2 s Log-Mel CRNN mainline: strongest support remains internal
  `paper/tables/table3_duration_comparison.csv` plus the paired-duration script.
  No direct external standard for 2 s pig-vocalization windows was found.
- CRNN audio classification: `@choi2017crnn` and `@cakir2017crnn_sed`.
- Log-Mel/MFCC animal/audio classification: `@mcfee2015librosa`,
  `@acevedo2009animalcalls`, `@huang2014anuran`, `@kahl2021birdnet`, and
  `@hershey2017cnn_audio`.
- PCEN and SpecAugment method provenance: `@lostanlen2019pcen` and
  `@park2019specaugment`.
- Prototypical networks: `@snell2017prototypical`.
- Bioacoustic prototype interpretability: `@heinrich2025audioprotopnet`
  provides analogical audio-prototype support, but it is a bird-sound paper,
  not a pig-vocalization result.
- Hierarchical classification/sound ontology: `@silla2011hierarchical` and
  `@gemmeke2017audioset` provide background only.
- Selective classification/risk-coverage: existing `@geifman2017selective` and
  `@fdshifts_riskcoverage` remain appropriate.
- DEMAND: existing `@demand2018` remains appropriate for public environmental
  noise recordings.

## Rejected or Not-Used Candidates

- ResearchGate, Google Scholar, and aggregator pages were treated only as
  discovery aids, not citation support.
- Guarino et al. field-test cough-detection work was not added because exact
  DOI/publisher metadata were not verified in this pass and the cough claim is
  already covered by four verified primary papers.
- Generic environmental sound CNN papers were omitted because the bibliography
  was kept within the requested 25--35 entries and more relevant audio/animal
  references were available.
- Additional pig cough fusion/monitoring papers beyond Yin 2021 and Shen 2022
  were not needed for the current claim set.
- Reviews were not used as specific experimental-result support.
- DEMAND-as-real-farm-validation was explicitly rejected as an interpretation.
- Prototype learning as a claimed novelty was explicitly rejected.

## Metrics and Scientific Boundaries Preserved

No scientific result was changed. Key locked metrics remain:

- 2 s Log-Mel clean mean Macro-F1 `0.9472`; 1 s Log-Mel `0.9226`; paired delta
  `0.0246`; Wilcoxon p `0.000162`.
- 3 s Log-Mel comparison `0.9434` remains a 15-run comparison.
- Hierarchical lambda=1.0 highest mean Macro-F1 `0.9515`.
- Lambda=0.5 balanced mean Macro-F1 `0.9510` and standard deviation `0.0124`.
- Hierarchical gain over the 2 s baseline remains nonsignificant.
- Clean hierarchical prototype mean Macro-F1 `0.9540`, but clean
  hierarchical - raw Softmax p `0.058253` remains nonsignificant.
- ALL_NOISY raw/main prototype/hierarchical prototype Macro-F1
  `0.6173/0.6230/0.6200`.
- ALL_NOISY main prototype - raw Softmax delta `0.005755`, 95% CI
  `[0.001606, 0.010894]`, Wilcoxon p `0.010511`.
- Hierarchical prototype - raw Softmax under ALL_NOISY remains nonsignificant
  with delta `0.002710` and p `0.312333`.
- Hierarchical prototype remains below main prototype under ALL_NOISY, delta
  `-0.003045`, p `0.000376`.
- 0 dB active-event SNR remains an extreme stress condition.
- Cough recognition at 0 dB remains unreliable.
- Prototype inference is not described as a universal uncertainty improvement.
- DEMAND is simulated additive noise only.
- Real-farm external validation was not performed.

## Data Boundaries and Leakage Audit

- No training, validation, or test role was changed.
- No test-set tuning was performed.
- No new model, prototype, calibration, threshold, fusion, SNR, noise draw,
  prediction, or checkpoint artifact was created.
- No raw clean audio or DEMAND audio was read or modified.
- No `paper/tables/*.csv`, `paper/appendix/*.csv`, prediction CSV/JSON, or
  source result file was modified.
- Path/source_id/MD5 disjointness is unchanged because this task only writes
  citation audit documents.
- Newly added citations do not justify any new deployment, real-farm, or
  statistical-significance claim.

## Verification

- `references.bib` entry count: `30`.
- `citation_audit.csv` row count: `32`.
- `citation_audit.csv` parsed successfully with `Import-Csv`.
- ASCII audit found zero non-ASCII characters in the three target reference
  files.
- Git diff was reviewed for the target reference files.

## Paper Usability

- `paper_usable`: true for citation expansion and audit.
- `submission_complete`: false.
- `new_results_created`: false.
- `source_csv_json_values_modified`: false.
- `simulated_noise_only`: true.
- `real_farm_external_validation`: false.

## Blockers

- Original pig-vocalization recording source, owner, license, access route and
  permission status remain unresolved.
- Animal ethics approval authority and approval number or exemption statement
  remain unresolved.
- Code release tag, archived DOI and software license remain unresolved.
- No direct external source was found for exactly 2 s pig-vocalization
  processing as a general standard.
- Spectral-gate denoising still lacks a formal citation if the final manuscript
  needs method provenance beyond an implementation note.

## Questions for GPT Pro

- Which of the new domain citations should be inserted into the final Related
  Work paragraph versus kept only in the bibliography?
- Should the final manuscript add inline citations now, or should the authors
  first approve the `citation_audit.csv` mapping?
- Can the authors provide clean pig-audio source, ethics and license details?
- Should MCTAFD be cited externally, defined internally, or removed from the
  final related-work/method framing?

## Recommended Next Step

Review `paper/references/citation_audit.csv` and insert approved inline
citations into the manuscript. Do not start new experiments or another paper
stage without a new explicit `APPROVED: true`.
