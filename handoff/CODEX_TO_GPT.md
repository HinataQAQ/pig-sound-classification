# Codex to GPT Handoff - English Manuscript Polishing

## Status

Completed. The task was approved by `handoff/GPT_TO_CODEX.md`
(`APPROVED: true`) and directly requested by the user.

This round used the installed `nature-polishing` skill. I loaded
`SKILL.md`, `manifest.yaml`, all manifest `always_load` files, and the relevant
axis fragments before polishing.

No model training, model inference, checkpoint/prototype/calibration
construction, threshold tuning, SNR/noise draw change, raw audio edit, or source
result CSV/JSON modification was performed.

## Branch

- Branch: `paper/sci-draft-v1`
- Base commit: `5bf1ced37ba27362eede3c3e6084df39d0b1e414`
- Commit SHA: see final Codex response after commit/push

## Changed Files

- Added `paper/manuscript/paper_en_full_story_polished.md`
- Added `paper/manuscript/polishing_change_log.md`
- Added `paper/manuscript/overclaim_audit.md`
- Updated `handoff/CODEX_TO_GPT.md`
- Updated `handoff/CODEX_TO_GPT.json`
- Updated `handoff/HISTORY.md`

## Commands

```powershell
Get-Content -LiteralPath "C:\Users\s1205\.codex\plugins\cache\openai-curated-remote\superpowers\6.1.1\skills\using-superpowers\SKILL.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\SKILL.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\manifest.yaml"
Get-Content -LiteralPath "C:\Users\s1205\.codex\plugins\cache\openai-curated-remote\superpowers\6.1.1\skills\using-superpowers\references\codex-tools.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\_shared\core\reader-workflow.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\_shared\core\paper-type-taxonomy.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\_shared\core\ethics.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\_shared\core\terminology-ledger.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\core\stance.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\core\failure-modes.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\core\output-format.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\fragments\paper_type\algorithmic.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\fragments\section\title.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\fragments\section\abstract.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\fragments\section\intro.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\fragments\section\methods.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\fragments\section\results.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\fragments\section\discussion.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\fragments\section\conclusion.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\fragments\language\en.md"
Get-Content -LiteralPath "C:\Users\s1205\.codex\skills\nature-polishing\static\fragments\journal\generic.md"
Get-Content -LiteralPath "paper\manuscript\paper_en_full_story.md"
Get-Content -LiteralPath "handoff\GPT_TO_CODEX.md"
git status --short -- paper/manuscript handoff
git branch --show-current
git rev-parse HEAD
git remote -v
rg -n -i "\b(prove|proves|proved|proven|novel|first)\b" paper/manuscript/paper_en_full_story_polished.md paper/manuscript/polishing_change_log.md paper/manuscript/overclaim_audit.md
$files = @('paper/manuscript/paper_en_full_story_polished.md','paper/manuscript/polishing_change_log.md','paper/manuscript/overclaim_audit.md'); foreach ($f in $files) { $text = Get-Content -LiteralPath $f -Raw; $non = [regex]::Matches($text, '[^\x00-\x7F]').Count; "$f non_ascii=$non" }
$env:PYTHONNOUSERSITE='1'; $env:PYTHONIOENCODING='utf-8'; @'<python QA script>'@ | & 'C:\py\anaconda3\envs\pigsound-gpu\python.exe' -
$env:PYTHONNOUSERSITE='1'; $env:PYTHONIOENCODING='utf-8'; & 'C:\py\anaconda3\envs\pigsound-gpu\python.exe' tools\summarize_hier_longcontext.py --help
git status --short -- reports/hier_longcontext_runs.csv reports/hier_longcontext_summary.csv
git diff -- reports/hier_longcontext_runs.csv reports/hier_longcontext_summary.csv
```

Notes:

- `conda activate pigsound-gpu` failed in this PowerShell session with a conda
  GBK `UnicodeEncodeError`. The text QA and script check were therefore run
  with the environment Python executable directly:
  `C:\py\anaconda3\envs\pigsound-gpu\python.exe`.
- `tools/summarize_hier_longcontext.py --help` is not a conventional help path.
  It executed the summary script and rewrote
  `reports/hier_longcontext_runs.csv` and
  `reports/hier_longcontext_summary.csv`. Git showed no diff for those files,
  so no tracked result content changed.

## Nature-Polishing Routing

- Detected paper type: `algorithmic`
- Detected sections: `title`, `abstract`, `intro`, `methods`, `results`,
  `discussion`, `conclusion`
- Detected language: `en`
- Detected journal: `generic`
- On-demand references: none loaded; the user did not request phrasebank,
  Nature article-pattern calibration, or LaTeX layout work.

## Language and Structural Work

- Rewrote the manuscript into cautious SCI applied-engineering English.
- Converted visible prose to British English, including `vocalisation`,
  `behaviour`, `organise`, `summarise`, `artefact`, `generalised`,
  `finalised`, and `licence`.
- Split long sentences; automated QA found `long_sentences_gt30 = 0`.
- Removed `prove`, `novel`, and `first` from the polished manuscript package.
- Preserved all numeric tokens from the source manuscript:
  `numeric_tokens_source = 168`, `numeric_tokens_polished = 168`,
  `missing = {}`, `added = {}`.
- Added a terminology ledger and a dedicated overclaim audit.
- Did not add references or fabricate missing evidence.

## Metrics and Scientific Boundaries Preserved

- 2 s Log-Mel clean mean Macro-F1 `0.9472`; 1 s Log-Mel `0.9226`; paired delta
  `0.0246`; Wilcoxon p `0.000162`.
- 3 s Log-Mel comparison `0.9434` remains a 15-run comparison.
- Hierarchical lambda=1.0 highest mean Macro-F1 `0.9515`.
- Lambda=0.5 balanced mean Macro-F1 `0.9510` and standard deviation `0.0124`.
- Hierarchical gain over the 2 s baseline is not described as statistically
  significant.
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

## Overclaim Audit Results

Main risks identified:

- The title phrase `duration-aware hierarchical representations` could imply a
  generally superior system unless the abstract clearly bounds the hierarchy
  result.
- Positive hierarchical means could invite a statistically significant clean
  performance claim. The polished text explicitly rejects that claim.
- Clean hierarchical prototype results could be confused with noise robustness.
  The polished text separates clean semantic explanation from noise robustness.
- Main prototype noise gains could be overgeneralised to real farms. The
  polished text restricts them to DEMAND simulated additive noise.
- Prototype selective-risk behaviour could be overgeneralised into uncertainty
  calibration. The polished text states that prototypes do not universally
  improve uncertainty.

## Evidence Still Needed from Authors

- Verified livestock-acoustics and pig-vocalisation references for Related
  Work.
- Clean pig audio availability, owner, licence, and access route.
- Code release tag, archived DOI, and software licence.
- Animal-use approval authority and approval number, if applicable.
- Real-farm external validation data, if the paper wants deployment claims.
- Additional evidence or methods for reliable cough recognition under 0 dB
  active-event SNR.

## Data Boundaries and Leakage Audit

- No training, validation, or test role was changed.
- No test-set tuning was performed.
- No new model, prototype, calibration, threshold, fusion, SNR, noise draw,
  prediction, or checkpoint artefact was created.
- No raw clean audio or DEMAND audio was read or modified.
- No `paper/tables/*.csv`, `paper/appendix/*.csv`, prediction CSV/JSON, or
  source result file was modified.
- Path/source_id/MD5 disjointness is unchanged because this task only writes
  manuscript documents.

## Verification

- Forbidden-word scan for `prove`, `proves`, `proved`, `proven`, `novel`, and
  `first` returned no matches after cleanup.
- Numeric-token audit matched source and polished manuscript exactly:
  168 tokens in both, with no missing or added numeric tokens.
- Sentence-length audit reported zero prose sentences above 30 words.
- ASCII audit reported zero non-ASCII characters in all three new manuscript
  files.
- `tools/summarize_hier_longcontext.py --help` executed with env Python but is
  not true argparse help. It rewrote summary files with no git diff.

## Paper Usability

- `paper_usable`: true for English language polishing and overclaim auditing.
- `submission_complete`: false.
- `new_results_created`: false.
- `source_csv_json_values_modified`: false.
- `simulated_noise_only`: true.
- `real_farm_external_validation`: false.

## Blockers

- Related Work still needs verified livestock-acoustics and pig-vocalisation
  citations.
- Data availability, clean-audio rights, and licence fields remain unresolved.
- Code release tag, archived DOI, and software licence remain unresolved.
- Animal ethics approval details remain unresolved.
- Real-farm external validation remains absent.

## Questions for GPT Pro

- Which livestock-acoustics and pig-vocalisation studies should be cited in
  Related Work?
- Should the title be shortened for the target SCI applied-engineering journal?
- Should `vocalisation` be kept as British English, or should the final target
  journal require US spelling?
- What exact ethics, data-rights, and code-release statements can the authors
  legally make?

## Recommended Next Step

Have GPT Pro or the authors fill the evidence gaps for references, data rights,
code release, and ethics. Do not start new experiments or another paper stage
without a new explicit `APPROVED: true`.
