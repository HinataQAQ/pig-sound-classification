# AGENTS.md — Pig Sound Classification Project

## Project mission

Develop a leakage-free and reproducible pig-vocalization classification and
prediction system suitable for an EI/SCI engineering paper and later deployment
in real pig-farm environments.

The current research focus is no longer ordinary MCTAFD feature stacking.
The validated mainline is:

- class-capped cap3x training expansion
- 2-second Log-Mel input
- CRNN backbone
- hierarchical subtype auxiliary supervision

Do not restart obsolete experimental directions unless explicitly requested.

## Repository and branch

Primary repository:
HinataQAQ/pig-sound-classification

Primary branch:
current-mctafd-ablation

Paper-ready results:
paper_results/

Generated experimental results:
reports/

Key manifests:
data/manifests_pigvocal_4class_expanded_train_cv5_cap3x/

## Environment

Operating system:
Windows

Shell:
PowerShell

Conda environment:
pigsound-gpu

Project root:
C:\py\pigsound\pig-sound-classification

Before running Python:

    conda activate pigsound-gpu
    cd C:\py\pigsound\pig-sound-classification
    $env:PYTHONNOUSERSITE="1"

Use commands that work in Windows PowerShell.

## Validated research state

Four main classes:

- cough
- calm_grunt
- feeding
- stress_vocal

Six auxiliary subtypes:

- dry_cough
- abdominal_cough
- calm_grunt
- feeding
- frightened_stress
- anxious_stress

Best established closed-set mainline:

- cap3x + 2-second Log-Mel CRNN
- 25 fold-seed runs
- mean Macro-F1 approximately 0.947237

Hierarchical auxiliary results:

- lambda=0.2: mean Macro-F1 approximately 0.950506
- lambda=0.5: mean Macro-F1 approximately 0.951050
- lambda=1.0: mean Macro-F1 approximately 0.951508

Interpretation:

- The statistically significant major gain comes from 2-second context.
- Hierarchical auxiliary supervision provides a small positive mean gain and
  improves feeding/stress_vocal boundary errors.
- The hierarchical gain over the 2-second baseline is not statistically
  significant; never claim otherwise.
- lambda=1.0 has the highest mean.
- lambda=0.5 has the lowest cross-run standard deviation and is the balanced
  setting.

## Key validated files

Training:

- tools/train_mctafd_crnn_ablation.py
- tools/train_hier_longcontext_crnn.py

Summaries:

- tools/summarize_cv5_expanded_variants.py
- tools/summarize_hier_longcontext.py
- tools/summarize_hier_longcontext_confusion_by_weight.py

Paired statistics:

- tools/paired_cv5_cap3x_logmel_dur2_vs_dur1.py
- tools/paired_cv5_cap3x_hier_vs_dur2_by_weight.py

Duration checks:

- tools/audit_manifest_duration.py
- tools/audit_duration_by_label.py
- tools/duration_only_baseline.py

Paper-ready outputs:

- paper_results/tables/
- paper_results/scripts/
- paper_results/manifests/
- paper_results/appendix/

## Negative or completed experimental directions

Do not repeat these as primary clean-audio directions unless explicitly asked:

- direct MCTAFD
- gated MCTAFD as final main model
- BiLSTM replacement
- PCEN as clean frontend
- spectral-gate denoising as clean frontend
- dual Log-Mel
- full or light SpecAugment on clean audio
- temporal attention pooling
- no-feeding pretraining
- late fusion
- stage-two feeding/stress expert
- freeze-five fine-tuning

These are retained as ablations or future noisy-environment components.

## Non-negotiable research constraints

1. Never tune parameters on the test set.
2. Train data may build model parameters and class prototypes.
3. Validation data may calibrate thresholds, temperatures, fusion weights, and
   rejection criteria.
4. Test data is evaluation only.
5. Maintain exact-path, source-ID, and MD5-hash disjointness.
6. Never mix folds or seeds when constructing a model-specific artifact.
7. Compare models using matched fold-seed pairs.
8. Screening experiments use 5 folds x 3 seeds.
9. Promising experiments must be completed to 5 folds x 5 seeds.
10. Report mean, standard deviation, minimum, maximum, bootstrap confidence
    interval, and Wilcoxon paired p-value where applicable.
11. Never describe a nonsignificant result as statistically significant.
12. Preserve existing result directories. Use new experiment names.
13. Do not overwrite established checkpoints or summary files.
14. All new scripts must support --help and validate missing files clearly.
15. Use zero_division=0 for classification metrics.

## Output contract for training and evaluation

Each experiment directory should contain:

- summary.json
- test_pred.csv

Where relevant, test_pred.csv should include:

- y_true
- y_pred
- class probabilities
- uncertainty or distance scores
- top-k predictions

Paper-ready aggregate files should be copied into:

- paper_results/tables/
- paper_results/scripts/

## Coding rules

- Prefer adding a new script over destabilizing an established baseline script.
- Reuse existing dataset, feature, model, and evaluation functions where safe.
- Add type hints and clear error messages.
- Keep random seeds deterministic.
- Do not add large dependencies without explaining why.
- Do not modify audio files.
- Do not require manual listening as part of the default workflow.
- Avoid OCR or manual label inference.
- Keep all thresholds and hyperparameters configurable through argparse.

## Definition of done

A task is complete only when:

1. Relevant code is implemented.
2. A one-fold one-seed debug run succeeds.
3. Existing established scripts still run.
4. New outputs are written to unique paths.
5. Leakage checks remain valid.
6. A short README or doc explains commands and output fields.
7. The diff is reviewed for accidental test-set tuning or data leakage.

## GPT Pro handoff protocol

After every research or experiment task:

1. Update:
   - handoff/CODEX_TO_GPT.md
   - handoff/CODEX_TO_GPT.json
   - handoff/HISTORY.md

2. Include:
   - exact commands
   - branch and commit
   - changed files
   - metrics
   - leakage audit
   - paper usability
   - blockers
   - questions requiring scientific judgment

3. Commit and push the handoff files.

4. Stop after producing the handoff.
   Do not automatically start the recommended next stage.

5. Before starting another stage, read:
   - handoff/GPT_TO_CODEX.md

6. Only execute work explicitly marked:
   APPROVED: true
