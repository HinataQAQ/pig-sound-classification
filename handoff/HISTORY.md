# Handoff History

## 2026-06-23 - Handoff protocol initialized

- Status: completed
- Task: Created Codex/GPT handoff templates and added the GPT Pro handoff protocol to `AGENTS.md`.
- Research outputs: none; documentation/setup only.
- Metrics: not applicable.
- Leakage audit: not applicable; no data, manifests, checkpoints, or predictions were modified.

## 2026-06-23 - Exact prototype final review fixes

- Status: completed
- Branch: `prototype-exact-review`
- Task: Fixed final PR review blockers for manifest content SHA enforcement,
  prediction input roles, result qualification fields, zero-valued calibration
  parameters, path normalization, and leakage audit errors.
- Metrics: fold0/seed3407 exact Softmax Macro-F1
  `0.946360153256705`; hierarchical prototype Macro-F1
  `0.9522727272727273`.
- Leakage audit: train/val/test path, source_id, and md5 disjointness remained
  valid; validation and frozen test manifest SHA mismatch negative tests passed.
- Paper usability: exact outputs are `feature_pipeline_equivalent=true` but
  `single_fold_debug=true`, so `paper_main_result=false`.

## 2026-06-23 - Exact prototype provenance qualification fix

- Status: completed
- Branch: `prototype-exact-review`
- Task: Restricted paper-main provenance to aggregate summaries only, required
  prediction metadata during evaluation, added prediction/calibration SHA
  verification, and added the exact CV aggregate gate script.
- Metrics: fold0/seed3407 exact metrics unchanged; hierarchical prototype
  Macro-F1 remained `0.9522727272727273`.
- Tests: 37 core unit tests passed, including metadata-missing, inference-role,
  edited prediction CSV SHA mismatch, fold1/seed42 non-paper-main, and aggregate
  paper-main gate tests.
- Scope: no full CV and no lambda=1.0 run was started.

## 2026-06-23 - Exact prototype aggregate summary protocol fix

- Branch: `prototype-exact-review`
- Task: Tightened `tools/summarize_cv5_exact_prototype.py` so aggregate paper
  provenance is allowed only for fixed 5 folds x 3 seeds screening and 5 folds
  x 5 seeds final protocols.
- Result qualification: 5x3 screening now sets `paper_candidate_result=true`
  and `paper_main_result=false`; 5x5 final sets both
  `paper_candidate_result=true` and `paper_main_result=true`.
- Statistics: aggregate summary now emits method means/std/min/max,
  fold-seed-paired deltas with bootstrap CI and Wilcoxon p-value, and aggregated
  confusion summaries for raw Softmax, prototype, and hierarchical methods.
- Tests: 47 core unit tests passed, including 1x1, 4x5, duplicate,
  unexpected-fold, mixed-lambda, paired-statistics, fixed-bootstrap, and
  confusion-aggregation cases.
- Scope: no full CV, no fold/seed expansion, and no model, feature,
  calibration, prediction, or evaluation core logic changes.

## 2026-06-24 - Lambda 0.5 exact prototype 5x3 screening

- Status: completed
- Branch: `prototype-cv5-screening-w05`
- Base integration: PR #1 was squash-merged into `current-mctafd-ablation`
  with merge commit `9ee233c04ef377a6acbaeaf233da7137f2add2d4`.
- Task: Ran the approved exact hierarchical acoustic prototype screening
  experiment for folds `0,1,2,3,4` and seeds `42,2024,3407`.
- Preflight: 15 checkpoints, 15 summaries, 15 reference `test_pred.csv` files,
  and all fold train/val/test manifests were present. Summaries passed
  `hier_aux_weight=0.5`, `dur_s=2.0`, `feature_mode=logmel`, seed, and label
  order checks. Checkpoints were loadable state_dict files.
- Screening status: 15/15 fold-seed runs completed. Each run performed exact
  Softmax reproduction, train-only prototype build, validation-only calibration,
  frozen test prediction, evaluation, leakage audit, and qualification gate.
- Aggregate provenance: `run_scope=aggregate`, `screening_result=true`,
  `final_25_run_result=false`, `paper_candidate_result=true`,
  `paper_main_result=false`, `n_runs=15`.
- Method Macro-F1 mean/std: raw Softmax `0.950244/0.013218`, calibrated
  Softmax `0.950244/0.013218`, prototype `0.952333/0.012754`,
  hierarchical `0.955499/0.010793`, fused `0.954289/0.009766`.
- Paired statistics: hierarchical - raw Softmax mean delta `0.005256`,
  bootstrap CI95 `[0.000026, 0.012109]`, Wilcoxon p `0.055808`,
  wins/ties/losses `9/4/2`; hierarchical - prototype mean delta `0.003166`,
  CI95 `[-0.000003, 0.006741]`, Wilcoxon p `0.176296`,
  wins/ties/losses `5/8/2`.
- Feeding/stress confusion: hierarchical reduced stress-to-feeding errors from
  `55` under raw Softmax to `43`; feeding-to-stress was `69` versus raw
  Softmax `70`.
- Leakage and qualification: all 15 runs used `training_exact`, `frozen_test`,
  manifest SHA verification, leakage audit ok, and
  `eligible_for_cv_aggregation=true`; no unsafe checkpoint SHA mismatch was
  used.
- Paper usability: usable as screening evidence and a paper candidate aggregate,
  not as final paper-main result. Do not claim statistical significance because
  Wilcoxon p is `0.055808`.
- Scope guard: no lambda 1.0 comparison, no 25-run final CV, and no
  unknown/open-set experiment was started.


## 2026-06-25 - Lambda 0.5 exact prototype 5x5 final aggregate

- Status: completed
- Branch: `prototype-cv5-final-w05`
- Task: Added the approved missing exact prototype runs for seeds `123` and `777` across folds `0,1,2,3,4`, then aggregated all 25 lambda 0.5 fold-seed runs.
- Scope: `feature_backend=training_exact`, `selection_method=hierarchical`, `target_coverage=0.95`; no lambda 1.0, noise robustness, unknown/open-set, or paper writing was started.
- Final aggregate provenance: `run_scope=aggregate`, `screening_result=false`, `final_25_run_result=true`, `paper_candidate_result=true`, `paper_main_result=true`, `n_runs=25`.
- Method Macro-F1 mean/std: raw Softmax `0.951050/0.012393`, calibrated Softmax `0.951050/0.012393`, prototype `0.951852/0.013104`, hierarchical `0.953986/0.012004`, fused `0.953486/0.010343`.
- Paired statistics: hierarchical - raw Softmax mean delta `0.002937`, CI95 `[-0.000478, 0.007294]`, Wilcoxon p `0.058253`, wins/ties/losses `13/8/4`; hierarchical - prototype mean delta `0.002134`, CI95 `[-0.000499, 0.004996]`, Wilcoxon p `0.221330`, wins/ties/losses `8/12/5`.
- Fold deltas: four of five folds had positive mean hierarchical - raw Softmax; fold0 was slightly negative.
- Feeding/stress confusion: hierarchical feeding->stress `117`, stress->feeding `76`; raw Softmax feeding->stress `110`, stress->feeding `95`.
- Leakage and qualification: 25/25 runs passed exact Softmax reproduction, reference path/y_true/y_pred matching, train/val/test path/source_id/MD5 disjointness, manifest SHA verification, leakage audit, and CV aggregation eligibility. No unsafe checkpoint SHA mismatch was used.
- Outputs: final aggregate CSV/provenance files plus by-fold, calibration-distribution, and selective-risk summaries under `reports/prototype_cv5_exact_w05_final_*`.
- Paper usability: final paper-main candidate aggregate, but the hierarchical gain over raw Softmax remains nonsignificant; do not claim significance.

## 2026-06-25 - DEMAND simulated noise robustness 1x1 debug

- Status: completed
- Branch: `codex/demand-noise-debug`
- Task: Prepared DEMAND noise provenance, implemented deterministic SNR mixing and
  frozen exact prototype noise robustness evaluation, then ran the approved
  fold0/seed3407 lambda 0.5 debug on `DWASHING` at 20 dB and 10 dB.
- Scope: DEMAND only; no MUSAN, ESC-50, UrbanSound8K, lambda 1.0, 0 dB,
  15-run screening, or real-farm external validation was started.
- Noise source decision: selected `DWASHING`, `TBUS`, and `STRAFFIC` for the
  planned protocol; debug used only `DWASHING`. Fixed channel policy is
  `ch01.wav` / `selected_channel=1`, and synchronized DEMAND channels are not
  treated as independent recordings.
- Provenance: DOI `10.5281/zenodo.1227121`; Zenodo metadata observed
  `cc-by-4.0`; DEMAND PDF text observed Creative Commons
  Attribution-ShareAlike 3.0 Unported. Both are recorded.
- Clean equivalence: passed with 168/168 path matches, 168/168 `y_true` matches,
  168/168 raw Softmax `y_pred` matches, and Macro-F1
  `0.946360153256705` exactly reproduced.
- Debug Macro-F1: clean raw/prototype/hierarchical
  `0.946360/0.946360/0.952273`; DWASHING 20 dB
  `0.660317/0.693259/0.688191`; DWASHING 10 dB
  `0.599470/0.618750/0.619464`.
- SNR audit: max active-region SNR absolute error
  `4.6514175444656303e-07` dB; full-window SNR differs by design because clean
  RMS uses the valid non-padding region.
- Leakage and qualification: train/val/test path, source_id, and md5
  disjointness passed; `feature_backend=training_exact`,
  `noise_protocol=zero_shot_frozen`, `real_farm_external_validation=false`,
  `run_scope=fold_seed`, `paper_main_result=false`.
- Outputs: DEMAND manifest/provenance, local reviewed provenance template,
  two noise protocol docs, new noise robustness scripts/tests, and structured
  debug CSV/JSON under `reports/prototype_noise_demand_w05_fold0_seed3407_debug`.
  Downloaded ZIP/WAV/PDF/cache/audio assets were not committed.
- Paper usability: pipeline/debug evidence only. Recommend 5 folds x 3 seeds
  DEMAND screening after explicit review approval.

## 2026-06-25 - DEMAND prescreen protocol fixes and 1x1 full-grid smoke

- Status: completed
- Branch: `codex/demand-noise-prescreen-fix`
- Approval state: full 15-run screening remained `APPROVED: false`; no 15-run
  screening was started.
- Task: Fixed prescreen blockers for DEMAND simulated noise robustness, including
  seed/SNR-independent noise offsets, method-specific clean-validation
  thresholds, strengthened SHA/provenance gates, selected-channel consistency,
  active-event SNR reporting, full per-class/confusion outputs, and hardened
  aggregate screening protocol checks.
- Smoke run: fold0/seed3407, lambda 0.5, `DWASHING`, `TBUS`, `STRAFFIC`, active
  SNR `20/10/0` dB, `noise_repeat=0`, zero-shot frozen.
- Clean equivalence: passed with 168/168 path, y_true, and raw Softmax y_pred
  matches; Macro-F1 exactly reproduced at `0.946360153256705`.
- Provenance gates: checkpoint, prototype bundle, calibration JSON, test
  manifest, noise manifest, DEMAND provenance JSON, selected noise file SHA,
  noise-vs-pig MD5 disjointness, and selected channel all verified true.
- Same offset proof: `same_offset_across_snr_verified=true` over 504
  clean/environment/repeat draw groups; unit tests verify same offset across SNR
  and model seed, with different environment/repeat changing the offset.
- Method-specific clean-validation thresholds: raw Softmax global `0.862944`,
  prototype `0.827906`, hierarchical `0.853217`.
- Smoke Macro-F1: clean raw/prototype/hierarchical
  `0.946360/0.946360/0.952273`; DWASHING 20
  `0.656321/0.665814/0.661102`; DWASHING 10
  `0.604815/0.614662/0.620090`; DWASHING 0
  `0.594665/0.605114/0.600000`; TBUS 20
  `0.614045/0.614045/0.619464`; TBUS 10
  `0.599470/0.604409/0.604815`; TBUS 0
  `0.563761/0.576143/0.564466`; STRAFFIC 20
  `0.594616/0.604815/0.599940`; STRAFFIC 10
  `0.533141/0.553182/0.540000`; STRAFFIC 0
  `0.377213/0.363202/0.355094`.
- Per-class warning: cough F1 was `0.0` for all methods across all three 0 dB
  active-event SNR conditions, so 0 dB should be framed as severe stress testing
  rather than expected deployment performance.
- SNR audit: max active-region SNR error `4.918568325962269e-07` dB; DWASHING
  cough 20 dB active mean `20.000000` while full-window mean was `12.920643`
  because cough valid-duration ratio averaged `0.234185`.
- Outputs: updated DEMAND manifest/provenance, noise protocol docs, smoke
  metrics/confusion/SNR/provenance CSV/JSON, and hardened smoke/aggregate
  summaries under `reports/prototype_noise_demand_w05_fold0_seed3407_prescreen_smoke`.
- Recommendation: protocol is ready for review before a fixed 15-run DEMAND
  screening, but do not start it without explicit `APPROVED: true`.

## 2026-06-25 - DEMAND lambda 0.5 fixed 15-run screening

- Status: completed
- Branch: `codex/demand-noise-screening-w05`
- Approval state: current user attachment was `APPROVED: true`, conditional on
  pre-run provenance patch. The stale `handoff/GPT_TO_CODEX.md` still says
  `APPROVED: false`; current explicit user instruction was followed.
- Pre-run patch: added `--run_stage prescreen_smoke|screening`; screening runs
  now record `run_stage=screening`, `run_scope=fold_seed`,
  `single_fold_debug=false`, `global_noise_seed=3407`, `noise_repeat=0`, and
  `offset_key_version=demand_noise_offset_v2`. Summarizer now enforces exact
  1 clean + 9 noisy condition grid, strict paired `n=15`, fixed DEMAND
  screening protocol, cross-seed draw audit, MODERATE_NOISE / EXTREME_STRESS /
  ALL_NOISY strata, and fold-cluster paired statistics.
- Verification before screening: all noise script `--help` commands passed,
  `py_compile` passed, `tests.test_noise_robustness_pipeline` passed with 29
  tests, `tests.test_prototype_pipeline_core` passed with 47 tests, and
  `git diff --check` passed.
- Screening scope: 5 folds x 3 seeds, lambda 0.5, `DWASHING`, `TBUS`,
  `STRAFFIC`, active-event SNR `20/10/0` dB, `global_noise_seed=3407`,
  `noise_repeat=0`, zero-shot frozen, no noisy validation or test tuning.
- Completion: 15/15 fold-seed runs completed; no run failed.
- Aggregate provenance: `run_scope=aggregate`, `screening_result=true`,
  `final_25_run_result=false`, `paper_candidate_result=true`,
  `paper_main_result=false`, `n_fold_seed_runs=15`,
  `simulated_noise=true`, `real_farm_external_validation=false`,
  `cross_seed_noise_draw_audit_ok=true`.
- Cross-seed draw audit: 2520 groups, all OK; same clean MD5/fold/environment
  uses identical draw ID, offset, noise SHA, selected channel, and global noise
  seed across seeds 42, 2024, and 3407.
- Mean/std Macro-F1:
  - MODERATE_NOISE raw/prototype/hierarchical/fused:
    `0.645442/0.027306`, `0.654077/0.024862`,
    `0.652557/0.027055`, `0.649658/0.025827`.
  - EXTREME_STRESS raw/prototype/hierarchical/fused:
    `0.556940/0.029653`, `0.565833/0.025803`,
    `0.557158/0.026776`, `0.561422/0.026679`.
  - ALL_NOISY raw/prototype/hierarchical/fused:
    `0.615941/0.025081`, `0.624662/0.022745`,
    `0.620757/0.023449`, `0.620246/0.022982`.
- ALL_NOISY paired stats:
  - prototype - raw: mean delta `0.008721`, p `0.002625`,
    wins/ties/losses `13/0/2`.
  - hierarchical - raw: mean delta `0.004816`, p `0.229309`,
    wins/ties/losses `9/0/6`.
  - hierarchical - prototype: mean delta `-0.003905`, p `0.000122`,
    wins/ties/losses `1/0/14`.
- ALL_NOISY aggregate class F1 from confusion:
  - raw: cough `0.087690`, calm `0.992294`, feeding `0.825939`,
    stress `0.598141`.
  - prototype: cough `0.098592`, calm `0.985739`, feeding `0.843309`,
    stress `0.606508`.
  - hierarchical: cough `0.096357`, calm `0.987542`, feeding `0.831763`,
    stress `0.605886`.
- Key interpretation: prototype-only is the most robust method in this
  simulated DEMAND screening. Hierarchical prototype is useful diagnostic
  evidence but does not beat prototype-only under noise. Cough remains the
  weakest class, with most cough errors going to `stress_vocal`.
- Outputs: `reports/prototype_noise_demand_w05_fold{fold}_seed{seed}_screening/`
  plus aggregate CSV/JSON files with prefix
  `reports/prototype_noise_demand_w05_screening`.
- Warnings: existing pandas DataFrame fragmentation warnings appeared during
  prediction output construction; no exit status or results were affected.
- Stop point: do not start lambda 1.0, noise-aware recalibration, noise
  augmentation training, local unlicensed noise, or real-farm validation without
  a new explicit approval.

## 2026-06-25 - DEMAND lambda 0.5 fixed 25-run final simulated-noise aggregate

- Status: completed
- Branch: `codex/demand-noise-final-w05`
- Base: `codex/demand-noise-screening-w05` at
  `45af0dc4ecb40182461f923d5206baead580af1d`
- Approval state: user explicitly provided `APPROVED: true`; before execution,
  `handoff/GPT_TO_CODEX.md` was updated to `APPROVED: true`.
- Task: Added `run_stage=final`, ran only the missing seeds `123` and `777`
  across folds `0,1,2,3,4`, and aggregated those 10 new runs with the existing
  15 screening runs into the fixed 5 folds x 5 seeds DEMAND final result.
- Scope: lambda `0.5`, `DWASHING/TBUS/STRAFFIC`, active-event SNR `20/10/0`,
  `global_noise_seed=3407`, `noise_repeat=0`, `selected_channel=1`,
  `feature_backend=training_exact`, zero-shot frozen. No lambda 1.0,
  noise-aware recalibration, noise augmentation, denoising, local reviewed
  noise, open-set, new SNR, new environment, or new repeat was started.
- Preflight: all 10 new fold-seed combinations had checkpoint, prototype
  bundle, calibration JSON, clean validation predictions, clean prediction
  metadata, reference `test_pred.csv`, train/val/test manifests, matching
  checkpoint/prototype/calibration SHA, test manifest SHA verification, DEMAND
  manifest/provenance, three noise files with matching SHA, selected channel 1,
  and noise/pig MD5 disjointness.
- Completion: 10/10 new final runs completed; 25/25 aggregate completeness
  verified.
- Aggregate provenance: `run_scope=aggregate`, `screening_result=false`,
  `final_25_run_result=true`, `paper_candidate_result=true`,
  `paper_main_result=true`, `simulated_noise_main_result=true`,
  `n_fold_seed_runs=25`, `simulated_noise=true`,
  `real_farm_external_validation=false`,
  `cross_seed_noise_draw_audit_ok=true`.
- Cross-seed draw audit: 2520 groups, all OK; seed set
  `42|123|777|2024|3407`.
- Mean/std Macro-F1:
  - MODERATE_NOISE raw/prototype/hierarchical/fused:
    `0.647225/0.025426`, `0.653354/0.023320`,
    `0.652175/0.025686`, `0.649915/0.024672`.
  - EXTREME_STRESS raw/prototype/hierarchical/fused:
    `0.557393/0.029311`, `0.562400/0.028306`,
    `0.555623/0.028425`, `0.560369/0.027277`.
  - ALL_NOISY raw/prototype/hierarchical/fused:
    `0.617281/0.022973`, `0.623036/0.021252`,
    `0.619991/0.022378`, `0.620066/0.021477`.
- ALL_NOISY paired stats:
  - prototype - raw: mean delta `0.005755`, CI95 `[0.001606, 0.010894]`,
    Wilcoxon p `0.010511`, wins/ties/losses `18/0/7`.
  - hierarchical - raw: mean delta `0.002710`, CI95
    `[-0.000999, 0.007085]`, Wilcoxon p `0.312333`,
    wins/ties/losses `13/0/12`.
  - hierarchical - prototype: mean delta `-0.003045`, CI95
    `[-0.004581, -0.001639]`, Wilcoxon p `0.000376`,
    wins/ties/losses `5/0/20`.
- ALL_NOISY class F1 from confusion:
  - raw: cough `0.086657`, calm `0.988748`, feeding `0.832270`,
    stress `0.600279`.
  - prototype: cough `0.097424`, calm `0.979829`, feeding `0.841560`,
    stress `0.607251`.
  - hierarchical: cough `0.095699`, calm `0.981712`, feeding `0.831944`,
    stress `0.607110`.
  - fused: cough `0.088961`, calm `0.987355`, feeding `0.838345`,
    stress `0.602811`.
- Key interpretation: prototype-only is the strongest simulated DEMAND noise
  method. Hierarchical prototype is above raw Softmax on ALL_NOISY but not
  significant and remains below prototype-only. Cough remains the weakest class,
  mostly failing into `stress_vocal`.
- Verification: related `--help` commands passed, `py_compile` passed,
  `tests.test_noise_robustness_pipeline` passed with 39 tests,
  `tests.test_prototype_pipeline_core` passed with 47 tests,
  `git diff --check` passed with line-ending warnings only, aggregate
  provenance gate passed, and cross-seed draw gate passed.
- Outputs: 10 new final run directories under
  `reports/prototype_noise_demand_w05_fold{fold}_seed{seed}_final/` plus
  aggregate CSV/JSON files under `reports/prototype_noise_demand_w05_final/`.
- Paper usability: paper-main candidate simulated-noise aggregate; not real-farm
  external validation.
- Stop point: do not start lambda 1.0, noisy recalibration, augmentation,
  denoising, local reviewed noise, open-set, new environments, new SNRs, or new
  repeats without new explicit approval.

## 2026-06-25 - SCI draft v1 paper package and corrected AUGRC

- Status: completed
- Branch: `paper/sci-draft-v1`
- Base: `codex/demand-noise-final-w05` at
  `c6de8d1541643da3d21d62032c4544f5d39cffae`
- Approval state: user explicitly provided `APPROVED: true`.
- Scope: experiment phase frozen. No new model training, inference,
  checkpoint, prototype, calibration, SNR, noise draw, or statistical protocol
  change was started.
- AUGRC correction: replaced the incorrect `cumsum(losses) / accepted_count`
  implementation with generalized risk
  `sum(loss_accepted) / total_sample_count`, integrated over coverage. The
  project stores unscaled 0..1 AUGRC values.
- Selective metrics: regenerated final selective summaries from existing
  frozen-test `noise_predictions.csv` files using
  `--recompute_selective_from_predictions`; no model inference was rerun.
- Corrected ALL_NOISY AURC/AUGRC:
  - raw_softmax: `0.177458 / 0.123881`
  - prototype: `0.210937 / 0.139362`
  - hierarchical: `0.212211 / 0.140298`
  - fused: `0.184808 / 0.127355`
- Macro-F1 unchanged:
  - ALL_NOISY raw/prototype/hierarchical/fused:
    `0.617281`, `0.623036`, `0.619991`, `0.620066`
  - MODERATE_NOISE raw/prototype/hierarchical/fused:
    `0.647225`, `0.653354`, `0.652175`, `0.649915`
  - EXTREME_STRESS raw/prototype/hierarchical/fused:
    `0.557393`, `0.562400`, `0.555623`, `0.560369`
- Paper package created under `paper/`:
  - bilingual manuscript drafts,
  - 9 main tables,
  - 8 main figures as SVG/PDF/300-dpi PNG,
  - appendix summaries,
  - references with 6 conservative verified entries,
  - reproducibility commands and claims/evidence notes.
- Paper guardrails: simulated DEMAND noise only; no real-farm external
  validation; no claim that hierarchical prototype is most robust; no claim of
  reliable cough recognition at 0 dB.
- Verification: AUGRC regression tests and summary recomputation tests added;
  full verification results are recorded in the final Codex response.
- Stop point: review the generated paper package. Do not start new experiments
  without a new explicit `APPROVED: true`.

## 2026-07-08 - Paper figures and Word-friendly tables V2

- Status: completed
- Branch: `paper/sci-draft-v1`
- Base: `a015ac2`
- Approval state: `handoff/GPT_TO_CODEX.md` contained `APPROVED: true`; the
  external task file
  `C:\Users\s1205\OneDrive\Desktop\论文\codex_paper_figures_and_tables_tasks.md`
  specified the figure/table generation scope.
- Scope: generated figures and Word-friendly table files only. No model
  training, inference, checkpoint/prototype/calibration construction, SNR/noise
  draw change, threshold tuning, or statistical protocol change was started.
- New script: `tools/generate_paper_figures_and_tables.py` validates inputs,
  generates all eight figures as SVG/PDF/300-dpi PNG, writes figure source
  manifests, and exports dependency-free Word-friendly table files.
- Documentation: added `paper/README_figures_tables.md` with the generation
  command and output descriptions.
- Figure outputs: regenerated
  `paper/figures/figure1_overall_method_architecture.{svg,pdf,png}` through
  `paper/figures/figure8_risk_coverage_curves.{svg,pdf,png}`.
- Table outputs: added `paper/tables/word_friendly/paper_main_tables.docx`,
  consolidated and per-table HTML files, and `table_sources.csv`.
- Source manifests: added `paper/figures/figure_data_sources.csv` and
  `paper/figures/figure_data_sources.json`.
- Metrics represented: duration Macro-F1 `1s=0.9226`, `2s=0.9472`,
  `3s=0.9434`, delta `0.0246`, p `0.000162`; ALL_NOISY
  raw/prototype/hierarchical Macro-F1 `0.6173/0.6230/0.6200`; ALL_NOISY
  raw/prototype/hierarchical AURC/AUGRC
  `0.1775/0.1239`, `0.2109/0.1394`, `0.2122/0.1403`; cough F1
  `0.0867/0.0974/0.0957`.
- Leakage audit: source result CSVs under `paper/tables/*.csv` and
  `paper/appendix/*.csv` have no diff; no audio, checkpoint, NPZ, cache,
  embedding, or prediction CSV file was added. Train/val/test boundaries and
  path/source_id/MD5 disjointness are unchanged because this task reads only
  existing aggregate paper files.
- Verification: generator `--help` passed, `py_compile` passed, full generation
  passed, DOCX `word/document.xml` parsed successfully, required caution labels
  were found in SVG/HTML outputs, and all eight PNGs were visually inspected.
- Paper usability: `paper_usable=true` for the figure/table package; this task
  does not create new scientific results and remains simulated-noise-only.
- Questions for GPT Pro: decide target-journal table format preference, whether
  Figure 1 needs a journal-specific visual style pass, and whether final
  captions need additional simulated-noise/0 dB caution wording.
- Stop point: review generated figures/tables and captions. Do not start new
  experiments or another paper stage without a new explicit `APPROVED: true`.

## 2026-07-09 - Full-story SCI manuscript rewrite with Nature Writing

- Status: completed
- Branch: `paper/sci-draft-v1`
- Base: `01853bf`
- Approval state: `handoff/GPT_TO_CODEX.md` contained `APPROVED: true`.
- Scope: used the installed `nature-writing` skill router, always-load files,
  selected research/full-manuscript/zh-to-en/generic fragments, and relevant
  writing references before drafting. No model training, inference, checkpoint,
  prototype, calibration, SNR, noise draw, threshold, raw audio, or result
  CSV/JSON change was started.
- Added manuscript files:
  `paper/manuscript/paper_cn_full_story.md`,
  `paper/manuscript/paper_en_full_story.md`,
  `paper/manuscript/section_plan.md`, and
  `paper/CLAIMS_AND_EVIDENCE_v2.md`.
- Story 1: reframed the clean mainline around event context rather than
  feature stacking. Locked evidence: 2 s Log-Mel Macro-F1 `0.9472` versus 1 s
  `0.9226`, mean delta `0.0246`, Wilcoxon p `0.000162`; 3 s `0.9434`; tabled
  MCTAFD/PCEN/spectral-gate/SpecAugment/attention variants did not surpass the
  2 s mainline.
- Story 2: reframed hierarchical auxiliary supervision as subtype semantics and
  boundary explanation, not a statistically significant main-performance
  contribution. Locked evidence: lambda=1.0 highest mean `0.9515`; lambda=0.5
  lowest std `0.0124`; clean hierarchical prototype `0.9540`; clean
  hierarchical - raw Softmax p `0.058253`.
- Story 3: reframed post-hoc prototypes as distinct clean semantic and
  simulated-noise inference mechanisms. Locked evidence: ALL_NOISY
  raw/prototype/hierarchical Macro-F1 `0.6173/0.6230/0.6200`; prototype - raw
  delta `0.005755`, CI95 `[0.001606, 0.010894]`, p `0.010511`; hierarchical -
  prototype delta `-0.003045`, p `0.000376`.
- Required boundaries written into the drafts: DEMAND is simulated additive
  noise only; no real-farm external validation; 0 dB active-event SNR is an
  extreme stress condition; cough is unreliable at 0 dB; prototypes do not
  comprehensively improve uncertainty; raw Softmax is better by AURC/AUGRC;
  prototype only slightly improves frozen-threshold selective risk; fused is an
  ablation.
- Leakage audit: source result CSVs and appendix CSVs were read only; no
  train/val/test roles changed; path/source_id/MD5 disjointness unchanged.
- Paper usability: `paper_usable=true` as a full-story manuscript draft grounded
  in existing paper outputs; `new_results_created=false`;
  `real_farm_external_validation=false`.
- Questions for GPT Pro: decide target journal and word limits, provide
  verified livestock-acoustic citations, authorize ethics/data/code
  availability statements, and choose whether English or Chinese is the
  authoritative submission source.
- Stop point: review and condense the full-story drafts. Do not start new
  experiments or another paper stage without a new explicit `APPROVED: true`.

## 2026-07-09 - Nature Figure V2 manuscript figure generation

- Status: completed
- Branch: `paper/sci-draft-v1`
- Base: `d481d98`
- Approval state: `handoff/GPT_TO_CODEX.md` contained `APPROVED: true`.
- Scope: used the installed `nature-figure` skill, loaded `SKILL.md`,
  `manifest.yaml`, always-load files, and the Python/matplotlib backend
  fragment. Python/matplotlib was used as the backend. No AI-generated
  schematic, OpenRouter call, model training, inference, checkpoint,
  prototype, calibration, SNR/noise draw, threshold, raw audio, or result
  CSV/JSON change was started.
- Added generator: `tools/generate_paper_figures_v2.py`.
- Added contracts: `paper/figures_v2/figure_contracts.md`.
- Generated exports: Figures 1-8 under `paper/figures_v2/`, each as editable
  SVG, PDF, and 300 dpi PNG.
- Source CSVs used by the generator:
  `paper/tables/table1_dataset_and_leakage_free_protocol.csv`,
  `paper/tables/table3_duration_comparison.csv`,
  `paper/tables/table4_hierarchical_aux_weight_ablation.csv`,
  `paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv`,
  `paper/tables/table6_simulated_noise_robustness_by_stratum.csv`,
  `paper/tables/table7_paired_statistics.csv`,
  `paper/tables/table8_per_class_noise_results_and_confusion_directions.csv`,
  `paper/tables/table9_selective_prediction_metrics_corrected_augrc.csv`,
  `reports/prototype_noise_demand_w05_final/final_summary.csv`, and
  `reports/prototype_noise_demand_w05_final/final_per_class_summary.csv`.
- Non-CSV source file used: `paper/CLAIMS_AND_EVIDENCE.md`.
- Metrics represented: duration Macro-F1 `1s=0.9226`, `2s=0.9472`,
  `3s=0.9434`, delta `0.0246`, p `0.000162`; hierarchy lambda means
  `0.9505/0.9510/0.9515`; clean raw/calibrated/prototype/hierarchical/fused
  Macro-F1 `0.9510/0.9510/0.9519/0.9540/0.9535`; MODERATE_NOISE
  raw/prototype/hierarchical `0.6472/0.6534/0.6522`; EXTREME_STRESS
  `0.5574/0.5624/0.5556`; ALL_NOISY `0.6173/0.6230/0.6200`; ALL_NOISY cough
  F1 `0.0867/0.0974/0.0957`; raw/prototype/hierarchical AURC/AUGRC
  `0.1775/0.1239`, `0.2109/0.1394`, `0.2122/0.1403`.
- Required boundaries encoded in contracts/figures: DEMAND is simulated noise
  only; no real-farm external validation; 0 dB is an extreme active-event SNR
  stress condition; cough is unreliable under 0 dB; hierarchical prototype is
  not the noise-optimal method; prototypes do not comprehensively improve
  uncertainty; raw Softmax ranks better by AURC/AUGRC.
- Leakage audit: no source result CSV/JSON was modified; no raw audio was read
  or modified; no training, validation, or test role changed;
  path/source_id/MD5 disjointness unchanged.
- Verification: generator `--help` passed; full generation passed; all eight
  PNG previews passed dimension and nonblank extrema checks; all eight PNGs
  were visually inspected and layout issues were corrected before final export.
- Paper usability: `paper_usable=true` for a manuscript figure package;
  `new_results_created=false`; `real_farm_external_validation=false`.
- Blockers: none for the requested figures.
- Questions for GPT Pro: decide target-journal-specific figure typography,
  final caption boundary wording, and whether to prepare a separate source-data
  archive from the same locked CSV inputs.
- Stop point: review generated figures and integrate final captions. Do not
  start new experiments or another paper stage without a new explicit
  `APPROVED: true`.

## 2026-07-09 - Nature Data availability, code, ethics, and reproducibility package

- Status: completed
- Branch: `paper/sci-draft-v1`
- Base: `855c9d7`
- Approval state: `handoff/GPT_TO_CODEX.md` contained `APPROVED: true`.
- Scope: used the installed `nature-data` skill, loaded `SKILL.md`,
  `manifest.yaml`, all always-load files, and relevant on-demand references for
  policy principles, repository/identifier strategy, statement patterns, FAIR
  metadata, Chinese-author alignment, and source-basis mapping.
- Added statements:
  `paper/reproducibility/DATA_AVAILABILITY.md`,
  `paper/reproducibility/CODE_AVAILABILITY.md`, and
  `paper/reproducibility/ETHICS_STATEMENT.md`.
- Added reproducibility files:
  `paper/reproducibility/REPRODUCIBILITY_CHECKLIST.md` and
  `paper/reproducibility/DATA_SOURCE_AUDIT.csv`.
- DATA_SOURCE_AUDIT rows: clean pig vocalization audio, auxiliary subtype
  labels, cap3x CV manifests, model checkpoints, prototype artifacts, DEMAND
  noise audio, DEMAND noise metadata, generated tables, generated appendix
  CSVs, generated figures, code repository, non-redistributed audio policy, and
  SHA provenance.
- Formally citable now: DEMAND DOI `10.5281/zenodo.1227121` and Zenodo record
  `https://zenodo.org/records/1227121`; code repository URL
  `https://github.com/HinataQAQ/pig-sound-classification`.
- Still unresolved: clean pig audio DOI/URL/license/owner/access route,
  auxiliary subtype label redistribution rights, source-data archive DOI and
  license, code release tag/software DOI/license, checkpoint archive plan, and
  prototype NPZ artifact archive plan.
- Ethics blocker: original animal collection ethics and permissions remain
  unresolved. The exact required fallback sentence is included:
  "This manuscript uses previously collected or public audio data; the original
  collection ethics and permissions must be confirmed before submission."
- Required boundaries encoded: DEMAND audio should be cited through DOI/Zenodo
  and not redistributed in GitHub; DEMAND license metadata conflict is retained;
  simulated DEMAND additive noise is not real-farm external validation; 0 dB is
  an extreme active-event SNR stress condition, not no noise.
- Metrics preserved: 2 s Log-Mel Macro-F1 `0.9472`; 1 s `0.9226`; paired delta
  `0.0246`; p `0.000162`; lambda=1.0 highest mean `0.9515`; lambda=0.5 std
  `0.0124`; clean hierarchical prototype `0.9540` with p `0.058253`; ALL_NOISY
  raw/prototype/hierarchical `0.6173/0.6230/0.6200`; prototype - raw delta
  `0.005755`, CI95 `[0.001606, 0.010894]`, p `0.010511`.
- Leakage audit: no training, inference, checkpoint, prototype, calibration,
  threshold, SNR/noise draw, prediction, source result CSV/JSON, raw clean
  audio, or DEMAND audio change was performed. Final diff does not modify
  `paper/tables/*.csv`, `paper/appendix/*.csv`, `paper/figures/*`,
  `paper/reproducibility/commands.md`, or
  `paper/reproducibility/paper_package_manifest.json`.
- Verification: `DATA_SOURCE_AUDIT.csv` parsed with 13 rows; keyword audit
  found required unresolved fields, DEMAND DOI, ethics fallback, checkpoint,
  prototype, SHA and MD5 language; all five new files contain zero non-ASCII
  characters; `tools/summarize_prototype_noise_robustness.py --help` passed.
  `create_paper_package.py --help` is not true argparse help and executed the
  package script; accidental figure regeneration was restored to HEAD.
- Paper usability: `paper_usable=true`; `new_results_created=false`;
  `source_csv_json_values_modified=false`; `simulated_noise_only=true`;
  `real_farm_external_validation=false`.
- Questions for GPT Pro: decide final clean-audio access route, provide exact
  ethics authority/approval number or confirm accepted fallback wording, decide
  checkpoint/prototype artifact release policy, and choose repository/DOI route
  for processed source data and code.
- Stop point: resolve clean-audio rights, ethics, and archival DOI fields. Do
  not start new experiments or another paper stage without a new explicit
  `APPROVED: true`.

## 2026-07-09 - Nature polishing of English full-story manuscript

- Status: completed
- Branch: `paper/sci-draft-v1`
- Base: `5bf1ced`
- Approval state: `handoff/GPT_TO_CODEX.md` contained `APPROVED: true`, and
  the user explicitly requested the polishing task.
- Scope: used the installed `nature-polishing` skill, loaded `SKILL.md`,
  `manifest.yaml`, all always-load files, and the matching fragments for
  `paper_type=algorithmic`, all manuscript sections, `language=en`, and
  `journal=generic`.
- Added polished manuscript:
  `paper/manuscript/paper_en_full_story_polished.md`.
- Added audit outputs:
  `paper/manuscript/polishing_change_log.md` and
  `paper/manuscript/overclaim_audit.md`.
- Main language work: SCI applied-engineering English; British spelling;
  shorter sentence structure; cautious claim verbs; clearer
  claim-evidence-boundary logic; no new references; no number changes; no
  deletion of limitations.
- Required boundaries preserved: 2 s context significantly improves clean
  classification; hierarchical auxiliary supervision is positive but not
  statistically significant over the 2 s baseline; clean hierarchical prototype
  has the highest Macro-F1; under simulated DEMAND noise, main prototype is the
  most robust; hierarchical prototype is not the most noise-robust method;
  prototype inference does not universally improve uncertainty; 0 dB
  active-event SNR is an extreme stress condition; real-farm external validation
  was not performed.
- Metrics preserved: 1 s/2 s Log-Mel Macro-F1 `0.9226/0.9472`, delta
  `0.0246`, p `0.000162`; 3 s Log-Mel `0.9434`; hierarchical lambda means
  `0.9505/0.9510/0.9515`; lambda=0.5 std `0.0124`; clean raw/main/hierarchical
  prototype Macro-F1 `0.9510/0.9519/0.9540`; clean hierarchical - raw p
  `0.058253`; ALL_NOISY raw/main/hierarchical `0.6173/0.6230/0.6200`; main -
  raw delta `0.005755`, CI95 `[0.001606, 0.010894]`, p `0.010511`;
  hierarchical - raw p `0.312333`; hierarchical - main p `0.000376`;
  ALL_NOISY raw/main/hierarchical AURC/AUGRC `0.1775/0.1239`,
  `0.2109/0.1394`, and `0.2122/0.1403`.
- Overclaim risks found: title/abstract could overgeneralise hierarchy;
  positive hierarchical means could imply significance; clean hierarchical
  prototype could be confused with noise robustness; main-prototype noise gains
  could be overgeneralised to real farms; selective-risk behaviour could be
  overgeneralised into uncertainty calibration.
- Evidence still needed: verified livestock-acoustics references; clean audio
  rights and licence; code release tag/DOI/licence; animal ethics approval
  details; real-farm external validation if deployment claims are desired; new
  evidence for reliable cough recognition under 0 dB if that claim is desired.
- Leakage audit: no training, inference, checkpoint, prototype, calibration,
  threshold, SNR/noise draw, prediction, source result CSV/JSON, raw clean
  audio, or DEMAND audio change was performed. No `paper/tables/*.csv`,
  `paper/appendix/*.csv`, prediction file, or source result file was modified.
- Verification: forbidden-word scan for `prove`, `proves`, `proved`, `proven`,
  `novel`, and `first` returned no matches; numeric-token audit matched 168
  source tokens and 168 polished tokens with no additions or removals;
  sentence-length audit found zero prose sentences above 30 words; ASCII audit
  found zero non-ASCII characters in all three new manuscript files.
  `tools/summarize_hier_longcontext.py --help` is not conventional help and
  executed the summary script, but git showed no diff in the rewritten report
  CSVs.
- Paper usability: `paper_usable=true` for English manuscript polishing;
  `submission_complete=false`; `new_results_created=false`;
  `source_csv_json_values_modified=false`; `simulated_noise_only=true`;
  `real_farm_external_validation=false`.
- Questions for GPT Pro: choose final Related Work citations; decide whether to
  shorten the title; confirm British vs US spelling for the final journal; fill
  legally accurate ethics, data-rights, and code-release statements.
- Stop point: review the polished English manuscript and fill evidence gaps. Do
  not start new experiments or another paper stage without a new explicit
  `APPROVED: true`.

## 2026-07-09 - Nature Citation reference expansion and claim audit

- Status: completed
- Branch: `paper/sci-draft-v1`
- Base: `1808464`
- Approval state: `handoff/GPT_TO_CODEX.md` contained `APPROVED: true`, and
  the user explicitly requested the citation-audit task.
- Scope: used the installed `nature-citation` skill, loaded `SKILL.md`,
  `manifest.yaml`, all always-load files, plus search-strategy and
  long-article/script-usage references. Broader academic search was used as
  allowed by the user because the domain citations should not be limited to
  Nature/CNS.
- Updated `paper/references/references.bib`: bibliography expanded from 6 to
  30 entries, with 24 newly added verified references.
- Added `paper/references/citation_audit.csv`: 32 claim rows with claim ID,
  claim text, required support type, current citation, missing-citation flag,
  candidate search query, recommended citation keys, support strength and audit
  notes.
- Added `paper/references/missing_citations.md`: strongest support by core
  claim group, rejected/not-used candidates, suggested insertion points and
  unresolved data-source/ethics/code gaps.
- Main added support: pig call valence/context (`Briefer 2022`, `Tallet 2013`,
  `Illmann 2013`, `Weary 1995`), pig cough recognition (`Ferrari 2008`,
  `Exadaktylos 2008`, `Yin 2021`, `Shen 2022`), noisy pig-vocalization
  classification (`Chung 2025`, `Xie 2024`), CRNN/sound-event classification,
  animal-sound features, PCEN, SpecAugment, hierarchical classification and
  audio prototype interpretability.
- Required boundaries preserved: 2 s remains an internal empirical result, not
  a universal literature-standard duration; DEMAND remains simulated additive
  noise, not real-farm external validation; reviews are background only; no
  prototype-learning novelty claim was introduced.
- Metrics preserved: 1 s/2 s Log-Mel Macro-F1 `0.9226/0.9472`, delta
  `0.0246`, p `0.000162`; 3 s Log-Mel `0.9434`; hierarchical lambda means
  `0.9505/0.9510/0.9515`; clean hierarchical - raw p `0.058253`; ALL_NOISY
  raw/main/hierarchical `0.6173/0.6230/0.6200`; main - raw delta `0.005755`,
  CI95 `[0.001606, 0.010894]`, p `0.010511`.
- Leakage audit: no training, inference, checkpoint, prototype, calibration,
  threshold, SNR/noise draw, prediction, source result CSV/JSON, raw clean
  audio, or DEMAND audio change was performed. No `paper/tables/*.csv`,
  `paper/appendix/*.csv`, prediction file, or source result file was modified.
- Verification: `references.bib` has 30 entries; `citation_audit.csv` parsed
  successfully with 32 rows; ASCII audit found zero non-ASCII characters in
  the three reference audit files; git diff was reviewed for target files.
- Paper usability: `paper_usable=true` for citation expansion/audit;
  `submission_complete=false`; `new_results_created=false`;
  `source_csv_json_values_modified=false`; `simulated_noise_only=true`;
  `real_farm_external_validation=false`.
- Remaining blockers: clean pig audio source/license/access, animal ethics
  details, code release tag/DOI/license, direct external support for exactly
  2 s pig-vocalization windows, formal spectral-gate citation if retained, and
  MCTAFD external citation/internal definition if retained.
- Stop point: review `citation_audit.csv` and insert approved inline citations
  into the manuscript. Do not start new experiments or another paper stage
  without a new explicit `APPROVED: true`.

## 2026-07-09 - Clean pig audio source recovery audit

- Status: completed.
- Branch: `paper/sci-draft-v1`.
- Approval state: `handoff/GPT_TO_CODEX.md` contained `APPROVED: true`, and the user explicitly requested only clean pig-audio provenance recovery.
- Scope: local evidence first, then web/source recovery. No model experiment, inference, training/evaluation edit, paper-result edit, raw-audio edit, Data Availability edit, paper-claim edit, or deletion was performed.
- Added `paper/reproducibility/PIG_AUDIO_SOURCE_RECOVERY_REPORT.md`.
- Added `paper/reproducibility/PIG_AUDIO_SOURCE_AUDIT.csv` with 14 source rows and the requested provenance/right columns.
- Added `paper/reproducibility/UNRESOLVED_DATA_SOURCES.md`.
- Updated `handoff/CODEX_TO_GPT.md` and `handoff/CODEX_TO_GPT.json` for this round.
- Verified current-mainline sources: `data/external/korea_raw/dry_cough` and `abdominal_cough` map to Smart Farm Korea / data.go.kr pig cough sound, high confidence; `data/raw/sow_call_dataset_labeled/*` maps to figshare Sow call dataset, CC BY 4.0, high confidence.
- Korea cough source identified: yes, high confidence.
- Verified non-current sources: SoundWel Zenodo 8252482, CC BY 4.0; aSwine GitHub, CC BY-NC 4.0; `data/raw/wu_pig_speech` is an exact local duplicate of figshare Sow call dataset.
- Unresolved/highest-risk source: `data/raw/scream_cough_small`, likely Kaggle `titpigrecognition/porcine-scream-sounds-and-cough-sounds`, but Kaggle metadata reports license `Unknown`; classification `E. unresolved_do_not_submit_as_final`.
- Current manifest audit metadata: 7,310 total cap3x fold/split rows; 3,812 unique paths; 0 blank md5; 0 blank source_id.
- Fold0 leakage checks passed read-only: cross-split MD5 duplicate groups 0; exact path overlap 0; source_id overlap 0.
- Final recommendation: `safe_only_after_source_citation_added`.
- Paper usability: `paper_usable=true` for source-provenance audit; `submission_complete=false`; source citations, license wording, Data Availability wording, and third-party ethics/exemption language still require author/GPT Pro judgment.
- Stop point: review the new source recovery files and decide citations/Data Availability wording. Do not start another paper or experiment stage without a new explicit `APPROVED: true`.

## 2026-07-09 - Source citation and statement synchronization

- Status: completed.
- Branch: `paper/sci-draft-v1`.
- Approval state: `handoff/GPT_TO_CODEX.md` contained `APPROVED: true`, and the user explicitly requested only data-source citation completion and statement updates.
- Scope: synchronized the clean pig-audio provenance audit into the paper submission package. No model experiment, inference, training/evaluation code edit, experimental result CSV/JSON edit, raw-audio edit, or paper core-claim change was performed.
- Updated `paper/references/references.bib`: DEMAND retained; added `sow_call_dataset_2021` for figshare DOI `10.6084/m9.figshare.16940389`; added `smartfarmkorea_pig_cough_voice` with the audited data.go.kr and Smart Farm Korea URLs and no fabricated DOI.
- Updated `paper/reproducibility/DATA_AVAILABILITY.md`: raw third-party pig audio is not redistributed; shareable code, commands, aggregate tables, figures, manifests, hash-based leakage audits, and provenance are distinguished from provider-controlled raw audio; DEMAND, Sow Call, and Korea sources are cited from audited identifiers.
- Updated `paper/reproducibility/ETHICS_STATEMENT.md`: no new animal experiment was conducted; pig audio is reused from public or third-party sources; original source-side ethics/permissions are attributed to original providers where available; unresolved permission items remain listed before submission.
- Added `paper/reproducibility/SOURCE_CITATION_STATUS.md`: DEMAND and Sow Call are `citation_complete`; Korea is `citation_partial`; Kaggle-like scream/cough is `unresolved_do_not_submit_until_fixed`; SoundWel/aSwine/placeholders are `excluded_from_main_results`.
- Updated `paper/manuscript/paper_en_full_story_polished.md` and `paper/manuscript/paper_cn_full_story.md` only in the data-source paragraph plus Data Availability/Ethics sections.
- Metrics preserved: no scientific result numbers, tables, figures, abstracts, or core conclusions were changed.
- Leakage/data boundaries: train/validation/test roles unchanged; no test-set tuning; no fold/seed artifacts constructed; existing path/source_id/MD5 audit statements preserved.
- Verification: `references.bib` parsed with 32 entries and required dataset keys; Data Availability DOI/URL guard passed; no result CSV/JSON diffs; no Python/training/evaluation code diffs; Korea not marked `citation_complete`; Kaggle-like source not marked verified; `git diff --check` had only line-ending warnings.
- Paper usability: `paper_usable=true`; `submission_complete=conditional`; manuscript is safer after these updates if raw third-party audio is not redistributed and the Korean provider-terms check remains explicit or is finalized.
- Remaining blockers: final Korea provider-terms confirmation before any raw-audio archive, final code/results release tag or DOI, target-journal ethics/exemption wording, and unresolved Kaggle-like source exclusion.
- Stop point: review synchronized citation/status wording. Do not start another stage without a new explicit `APPROVED: true`.
