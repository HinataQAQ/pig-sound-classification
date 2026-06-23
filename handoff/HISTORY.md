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
