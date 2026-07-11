# Final Validation Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for code changes and superpowers:verification-before-completion before each completion claim.

**Goal:** Produce the locked validation-only lambda sensitivity audit, validation-selected B2/B3 framework evaluation, prototype prediction-value analysis, convergence/stability audit, preliminary audit figures, and final evidence report without retraining or changing validated artifacts.

**Architecture:** A new read-only generator validates and hashes frozen sources, builds the fold-wise selection table, aggregates matched fold-seed results, computes functional and stability metrics, exports five Python/matplotlib audit figures, and writes the final report. Missing selected-lambda B3 artifacts are produced separately by the established five-command exact prototype pipeline (Softmax reproduction, prototype build, validation calibration, frozen-test prediction, and evaluation) using existing checkpoints and validation-only calibration in unique, overwrite-protected directories.

**Tech Stack:** Python 3 in `pigsound-gpu`; pandas, NumPy, SciPy, scikit-learn, matplotlib; `unittest`; Windows PowerShell.

## Global Constraints

- Candidate lambdas are exactly `0.2`, `0.5`, and `1.0`; folds are `0-4`; seeds are `42,123,777,2024,3407`.
- Selection uses five-seed mean `best_val_macro_f1`, then validation SD when means differ by at most `1e-6`, then the smallest lambda. Test values never participate.
- No backbone training, new grids, test-set tuning, source substitution, artifact overwrite, manuscript regeneration, or edits to validated result/checkpoint/manuscript/figure/training files.
- B3 prototypes use the matching checkpoint, train-only prototypes, that fold's validation split for existing-protocol calibration, and the frozen test split once.
- All performance comparisons are strictly matched by `(fold, seed)` and report n, means/SDs, paired mean delta, 10,000-resample percentile CI, two-sided Wilcoxon raw P, Holm P across the seven requested framework contrasts, wins/ties/losses, five fold deltas, and five-fold cluster-bootstrap CI.
- Main/subtype distances use stored cosine distances. Favorable margin is `nearest wrong prototype distance - true prototype distance`; error-detection score is its negative. No test-derived threshold is fitted.
- B3 `best_epoch` and validation metric inherit the selected B2 checkpoint and must be labelled as inherited, because post-hoc prototype inference has no training epoch.
- Epoch curves are emitted only if every required run has complete epoch-level train-loss and validation-Macro-F1 logs; missing curves are never reconstructed.
- Figure backend is Python only. Export every required figure as editable-text SVG, TrueType PDF, and 300-dpi PNG.
- Preserve all pre-existing untracked files and refuse to overwrite any new selected-prototype directory or final-validation output.
- Final report must state that this is a locked validation-only sensitivity analysis, not fully prospective or fully confirmatory, and must use the representative wording: “a training sample closest to the predicted class prototype”.

---

### Task 1: Test-first statistical and provenance core

**Files:**
- Create: `tests/test_generate_final_validation_audit.py`
- Create: `tools/generate_final_validation_audit.py`

**Interfaces:**
- `select_lambda_by_fold(records: pd.DataFrame, tie_tolerance: float = 1e-6) -> tuple[pd.DataFrame, dict[int, float]]`
- `holm_adjust(p_values: Sequence[float]) -> np.ndarray`
- `paired_statistics(runs: pd.DataFrame, comparisons: Sequence[tuple[str, str]], n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]`
- `sha256_file(path: Path) -> str`
- `refuse_existing(paths: Iterable[Path]) -> None`

- [ ] **Step 1: Write failing tests.** Cover mean-first selection, SD tie-break, smallest-lambda tie-break, the exact 25-key pairing gate, Holm monotonicity/order restoration, deterministic bootstrap output, all-zero Wilcoxon handling, missing/non-finite validation metrics, and overwrite refusal.
- [ ] **Step 2: Run the focused test and confirm RED.**

```powershell
conda activate pigsound-gpu
$env:PYTHONNOUSERSITE="1"
python -m unittest tests.test_generate_final_validation_audit -v
```

Expected: import failure because `tools.generate_final_validation_audit` does not yet exist.

- [ ] **Step 3: Implement the minimum core.** Use sample SD (`ddof=1`), `numpy.random.default_rng(3407)`, `scipy.stats.wilcoxon`, stable sorting, `np.isclose(..., atol=1e-12, rtol=0)` for W/T/L, and strict finite-value checks.
- [ ] **Step 4: Re-run the focused test and confirm GREEN.** Expected: every Task 1 test passes.

### Task 2: Frozen artifact loading and selected B3 reconstruction

**Files:**
- Modify: `tests/test_generate_final_validation_audit.py`
- Modify: `tools/generate_final_validation_audit.py`
- Create generated directories only when absent: `reports/prototype_cv5_exact_valsel_w02_fold{1,2,3}_seed{seed}/`
- Create generated directories only when absent: `reports/prototype_cv5_exact_valsel_w10_fold0_seed{seed}/`

**Interfaces:**
- `load_lambda_selection_sources(root: Path) -> tuple[pd.DataFrame, dict[str, str]]`
- `load_validation_selected_runs(root: Path, selected: Mapping[int, float]) -> tuple[pd.DataFrame, list[Path]]`
- `validate_prototype_run(root: Path, run_dir: Path, fold: int, seed: int, expected_lambda: float, summary_path: Path) -> dict[str, Any]`

- [ ] **Step 1: Add failing fixture tests.** Build temporary B0/B1/hierarchy summaries and prototype metadata/metrics/predictions; require exact fold/seed/lambda, matching raw-Softmax Macro-F1, `training_exact`, frozen-test role, validation-only calibration, manifest SHA match, passed path/source-ID/MD5 audit, and complete probability/distance columns.
- [ ] **Step 2: Run focused tests and confirm failures identify the absent loaders/validators.**
- [ ] **Step 3: Implement loaders and validators without fallback paths.** Record project-relative path plus SHA256 for every selection source; stop with the full missing `(fold, seed, lambda, artifact)` list before inference or output writes.
- [ ] **Step 4: Preflight the real 25 selected combinations.** Expected selection: fold 0=`1.0`, folds 1-3=`0.2`, fold 4=`0.5`; all selected checkpoints/summaries/manifests present; 20 prototype bundles absent and five fold-4 lambda-0.5 bundles reusable.
- [ ] **Step 5: Run one fold-seed debug through the existing exact protocol.** Use fold 0, seed 42, lambda 1.0 and a new `prototype_cv5_exact_valsel_w10_...` directory. Pass `--hier_aux_prob_weight 1.0` to build and calibration; run build, calibration, prediction, evaluation; confirm raw Softmax exactly reproduces the selected training summary and provenance checks pass.
- [ ] **Step 6: Run the remaining 19 combinations sequentially.** Use the identical locked grids and explicit selected lambda, never `--allow_overwrite`, and stop immediately on the first failed command.
- [ ] **Step 7: Re-run fixture tests and a real-data validate-only load.** Expected: 25/25 strict selected prototype runs, no missing/duplicate keys, no source changes.

### Task 3: Prototype functional value and convergence metrics

**Files:**
- Modify: `tests/test_generate_final_validation_audit.py`
- Modify: `tools/generate_final_validation_audit.py`

**Interfaces:**
- `compute_functional_metrics(predictions_by_run: Mapping[tuple[int, int], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
- `compute_convergence_tables(run_sources: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]`

- [ ] **Step 1: Add failing synthetic prediction tests.** Verify main/subtype Top-1/Top-2, deterministic true-class ranks, subtype-to-main mapping, four hierarchy/correctness states, conditional error rates, favorable main/subtype margins, per-run AUROC using negative margin, Raw/prototype disagreement counts, and feeding/stress Top-2 coverage.
- [ ] **Step 2: Confirm RED for absent metric functions.**
- [ ] **Step 3: Implement per-run-first aggregation.** Methods are `raw_softmax`, `main_prototype`, and `hierarchical_prototype`; aggregate run means/SDs and five fold means/cluster CIs without treating repeated-seed clips as independent.
- [ ] **Step 4: Implement convergence extraction.** Emit all 25 points for each B0-B3 stage, distribution summaries, validation-test gaps, and validation-vs-test scatter inputs; B3 inherits B2 values with an explicit provenance field. Search for epoch logs and set `curves_available=false` unless coverage is 25/25 for every stage.
- [ ] **Step 5: Run focused tests and confirm GREEN.**

### Task 4: Outputs, report, and audit figures

**Files:**
- Modify: `tests/test_generate_final_validation_audit.py`
- Modify: `tools/generate_final_validation_audit.py`
- Create: `paper/final_validation/*.csv`
- Create: `paper/final_validation/lambda_selection_provenance.json`
- Create: `paper/final_validation/FINAL_VALIDATION_REPORT.md`
- Create: `paper/final_validation/figures/*.{svg,pdf,png}`

**Interfaces:**
- `generate(root: Path, output_dir: Path, n_boot: int = 10_000, seed: int = 3407, validate_only: bool = False) -> dict[str, Any]`
- `build_final_report(...) -> str`
- `export_figure(fig: Figure, stem: Path) -> tuple[Path, Path, Path]`

- [ ] **Step 1: Add failing output-contract tests.** Assert exact required filenames, `--help`, validate-only no-write behavior, output refusal, 75-source lambda provenance, seven Holm-adjusted comparisons, explicit missing-artifact reporting, and the required claim-boundary/representative wording.
- [ ] **Step 2: Confirm RED.**
- [ ] **Step 3: Implement CSV/JSON/report export.** Include an ARS validate-mode Material Passport, all 11 statistical fallacy checks, source hash before/after equality, missing-artifact list, commands, split boundaries, and literal A/B/C evidence mapping.
- [ ] **Step 4: Implement the five quantitative-grid figures.** Contracts:
  - lambda figure: validation means/SDs and selected lambda only;
  - cumulative figure: B0/B1/B2_valsel/B3_valsel plus fixed-0.5 sensitivity and paired CIs;
  - functional figure: Top-k, hierarchy consistency, and feeding/stress coverage;
  - margin figure: correct-vs-error favorable margins plus per-run AUROC;
  - convergence figure: best-epoch distribution, validation-test gaps, and all validation-test points.
- [ ] **Step 5: Confirm GREEN, then run the full generator once.** Hash every frozen source before and after; abort if any digest differs.

### Task 5: Verification, review, handoff, and publication of the audit branch

**Files:**
- Modify: `handoff/CODEX_TO_GPT.md`
- Modify: `handoff/CODEX_TO_GPT.json`
- Modify: `handoff/HISTORY.md`

- [ ] **Step 1: Run every relevant help command.** Include the new generator and the existing build/calibrate/predict/evaluate prototype scripts.
- [ ] **Step 2: Compile the new generator.**

```powershell
python -m py_compile tools\generate_final_validation_audit.py tests\test_generate_final_validation_audit.py
```

- [ ] **Step 3: Run tests.** Run `tests.test_generate_final_validation_audit`, `tests.test_prototype_pipeline_core`, and `tests.test_generate_cumulative_framework_analysis` in the `pigsound-gpu` environment.
- [ ] **Step 4: Inspect all five PNGs at original resolution.** Check clipping, overlap, readable labels, interval definitions, and agreement with source CSV values. Confirm SVG text nodes and non-empty PDF/PNG exports.
- [ ] **Step 5: Run integrity checks.** Recompute the full frozen-source SHA map, `git diff --check`, `git status --short`, and inspect the complete diff for test-set tuning, leakage, accidental artifact edits, prohibited claims, or manuscript/validated-figure changes.
- [ ] **Step 6: Update all three handoff files.** Record exact commands, branch/base/commit, changed and generated files, all core metrics, path/source-ID/MD5 audit, source hashes, paper usability, blockers, and scientific-judgment questions.
- [ ] **Step 7: Commit and push.** Stage only new audit code/tests/docs/required final-validation outputs and handoff files; exclude generated checkpoint/prototype directories and all pre-existing untracked files. Push `paper/final-validation-audit`, report the commit SHA, and stop.
