# Final Figure QA Report

## Scope and figure contract

- Main-figure count: 5 (PASS)
- Supplementary-figure count: 3 (PASS)
- Export bundle completeness: PASS (SVG/PDF/300 dpi PNG/600 dpi TIFF)
- Primary font: SimSun; fallback stack: Arial, DejaVu Sans.
- SVG font policy: editable text nodes (`svg.fonttype=none`); SimSun is referenced but not embedded or redistributed.
- Figure width: fixed 183 mm for every main and supplementary figure; no `bbox_inches='tight'` size drift.
- Background: white. No gradient, shadow, 3D, decorative icon, rounded dashboard card, or embedded table screenshot is used.
- All Macro-F1 and recall axes use 0–1. Small paired effects are shown only on explicit difference axes around zero.

## Evidence contracts

- Figure 1 conclusion: the pipeline separates 2 s Log-Mel/CRNN representation, hierarchical training, train-only prototypes, validation-only calibration, and frozen clean/simulated-noise evaluation.
- Figure 2 conclusion: the major clean gain is the matched 2 s context effect; unmatched n=15 and n=25 ablations remain descriptive.
- Figure 3 conclusion: hierarchy provides semantic structure; λ=1.0 has the highest mean, λ=0.5 the lowest SD, and clean increments are nonsignificant.
- Figure 4 conclusion: main prototypes show the strongest bounded evidence under moderate simulated noise, while fold clustering tempers some run-level claims.
- Figure 5 conclusion: cough collapses under severe simulated noise; raw Softmax ranks uncertainty better by AURC/AUGRC, with only small prototype fixed-operating-point risk advantages.

## Machine checks

- SVG editable text/font/no-raster/no-gradient/no-filter check: PASS.
- Non-empty export check: PASS.
- Source-hash preservation is checked before and after generation; all required evidence sources were unchanged.
- Figure/source and figure/claim coverage: all 25 planned panels have entries.

| File | Bytes | Text nodes | SimSun refs | Pixels | DPI | Content margins (px) |
|---|---:|---:|---:|---|---|---|
| `paper/figures_journal/figure1_overall_framework.svg` | 20668 | 38 | 38 |  |  |  |
| `paper/figures_journal/figure1_overall_framework.pdf` | 19794 |  |  |  |  |  |
| `paper/figures_journal/figure1_overall_framework.png` | 209210 |  |  | 2161×1488 | (299.9994, 299.9994) | L80/R61/T97/B109 |
| `paper/figures_journal/figure1_overall_framework.tiff` | 1348486 |  |  | 4322×2976 | (600.0, 600.0) | L159/R121/T194/B219 |
| `paper/figures_journal/figure2_duration_clean_ablations.svg` | 171283 | 50 | 50 |  |  |  |
| `paper/figures_journal/figure2_duration_clean_ablations.pdf` | 43327 |  |  |  |  |  |
| `paper/figures_journal/figure2_duration_clean_ablations.png` | 159368 |  |  | 2161×1677 | (299.9994, 299.9994) | L13/R13/T13/B12 |
| `paper/figures_journal/figure2_duration_clean_ablations.tiff` | 1418312 |  |  | 4322×3354 | (600.0, 600.0) | L25/R26/T25/B25 |
| `paper/figures_journal/figure3_hierarchical_clean_prototypes.svg` | 172615 | 59 | 59 |  |  |  |
| `paper/figures_journal/figure3_hierarchical_clean_prototypes.pdf` | 43520 |  |  |  |  |  |
| `paper/figures_journal/figure3_hierarchical_clean_prototypes.png` | 194844 |  |  | 2161×1559 | (299.9994, 299.9994) | L14/R14/T13/B15 |
| `paper/figures_journal/figure3_hierarchical_clean_prototypes.tiff` | 2149422 |  |  | 4322×3118 | (600.0, 600.0) | L28/R28/T25/B33 |
| `paper/figures_journal/figure4_simulated_noise_robustness.svg` | 136878 | 88 | 88 |  |  |  |
| `paper/figures_journal/figure4_simulated_noise_robustness.pdf` | 38454 |  |  |  |  |  |
| `paper/figures_journal/figure4_simulated_noise_robustness.png` | 206008 |  |  | 2161×1984 | (299.9994, 299.9994) | L13/R10/T13/B15 |
| `paper/figures_journal/figure4_simulated_noise_robustness.tiff` | 1488072 |  |  | 4322×3968 | (600.0, 600.0) | L26/R21/T25/B34 |
| `paper/figures_journal/figure5_failure_selective_prediction.svg` | 123590 | 77 | 77 |  |  |  |
| `paper/figures_journal/figure5_failure_selective_prediction.pdf` | 38094 |  |  |  |  |  |
| `paper/figures_journal/figure5_failure_selective_prediction.png` | 237356 |  |  | 2161×1842 | (299.9994, 299.9994) | L12/R10/T13/B13 |
| `paper/figures_journal/figure5_failure_selective_prediction.tiff` | 1392128 |  |  | 4322×3685 | (600.0, 600.0) | L25/R21/T25/B28 |
| `paper/supplementary/figures/figure_s1_clean_ablations.svg` | 22008 | 27 | 27 |  |  |  |
| `paper/supplementary/figures/figure_s1_clean_ablations.pdf` | 18464 |  |  |  |  |  |
| `paper/supplementary/figures/figure_s1_clean_ablations.png` | 136341 |  |  | 2161×1724 | (299.9994, 299.9994) | L14/R18/T21/B13 |
| `paper/supplementary/figures/figure_s1_clean_ablations.tiff` | 783808 |  |  | 4322×3448 | (600.0, 600.0) | L28/R35/T42/B26 |
| `paper/supplementary/figures/figure_s2_demand_environment_snr.svg` | 51282 | 33 | 33 |  |  |  |
| `paper/supplementary/figures/figure_s2_demand_environment_snr.pdf` | 21741 |  |  |  |  |  |
| `paper/supplementary/figures/figure_s2_demand_environment_snr.png` | 136945 |  |  | 2161×1677 | (299.9994, 299.9994) | L13/R10/T16/B14 |
| `paper/supplementary/figures/figure_s2_demand_environment_snr.tiff` | 872158 |  |  | 4322×3354 | (600.0, 600.0) | L26/R21/T31/B28 |
| `paper/supplementary/figures/figure_s3_confusion_calibration_fold_deltas.svg` | 111789 | 165 | 165 |  |  |  |
| `paper/supplementary/figures/figure_s3_confusion_calibration_fold_deltas.pdf` | 37103 |  |  |  |  |  |
| `paper/supplementary/figures/figure_s3_confusion_calibration_fold_deltas.png` | 313593 |  |  | 2161×2645 | (299.9994, 299.9994) | L13/R10/T12/B12 |
| `paper/supplementary/figures/figure_s3_confusion_calibration_fold_deltas.tiff` | 1959852 |  |  | 4322×5291 | (600.0, 600.0) | L26/R21/T25/B25 |

## Source hashes

The following SHA-256 values were captured before generation and rechecked after generation.

| Source | SHA-256 |
|---|---|
| `docs/NOISE_ROBUSTNESS_PROTOCOL.md` | `7abdf6d9b942a084fec2e803f84e2911972b4cb883293eeff713ce72e852b94f` |
| `paper/CLAIMS_AND_EVIDENCE_v2.md` | `ec7fdb44229659095dec5eba360ca447a53f375e189befeb546f1f1949e25432` |
| `paper/appendix/clean_25_run_prototype_paired_stats.csv` | `29037e49a83ac2797bf0a960bf731a05cd741ff7b42aa8896123a839ec80cb5d` |
| `paper/appendix/clean_calibration_distribution.csv` | `f253e9a2d5ea9aae5b81b0c0bde9827287c927f954b70d2cadf0edfedb327d4f` |
| `paper/appendix/noise_25_run_aurc_augrc_summary.csv` | `6b0af0016b61903c4821534b246b3cf8ea23f4a45ff345526af0bc2aa6906d77` |
| `paper/appendix/noise_25_run_cluster_bootstrap.csv` | `773fa0b69981fafed8d03990661f6db8b9e31a0832ba11d862eb122c39994d5e` |
| `paper/appendix/noise_25_run_fold_level_deltas.csv` | `a7c6de15a104cb38e766d94a2eb07ff903ca82d8ab9a9b6662405407b51ee068` |
| `paper/appendix/noise_25_run_paired_stats.csv` | `514617a93c4cb871164c0b1493b29cd87d6685d8e834581a5ea95f6824c9705d` |
| `paper/appendix/noise_25_run_per_class_summary.csv` | `3c9e66d209d6e4a4fc82d01b948eeb7a65a5dfa90c966c9db50ca39deebb6a36` |
| `paper/appendix/noise_25_run_selective_summary.csv` | `d3f88ed904567dd0d1df7ed6455ebde85f6b16afefa4d81fd71a2af07f8ab1b1` |
| `paper/appendix/noise_25_run_summary.csv` | `b0df912c6197944196debfbe118979023b73a0793cbc3e25eb5198954e1cd296` |
| `paper/manuscript/paper_en_full_story_polished.md` | `96ada4c2a2e7faeb77f5d71c193e24223e9537e4f05f9e8b3930b9d8a78ce6ca` |
| `paper/tables/table1_dataset_and_leakage_free_protocol.csv` | `816ae354bd5cecd603c804241a5bd2e45e9cf728e24bdd57caf11d09cd37f2af` |
| `paper/tables/table2_clean_baseline_and_architecture_ablations.csv` | `81880bac54ffe9abe4de559b2404679b496630f727f251deabe66b9d8642b9b7` |
| `paper/tables/table3_duration_comparison.csv` | `3b3fd2e4dff79f4064d4b64ebdd0ef3b19e238a4e83db9a0ea34fee8523e962d` |
| `paper/tables/table4_hierarchical_aux_weight_ablation.csv` | `e0afcaadaa5304ad9acce9f8d96209424cbfdde431f90f3ba3608026acc39129` |
| `paper/tables/table5_clean_softmax_main_prototype_hierarchical_prototype.csv` | `bb8bdf843524486be34b3561656547432532a0303d04575b176cf2e4581a44a8` |
| `paper/tables/table6_simulated_noise_robustness_by_stratum.csv` | `fc2157c46a20c24f54cbf882c133338ad17289086f860931d2ed51d0478b3249` |
| `paper/tables/table8_per_class_noise_results_and_confusion_directions.csv` | `3bf61d28bb3ab59d53b65c58a4b3a496d8e63ac91fb79bcd83367fc11516d2f3` |
| `paper_results/tables/cv5_cap3x_hier_w02_vs_dur2_paired_stats.csv` | `3713195e86d947f5cec62604757483e12697ee54f155c9a0c2b64ca12cf23519` |
| `paper_results/tables/cv5_cap3x_hier_w05_vs_dur2_paired_stats.csv` | `11df39b1ef854bef555a2f83889b1efd8ee62aaaa3cd4624dd0bd3e1b31e767a` |
| `paper_results/tables/cv5_cap3x_hier_w10_vs_dur2_paired_stats.csv` | `cafa8a733025c0d21528d3ae9f283196d70f5b8fb09049964b38ebfc7dc633fc` |
| `paper_results/tables/cv5_cap3x_logmel_dur2_vs_dur1_paired_stats.csv` | `3ffd03eaeac8f3177e13f844c892a203979ab3114f8a213890c00ac87a124683` |
| `reports/prototype_noise_demand_w05_final/final_confusion_summary.csv` | `5ea8fb76016e61fa1d3ac214143923ed4da6e49904e7c60cb6a4bec0a0517a98` |
| `reports/prototype_noise_demand_w05_final/final_runs.csv` | `62ca7e95fa71dd76dc4d77fdaf9316e3a4ec969411a215782b466ef6a0f26b6a` |
| `tools/prototype_model_adapter.py` | `0261411d35aba495f29ad3c198cfb4f00225d4870771f7913d6e3648c9778ecb` |

## Manual visual QA

- PASS: all five 300 dpi main-figure PNGs and all three supplementary PNGs were inspected at original resolution in the Codex app.
- PASS: no title, panel label, axis label, exact P value, boundary statement, or legend is clipped; Figure 4 retains positive right content margins (PNG R10 px; TIFF R21 px).
- PASS: legends do not cover data; Figure 4's method legend occupies unused panel-b space, its interval legend is below panel c, and Supplementary Figure S2 uses a dedicated top strip.
- PASS: Figure 1 shows independent prototype branches, keeps the simulated-noise versus real-farm boundary visible, and contains no automatic-routing mechanism.
- PASS: Figures 2–5 retain 0–1 Macro-F1/recall axes; only explicit paired-difference panels use centered delta axes.
- PASS: Figure 2 panel-c value labels sit beyond the SD interval rather than crossing its error bars, with n retained in each row label.
- PASS: Figure 3 explicitly labels the incremental clean hierarchy result as nonsignificant and identifies panel-c raw/main/hierarchical outputs as the same lambda=0.5 cohort.
- PASS: Figure 4 highlights the moderate-noise evidence boundary, separates run-level and fold-cluster intervals, and prints exact P values without significance stars; the legends identify the stored P values as unadjusted.
- PASS: Figure 5 states that lower selective-risk metrics are better and preserves the raw-Softmax AURC/AUGRC advantage.
- PASS: Supplementary Figures S1–S3 retain all locked ablations, environment-by-SNR detail, complete pooled confusion matrices, validation-calibration distributions, and all five fold deltas.

## Known limitations

- SimSun remains editable but is not embedded in SVG; recipients without SimSun may see font substitution/reflow.
- SimSun has no separate native bold face in the local font file; bold panel labels may be synthetically emboldened.
- No external PDF font-inspection executable is installed; PDF validity is checked by successful Matplotlib export and non-empty file size.
- Per-environment/SNR fold-cluster CIs and selective-metric paired CIs/P values do not exist in the locked source package and are not invented.
- ALL_NOISY confusion counts pool repeated seed decisions; they are not independent recordings.
