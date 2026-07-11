# Final Functional Figure Package QA

## Machine checks

All exports were checked for non-empty content, fixed 183-mm width, white background, editable SVG text, 300-dpi PNG, and 600-dpi TIFF.

| File | Bytes | Detail | Status |
|---|---:|---|---|
| paper/figures_final/figure1_final_framework.svg | 25339 | 43 text nodes | PASS |
| paper/figures_final/figure1_final_framework.pdf | 51650 | vector text nodes | PASS |
| paper/figures_final/figure1_final_framework.png | 225668 | 2161x1393 | PASS |
| paper/figures_final/figure1_final_framework.tiff | 1346264 | 4322x2787 | PASS |
| paper/figures_final/figure2_primary_classification_evidence.svg | 51009 | 55 text nodes | PASS |
| paper/figures_final/figure2_primary_classification_evidence.pdf | 53565 | vector text nodes | PASS |
| paper/figures_final/figure2_primary_classification_evidence.png | 169124 | 2161x921 | PASS |
| paper/figures_final/figure2_primary_classification_evidence.tiff | 980508 | 4322x1842 | PASS |
| paper/figures_final/figure3_prototype_functional_value.svg | 102986 | 92 text nodes | PASS |
| paper/figures_final/figure3_prototype_functional_value.pdf | 64523 | vector text nodes | PASS |
| paper/figures_final/figure3_prototype_functional_value.png | 228658 | 2161x1417 | PASS |
| paper/figures_final/figure3_prototype_functional_value.tiff | 1421708 | 4322x2834 | PASS |
| paper/figures_final/figure4_prototype_candidate_cases.svg | 62467 | 119 text nodes | PASS |
| paper/figures_final/figure4_prototype_candidate_cases.pdf | 49533 | vector text nodes | PASS |
| paper/figures_final/figure4_prototype_candidate_cases.png | 279901 | 2161x1677 | PASS |
| paper/figures_final/figure4_prototype_candidate_cases.tiff | 1690464 | 4322x3354 | PASS |
| paper/figures_final/figure_s1_complete_clean_ablations.svg | 18098 | 18 text nodes | PASS |
| paper/figures_final/figure_s1_complete_clean_ablations.pdf | 40669 | vector text nodes | PASS |
| paper/figures_final/figure_s1_complete_clean_ablations.png | 95740 | 2161x1015 | PASS |
| paper/figures_final/figure_s1_complete_clean_ablations.tiff | 497380 | 4322x2031 | PASS |
| paper/figures_final/figure_s2_fixed_lambda_retrospective_cumulative.svg | 46927 | 33 text nodes | PASS |
| paper/figures_final/figure_s2_fixed_lambda_retrospective_cumulative.pdf | 54575 | vector text nodes | PASS |
| paper/figures_final/figure_s2_fixed_lambda_retrospective_cumulative.png | 115752 | 2161x921 | PASS |
| paper/figures_final/figure_s2_fixed_lambda_retrospective_cumulative.tiff | 590234 | 4322x1842 | PASS |
| paper/figures_final/figure_s3_lambda_selection_by_fold.svg | 33277 | 32 text nodes | PASS |
| paper/figures_final/figure_s3_lambda_selection_by_fold.pdf | 43087 | vector text nodes | PASS |
| paper/figures_final/figure_s3_lambda_selection_by_fold.png | 138741 | 2161x897 | PASS |
| paper/figures_final/figure_s3_lambda_selection_by_fold.tiff | 589236 | 4322x1795 | PASS |
| paper/figures_final/figure_s4_best_epoch_validation_test_gap.svg | 28843 | 28 text nodes | PASS |
| paper/figures_final/figure_s4_best_epoch_validation_test_gap.pdf | 39085 | vector text nodes | PASS |
| paper/figures_final/figure_s4_best_epoch_validation_test_gap.png | 64969 | 2161x921 | PASS |
| paper/figures_final/figure_s4_best_epoch_validation_test_gap.tiff | 665804 | 4322x1842 | PASS |
| paper/figures_final/figure_s5_demand_simulated_noise.svg | 65939 | 72 text nodes | PASS |
| paper/figures_final/figure_s5_demand_simulated_noise.pdf | 52587 | vector text nodes | PASS |
| paper/figures_final/figure_s5_demand_simulated_noise.png | 130447 | 2161x1370 | PASS |
| paper/figures_final/figure_s5_demand_simulated_noise.tiff | 1006660 | 4322x2740 | PASS |
| paper/figures_final/figure_s6_aurc_augrc.svg | 21357 | 29 text nodes | PASS |
| paper/figures_final/figure_s6_aurc_augrc.pdf | 40428 | vector text nodes | PASS |
| paper/figures_final/figure_s6_aurc_augrc.png | 68795 | 2161x897 | PASS |
| paper/figures_final/figure_s6_aurc_augrc.tiff | 899518 | 4322x1795 | PASS |
| paper/figures_final/figure_s7_label_aware_true_class_margin.svg | 23860 | 26 text nodes | PASS |
| paper/figures_final/figure_s7_label_aware_true_class_margin.pdf | 43454 | vector text nodes | PASS |
| paper/figures_final/figure_s7_label_aware_true_class_margin.png | 77480 | 2161x921 | PASS |
| paper/figures_final/figure_s7_label_aware_true_class_margin.tiff | 1068386 | 4322x1842 | PASS |

## Provenance checks

- Panel rows: 30; duplicate figure/panel rows: 0.
- Every panel row records source paths and freshly computed SHA-256 values.
- Full frozen-source audit: 921 files hashed before and after generation; no digest changed.
- Shared prototype margin scope: `shared_main_prototype_geometry`.
- No mapped-subtype or hierarchical cosine-distance vector was constructed.
- No training, checkpoint inference, Atlas generation, or test-set tuning was performed.

## Original-resolution visual inspection

- Inspection date: 2026-07-11.
- Scope: all 11 final PNG exports at original resolution, with the final Figure 4 export re-inspected after the panel-d title correction.
- Clipping and overlap: PASS; titles, panel labels, legends, tables, axis labels, and annotations remain inside their intended regions.
- Typography and rendering: PASS; text is legible, symbols render correctly, and no raster corruption or transparent-background artifact was observed.
- Axis integrity: PASS; scales, tick labels, category order, reference lines, and secondary-axis mappings agree with the plotted quantities.
- Figure 1: PASS; split roles and framework functions are visually separated.
- Figure 2: PASS; the duration-supported primary gain, exploratory hierarchy result, and B3 non-superiority are visually distinct.
- Figure 3: PASS; predicted-margin use-time status, label-aware evaluation, rare inconsistency prevalence, available-run counts, and low recall remain visible.
- Figure 4: PASS; the hierarchical-route representative is explicit and the shortened panel title is unclipped.
- Figures S1-S7: PASS; clean ablations, retrospective fixed-lambda evidence, selection frequencies, validation-test gaps, fixed-lambda simulated noise, selective-risk summaries, and label-aware true-class margins remain supplementary and legible.
