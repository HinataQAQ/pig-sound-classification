# GPT paragraph-level review notes for `paper_en_full_story_polished.md`

Reviewed branch: `paper/sci-draft-v1`

Output manuscript produced in this pass:

- `paper/manuscript/paper_en_full_story_polished_gpt_reviewed.md`

## Overall decision

The current manuscript is scientifically usable but still too close to an internal project narrative. The revised version keeps the three evidence-supported contributions while making the prose more journal-facing, less conversational, and more explicit about methodological boundaries.

## Main structural changes

### Title

Kept the core title, but standardised `Post-Hoc` to `Post-hoc`. The title is within the typical short-title range and states the system scope without claiming real-farm deployment.

### Abstract

Converted the multi-paragraph abstract into a single journal-style abstract. The revised abstract retains:

- the four-class closed-set task;
- six auxiliary subtypes;
- 2 s duration result;
- clean hierarchical supervision boundary;
- DEMAND simulated-noise result;
- no real-farm external validation;
- no reliable 0 dB cough recognition claim.

It removes project-internal wording such as `locked result package` and avoids presenting all negative ablations in the abstract.

### Introduction

Reorganised the Introduction into the following logic:

1. livestock monitoring motivation;
2. event-duration problem;
3. subtype semantic problem;
4. frozen prototype inference problem;
5. bounded contributions.

The revised text removes repeated statements and keeps the three contributions separate.

### Related Work

Kept pig acoustics, CRNN, prototype inference, and selective prediction as separate paragraphs. The revision keeps citation placeholders as BibTeX keys, but removes unsupported novelty framing.

Key protection: the manuscript does not claim first use of pig vocal recognition, 2 s input, CRNN, or prototype learning.

### Materials and methods

Split the methods into:

- data sources and task definition;
- feature extraction and CRNN backbone;
- hierarchical auxiliary supervision;
- post-hoc prototype inference;
- simulated additive-noise protocol;
- evaluation and statistics.

Added explicit feature parameters and a clearer prototype definition. This improves reproducibility and reduces reviewer uncertainty.

### Results

Changed `Story 1/2/3` section titles into formal result headings:

- `Duration selection identifies the clean mainline`;
- `Hierarchical supervision adds semantic structure`;
- `Main-class prototypes improve simulated-noise robustness`.

This preserves the narrative but removes informal wording from the manuscript body.

### Noise statistics

The revision distinguishes:

- run-level paired statistics;
- fold-cluster evidence;
- MODERATE_NOISE versus EXTREME_STRESS;
- ALL_NOISY aggregate interpretation.

Main wording change: the strongest noise claim is now the 20/10 dB MODERATE_NOISE result. ALL_NOISY is reported, but not overclaimed because its fold-cluster interval crosses zero.

### Per-class failure

The revision keeps cough collapse as a central limitation. It explicitly states that 0 dB active-event SNR is a stress test and that cough recognition remains unreliable under that setting.

### Selective prediction

The revised text states that prototypes do not comprehensively improve uncertainty ranking. It preserves the small frozen-threshold selective-risk result but does not present it as universal calibration improvement.

### Discussion

Rewritten as a mechanism interpretation rather than a second results section. The revised discussion links:

- duration → event context;
- hierarchy → subtype semantics;
- prototype inference → coarse class-centre robustness;
- DEMAND → controlled but incomplete simulated noise.

### Limitations

Condensed repeated limitation paragraphs and made limitations explicit:

- no real-farm external validation;
- closed-set task;
- 0 dB cough failure;
- nonsignificant hierarchical gains;
- no universal uncertainty improvement;
- third-party raw audio restrictions.

### Data, code, and ethics statements

Kept the conservative stance:

- raw third-party pig audio not redistributed;
- Smart Farm Korea / data.go.kr source remains under provider terms;
- Sow Call Dataset and DEMAND are cited;
- code release URL uses `v0.1-paper-draft-r1`;
- no new animal experiment or prospective farm recording was performed.

## Remaining author-side blockers

1. Final target journal.
2. Korea provider terms final screenshot / terms archive.
3. Author names and affiliations.
4. Funding statement.
5. Competing interests statement.
6. Decision on whether model checkpoints and prototype artifacts will ever be archived.
7. Final reference-format conversion for target journal.
8. Supplementary information package assembly.

## Claims that remain prohibited

Do not claim:

- 2 s is a universal pig-audio standard;
- hierarchical supervision significantly improves clean performance;
- hierarchical prototype is the most noise robust;
- prototype inference universally improves uncertainty;
- 0 dB cough recognition is reliable;
- DEMAND additive-noise testing is real-farm external validation;
- raw third-party pig audio can be redistributed by this repository.
