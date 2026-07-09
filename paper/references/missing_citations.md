# Citation expansion and audit

Search date: 2026-07-09.

Skill scope used: `nature-citation` segmentation and conservative support grading, with broader academic search allowed by the user for domain literature. Bibliographic metadata were checked through DOI/Crossref records and official publisher, dataset, or archive URLs where available.

## Summary

- Existing bibliography entries before this audit: 6.
- New bibliography entries added: 24.
- Total bibliography entries after this audit: 30.
- New audit artifact: `paper/references/citation_audit.csv`.
- Manuscript body was not edited in this pass; citation insertion can be done after author review of the audit table.

## Strongest support for core claim groups

| Claim group | Strongest support | Support grade | Notes |
|---|---|---:|---|
| Pig call emotional valence and context | `@briefer2022pigcalls`; `@tallet2013piglets`; `@illmann2013signalneed`; `@weary1995calling` | strong | Use for pig calls as context/valence/need signals. |
| Pig vocalization classification and livestock acoustic monitoring | `@briefer2022pigcalls`; `@xie2024abnormalpig`; `@chung2025pvmc`; `@mcloughlin2019automated`; `@coutant2024scoping` | strong/background | Use reviews only for field framing, not for specific experimental claims. |
| Pig cough recognition | `@ferrari2008cough`; `@exadaktylos2008cough`; `@yin2021pigcoughcnn`; `@shen2022pigcoughfusion` | strong | Supports pig-cough recognition context, not this paper's 0 dB cough failure. |
| Noisy pig vocalization recognition | `@chung2025pvmc`; `@xie2024abnormalpig` | partial/strong background | Supports the existence of noisy or smart-farm pig-vocalization work; does not validate this paper's DEMAND protocol. |
| DEMAND environmental noise source | `@demand2018` | strong | Must be described as simulated additive noise in this paper, not real-farm validation. |
| 2 s clean mainline | internal `paper/tables/table3_duration_comparison.csv`; internal paired-duration script | strong internal | No direct external source was found that establishes 2 s as a general pig-vocalization standard. |
| CRNN audio classification | `@choi2017crnn`; `@cakir2017crnn_sed` | strong | Choi supports CRNN audio classification; Cakir strengthens sound-event framing. |
| Log-Mel/MFCC and animal/audio classification | `@mcfee2015librosa`; `@acevedo2009animalcalls`; `@huang2014anuran`; `@kahl2021birdnet`; `@hershey2017cnn_audio` | background | Good for method context; exact Log-Mel settings remain internal. |
| PCEN and SpecAugment methods | `@lostanlen2019pcen`; `@park2019specaugment` | strong method background | Cite only because PCEN and SpecAugment are actually discussed as ablations. |
| Prototypical networks | `@snell2017prototypical` | strong | Use only for prototype/class-center background. Do not claim prototype learning is first introduced here. |
| Hierarchical labels/classification | `@silla2011hierarchical`; `@gemmeke2017audioset` | background/partial | Supports hierarchical classification and sound ontology context, not this paper's statistical result. |
| Bioacoustic prototype interpretability | `@heinrich2025audioprotopnet` | partial | Strongly relevant to audio prototypes, but bird rather than pig; use as analogical support. |
| Selective classification and risk coverage | `@geifman2017selective`; `@fdshifts_riskcoverage` | strong method support | Existing citations remain appropriate. Negative uncertainty results are internal. |

## Still missing or author-dependent citations

1. Original pig-vocalization recording source: dataset paper, collection report, ethics protocol, repository page, or permission statement is still missing.
2. Animal ethics approval: authority, approval number, waiver/exemption, or statement that this work is computational reanalysis only is still missing.
3. Code availability: public repository URL, release tag, archived DOI, or embargo/access statement is still missing.
4. Direct external evidence for a universally preferred 2 s pig-vocalization window was not found. The 2 s claim should remain an internal empirical result.
5. If `MCTAFD` remains in the final manuscript as an externally named method rather than a project-specific feature stack, cite its original definition or add a short internal definition.
6. Spectral-gate denoising is discussed as a negative ablation but no formal method citation was added in this pass; add one only if the final manuscript needs method provenance beyond an implementation note.
7. The final data availability statement must specify whether original pig recordings can be shared, restricted, or only summarized.

## Rejected or not-used candidates

These were not added to `references.bib` in this pass.

| Candidate | Reason not used |
|---|---|
| ResearchGate, Google Scholar, or aggregator pages for pig-cough papers | Discovery-only pages are not formal support sources. Publisher/DOI metadata were used instead. |
| Guarino et al. field-test cough-detection work | Potentially relevant, but the exact DOI/publisher metadata were not verified during this audit and the cough claim is already covered by four verified primary papers. |
| Generic environmental sound CNN papers such as urban-sound-only studies | Relevant to audio ML but less specific than CRNN, AudioSet, BirdNET and animal-call references; omitted to keep the bibliography within 25--35 entries. |
| Additional pig cough fusion/monitoring papers beyond Shen 2022 and Yin 2021 | Overlapping support; not needed for the current claim set unless the final related-work section expands. |
| Reviews as support for specific numerical claims | Manteuffel 2004, McLoughlin 2019, Stowell 2022 and Coutant 2024 were kept only for background or scoping context. |
| DEMAND interpreted as real-farm validation | Rejected interpretation. DEMAND supports public environmental noise recordings only; this paper uses simulated additive mixing. |
| Prototype learning as a claimed novelty | Rejected wording. The manuscript should say post-hoc prototype inference, not first proposal of prototype learning. |

## Suggested insertion points

- Related Work paragraph 1: add pig vocalization and livestock acoustic monitoring citations: `@briefer2022pigcalls; @tallet2013piglets; @ferrari2008cough; @exadaktylos2008cough; @yin2021pigcoughcnn; @shen2022pigcoughfusion; @xie2024abnormalpig; @chung2025pvmc`.
- CRNN paragraph: keep `@choi2017crnn` and add `@cakir2017crnn_sed`.
- Log-Mel/animal-sound methods paragraph: keep `@mcfee2015librosa`; optionally add `@acevedo2009animalcalls; @huang2014anuran; @kahl2021birdnet`.
- PCEN/SpecAugment sentence: add `@lostanlen2019pcen; @park2019specaugment`.
- Prototype paragraph: keep `@snell2017prototypical`; add `@heinrich2025audioprotopnet` only for interpretable audio-prototype background.
- Hierarchical label paragraph: add `@silla2011hierarchical; @gemmeke2017audioset` as background, while keeping the performance claim internal.
- DEMAND/noise protocol paragraph: keep `@demand2018`; explicitly say "simulated additive noise" and "not real-farm external validation."
