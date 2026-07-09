# Polishing Change Log

## Scope

- Input: `paper/manuscript/paper_en_full_story.md`
- Output: `paper/manuscript/paper_en_full_story_polished.md`
- Skill used: `nature-polishing`
- Detected axes: `paper_type=algorithmic`, `section=title, abstract, intro, methods, results, discussion, conclusion`, `language=en`, `journal=generic`
- Target register: SCI applied-engineering English with cautious evidence boundaries

## Terminology Ledger

| Canonical term | Initial-use definition | Variants normalised | Decision |
|---|---|---|---|
| pig vocalisation recognition | automatic classification of pig vocal events | pig vocalization recognition, pig sound recognition | Use British-English `vocalisation` in prose. |
| 2 s Log-Mel CRNN | validated clean closed-set mainline | 2-second Log-Mel CRNN, 2 s mainline | Use `2 s Log-Mel CRNN`. |
| hierarchical auxiliary supervision | subtype auxiliary head added to the main classifier | hierarchical supervision, hierarchy | Use full term for claims, shorter form only when context is clear. |
| main prototype | four-main-class post-hoc prototype rule | main-class prototype | Use `main prototype` or `main-class prototype` consistently. |
| hierarchical prototype | prototype rule using subtype information | subtype prototype, hierarchical prototype method | Use `hierarchical prototype`. |
| DEMAND simulated additive noise | deterministic additive noise protocol using DEMAND recordings | simulated DEMAND noise, DEMAND noise | Use `DEMAND simulated additive noise` when making claims. |
| 0 dB active-event SNR | extreme stress condition in the noisy protocol | 0 dB, extreme-stress stratum | Always state the stress-test boundary near strong claims. |

## Main Language Modifications

- Split long sentences into shorter claim-evidence-boundary units, aiming for no more than 30 words per sentence.
- Converted wording to British English where visible in prose, including `vocalisation`, `behaviour`, `organise`, `summarise`, `artefact`, `generalised`, `finalised`, and `licence`.
- Replaced stronger or marketing-like phrasing with cautious verbs, including `support`, `indicate`, `suggest`, and `under the tested protocol`.
- Kept the manuscript as an algorithmic/applied-engineering paper rather than a discovery claim.
- Made Results sections report observed outcomes before interpretation.
- Made Discussion sections explicitly state when interpretation may fail.
- Preserved all numerical results, p-values, confidence intervals, fold-seed counts, class names, citation keys, and limitations.
- Retained unresolved citation, data-availability, code-availability, and ethics placeholders rather than inventing evidence.

## Scientific Boundaries Preserved

- The significant clean gain is attributed to 2 s context.
- Hierarchical auxiliary supervision remains positive in mean but not statistically significant over the 2 s baseline.
- Clean hierarchical prototype remains the highest Macro-F1 prototype variant.
- Under simulated DEMAND noise, the main prototype remains the most robust.
- The hierarchical prototype is not presented as the most noise-robust method.
- Prototype inference is not described as a universal uncertainty improvement.
- The 0 dB active-event SNR condition remains an extreme stress condition.
- Real-farm external validation is explicitly stated as not performed.

## Structural Changes

- Abstract was reorganised into context, approach, clean result, hierarchy result, prototype result, and boundary.
- Introduction was tightened around three bounded contributions.
- Related Work now separates prior-method positioning from the manuscript's own contribution.
- Methods now emphasise train, validation, and test boundaries for prototype and calibration artefacts.
- Results keep the three-story structure but reduce interpretive overreach within result paragraphs.
- Discussion now separates temporal, semantic, and inferential lessons.
- Limitations were retained and made more explicit.

## Verification Notes

- No new references were added.
- No numbers were intentionally changed.
- No model training, inference, threshold tuning, prototype construction, or audio processing was performed.
- The original manuscript file was not edited.
