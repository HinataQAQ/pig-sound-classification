# Overclaim Audit

## Summary

The source draft was already cautious on most scientific claims. The main overclaim risk was not direct exaggeration, but sentences that could be read as broader than the evidence supports. The polished manuscript narrows those statements to the tested closed-set, clean, or DEMAND simulated-noise protocols.

## Overclaim Risks Found and Edited

| Location | Risk | Polishing action |
|---|---|---|
| Title and abstract | `Duration-aware hierarchical representations` could imply a generally superior system. | Kept the title but made the abstract state that hierarchy is semantic and not significantly better than the 2 s baseline. |
| Abstract | `decisive clean-data gain` could sound too strong outside the tested protocol. | Reframed the result as the main clean-data gain and tied it to paired evidence. |
| Introduction | `clean-data advance` could be read as universal. | Added boundaries: closed-set protocol, validated result package, and no universal duration claim. |
| Hierarchical supervision sections | Positive means could invite a significant-performance claim. | Repeated that the hierarchical gain over the 2 s baseline was not statistically significant. |
| Prototype sections | Clean hierarchical prototype performance could imply noise robustness. | Separated clean semantic explanation from simulated-noise robustness. |
| Noise robustness sections | Main prototype improvement could be overgeneralised to real farms. | Restricted the claim to DEMAND simulated additive noise under the frozen protocol. |
| Selective prediction sections | Prototype improvements could imply better uncertainty calibration. | Stated that prototypes do not comprehensively improve uncertainty or ranking. |
| 0 dB discussion | 0 dB could be misread as ordinary operating noise. | Described it as an extreme active-event SNR stress condition. |
| Conclusion | Reproducible path could imply deployment readiness. | Added `under controlled simulated conditions` and retained real-farm validation as future work. |

## Required Claims That Remain Explicit

- 2 s context significantly improves clean classification.
- Hierarchical auxiliary supervision is positive but not statistically significant over the 2 s baseline.
- The clean hierarchical prototype has the highest Macro-F1 among prototype variants.
- Under simulated DEMAND noise, the main prototype is the most robust method.
- The hierarchical prototype is not the most noise-robust method.
- Prototype inference does not universally improve uncertainty.
- 0 dB active-event SNR is an extreme stress condition.
- Real-farm external validation was not performed.

## Positions That Still Need Author Evidence

| Location | Evidence still needed | Why polishing cannot resolve it |
|---|---|---|
| Related Work | Verified livestock-acoustics and pig-vocalisation references | The manuscript contains a citation placeholder and should not invent sources. |
| Data Availability | Clean pig audio availability, owner, licence, and access route | These are rights and provenance facts, not language issues. |
| Code Availability | Repository release tag, archived DOI, and software licence | The draft cannot claim public release until authors complete it. |
| Ethics Statement | Animal-use approval authority and approval number, if applicable | Ethics provenance must come from the original data collection record. |
| Real-farm deployment discussion | External farm validation data | The current study used simulated DEMAND additive noise only. |
| Severe-noise cough recognition | Additional evidence or methods for 0 dB cough recovery | Existing results show cough collapse under 0 dB, so language cannot make it reliable. |

## Audit Verdict

The polished manuscript is paper-usable as an English SCI applied-engineering draft. It is not submission-complete until the missing references, ethics details, data rights, and release information are supplied by the authors.
