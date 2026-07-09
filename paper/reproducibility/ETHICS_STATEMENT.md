# Ethics Statement

This ethics statement was prepared from the reviewed manuscript and
reproducibility files plus the 2026-07-09 clean pig-audio source-recovery audit.
The audit identifies the current clean pig-audio sources as reused public or
third-party datasets rather than newly collected recordings. Original
source-side animal ethics approvals, farm permissions, and data-use terms should
be attributed to the original providers where available.

## Submission Draft

This manuscript uses previously collected public or third-party pig-audio data;
no new animal experiment was conducted for this manuscript.

No new animal experiment, animal intervention, or manual listening requirement
is documented in the reviewed reproducibility package for this statement stage.
The computational work uses existing pig-vocalization audio, auxiliary subtype
metadata, cross-validation manifests, model outputs, and simulated additive
environmental-noise mixtures. The current clean manifests reuse Korean pig cough
audio from Smart Farm Korea / data.go.kr and Sow Call Dataset audio from
figshare. The Korean source is cited through its public data.go.kr and Smart
Farm Korea pages; raw audio is not redistributed here and should be obtained
from the original provider portal unless the authors complete and record a final
provider-terms check. The Sow Call Dataset is cited through DOI
`10.6084/m9.figshare.16940389` and is listed as CC BY 4.0 in the source audit.
DEMAND environmental recordings are reused as a public third-party noise source
and should be cited through the Zenodo DOI `10.5281/zenodo.1227121`; DEMAND
audio should not be redistributed in the GitHub repository.

The original data-source ethics and permissions remain the responsibility of
the original data providers or repositories. This manuscript should not present
any third-party audio as newly collected by the authors. If the target journal
requires animal-use approval details for secondary analysis of animal audio, the
authors should provide the original provider statement, repository terms, or an
explicit exemption/permission explanation.

The simulated DEMAND noise evaluation does not constitute real-farm external
validation. It is an additive noise stress protocol using selected public
environmental recordings and fixed active-event SNR levels. The manuscript
should not imply that the system has been validated prospectively in commercial
pig-farm environments unless such a dataset and ethics/permission record are
added later.

## Ethics and Permission Fields Required Before Submission

- Final confirmation that the target journal accepts a no-new-animal-experiment
  statement for computational reuse of third-party pig audio.
- Original clean audio source name and data owner or repository, already
  identified for the current mainline as Smart Farm Korea / data.go.kr and the
  figshare Sow Call Dataset.
- Animal ethics approval authority and approval number, if applicable.
- Farm, institution, or data-owner permission statement, if applicable.
- Confirmation that reuse of the clean audio and auxiliary subtype labels is
  permitted for this manuscript.
- Confirmation of whether raw clean audio, metadata, and subtype labels may be
  redistributed, deposited under controlled access, or only described as
  restricted third-party material.
- Any data-use agreement, confidentiality term, farm privacy condition, or
  commercial restriction that affects audio redistribution.
- Any statement required by the target journal for computational reanalysis of
  previously collected animal audio.
- The unresolved Kaggle-like scream/cough source remains excluded from final
  evidence until licence, rights, and ethics/permission status are resolved.

## Current Risk Flag

The manuscript may be framed as computational reuse of public or third-party
audio with no new animal experiment. It should not redistribute raw third-party
pig audio or claim provider-side ethics approvals that have not been recovered.
Before submission, keep the unresolved source and permission items listed in
`paper/reproducibility/SOURCE_CITATION_STATUS.md` and
`paper/reproducibility/UNRESOLVED_DATA_SOURCES.md` aligned with the final Data
Availability statement.
