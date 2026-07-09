# Reference Verification Report

## Scope

Verified `paper/manuscript/paper_en_full_story_polished.md` against `paper/references/references.bib`, `paper/references/citation_audit.csv`, and `paper/CLAIMS_AND_EVIDENCE_v2.md`. No model experiments were run. No result CSV/JSON, figures, data files, source code, manuscript text, or `references.bib` were modified.

## Summary

- References checked: 32
- Citation keys used in manuscript: 21
- Citation keys present in `references.bib`: 32
- Broken manuscript citation keys: none
- Unused BibTeX entries retained for possible background/method use: 11

## Classification Counts

| classification | all references | used in manuscript |
|---|---:|---:|
| verified_strong | 13 | 13 |
| verified_background | 7 | 3 |
| verified_method | 11 | 5 |
| weak_support | 0 | 0 |
| metadata_only | 1 | 0 |
| remove_or_replace | 0 | 0 |

## Verification Sources Used

- Crossref Works API for DOI-bearing journal/conference papers.
- DataCite DOI API for Zenodo and figshare dataset DOIs.
- Zenodo DEMAND landing page: https://zenodo.org/records/1227121
- figshare Sow Call Dataset landing page: https://figshare.com/articles/dataset/Sow_call_dataset/16940389
- data.go.kr pig-cough dataset page: https://www.data.go.kr/data/15117368/fileData.do
- arXiv stable records for `snell2017prototypical`, `choi2017crnn`, and `geifman2017selective`.
- SciPy proceedings page for `mcfee2015librosa`.
- GitHub source page for `fdshifts_riskcoverage`.

## Used Citation Keys

acevedo2009animalcalls, briefer2022pigcalls, cakir2017crnn_sed, choi2017crnn, chung2025pvmc, demand2018, exadaktylos2008cough, ferrari2008cough, geifman2017selective, huang2014anuran, illmann2013signalneed, kahl2021birdnet, mcfee2015librosa, shen2022pigcoughfusion, smartfarmkorea_pig_cough_voice, snell2017prototypical, sow_call_dataset_2021, tallet2013piglets, weary1995calling, xie2024abnormalpig, yin2021pigcoughcnn

## Broken Citation Keys

No used manuscript citation key is missing from `references.bib`. No manuscript citation-key edits were needed.

## Metadata Issues Requiring Attention Before Final Reference Cleanup

| citation_key | issue | action |
|---|---|---|
| demand2018 | Zenodo/DataCite verifies DOI and authors, but the source page is published in 2013 while current BibTeX year is 2018; the page also contains a license-text/rights-metadata mismatch. | Correct or explain year/version and license wording before final submission. |
| sow_call_dataset_2021 | figshare/DataCite verifies DOI, but landing page version 2 is posted in 2022 and authored by Jie Liao; current BibTeX has no author and year 2021. | Add author and clarify whether citing version 1 (2021) or current version 2 (2022). |
| smartfarmkorea_pig_cough_voice | Official data.go.kr URL is verified and has no DOI; current BibTeX uses an English descriptive title and no institutional author field. | Keep official URL citation; consider using official Korean title plus institutional author/publisher in final BibTeX. |
| choi2017crnn | Stable arXiv URL is verified; Crossref also verifies IEEE ICASSP DOI `10.1109/ICASSP.2017.7952585` and pages `2392-2396`, absent from current BibTeX. | Optional metadata improvement: add DOI/pages or explicitly keep arXiv citation. |

## Claim-Support Findings

- Pig-vocalisation background citations used in Related Work are supported by verified pig-call, piglet-call, pig-cough, abnormal pig-vocalisation, and noisy farm-environment papers.
- Method citations for CRNN, Log-Mel/audio tooling, prototypes, and selective prediction support method framing only. They do not support internal Macro-F1, Wilcoxon, or leakage claims.
- Different-species citations are used only in broad animal-audio/method background: `acevedo2009animalcalls, huang2014anuran, kahl2021birdnet, heinrich2025audioprotopnet`. They should not be used as pig-specific evidence.
- Review or general-method entries that are not currently cited should not be used to support specific internal experiments unless the manuscript adds the relevant caveat.

## Dataset Citation Issues

- `smartfarmkorea_pig_cough_voice`: exact official data.go.kr page is verified, including provider, WAV format, dry/abdominal cough keywords, free access, and usage-permission range listed as unrestricted. Remaining risk is source-side ethics/redistribution wording and lack of DOI.
- `sow_call_dataset_2021`: DOI and figshare page are verified; metadata should be normalized for author/version/year.
- `demand2018`: DOI and Zenodo page are verified; year/license wording should be normalized because page text and rights metadata are not perfectly aligned.

## References To Remove Or Replace

No used reference is classified as `remove_or_replace`. `fdshifts_riskcoverage` is `metadata_only` because it is not cited in the current polished manuscript; remove it from the final `.bib` only if the journal requires a cited-only BibTeX file. Other unused entries are verified background/method entries that can remain harmlessly in the `.bib` if the bibliography processor filters unused keys.

## Output Table

Detailed per-reference verification is in `paper/references/REFERENCE_VERIFICATION_TABLE.csv`.
