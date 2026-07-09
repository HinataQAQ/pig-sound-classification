# Unresolved Pig Audio Data Sources

This file records local sources that must not be used as final submission evidence until rights or provenance are resolved.

## E. unresolved_do_not_submit_as_final

### `data/raw/scream_cough_small`

- Suspected source: Kaggle `porcine scream sounds and cough sounds` by `titpigrecognition`.
- URL: https://www.kaggle.com/datasets/titpigrecognition/porcine-scream-sounds-and-cough-sounds
- Local evidence: Edge download history records `archive.zip` from the Kaggle dataset URL; local folders are `trainingdata/testdata` with `cough` and `screams`.
- Blocker: Kaggle JSON-LD metadata reports license as `Unknown`; no explicit rights, ethics, or redistribution statement recovered.
- Action: do not include in final paper claims, do not redistribute raw audio, and do not publicly share derived features/labels until license and owner rights are confirmed.

### Empty placeholders

These roots contained no audio files during this audit. If repopulated later, they require a new provenance audit before use.

- `data/external/kaggle_porcine`
- `data/external/sow_call`
- `data/korea_subtype/raw/dry_cough`
- `data/korea_subtype/raw/abdominal_cough`

## Terms Needing Final Sanity Check

### `data/external/korea_raw/*`

- Source identified with high confidence as Smart Farm Korea / data.go.kr pig cough sound.
- data.go.kr lists no usage-permission restriction, but Smart Farm Korea terms should still be checked before redistributing raw audio in a public archive.
- Action before submission: cite data.go.kr/SmartFarm Korea and decide whether raw audio will be redistributed, linked externally, or referenced through source URLs plus manifests/checksums.

## Candidate Non-Matches

- AI Hub / aihub searches did not produce a high-confidence match to local Korea cough files.
- Literature pages about pig cough recognition were found in browser history but are not dataset download sources for the local files.
