# DEMAND Noise Source Decision

This stage uses DEMAND as the only public noise source for the simulated pig-farm
noise robustness protocol. MUSAN, ESC-50, and UrbanSound8K are intentionally not
downloaded or included in this stage.

## Public Source

- Dataset: DEMAND, a collection of multi-channel recordings of acoustic noise in
  diverse environments
- DOI: 10.5281/zenodo.1227121
- Zenodo record: https://zenodo.org/records/1227121
- Version recorded locally: 1.0
- Download date recorded locally: 2026-06-25
- Source type: synchronized multi-channel environmental noise recordings

The DEMAND paper/PDF describes synchronized 16-channel recordings. Channels are
therefore not treated as independent recordings. This project fixes one channel
per selected environment before evaluation and records that channel in the noise
manifest.

## License Note

The local provenance records both license observations and must not collapse them
into one unambiguous license:

- `description_license=CC-BY-SA-3.0`
- `zenodo_rights_license=CC-BY-4.0`
- `license_status=conflicting_metadata`
- `redistribution_policy=do_not_redistribute_in_repository`

Paper text should cite DEMAND and explicitly state which license metadata was
observed from Zenodo and from the DEMAND PDF. If strict redistribution language
is required, use links and checksums rather than redistributing DEMAND audio.

## Selected DEMAND Environments

Exactly three DEMAND environment families were selected for the planned
screening protocol:

- `DWASHING`: steady indoor/mechanical noise. This is a domestic washroom
  recording with a front-loading washing machine running.
- `TBUS`: engine/transport machinery noise. This is a public transit bus
  recording.
- `STRAFFIC`: mixed environmental noise. This is a busy street traffic
  intersection recording.

The fixed screening uses all three selected environments over 5 folds x 3 seeds
at 20, 10, and 0 dB active-event SNR. The 0 dB condition is an extreme
simulated-noise stress condition, not an ordinary deployment noise level.

## Channel Policy

- Selected channel: `ch01.wav`
- Channel numbering: one-indexed in the manifest as `selected_channel=1`
- Recording identity: all channels from the same DEMAND environment collapse to
  `DEMAND:<environment>` for leakage/provenance purposes
- Independence rule: synchronized channels must not be counted as independent
  noise recordings

## Local Reviewed Noise

Local `fan_noise`, `equipment`, and `mixed_noise` candidates are internal debug
assets only until provenance is complete. They are not paper-facing in this
stage.

The template at `data/noise_sources/local_reviewed/PROVENANCE_TEMPLATE.csv`
defines the required provenance fields. Unknown provenance must be recorded with
`allowed_for_publication=false`.

Excluded from the main protocol:

- `det_round1_clips/other_clean`
- `round4_noise_mix*`
- pig vocal background or target-class leakage sources
- human voice recordings

## Generated Metadata

The DEMAND metadata generated for this stage is:

- `data/noise_sources/demand/NOISE_SOURCE_MANIFEST.csv`
- `data/noise_sources/demand/PROVENANCE.json`

`PROVENANCE.json` records the SHA256 of `NOISE_SOURCE_MANIFEST.csv`. The noisy
evaluator verifies this before running.

Downloaded ZIP files, extracted WAV files, cache files, and derived audio are not
committed.
