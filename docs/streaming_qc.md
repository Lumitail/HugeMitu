# Streaming waterfall / QC pipeline

This stage builds bounded-memory, review-oriented QC products for one logical observation
(`ObservationManifest`) without materializing the full observation cube.

## Inputs and data contract

The implementation preserves the existing raw `.dat` contract:

- headerless payload
- row-major `(rows, 256)`
- little-endian `<u2`
- each word packs signed int8 `I/Q`
- metadata remain external
- STFT runs along **time** inside each coarse channel (`axis=0` on `(time, 256)`)

## Two-pass design

`build_streaming_waterfall(...)` in `acs.preproc.waterfall` runs two streaming passes over
`iter_observation_blocks(...)`.

### Pass 1: bounded baseline sampling

1. Stream blocks across file boundaries.
2. Decode packed `I/Q` per block.
3. Maintain a carry-buffer of trailing samples so STFT frames crossing block boundaries are
   complete and continuous.
4. Emit non-overlapping global STFT frame starts using `hop` stepping (no double counting).
5. Capture only up to `baseline_sample_frames` coarse-power vectors for baseline estimation.

Baseline estimation:

- median across sampled frames per coarse channel
- broad smoothing with moving average across channels (`baseline_smooth_width`)

This produces a broad bandpass estimate appropriate for first-pass H I-adjacent QC flattening.

### Pass 2: flattened products

1. Re-stream frame coarse-power vectors.
2. Divide by baseline to produce excess power.
3. Accumulate mean-excess spectrum incrementally.
4. Produce display waterfall by temporal decimation (`time_decimation`) using group means.

Outputs are converted to dB for review visualization.

## Memory model

The pipeline keeps memory bounded by:

- one streamed block of decoded rows
- one carry-buffer tail (at most `nfft - 1` rows plus short remainder)
- bounded baseline sample bank (`baseline_sample_frames x 256`)
- decimated display waterfall (`display_frames x 256`), where `display_frames` is reduced by
  `time_decimation`
- small running accumulators (mean spectrum sums and metadata)

It does **not** allocate:

- the full decoded observation
- the full-resolution `(all_frames, nfft, 256)` STFT cube

## Review artifacts

`save_qc_review_bundle(...)` in `acs.review.qc` writes:

- waterfall image PNG
- mean-excess spectrum PNG
- metadata summary JSON

The JSON includes STFT parameters, frame counts, display frame row coordinates, and basic
baseline statistics.
