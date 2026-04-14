"""Streaming waterfall/QC construction for one logical observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from acs.io.dat_reader import decode_packed_iq
from acs.io.manifest import ObservationManifest
from acs.io.streaming import ObservationBlock, iter_observation_blocks


@dataclass(frozen=True)
class StreamingWaterfallResult:
    """Display-ready waterfall/QC products computed in bounded memory."""

    waterfall_db: np.ndarray
    mean_excess_db: np.ndarray
    baseline_power: np.ndarray
    display_frame_start_rows: np.ndarray
    metadata: dict[str, int | float | str]


def estimate_baseline(sample_frames: np.ndarray, smooth_width: int = 33) -> np.ndarray:
    """Estimate broad per-channel bandpass from sampled frame powers.

    The estimate is intentionally broad-band: median across sampled frames followed by
    moving-average smoothing across coarse-channel index.
    """

    if sample_frames.ndim != 2:
        raise ValueError("sample_frames must have shape (n_frames, n_channels).")
    if sample_frames.shape[0] == 0:
        raise ValueError("sample_frames must contain at least one frame.")

    raw = np.median(sample_frames, axis=0)
    width = int(max(1, smooth_width))
    if width % 2 == 0:
        width += 1
    if width == 1:
        return np.maximum(raw, 1e-12)

    kernel = np.ones(width, dtype=np.float64) / float(width)
    padded = np.pad(raw.astype(np.float64), (width // 2, width // 2), mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    return np.maximum(smooth.astype(np.float32), 1e-12)


def _iter_frame_coarse_power(
    manifest: ObservationManifest,
    *,
    nfft: int,
    hop: int,
    block_rows: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (global_frame_start_row, coarse_power[256]) for the observation."""

    if nfft <= 0:
        raise ValueError("nfft must be > 0")
    if hop <= 0:
        raise ValueError("hop must be > 0")
    if block_rows <= 0:
        raise ValueError("block_rows must be > 0")

    carry = np.empty((0, 256), dtype=np.complex64)

    for block in iter_observation_blocks(manifest, block_rows=block_rows, overlap_rows=0):
        if not isinstance(block, ObservationBlock):
            raise TypeError("iter_observation_blocks yielded unexpected object.")

        decoded = decode_packed_iq(block.words)
        combined = np.concatenate((carry, decoded), axis=0)
        combined_start = block.start_row - carry.shape[0]

        frame_start = 0
        while frame_start + nfft <= combined.shape[0]:
            frame = combined[frame_start : frame_start + nfft, :]
            fft_out = np.fft.fft(frame, axis=0)
            power = np.abs(fft_out) ** 2
            coarse_power = power.mean(axis=0).astype(np.float32)
            yield combined_start + frame_start, coarse_power
            frame_start += hop

        if combined.shape[0] >= nfft:
            carry_start = frame_start
        else:
            carry_start = 0
        carry = np.array(combined[carry_start:], copy=True)


def build_streaming_waterfall(
    manifest: ObservationManifest,
    *,
    nfft: int,
    hop: int,
    block_rows: int,
    overlap_rows: int = 0,
    baseline_sample_frames: int = 128,
    baseline_smooth_width: int = 33,
    time_decimation: int = 4,
) -> StreamingWaterfallResult:
    """Build streaming two-pass waterfall/QC products for one observation.

    Pass 1 samples a bounded number of coarse-power STFT frames for baseline
    estimation and captures global frame coordinates. Pass 2 computes baseline-flattened
    waterfall/spectrum products without materializing full-resolution cubes.
    """

    if overlap_rows != 0:
        raise ValueError(
            "build_streaming_waterfall uses internal carry-buffer continuity; set overlap_rows=0."
        )
    if baseline_sample_frames <= 0:
        raise ValueError("baseline_sample_frames must be > 0")
    if time_decimation <= 0:
        raise ValueError("time_decimation must be > 0")

    sampled: list[np.ndarray] = []
    total_frames = 0
    first_start_row: int | None = None
    last_start_row: int | None = None

    for frame_start_row, coarse_power in _iter_frame_coarse_power(
        manifest, nfft=nfft, hop=hop, block_rows=block_rows
    ):
        if first_start_row is None:
            first_start_row = frame_start_row
        last_start_row = frame_start_row
        total_frames += 1
        if len(sampled) < baseline_sample_frames:
            sampled.append(coarse_power)

    if total_frames == 0:
        raise ValueError(
            "Observation is shorter than one STFT frame; increase data length or reduce nfft."
        )

    sample_array = np.stack(sampled, axis=0)
    baseline = estimate_baseline(sample_array, smooth_width=baseline_smooth_width)

    decim_buffer: list[np.ndarray] = []
    decim_rows: list[np.ndarray] = []
    decim_times: list[int] = []
    sum_excess = np.zeros(256, dtype=np.float64)
    frame_count = 0

    for frame_start_row, coarse_power in _iter_frame_coarse_power(
        manifest, nfft=nfft, hop=hop, block_rows=block_rows
    ):
        excess = coarse_power / baseline
        sum_excess += excess
        frame_count += 1

        decim_buffer.append(excess)
        if len(decim_buffer) == time_decimation:
            decim_rows.append(np.mean(np.stack(decim_buffer, axis=0), axis=0))
            decim_times.append(frame_start_row - (time_decimation - 1) * hop)
            decim_buffer.clear()

    if decim_buffer:
        decim_rows.append(np.mean(np.stack(decim_buffer, axis=0), axis=0))
        decim_times.append(last_start_row - (len(decim_buffer) - 1) * hop)

    mean_excess = (sum_excess / float(frame_count)).astype(np.float32)
    waterfall = np.stack(decim_rows, axis=0).astype(np.float32)

    waterfall_db = (10.0 * np.log10(np.maximum(waterfall, 1e-12))).astype(np.float32)
    mean_excess_db = (10.0 * np.log10(np.maximum(mean_excess, 1e-12))).astype(np.float32)

    metadata: dict[str, int | float | str] = {
        "nfft": int(nfft),
        "hop": int(hop),
        "block_rows": int(block_rows),
        "total_stft_frames": int(total_frames),
        "display_frames": int(waterfall_db.shape[0]),
        "baseline_sample_frames": int(sample_array.shape[0]),
        "first_frame_start_row": int(first_start_row),
        "last_frame_start_row": int(last_start_row),
        "time_decimation": int(time_decimation),
    }

    return StreamingWaterfallResult(
        waterfall_db=waterfall_db,
        mean_excess_db=mean_excess_db,
        baseline_power=baseline.astype(np.float32),
        display_frame_start_rows=np.asarray(decim_times, dtype=np.int64),
        metadata=metadata,
    )
