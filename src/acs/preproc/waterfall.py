"""Streaming waterfall/QC construction for one logical observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
try:
    from scipy.ndimage import median_filter
except Exception:  # pragma: no cover - scipy is optional at runtime.
    median_filter = None

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
    freq_hz_display: np.ndarray
    time_s_display: np.ndarray
    metadata: dict[str, int | float | str]


def estimate_baseline(sample_frames: np.ndarray, smooth_width: int = 33) -> np.ndarray:
    """Estimate broad per-bin bandpass from sampled frame powers.

    Parity note:
    This mirrors the reference `direct_dat_waterfall_qc_v3_optimized.py` semantics:
    per-bin median in dB over sampled frames, then broad robust smoothing in dB.
    """

    if sample_frames.ndim != 2:
        raise ValueError("sample_frames must have shape (n_frames, n_bins).")
    if sample_frames.shape[0] == 0:
        raise ValueError("sample_frames must contain at least one frame.")

    width = int(max(1, smooth_width))
    if width % 2 == 0:
        width += 1
    eps = np.float32(1e-12)

    time_median_db = np.median(10.0 * np.log10(np.maximum(sample_frames, eps)), axis=0).astype(np.float32)
    if width < 3:
        baseline_db = time_median_db
    elif median_filter is not None:
        baseline_db = median_filter(time_median_db, size=width, mode="nearest").astype(np.float32)
    else:
        baseline_db = _robust_bandpass_db_fallback(time_median_db, width_bins=width)
    baseline_power = np.power(10.0, baseline_db / 10.0).astype(np.float32)
    return np.maximum(baseline_power, eps)


def _robust_bandpass_db_fallback(time_median_db: np.ndarray, width_bins: int) -> np.ndarray:
    """Numpy-only approximation of the legacy robust dB baseline smoother."""

    if width_bins < 3:
        return time_median_db.copy()
    if width_bins % 2 == 0:
        width_bins += 1

    group = max(5, width_bins // 2)
    n = len(time_median_db)
    centers: list[float] = []
    vals: list[float] = []
    for s in range(0, n, group):
        e = min(s + group, n)
        centers.append((s + e - 1) / 2.0)
        vals.append(float(np.median(time_median_db[s:e])))

    centers_arr = np.asarray(centers, dtype=np.float64)
    vals_arr = np.asarray(vals, dtype=np.float64)
    vals2 = vals_arr.copy()
    if len(vals_arr) >= 5:
        for i in range(2, len(vals_arr) - 2):
            vals2[i] = np.median(vals_arr[i - 2 : i + 3])
    return np.interp(np.arange(n, dtype=np.float64), centers_arr, vals2).astype(np.float32)


def _iter_frame_fine_power(
    manifest: ObservationManifest,
    *,
    nfft: int,
    hop: int,
    block_rows: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (global_frame_start_row, fine_power[256*nfft]) for the observation."""

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
            power = np.abs(np.fft.fftshift(fft_out, axes=0)) ** 2
            fine_power = power.T.reshape(-1).astype(np.float32)
            yield combined_start + frame_start, fine_power
            frame_start += hop

        if combined.shape[0] >= nfft:
            carry_start = frame_start
        else:
            carry_start = 0
        carry = np.array(combined[carry_start:], copy=True)


def _compute_freq_hz_display(
    *, nfft: int, channels: int, start_freq_hz: float, coarse_channel_spacing_hz: float, sample_rate_hz: float
) -> np.ndarray:
    _ = sample_rate_hz
    fine_offsets_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / coarse_channel_spacing_hz))
    coarse_centers_hz = start_freq_hz + coarse_channel_spacing_hz * np.arange(channels, dtype=np.float64)
    return (coarse_centers_hz[:, None] + fine_offsets_hz[None, :]).reshape(-1).astype(np.float64)


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
    freq_decimation: int = 1,
    sample_rate_hz: float = 1.0,
    start_freq_hz: float = 0.0,
    coarse_channel_spacing_hz: float = 1.0,
) -> StreamingWaterfallResult:
    """Build streaming two-pass waterfall/QC products for one observation.

    Pass 1 samples a bounded number of fine-power STFT frames for baseline estimation.
    Pass 2 computes baseline-flattened fine-frequency waterfall/spectrum products and then
    optionally decimates for display.
    """

    if overlap_rows != 0:
        raise ValueError(
            "build_streaming_waterfall uses internal carry-buffer continuity; set overlap_rows=0."
        )
    if baseline_sample_frames <= 0:
        raise ValueError("baseline_sample_frames must be > 0")
    if time_decimation <= 0:
        raise ValueError("time_decimation must be > 0")
    if freq_decimation <= 0:
        raise ValueError("freq_decimation must be > 0")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    sampled: list[np.ndarray] = []
    total_frames = 0
    first_start_row: int | None = None
    last_start_row: int | None = None

    for frame_start_row, fine_power in _iter_frame_fine_power(
        manifest, nfft=nfft, hop=hop, block_rows=block_rows
    ):
        if first_start_row is None:
            first_start_row = frame_start_row
        last_start_row = frame_start_row
        total_frames += 1
        if len(sampled) < baseline_sample_frames:
            sampled.append(fine_power)

    if total_frames == 0:
        raise ValueError(
            "Observation is shorter than one STFT frame; increase data length or reduce nfft."
        )

    sample_array = np.stack(sampled, axis=0)
    baseline = estimate_baseline(sample_array, smooth_width=baseline_smooth_width)

    fine_bins = 256 * nfft
    sum_excess = np.zeros(fine_bins, dtype=np.float64)
    frame_count = 0

    decim_buffer: list[np.ndarray] = []
    decim_rows: list[np.ndarray] = []
    decim_start_rows: list[int] = []
    decim_time_s: list[float] = []

    for frame_start_row, fine_power in _iter_frame_fine_power(
        manifest, nfft=nfft, hop=hop, block_rows=block_rows
    ):
        excess = fine_power / baseline
        sum_excess += excess
        frame_count += 1

        decim_buffer.append(excess)
        if len(decim_buffer) == time_decimation:
            decim_rows.append(np.mean(np.stack(decim_buffer, axis=0), axis=0))
            first_row = frame_start_row - (time_decimation - 1) * hop
            decim_start_rows.append(int(first_row))
            group_starts = first_row + hop * np.arange(time_decimation, dtype=np.float64)
            center_rows = group_starts + (nfft / 2.0)
            decim_time_s.append(float(np.mean(center_rows) / sample_rate_hz))
            decim_buffer.clear()

    if decim_buffer:
        decim_rows.append(np.mean(np.stack(decim_buffer, axis=0), axis=0))
        first_row = last_start_row - (len(decim_buffer) - 1) * hop
        decim_start_rows.append(int(first_row))
        group_starts = first_row + hop * np.arange(len(decim_buffer), dtype=np.float64)
        center_rows = group_starts + (nfft / 2.0)
        decim_time_s.append(float(np.mean(center_rows) / sample_rate_hz))

    mean_excess = (sum_excess / float(frame_count)).astype(np.float32)
    waterfall = np.stack(decim_rows, axis=0).astype(np.float32)
    mean_excess_db_full = (10.0 * np.log10(np.maximum(mean_excess, 1e-12))).astype(np.float32)
    waterfall_db_full = (10.0 * np.log10(np.maximum(waterfall, 1e-12))).astype(np.float32)

    if freq_decimation > 1:
        usable_bins = (fine_bins // freq_decimation) * freq_decimation
        waterfall_db = (
            waterfall_db_full[:, :usable_bins]
            .reshape(waterfall_db_full.shape[0], -1, freq_decimation)
            .mean(axis=2)
            .astype(np.float32)
        )
        mean_excess_db = (
            mean_excess_db_full[:usable_bins]
            .reshape(-1, freq_decimation)
            .mean(axis=1)
            .astype(np.float32)
        )
    else:
        waterfall_db = waterfall_db_full
        mean_excess_db = mean_excess_db_full

    freq_hz = _compute_freq_hz_display(
        nfft=nfft,
        channels=256,
        start_freq_hz=float(start_freq_hz),
        coarse_channel_spacing_hz=float(coarse_channel_spacing_hz),
        sample_rate_hz=float(sample_rate_hz),
    )
    if freq_decimation > 1:
        usable_bins = (freq_hz.shape[0] // freq_decimation) * freq_decimation
        freq_hz_display = freq_hz[:usable_bins].reshape(-1, freq_decimation).mean(axis=1)
    else:
        freq_hz_display = freq_hz

    time_s_display = np.asarray(decim_time_s, dtype=np.float64)

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
        "freq_decimation": int(freq_decimation),
        "fine_bins_total": int(fine_bins),
        "display_freq_bins": int(waterfall_db.shape[1]),
        "display_time_start_s": float(time_s_display[0]),
        "display_time_end_s": float(time_s_display[-1]),
        "display_freq_start_hz": float(freq_hz_display[0]),
        "display_freq_end_hz": float(freq_hz_display[-1]),
        "sample_rate_hz": float(sample_rate_hz),
        "start_freq_hz": float(start_freq_hz),
        "coarse_channel_spacing_hz": float(coarse_channel_spacing_hz),
        "fine_bin_spacing_hz": float(coarse_channel_spacing_hz / nfft),
        "frame_time_origin": "stft_window_center",
        "display_decimation_domain": "dB",
    }

    return StreamingWaterfallResult(
        waterfall_db=waterfall_db,
        mean_excess_db=mean_excess_db,
        baseline_power=baseline.astype(np.float32),
        display_frame_start_rows=np.asarray(decim_start_rows, dtype=np.int64),
        freq_hz_display=freq_hz_display,
        time_s_display=time_s_display,
        metadata=metadata,
    )
