#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
direct_dat_waterfall_qc_v3_optimized.py

Memory-optimized QC/visualization script for FAST / SERENDIP-style channelized
.dat files. This version is designed to run on multi-minute, multi-GB files in
Spyder without materializing the full decoded complex matrix or the full
waterfall cube in RAM.

Core design change
------------------
The original implementation built these large in-memory objects:
    1) full decoded complex array       : (rows, channels)
    2) full linear-power waterfall      : (n_frames, channels*nfft)
    3) full normalized waterfall in dB  : (n_frames, channels*nfft)

That is practical for short test files, but not for real 2 GB / 4 min .dat
files. This optimized version switches to a two-pass streaming design:
    - read/decode only one STFT frame at a time from a raw memmap
    - estimate the bandpass baseline from a bounded set of sampled frames
    - store only the display-ready waterfall, not the full-resolution 2D cube
    - keep full-resolution 1D spectra (mean_excess_db, baseline_db, freq_hz),
      which are cheap

Result: peak RAM is reduced from multi-GB / tens-of-GB territory to roughly the
size of:
    - one STFT frame
    - one baseline sample bank
    - one display waterfall
which is typically hundreds of MB rather than many GB.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import median_filter
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# =============================================================================
# DEFAULT HARDWARE / PYWU GEOMETRY
# =============================================================================
LO_HZ_DEFAULT = 1_000_000_000.0
FS_HZ_DEFAULT = 1_000_000_000.0
FFT_POINT_DEFAULT = 65536
START_CH_DEFAULT = 27392
CHANNELS_DEFAULT = 256
BYTES_PER_SAMPLE_DEFAULT = 2

# =============================================================================
# DEFAULT DSP SETTINGS
# =============================================================================
NFFT_DEFAULT = 2048
HOP_DEFAULT = 1024
WINDOW_DEFAULT = "hann"

# Robust bandpass baseline settings
BASELINE_FILTER_BINS_DEFAULT = 1001   # should be odd
BASELINE_SAMPLE_FRAMES_DEFAULT = 64   # bounded-memory approximation to time median
EPS_DEFAULT = 1e-12

# Display defaults
DISPLAY_VMIN_PERCENTILE_DEFAULT = 1.0
DISPLAY_VMAX_PERCENTILE_DEFAULT = 99.7
DISPLAY_DECIMATE_DEFAULT = 4096
COLORMAP_DEFAULT = "viridis"

# Injection / streaming utility defaults
RMS_CHUNK_ROWS_DEFAULT = 262144

# Optional reference marker
HI_LINE_HZ = 1420.40575177e6

# =============================================================================
# SPYDER / IPYTHON DIRECT-RUN SETTINGS
# =============================================================================
SPYDER_RUN_CONFIG: Dict[str, Any] = {
    "dat": "serendip6_m13_1.05G-1.45G_MB_01_00_20230511_165609_868843681_raw_2s.dat",
    "json": None,
    "out": None,
    "save_npy_prefix": None,
    "show": True,
    "title": None,
    "display_f0_hz": None,
    "display_span_hz": None,
    "inject": False,
    "inject_start_freq_hz": HI_LINE_HZ,
    "inject_drift_hz_per_s": 35.0,
    "inject_start_time_s": 0.10,
    "inject_duration_s": 1.8,
    "inject_snr_db": 18.0,
    "inject_edge_taper_s": None,
    "inject_bandwidth_hz": 5.0,
    "baseline_sample_frames": BASELINE_SAMPLE_FRAMES_DEFAULT,
    "rms_chunk_rows": RMS_CHUNK_ROWS_DEFAULT,
}

def make_output_path(dat_path: Path, out_arg: Optional[str]) -> Path:
    """
    自动生成与输入 DAT 文件名关联的输出 PDF 路径。

    规则：
    1. 如果用户显式给了 --out，就优先使用它
    2. 如果没给，就默认保存为：
       原DAT文件名_stem + "_waterfall_qc.pdf"
    3. 默认保存在 DAT 文件所在目录
    """
    if out_arg:
        out_path = Path(out_arg)
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".pdf")
        return out_path

    return dat_path.with_name(f"{dat_path.stem}_waterfall_qc.pdf")
# =============================================================================
# ARGPARSE / RUNTIME SETUP
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Memory-optimized QC waterfall / mean-excess-power plotter for FAST/SERENDIP-style .dat files."
    )
    p.add_argument("--dat", default=None, help="Path to the input .dat file")
    p.add_argument("--json", default=None, help="Optional JSON sidecar path")
    p.add_argument("--out", default=None, help="Optional output figure path (e.g. .pdf, .png)")
    p.add_argument("--save-npy-prefix", default=None, help="Optional prefix to save processed arrays (.npy)")

    # Geometry
    p.add_argument("--lo-hz", type=float, default=LO_HZ_DEFAULT)
    p.add_argument("--fs-hz", type=float, default=FS_HZ_DEFAULT)
    p.add_argument("--fft-point", type=int, default=FFT_POINT_DEFAULT)
    p.add_argument("--start-ch", type=int, default=START_CH_DEFAULT)
    p.add_argument("--channels", type=int, default=CHANNELS_DEFAULT)
    p.add_argument("--bytes-per-sample", type=int, default=BYTES_PER_SAMPLE_DEFAULT)

    # STFT
    p.add_argument("--nfft", type=int, default=NFFT_DEFAULT)
    p.add_argument("--hop", type=int, default=HOP_DEFAULT)
    p.add_argument("--window", type=str, default=WINDOW_DEFAULT,
                   choices=["rect", "hann", "hamming", "blackman"])

    # Display / baseline / streaming
    p.add_argument("--baseline-filter-bins", type=int, default=BASELINE_FILTER_BINS_DEFAULT)
    p.add_argument("--baseline-sample-frames", type=int, default=BASELINE_SAMPLE_FRAMES_DEFAULT,
                   help="Number of representative frames used to approximate the time-median bandpass baseline")
    p.add_argument("--display-vmin-percentile", type=float, default=DISPLAY_VMIN_PERCENTILE_DEFAULT)
    p.add_argument("--display-vmax-percentile", type=float, default=DISPLAY_VMAX_PERCENTILE_DEFAULT)
    p.add_argument("--display-decimate", type=int, default=DISPLAY_DECIMATE_DEFAULT)
    p.add_argument("--colormap", type=str, default=COLORMAP_DEFAULT)
    p.add_argument("--rms-chunk-rows", type=int, default=RMS_CHUNK_ROWS_DEFAULT,
                   help="Chunk length used when estimating target-channel RMS for streaming injection")

    # Optional zoom region
    p.add_argument("--display-f0-hz", type=float, default=None,
                   help="Optional center frequency for display zoom. If omitted, show full band.")
    p.add_argument("--display-span-hz", type=float, default=None,
                   help="Optional total frequency span for display zoom. Used only if --display-f0-hz is set.")

    # Synthetic signal injection
    p.add_argument("--inject", action="store_true", help="Inject a synthetic drifting narrowband CW into the data before plotting")
    p.add_argument("--inject-start-freq-hz", type=float, default=HI_LINE_HZ)
    p.add_argument("--inject-drift-hz-per-s", type=float, default=30.0)
    p.add_argument("--inject-start-time-s", type=float, default=0.0)
    p.add_argument("--inject-duration-s", type=float, default=None)
    p.add_argument("--inject-snr-db", type=float, default=12.0)
    p.add_argument("--inject-edge-taper-s", type=float, default=None)
    p.add_argument("--inject-bandwidth-hz", type=float, default=0.0,
                   help="Intrinsic injected signal bandwidth in Hz. 0 => pure single-tone CW")

    # Plot cosmetics
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--show", action="store_true", help="Display the figure interactively")
    return p



def running_in_spyder_or_ipython() -> bool:
    return (
        "SPYDER_ARGS" in os.environ
        or "spyder_kernels" in sys.modules
        or "ipykernel" in sys.modules
        or "IPython" in sys.modules
    )



def namespace_from_dict(base: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    data = vars(base).copy()
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        data[key] = value
    return argparse.Namespace(**data)



def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(argv)

    if args.dat:
        if unknown:
            print(f"[INFO] Ignoring unknown arguments: {unknown}")
        return args

    if running_in_spyder_or_ipython():
        args = namespace_from_dict(args, SPYDER_RUN_CONFIG)
        if args.dat:
            if unknown:
                print(f"[INFO] Ignoring unknown IDE arguments: {unknown}")
            return args

    parser.error(
        "missing input .dat file. Use --dat in terminal mode, or set "
        "SPYDER_RUN_CONFIG['dat'] when running directly inside Spyder/IPython."
    )


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def get_window(name: str, n: int) -> np.ndarray:
    name = name.lower()
    if name == "rect":
        return np.ones(n, dtype=np.float32)
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "blackman":
        return np.blackman(n).astype(np.float32)
    raise ValueError(f"Unsupported window: {name}")



def file_size_bytes(path: Path) -> int:
    return path.stat().st_size



def parse_filename_metadata(path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"filename": path.name}
    stem = path.stem
    parts = stem.split("_")

    if len(parts) >= 8:
        meta["software"] = parts[0]
        meta["source"] = parts[1]
        meta["band_label"] = parts[2]
        meta["receiver"] = parts[3]

    beam_match = re.search(r"_MB_(\d+)_(\d+)_([0-9]{8})_([0-9]{6})_([0-9]+)", stem)
    if beam_match:
        meta["beam_1based"] = int(beam_match.group(1))
        meta["pol"] = int(beam_match.group(2))
        meta["date_yyyymmdd"] = beam_match.group(3)
        meta["time_hhmmss"] = beam_match.group(4)
        meta["nanosec"] = int(beam_match.group(5))
    if "beam_1based" in meta:
        meta["beam_0based"] = meta["beam_1based"] - 1
    return meta



def load_json_sidecar(json_path: Optional[Path]) -> Dict[str, Any]:
    if json_path is None:
        return {"json_loaded": False}
    if not json_path.exists():
        return {"json_loaded": False, "json_error": f"File does not exist: {json_path}"}

    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception as e:
        return {"json_loaded": False, "json_error": f"Failed to parse JSON: {e}"}

    summary: Dict[str, Any] = {"json_loaded": True, "json_type": type(obj).__name__}

    def count_coord_like(node: Any) -> int:
        count = 0
        if isinstance(node, dict):
            keys = set(k.lower() for k in node.keys())
            if ("ra" in keys and "dec" in keys) or ("time" in keys and "ra" in keys):
                count += 1
            for v in node.values():
                count += count_coord_like(v)
        elif isinstance(node, list):
            for item in node:
                count += count_coord_like(item)
        return count

    summary["coord_like_records"] = count_coord_like(obj)
    summary["json_preview"] = str(obj)[:300]
    summary["raw"] = obj
    return summary



def validate_geometry(dat_path: Path, channels: int, bytes_per_sample: int) -> Tuple[int, int]:
    total_bytes = file_size_bytes(dat_path)
    bytes_per_row = channels * bytes_per_sample
    if total_bytes % bytes_per_row != 0:
        raise ValueError(
            f"Illegal file size for the requested geometry: "
            f"{total_bytes} bytes is not divisible by {bytes_per_row} bytes/row"
        )
    rows = total_bytes // bytes_per_row
    return total_bytes, rows



def open_raw_u16_memmap(dat_path: Path, rows: int, channels: int) -> np.memmap:
    return np.memmap(dat_path, dtype="<u2", mode="r", shape=(rows, channels))



def decode_u16_block_to_complex(u16_block: np.ndarray) -> np.ndarray:
    i_part = (u16_block & 0x00FF).astype(np.uint8).view(np.int8).astype(np.float32)
    q_part = ((u16_block >> 8) & 0x00FF).astype(np.uint8).view(np.int8).astype(np.float32)
    return i_part + 1j * q_part



def coarse_centers_hz(lo_hz: float, fs_hz: float, fft_point: int, start_ch: int, channels: int) -> np.ndarray:
    coarse_df = fs_hz / fft_point
    return lo_hz + (start_ch + np.arange(channels)) * coarse_df



def fine_frequency_axis_hz(lo_hz: float, fs_hz: float, fft_point: int, start_ch: int,
                           channels: int, nfft: int) -> np.ndarray:
    coarse_df = fs_hz / fft_point
    ccent = coarse_centers_hz(lo_hz, fs_hz, fft_point, start_ch, channels)
    fine_offsets = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / coarse_df))
    return (ccent[:, None] + fine_offsets[None, :]).reshape(-1).astype(np.float64)



def frame_time_axis_s(n_frames: int, hop: int, nfft: int, coarse_sr_hz: float) -> np.ndarray:
    return ((np.arange(n_frames) * hop) + (nfft / 2.0)) / coarse_sr_hz



def robust_bandpass_db(time_median_db: np.ndarray, width_bins: int) -> np.ndarray:
    width_bins = int(width_bins)
    if width_bins < 3:
        return time_median_db.copy()
    if width_bins % 2 == 0:
        width_bins += 1

    if HAVE_SCIPY:
        return median_filter(time_median_db, size=width_bins, mode="nearest")

    group = max(5, width_bins // 2)
    n = len(time_median_db)
    centers = []
    vals = []
    for s in range(0, n, group):
        e = min(s + group, n)
        centers.append((s + e - 1) / 2.0)
        vals.append(np.median(time_median_db[s:e]))
    centers = np.array(centers)
    vals = np.array(vals)
    vals2 = vals.copy()
    if len(vals) >= 5:
        for i in range(2, len(vals) - 2):
            vals2[i] = np.median(vals[i - 2:i + 3])
    return np.interp(np.arange(n), centers, vals2)



def choose_display_region(freq_hz: np.ndarray, f0_hz: Optional[float], span_hz: Optional[float]) -> np.ndarray:
    if f0_hz is None:
        return np.ones_like(freq_hz, dtype=bool)
    if span_hz is None:
        raise ValueError("If --display-f0-hz is supplied, you must also supply --display-span-hz.")
    half = span_hz / 2.0
    return (freq_hz >= f0_hz - half) & (freq_hz <= f0_hz + half)



def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.nfft <= 0 or args.hop <= 0:
        raise ValueError("nfft and hop must both be positive integers")
    if args.channels <= 0 or args.bytes_per_sample <= 0:
        raise ValueError("channels and bytes_per_sample must both be positive")
    if args.display_decimate <= 0:
        raise ValueError("display_decimate must be >= 1")
    if args.display_vmin_percentile >= args.display_vmax_percentile:
        raise ValueError("display_vmin_percentile must be smaller than display_vmax_percentile")
    if args.inject_duration_s is not None and args.inject_duration_s <= 0:
        raise ValueError("inject_duration_s must be positive when provided")
    if args.baseline_sample_frames <= 0:
        raise ValueError("baseline_sample_frames must be >= 1")
    if args.rms_chunk_rows <= 0:
        raise ValueError("rms_chunk_rows must be >= 1")


# =============================================================================
# INJECTION (STREAMING-SAFE)
# =============================================================================

def _raised_cosine_envelope_asymmetric(
    local_t: np.ndarray,
    duration_s: float,
    attack_s: float,
    release_s: float,
) -> np.ndarray:
    env = np.ones_like(local_t, dtype=np.float32)

    attack_s = float(max(0.0, min(attack_s, duration_s / 2.0)))
    release_s = float(max(0.0, min(release_s, duration_s / 2.0)))

    if attack_s > 0.0:
        lead = local_t < attack_s
        if np.any(lead):
            env[lead] = 0.5 * (1.0 - np.cos(np.pi * local_t[lead] / attack_s))

    if release_s > 0.0:
        remain = duration_s - local_t
        trail = remain < release_s
        if np.any(trail):
            env[trail] = np.minimum(env[trail], 0.5 * (1.0 - np.cos(np.pi * remain[trail] / release_s)))

    return env



def compute_channel_rms_from_raw_u16(
    raw_u16: np.ndarray,
    rows: int,
    channel: int,
    chunk_rows: int,
) -> float:
    sumsq = 0.0
    count = 0
    for s in range(0, rows, chunk_rows):
        e = min(rows, s + chunk_rows)
        chunk = raw_u16[s:e, channel]
        i_part = (chunk & 0x00FF).astype(np.uint8).view(np.int8).astype(np.float32)
        q_part = ((chunk >> 8) & 0x00FF).astype(np.uint8).view(np.int8).astype(np.float32)
        sumsq += float(np.sum(i_part * i_part + q_part * q_part, dtype=np.float64))
        count += (e - s)
    if count == 0:
        raise ValueError("Cannot estimate RMS from empty file")
    return float(np.sqrt(sumsq / count))



def prepare_injection_plan(
    raw_u16: np.ndarray,
    rows: int,
    lo_hz: float,
    fs_hz: float,
    fft_point: int,
    start_ch: int,
    channels: int,
    start_frequency_hz: float,
    drift_rate_hz_per_s: float,
    insertion_start_time_s: float,
    bandwidth_hz: float,
    signal_duration_s: Optional[float],
    snr_db: float,
    rms_chunk_rows: int,
    edge_taper_s: Optional[float] = None,
    reference_nfft: Optional[int] = None,
) -> Dict[str, Any]:
    coarse_sr_hz = fs_hz / fft_point
    coarse_centers = coarse_centers_hz(lo_hz, fs_hz, fft_point, start_ch, channels)
    ch = int(np.argmin(np.abs(coarse_centers - start_frequency_hz)))
    f0_baseband_hz = float(start_frequency_hz - coarse_centers[ch])

    start_idx = max(0, int(round(insertion_start_time_s * coarse_sr_hz)))
    if signal_duration_s is None:
        end_idx = rows
    else:
        end_idx = min(rows, start_idx + int(round(signal_duration_s * coarse_sr_hz)))
    if start_idx >= end_idx:
        raise ValueError("Injection interval is empty. Check insertion_start_time_s and signal_duration_s.")

    sigma = compute_channel_rms_from_raw_u16(raw_u16, rows, ch, rms_chunk_rows)
    amp = float(sigma * (10.0 ** (snr_db / 20.0)))

    active_duration_s = (end_idx - start_idx) / coarse_sr_hz
    if edge_taper_s is None:
        if reference_nfft is not None:
            frame_len_s = reference_nfft / coarse_sr_hz
            edge_taper_s = min(active_duration_s / 4.0, frame_len_s)
        else:
            edge_taper_s = active_duration_s / 4.0

    taper_attack = float(edge_taper_s if start_idx > 0 else 0.0)
    taper_release = float(edge_taper_s if end_idx < rows else 0.0)

    plan: Dict[str, Any] = {
        "target_coarse_channel": ch,
        "target_coarse_center_hz": float(coarse_centers[ch]),
        "injected_start_freq_hz": float(start_frequency_hz),
        "injected_end_freq_hz": float(start_frequency_hz + drift_rate_hz_per_s * max(0.0, active_duration_s - (1.0 / coarse_sr_hz))),
        "f0_baseband_hz": f0_baseband_hz,
        "drift_rate_hz_per_s": float(drift_rate_hz_per_s),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "snr_db": float(snr_db),
        "estimated_channel_rms": float(sigma),
        "injected_amplitude": float(amp),
        "edge_taper_s": float(edge_taper_s),
        "taper_attack_s": taper_attack,
        "taper_release_s": taper_release,
        "active_duration_s": float(active_duration_s),
        "coarse_sr_hz": float(coarse_sr_hz),
        "bandwidth_hz": float(bandwidth_hz),
    }

    if bandwidth_hz > 0.0:
        n_subtones = 21
        offsets = np.linspace(-bandwidth_hz / 2.0, bandwidth_hz / 2.0, n_subtones, dtype=np.float64)
        sigma_bw = bandwidth_hz / 2.355
        weights = np.exp(-0.5 * (offsets / max(sigma_bw, 1e-12)) ** 2)
        weights = weights / np.sqrt(np.sum(weights ** 2))
        rng = np.random.default_rng(12345)
        phases0 = rng.uniform(0.0, 2.0 * np.pi, size=n_subtones)
        plan["offsets_hz"] = offsets.astype(np.float64)
        plan["weights"] = weights.astype(np.float32)
        plan["phases0"] = phases0.astype(np.float64)

    return plan



def apply_injection_to_segment(seg: np.ndarray, seg_start_idx: int, plan: Dict[str, Any]) -> None:
    seg_len = seg.shape[0]
    seg_end_idx = seg_start_idx + seg_len

    ov_start = max(seg_start_idx, plan["start_idx"])
    ov_end = min(seg_end_idx, plan["end_idx"])
    if ov_start >= ov_end:
        return

    ch = int(plan["target_coarse_channel"])
    coarse_sr_hz = float(plan["coarse_sr_hz"])
    amp = float(plan["injected_amplitude"])
    f0_baseband_hz = float(plan["f0_baseband_hz"])
    drift_rate_hz_per_s = float(plan["drift_rate_hz_per_s"])

    local_sample_idx = np.arange(ov_start, ov_end, dtype=np.int64)
    local_t = (local_sample_idx - plan["start_idx"]) / coarse_sr_hz

    env = _raised_cosine_envelope_asymmetric(
        local_t=local_t.astype(np.float32),
        duration_s=float(plan["active_duration_s"]),
        attack_s=float(plan["taper_attack_s"]),
        release_s=float(plan["taper_release_s"]),
    )

    if plan["bandwidth_hz"] <= 0.0:
        phase = 2.0 * np.pi * (
            f0_baseband_hz * local_t + 0.5 * drift_rate_hz_per_s * (local_t ** 2)
        )
        injected = amp * np.exp(1j * phase)
    else:
        offsets = plan["offsets_hz"][:, None]
        weights = plan["weights"][:, None]
        phases0 = plan["phases0"][:, None]
        local_t2 = local_t[None, :]
        phase = 2.0 * np.pi * (
            (f0_baseband_hz + offsets) * local_t2
            + 0.5 * drift_rate_hz_per_s * (local_t2 ** 2)
        ) + phases0
        injected = amp * np.sum(weights * np.exp(1j * phase), axis=0)

    injected = (injected * env).astype(np.complex64)
    dst_s = ov_start - seg_start_idx
    dst_e = ov_end - seg_start_idx
    seg[dst_s:dst_e, ch] += injected


# =============================================================================
# STREAMING WATERFALL CORE
# =============================================================================

def build_display_geometry(
    freq_hz: np.ndarray,
    display_mask: np.ndarray,
    requested_display_decimate: int,
) -> Dict[str, Any]:
    selected_idx = np.flatnonzero(display_mask)
    if selected_idx.size == 0:
        raise ValueError("Display selection is empty. Adjust display_f0_hz / display_span_hz.")

    effective_decimate = int(max(1, requested_display_decimate))
    if selected_idx.size < effective_decimate:
        effective_decimate = 1

    usable = (selected_idx.size // effective_decimate) * effective_decimate
    if usable == 0:
        effective_decimate = 1
        usable = selected_idx.size

    selected_idx = selected_idx[:usable]
    if effective_decimate == 1:
        groups = selected_idx.reshape(-1, 1)
    else:
        groups = selected_idx.reshape(-1, effective_decimate)

    freq_disp_hz = freq_hz[groups].mean(axis=1)
    return {
        "selected_idx": selected_idx,
        "groups": groups,
        "freq_disp_hz": freq_disp_hz,
        "effective_display_decimate": effective_decimate,
    }



def estimate_memory_usage(
    rows: int,
    channels: int,
    nfft: int,
    n_frames: int,
    n_bins: int,
    n_display_bins: int,
    baseline_sample_frames: int,
) -> Dict[str, float]:
    legacy_decoded_complex = rows * channels * np.dtype(np.complex64).itemsize
    legacy_full_2d = n_frames * n_bins * np.dtype(np.float32).itemsize
    legacy_total_very_rough = legacy_decoded_complex + 3.0 * legacy_full_2d

    streaming_frame_decode = nfft * channels * np.dtype(np.complex64).itemsize
    streaming_baseline_bank = baseline_sample_frames * n_bins * np.dtype(np.float32).itemsize
    streaming_display_waterfall = n_frames * n_display_bins * np.dtype(np.float32).itemsize
    streaming_1d_vectors = 5 * n_bins * np.dtype(np.float32).itemsize
    streaming_peak_rough = (
        streaming_frame_decode + streaming_baseline_bank + streaming_display_waterfall + streaming_1d_vectors
    )

    return {
        "legacy_decoded_complex_gib": legacy_decoded_complex / (1024 ** 3),
        "legacy_single_2d_waterfall_gib": legacy_full_2d / (1024 ** 3),
        "legacy_total_rough_gib": legacy_total_very_rough / (1024 ** 3),
        "streaming_baseline_bank_gib": streaming_baseline_bank / (1024 ** 3),
        "streaming_display_waterfall_gib": streaming_display_waterfall / (1024 ** 3),
        "streaming_peak_rough_gib": streaming_peak_rough / (1024 ** 3),
    }



def frame_power_from_raw(
    raw_u16: np.ndarray,
    frame_start: int,
    nfft: int,
    window_2d: np.ndarray,
    injection_plan: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    u16_block = raw_u16[frame_start:frame_start + nfft, :]
    seg = decode_u16_block_to_complex(u16_block)
    if injection_plan is not None:
        apply_injection_to_segment(seg, frame_start, injection_plan)

    seg -= seg.mean(axis=0, keepdims=True)
    spec = np.fft.fftshift(np.fft.fft(seg * window_2d, axis=0), axes=0)
    pwr = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
    return pwr.T.reshape(-1)



def choose_baseline_sample_frame_ids(n_frames: int, baseline_sample_frames: int) -> np.ndarray:
    n_keep = min(max(1, baseline_sample_frames), n_frames)
    ids = np.linspace(0, n_frames - 1, num=n_keep, dtype=np.int64)
    return np.unique(ids)



def build_waterfall_streaming(
    raw_u16: np.ndarray,
    rows: int,
    lo_hz: float,
    fs_hz: float,
    fft_point: int,
    start_ch: int,
    channels: int,
    nfft: int,
    hop: int,
    window_name: str,
    baseline_filter_bins: int,
    baseline_sample_frames: int,
    display_mask: np.ndarray,
    display_decimate: int,
    eps: float = EPS_DEFAULT,
    injection_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    coarse_sr_hz = fs_hz / fft_point
    if rows < nfft:
        raise ValueError(f"File too short for NFFT={nfft}: only {rows} rows")

    n_frames = 1 + (rows - nfft) // hop
    n_bins = channels * nfft
    win = get_window(window_name, nfft)
    window_2d = win[:, None]
    time_s = frame_time_axis_s(n_frames, hop, nfft, coarse_sr_hz)
    freq_hz = fine_frequency_axis_hz(lo_hz, fs_hz, fft_point, start_ch, channels, nfft)

    display_geom = build_display_geometry(freq_hz, display_mask, display_decimate)
    display_groups = display_geom["groups"]
    freq_disp_hz = display_geom["freq_disp_hz"]
    effective_display_decimate = display_geom["effective_display_decimate"]
    n_display_bins = len(freq_disp_hz)

    sample_ids = choose_baseline_sample_frame_ids(n_frames, baseline_sample_frames)
    baseline_samples = np.empty((len(sample_ids), n_bins), dtype=np.float32)
    sample_ptr = 0

    for fi in range(n_frames):
        frame_start = fi * hop
        pwr_flat = frame_power_from_raw(raw_u16, frame_start, nfft, window_2d, injection_plan)
        if sample_ptr < len(sample_ids) and fi == sample_ids[sample_ptr]:
            baseline_samples[sample_ptr, :] = 10.0 * np.log10(pwr_flat + eps)
            sample_ptr += 1

    time_median_db = np.median(baseline_samples, axis=0).astype(np.float32)
    baseline_db = robust_bandpass_db(time_median_db, baseline_filter_bins).astype(np.float32)
    baseline_lin = np.power(10.0, baseline_db / 10.0).astype(np.float32)
    baseline_den = baseline_lin + np.float32(eps)

    mean_norm_sum = np.zeros(n_bins, dtype=np.float64)
    display_waterfall_db = np.empty((n_frames, n_display_bins), dtype=np.float32)
    norm_work = np.empty(n_bins, dtype=np.float32)
    db_work = np.empty(n_bins, dtype=np.float32)

    injection_qc: Optional[Dict[str, float]] = None
    if injection_plan is not None:
        injection_qc = {
            "track_sum": 0.0,
            "track_count": 0.0,
            "off_sum": 0.0,
            "off_count": 0.0,
        }

    for fi in range(n_frames):
        frame_start = fi * hop
        pwr_flat = frame_power_from_raw(raw_u16, frame_start, nfft, window_2d, injection_plan)
        np.divide(pwr_flat, baseline_den, out=norm_work)
        mean_norm_sum += norm_work

        np.copyto(db_work, norm_work)
        db_work += np.float32(eps)
        np.log10(db_work, out=db_work)
        db_work *= np.float32(10.0)

        display_waterfall_db[fi, :] = db_work[display_groups].mean(axis=1, dtype=np.float32)

        if injection_qc is not None:
            track_hz = (
                float(injection_plan["injected_start_freq_hz"])
                + float(injection_plan["drift_rate_hz_per_s"]) * (time_s[fi] - time_s[0])
            )
            b = int(np.argmin(np.abs(freq_hz - track_hz)))
            injection_qc["track_sum"] += float(db_work[b])
            injection_qc["track_count"] += 1.0
            for d in range(-8, 9):
                if abs(d) <= 2:
                    continue
                j = b + d
                if 0 <= j < n_bins:
                    injection_qc["off_sum"] += float(db_work[j])
                    injection_qc["off_count"] += 1.0

    mean_excess_db = (10.0 * np.log10((mean_norm_sum / n_frames) + eps)).astype(np.float32)
    display_mean_excess_db = mean_excess_db[display_groups].mean(axis=1, dtype=np.float32)

    memory_report = estimate_memory_usage(
        rows=rows,
        channels=channels,
        nfft=nfft,
        n_frames=n_frames,
        n_bins=n_bins,
        n_display_bins=n_display_bins,
        baseline_sample_frames=len(sample_ids),
    )

    return {
        "waterfall_db_display": display_waterfall_db,
        "mean_excess_db": mean_excess_db,
        "mean_excess_db_display": display_mean_excess_db,
        "baseline_db": baseline_db,
        "freq_hz": freq_hz,
        "freq_hz_display": freq_disp_hz,
        "time_s": time_s,
        "coarse_sr_hz": np.array([coarse_sr_hz], dtype=np.float64),
        "effective_display_decimate": int(effective_display_decimate),
        "baseline_sample_frames_used": int(len(sample_ids)),
        "memory_report": memory_report,
        "injection_qc": injection_qc,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_dual_panel(
    waterfall_db: np.ndarray,
    mean_excess_db: np.ndarray,
    freq_hz: np.ndarray,
    time_s: np.ndarray,
    vmin_percentile: float,
    vmax_percentile: float,
    display_decimate: int,
    colormap: str,
    title: str,
    f0_ref_hz: Optional[float] = None,
    display_mask: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    # This function is kept backward-friendly, but the optimized pipeline calls
    # it with display-ready arrays and display_decimate=1.
    if display_mask is None:
        display_mask = np.ones_like(freq_hz, dtype=bool)

    freq_sel = freq_hz[display_mask]
    W_sel = waterfall_db[:, display_mask] if waterfall_db.ndim == 2 and waterfall_db.shape[1] == len(freq_hz) else waterfall_db
    M_sel = mean_excess_db[display_mask] if mean_excess_db.ndim == 1 and len(mean_excess_db) == len(freq_hz) else mean_excess_db

    if display_decimate > 1 and W_sel.shape[1] == len(freq_sel):
        usable = (len(freq_sel) // display_decimate) * display_decimate
        freq_sel = freq_sel[:usable].reshape(-1, display_decimate).mean(axis=1)
        W_sel = W_sel[:, :usable].reshape(W_sel.shape[0], usable // display_decimate, display_decimate).mean(axis=2)
        M_sel = M_sel[:usable].reshape(-1, display_decimate).mean(axis=1)

    vmin = float(np.percentile(W_sel, vmin_percentile))
    vmax = float(np.percentile(W_sel, vmax_percentile))

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.15], hspace=0.18)

    if len(freq_sel) > 1:
        x0 = freq_sel[0] / 1e3
        x1 = freq_sel[-1] / 1e3
    else:
        dx = 0.5
        x0 = freq_sel[0] / 1e3 - dx
        x1 = freq_sel[0] / 1e3 + dx

    if len(time_s) > 1:
        y0 = time_s[0]
        y1 = time_s[-1]
    else:
        dt = 0.5
        y0 = time_s[0] - dt
        y1 = time_s[0] + dt

    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(
        W_sel,
        aspect="auto",
        origin="lower",
        extent=[x0, x1, y0, y1],
        vmin=vmin,
        vmax=vmax,
        cmap=colormap,
        interpolation="nearest",
    )
    if f0_ref_hz is not None and freq_sel[0] <= f0_ref_hz <= freq_sel[-1]:
        ax0.axvline(f0_ref_hz / 1e3, linestyle="--", linewidth=1.0)
    ax0.set_ylabel("Time (s)")
    ax0.set_title(title)
    ax0.ticklabel_format(axis="x", style="plain", useOffset=False)
    cbar = fig.colorbar(im, ax=ax0)
    cbar.set_label("Bandpass-flattened power (dB)")

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(freq_sel / 1e3, M_sel, linewidth=1.0, label="Mean excess power")
    if f0_ref_hz is not None and freq_sel[0] <= f0_ref_hz <= freq_sel[-1]:
        ax1.axvline(f0_ref_hz / 1e3, linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("Mean excess\npower (dB)")
    ax1.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax1.legend(loc="best")
    ax1.grid(alpha=0.2)

    if out_path is not None:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    validate_runtime_args(args)

    dat_path = Path(args.dat)
    json_path = Path(args.json) if args.json else None
    if not dat_path.exists():
        raise FileNotFoundError(f"DAT file not found: {dat_path}")

    total_bytes, rows = validate_geometry(dat_path, args.channels, args.bytes_per_sample)
    fn_meta = parse_filename_metadata(dat_path)
    json_meta = load_json_sidecar(json_path)

    coarse_sr_hz = args.fs_hz / args.fft_point
    dt_native_s = 1.0 / coarse_sr_hz
    coarse_df_hz = coarse_sr_hz

    freq_hz_full = fine_frequency_axis_hz(
        args.lo_hz, args.fs_hz, args.fft_point, args.start_ch, args.channels, args.nfft
    )

    display_f0_hz = args.display_f0_hz
    display_span_hz = args.display_span_hz
    if args.inject and display_f0_hz is None:
        display_f0_hz = args.inject_start_freq_hz
        display_span_hz = 20_000.0
    display_mask = choose_display_region(freq_hz_full, display_f0_hz, display_span_hz)

    raw_u16 = open_raw_u16_memmap(dat_path, rows, args.channels)

    injection_info = None
    if args.inject:
        injection_info = prepare_injection_plan(
            raw_u16=raw_u16,
            rows=rows,
            lo_hz=args.lo_hz,
            fs_hz=args.fs_hz,
            fft_point=args.fft_point,
            start_ch=args.start_ch,
            channels=args.channels,
            start_frequency_hz=args.inject_start_freq_hz,
            drift_rate_hz_per_s=args.inject_drift_hz_per_s,
            insertion_start_time_s=args.inject_start_time_s,
            bandwidth_hz=args.inject_bandwidth_hz,
            signal_duration_s=args.inject_duration_s,
            snr_db=args.inject_snr_db,
            rms_chunk_rows=args.rms_chunk_rows,
            edge_taper_s=args.inject_edge_taper_s,
            reference_nfft=args.nfft,
        )

    products = build_waterfall_streaming(
        raw_u16=raw_u16,
        rows=rows,
        lo_hz=args.lo_hz,
        fs_hz=args.fs_hz,
        fft_point=args.fft_point,
        start_ch=args.start_ch,
        channels=args.channels,
        nfft=args.nfft,
        hop=args.hop,
        window_name=args.window,
        baseline_filter_bins=args.baseline_filter_bins,
        baseline_sample_frames=args.baseline_sample_frames,
        display_mask=display_mask,
        display_decimate=args.display_decimate,
        injection_plan=injection_info,
    )

    waterfall_db_display = products["waterfall_db_display"]
    mean_excess_db = products["mean_excess_db"]
    mean_excess_db_display = products["mean_excess_db_display"]
    freq_hz = products["freq_hz"]
    freq_hz_display = products["freq_hz_display"]
    time_s = products["time_s"]
    effective_display_decimate = products["effective_display_decimate"]
    display_df_hz = float(np.median(np.diff(freq_hz_display))) if len(freq_hz_display) > 1 else float(coarse_df_hz / args.nfft * effective_display_decimate)

    if args.title:
        title = args.title
    else:
        title = (
            f"Direct-to-.dat waterfall QC | NFFT={args.nfft}, hop={args.hop}, window={args.window} | "
            f"native Δt={dt_native_s * 1e6:.3f} µs, coarse Δf={coarse_df_hz:.6f} Hz, "
            f"fine Δf={coarse_df_hz / args.nfft:.6f} Hz, display Δf≈{display_df_hz:.6f} Hz"
        )
        if args.inject:
            title += (
                f"\nSynthetic CW injected at {args.inject_start_freq_hz / 1e6:.6f} MHz "
                f"with drift {args.inject_drift_hz_per_s:+.3f} Hz/s"
            )

    out_path = make_output_path(dat_path, args.out)
    fig = plot_dual_panel(
        waterfall_db=waterfall_db_display,
        mean_excess_db=mean_excess_db_display,
        freq_hz=freq_hz_display,
        time_s=time_s,
        vmin_percentile=args.display_vmin_percentile,
        vmax_percentile=args.display_vmax_percentile,
        display_decimate=1,
        colormap=args.colormap,
        title=title,
        f0_ref_hz=(display_f0_hz if display_f0_hz is not None else HI_LINE_HZ),
        display_mask=None,
        out_path=out_path,
        show=args.show or (out_path is None),
    )

    if args.save_npy_prefix:
        prefix = Path(args.save_npy_prefix)
        np.save(str(prefix) + "_waterfall_display_db.npy", waterfall_db_display)
        np.save(str(prefix) + "_mean_excess_db.npy", mean_excess_db)
        np.save(str(prefix) + "_mean_excess_display_db.npy", mean_excess_db_display)
        np.save(str(prefix) + "_baseline_db.npy", products["baseline_db"])
        np.save(str(prefix) + "_freq_hz.npy", freq_hz)
        np.save(str(prefix) + "_freq_display_hz.npy", freq_hz_display)
        np.save(str(prefix) + "_time_s.npy", time_s)

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"DAT file: {dat_path}")
    if json_path is not None:
        print(f"JSON sidecar: {json_path}")
        print(f"JSON loaded: {json_meta.get('json_loaded')}")
        if json_meta.get("json_error"):
            print(f"JSON error: {json_meta.get('json_error')}")
        if json_meta.get("coord_like_records") is not None:
            print(f"coord_like_records: {json_meta.get('coord_like_records')}")

    print(f"rows = {rows}")
    print(f"channels = {args.channels}")
    print(f"total_bytes = {total_bytes}")
    print(f"native_dt_s = {dt_native_s}")
    print(f"native_coarse_df_hz = {coarse_df_hz}")
    print(f"stft_nfft = {args.nfft}")
    print(f"stft_hop = {args.hop}")
    print(f"frame_length_s = {args.nfft / coarse_sr_hz}")
    print(f"hop_s = {args.hop / coarse_sr_hz}")
    print(f"fine_df_hz = {coarse_df_hz / args.nfft}")
    print(f"display_df_hz ~= {display_df_hz}")
    print(f"frequency_start_mhz = {freq_hz[0] / 1e6}")
    print(f"frequency_stop_mhz = {freq_hz[-1] / 1e6}")
    print(f"display_frequency_start_khz = {freq_hz_display[0] / 1e3}")
    print(f"display_frequency_stop_khz = {freq_hz_display[-1] / 1e3}")
    print(f"effective_display_decimate = {effective_display_decimate}")
    print(f"baseline_sample_frames_used = {products['baseline_sample_frames_used']}")
    print(f"filename_metadata = {fn_meta}")

    mem = products["memory_report"]
    print("-" * 72)
    print("MEMORY REPORT")
    print(f"legacy_decoded_complex_gib ~= {mem['legacy_decoded_complex_gib']:.3f}")
    print(f"legacy_single_2d_waterfall_gib ~= {mem['legacy_single_2d_waterfall_gib']:.3f}")
    print(f"legacy_total_rough_gib ~= {mem['legacy_total_rough_gib']:.3f}")
    print(f"streaming_baseline_bank_gib ~= {mem['streaming_baseline_bank_gib']:.3f}")
    print(f"streaming_display_waterfall_gib ~= {mem['streaming_display_waterfall_gib']:.3f}")
    print(f"streaming_peak_rough_gib ~= {mem['streaming_peak_rough_gib']:.3f}")

    injection_qc = products.get("injection_qc")
    if injection_info is not None:
        print("-" * 72)
        print("SYNTHETIC INJECTION")
        for k, v in injection_info.items():
            if k in {"offsets_hz", "weights", "phases0"}:
                continue
            print(f"{k} = {v}")
        if injection_qc is not None and injection_qc["track_count"] > 0:
            on_track = injection_qc["track_sum"] / injection_qc["track_count"]
            off_track = (
                injection_qc["off_sum"] / injection_qc["off_count"]
                if injection_qc["off_count"] > 0
                else float("nan")
            )
            peak_idx = int(np.argmax(mean_excess_db))
            print(f"track_contrast_db = {on_track - off_track}")
            print(f"peak_mean_excess_db = {float(mean_excess_db[peak_idx])}")
            print(f"peak_mean_excess_freq_mhz = {float(freq_hz[peak_idx] / 1e6)}")

    return {
        "args": args,
        "figure": fig,
        "products": products,
        "filename_metadata": fn_meta,
        "json_metadata": json_meta,
        "injection_info": injection_info,
    }


if __name__ == "__main__":
    main()
