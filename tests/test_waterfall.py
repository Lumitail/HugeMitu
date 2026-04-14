from __future__ import annotations

import numpy as np
import pytest

from acs.io.manifest import DatFileEntry, ObservationManifest
from acs.preproc.waterfall import build_streaming_waterfall, estimate_baseline


ROW_WIDTH = 256


def _pack_iq(i_vals: np.ndarray, q_vals: np.ndarray) -> np.ndarray:
    i_u8 = i_vals.astype(np.int8).view(np.uint8).astype(np.uint16)
    q_u8 = q_vals.astype(np.int8).view(np.uint8).astype(np.uint16)
    return (q_u8 << 8) | i_u8


def _write_dat(path, words: np.ndarray) -> None:
    path.write_bytes(words.astype("<u2").tobytes())


def _reference_coarse_power(decoded: np.ndarray, *, nfft: int, hop: int) -> np.ndarray:
    frames = []
    for start in range(0, decoded.shape[0] - nfft + 1, hop):
        frame = decoded[start : start + nfft, :]
        power = np.abs(np.fft.fft(frame, axis=0)) ** 2
        frames.append(power.mean(axis=0))
    return np.stack(frames, axis=0)


def test_build_streaming_waterfall_matches_reference_across_file_boundary(tmp_path):
    rng = np.random.default_rng(12)

    rows_total = 28
    i_vals = rng.integers(-20, 20, size=(rows_total, ROW_WIDTH), dtype=np.int16)
    q_vals = rng.integers(-20, 20, size=(rows_total, ROW_WIDTH), dtype=np.int16)
    words = _pack_iq(i_vals, q_vals)

    dat_a = tmp_path / "a.dat"
    dat_b = tmp_path / "b.dat"
    _write_dat(dat_a, words[:13])
    _write_dat(dat_b, words[13:])

    manifest = ObservationManifest(
        files=(DatFileEntry(dat_path=dat_a), DatFileEntry(dat_path=dat_b)),
        observation_id="obs-test",
    )

    result = build_streaming_waterfall(
        manifest,
        nfft=8,
        hop=4,
        block_rows=7,
        baseline_sample_frames=3,
        baseline_smooth_width=5,
        time_decimation=2,
    )

    decoded = i_vals.astype(np.float32) + 1j * q_vals.astype(np.float32)
    coarse = _reference_coarse_power(decoded, nfft=8, hop=4)
    baseline = estimate_baseline(coarse[:3], smooth_width=5)
    mean_excess_ref = (coarse / baseline).mean(axis=0)
    mean_excess_ref_db = 10.0 * np.log10(np.maximum(mean_excess_ref, 1e-12))

    assert result.metadata["total_stft_frames"] == coarse.shape[0]
    assert result.metadata["first_frame_start_row"] == 0
    assert result.metadata["last_frame_start_row"] == 20
    # Two-pass bounded-memory float32 streaming can differ from full-array reference by
    # sub-micro-dB roundoff, especially near zero-dB bins.
    np.testing.assert_allclose(
        result.mean_excess_db,
        mean_excess_ref_db.astype(np.float32),
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.waterfall_db.shape[1] == ROW_WIDTH
    np.testing.assert_array_equal(result.display_frame_start_rows, np.array([0, 8, 16], dtype=np.int64))


def test_build_streaming_waterfall_rejects_observation_shorter_than_frame(tmp_path):
    rows_total = 6
    words = np.zeros((rows_total, ROW_WIDTH), dtype=np.uint16)

    dat_a = tmp_path / "short.dat"
    _write_dat(dat_a, words)

    manifest = ObservationManifest(files=(DatFileEntry(dat_path=dat_a),))

    with pytest.raises(ValueError, match="shorter than one STFT frame"):
        build_streaming_waterfall(
            manifest,
            nfft=8,
            hop=4,
            block_rows=4,
        )
