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


def _reference_fine_power(decoded: np.ndarray, *, nfft: int, hop: int) -> np.ndarray:
    frames = []
    for start in range(0, decoded.shape[0] - nfft + 1, hop):
        frame = decoded[start : start + nfft, :]
        power = np.abs(np.fft.fftshift(np.fft.fft(frame, axis=0), axes=0)) ** 2
        frames.append(power.T.reshape(-1))
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
        sample_rate_hz=8.0,
        start_freq_hz=1000.0,
        coarse_channel_spacing_hz=2.0,
    )

    decoded = i_vals.astype(np.float32) + 1j * q_vals.astype(np.float32)
    fine = _reference_fine_power(decoded, nfft=8, hop=4)
    baseline = estimate_baseline(fine[:3], smooth_width=5)
    mean_excess_ref = (fine / baseline).mean(axis=0)
    mean_excess_ref_db = 10.0 * np.log10(np.maximum(mean_excess_ref, 1e-12))

    assert result.metadata["total_stft_frames"] == fine.shape[0]
    assert result.metadata["first_frame_start_row"] == 0
    assert result.metadata["last_frame_start_row"] == 20
    np.testing.assert_allclose(
        result.mean_excess_db,
        mean_excess_ref_db.astype(np.float32),
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.waterfall_db.shape[1] == ROW_WIDTH * 8
    assert result.metadata["fine_bins_total"] == ROW_WIDTH * 8
    np.testing.assert_array_equal(result.display_frame_start_rows, np.array([0, 8, 16], dtype=np.int64))
    np.testing.assert_allclose(result.time_s_display, np.array([0.0, 1.0, 2.0]))


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


def test_build_streaming_waterfall_retains_fine_bins_without_freq_decimation(tmp_path):
    rows_total = 32
    i_vals = np.zeros((rows_total, ROW_WIDTH), dtype=np.int16)
    q_vals = np.zeros((rows_total, ROW_WIDTH), dtype=np.int16)
    i_vals[:, 7] = 12
    words = _pack_iq(i_vals, q_vals)

    dat = tmp_path / "obs.dat"
    _write_dat(dat, words)

    manifest = ObservationManifest(files=(DatFileEntry(dat_path=dat),))
    result = build_streaming_waterfall(
        manifest,
        nfft=8,
        hop=4,
        block_rows=10,
        baseline_sample_frames=2,
        baseline_smooth_width=1,
        time_decimation=2,
        freq_decimation=1,
    )

    assert result.mean_excess_db.shape == (ROW_WIDTH * 8,)
    assert result.waterfall_db.shape[1] == ROW_WIDTH * 8
