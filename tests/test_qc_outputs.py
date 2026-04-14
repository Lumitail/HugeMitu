from __future__ import annotations

import json

import numpy as np

from acs.io.manifest import DatFileEntry, ObservationManifest
from acs.preproc.waterfall import build_streaming_waterfall
from acs.review.qc import save_qc_review_bundle


def _write_dat(path, words: np.ndarray) -> None:
    path.write_bytes(words.astype("<u2").tobytes())


def test_save_qc_review_bundle_writes_expected_artifacts(tmp_path):
    rows = 20
    i_lane = np.tile(np.arange(256, dtype=np.uint16), (rows, 1)) & 0x00FF
    q_lane = (255 - i_lane) & 0x00FF
    words = (q_lane << 8) | i_lane

    dat = tmp_path / "obs.dat"
    _write_dat(dat, words)

    manifest = ObservationManifest(files=(DatFileEntry(dat_path=dat),), observation_id="obs-qc")
    result = build_streaming_waterfall(
        manifest,
        nfft=8,
        hop=4,
        block_rows=9,
        baseline_sample_frames=2,
        time_decimation=2,
        sample_rate_hz=100.0,
        start_freq_hz=1.42e9,
        coarse_channel_spacing_hz=10_000.0,
    )

    outputs = save_qc_review_bundle(result, tmp_path / "qc_out", stem="review")

    assert outputs["waterfall_image"].exists()
    assert outputs["spectrum_image"].exists()
    assert outputs["metadata_json"].exists()

    metadata = json.loads(outputs["metadata_json"].read_text())
    assert metadata["nfft"] == 8
    assert metadata["hop"] == 4
    assert metadata["display_frames"] == result.waterfall_db.shape[0]
    assert len(metadata["display_frame_start_rows"]) == result.waterfall_db.shape[0]
    assert metadata["waterfall_shape"][1] == 256 * 8
    assert metadata["mean_excess_shape"][0] == 256 * 8
    assert len(metadata["freq_hz_display"]) == 256 * 8
    assert len(metadata["time_s_display"]) == result.waterfall_db.shape[0]
