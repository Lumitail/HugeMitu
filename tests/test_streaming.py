from __future__ import annotations

import numpy as np
import pytest

from acs.io.dat_reader import ROW_WIDTH
from acs.io.manifest import DatFileEntry, ObservationManifest
from acs.io.streaming import iter_observation_blocks


def _write_dat_rows(path, row_values: list[int]) -> None:
    rows = np.zeros((len(row_values), ROW_WIDTH), dtype=np.uint16)
    for i, value in enumerate(row_values):
        rows[i, :] = value
    path.write_bytes(rows.astype("<u2").tobytes())


def test_iter_observation_blocks_crosses_file_boundaries(tmp_path):
    dat_0 = tmp_path / "part0.dat"
    dat_1 = tmp_path / "part1.dat"

    _write_dat_rows(dat_0, [0, 1, 2, 3, 4])
    _write_dat_rows(dat_1, [5, 6, 7])

    manifest = ObservationManifest(
        files=(DatFileEntry(dat_path=dat_0), DatFileEntry(dat_path=dat_1))
    )

    blocks = list(iter_observation_blocks(manifest, block_rows=3))

    assert [(b.start_row, b.stop_row) for b in blocks] == [(0, 3), (3, 6), (6, 8)]
    assert [int(b.words[0, 0]) for b in blocks] == [0, 3, 6]
    assert [int(b.words[-1, 0]) for b in blocks] == [2, 5, 7]


def test_iter_observation_blocks_supports_overlap(tmp_path):
    dat_0 = tmp_path / "part0.dat"
    dat_1 = tmp_path / "part1.dat"

    _write_dat_rows(dat_0, [0, 1, 2])
    _write_dat_rows(dat_1, [3, 4, 5, 6])

    manifest = ObservationManifest(
        files=(DatFileEntry(dat_path=dat_0), DatFileEntry(dat_path=dat_1))
    )

    blocks = list(iter_observation_blocks(manifest, block_rows=4, overlap_rows=2))

    assert [(b.start_row, b.stop_row) for b in blocks] == [(0, 4), (2, 6), (4, 7), (6, 7)]
    assert [int(b.words[0, 0]) for b in blocks] == [0, 2, 4, 6]


def test_iter_observation_blocks_validates_overlap(tmp_path):
    dat_0 = tmp_path / "part0.dat"
    _write_dat_rows(dat_0, [0, 1])

    manifest = ObservationManifest(files=(DatFileEntry(dat_path=dat_0),))

    with pytest.raises(ValueError, match="overlap_rows must be < block_rows"):
        list(iter_observation_blocks(manifest, block_rows=2, overlap_rows=2))
