from __future__ import annotations

import numpy as np
import pytest

from acs.io.dat_reader import BYTES_PER_ROW, decode_packed_iq, read_dat_words


def _write_bytes(path, payload: bytes) -> None:
    path.write_bytes(payload)


def test_read_dat_words_rejects_non_row_aligned_file(tmp_path):
    dat_path = tmp_path / "bad.dat"
    _write_bytes(dat_path, b"\x00" * (BYTES_PER_ROW + 1))

    with pytest.raises(ValueError, match="must be divisible"):
        read_dat_words(dat_path)


def test_read_dat_words_memmaps_expected_shape_and_dtype(tmp_path):
    dat_path = tmp_path / "tiny.dat"

    words = np.arange(2 * 256, dtype=np.uint16).reshape(2, 256)
    _write_bytes(dat_path, words.astype("<u2").tobytes())

    grid = read_dat_words(dat_path)

    assert grid.rows == 2
    assert grid.words.shape == (2, 256)
    assert grid.words.dtype == np.dtype("<u2")
    np.testing.assert_array_equal(np.asarray(grid.words), words)


def test_decode_packed_iq_interprets_low_high_bytes_as_signed_int8():
    packed = np.array(
        [
            [0x0000, 0xFF01, 0x0180, 0x807F],
            [0x7F80, 0x8080, 0x7FFF, 0x8001],
        ],
        dtype=np.uint16,
    )

    decoded = decode_packed_iq(packed)

    expected = np.array(
        [
            [0 + 0j, 1 - 1j, -128 + 1j, 127 - 128j],
            [-128 + 127j, -128 - 128j, -1 + 127j, 1 - 128j],
        ],
        dtype=np.complex64,
    )

    assert decoded.dtype == np.complex64
    np.testing.assert_array_equal(decoded, expected)
