"""Raw .dat reader for packed complex I/Q words.

The on-disk contract is:
- headerless binary payload
- row-major shape of (rows, 256)
- each cell is a little-endian uint16
- low/high bytes are signed int8 I/Q lanes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROW_WIDTH = 256
BYTES_PER_CELL = 2
BYTES_PER_ROW = ROW_WIDTH * BYTES_PER_CELL


@dataclass(frozen=True)
class DatWordGrid:
    """Raw-word view of a `.dat` payload.

    This keeps the memmapped uint16 words separate from any decoded complex view.
    """

    path: Path
    rows: int
    words: np.memmap


def read_dat_words(path: str | Path) -> DatWordGrid:
    """Open a raw `.dat` file as a memmapped uint16 grid.

    Raises:
        ValueError: if byte length is not divisible by one full row (512 bytes).
    """

    path = Path(path)
    total_bytes = path.stat().st_size

    if total_bytes % BYTES_PER_ROW != 0:
        raise ValueError(
            f"Invalid .dat size {total_bytes} bytes; must be divisible by {BYTES_PER_ROW}."
        )

    rows = total_bytes // BYTES_PER_ROW
    words = np.memmap(path, mode="r", dtype="<u2", shape=(rows, ROW_WIDTH))
    return DatWordGrid(path=path, rows=rows, words=words)


def decode_packed_iq(words: np.ndarray) -> np.ndarray:
    """Decode packed uint16 words into complex64 I/Q samples.

    Low byte is signed int8 I, high byte is signed int8 Q.
    """

    u16 = np.asarray(words, dtype=np.uint16)

    i_lane = (u16 & 0x00FF).astype(np.uint8).view(np.int8)
    q_lane = ((u16 >> 8) & 0x00FF).astype(np.uint8).view(np.int8)

    decoded = i_lane.astype(np.float32) + 1j * q_lane.astype(np.float32)
    return decoded.astype(np.complex64, copy=False)
