"""Streaming helpers for logical observations spanning multiple `.dat` files."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acs.io.dat_reader import ROW_WIDTH, read_dat_words
from acs.io.manifest import ObservationManifest


@dataclass(frozen=True)
class ObservationBlock:
    """One streamed block from a logical observation."""

    start_row: int
    stop_row: int
    words: np.ndarray


def _iter_dat_row_chunks(
    manifest: ObservationManifest,
    chunk_rows: int,
):
    """Yield consecutive row chunks across all dat files in manifest order."""

    for entry in manifest.files:
        grid = read_dat_words(entry.dat_path)
        row_start = 0
        while row_start < grid.rows:
            row_stop = min(row_start + chunk_rows, grid.rows)
            yield np.asarray(grid.words[row_start:row_stop])
            row_start = row_stop


def iter_observation_blocks(
    manifest: ObservationManifest,
    block_rows: int,
    overlap_rows: int = 0,
):
    """Iterate row blocks across `.dat` file boundaries.

    Blocks are emitted in global observation row coordinates. Data are streamed from
    file-atoms in manifest order and are never materialized as a full observation.
    """

    if block_rows <= 0:
        raise ValueError("block_rows must be > 0")
    if overlap_rows < 0:
        raise ValueError("overlap_rows must be >= 0")
    if overlap_rows >= block_rows:
        raise ValueError("overlap_rows must be < block_rows")

    step_rows = block_rows - overlap_rows

    buffer = np.empty((0, ROW_WIDTH), dtype=np.dtype("<u2"))
    buffer_start_row = 0
    next_global_row = 0

    for chunk in _iter_dat_row_chunks(manifest=manifest, chunk_rows=block_rows):
        if chunk.size == 0:
            continue

        if chunk.shape[1] != ROW_WIDTH:
            raise ValueError(f"Expected row width {ROW_WIDTH}, got {chunk.shape[1]}.")

        chunk = np.asarray(chunk, dtype=np.dtype("<u2"))

        if buffer.size == 0:
            buffer = np.array(chunk, copy=True)
            buffer_start_row = next_global_row
        else:
            buffer = np.concatenate((buffer, chunk), axis=0)

        next_global_row += chunk.shape[0]

        while buffer.shape[0] >= block_rows:
            block_words = np.array(buffer[:block_rows], copy=True)
            start = buffer_start_row
            stop = start + block_words.shape[0]
            yield ObservationBlock(start_row=start, stop_row=stop, words=block_words)

            buffer = np.array(buffer[step_rows:], copy=True)
            buffer_start_row += step_rows

    while buffer.shape[0] > 0:
        block_words = np.array(buffer[:block_rows], copy=True)
        start = buffer_start_row
        stop = start + block_words.shape[0]
        yield ObservationBlock(start_row=start, stop_row=stop, words=block_words)

        if block_words.shape[0] < block_rows:
            break

        buffer = np.array(buffer[step_rows:], copy=True)
        buffer_start_row += step_rows
