# Observation Manifest and Multi-File Streaming

## Why a manifest exists

Real observations can span many `.dat` files. A single file atom is only one physical chunk.
`ObservationManifest` defines one logical observation as an ordered list of those chunks.

This preserves the raw `.dat` contract:

- headerless binary payload
- row-major shape `(rows, 256)`
- `2` bytes per cell (`<u2`, little-endian)
- packed signed int8 I/Q in low/high bytes
- metadata must be external (never inferred from payload)

## Manifest JSON format

```json
{
  "observation_id": "optional-observation-id",
  "files": [
    {
      "dat_path": "capture_000.dat",
      "metadata_path": "capture_000.meta.json"
    },
    {
      "dat_path": "capture_001.dat"
    }
  ]
}
```

Notes:

- `files` is required and ordered.
- `dat_path` is required for each file atom.
- `metadata_path` is optional for each file atom.
- Relative paths are resolved against the manifest file directory.

## Models

- `DatFileEntry`: one physical `.dat` file plus optional external metadata reference.
- `ObservationManifest`: ordered tuple of `DatFileEntry` records for one logical observation.

Keeping these separate prevents accidental treatment of each file as an independent science observation.

## Streaming model

`iter_observation_blocks(manifest, block_rows, overlap_rows=0)` streams row blocks from the
logical observation in global row order.

Behavior:

- reads file atoms in manifest order
- transparently crosses file boundaries
- emits `ObservationBlock(start_row, stop_row, words)`
- supports overlapping windows via `overlap_rows`
- uses stride `block_rows - overlap_rows`
- allows one terminal partial block that reaches observation end
- stops after the first block with `stop_row == total_rows` (no fully contained tail block)
- never materializes the full observation in memory

Memory is bounded by the configured block size (plus a small working buffer) rather than total
observation size.
