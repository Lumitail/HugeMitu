# Raw `.dat` data contract

This project treats raw `.dat` files as **headerless binary payloads** with no embedded metadata.

## Binary layout

- Geometry is row-major `(rows, 256)`.
- Each cell is 2 bytes.
- Therefore each row is `256 * 2 = 512` bytes.
- A valid file size must be divisible by 512 bytes.
- `rows = total_bytes / 512`.

## Cell encoding

Each cell is one little-endian 16-bit word (`<u2`) that packs two signed int8 lanes:

- Low byte: `I` lane (`int8`)
- High byte: `Q` lane (`int8`)

Decoded complex sample is `complex(I, Q)` represented as `complex64` for downstream processing.

## Metadata policy

- Do not assume any internal file header.
- Do not invent metadata from payload contents.
- Observation metadata (sample rate, start time, tuning, etc.) must come from external manifests.

## Processing directionality

For spectral processing, FFTs must run along time inside each coarse channel, not across the 256 coarse-channel columns.
