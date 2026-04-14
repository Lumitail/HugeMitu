# AGENTS.md

## Project scope
This repository is for a new SETI narrowband search tool working directly on raw `.dat` files.

## Non-negotiable data contract
- Raw `.dat` files are headerless binary payloads.
- File geometry is row-major `(rows, 256)`.
- Each cell is 2 bytes.
- Each 16-bit word stores two signed int8 lanes as packed complex I/Q.
- Metadata are external and must not be invented from the payload.
- FFTs must run along time inside each coarse channel, not across the 256 columns.

## Engineering rules
- Prefer small, reviewable commits.
- Do not add heavy dependencies unless explicitly requested.
- Keep implementations simple and exact before optimizing.
- Add or update tests for every non-trivial code change.
- Do not silently change the data contract.
- Preserve bounded-memory streaming behavior for large multi-file observations.

## Current priority
1. repository skeleton
2. exact `.dat` parser
3. observation manifest and multi-file stitching
4. streaming waterfall / QC
5. first-pass narrowband search
6. review plot generation

## Done means
- The requested change is minimal and reviewable.
- Tests are added or updated when applicable.
- No unsupported assumptions are introduced.

Done when:
- AGENTS.md exists at the repo root
- No other files are changed
