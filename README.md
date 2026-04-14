# ACS (Initial Skeleton)

ACS is a new Python project for a SETI narrowband search workflow.

## Current scope

This repository currently provides only the initial project layout:
- package/module skeleton
- minimal CLI placeholder
- basic test + CI wiring

No full search pipeline is implemented yet.

## Data note

Raw `.dat` files are treated as **headerless payloads**. Required observation metadata is expected to come from external sources and is not embedded in an internal file header format.

## Quickstart (placeholder)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
acs
pytest
```

Further usage instructions will be added as pipeline components land.
