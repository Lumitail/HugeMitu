"""Observation manifest models for multi-file `.dat` observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class DatFileEntry:
    """One physical `.dat` file and optional external metadata reference."""

    dat_path: Path
    metadata_path: Path | None = None


@dataclass(frozen=True)
class ObservationManifest:
    """Ordered set of `.dat` files that forms one logical observation."""

    files: tuple[DatFileEntry, ...]
    observation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("ObservationManifest.files must contain at least one DatFileEntry.")


def load_manifest(path: str | Path) -> ObservationManifest:
    """Load an observation manifest from JSON.

    Expected format:
    {
      "observation_id": "optional-id",
      "files": [
        {"dat_path": "chunk_000.dat", "metadata_path": "chunk_000.json"},
        {"dat_path": "chunk_001.dat"}
      ]
    }

    Relative file paths are resolved relative to the manifest file parent directory.
    """

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())

    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("Manifest must contain a non-empty 'files' list.")

    base_dir = manifest_path.parent
    entries: list[DatFileEntry] = []

    for item in raw_files:
        if not isinstance(item, dict):
            raise ValueError("Each manifest file entry must be an object.")

        dat_path_raw = item.get("dat_path")
        if not isinstance(dat_path_raw, str) or not dat_path_raw:
            raise ValueError("Each manifest file entry requires non-empty string 'dat_path'.")

        dat_path = Path(dat_path_raw)
        if not dat_path.is_absolute():
            dat_path = base_dir / dat_path

        metadata_raw = item.get("metadata_path")
        metadata_path: Path | None = None
        if metadata_raw is not None:
            if not isinstance(metadata_raw, str) or not metadata_raw:
                raise ValueError("'metadata_path' must be a non-empty string when provided.")
            metadata_path = Path(metadata_raw)
            if not metadata_path.is_absolute():
                metadata_path = base_dir / metadata_path

        entries.append(DatFileEntry(dat_path=dat_path, metadata_path=metadata_path))

    observation_id = payload.get("observation_id")
    if observation_id is not None and not isinstance(observation_id, str):
        raise ValueError("'observation_id' must be a string when provided.")

    return ObservationManifest(files=tuple(entries), observation_id=observation_id)
