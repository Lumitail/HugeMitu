from __future__ import annotations

import json

import pytest

from acs.io.manifest import ObservationManifest, load_manifest


def test_load_manifest_resolves_relative_paths(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    dat_a = tmp_path / "a.dat"
    dat_b = tmp_path / "b.dat"
    meta_b = tmp_path / "b_meta.json"

    manifest_path.write_text(
        json.dumps(
            {
                "observation_id": "obs-001",
                "files": [
                    {"dat_path": dat_a.name},
                    {"dat_path": dat_b.name, "metadata_path": meta_b.name},
                ],
            }
        )
    )

    manifest = load_manifest(manifest_path)

    assert isinstance(manifest, ObservationManifest)
    assert manifest.observation_id == "obs-001"
    assert manifest.files[0].dat_path == dat_a
    assert manifest.files[0].metadata_path is None
    assert manifest.files[1].dat_path == dat_b
    assert manifest.files[1].metadata_path == meta_b


def test_load_manifest_requires_non_empty_files_list(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"files": []}))

    with pytest.raises(ValueError, match="non-empty 'files' list"):
        load_manifest(manifest_path)


def test_load_manifest_requires_dat_path_for_each_entry(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"files": [{"metadata_path": "x.json"}]}))

    with pytest.raises(ValueError, match="requires non-empty string 'dat_path'"):
        load_manifest(manifest_path)
