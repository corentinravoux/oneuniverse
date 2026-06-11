"""OUF 2.6 — partition stats live in the `_index.parquet` sidecar.

The manifest JSON is identity-only (stable, O(1) in partition count); the
sidecar carries per-partition stats; `read_manifest` reconstructs
`Manifest.partitions` transparently, and pre-2.6 manifests with embedded
partition arrays still read (back-compat).
"""
import json
import sys
from pathlib import Path

import pytest

from oneuniverse.data.format_spec import FORMAT_VERSION
from oneuniverse.data.manifest import (PARTITION_INDEX_FILENAME,
                                       ManifestValidationError,
                                       read_manifest, write_manifest)

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402


def test_v26_manifest_is_identity_only_with_sidecar(tmp_path):
    view = synthetic_point_view(tmp_path, n=3000, seed=1)
    raw = json.loads((view.ou_dir / "manifest.json").read_text())
    assert raw["oneuniverse_format_version"] == FORMAT_VERSION == "2.6.0"
    assert "partitions" not in raw                       # moved out
    assert raw["partition_index"] == PARTITION_INDEX_FILENAME
    assert raw["n_partitions"] >= 1
    assert raw["n_rows_total"] == 3000
    assert (view.ou_dir / PARTITION_INDEX_FILENAME).is_file()


def test_v26_round_trip_reconstructs_partitions(tmp_path):
    view = synthetic_point_view(tmp_path, n=3000, seed=2)
    m = view.manifest                                    # read via sidecar
    assert m.n_rows == 3000
    assert m.n_partitions == len(m.partitions) >= 1
    p = m.partitions[0]
    assert p.healpix_cell is not None and p.n_rows > 0
    assert p.stats.z_min is not None and p.stats.z_max >= p.stats.z_min
    # write -> read -> identical specs (the full loop through the sidecar)
    out = tmp_path / "copy" / "manifest.json"
    out.parent.mkdir()
    write_manifest(out, m)
    m2 = read_manifest(out)
    assert m2.partitions == m.partitions


def test_pre26_embedded_partitions_still_read(tmp_path):
    """A 2.5-style manifest (partitions embedded in the JSON) must keep
    loading — readers accept both layouts."""
    view = synthetic_point_view(tmp_path, n=500, seed=3)
    raw = json.loads((view.ou_dir / "manifest.json").read_text())
    # forge the old layout: embed the partitions, drop the sidecar pointer
    old = dict(raw)
    old.pop("partition_index"); old.pop("n_partitions"); old.pop("n_rows_total")
    old["oneuniverse_format_version"] = "2.5.0"
    old["partitions"] = [
        {"name": p.name, "n_rows": p.n_rows, "sha256": p.sha256,
         "size_bytes": p.size_bytes, "healpix_cell": p.healpix_cell,
         "stats": {"z_min": p.stats.z_min, "z_max": p.stats.z_max}}
        for p in view.manifest.partitions]
    legacy = tmp_path / "legacy" / "manifest.json"
    legacy.parent.mkdir()
    legacy.write_text(json.dumps(old))
    m = read_manifest(legacy)                            # no sidecar needed
    assert m.oneuniverse_format_version == "2.5.0"
    assert m.n_rows == 500


def test_missing_sidecar_raises_clearly(tmp_path):
    view = synthetic_point_view(tmp_path, n=200, seed=4)
    (view.ou_dir / PARTITION_INDEX_FILENAME).unlink()
    with pytest.raises(ManifestValidationError, match="sidecar file is missing"):
        read_manifest(view.ou_dir / "manifest.json")
