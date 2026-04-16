"""
Tests for DatasetEntry consolidation (Phase 6 Task 1).

Verify the frozen dataclass behaviour and that the database exposes entries
that agree with the legacy per-field accessors.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data import OneuniverseDatabase, convert_survey
from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._dataset_entry import DatasetEntry
from oneuniverse.data._registry import _REGISTRY


@pytest.fixture
def tmp_path_clean():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _register_synth(name: str):
    class _SynthLoader(BaseSurveyLoader):
        config = SurveyConfig(
            name=name,
            survey_type="test",
            description="synth",
            data_subpath=f"test/{name}",
            data_filename="synth.csv",
            data_format="csv",
        )

        def _load_raw(self, data_path=None, **kwargs):
            return pd.read_csv(Path(data_path) / self.config.data_filename)

    _REGISTRY[name] = _SynthLoader
    return _SynthLoader


def _make_csv(dir_: Path, n: int = 40) -> None:
    import healpy as hp

    dir_.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    df = pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "z": rng.uniform(0.01, 0.5, n),
        "z_type": ["spec"] * n,
        "z_err": np.full(n, 1e-4, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": "synth",
        "_original_row_index": np.arange(n, dtype=np.int64),
        "_healpix32": hp.ang2pix(
            32, np.radians(90.0 - dec), np.radians(ra), nest=True,
        ).astype(np.int32),
    })
    df.to_csv(dir_ / "synth.csv", index=False)


def test_entry_frozen():
    e = DatasetEntry(loader=None, manifest=None, path=Path("/tmp"))  # type: ignore[arg-type]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        e.loader = object()  # type: ignore[misc]


def test_entry_matches_legacy_accessors(tmp_path_clean):
    name = "ds_entry_synth"
    _register_synth(name)
    try:
        raw_dir = tmp_path_clean / "raw"
        _make_csv(raw_dir)
        out = tmp_path_clean / "db" / "test" / "synth"
        out.mkdir(parents=True)
        convert_survey(
            survey_name=name, raw_path=raw_dir,
            output_dir=out, overwrite=True,
        )

        db = OneuniverseDatabase(tmp_path_clean / "db")
        key = next(iter(db))
        e = db.entry(key)

        assert isinstance(e, DatasetEntry)
        assert e.path == db.get_path(key)
        assert e.manifest is db.get_manifest(key)
        assert isinstance(e.loader(), BaseSurveyLoader)
        assert e.loader.config is db.get_config(key)
        # Unknown dataset → KeyError via entry()
        with pytest.raises(KeyError):
            db.entry("__missing__")
    finally:
        _REGISTRY.pop(name, None)
