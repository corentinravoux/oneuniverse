import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.database import OneuniverseDatabase
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec
from oneuniverse.data.validity import DatasetValidity


def _write_syn(root: Path, sub: str, name: str, validity: DatasetValidity):
    df = pd.DataFrame({
        "ra": [0.0], "dec": [0.0], "z": [0.1],
        "z_type": ["spec"], "z_err": [0.01],
        "galaxy_id": np.array([0], dtype=np.int64),
        "survey_id": ["syn0"],
        "_original_row_index": np.array([0], dtype=np.int64),
        "_healpix32": np.array([0], dtype=np.int64),
    })
    survey_dir = root / sub
    ou_dir = survey_dir / "oneuniverse"
    ou_dir.mkdir(parents=True, exist_ok=True)
    write_ouf_dataset(
        df, ou_dir,
        survey_name=name, survey_type=sub.split("/")[0],
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="syn", version="0"),
        validity=validity,
    )


def test_as_of_selects_version_valid_at_timestamp(tmp_path):
    v1 = DatasetValidity(
        valid_from_utc="2020-01-01T00:00:00+00:00",
        valid_to_utc="2021-01-01T00:00:00+00:00",
        version="dr16",
    )
    v2 = DatasetValidity(
        valid_from_utc="2021-01-01T00:00:00+00:00",
        version="dr17",
        supersedes=("spectroscopic_eboss_qso_v_dr16",),
    )
    _write_syn(tmp_path, "spectroscopic/eboss_qso/v_dr16",
               "spectroscopic_eboss_qso_v_dr16", v1)
    _write_syn(tmp_path, "spectroscopic/eboss_qso/v_dr17",
               "spectroscopic_eboss_qso_v_dr17", v2)

    db = OneuniverseDatabase.from_root(tmp_path)
    names = set(db.list())
    assert "spectroscopic_eboss_qso_v_dr17" in names
    assert "spectroscopic_eboss_qso_v_dr16" not in names

    t = dt.datetime(2020, 6, 1, tzinfo=dt.timezone.utc)
    snap = db.as_of(t)
    snap_names = set(snap.list())
    assert "spectroscopic_eboss_qso_v_dr16" in snap_names
    assert "spectroscopic_eboss_qso_v_dr17" not in snap_names


def test_versions_of_lists_all_versions(tmp_path):
    v1 = DatasetValidity(
        valid_from_utc="2020-01-01T00:00:00+00:00",
        valid_to_utc="2021-01-01T00:00:00+00:00",
        version="dr16",
    )
    v2 = DatasetValidity(
        valid_from_utc="2021-01-01T00:00:00+00:00",
        version="dr17",
        supersedes=("spectroscopic_eboss_qso_v_dr16",),
    )
    _write_syn(tmp_path, "spectroscopic/eboss_qso/v_dr16",
               "spectroscopic_eboss_qso_v_dr16", v1)
    _write_syn(tmp_path, "spectroscopic/eboss_qso/v_dr17",
               "spectroscopic_eboss_qso_v_dr17", v2)

    db = OneuniverseDatabase.from_root(tmp_path)
    versions = db.versions_of_root("spectroscopic/eboss_qso")
    labels = [e.manifest.validity.version for e in versions]
    assert labels == ["dr16", "dr17"]


def test_as_of_rejects_naive_timestamp(tmp_path):
    db = OneuniverseDatabase.from_root(tmp_path)
    with pytest.raises(ValueError, match="timezone"):
        db.as_of(dt.datetime(2026, 3, 1))


def test_default_walker_warns_on_mixed_validity(tmp_path, caplog):
    v1 = DatasetValidity(valid_from_utc="2020-01-01T00:00:00+00:00")
    v2 = DatasetValidity(valid_from_utc="2020-02-01T00:00:00+00:00")
    _write_syn(tmp_path, "spectroscopic/a", "spectroscopic_a", v1)
    _write_syn(tmp_path, "spectroscopic/a_overlap",
               "spectroscopic_a_overlap", v2)
    with caplog.at_level("WARNING"):
        OneuniverseDatabase.from_root(tmp_path)
