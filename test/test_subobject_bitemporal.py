"""Tests: rebuild archives previous, and load_subobject_links(as_of=T)
resolves correct version."""
import datetime as dt
import time

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.database import OneuniverseDatabase
from oneuniverse.data.oneuid_rules import CrossMatchRules
from oneuniverse.data.subobject_rules import SubobjectRules

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from test_subobject_build import (  # noqa: E402
    _synthetic_host_catalog,
    _synthetic_sn_catalog,
)


def _setup(tmp_path):
    root = tmp_path / "db"
    root.mkdir()
    _synthetic_host_catalog(root, "host_galaxies", n_host=5)
    from oneuniverse.data.dataset_view import DatasetView
    host_df = DatasetView.from_path(root / "host_galaxies").read(
        columns=["ra", "dec", "z"]
    )
    _synthetic_sn_catalog(root, "sne", host_df)
    db = OneuniverseDatabase(root)
    db.build_oneuid(
        datasets=["host_galaxies", "sne"],
        rules=CrossMatchRules(sky_tol_arcsec=0.05),
        name="default",
    )
    return db


def test_rebuild_archives_previous(tmp_path):
    db = _setup(tmp_path)
    r1 = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=2.0, dz_tol=1e-2,
    )
    db.build_subobject_links(
        rules=r1, parent_datasets=["host_galaxies"],
        child_datasets=["sne"], name="sne_in_hosts",
    )

    assert db.list_subobject_link_sets() == ["sne_in_hosts"]
    assert db.list_subobject_link_sets(include_archived=True) == ["sne_in_hosts"]

    time.sleep(1.1)  # second-granular archive suffix
    r2 = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=0.3, dz_tol=1e-2,
    )
    db.build_subobject_links(
        rules=r2, parent_datasets=["host_galaxies"],
        child_datasets=["sne"], name="sne_in_hosts",
    )

    live = db.list_subobject_link_sets()
    archived = db.list_subobject_link_sets(include_archived=True)
    assert live == ["sne_in_hosts"]
    archive_only = [a for a in archived if "__" in a]
    assert len(archive_only) == 1


def test_as_of_resolves_correct_version(tmp_path):
    db = _setup(tmp_path)
    r1 = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=2.0, dz_tol=1e-2,
    )
    db.build_subobject_links(
        rules=r1, parent_datasets=["host_galaxies"],
        child_datasets=["sne"], name="sne_in_hosts",
    )
    time.sleep(1.1)
    t_mid = dt.datetime.now(tz=dt.timezone.utc)
    time.sleep(1.1)
    r2 = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=0.3, dz_tol=1e-2,
    )
    db.build_subobject_links(
        rules=r2, parent_datasets=["host_galaxies"],
        child_datasets=["sne"], name="sne_in_hosts",
    )

    now = db.load_subobject_links("sne_in_hosts")
    assert now.rules == r2

    old = db.load_subobject_links("sne_in_hosts", as_of=t_mid)
    assert old.rules == r1


def test_as_of_no_version_raises(tmp_path):
    db = _setup(tmp_path)
    r1 = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=2.0, dz_tol=1e-2,
    )
    db.build_subobject_links(
        rules=r1, parent_datasets=["host_galaxies"],
        child_datasets=["sne"], name="sne_in_hosts",
    )
    ancient = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    with pytest.raises(FileNotFoundError, match="valid at"):
        db.load_subobject_links("sne_in_hosts", as_of=ancient)
