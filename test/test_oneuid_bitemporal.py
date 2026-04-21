import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.database import OneuniverseDatabase
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec
from oneuniverse.data.oneuid_rules import CrossMatchRules
from oneuniverse.data.oneuid import ONEUID_DIR
from oneuniverse.data.validity import DatasetValidity


def _syn(root: Path, sub: str, name: str, n: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ra":    rng.uniform(0.0, 10.0, n),
        "dec":   rng.uniform(-5.0, 5.0, n),
        "z":     rng.uniform(0.1, 0.3, n),
        "z_type": ["spec"] * n,
        "z_err":  rng.uniform(1e-4, 1e-3, n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": [f"{name}_{i:03d}" for i in range(n)],
        "_original_row_index": np.arange(n, dtype=np.int64),
        "_healpix32": np.zeros(n, dtype=np.int64),
    })
    survey_dir = root / sub
    ou_dir = survey_dir / "oneuniverse"
    ou_dir.mkdir(parents=True, exist_ok=True)
    write_ouf_dataset(
        df, ou_dir,
        survey_name=name, survey_type=sub.split("/")[0],
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="syn", version="0"),
        validity=DatasetValidity(
            valid_from_utc="2020-01-01T00:00:00+00:00"),
    )


def _set_up_db(tmp_path):
    _syn(tmp_path, "spectroscopic/a", "spectroscopic_a", seed=1)
    _syn(tmp_path, "spectroscopic/b", "spectroscopic_b", seed=2)
    return OneuniverseDatabase.from_root(tmp_path)


def test_build_index_stamps_validity(tmp_path):
    db = _set_up_db(tmp_path)
    rules = CrossMatchRules(sky_tol_arcsec=3.0)
    db.build_oneuid(
        datasets=["spectroscopic_a", "spectroscopic_b"],
        rules=rules, name="v1",
    )
    idx = db.load_oneuid(name="v1")
    assert idx.validity is not None
    assert idx.validity.is_current()


def test_rebuild_closes_previous_version(tmp_path):
    db = _set_up_db(tmp_path)
    rules = CrossMatchRules(sky_tol_arcsec=3.0)
    db.build_oneuid(datasets=["spectroscopic_a"], rules=rules, name="v1")
    time.sleep(1.1)
    db.build_oneuid(
        datasets=["spectroscopic_a", "spectroscopic_b"],
        rules=rules, name="v1",
    )
    old_manifests = list(
        (Path(tmp_path) / ONEUID_DIR).glob("v1__*.manifest.json")
    )
    assert len(old_manifests) == 1
    closed = json.loads(old_manifests[0].read_text())["validity"]
    assert closed["valid_to_utc"] is not None


def test_load_oneuid_as_of_picks_earlier_version(tmp_path):
    db = _set_up_db(tmp_path)
    rules = CrossMatchRules(sky_tol_arcsec=3.0)
    db.build_oneuid(datasets=["spectroscopic_a"], rules=rules, name="v1")
    time.sleep(1.1)
    db.build_oneuid(
        datasets=["spectroscopic_a", "spectroscopic_b"],
        rules=rules, name="v1",
    )
    later_current = db.load_oneuid(name="v1")
    t_mid = dt.datetime.fromisoformat(
        later_current.validity.valid_from_utc
    ) - dt.timedelta(seconds=1)
    earlier = db.load_oneuid(name="v1", as_of=t_mid)
    assert earlier is not None
    assert tuple(earlier.datasets()) == ("spectroscopic_a",)


def test_list_oneuids_bitemporal_contains_archived(tmp_path):
    db = _set_up_db(tmp_path)
    rules = CrossMatchRules(sky_tol_arcsec=3.0)
    db.build_oneuid(datasets=["spectroscopic_a"], rules=rules, name="v1")
    time.sleep(1.1)
    db.build_oneuid(
        datasets=["spectroscopic_a", "spectroscopic_b"],
        rules=rules, name="v1",
    )
    current = db.list_oneuids()
    assert "v1" in current

    archived = db.list_oneuids(include_archived=True)
    assert "v1" in archived
    assert any(label.startswith("v1__") for label in archived)
