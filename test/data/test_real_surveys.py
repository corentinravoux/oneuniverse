"""Conversion and cross-match of the real eBOSS DR16Q and DESI DR1 QSO catalogues.

Gated on ``ONEUNIVERSE_DATA_ROOT`` (a directory laid out like the survey archive);
skipped when it is unset or the files are missing. Run nightly with the slow suite.
"""
import os
from pathlib import Path

import pyarrow.compute as pc
import pytest

from oneuniverse._core.geometry import Cone
from oneuniverse.data import Database, DatasetView, convert
from oneuniverse.data.crossmatch import CrossMatchRules
from oneuniverse.data.loaders import get_loader

pytestmark = pytest.mark.slow


def _data_root() -> Path:
    root = os.environ.get("ONEUNIVERSE_DATA_ROOT")
    if not root:
        pytest.skip("ONEUNIVERSE_DATA_ROOT is not set")
    return Path(root)


def _source(survey: str) -> Path:
    """Raw-file directory of *survey* in the archive layout (the loader's ``archive_path``,
    shared with ``scripts/convert_surveys.py``)."""
    loader = get_loader(survey)
    src = _data_root() / loader.archive_path
    if not (src / loader.filename).is_file():
        pytest.skip(f"{src / loader.filename} not found")
    return src


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Both catalogues converted once into a temporary Database root."""
    eboss = _source("eboss_qso")
    desi = _source("desi_qso")
    root = tmp_path_factory.mktemp("real_surveys")
    convert("eboss_qso", source=eboss, out=root / "eboss_qso")
    convert("desi_qso", source=desi, out=root / "desi_qso")
    return root


def _check(root: Path, name: str, n_min: int) -> DatasetView:
    view = DatasetView.from_path(root / name)
    assert view.n_rows > n_min
    z_type = view.table(columns=["z_type"])["z_type"]
    assert set(pc.unique(z_type).to_pylist()) == {"spec"}
    # a cone on the survey's own first rows must return rows
    first = view.table(columns=["ra", "dec"]).slice(0, 1).to_pylist()[0]
    assert view.table(cone=Cone(first["ra"], first["dec"], 1.0)).num_rows > 0
    return view


def test_real_eboss_and_desi_convert(converted):
    _check(converted, "eboss_qso", 900_000)
    _check(converted, "desi_qso", 1_200_000)


def test_real_cross_match_finds_shared_quasars(converted):
    db = Database(converted)
    idx = db.build_oneuid(["eboss_qso", "desi_qso"],
                          rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol=5e-3), name="real")
    assert idx.n_multi > 200_000
