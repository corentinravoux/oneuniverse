"""Shared fixtures. Expensive databases are built once per session and cloned."""
import numpy as np
import pytest
from fixtures.cosmology import COSMO
from fixtures.synthetic import write_synthetic_dataset
from fixtures.two_surveys import clone_database, make_two_survey_db

from oneuniverse._core.healpix import NSIDE_FINE
from oneuniverse.data import DatasetView
from oneuniverse.data.database import Database


@pytest.fixture(scope="session")
def _two_survey_template(tmp_path_factory):
    """Small a/b pair (2000 rows each, 500 shared objects), written once."""
    root = tmp_path_factory.mktemp("two_survey_small")
    make_two_survey_db(root, 2_000, 500, partition_nside=2)
    return root


@pytest.fixture(scope="session")
def _two_survey_large_template(tmp_path_factory):
    """Large a/b pair (12 000 rows each, 4 000 shared objects), written once."""
    root = tmp_path_factory.mktemp("two_survey_large")
    make_two_survey_db(root, 12_000, 4_000, partition_nside=2)
    return root


@pytest.fixture
def two_survey_db(_two_survey_template, tmp_path) -> Database:
    """Private small a/b database (no index built); safe to mutate."""
    return clone_database(_two_survey_template, tmp_path / "db")


@pytest.fixture(scope="module")
def two_survey_db_large(_two_survey_large_template, tmp_path_factory) -> Database:
    """Module-private large a/b database; tests in one module share it."""
    return clone_database(_two_survey_large_template, tmp_path_factory.mktemp("two_survey_clone") / "db")


@pytest.fixture(scope="session")
def synthetic_view(tmp_path_factory) -> DatasetView:
    """Read-only DatasetView on a 20 000-row synthetic POINT dataset (nside 4)."""
    path = tmp_path_factory.mktemp("synthetic_view") / "ds"
    write_synthetic_dataset(path, 20_000, 0, partition_nside=4)
    return DatasetView.from_path(path)


@pytest.fixture(autouse=True)
def _coarse_stage_buckets(request, monkeypatch):
    """Stage the writer into 12 buckets instead of 768 (about 0.8 s saved per full-sky write).

    The bucket size is an internal staging detail whose output is invariant (row order,
    partitions and statistics are pinned by tests). Tests marked ``full_stage`` keep the
    production setting.
    """
    if request.node.get_closest_marker("full_stage") is None:
        from oneuniverse.data import writer
        # Fail loudly if the internal knob is renamed or its production value changes:
        # otherwise the speedup (or the patch) would silently stop doing what we think.
        # Production staging stays covered by the ``full_stage`` tests in test/data/test_writer.py.
        assert getattr(writer, "_BUCKET_SHIFT", None) == 2 * (int(np.log2(NSIDE_FINE)) - 3), (
            "writer._BUCKET_SHIFT changed; update test/conftest.py and the full_stage tests")
        monkeypatch.setattr(writer, "_BUCKET_SHIFT", 2 * int(np.log2(NSIDE_FINE)))


@pytest.fixture(scope="session")
def cosmo():
    """The shared reference cosmology (same object as ``fixtures.cosmology.COSMO``)."""
    return COSMO
