"""Phase 12 D1: module-level data-root state is gone."""
from __future__ import annotations

import pytest


def test_set_data_root_no_longer_importable_from_data():
    with pytest.raises(ImportError):
        from oneuniverse.data import set_data_root  # noqa: F401


def test_get_data_root_no_longer_importable_from_data():
    with pytest.raises(ImportError):
        from oneuniverse.data import get_data_root  # noqa: F401


def test_top_level_set_data_root_removed():
    with pytest.raises(ImportError):
        from oneuniverse import set_data_root  # noqa: F401


def test_env_data_root_still_available():
    from oneuniverse.data._config import env_data_root  # noqa: F401


def test_resolve_survey_path_accepts_data_root_kwarg(tmp_path):
    from oneuniverse.data._config import resolve_survey_path

    got = resolve_survey_path(
        "spectroscopic", "fake", "spectroscopic/fake",
        data_root=tmp_path,
    )
    assert got == tmp_path / "spectroscopic" / "fake"


def test_resolve_survey_path_returns_none_when_no_data_root(monkeypatch):
    from oneuniverse.data._config import resolve_survey_path

    monkeypatch.delenv("ONEUNIVERSE_DATA_ROOT", raising=False)
    assert resolve_survey_path("spectroscopic", "fake") is None
