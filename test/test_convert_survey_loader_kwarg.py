"""Phase 12 D3: convert_survey accepts an explicit loader= instance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data.converter import convert_survey
from oneuniverse.data.dataset_view import DatasetView


class _InlineLoader(BaseSurveyLoader):
    """Loader instance built on-the-fly, NOT decorated with @register."""

    config = SurveyConfig(
        name="inline_fake",
        survey_type="spectroscopic",
        description="inline-built loader, not registered",
        column_groups=("core",),
    )

    def __init__(self, df):
        self._df = df

    def _load_raw(self, data_path=None, **kwargs):
        return self._df.copy()


def _df(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({
        "ra": np.linspace(0.0, 90.0, n, dtype=np.float64),
        "dec": np.linspace(-10.0, 10.0, n, dtype=np.float64),
        "z": np.full(n, 0.5, dtype=np.float32),
        "z_type": np.array(["spec"] * n, dtype="<U4"),
        "z_spec_err": np.full(n, 1e-3, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.array(["inline_fake"] * n, dtype="<U16"),
    })


def test_convert_survey_accepts_loader_instance(tmp_path):
    loader = _InlineLoader(_df())
    out = tmp_path / "inline_fake"
    convert_survey(loader=loader, output_dir=out, overwrite=True)

    view = DatasetView.from_path(out)
    got = view.read(columns=["ra", "dec", "z", "z_type"])
    assert len(got) == 50
    assert set(got["z_type"].unique()) <= {"spec"}


def test_convert_survey_requires_name_or_loader():
    with pytest.raises(TypeError, match="survey_name|loader"):
        convert_survey()
