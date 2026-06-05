"""Regression — the LIGHTCURVE writer stamps the canonical OUF version.

Previously it hard-coded "2.1.0" while POINT/SIGHTLINE used FORMAT_VERSION
(2.5.0), so LIGHTCURVE datasets were mislabelled. No test caught it.
"""
import numpy as np
import pandas as pd

from oneuniverse.data._converter_lightcurve import write_ouf_lightcurve_dataset
from oneuniverse.data.format_spec import (FORMAT_VERSION, ONEUNIVERSE_SUBDIR,
                                          SCHEMA_VERSION)
from oneuniverse.data.manifest import read_manifest


def test_lightcurve_writer_uses_canonical_version(tmp_path):
    objects = pd.DataFrame({
        "object_id": np.arange(3, dtype=np.int64),
        "ra": [10.0, 11.0, 12.0], "dec": [0.0, 1.0, 2.0],
        "z": [0.1, 0.2, 0.3], "z_type": ["spec"] * 3, "z_err": [1e-3] * 3})
    epochs = pd.DataFrame({
        "object_id": np.repeat(np.arange(3), 4),
        "mjd": np.tile(np.linspace(58000, 59000, 4), 3),
        "filter": ["g"] * 12, "flux": np.ones(12), "flux_err": np.ones(12),
        "flag": np.zeros(12, dtype=int)})
    survey = tmp_path / "lc"
    write_ouf_lightcurve_dataset(objects=objects, epochs=epochs,
                                 survey_path=survey, survey_name="lc",
                                 survey_type="transient", loader_name="x",
                                 loader_version="0")
    m = read_manifest(survey / ONEUNIVERSE_SUBDIR / "manifest.json")
    assert m.oneuniverse_format_version == FORMAT_VERSION == "2.5.0"
    assert m.oneuniverse_schema_version == SCHEMA_VERSION == "2.5.0"
