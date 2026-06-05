"""measure Lyα — Sightline subtype + sightline_from_view + build_lya."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from oneuniverse.measure import MeasurementSet
from oneuniverse.measure.dataproduct import Sightline
from oneuniverse.measure.lya import build_lya
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.sightline import sightline_from_view

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_sightline_view  # noqa: E402


def test_sightline_subtype_holds_los_and_pixels():
    los = pd.DataFrame({"sightline_id": [0, 1], "ra": [10.0, 11.0],
                        "dec": [0.0, 1.0], "z_source": [2.3, 2.5]})
    sl = Sightline(
        los=los, delta=[np.zeros(3), np.zeros(4)],
        mask=[np.ones(3), np.ones(4)], continuum=None,
        region_map=np.array([0, 1], dtype=np.int64),
        metadata=ProductMetadata(frame="icrs", epoch=2000.0,
                                 length_unit="deg", nside_region=8),
        provenance=Provenance(dataset_ids=("lya",)))
    assert sl.kind == "sightline" and sl.n_sightlines == 2
    assert len(sl.delta[1]) == 4


def test_sightline_from_view(tmp_path):
    view = synthetic_sightline_view(tmp_path, n_los=12, n_pix=20, seed=1)
    sl = sightline_from_view(view)
    assert sl.kind == "sightline" and sl.n_sightlines == 12
    assert {"sightline_id", "ra", "dec"} <= set(sl.los.columns)
    assert len(sl.delta) == 12 and len(sl.delta[0]) == 20


def test_build_lya_p1d(tmp_path):
    view = synthetic_sightline_view(tmp_path, n_los=20, n_pix=24, seed=2)
    ms = build_lya(view, tracer="lya", statistic="p1d", nside_region=16)
    assert isinstance(ms, MeasurementSet)
    sl = ms.products["lya"]
    assert sl.kind == "sightline" and sl.n_sightlines == 20
    assert ms.spec.statistic == "p1d" and ms.spec.estimator_family == "lya"
    ms.check_invariants()
