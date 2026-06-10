"""Regressions for the 2026-06-10 structural-review bugs B1-B5.

B1 FieldMap axes/beam silently dropped on save  → preserved (arrays as lists),
   truly-unserialisable values raise instead of vanishing.
B2 weight component named 'total' crashed np.savez → namespaced, round-trips.
B3 generate_randoms with zero n(z) returned silent garbage → raises.
B4 ingested randoms not z-filtered / weight-less → filtered + weight=1.
B5 unclosed file handles in measure/io.py → none remain (source scan).
"""
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pytest

from oneuniverse.combine.weights import ColumnWeight
from oneuniverse.measure import MeasurementSet, build_galaxy_clustering
from oneuniverse.measure.dataproduct import FieldMap, PointSet
from oneuniverse.measure.measurement_set import MeasurementSet as MS
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.nz import Nz
from oneuniverse.measure.randoms import generate_randoms
from oneuniverse.measure.spec import MeasurementSpec
from oneuniverse.measure.weighting import NamedWeights
from oneuniverse.measure.window import footprint_from_positions

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402


def _meta(nside_region=8):
    return ProductMetadata(frame="icrs", epoch=2000.0, length_unit="deg",
                           nside_region=nside_region)


def test_b1_fieldmap_axes_round_trip_and_raise(tmp_path):
    nside = 8
    npix = hp.nside2npix(nside)
    fm = FieldMap(values=np.random.rand(npix), mask=np.ones(npix, bool),
                  nside=nside, region_map=np.array([], dtype=np.int64),
                  metadata=ProductMetadata(frame="galactic", epoch=2000.0,
                                           length_unit="dimensionless",
                                           nside_region=0),
                  provenance=Provenance(dataset_ids=("m",)),
                  axes={"freq_ghz": np.linspace(100, 120, 8)},
                  beam={"fwhm_arcmin": np.float32(6.0)})
    ms = MS({"m": fm}, MeasurementSpec(("m",), (("m", "m"),), "pk", "lim"),
            fm.metadata)
    ms.to_dir(tmp_path / "a")
    back = MS.from_dir(tmp_path / "a")
    bm = back.products["m"]
    assert bm.axes is not None                              # was silently None
    np.testing.assert_allclose(bm.axes["freq_ghz"], np.linspace(100, 120, 8))
    assert bm.beam["fwhm_arcmin"] == pytest.approx(6.0)
    # truly unserialisable -> raise, never drop
    fm2 = FieldMap(values=fm.values, mask=fm.mask, nside=nside,
                   region_map=fm.region_map, metadata=fm.metadata,
                   provenance=fm.provenance, axes={"bad": object()})
    ms2 = MS({"m": fm2}, ms.spec, fm.metadata)
    with pytest.raises(TypeError, match="non-serialisable"):
        ms2.to_dir(tmp_path / "b")


def test_b2_weight_component_named_total_round_trips(tmp_path):
    n = 5
    nw = NamedWeights(total=np.ones(n), components={"total": np.full(n, 2.0),
                                                    "fkp": np.full(n, 0.5)},
                      recipe=("total=x", "fkp=y"))
    ps = PointSet(catalog=pd.DataFrame({"ra": np.zeros(n), "dec": np.zeros(n),
                                        "z": np.full(n, 0.5)}),
                  region_map=np.zeros(n, dtype=np.int64), metadata=_meta(),
                  provenance=Provenance(dataset_ids=("x",)), weights=nw)
    ms = MS({"g": ps}, MeasurementSpec(("g",), (("g", "g"),),
                                       "pk_multipole", "clustering"),
            ps.metadata)
    ms.to_dir(tmp_path / "w")                               # crashed before
    back = MS.from_dir(tmp_path / "w").products["g"].weights
    np.testing.assert_allclose(back.total, 1.0)             # not overwritten
    np.testing.assert_allclose(back.components["total"], 2.0)
    np.testing.assert_allclose(back.components["fkp"], 0.5)


def test_b3_zero_nz_randoms_raises():
    win = footprint_from_positions(np.array([10.0, 11.0]),
                                   np.array([0.0, 1.0]), nside=16)
    nz0 = Nz(np.linspace(0, 1, 5), np.zeros(4), "spec_hist")
    with pytest.raises(ValueError, match="zero/invalid total weight"):
        generate_randoms(win, nz0, n_randoms=10, seed=0)    # silent NaNs before


def test_b4_ingested_randoms_filtered_and_weighted(tmp_path):
    data = synthetic_point_view(tmp_path, n=2000, seed=8, name="data")
    rand = synthetic_point_view(tmp_path, n=6000, seed=42, name="rand")
    zlo, zhi = 0.4, 0.6
    ms = build_galaxy_clustering(data, tracer="gal", z_range=(zlo, zhi),
                                 weights=[ColumnWeight("weight_comp")],
                                 nz_edges=np.linspace(0.3, 0.7, 9),
                                 randoms=rand, nside_region=4)
    rnd = ms.products["gal"].randoms
    assert rnd["z"].between(zlo, zhi).all()                 # z-cut applied
    assert "weight" in rnd.columns and (rnd["weight"] == 1.0).all()
    assert len(rnd) < 6000                                  # cut removed rows


def test_b5_no_unclosed_handles_in_io():
    src = (Path(__file__).parent.parent / "oneuniverse" / "measure"
           / "io.py").read_text()
    import re
    assert not re.search(r"json\.(dump|load)\(.*open\(", src)
