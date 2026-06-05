"""measure — MeasurementSet.summary() introspection across subtypes."""
import sys
from pathlib import Path

import numpy as np

from oneuniverse.combine.weights import ColumnWeight
from oneuniverse.measure import (build_galaxy_clustering, build_lya,
                                 build_map_cross)
from oneuniverse.measure.fieldmap import fieldmap_from_healpix

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import (synthetic_healpix_map,  # noqa: E402
    synthetic_point_view, synthetic_sightline_view)


def test_summary_pointset(tmp_path):
    view = synthetic_point_view(tmp_path, n=2000, seed=1)
    ms = build_galaxy_clustering(view, tracer="gal", z_range=(0.1, 1.0),
                                 weights=[ColumnWeight("weight_comp")],
                                 nz_edges=np.linspace(0, 1.2, 13),
                                 randoms="generate", n_randoms=4000, seed=1)
    s = ms.summary()
    assert s["cosmology_free"] is True
    assert s["n_products"] == 1
    g = s["products"]["gal"]
    assert g["kind"] == "pointset" and g["has_randoms"] and g["has_nz"]
    assert g["has_window"] and g["n"] > 0
    assert "MeasurementSet" in repr(ms)


def test_summary_sightline_and_fieldmap(tmp_path):
    lview = synthetic_sightline_view(tmp_path, n_los=10, n_pix=15, seed=2)
    s = build_lya(lview).summary()
    assert s["products"]["lya"]["kind"] == "sightline"
    assert s["products"]["lya"]["n_sightlines"] == 10

    gview = synthetic_point_view(tmp_path, n=2000, seed=3, name="g")
    vals, mask = synthetic_healpix_map(nside=32, seed=4)
    fm = fieldmap_from_healpix(vals, mask=mask, nside=32)
    s2 = build_map_cross(gview, fm, nside_region=4, z_range=(0.1, 1.0)).summary()
    assert set(s2["products"]) == {"gal", "kappa"}
    assert s2["products"]["kappa"]["kind"] == "fieldmap"
    assert s2["products"]["kappa"]["covered_pixels"] > 0
    assert s2["spec"]["statistic"] == "cl"
