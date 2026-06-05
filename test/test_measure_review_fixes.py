"""M2 / L1 / L2 review fixes — cosmology guard, map randoms, shape attributes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.measure import (build_cosmic_shear, build_map_cross,
                                 build_galaxy_clustering)
from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.fieldmap import fieldmap_from_healpix
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.spec import MeasurementSpec

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import (synthetic_healpix_map,  # noqa: E402
    synthetic_point_view, synthetic_shear_view)


# ── M2: cosmology-derived columns are rejected by the invariant ─────────────
def test_cosmology_column_leak_is_rejected():
    n = 20
    cat = pd.DataFrame({"ra": np.zeros(n), "dec": np.zeros(n),
                        "z": np.linspace(0.1, 1, n),
                        "comoving_distance": np.linspace(300, 3000, n)})
    ps = PointSet(catalog=cat, region_map=np.zeros(n, dtype=np.int64),
                  metadata=ProductMetadata(frame="icrs", epoch=2000.0,
                                           length_unit="deg", nside_region=8),
                  provenance=Provenance(dataset_ids=("x",)))
    ms = MeasurementSet({"g": ps}, MeasurementSpec(
        ("g",), (("g", "g"),), "pk_multipole", "clustering"), ps.metadata)
    with pytest.raises(ValueError, match="cosmology-derived column"):
        ms.check_invariants()


def test_clean_catalog_passes_invariant(tmp_path):
    view = synthetic_point_view(tmp_path, n=500, seed=1)
    ms = build_galaxy_clustering(view, z_range=(0.1, 1.0),
                                 weights=[__import__("oneuniverse.combine.weights",
                                          fromlist=["ColumnWeight"]).ColumnWeight("weight_comp")],
                                 nz_edges=np.linspace(0, 1.2, 13),
                                 randoms="none")
    ms.check_invariants()                       # no forbidden columns -> ok


# ── L2: cosmic shear records the shape columns in attributes ────────────────
def test_cosmic_shear_populates_shape_attributes(tmp_path):
    view = synthetic_shear_view(tmp_path, n=1500, seed=3, with_pdf=True, n_tomo=2)
    ms = build_cosmic_shear(view, z_grid=np.linspace(0, 2, 41), nside_region=4)
    attrs = ms.products["src"].attributes
    assert attrs is not None and "e1" in attrs["shapes"] and "e2" in attrs["shapes"]


# ── L1: build_map_cross can attach galaxy randoms ───────────────────────────
def test_map_cross_can_generate_galaxy_randoms(tmp_path):
    gview = synthetic_point_view(tmp_path, n=2000, seed=3, name="g")
    vals, mask = synthetic_healpix_map(nside=32, seed=4)
    fm = fieldmap_from_healpix(vals, mask=mask, nside=32)
    ms = build_map_cross(gview, fm, nside_region=4, z_range=(0.1, 1.0),
                         randoms="generate", n_randoms=4000, seed=1)
    assert ms.products["gal"].randoms is not None
    assert len(ms.products["gal"].randoms) == 4000
    ms.check_invariants()
