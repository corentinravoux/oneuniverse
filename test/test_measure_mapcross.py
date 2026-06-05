"""measure map×catalog — FieldMap subtype + build_map_cross."""
import sys
from pathlib import Path

import healpy as hp
import numpy as np

from oneuniverse.measure import MeasurementSet
from oneuniverse.measure.dataproduct import FieldMap
from oneuniverse.measure.fieldmap import fieldmap_from_healpix
from oneuniverse.measure.mapcross import build_map_cross
from oneuniverse.measure.metadata import ProductMetadata, Provenance

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_healpix_map, synthetic_point_view  # noqa: E402


def test_fieldmap_subtype_and_ingest():
    vals, mask = synthetic_healpix_map(nside=32, seed=1)
    fm = fieldmap_from_healpix(vals, mask=mask, nside=32, frame="galactic")
    assert fm.kind == "fieldmap" and fm.nside == 32 and fm.npix == hp.nside2npix(32)
    assert fm.metadata.frame == "galactic" and fm.mask.dtype == bool
    # subtype is directly constructible too
    fm2 = FieldMap(values=vals, mask=mask, nside=32, nest=True,
                   region_map=np.array([], dtype=np.int64),
                   metadata=ProductMetadata(frame="galactic", epoch=2000.0,
                                            length_unit="dimensionless",
                                            nside_region=8),
                   provenance=Provenance(dataset_ids=("k",)))
    assert fm2.values.shape == fm2.mask.shape


def test_build_map_cross(tmp_path):
    gview = synthetic_point_view(tmp_path, n=4000, seed=3, name="gal")
    vals, mask = synthetic_healpix_map(nside=64, seed=4)
    fm = fieldmap_from_healpix(vals, mask=mask, nside=64, dataset_id="cmbk")
    ms = build_map_cross(gview, fm, gal_tracer="gal", map_tracer="kappa",
                         z_range=(0.1, 1.0), gal_weights_columns=("weight_comp",),
                         nside_region=4)
    assert isinstance(ms, MeasurementSet)
    assert set(ms.products) == {"gal", "kappa"}
    assert ("gal", "kappa") in ms.spec.pairs
    assert ms.spec.statistic == "cl" and ms.spec.estimator_family == "cross"
    ms.check_invariants()
