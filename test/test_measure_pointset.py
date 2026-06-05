"""measure T1 — synthetic view + PointSet carrier."""
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402


def test_synthetic_view_reads_point(tmp_path):
    view = synthetic_point_view(tmp_path, n=500, seed=1)
    df = view.read(columns=["ra", "dec", "z"])
    assert len(df) == 500 and {"ra", "dec", "z"} <= set(df.columns)


def test_pointset_holds_catalog_and_metadata(tmp_path):
    view = synthetic_point_view(tmp_path, n=500, seed=1)
    df = view.read()
    ps = PointSet(
        catalog=df, randoms=None, nz=None, window=None,
        region_map=np.zeros(len(df), dtype=np.int64),
        metadata=ProductMetadata(frame="icrs", epoch=2000.0,
                                 length_unit="deg", nside_region=8),
        provenance=Provenance(dataset_ids=("synth",)),
    )
    assert ps.kind == "pointset"
    assert ps.metadata.frame == "icrs"
    assert "cosmology" not in vars(ps.metadata)   # cosmology-free invariant
