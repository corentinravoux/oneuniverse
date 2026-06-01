"""Phase S2 T8 — RegionSpec."""
import pytest

from oneuniverse.simulation.region import RegionSpec
from oneuniverse.simulation.selectors import Cone


def test_eulerian_bbox_ok():
    r = RegionSpec(
        region_id="coma", kind="cluster",
        eulerian_bbox=(100.0, 110.0, 100.0, 110.0, 100.0, 110.0),
    )
    assert r.eulerian_bbox[1] == 110.0


def test_cone_region_ok():
    r = RegionSpec(
        region_id="patch1", kind="observed",
        cone=Cone(lon=120.0, lat=0.0, radius_deg=2.0),
        refs=("/data/oneuniverse/clusters/redmapper.parquet",),
    )
    assert r.cone.radius_deg == 2.0
    assert r.refs[0].endswith("redmapper.parquet")


def test_requires_at_least_one_geometry():
    with pytest.raises(ValueError, match="geometry"):
        RegionSpec(region_id="x", kind="void")


def test_roundtrip_with_cone():
    r = RegionSpec(
        region_id="patch1", kind="observed",
        cone=Cone(lon=120.0, lat=0.0, radius_deg=2.0),
        z=0.3, mass=1e14, refs=("/a.parquet",),
    )
    assert RegionSpec.from_dict(r.to_dict()) == r


def test_roundtrip_with_bbox_and_lagrangian():
    r = RegionSpec(
        region_id="zoom1", kind="lagrangian",
        eulerian_bbox=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        lagrangian_patch=(0.0, 0.5, 0.0, 0.5, 0.0, 0.5),
    )
    assert RegionSpec.from_dict(r.to_dict()) == r
