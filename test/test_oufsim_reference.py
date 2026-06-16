"""Phase S15 — wrap-in-place (`reference`) vs re-encode field projection."""
import json

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.oufsim.native import NumpyFieldAdapter
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _dirsize(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def test_reference_is_index_only_and_matches_reencode(tmp_path):
    c = _cosmo()
    native = generate_linear_sim(tmp_path / "n", c, box_size=200.0, n_grid=48,
                                 redshifts=(0.0,), seed=2)
    re = write_oufsim_store(native, tmp_path / "re", sim_name="re")
    rf = write_oufsim_store(native, tmp_path / "rf", sim_name="rf",
                            field_projection="reference")
    # reference field dir has NO copied tiles, only the sidecar index
    rf_fields = rf / "fields" / "z0.000"
    assert not list(rf_fields.glob("tile_*.npy"))
    assert (rf_fields / "_index.parquet").is_file()
    # the layout sidecar records the projection (S11: moved out of manifest)
    from oneuniverse.simulation.oufsim._layout import read_store_layout
    layout = read_store_layout(rf)
    assert layout["fields"]["z0.000"]["projection"] == "reference"
    # reads are identical (reference memmaps the native field)
    cube = Cube(0, 80, 0, 80, 0, 80)
    a, _ = SimStore(re).read_field_box(0.0, cube)
    b, _ = SimStore(rf).read_field_box(0.0, cube)
    np.testing.assert_array_equal(a, b)
    # the reference fields product is far smaller (index-only, no copy)
    assert _dirsize(rf / "fields") < 0.2 * _dirsize(re / "fields")


def test_native_adapter_reads_subregion(tmp_path):
    arr = np.arange(64 ** 3, dtype=np.float64).reshape(64, 64, 64)
    p = tmp_path / "f.npy"; np.save(p, arr)
    sub = NumpyFieldAdapter().read_field_region(
        p, (slice(0, 8), slice(0, 8), slice(0, 8)))
    np.testing.assert_array_equal(sub, arr[:8, :8, :8])
