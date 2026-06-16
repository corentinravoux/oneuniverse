"""Phase S5 T1 — streaming bucket chunker (bounded memory, fused bbox)."""
import json
import tracemalloc

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _rowcount(store, product, zt):
    from oneuniverse.simulation.oufsim._layout import read_store_layout
    info = read_store_layout(store)  # S11: layout moved to its own sidecar
    return info[product][zt]["n_rows"]


def test_streaming_preserves_rows_and_bbox(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=32, redshifts=(0.0,), seed=2)
    store = write_oufsim_store(native, tmp_path / "s", sim_name="d",
                               particle_chunk_nside=4)
    s = SimStore(store)
    cube = Cube(0, 50, 0, 50, 0, 50)
    sel = s.read_box("snapshots", 0.0, cube)
    assert sel["x"].max() <= 50.0 and sel["x"].min() >= 0.0
    assert _rowcount(store, "snapshots", "z0.000") == 32 ** 3


def test_peak_memory_is_bounded(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=300.0,
                                 n_grid=64, redshifts=(0.0,), seed=2)
    tracemalloc.start()
    write_oufsim_store(native, tmp_path / "s", sim_name="d",
                       particle_chunk_nside=4, batch_rows=200_000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # no full sorted copy of every column (~6×8×64^3 ≈ 100 MB) materialised
    assert peak < 80 * 1e6
