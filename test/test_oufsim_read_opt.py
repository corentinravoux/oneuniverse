"""Phase S6 T1/T2/T4 — read benchmark harness, column projection, index cache."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.oufsim.bench import ReadBenchmark, measure_read
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _store(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=32, redshifts=(0.0,), seed=2)
    return write_oufsim_store(native, tmp_path / "s", sim_name="d",
                              particle_chunk_nside=4)


def test_measure_read_reports_fields(tmp_path):
    s = SimStore(_store(tmp_path))
    bm = measure_read(lambda: s.read_box("snapshots", 0.0,
                                         Cube(0, 80, 0, 80, 0, 80)))
    assert isinstance(bm, ReadBenchmark)
    assert bm.wall_s >= 0.0 and bm.peak_bytes > 0 and bm.n_rows > 0


def test_projection_returns_only_requested(tmp_path):
    s = SimStore(_store(tmp_path))
    cube = Cube(0, 80, 0, 80, 0, 80)
    full = s.read_box("snapshots", 0.0, cube)
    proj = s.read_box("snapshots", 0.0, cube, columns=["x", "y", "z"])
    assert set(proj) == {"x", "y", "z"}
    assert len(proj["x"]) == len(full["x"])
    np.testing.assert_array_equal(np.sort(proj["x"]), np.sort(full["x"]))


def test_projection_reads_fewer_bytes(tmp_path):
    s = SimStore(_store(tmp_path))
    cube = Cube(0, 120, 0, 120, 0, 120)
    full = measure_read(lambda: s.read_box("snapshots", 0.0, cube))
    proj = measure_read(lambda: s.read_box("snapshots", 0.0, cube,
                                           columns=["x"]))
    assert proj.peak_bytes < full.peak_bytes


def test_index_cache_avoids_reread(tmp_path):
    s = SimStore(_store(tmp_path))
    cube = Cube(0, 50, 0, 50, 0, 50)
    s.read_box("snapshots", 0.0, cube)
    n_cached = len(s._index_cache)
    assert n_cached >= 1
    calls = {"n": 0}
    import pyarrow.parquet as pq
    orig = pq.read_table

    def counting(path, *a, **k):
        if str(path).endswith("_index.parquet"):
            calls["n"] += 1
        return orig(path, *a, **k)

    pq.read_table = counting
    try:
        s.read_box("snapshots", 0.0, cube)     # second identical query
    finally:
        pq.read_table = orig
    assert calls["n"] == 0                      # index served from cache


def test_parallel_read_matches_serial(tmp_path):
    s = SimStore(_store(tmp_path))
    cube = Cube(0, 150, 0, 150, 0, 150)
    a = s.read_box("snapshots", 0.0, cube, n_threads=1)
    b = s.read_box("snapshots", 0.0, cube, n_threads=4)
    assert len(a["x"]) == len(b["x"])
    np.testing.assert_array_equal(np.sort(a["x"]), np.sort(b["x"]))
