import numpy as np
import pyarrow.parquet as pq
import pytest
from fixtures.native_sim import (
    make_native_sim,
    native_manifest,
    native_products,
    write_packed_native_fixture,
)

from oneuniverse._core.errors import NotFoundError, ValidationError
from oneuniverse.simulation import Cone, Cube
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.store.build import build_store
from oneuniverse.simulation.store.index import split_periodic
from oneuniverse.simulation.store.read import SimStore


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    src = make_native_sim(tmp_path_factory.mktemp("n") / "sim", n_particles=200_000)
    return build_store(tmp_path_factory.mktemp("s") / "store", manifest=native_manifest(src),
                       products=native_products(src), plan=ExecutionPlan(), n_side=4, tile_cells=16), src


def test_read_box_exact_and_pruned(store):
    st, src = store
    cube = Cube((10, 10, 10), (30, 30, 30))
    t = st.read_box("particles", 0.0, cube)
    raw = np.load(src / "particles_z0.000.npy")
    inside = cube.contains(raw["x"], raw["y"], raw["z"])
    assert t.num_rows == inside.sum()
    assert st.last_read_stats["chunks_read"] <= 8 and st.last_read_stats["chunks_total"] == 64
    assert st.last_read_stats["rows"] == t.num_rows and st.last_read_stats["bytes"] == t.nbytes


def test_read_box_periodic_wrap(store):
    st, _ = store
    cube = Cube((95, 95, 95), (105, 105, 105))
    t = st.read_box("particles", 0.0, cube)
    parts = [st.read_box("particles", 0.0, c) for c in split_periodic(cube, 100.0)]
    assert t.num_rows == sum(p.num_rows for p in parts) and t.num_rows > 0 and len(parts) == 8


def test_read_box_columns_projection(store):
    st, _ = store
    cube = Cube((10, 10, 10), (30, 30, 30))
    full = st.read_box("particles", 0.0, cube)
    t = st.read_box("particles", 0.0, cube, columns=["vx"])
    assert t.column_names == ["vx"] and t["vx"].equals(full["vx"])


def test_iter_box_batches_and_columns(store):
    st, _ = store
    it = st.iter_box("particles", 0.0, Cube((0, 0, 0), (100, 100, 100)), columns=["x"],
                     plan=ExecutionPlan(memory_budget_bytes=2**16))
    b = next(it)
    assert b.column_names == ["x"] and b.num_rows <= 2**16 // 8


def test_iter_box_total_matches_read_box(store):
    st, _ = store
    cube = Cube((40, 40, 40), (70, 80, 90))
    n = sum(b.num_rows for b in st.iter_box("particles", 0.0, cube))
    assert n == st.read_box("particles", 0.0, cube).num_rows


def test_read_box_threads_deterministic(store):
    st, _ = store
    cube = Cube((0, 0, 0), (60, 60, 60))
    a = st.read_box("particles", 0.0, cube, plan=ExecutionPlan(n_threads=1))
    b = st.read_box("particles", 0.0, cube, plan=ExecutionPlan(n_threads=6))
    assert a.equals(b)


class _FakeComm:
    def __init__(self, rank, size):
        self._r, self._s = rank, size

    def Get_rank(self):
        return self._r

    def Get_size(self):
        return self._s


def test_read_box_mpi_partitions_chunks(store):
    st, _ = store
    cube = Cube((0, 0, 0), (100, 100, 100))
    whole = st.read_box("particles", 0.0, cube)
    parts = [st.read_box("particles", 0.0, cube,
                         plan=ExecutionPlan(mode=ExecutionMode.MPI, mpi_comm=_FakeComm(r, 3)))
             for r in range(3)]
    assert all(0 < p.num_rows < whole.num_rows for p in parts)
    assert sum(p.num_rows for p in parts) == whole.num_rows
    assert sorted(np.concatenate([p["x"].to_numpy() for p in parts])) == sorted(whole["x"].to_numpy())


def test_read_field_box_matches_native_with_wrap(store):
    # cell i covers [i*cell, (i+1)*cell); every cell intersecting the cube, in wrapped order
    st, src = store
    f = np.load(src / "field_z0.000.npy")
    cell = 100.0 / 32
    sub = st.read_field_box("field", 0.0, Cube((90.0, 0.0, 0.0), (110.0, 20.0, 20.0)))
    i_cells = np.arange(int(90.0 // cell), int(np.ceil(110.0 / cell))) % 32
    jk = np.arange(0, int(np.ceil(20.0 / cell)))
    assert sub.dtype == np.float32
    np.testing.assert_array_equal(sub, f[np.ix_(i_cells, jk, jk)])


def test_read_cone_lightcone(store):
    st, src = store
    cone = Cone(45.0, 10.0, 5.0)
    t = st.read_cone("lightcone", cone)
    raw = pq.read_table(src / "lightcone.parquet")
    assert t.num_rows == cone.contains(raw["ra"].to_numpy(), raw["dec"].to_numpy()).sum()


def test_read_cone_z_range_and_columns(store):
    st, src = store
    cone = Cone(45.0, 10.0, 20.0)
    t = st.read_cone("lightcone", cone, z_range=(0.2, 0.6), columns=["ra"])
    raw = pq.read_table(src / "lightcone.parquet")
    ra, dec, z = (raw[c].to_numpy() for c in ("ra", "dec", "z"))
    want = cone.contains(ra, dec) & (z >= 0.2) & (z <= 0.6)
    assert t.column_names == ["ra"] and t.num_rows == want.sum() > 0


def test_scan_full_is_explicit(store):
    st, _ = store
    n = sum(b.num_rows for b in st.scan_full("particles", 0.0))
    assert n == 200_000 and "whole" in SimStore.scan_full.__doc__.lower()


def test_reference_store_reads_identically(tmp_path, store):
    st, src = store
    packed = write_packed_native_fixture(src, n_side=4)
    ref = build_store(tmp_path / "ref", manifest=native_manifest(src),
                      products=native_products(packed, adapter_format="packed_npy"),
                      plan=ExecutionPlan(), projection="reference", n_side=4)
    cube = Cube((10, 10, 10), (30, 30, 30))
    a = st.read_box("particles", 0.0, cube).sort_by("x")
    b = ref.read_box("particles", 0.0, cube).sort_by("x")
    assert a.equals(b)
    c = Cube((90.0, 0.0, 0.0), (110.0, 20.0, 20.0))
    np.testing.assert_array_equal(ref.read_field_box("field", 0.0, c), st.read_field_box("field", 0.0, c))
    assert sum(b.num_rows for b in ref.scan_full("particles", 0.0)) == 200_000


def test_errors(store):
    st, _ = store
    cube = Cube((0, 0, 0), (10, 10, 10))
    with pytest.raises(NotFoundError):
        st.read_box("nope", 0.0, cube)
    with pytest.raises(NotFoundError):
        st.read_box("particles", 3.0, cube)
    with pytest.raises(ValidationError, match="read_field_box"):
        st.read_box("field", 0.0, cube)
    with pytest.raises(ValidationError, match="read_cone"):
        st.read_box("lightcone", 0.0, cube)
    with pytest.raises(ValidationError, match="read_cone"):
        st.iter_box("lightcone", 0.0, cube)
    with pytest.raises(ValidationError, match="read_cone"):
        st.scan_full("lightcone", 0.0)
    with pytest.raises(ValidationError, match="read_box"):
        st.read_field_box("particles", 0.0, cube)
    with pytest.raises(ValidationError, match="read_box"):
        st.read_cone("particles", Cone(0.0, 0.0, 1.0))


def test_unknown_column_is_validation_error(store):
    st, _ = store
    with pytest.raises(ValidationError, match="nope"):
        st.read_box("particles", 0.0, Cube((0, 0, 0), (10, 10, 10)), columns=["nope"])


def test_read_field_box_oversized_cube(store):
    st, _ = store
    with pytest.raises(ValidationError, match="exceeds box_size"):
        st.read_field_box("field", 0.0, Cube((0, 0, 0), (120, 10, 10)))


def _oracle(raw, cube, box=100.0):
    """Rows selected on positions wrapped into [0, box); returned with their stored positions."""
    w = {a: np.mod(raw[a].astype(np.float64), box) for a in "xyz"}
    inside = np.zeros(len(raw), dtype=bool)
    for piece in split_periodic(cube, box):
        inside |= piece.contains(w["x"], w["y"], w["z"])
    return np.sort(np.stack([raw[a][inside] for a in "xyz"], axis=1), axis=0)


def _rows(t):
    return np.sort(np.stack([t[a].to_numpy() for a in "xyz"], axis=1), axis=0)


def test_read_box_periodic_wrap_matches_raw(store):
    st, src = store
    raw = np.load(src / "particles_z0.000.npy")
    for cube in (Cube((95, 95, 95), (105, 105, 105)),
                 Cube((90, 0, 20), (110, 15, 60)),
                 Cube((25, 50, 75), (50, 75, 100))):  # faces exactly on chunk edges
        t = st.read_box("particles", 0.0, cube)
        want = _oracle(raw, cube)
        assert t.num_rows == len(want) > 0
        np.testing.assert_array_equal(_rows(t), want)


def test_read_box_particles_on_box_and_chunk_faces(tmp_path):
    src = make_native_sim(tmp_path / "sim", n_particles=2_000)
    path = src / "particles_z0.000.npy"
    raw = np.load(path)
    raw["x"][:4] = [0.0, 100.0, 25.0, 50.0]
    raw["y"][:4] = [10.0, 10.0, 25.0, 75.0]
    raw["z"][:4] = [10.0, 10.0, 25.0, 100.0]
    np.save(path, raw)
    st = build_store(tmp_path / "st", manifest=native_manifest(src), products=native_products(src),
                     plan=ExecutionPlan(), n_side=4, tile_cells=16)
    for cube in (Cube((0, 0, 0), (5, 20, 20)), Cube((95, 0, 0), (105, 20, 20)),
                 Cube((99.5, 5, 5), (100.5, 15, 15)), Cube((25, 25, 25), (50, 75, 100)),
                 Cube((10, 10, 10), (25, 25, 25))):
        t = st.read_box("particles", 0.0, cube)
        want = _oracle(raw, cube)
        np.testing.assert_array_equal(_rows(t), want)
    # the x=0 and x=100 particles are each returned exactly once by a cube spanning both
    t = st.read_box("particles", 0.0, Cube((99.5, 9.5, 9.5), (100.5, 10.5, 10.5)))
    xs = t["x"].to_numpy().tolist()
    assert xs.count(0.0) == 1 and xs.count(100.0) == 1


def _write_oob_sim(tmp_path, x, n=2_000):
    src = make_native_sim(tmp_path / "sim", n_particles=n)
    path = src / "particles_z0.000.npy"
    raw = np.load(path)
    raw["x"][:len(x)] = x
    raw["y"][:len(x)] = 10.0
    raw["z"][:len(x)] = 10.0
    np.save(path, raw)
    return src, raw


def test_read_box_out_of_box_positions_reencode(tmp_path):
    src, raw = _write_oob_sim(tmp_path, [-0.1, 100.5, 100.0, 50.0, 150.0])
    st = build_store(tmp_path / "st", manifest=native_manifest(src), products=native_products(src),
                     plan=ExecutionPlan(), n_side=4, tile_cells=16)
    whole = Cube((0, 0, 0), (100, 100, 100))
    assert st.read_box("particles", 0.0, whole).num_rows == len(raw)
    assert sum(t.num_rows for t in st.scan_full("particles", 0.0)) == len(raw)
    for cube in (Cube((0, 0, 0), (1, 20, 20)), Cube((99, 0, 0), (101, 20, 20)),
                 Cube((49, 0, 0), (51, 20, 20)), Cube((95, 5, 5), (105, 15, 15))):
        np.testing.assert_array_equal(_rows(st.read_box("particles", 0.0, cube)),
                                      _oracle(raw, cube))
    t = st.read_box("particles", 0.0, Cube((0, 5, 5), (1, 15, 15)))
    xs = t["x"].to_numpy()
    assert all(v in xs for v in np.float32([100.0, 100.5])) and np.float32(-0.1) not in xs
    t = st.read_box("particles", 0.0, Cube((99, 5, 5), (100, 15, 15)))
    assert np.float32(-0.1) in t["x"].to_numpy()


def test_read_box_out_of_box_positions_reference(tmp_path):
    src, _ = _write_oob_sim(tmp_path, [-0.1, 100.5, 100.0])
    packed = write_packed_native_fixture(src, n_side=4)
    raw = np.load(packed / "particles_z0.000.npy")
    st = build_store(tmp_path / "st", manifest=native_manifest(packed), products=native_products(packed, adapter_format="packed_npy"),
                     plan=ExecutionPlan(), projection="reference", tile_cells=16)
    for cube in (Cube((0, 0, 0), (100, 100, 100)), Cube((0, 5, 5), (1, 15, 15)),
                 Cube((99, 5, 5), (101, 15, 15))):
        np.testing.assert_array_equal(_rows(st.read_box("particles", 0.0, cube)),
                                      _oracle(raw, cube))


def test_write_refuses_garbage_positions(tmp_path):
    for bad in (-100.5, 200.0):
        src, _ = _write_oob_sim(tmp_path / f"b{bad}", [bad])
        with pytest.raises(ValidationError, match="positions"):
            build_store(tmp_path / f"st{bad}", manifest=native_manifest(src),
                        products=native_products(src), plan=ExecutionPlan(), n_side=4)


def test_read_box_empty_region_keeps_schema(tmp_path):
    src, _ = _write_oob_sim(tmp_path, [10.0, 12.0, 14.0], n=3)
    st = build_store(tmp_path / "st", manifest=native_manifest(src), products=native_products(src),
                     plan=ExecutionPlan(), n_side=4, tile_cells=16)
    cube = Cube((60, 60, 60), (70, 70, 70))
    t = st.read_box("particles", 0.0, cube)
    assert t.num_rows == 0 and t.column_names == st.read_box(
        "particles", 0.0, Cube((0, 0, 0), (100, 100, 100))).column_names
    p = st.read_box("particles", 0.0, cube, columns=["vx", "x"])
    assert p.num_rows == 0 and p.column_names == ["vx", "x"]


def test_scan_full_mpi_partition(store):
    st, _ = store
    whole = sum(b.num_rows for b in st.scan_full("particles", 0.0))
    parts = [sum(b.num_rows for b in st.scan_full(
        "particles", 0.0, plan=ExecutionPlan(mode=ExecutionMode.MPI, mpi_comm=_FakeComm(r, 3))))
        for r in range(3)]
    assert sum(parts) == whole and all(p < whole for p in parts)


def test_read_cone_z_range_without_z_column(tmp_path):
    from oneuniverse._core.errors import SchemaError
    src = make_native_sim(tmp_path / "sim", n_particles=100)
    prods = native_products(src)
    for p in prods:
        if p.kind_layout == "lightcone":
            orig = p.batches
            p.batches = lambda z, o=orig: (b.drop_columns(["z"]) for b in o(z))
    st = build_store(tmp_path / "st", manifest=native_manifest(src), products=prods,
                     plan=ExecutionPlan(), n_side=2, tile_cells=16)
    with pytest.raises(SchemaError, match="'z'"):
        st.read_cone("lightcone", Cone(10.0, 10.0, 60.0), z_range=(0.0, 1.0))
    assert st.read_cone("lightcone", Cone(10.0, 10.0, 60.0)).num_rows > 0
