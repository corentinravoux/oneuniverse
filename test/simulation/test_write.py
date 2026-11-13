import subprocess
import sys
import textwrap
import tracemalloc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fixtures.native_sim import (
    make_native_sim,
    native_manifest,
    native_products,
    write_packed_native_fixture,
)

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse.simulation.execution import ExecutionPlan
from oneuniverse.simulation.store import write as W
from oneuniverse.simulation.store.build import build_store
from oneuniverse.simulation.store.native import get_adapter
from oneuniverse.simulation.store.read import SimStore
from oneuniverse.simulation.store.write import (
    ingest_field,
    write_catalog_product,
    write_field_product,
    write_lightcone_product,
)


def _batches(n, box=100.0, seed=0, nb=4):
    rng = np.random.default_rng(seed)
    for _ in range(nb):
        yield pa.RecordBatch.from_pydict(
            {c: rng.uniform(0, box, n // nb) for c in "xyz"} | {"m": rng.uniform(size=n // nb)})


def test_build_store_reencode_layout(tmp_path):
    src = make_native_sim(tmp_path / "sim")
    store = build_store(tmp_path / "store", manifest=native_manifest(src), products=native_products(src),
                        plan=ExecutionPlan(), n_side=4, tile_cells=16)
    assert set(store.products) == {"particles", "field", "halos", "lightcone"}
    assert store.redshifts("particles") == (1.0, 0.0)
    assert (tmp_path / "store/manifest.json").exists() and (tmp_path / "store/layout.json").exists()
    assert store.index("particles", 0.0).num_rows == 64
    assert store.manifest.product("field").n_grid == 32
    assert store.layout["products"]["field"]["0.000"]["layout"] == "tiles"
    assert store.layout["products"]["lightcone"]["0.000"]["layout"] == "sky"
    idx = store.index("particles", 0.0)
    assert idx["row_stop"].to_numpy()[-1] == 50_000
    assert not (tmp_path / "store.tmp").exists()


def test_catalog_write_streams_under_budget(tmp_path):
    def gen():
        for i in range(20):
            rng = np.random.default_rng(i)
            yield pa.RecordBatch.from_pydict({c: rng.uniform(0, 100, 100_000) for c in "xyz"})
    tracemalloc.start()
    write_catalog_product(tmp_path / "s", "p", 0.0, gen(), box_size=100.0, n_side=4,
                          plan=ExecutionPlan(memory_budget_bytes=256 * 2**20))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    idx = pq.read_table(tmp_path / "s/products/p/z=0.000/_index.parquet")
    assert idx["row_stop"].to_numpy().sum() - idx["row_start"].to_numpy().sum() == 2_000_000
    assert peak < 300 * 2**20


def test_catalog_write_rss_growth_fresh_process(tmp_path):
    code = textwrap.dedent("""
        import sys, numpy as np, pyarrow as pa, pyarrow.parquet
        from oneuniverse.simulation.execution import ExecutionPlan
        from oneuniverse.simulation.store.write import write_catalog_product
        def rss():
            for l in open('/proc/self/status'):
                if l.startswith('VmRSS'): return int(l.split()[1]) * 1024
        def peak():
            for l in open('/proc/self/status'):
                if l.startswith('VmHWM'): return int(l.split()[1]) * 1024
        def gen():
            for i in range(20):
                rng = np.random.default_rng(i)
                yield pa.RecordBatch.from_pydict({c: rng.uniform(0, 100, 100_000) for c in 'xyz'})
        base = rss()
        write_catalog_product(sys.argv[1], 'p', 0.0, gen(), box_size=100.0, n_side=4,
                              plan=ExecutionPlan(memory_budget_bytes=256 * 2**20))
        print(peak() - base)
    """)
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "s")], capture_output=True,
                         text=True, check=True)
    growth = int(out.stdout.strip().splitlines()[-1])
    assert growth < 300 * 2**20, growth


def test_field_write_uses_memmap_slices(tmp_path, monkeypatch):
    f = np.lib.format.open_memmap(tmp_path / "f.npy", mode="w+", dtype=np.float32, shape=(64, 64, 64))
    f[:] = 1.0
    f.flush()
    mm = np.load(tmp_path / "f.npy", mmap_mode="r")
    import numpy
    spy = []
    monkeypatch.setattr(numpy, "asarray",
                        lambda a, *k, **kw: (spy.append(getattr(a, "shape", None)) or np.array(a, *k, **kw)))
    write_field_product(tmp_path / "s", "field", 0.0, mm, tile_cells=16, plan=ExecutionPlan())
    assert all(s != (64, 64, 64) for s in spy if s)


def test_field_tiles_uneven_and_exact(tmp_path):
    f = np.random.default_rng(1).normal(size=(10, 10, 10)).astype(np.float32)
    entry = write_field_product(tmp_path / "s", "f", 0.0, f, tile_cells=4)
    assert entry["n_tiles"] == 27
    idx = pq.read_table(tmp_path / "s/products/f/z=0.000/_index.parquet").to_pydict()
    rebuilt = np.zeros_like(f)
    for tid, p, i0, i1, j0, j1, k0, k1 in zip(idx["tile_id"], idx["path"], idx["i0"], idx["i1"],
                                              idx["j0"], idx["j1"], idx["k0"], idx["k1"]):
        t = np.load(tmp_path / "s/products/f/z=0.000" / p)
        assert t.shape == (i1 - i0, j1 - j0, k1 - k0)
        rebuilt[i0:i1, j0:j1, k0:k1] = t
    assert np.array_equal(rebuilt, f)
    with pytest.raises(ValidationError):
        write_field_product(tmp_path / "s", "g", 0.0, np.zeros((4, 4, 5), np.float32), tile_cells=4)


def test_catalog_index_rows_bbox_and_empty_chunks(tmp_path):
    pos = {"x": [1.0, 2.0, 60.0], "y": [1.0, 3.0, 60.0], "z": [1.0, 4.0, 60.0]}
    write_catalog_product(tmp_path / "s", "p", 0.0, [pa.RecordBatch.from_pydict(pos)],
                          box_size=100.0, n_side=2)
    idx = pq.read_table(tmp_path / "s/products/p/z=0.000/_index.parquet").to_pydict()
    assert idx["row_start"] == [0, 2, 2, 2, 2, 2, 2, 2] and idx["row_stop"] == [2, 2, 2, 2, 2, 2, 2, 3]
    assert idx["x_min"][0] == 1.0 and idx["x_max"][0] == 2.0 and idx["z_max"][0] == 4.0
    assert np.isnan(idx["x_min"][1]) and idx["path"][1] == ""
    files = sorted(p.name for p in (tmp_path / "s/products/p/z=0.000").iterdir())
    assert files == ["_index.parquet", "chunk_0.parquet", "chunk_7.parquet"]


def test_catalog_missing_positions_and_duplicate(tmp_path):
    with pytest.raises(ValidationError):
        write_catalog_product(tmp_path / "s", "p", 0.0, [pa.RecordBatch.from_pydict({"x": [1.0]})],
                              box_size=10.0, n_side=2)
    assert not list((tmp_path / "s/products/p").glob("*"))  # staging cleaned up
    b = list(_batches(40))
    write_catalog_product(tmp_path / "s", "p", 0.0, b, box_size=100.0, n_side=2)
    with pytest.raises(FormatError):
        write_catalog_product(tmp_path / "s", "p", 0.0, b, box_size=100.0, n_side=2)


def _chunk_contents(root, n):
    d = root / "products/p/z=0.000"
    return {c: pq.read_table(d / f"chunk_{c}.parquet").to_pydict()
            for c in range(n) if (d / f"chunk_{c}.parquet").exists()}


def test_bucketed_path_matches_direct_and_bounds_open_writers(tmp_path, monkeypatch):
    data = list(_batches(4000, seed=3, nb=5))
    write_catalog_product(tmp_path / "a", "p", 0.0, data, box_size=100.0, n_side=4)
    ref = _chunk_contents(tmp_path / "a", 64)
    live, peak = [0], [0]
    real = pq.ParquetWriter

    class Spy(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            live[0] += 1
            peak[0] = max(peak[0], live[0])

        def close(self):
            if not self.is_open:
                return
            super().close()
            live[0] -= 1

    monkeypatch.setattr(W.pq, "ParquetWriter", Spy)
    monkeypatch.setattr(W, "_MAX_OPEN", 5)  # forces two levels of bucketing for 64 chunks
    write_catalog_product(tmp_path / "b", "p", 0.0, data, box_size=100.0, n_side=4)
    assert peak[0] <= 5 and live[0] == 0
    got = _chunk_contents(tmp_path / "b", 64)
    assert got == ref
    ia = pq.read_table(tmp_path / "a/products/p/z=0.000/_index.parquet")
    ib = pq.read_table(tmp_path / "b/products/p/z=0.000/_index.parquet")
    assert ia.equals(ib)
    assert not (tmp_path / "b/products/p/z=0.000/_stage").exists()


def test_n_side_8_uses_at_most_256_writers(tmp_path, monkeypatch):
    live, peak = [0], [0]
    real = pq.ParquetWriter

    class Spy(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            live[0] += 1
            peak[0] = max(peak[0], live[0])

        def close(self):
            if not self.is_open:
                return
            super().close()
            live[0] -= 1

    monkeypatch.setattr(W.pq, "ParquetWriter", Spy)
    write_catalog_product(tmp_path / "s", "p", 0.0, _batches(20_000), box_size=100.0, n_side=8)
    assert peak[0] <= 256
    idx = pq.read_table(tmp_path / "s/products/p/z=0.000/_index.parquet")
    assert idx.num_rows == 512 and idx["row_stop"].to_numpy()[-1] == 20_000


def test_lightcone_partition(tmp_path):
    rng = np.random.default_rng(0)
    b = pa.RecordBatch.from_pydict({"ra": rng.uniform(0, 360, 2000),
                                    "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, 2000)))})
    entry = write_lightcone_product(tmp_path / "s", "lc", [b], nside_part=2, plan=ExecutionPlan())
    idx = pq.read_table(tmp_path / "s/products/lc/z=0.000/_index.parquet")
    assert idx["n_rows"].to_numpy().sum() == 2000 and entry["layout"] == "sky"
    total = sum(pq.read_table(tmp_path / "s/products/lc/z=0.000" / p).num_rows for p in idx["path"].to_pylist())
    assert total == 2000


def test_reference_projection_is_index_only(tmp_path):
    src = make_native_sim(tmp_path / "sim")
    packed = write_packed_native_fixture(src, n_side=4)
    store = build_store(tmp_path / "ref", manifest=native_manifest(src),
                        products=native_products(packed, adapter_format="packed_npy"),
                        plan=ExecutionPlan(), projection="reference", n_side=4)
    size_ref = sum(f.stat().st_size for f in (tmp_path / "ref").rglob("*") if f.is_file())
    size_native = sum(f.stat().st_size for f in packed.rglob("*.npy"))
    assert size_ref < 0.2 * size_native
    assert store.layout["products"]["particles"]["0.000"]["layout"] == "reference"
    assert store.layout["products"]["field"]["0.000"]["layout"] == "reference"
    assert store.layout["products"]["halos"]["0.000"]["layout"] == "chunks"  # parquet: re-encoded
    entry = store.layout["products"]["particles"]["0.000"]
    native = (tmp_path / "ref" / entry["native"]).resolve()
    assert native == (packed / "particles_z0.000.npy").resolve()
    idx = store.index("particles", 0.0)
    ad = get_adapter("packed_npy")
    r = idx.to_pylist()
    r = next(x for x in r if x["row_stop"] > x["row_start"])
    t = ad.read_rows(native, r["row_start"], r["row_stop"])
    lo, hi = (r["x_min"], r["y_min"], r["z_min"]), (r["x_max"], r["y_max"], r["z_max"])
    for c, a, b in zip("xyz", lo, hi):
        v = t[c].to_numpy()
        assert (v >= a).all() and (v <= b).all()


def test_reference_rejects_unsorted_adapter(tmp_path):
    src = make_native_sim(tmp_path / "sim", n_particles=1000)
    with pytest.raises(ValidationError):
        W.write_reference_catalog(tmp_path / "s", "p", 0.0, src / "particles_z0.000.npy",
                                  get_adapter("npy"), box_size=100.0, n_side=4)


def test_adapters_read_rows_and_regions(tmp_path):
    src = make_native_sim(tmp_path / "sim", n_particles=500, redshifts=(0.0,))
    ad = get_adapter("npy")
    arr = np.load(src / "particles_z0.000.npy")
    t = ad.read_rows(src / "particles_z0.000.npy", 10, 20, columns=["x", "vz"])
    assert t.column_names == ["x", "vz"] and t["x"].type == pa.float32()
    assert np.array_equal(t["x"].to_numpy(), arr["x"][10:20])
    assert ad.num_rows(src / "particles_z0.000.npy") == 500
    with pytest.raises(ValidationError):
        ad.read_rows(src / "particles_z0.000.npy", 0, 10, columns=["nope"])
    f = np.load(src / "field_z0.000.npy")
    assert ad.field_shape(src / "field_z0.000.npy") == (32, 32, 32)
    assert np.array_equal(ad.read_field_region(src / "field_z0.000.npy", (1, 2, 3), (5, 6, 7)), f[1:5, 2:6, 3:7])
    with pytest.raises(ValidationError):
        ad.read_field_region(src / "field_z0.000.npy", (0, 0, 0), (40, 1, 1))
    with pytest.raises(Exception, match="unknown"):
        get_adapter("nope")


def test_ingest_field_adds_product(tmp_path):
    src = make_native_sim(tmp_path / "sim")
    build_store(tmp_path / "s", manifest=native_manifest(src), products=native_products(src)[:1],
                plan=ExecutionPlan())
    ingest_field(tmp_path / "s", "delta_recon", 0.0, np.zeros((32, 32, 32), np.float32), tile_cells=16)
    store2 = SimStore(tmp_path / "s")
    assert "delta_recon" in store2.products and store2.manifest.product("delta_recon").kind == "field"
    assert store2.manifest.product("delta_recon").n_grid == 32
    assert store2.redshifts("delta_recon") == (0.0,) and store2.index("delta_recon", 0.0).num_rows == 8
    with pytest.raises(FormatError):
        ingest_field(tmp_path / "s", "delta_recon", 0.0, np.zeros((32, 32, 32), np.float32))
    ingest_field(tmp_path / "s", "delta_recon", 1.0, np.zeros((32, 32, 32), np.float32))
    assert SimStore(tmp_path / "s").redshifts("delta_recon") == (0.0, 1.0)
