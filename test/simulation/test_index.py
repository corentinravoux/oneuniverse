import json

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

from oneuniverse._core.errors import ValidationError
from oneuniverse._core.geometry import Cube
from oneuniverse.simulation.store.index import (
    CATALOG_INDEX_SCHEMA,
    FIELD_INDEX_SCHEMA,
    SKY_INDEX_SCHEMA,
    chunk_bbox,
    chunk_ids,
    overlapping_chunks,
    overlapping_tiles,
    split_periodic,
)


def _chunk_index(box, n):
    k = n ** 3
    cols = {f"{a}_{m}": [chunk_bbox(i, box, n)[0 if m == "min" else 1][c] for i in range(k)]
            for c, a in enumerate("xyz") for m in ("min", "max")}
    return pa.table({"chunk_id": pa.array(np.arange(k), pa.int32()), "path": [""] * k,
                     "row_start": [0] * k, "row_stop": [0] * k, **cols})


def test_chunk_ids_and_bbox_round_trip():
    pos = np.array([[0.1, 0.1, 0.1], [99.9, 50.0, 12.5]])
    ids = chunk_ids(pos, 100.0, 4)
    lo, hi = chunk_bbox(ids[1], 100.0, 4)
    assert ids.dtype == np.int32 and lo[0] == 75.0 and hi[0] == 100.0
    assert lo[1] == 50.0 and lo[2] == 0.0
    assert ids[0] == 0


def test_chunk_ids_edges_and_wrap():
    eps = 1e-9
    pos = np.array([[100.0 - eps, 0.0, 0.0], [0.0, 0.0, 0.0], [100.0, 0.0, 0.0],
                    [-eps, 0.0, 0.0], [125.0, 0.0, 0.0]])
    ids = chunk_ids(pos, 100.0, 4)
    assert ids[0] == 3 * 16 and ids[1] == 0
    assert ids[2] == 0            # x == box wraps to 0
    assert ids[3] == 3 * 16       # tiny negative wraps into last chunk
    assert ids[4] == 1 * 16
    for i in range(64):
        lo, hi = chunk_bbox(i, 100.0, 4)
        c = (np.array(lo) + np.array(hi)) / 2
        assert chunk_ids(c[None], 100.0, 4)[0] == i


def test_split_periodic():
    assert split_periodic(Cube((10, 10, 10), (20, 20, 20)), 100.0) == [Cube((10, 10, 10), (20, 20, 20))]
    parts = split_periodic(Cube((95, 10, 10), (105, 20, 20)), 100.0)
    assert len(parts) == 2 and sum(p.volume for p in parts) == pytest.approx(1000.0)
    assert all(0 <= p.lo[0] and p.hi[0] <= 100 for p in parts)
    assert len(split_periodic(Cube((-5, -5, -5), (5, 5, 5)), 100.0)) == 8
    assert split_periodic(Cube((95, 10, 10), (105, 20, 20)), 100.0) == parts  # deterministic
    with pytest.raises(ValidationError):
        split_periodic(Cube((0, 0, 0), (101, 5, 5)), 100.0)
    full = split_periodic(Cube((0, 0, 0), (100, 100, 100)), 100.0)
    assert len(full) == 1


def test_overlapping_chunks_is_exact_superset():
    pos = np.random.default_rng(0).uniform(0, 100, (20_000, 3))
    ids = chunk_ids(pos, 100.0, 4)
    idx = _chunk_index(100.0, 4)
    cube = Cube((30, 30, 30), (55, 55, 55))
    hit = overlapping_chunks(idx, cube, 100.0)
    inside = cube.contains(*pos.T)
    assert np.isin(ids[inside], hit).all() and len(hit) == 8
    assert hit.dtype == np.int32 and (np.diff(hit) > 0).all()


def test_overlapping_chunks_periodic():
    idx = _chunk_index(100.0, 4)
    hit = overlapping_chunks(idx, Cube((95, 10, 10), (105, 20, 20)), 100.0)
    xs = {int(h) // 16 for h in hit}
    assert xs == {0, 3}


def test_overlapping_tiles():
    n_grid, tile = 16, 8
    rows = []
    for t, (i, j, k) in enumerate(np.ndindex(2, 2, 2)):
        rows.append((t, "", i * tile, (i + 1) * tile, j * tile, (j + 1) * tile,
                     k * tile, (k + 1) * tile))
    cols = ["tile_id", "path", "i0", "i1", "j0", "j1", "k0", "k1"]
    idx = pa.table({c: [r[n] for r in rows] for n, c in enumerate(cols)})
    # box 16: cell = 1; cube [1, 7] stays in tile (0,0,0); hi == 8 touches cell 8
    assert overlapping_tiles(idx, Cube((1, 1, 1), (7, 7, 7)), 16.0, n_grid).tolist() == [0]
    assert len(overlapping_tiles(idx, Cube((1, 1, 1), (8, 7, 7)), 16.0, n_grid)) == 2
    assert len(overlapping_tiles(idx, Cube((14, 14, 14), (18, 18, 18)), 16.0, n_grid)) == 8
    assert overlapping_tiles(idx, Cube((1, 1, 1), (7, 7, 7)), 16.0, n_grid).dtype == np.int32


def test_schemas():
    assert CATALOG_INDEX_SCHEMA.names == [
        "chunk_id", "path", "row_start", "row_stop",
        "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    assert CATALOG_INDEX_SCHEMA.field("chunk_id").type == pa.int32()
    assert FIELD_INDEX_SCHEMA.names == ["tile_id", "path", "i0", "i1", "j0", "j1", "k0", "k1"]
    assert SKY_INDEX_SCHEMA.names == ["pix", "path", "n_rows"]


def test_native_fixture_layout(tmp_path):
    p = make_native_sim(tmp_path / "sim", n_particles=2000, n_grid=8)
    assert (p / "particles_z1.000.npy").exists() and (p / "field_z0.000.npy").exists()
    assert (p / "lightcone.parquet").exists() and (p / "halos_z0.000.parquet").exists()
    cfg = json.loads((p / "config.json").read_text())
    assert cfg["box_size"] == 100.0 and cfg["n_grid"] == 8 and cfg["redshifts"] == [1.0, 0.0]
    prods = native_products(p)
    assert {x.kind_layout for x in prods} == {"catalog", "field", "lightcone"}
    assert native_manifest(p).box_size == 100.0 and native_manifest(p).products == ()
    arr = np.load(p / "particles_z1.000.npy", mmap_mode="r")
    assert arr.dtype.names == ("x", "y", "z", "vx", "vy", "vz") and len(arr) == 2000
    assert np.load(p / "field_z0.000.npy").shape == (8, 8, 8)
    assert {"x", "y", "z", "mass"} <= set(pq.read_schema(p / "halos_z0.000.parquet").names)
    assert pq.read_schema(p / "lightcone.parquet").names[:3] == ["ra", "dec", "z"]
    part = next(x for x in prods if x.decl.kind == "particles")
    tbl = pa.Table.from_batches(list(part.batches(1.0)))
    assert tbl.num_rows == 2000 and tbl.column_names == list(part.decl.columns)


def test_packed_fixture(tmp_path):
    p = make_native_sim(tmp_path / "sim", n_particles=3000, n_grid=8)
    q = write_packed_native_fixture(p, n_side=4)
    assert q != p and q.exists()
    hdr = json.loads((q / "header.json").read_text())
    assert hdr["n_side"] == 4 and hdr["box_size"] == 100.0
    arr = np.load(q / "particles_z1.000.npy")
    pos = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    ids = chunk_ids(pos, 100.0, 4)
    assert (np.diff(ids) >= 0).all()
    for cid, a, b in hdr["chunks"]["1.000"]:
        assert (ids[a:b] == cid).all()
    assert sum(b - a for _, a, b in hdr["chunks"]["1.000"]) == 3000
    assert (q / "lightcone.parquet").exists() and (q / "config.json").exists()
    prods = native_products(q, adapter_format="packed_npy")
    assert {x.decl.kind: x.adapter_format for x in prods}["particles"] == "packed_npy"
    assert {x.decl.kind: x.adapter_format for x in prods}["halos"] is None
