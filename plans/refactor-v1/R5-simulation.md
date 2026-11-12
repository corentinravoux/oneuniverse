# R5 — simulation substrate: specs, manifest, execution, store, converter, SimDatabase

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `oneuniverse.simulation` as substrate only: typed specs, one JSON manifest validated on read, an execution plan without GPU stubs, a partial-access store (catalog chunks, field tiles, lightcone sky partitions, wrap-in-place projection) with periodic-aware reads, the converter/adapter seams, and a `SimDatabase` whose lineage matches what is on disk. No physics in this step; the fixture is a random "native simulation".

**Architecture:** Store layout per simulation: `<root>/<sim>/manifest.json` + `layout.json` + `products/<name>/z=<tag>/` holding either re-encoded parquet chunks / `.npy` tiles or a reference index over native files, each product with `_index.parquet`. `SimStore` reads only chunks/tiles overlapping the selector; a straddling cube is split into up to 8 wrapped sub-cubes. `SimDatabase.dispatch(request, engine)` calls a user-supplied engine and ingests its field as a child store.

**Tech Stack:** numpy (memmap), pyarrow, healpy, concurrent.futures threads; `mpi4py` optional.

**Spec:** design §6.8, §5 (`simulation/`), §8 items 1–2. Master plan Global Constraints (`simulation` imports only `_core`; no `data`/`combine`/`measure`/`twin`/`reference`).

## Interfaces consumed (R0)

`_core.errors`, `_core.fs` (`atomic_dir`, `atomic_write_json`, `hash_file`), `_core.geometry` (`Cube`, `Cone`, `SkyPatch`), `_core.healpix` (`radec_to_pix`, `cone_pixels`, `patch_pixels`), `_core.registry.Registry`.

## Interfaces produced

```
oneuniverse.simulation.specs
  CosmologySpec(h, omega_m, omega_b, sigma8, n_s, w0=-1.0, wa=0.0) .to_dict .from_dict      # the ONLY cosmology container in the package; used by simulation/reference/twin
  UnitFrameSpec(length="Mpc/h", velocity="km/s", mass="Msun/h", comoving=True)
  ProvenanceSpec(code: str, version: str, seed: int | None, config_hash: str, extra: dict = {})
  PRODUCT_KINDS = ("particles", "halos", "field", "lightcone", "checkpoint")
  ProductDecl(name: str, kind: str, redshifts: tuple[float, ...], columns: tuple[str, ...] = (), n_grid: int | None = None, extra: dict = {})
  RegionSpec(cube: Cube, buffer: float = 0.0, n_grid: int | None = None) .outer -> Cube (cube grown by buffer, may exceed [0, box))
  SimulationRequest(id: str, parent: str, region: RegionSpec, physics: Literal["dm", "hydro"] = "dm", ic_strategy: Literal["parent_field", "resample"] = "parent_field",
                    status: Literal["pending", "running", "ingested", "failed"] = "pending", child: str | None = None, error: str | None = None)
oneuniverse.simulation.manifest
  OUFSIM_FORMAT_VERSION = "1.0.0" ; MANIFEST_FILENAME = "manifest.json" ; LAYOUT_FILENAME = "layout.json"
  OUFSimManifest(format_version, name: str, box_size: float, cosmology: CosmologySpec, units: UnitFrameSpec, provenance: ProvenanceSpec, products: tuple[ProductDecl, ...],
                 native: dict (format, root, projection), created_utc: str, parent: str | None = None, lineage: dict = {})   # lineage: {"request_id", "region": RegionSpec dict} when a child
    .to_dict .from_dict ; .product(name) -> ProductDecl
  write_manifest(path, m) ; read_manifest(path) -> OUFSimManifest   # FormatError on major mismatch / missing key
  layout.json: {"products": {name: {ztag: {"dir": rel, "layout": "chunks"|"tiles"|"sky"|"reference", "n_chunks"|"n_tiles"|"n_pix": int}}}}  written by the store writers, read by SimStore
oneuniverse.simulation.execution
  ExecutionMode(str, Enum): STREAMED = "streamed"; MPI = "mpi"
  ExecutionPlan(mode: ExecutionMode = STREAMED, memory_budget_bytes: int = 2**30, mpi_comm=None, n_threads: int = 4) .batch_for(bytes_per_row, *, safety=0.5) -> int ; __post_init__: MPI requires mpi_comm
oneuniverse.simulation.store.index
  CATALOG_INDEX_SCHEMA = chunk_id i4, path string, row_start i8, row_stop i8, x_min..z_max f8 ; FIELD_INDEX_SCHEMA = tile_id i4, path string, i0 i4, i1 i4, j0, j1, k0, k1 ; SKY_INDEX_SCHEMA = pix i4, path string, n_rows i8
  chunk_ids(pos: ndarray (n,3), box_size, n_side) -> int32[] ; chunk_bbox(chunk_id, box_size, n_side) -> (lo, hi) ; split_periodic(cube: Cube, box_size) -> list[Cube]   # 1..8 cubes inside [0, box)
  overlapping_chunks(index: pa.Table, cube: Cube, box_size) -> int32[] (chunk ids) ; overlapping_tiles(index, cube, box_size, n_grid) -> int32[] ; cone_pixels for sky
oneuniverse.simulation.store.native
  class NativeReaderAdapter(ABC): format: ClassVar[str] ; read_rows(path, start, stop, *, columns=None) -> pa.Table ; read_field_region(path, lo: tuple[int,int,int], hi: tuple[int,int,int]) -> ndarray
  register_adapter(cls) ; get_adapter(format) ; NpyFieldAdapter ("npy") ; PackedNpyAdapter ("packed_npy": sorted particle slab + header.json with chunk ranges)
oneuniverse.simulation.store.write
  write_catalog_product(store_root, name, z, batches: Iterable[pa.RecordBatch], *, box_size, n_side, plan) -> dict   # streams: assigns chunk ids per batch, appends to per-chunk parquet writers (bounded open files = n_side³ ≤ 4096 ok; else two-pass by chunk ranges), writes _index.parquet + layout entry
  write_field_product(store_root, name, z, field: np.memmap | ndarray, *, tile_cells, plan) -> dict              # tiles written from slices of the memmap; never np.asarray(field) whole
  write_lightcone_product(store_root, name, batches, *, nside_part, plan) -> dict
  write_reference_catalog(store_root, name, z, native_path, adapter, *, box_size, n_side) -> dict ; write_reference_field(store_root, name, z, native_path, adapter, *, tile_cells) -> dict   # index only
  ingest_field(store_root, name, z, field, *, tile_cells=32, plan=None) -> None   # adds a field product (+ manifest ProductDecl + layout) to an existing store, atomically
oneuniverse.simulation.store.build
  @dataclass NativeProduct(decl: ProductDecl, kind_layout: Literal["catalog", "field", "lightcone"], sources: dict[float, Path], adapter_format: str | None = None, batches: Callable[[float], Iterable[pa.RecordBatch]] | None = None)
  build_store(out: Path, *, manifest: OUFSimManifest (without products), products: Sequence[NativeProduct], plan: ExecutionPlan, projection: Literal["reencode", "reference"] = "reencode", n_side=8, tile_cells=32) -> SimStore
oneuniverse.simulation.store.read
  class SimStore(root): .manifest .layout .products -> tuple[str, ...] ; .redshifts(product) -> tuple[float, ...] ; .index(product, z) -> pa.Table (cached)
    .read_box(product, z, cube, *, columns=None, plan=None) -> pa.Table        # periodic split; exact filter on positions; MPI: rank slice of chunks, no gather (documented)
    .iter_box(product, z, cube, *, columns=None, plan=None) -> Iterator[pa.Table]   # one chunk at a time under plan.batch_for
    .read_field_box(name, z, cube) -> ndarray (float32 sub-grid, periodic wrap handled by index arithmetic)
    .read_cone(product, cone: Cone, *, z_range=None, columns=None) -> pa.Table   # lightcone products only
    .scan_full(product, z, *, columns=None) -> Iterator[pa.Table]                 # explicit whole-product stream; docstring warning
    .last_read_stats -> dict {chunks_total, chunks_read, rows, bytes}
oneuniverse.simulation.converter
  class SimConverter(ABC): code: ClassVar[str] ; detect(path) -> bool ; manifest(path) -> OUFSimManifest ; products(path) -> list[NativeProduct] ; convert(path, out, *, plan, projection="reencode") -> SimStore (default impl = build_store)
  register(cls) ; get_converter(code) ; detect_converter(path) -> type[SimConverter] | None ; registered_codes()
oneuniverse.simulation.db
  class SimDatabase(root): .scan() -> self ; .stores() -> dict[str, OUFSimManifest] ; .store(name) -> SimStore ; .lineage() -> pa.Table (parent, child, request_id, created_utc)
    .request_region(parent, region: RegionSpec, *, physics="dm", ic_strategy="parent_field") -> SimulationRequest (persisted pending)
    .dispatch(request, engine: Callable[[SimulationRequest, SimStore], tuple[np.ndarray, float]], *, child_name=None, tile_cells=32) -> SimStore
        # engine returns (field, z); dispatch writes child store <parent>__<request.id> with manifest.parent + lineage, ingests the field at z, records lineage row, status "ingested"; on exception status "failed" + error, re-raise
    .requests() -> list[SimulationRequest] ; .save() ; .load()   # requests.json + lineage.parquet under <root>/_simdb/
oneuniverse.simulation (__init__) exports: CosmologySpec, UnitFrameSpec, ProvenanceSpec, ProductDecl, RegionSpec, SimulationRequest, OUFSimManifest, read_manifest, ExecutionMode, ExecutionPlan, SimConverter, register, get_converter, detect_converter, SimStore, build_store, NativeProduct, ingest_field, SimDatabase, Cube, Cone, SkyPatch (re-exported from _core.geometry)
test/fixtures/native_sim.py
  make_native_sim(out: Path, *, n_particles=50_000, n_grid=32, box_size=100.0, redshifts=(1.0, 0.0), seed=0) -> Path   # writes particles_z<tag>.npy (x,y,z,vx,vy,vz float32 structured), field_z<tag>.npy (n_grid³ float32), halos_z<tag>.parquet, lightcone.parquet (ra, dec, z), config.json ; random, no physics
  native_products(path) -> list[NativeProduct] ; native_manifest(path) -> OUFSimManifest
```

---

### Task 1: Delete the old simulation tree; specs + manifest + execution

**Files:**
- Delete: `oneuniverse/simulation/`, `test/test_sim*.py`, `test/test_oufsim*.py`, `test/test_lin*.py`, `test/test_pm*.py`, `test/test_resim*.py`, `test/test_execution*.py`, `test/test_simdatabase*.py`, `test/test_visual_*.py`, `scripts/{build_demo_oufsim,build_s17_demo,resim_feasibility}.py`; remove `pyyaml` from `dev` extras.
- Create: `oneuniverse/simulation/__init__.py`, `specs.py`, `manifest.py`, `execution.py`
- Test: `test/simulation/__init__.py`, `test/simulation/test_specs_manifest.py`

- [ ] **Step 1: `git rm -r` the paths; run `pytest -n 6`** — expected: only `test/core`, `test/data`, `test/combine`, `test/measure`, `test/test_guards.py` remain, all green. Commit — `refactor!: remove old simulation tree (rebuilt in R5/R6)`.

- [ ] **Step 2: Write the failing tests**

```python
def _manifest(**kw): return OUFSimManifest(OUFSIM_FORMAT_VERSION, "sim", 100.0, CosmologySpec(0.67, 0.31, 0.049, 0.81, 0.965), UnitFrameSpec(), ProvenanceSpec("fixture", "1", 0, "abc"), (ProductDecl("particles", "particles", (1.0, 0.0), ("x", "y", "z")),), {"format": "npy", "root": ".", "projection": "reencode"}, "2026-11-12T00:00:00Z", **kw)

def test_manifest_round_trip_and_version_gate(tmp_path):
    m = _manifest(); write_manifest(tmp_path / "manifest.json", m); assert read_manifest(tmp_path / "manifest.json") == m
    d = m.to_dict(); d["format_version"] = "0.1.0"; (tmp_path / "m.json").write_text(json.dumps(d))
    with pytest.raises(FormatError, match="0.1.0"): read_manifest(tmp_path / "m.json")
    with pytest.raises(ValueError): ProductDecl("x", "weird", (0.0,))

def test_region_outer_and_request_validation():
    r = RegionSpec(Cube((10, 10, 10), (20, 20, 20)), buffer=5.0); assert r.outer == Cube((5, 5, 5), (25, 25, 25))
    with pytest.raises(ValueError): SimulationRequest("r1", "sim", r, physics="mhd")

def test_execution_plan_batch_and_mpi_guard():
    assert ExecutionPlan(memory_budget_bytes=2**20).batch_for(100) == 2**20 // 100 // 2
    with pytest.raises(ValueError): ExecutionPlan(mode=ExecutionMode.MPI)
    assert not hasattr(ExecutionMode, "GPU")
```

- [ ] **Step 3–5:** Run (fail) → implement → run (pass) → commit `feat(simulation): typed specs, JSON manifest, execution plan`.

---

### Task 2: `store.index` and the native fixture

**Files:**
- Create: `oneuniverse/simulation/store/__init__.py`, `oneuniverse/simulation/store/index.py`, `test/fixtures/native_sim.py`
- Test: `test/simulation/test_index.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_chunk_ids_and_bbox_round_trip():
    pos = np.array([[0.1, 0.1, 0.1], [99.9, 50.0, 12.5]]); ids = chunk_ids(pos, 100.0, 4); lo, hi = chunk_bbox(ids[1], 100.0, 4)
    assert ids.dtype == np.int32 and lo[0] == 75.0 and hi[0] == 100.0 and lo[1] == 50.0 and lo[2] == 0.0

def test_split_periodic():
    assert split_periodic(Cube((10, 10, 10), (20, 20, 20)), 100.0) == [Cube((10, 10, 10), (20, 20, 20))]
    parts = split_periodic(Cube((95, 10, 10), (105, 20, 20)), 100.0); assert len(parts) == 2 and sum(p.volume for p in parts) == pytest.approx(1000.0) and all(0 <= p.lo[0] and p.hi[0] <= 100 for p in parts)
    assert len(split_periodic(Cube((-5, -5, -5), (5, 5, 5)), 100.0)) == 8

def test_overlapping_chunks_is_exact_superset():
    pos = np.random.default_rng(0).uniform(0, 100, (20_000, 3)); ids = chunk_ids(pos, 100.0, 4)
    idx = pa.table({"chunk_id": pa.array(np.arange(64), pa.int32()), "path": [""] * 64, "row_start": [0] * 64, "row_stop": [0] * 64, **{f"{a}_{m}": [chunk_bbox(i, 100.0, 4)[0 if m == "min" else 1][k] for i in range(64)] for k, a in enumerate("xyz") for m in ("min", "max")}})
    cube = Cube((30, 30, 30), (55, 55, 55)); hit = overlapping_chunks(idx, cube, 100.0); inside = cube.contains(*pos.T)
    assert np.isin(ids[inside], hit).all() and len(hit) == 8

def test_native_fixture_layout(tmp_path):
    p = make_native_sim(tmp_path / "sim"); assert (p / "particles_z1.000.npy").exists() and (p / "field_z0.000.npy").exists() and (p / "lightcone.parquet").exists()
    prods = native_products(p); assert {np.kind_layout for np in prods} == {"catalog", "field", "lightcone"} and native_manifest(p).box_size == 100.0
```

- [ ] **Step 2–5:** Run (fail) → implement (z tag `f"{z:.3f}"`; `split_periodic` by splitting each axis at 0 and `box_size` after taking the cube modulo the box) → run (pass) → commit `feat(simulation): chunk/tile index helpers and native fixture`.

---

### Task 3: `store.native`, `store.write`, `store.build`

**Files:**
- Create: `oneuniverse/simulation/store/native.py`, `write.py`, `build.py`
- Test: `test/simulation/test_write.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_store_reencode_layout(tmp_path):
    src = make_native_sim(tmp_path / "sim"); store = build_store(tmp_path / "store", manifest=native_manifest(src), products=native_products(src), plan=ExecutionPlan(), n_side=4, tile_cells=16)
    assert set(store.products) == {"particles", "field", "halos", "lightcone"} and store.redshifts("particles") == (1.0, 0.0)
    assert (tmp_path / "store/manifest.json").exists() and (tmp_path / "store/layout.json").exists() and store.index("particles", 0.0).num_rows == 64
    assert store.manifest.product("field").n_grid == 32 and store.layout["products"]["field"]["0.000"]["layout"] == "tiles"

def test_catalog_write_streams_under_budget(tmp_path):
    def gen():  # 20 batches, 100k rows each; never materialised together
        for i in range(20): rng = np.random.default_rng(i); yield pa.RecordBatch.from_pydict({"x": rng.uniform(0, 100, 100_000), "y": rng.uniform(0, 100, 100_000), "z": rng.uniform(0, 100, 100_000)})
    tracemalloc.start(); write_catalog_product(tmp_path / "s", "p", 0.0, gen(), box_size=100.0, n_side=4, plan=ExecutionPlan(memory_budget_bytes=256 * 2**20)); peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    idx = pq.read_table(tmp_path / "s/products/p/z=0.000/_index.parquet"); assert idx["row_stop"].to_numpy().sum() - idx["row_start"].to_numpy().sum() == 2_000_000 and peak < 300 * 2**20

def test_field_write_uses_memmap_slices(tmp_path, monkeypatch):
    f = np.lib.format.open_memmap(tmp_path / "f.npy", mode="w+", dtype=np.float32, shape=(64, 64, 64)); f[:] = 1.0; f.flush()
    mm = np.load(tmp_path / "f.npy", mmap_mode="r"); import numpy; spy = []
    monkeypatch.setattr(numpy, "asarray", lambda a, *k, **kw: (spy.append(getattr(a, "shape", None)) or np.array(a, *k, **kw)))
    write_field_product(tmp_path / "s", "field", 0.0, mm, tile_cells=16, plan=ExecutionPlan()); assert all(s != (64, 64, 64) for s in spy if s)

def test_reference_projection_is_index_only(tmp_path):
    src = make_native_sim(tmp_path / "sim"); packed = write_packed_native_fixture(src, n_side=4)   # fixture helper in native_sim.py producing the packed_npy layout
    store = build_store(tmp_path / "ref", manifest=native_manifest(src), products=native_products(packed, adapter_format="packed_npy"), plan=ExecutionPlan(), projection="reference", n_side=4)
    size_ref = sum(f.stat().st_size for f in (tmp_path / "ref").rglob("*") if f.is_file()); size_native = sum(f.stat().st_size for f in packed.rglob("*.npy"))
    assert size_ref < 0.2 * size_native and store.layout["products"]["particles"]["0.000"]["layout"] == "reference"

def test_ingest_field_adds_product(tmp_path):
    src = make_native_sim(tmp_path / "sim"); store = build_store(tmp_path / "s", manifest=native_manifest(src), products=native_products(src)[:1], plan=ExecutionPlan())
    ingest_field(tmp_path / "s", "delta_recon", 0.0, np.zeros((32, 32, 32), np.float32), tile_cells=16); store2 = SimStore(tmp_path / "s")
    assert "delta_recon" in store2.products and store2.manifest.product("delta_recon").kind == "field"
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement.** Catalog writer: one `pq.ParquetWriter` per chunk opened lazily (n_side ≤ 16 → ≤ 4096 writers; above that raise `ValueError`), batches split by `chunk_ids`, bbox accumulated with `np.minimum/maximum` per chunk, closed at the end; index written; layout entry merged into `layout.json` via `atomic_write_json`. Field writer: iterate tile index triples, slice the memmap, `np.save` each tile. Reference writers: index only (chunk ranges from the packed header; tile index from shape). `build_store` = `atomic_dir(out)` → writers → manifest with products → layout. `ingest_field` = write the tiles into a temp product dir, then rename, then rewrite manifest + layout.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(simulation): streaming store writers, adapters, build_store`

---

### Task 4: `store.read` (SimStore)

**Files:**
- Create: `oneuniverse/simulation/store/read.py`
- Test: `test/simulation/test_read.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.fixture(scope="module")
def store(tmp_path_factory):
    src = make_native_sim(tmp_path_factory.mktemp("n") / "sim", n_particles=200_000); return build_store(tmp_path_factory.mktemp("s") / "store", manifest=native_manifest(src), products=native_products(src), plan=ExecutionPlan(), n_side=4, tile_cells=16), src

def test_read_box_exact_and_pruned(store):
    st, src = store; cube = Cube((10, 10, 10), (30, 30, 30)); t = st.read_box("particles", 0.0, cube)
    raw = np.load(src / "particles_z0.000.npy"); inside = cube.contains(raw["x"], raw["y"], raw["z"])
    assert t.num_rows == inside.sum() and st.last_read_stats["chunks_read"] <= 8 and st.last_read_stats["chunks_total"] == 64

def test_read_box_periodic_wrap(store):
    st, src = store; cube = Cube((95, 95, 95), (105, 105, 105)); t = st.read_box("particles", 0.0, cube)
    parts = [st.read_box("particles", 0.0, c) for c in split_periodic(cube, 100.0)]
    assert t.num_rows == sum(p.num_rows for p in parts) and t.num_rows > 0 and len(parts) == 8

def test_iter_box_batches_and_columns(store):
    st, _ = store; it = st.iter_box("particles", 0.0, Cube((0, 0, 0), (100, 100, 100)), columns=["x"], plan=ExecutionPlan(memory_budget_bytes=2**16)); b = next(it)
    assert b.column_names == ["x"] and b.num_rows <= 2**16 // 8

def test_read_field_box_matches_native_with_wrap(store):
    # convention: cell i covers [i*cell, (i+1)*cell); read_field_box returns every cell intersecting the cube, in wrapped order
    st, src = store; f = np.load(src / "field_z0.000.npy"); cell = 100.0 / 32
    sub = st.read_field_box("field", 0.0, Cube((90.0, 0.0, 0.0), (110.0, 20.0, 20.0)))
    i_cells = np.arange(int(90.0 // cell), int(np.ceil(110.0 / cell))) % 32      # 28..31, 0..3
    jk = np.arange(0, int(np.ceil(20.0 / cell)))                                  # 0..6
    np.testing.assert_array_equal(sub, f[np.ix_(i_cells, jk, jk)])

def test_read_cone_lightcone(store):
    st, src = store; cone = Cone(45.0, 10.0, 5.0); t = st.read_cone("lightcone", cone); raw = pq.read_table(src / "lightcone.parquet")
    assert t.num_rows == cone.contains(raw["ra"].to_numpy(), raw["dec"].to_numpy()).sum()

def test_scan_full_is_explicit(store):
    st, _ = store; n = sum(b.num_rows for b in st.scan_full("particles", 0.0)); assert n == 200_000 and "whole" in SimStore.scan_full.__doc__.lower()

def test_reference_store_reads_identically(tmp_path, store):
    st, src = store; packed = write_packed_native_fixture(src, n_side=4)
    ref = build_store(tmp_path / "ref", manifest=native_manifest(src), products=native_products(packed, adapter_format="packed_npy"), plan=ExecutionPlan(), projection="reference", n_side=4)
    cube = Cube((10, 10, 10), (30, 30, 30)); a = st.read_box("particles", 0.0, cube).sort_by("x"); b = ref.read_box("particles", 0.0, cube).sort_by("x"); assert a.equals(b)
```

- [ ] **Step 2–5:** Run (fail) → implement (threaded chunk reads under `plan.n_threads`; MPI mode partitions chunk ids by rank and returns that rank's rows, documented) → run (pass) → commit `feat(simulation): SimStore partial-access reads with periodic wrap`.

---

### Task 5: `converter` and `db`

**Files:**
- Create: `oneuniverse/simulation/converter.py`, `oneuniverse/simulation/db.py`
- Modify: `test/fixtures/native_sim.py` (add `FixtureConverter(SimConverter)`, code `"fixture"`, registered inside the fixture module)
- Test: `test/simulation/test_converter_db.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_converter_registry_and_convert(tmp_path):
    src = make_native_sim(tmp_path / "sim"); assert detect_converter(src).code == "fixture" and "fixture" in registered_codes()
    st = get_converter("fixture")().convert(src, tmp_path / "store", plan=ExecutionPlan()); assert isinstance(st, SimStore) and st.manifest.provenance.code == "fixture"
    with pytest.raises(NotFoundError): get_converter("gadget")

def test_simdatabase_scan_request_dispatch(tmp_path):
    src = make_native_sim(tmp_path / "sim"); get_converter("fixture")().convert(src, tmp_path / "db/parent", plan=ExecutionPlan())
    db = SimDatabase(tmp_path / "db").scan(); assert list(db.stores()) == ["parent"]
    req = db.request_region("parent", RegionSpec(Cube((10, 10, 10), (40, 40, 40)), buffer=5.0)); assert req.status == "pending" and db.requests()[0].id == req.id
    def engine(request, store):  # a stand-in engine: mean field of the outer region at z=0
        sub = store.read_field_box("field", 0.0, request.region.outer); return sub * 0 + 1.0, 0.0
    child = db.dispatch(req, engine); assert child.manifest.parent == "parent" and child.manifest.lineage["request_id"] == req.id and "field" in child.products
    db2 = SimDatabase(tmp_path / "db").scan(); assert db2.requests()[0].status == "ingested" and db2.lineage().num_rows == 1 and db2.store(req.child).read_field_box("field", 0.0, Cube((0, 0, 0), (10, 10, 10))).mean() == 1.0

def test_dispatch_failure_recorded(tmp_path):
    ...; def bad(request, store): raise RuntimeError("boom")
    with pytest.raises(RuntimeError): db.dispatch(req, bad)
    assert db.requests()[0].status == "failed" and "boom" in db.requests()[0].error and db.lineage().num_rows == 0
```

- [ ] **Step 2–5:** Run (fail) → implement → run (pass) → commit `feat(simulation): converter registry and SimDatabase with on-disk lineage`.

---

### Task 6: Surface, guards, close-out

- [ ] **Step 1:** fill `simulation/__init__.py` exports; extend `test/test_guards.py` with the rule "`oneuniverse/simulation` imports nothing from `oneuniverse.reference`" (tree may not exist yet → skip).
- [ ] **Step 2:** `pytest -n 6`, single-process (< 60 s); guards green.
- [ ] **Step 3:** STATUS.md + memory; commit + push claude_config. Commit — `docs: R5 close-out`.

**Exit criterion:** reencode and reference stores read identically; periodic cube read equals the union of its wrapped parts; `SimDatabase.dispatch` produces a child store that `scan()` finds; no physics imported anywhere under `simulation/`.
