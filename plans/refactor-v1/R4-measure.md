# R4 — measure: Product, MeasurementSet, pipeline, presets, I/O

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `oneuniverse.measure` as the minimal honest handoff: one `Product` base with a fixed catalog contract and typed optional atoms, two subclasses (`FieldMap`, `Sightline`), a `MeasurementSet` validated at construction, one pipeline spine (`build_pointset`) with probe presets, and a generic, versioned, atomic save/load that drops nothing.

**Architecture:** Builders operate on `DatasetView` scans (pyarrow), weights come from `combine`, the window is a survey mask when given (data-inferred footprint only as a recorded fallback), regions are a HEALPix→region map shared across products, randoms are drawn from window × n(z). Catalog rows keep `source_row`, `partition`, `offset` so later attachments (photo-z PDFs) align by position.

**Tech Stack:** pyarrow, numpy, healpy, scipy (`kmeans2` for regions).

**Spec:** design §6.7, §9b (photo-z alignment, window). Master plan Global Constraints (no cosmology; `measure` imports nothing from `simulation`, `reference`, `twin`).

## Interfaces consumed (R1–R3)

`DatasetView` (`scan`, `table`, `take`, `load_pdf`, `manifest`), `ProbabilisticRedshift`, `Weight`, `ShearWeight`, `_core.healpix` (`radec_to_pix`, `coarsen`), `_core.fs` (`atomic_dir`, `atomic_write_json`), `_core.errors`.

## Interfaces produced

```
oneuniverse.measure.atoms
  @dataclass(frozen=True) Nz(edges: np.ndarray[float64, (k+1,)], counts: np.ndarray[float64, (k,) | (b, k)], method: Literal["spec_hist", "photoz_stack", "external"], bins: tuple[int, ...] | None = None)
      .pdf(z) -> ndarray ; .normalised() -> Nz ; __post_init__: counts >= 0, shapes consistent
  @dataclass(frozen=True) Window(nside: int, nest: bool, mask: np.ndarray[float32, (12 nside²,)] in [0,1], source: Literal["mask", "footprint"], systematics: dict[str, np.ndarray] = {})
      .area_sr -> float ; .fraction -> float ; .contains(ra, dec) -> bool[] (mask > 0)
      @classmethod from_mask(array, *, nside, nest=True) ; @classmethod footprint(ra, dec, *, nside) -> Window(source="footprint")
  @dataclass(frozen=True) Regions(nside: int, nest: bool, n_regions: int, pixel_region: np.ndarray[int32, (12 nside²,)] (-1 outside window), method: str)
      .assign(ra, dec) -> int32[] ; @classmethod from_window(window, n_regions, *, nside=None, seed=0) -> Regions   # kmeans2 on unit vectors of masked pixels, deterministic
oneuniverse.measure.product
  CATALOG_CORE = ("ra", "dec", "z", "weight", "region_id", "source_row", "partition", "offset")
  @dataclass(frozen=True) Product(catalog: pa.Table, randoms: pa.Table | None = None, window: Window | None = None, nz: Nz | None = None,
                                  extras: dict[str, np.ndarray | pa.Table] = {}, provenance: dict = {})
      __post_init__: catalog has CATALOG_CORE with float/int families; num_rows > 0 (ValidationError "empty"); randoms (if any) have ra, dec, z, weight, region_id; no cosmology column names (FORBIDDEN_COLUMNS = {"comoving_distance", "r_comoving", "dist_mpc", "dist_mpc_h", "chi"}); extras keys are identifiers; extras values are ndarray or pa.Table
      .n -> int ; .with_extra(name, value) -> Product ; .kind -> "pointset"
  @dataclass(frozen=True) FieldMap(Product): values: np.ndarray ; mask: np.ndarray ; nside: int ; nest: bool = True   # catalog may be a 0-row placeholder: FieldMap overrides the empty check; .kind = "fieldmap"
  @dataclass(frozen=True) Sightline(Product): # catalog: sightline_id i8, ra, dec, z, weight, region_id, wavelength list<f4>, delta list<f4>, pixel_weight list<f4>; .kind = "sightline" ; .n_pixels
oneuniverse.measure.measurement_set
  @dataclass(frozen=True) MeasurementSpec(name: str, tracers: tuple[str, ...], statistics: tuple[str, ...], pairs: tuple[tuple[str, str], ...] = (), covariance: Literal["jackknife", "mocks", "analytic", "external"] = "jackknife", notes: dict = {})
  @dataclass(frozen=True) MeasurementSet(products: dict[str, Product], spec: MeasurementSpec, regions: Regions)
      __post_init__: spec.tracers == tuple(products); every pair in spec.pairs names two tracers; every product's region_id < regions.n_regions; FieldMap exempt from region check
      .summary() -> dict (json-safe)
oneuniverse.measure.pipeline
  build_pointset(view: DatasetView, *, z: tuple[float, float], weight: Weight | None = None, nz_edges: Sequence[float], mask: Window | None = None,
                 regions: Regions | int = 20, randoms: Literal["generate", "none"] | DatasetView = "generate", random_factor: int = 10,
                 keep_columns: Sequence[str] = (), quality: tuple[str, str, float] | None = None, cone=None, patch=None, seed: int = 0) -> Product
      # steps: scan(z, cone, patch, columns=core + weight.columns + keep + quality col) -> drop rows with nan ra/dec/z -> quality mask -> weight -> window (mask or footprint(nside=256)) -> regions (from_window if int) -> region_id -> nz (spec_hist, weighted) -> randoms -> provenance {dataset, version, z, weight: describe(), window_source, randoms_source, nz_method, n, seed}
  randoms_from_view(view, *, z, window, regions, n_max=None) -> pa.Table ; generate_randoms(window, nz, regions, *, n, seed) -> pa.Table
  build_sightlines(view: DatasetView, *, z: tuple[float, float], regions: Regions | int = 20, mask: Window | None = None, cone=None, patch=None, seed=0) -> Sightline   # SIGHTLINE geometry only (FormatError otherwise); catalog = sightline_id, ra, dec, z, weight=1, region_id, source_row, partition, offset, wavelength, delta, pixel_weight (renamed from the OUF "weight" list column)
oneuniverse.measure.photoz
  attach_photoz(product: Product, view: DatasetView, *, n_draws: int = 1, seed: int = 0) -> Product   # uses view.take(partition, offset, columns=[pdf value column]) -> ProbabilisticRedshift ; extras["z_draws"] (n, n_draws), extras["z_pdf_mean"], ["z_pdf_std"]
  tomographic_nz(product: Product, *, bin_edges: Sequence[float], nz_edges: Sequence[float], by: Literal["z_pdf_mean", "z"] = "z_pdf_mean") -> tuple[Product, Nz]   # adds extras["tomo_bin"]; Nz.counts shape (n_bins, k), method "photoz_stack" from z_draws
oneuniverse.measure.presets
  galaxy_clustering(view, *, z, weight=None, nz_edges, mask=None, regions=20, randoms="generate", statistics=("pk_multipoles",), name="clustering", **kw) -> MeasurementSet
  cosmic_shear(view, *, z, shear_weight: ShearWeight, tomo_edges, nz_edges, mask=None, regions=20, n_draws=20, name="cosmic_shear", **kw) -> MeasurementSet   # extras e1, e2, tomo_bin, z_draws ; statistics ("xi_plus", "xi_minus")
  three_by_two(lens: MeasurementSet, source: MeasurementSet, *, name="3x2pt") -> MeasurementSet   # ValidationError if regions differ; pairs = (lens,lens), (lens,src), (src,src)
oneuniverse.measure.io
  MEASURE_FORMAT_VERSION = "1.0.0"
  save(ms: MeasurementSet, path: Path, *, overwrite=False) -> None ; load(path) -> MeasurementSet
  layout: path/measurement_set.json {format_version, spec, regions{nside,nest,n_regions,method}, products: {name: {kind, has_randoms, has_window, has_nz, extras: {key: "npy"|"parquet"}, provenance, fieldmap|sightline fields}}}
          path/regions.npy ; path/products/<name>/catalog.parquet, randoms.parquet, window.npz (mask + systematics), nz.npz, values.npy + mask.npy (FieldMap), extras/<key>.npy|.parquet
  # unknown extra type -> TypeError at save; format major mismatch -> FormatError at load; load reconstructs the exact subclass
oneuniverse.measure (__init__) exports: Nz, Window, Regions, Product, FieldMap, Sightline, MeasurementSpec, MeasurementSet, build_pointset, build_sightlines, attach_photoz, tomographic_nz, galaxy_clustering, cosmic_shear, three_by_two, save, load
test/fixtures/synthetic.py additions: synthetic_point_table(..., with_shear=False)  # adds e1, e2, R11, R22, R_S, shear_weight columns
```

---

### Task 1: `measure.atoms`

**Files:**
- Create: `oneuniverse/measure/__init__.py`, `oneuniverse/measure/atoms.py`
- Test: `test/measure/__init__.py`, `test/measure/test_atoms.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_nz_normalised_and_pdf():
    nz = Nz(np.array([0.0, 0.5, 1.0]), np.array([2.0, 6.0]), "spec_hist"); n = nz.normalised()
    assert np.isclose(np.trapezoid(n.pdf(np.linspace(0, 1, 1001)), np.linspace(0, 1, 1001)), 1.0, atol=1e-3) and n.pdf(np.array([0.25, 0.75]))[1] == 3 * n.pdf(np.array([0.25, 0.75]))[0]
    with pytest.raises(ValidationError): Nz(np.array([0.0, 1.0]), np.array([-1.0]), "spec_hist")

def test_window_from_mask_and_footprint():
    m = np.zeros(12 * 16**2, np.float32); m[:100] = 1.0; w = Window.from_mask(m, nside=16); assert w.fraction == pytest.approx(100 / len(m)) and w.source == "mask"
    with pytest.raises(ValidationError): Window.from_mask(m[:-1], nside=16)
    rng = np.random.default_rng(0); ra = rng.uniform(0, 10, 500); dec = rng.uniform(0, 10, 500); f = Window.footprint(ra, dec, nside=64)
    assert f.source == "footprint" and f.contains(ra, dec).all() and not f.contains(np.array([180.0]), np.array([0.0]))[0]

def test_regions_from_window_deterministic_and_balanced():
    m = np.zeros(12 * 32**2, np.float32); m[:3000] = 1.0; w = Window.from_mask(m, nside=32)
    r1 = Regions.from_window(w, 10, seed=1); r2 = Regions.from_window(w, 10, seed=1)
    assert np.array_equal(r1.pixel_region, r2.pixel_region) and r1.n_regions == 10 and (r1.pixel_region[m == 0] == -1).all()
    counts = np.bincount(r1.pixel_region[r1.pixel_region >= 0]); assert counts.min() > 0.5 * counts.mean()
    ra, dec = hp.pix2ang(32, np.arange(10), nest=True, lonlat=True); assert (r1.assign(ra, dec) >= 0).all()
```

- [ ] **Step 2–5:** Run (fail) → implement → run (pass) → commit `feat(measure): Nz, Window, Regions atoms`.

---

### Task 2: `measure.product` and `measure.measurement_set`

**Files:**
- Create: `oneuniverse/measure/product.py`, `oneuniverse/measure/measurement_set.py`
- Test: `test/measure/test_product.py`

- [ ] **Step 1: Write the failing tests**

```python
def _cat(n=10): return pa.table({"ra": np.zeros(n), "dec": np.zeros(n), "z": np.full(n, 0.5), "weight": np.ones(n), "region_id": np.zeros(n, np.int32), "source_row": np.arange(n), "partition": np.zeros(n, np.int32), "offset": np.arange(n)})

def test_product_validates_core_and_extras():
    p = Product(_cat()); assert p.n == 10 and p.kind == "pointset"
    with pytest.raises(ValidationError, match="region_id"): Product(_cat().drop(["region_id"]))
    with pytest.raises(TypeError): Product(_cat(), extras={"x": [1, 2]})
    assert p.with_extra("e1", np.zeros(10)).extras["e1"].shape == (10,)

def test_product_rejects_empty_catalog():
    with pytest.raises(ValidationError, match="empty"): Product(_cat(0))

def test_product_rejects_cosmology_columns():
    with pytest.raises(ValidationError, match="comoving_distance"): Product(_cat().append_column("comoving_distance", pa.array(np.zeros(10))))

def test_fieldmap_and_sightline_kinds():
    fm = FieldMap(_cat(0), values=np.zeros(12 * 4 * 4), mask=np.ones(12 * 4 * 4), nside=4); assert fm.kind == "fieldmap"
    sl_cat = _cat(3).append_column("sightline_id", pa.array([1, 2, 3])).append_column("wavelength", pa.array([[3600.0, 3601.0]] * 3, pa.list_(pa.float32()))).append_column("delta", pa.array([[0.1, -0.1]] * 3, pa.list_(pa.float32()))).append_column("pixel_weight", pa.array([[1.0, 1.0]] * 3, pa.list_(pa.float32())))
    sl = Sightline(sl_cat); assert sl.kind == "sightline" and sl.n_pixels == 6
    with pytest.raises(ValidationError, match="wavelength"): Sightline(_cat(3))

def test_measurement_set_invariants():
    reg = Regions(4, True, 2, np.zeros(12 * 16, np.int32), "test"); p = Product(_cat())
    ms = MeasurementSet({"gal": p}, MeasurementSpec("t", ("gal",), ("pk",)), reg); json.dumps(ms.summary())
    with pytest.raises(ValidationError, match="tracers"): MeasurementSet({"gal": p}, MeasurementSpec("t", ("other",), ("pk",)), reg)
    bad = Product(_cat().set_column(4, "region_id", pa.array(np.full(10, 7, np.int32))))
    with pytest.raises(ValidationError, match="region"): MeasurementSet({"gal": bad}, MeasurementSpec("t", ("gal",), ("pk",)), reg)
    with pytest.raises(ValidationError, match="pairs"): MeasurementSet({"gal": p}, MeasurementSpec("t", ("gal",), ("pk",), pairs=(("gal", "x"),)), reg)
```

- [ ] **Step 2–5:** Run (fail) → implement → run (pass) → commit `feat(measure): Product, FieldMap, Sightline, MeasurementSet with construction-time validation`.

---

### Task 3: `measure.pipeline` (build_pointset, randoms)

**Files:**
- Create: `oneuniverse/measure/pipeline.py`
- Modify: `test/fixtures/synthetic.py` (add `quality` int column and a `with_shear` option)
- Test: `test/measure/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.fixture(scope="module")
def view(tmp_path_factory):  # 50k synthetic rows in a 60x40 deg patch, z in (0.1, 1.5), with quality column
    p = tmp_path_factory.mktemp("m") / "ds"; write_synthetic_dataset(p, 50_000, 0, ra_range=(100, 160), dec_range=(-10, 30)); return DatasetView.from_path(p)

def test_build_pointset_footprint_fallback_recorded(view):
    prod = build_pointset(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), regions=12, seed=1)
    assert prod.window.source == "footprint" and prod.provenance["window_source"] == "footprint" and prod.nz.method == "spec_hist"
    assert prod.catalog.num_rows > 0 and pc.min(prod.catalog["z"]).as_py() >= 0.3 and (prod.catalog["region_id"].to_numpy() >= 0).all()
    assert prod.randoms.num_rows == 10 * prod.n and prod.provenance["randoms_source"] == "generated"

def test_build_pointset_uses_given_mask(view):
    m = np.zeros(12 * 64**2, np.float32); m[radec_to_pix(64, np.array([130.0]), np.array([10.0]))[0]] = 1.0   # single pixel mask
    prod = build_pointset(view, z=(0.1, 1.5), nz_edges=np.linspace(0.1, 1.5, 5), mask=Window.from_mask(m, nside=64), randoms="generate")
    assert prod.window.source == "mask" and prod.catalog.num_rows < 500 and Window.from_mask(m, nside=64).contains(prod.randoms["ra"].to_numpy(), prod.randoms["dec"].to_numpy()).all()

def test_randoms_follow_window_and_nz(view):
    prod = build_pointset(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), random_factor=20, seed=2)
    assert prod.window.contains(prod.randoms["ra"].to_numpy(), prod.randoms["dec"].to_numpy()).all()
    h_d, _ = np.histogram(prod.catalog["z"].to_numpy(), prod.nz.edges, weights=prod.catalog["weight"].to_numpy()); h_r, _ = np.histogram(prod.randoms["z"].to_numpy(), prod.nz.edges)
    assert np.allclose(h_r / h_r.sum(), h_d / h_d.sum(), atol=0.02) and ks_2samp(prod.catalog["z"].to_numpy(), prod.randoms["z"].to_numpy()).pvalue > 1e-3

def test_weight_and_quality_and_keep_columns(view):
    prod = build_pointset(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), weight=InverseVarianceWeight("z_err"), quality=("quality", ">=", 1), keep_columns=["mag"], randoms="none")
    assert prod.randoms is None and "mag" in prod.catalog.column_names and (prod.catalog["weight"].to_numpy() > 0).all() and prod.provenance["weight"].startswith("InverseVariance")

def test_empty_selection_raises(view):
    with pytest.raises(ValidationError, match="empty"): build_pointset(view, z=(5.0, 6.0), nz_edges=[5, 6])

def test_build_sightlines_from_sightline_view(tmp_path):
    # reuse the SIGHTLINE table of test/data/test_writer.py::test_sightline_geometry_round_trip via a fixture helper write_synthetic_sightlines(out, n, seed) added to test/fixtures/synthetic.py
    write_synthetic_sightlines(tmp_path / "sl", 300, 0); v = DatasetView.from_path(tmp_path / "sl")
    sl = build_sightlines(v, z=(2.0, 2.5), regions=5); assert isinstance(sl, Sightline) and sl.n_pixels == sl.n * 50 and (sl.catalog["region_id"].to_numpy() >= 0).all()
    with pytest.raises(FormatError): build_sightlines(view, z=(0.3, 1.0))   # POINT view

def test_randoms_from_view_ingest(view, tmp_path):
    prod = build_pointset(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), randoms=view)   # a view as randoms source (here the data itself)
    assert prod.provenance["randoms_source"] == "ingested" and prod.randoms.num_rows == prod.n and (prod.randoms["region_id"].to_numpy() >= 0).all()
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement.** `generate_randoms`: draw pixel indices with probability ∝ mask, uniform position inside the pixel at a finer NSIDE (`nside * 8`, `hp.pix2ang` of random sub-pixels), z from `nz.normalised()` by inverse CDF, `weight = 1`, `region_id = regions.assign`. `randoms_from_view`: scan with `z` + window containment + region assignment, `weight` column if present else 1.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(measure): build_pointset spine and randoms`

---

### Task 4: `measure.photoz` and `measure.presets`

**Files:**
- Create: `oneuniverse/measure/photoz.py`, `oneuniverse/measure/presets.py`
- Test: `test/measure/test_photoz.py`, `test/measure/test_presets.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_attach_photoz_aligns_rows_after_cuts(tmp_path):
    spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 31))); write_synthetic_dataset(tmp_path / "ds", 5_000, 0, with_pdf=spec); v = DatasetView.from_path(tmp_path / "ds")
    prod = build_pointset(v, z=(0.5, 1.5), nz_edges=np.linspace(0.5, 1.5, 6), quality=("quality", ">=", 1), randoms="none")   # cuts drop rows
    p2 = attach_photoz(prod, v, n_draws=3, seed=0)
    full = v.table(columns=["source_row", "z_pdf"]); pz_full = v.load_pdf(full); mean_by_row = dict(zip(full["source_row"].to_pylist(), pz_full.mean()))
    assert p2.extras["z_draws"].shape == (prod.n, 3) and np.allclose(p2.extras["z_pdf_mean"], [mean_by_row[r] for r in prod.catalog["source_row"].to_pylist()])

def test_tomographic_nz_shapes(tmp_path):
    ...; p3, nz = tomographic_nz(p2, bin_edges=[0.5, 1.0, 1.5], nz_edges=np.linspace(0, 3, 31))
    assert nz.counts.shape == (2, 30) and nz.method == "photoz_stack" and set(np.unique(p3.extras["tomo_bin"])) <= {-1, 0, 1}

def test_galaxy_clustering_preset(view):
    ms = galaxy_clustering(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8)); assert ms.spec.tracers == ("gal",) and ms.spec.statistics == ("pk_multipoles",) and ms.products["gal"].randoms is not None

def test_cosmic_shear_preset(tmp_path):
    spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 31))); write_synthetic_dataset(tmp_path / "s", 20_000, 0, with_pdf=spec, with_shear=True); v = DatasetView.from_path(tmp_path / "s")
    ms = cosmic_shear(v, z=(0.2, 2.0), shear_weight=ShearWeight("metacal"), tomo_edges=[0.2, 0.8, 2.0], nz_edges=np.linspace(0, 3, 31), n_draws=5)
    src = ms.products["src"]; assert {"e1", "e2", "tomo_bin", "z_draws"} <= set(src.extras) and src.nz.counts.shape[0] == 2 and ms.spec.statistics == ("xi_plus", "xi_minus")

def test_three_by_two_requires_shared_regions(view, tmp_path):
    lens = galaxy_clustering(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), regions=10); src = galaxy_clustering(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), regions=10, name="src")
    with pytest.raises(ValidationError, match="regions"): three_by_two(lens, galaxy_clustering(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), regions=12))
    src2 = MeasurementSet({"src": src.products["gal"]}, MeasurementSpec("src", ("src",), ("xi_plus",)), lens.regions)
    ms = three_by_two(lens, src2); assert ms.spec.pairs == (("gal", "gal"), ("gal", "src"), ("src", "src")) and set(ms.products) == {"gal", "src"}
```

- [ ] **Step 2–5:** Run (fail) → implement (`tomographic_nz` stacks `z_draws` histograms per bin; `cosmic_shear` = `build_pointset(..., weight=shear_weight, keep_columns=shear columns, randoms="none")` → extras `e1`, `e2` → `attach_photoz` → `tomographic_nz`) → run (pass) → commit `feat(measure): photo-z attachment, tomographic n(z), probe presets`.

---

### Task 5: `measure.io`

**Files:**
- Create: `oneuniverse/measure/io.py`
- Test: `test/measure/test_io.py`

- [ ] **Step 1: Write the failing tests**

```python
def _ms_all_kinds(view):  # clustering product + FieldMap + Sightline, with extras of both ndarray and pa.Table types, window systematics, nz
    ...

def test_round_trip_drops_nothing(view, tmp_path):
    ms = _ms_all_kinds(view); save(ms, tmp_path / "ms"); back = load(tmp_path / "ms")
    assert set(back.products) == set(ms.products) and back.spec == ms.spec and np.array_equal(back.regions.pixel_region, ms.regions.pixel_region)
    for name, p in ms.products.items():
        q = back.products[name]; assert type(q) is type(p) and q.catalog.equals(p.catalog) and (q.randoms is None) == (p.randoms is None) and set(q.extras) == set(p.extras)
        for k, v in p.extras.items(): assert (v.equals(q.extras[k]) if isinstance(v, pa.Table) else np.array_equal(v, q.extras[k]))
        if p.window: assert np.array_equal(q.window.mask, p.window.mask) and set(q.window.systematics) == set(p.window.systematics)
        if p.nz: assert np.array_equal(q.nz.counts, p.nz.counts) and q.nz.method == p.nz.method
        assert q.provenance == p.provenance
    fm = back.products["kappa"]; assert isinstance(fm, FieldMap) and fm.nside == ms.products["kappa"].nside

def test_save_refuses_unknown_extra_type(view, tmp_path):
    ms = _ms_all_kinds(view); bad = dataclasses.replace(ms.products["gal"], extras={"x": {"a": 1}})
    with pytest.raises(TypeError): Product(bad.catalog, extras={"x": {"a": 1}})   # already refused at construction
def test_save_is_atomic_and_versioned(view, tmp_path):
    ms = _ms_all_kinds(view); save(ms, tmp_path / "ms"); d = json.loads((tmp_path / "ms/measurement_set.json").read_text()); assert d["format_version"] == "1.0.0"
    with pytest.raises(FormatError): save(ms, tmp_path / "ms")
    d["format_version"] = "2.0.0"; (tmp_path / "ms/measurement_set.json").write_text(json.dumps(d))
    with pytest.raises(FormatError): load(tmp_path / "ms")
```

- [ ] **Step 2–5:** Run (fail) → implement → run (pass) → commit `feat(measure): versioned atomic save/load`.

---

### Task 6: Surface and close-out

- [ ] **Step 1:** fill `measure/__init__.py` exports; `test/measure/test_public_api.py` asserts the 17 names.
- [ ] **Step 2:** `pytest -n 6`, single-process (< 60 s), guards green (`measure` imports no `simulation`/`reference`/`twin`; no cosmology identifiers).
- [ ] **Step 3:** STATUS.md + memory; commit + push claude_config. Commit — `docs: R4 close-out`.

**Exit criterion:** clustering and cosmic-shear presets build on synthetic OUF; photo-z alignment test green; save/load round-trip equality on all three product kinds.
