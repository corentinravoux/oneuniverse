# R1 — Data format: schema, manifest, writer, view, PDFs, loaders

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old `oneuniverse.data` with OUF 3.0: one schema module, a small manifest plus a columnar partition index, one streaming writer, one pyarrow-backed view with vectorised pruning, trimmed photo-z PDFs, and declarative batch-streaming survey loaders for eBOSS DR16Q and DESI DR1 QSO.

**Architecture:** Loaders yield `pa.RecordBatch`es; the writer stages them, chooses the partition NSIDE from the row count, writes HEALPix-partitioned parquet with `pyarrow.dataset.write_dataset`, derives per-file statistics from parquet metadata into `_index.parquet`, writes `manifest.json` last inside an `atomic_dir`. `DatasetView` opens the hive-partitioned dataset once and prunes files by `cell.isin(...)` computed from the index with numpy.

**Tech Stack:** pyarrow (`dataset`, `parquet`, `compute`), numpy, healpy, astropy.io.fits (FITS reading, memmap, row chunks), pandas at the `read()` boundary.

**Spec:** design §5 (`data/`), §6.1, §6.2, §6.5, §6.6, §9b. Master plan Global Constraints apply.

## Interfaces consumed (from R0)

`_core.errors`, `_core.healpix` (`radec_to_pix`, `coarsen`, `cone_pixels`, `patch_pixels`, `NSIDE_FINE`), `_core.geometry` (`Cone`, `SkyPatch`, `ZRange`), `_core.fs` (`atomic_dir`, `atomic_write_json`, `hash_file`), `_core.registry.Registry`.

## Interfaces produced

```
oneuniverse.data.schema
  class Geometry(str, Enum): POINT = "point"; SIGHTLINE = "sightline"; LIGHTCURVE = "lightcurve"
  Z_TYPES: frozenset[str] = {"spec", "phot", "phot_pdf", "pv", "none"} ; register_z_type(name: str) -> None ; z_types() -> frozenset[str]
  CORE_POINT: pa.Schema  = ra f8, dec f8, z f8, z_type dictionary<int8,string>, z_err f4, object_id i8, survey dictionary<int16,string>, source_row i8, hpix32 i4
  CORE_SIGHTLINE: pa.Schema = sightline_id i8, ra f8, dec f8, z f8, survey dict, source_row i8, hpix32 i4, wavelength list<f4>, delta list<f4>, weight list<f4>
  CORE_LIGHTCURVE: pa.Schema = CORE_POINT + mjd list<f8>, flux list<f4>, flux_err list<f4>, band list<dictionary<int8,string>>
  OPTIONAL_COLUMNS: dict[str, pa.Schema]   # groups "spectroscopic", "photometric", "peculiar_velocity", "snia", "shear", "pdf": name -> type used only to type-check when present
  core_schema(geometry) -> pa.Schema ; validate_schema(schema: pa.Schema, geometry: Geometry) -> None   # SchemaError: missing core column, wrong type family, z_type dictionary values outside z_types()
  stable_object_id(survey: str, source_row: np.ndarray[int64]) -> np.ndarray[int64]   # (int(sha256(survey)[:8],16) << 32) | source_row ; raises ValueError if source_row >= 2**32
  cast_to_core(table: pa.Table, geometry) -> pa.Table   # casts core columns to the core types (dictionary-encodes z_type/survey), leaves others

oneuniverse.data.manifest
  FORMAT_VERSION = "3.0.0" ; MANIFEST_FILENAME = "manifest.json" ; INDEX_FILENAME = "_index.parquet" ; DATA_DIR = "data"
  @dataclass(frozen=True, slots=True) SurveyMeta(name, kind, description="", reference="", url="")
  LoaderInfo(name, version, kwargs: dict) ; OriginalFile(path, sha256, n_rows: int | None) ; Validity(valid_from: str, valid_to: str | None, version: int) ; Partitioning(nside: int, nest: bool = True)
  Manifest(format_version, geometry: Geometry, survey: SurveyMeta, loader: LoaderInfo, created_utc: str, schema: pa.Schema, partitioning: Partitioning,
           original_files: tuple[OriginalFile, ...], n_rows: int, validity: Validity, extensions: dict[str, dict])
    .to_dict() -> dict (schema as {"ipc_base64": ..., "columns": [{"name","type"}]}) ; Manifest.from_dict(d) ; .extension(name) -> dict | None
  write_manifest(path: Path, m: Manifest) -> None (atomic) ; read_manifest(path) -> Manifest  # FormatError if format_version major != 3 or any required key missing
  EXTENSION_VERSIONS = {"coordinate": 1, "spectrum": 1, "temporal": 1, "pdf": 1}  # each extension dict carries "version"
  INDEX_SCHEMA: path string, cell int32, n_rows int64, ra_min/ra_max/dec_min/dec_max/z_min/z_max float64, then optional <col>_min/<col>_max float64 for extra columns
  write_partition_index(path, table: pa.Table) ; read_partition_index(path) -> pa.Table

oneuniverse.data.writer
  MIN_ROWS_PER_PARTITION = 5_000 ; MAX_ROWS_PER_FILE = 2_000_000 ; auto_partition_nside(n_rows: int) -> int  # finest power of 2 <= 32 with n_rows / (12 nside²) >= MIN_ROWS_PER_PARTITION, min 1
  write_dataset(batches: Iterable[pa.RecordBatch] | pa.Table, out: Path, *, geometry: Geometry, survey: SurveyMeta, loader: LoaderInfo,
                original_files: Sequence[Path | OriginalFile] = (), partition_nside: int | None = None, extra_stats_columns: Sequence[str] = (),
                extensions: dict[str, dict] | None = None, overwrite: bool = False) -> Manifest
  # out exists and not overwrite -> FormatError; hpix32 computed from ra/dec when absent; z_type validated per batch; schema validated on the first batch

oneuniverse.data.view
  class DatasetView: from_path(path) ; .path .manifest .index (pa.Table) .geometry .n_rows .n_partitions .columns .schema .nside
    .cells_for(*, cone=None, patch=None, z=None, t=None, ranges: dict[str, tuple[float, float]] | None = None) -> np.ndarray[int32]   # vectorised over .index; RA-wrap aware
    .scan(*, columns=None, cone=None, patch=None, z=None, t=None, ranges=None, filter: pc.Expression | None = None, batch_size=65_536) -> Iterator[pa.RecordBatch]
    .table(**same) -> pa.Table ; .read(**same) -> pd.DataFrame
    .take(partition: np.ndarray[int32], offset: np.ndarray[int64], *, columns=None) -> pa.Table   # rows in the order given; partition = row id in .index
    .iter_partitions(columns=None) -> Iterator[tuple[int, pa.Table]]   # (partition id, whole file) — the builder for offsets
    .load_pdf(rows: pa.Table | None = None) -> ProbabilisticRedshift   # FormatError if no "pdf" extension
  cone_expr(cone) -> pc.Expression ; patch_expr(patch) -> pc.Expression ; range_expr(column, lo, hi) -> pc.Expression   # module-level, reused by R2/R4

oneuniverse.data.pdf
  PdfSpec(kind: Literal["interp", "sample", "quant"], grid: tuple[float, ...] | None = None, quant_levels: tuple[float, ...] | None = None, value_column: str = "z_pdf")
    .to_dict() .from_dict() ; .n_components
  ProbabilisticRedshift(spec, values: np.ndarray[float32, (n, n_components)] | list-of-arrays for "sample")
    .mean() .std() .sample(n_per: int, seed: int) -> (n, n_per) ; .cdf(z: float) -> (n,) [interp only] ; .ppf(q: float) -> (n,) [interp, quant]  # NotImplementedError otherwise, never a crash
    from_table(table: pa.Table, spec) -> ProbabilisticRedshift

oneuniverse.data.loaders.base
  class SurveyLoader(ABC): meta: ClassVar[SurveyMeta]; geometry: ClassVar[Geometry] = Geometry.POINT; z_type: ClassVar[str] = "spec"; version: ClassVar[str] = "1"
                           extensions: ClassVar[dict[str, dict]] = {}
    @abstractmethod batches(self, source: Path, *, batch_size: int = 250_000, **kwargs) -> Iterator[pa.RecordBatch]   # yields core-complete batches
    original_files(self, source: Path) -> list[Path]
  class FitsCatalogLoader(SurveyLoader): filename: ClassVar[str]; hdu: ClassVar[int] = 1; columns: ClassVar[dict[str, str]]  # native -> standard
    def row_filter(self, native: pa.Table, **options) -> pa.BooleanArray | None: return None   # receives the native-named batch; options = loader kwargs minus z_min/z_max/batch_size
    z_err_column: ClassVar[str | None] = None   # native name; None -> z_err = NaN
    def derive(self, table: pa.Table) -> pa.Table: return table   # after renaming, before core stamping
    batches(...) implemented: reads hdu rows [start, stop) with astropy memmap, converts the selected native columns to arrow (byteswap to native, strip fixed-width strings, 2-D columns -> fixed_size_list), applies row_filter, sets source_row = start + flatnonzero(mask), renames, derive(), stamps survey/z_type/z_err/object_id/hpix32, cast_to_core
  read_fits_batches(path, hdu, columns, batch_size) -> Iterator[tuple[int, pa.Table]]   # (start_row, native table) — module-level helper
oneuniverse.data.loaders.registry
  register(cls) -> cls ; get_loader(name) -> SurveyLoader ; list_surveys() -> dict[str, SurveyMeta] ; load_entry_points() (group "oneuniverse.survey_loaders", called once at import of oneuniverse.data.loaders)
oneuniverse.data.loaders.eboss_qso: class EbossQso(FitsCatalogLoader)  meta.name = "eboss_qso", kind = "spectroscopic"
oneuniverse.data.loaders.desi_qso:  class DesiQso(FitsCatalogLoader)   meta.name = "desi_qso",  kind = "spectroscopic"
oneuniverse.data.convert
  convert(survey: str | SurveyLoader, *, source: Path, out: Path, overwrite=False, partition_nside=None, batch_size=250_000, **loader_kwargs) -> Manifest

oneuniverse.data (package __init__)  exports: Geometry, SurveyMeta, Manifest, read_manifest, DatasetView, write_dataset, convert, list_surveys, get_loader, register, SurveyLoader, FitsCatalogLoader,
                                              PdfSpec, ProbabilisticRedshift, Cone, SkyPatch, ZRange (re-exported from _core.geometry)
test/fixtures/synthetic.py
  synthetic_point_table(n: int, seed: int, *, survey="synthetic", ra_range=(0, 360), dec_range=(-90, 90), z_range=(0.01, 2.0), with_pdf: PdfSpec | None = None) -> pa.Table
     # core columns + "mag" f4 + "z_err" realistic (1e-4..1e-3) + optional pdf column; source_row = arange(n); shuffled row order (so file order != source_row order)
  write_synthetic_dataset(out: Path, n: int, seed: int, *, partition_nside=None, **kw) -> Manifest
  write_synthetic_sightlines(out: Path, n: int, seed: int, *, n_pixels=50) -> Manifest   # SIGHTLINE geometry: sightline_id, ra, dec, z in (2, 3), survey "lya", wavelength/delta/weight list<f4>
test/fixtures/fits_like.py
  write_fake_desi_dr1_fits(out_dir, n_rows=5000, seed=0) -> Path   (port of main:test/fixtures/desi_dr1_like.py, same columns)
  write_fake_eboss_dr16q_fits(out_dir, n_rows=5000, seed=0) -> Path  (columns from main:oneuniverse/data/surveys/spectroscopic/eboss_qso/loader.py `_FITS_COLUMNS`, incl. 2-D Z_DLA/NHI_DLA/CONF_DLA (n,5) with -1 sentinels, IS_QSO_FINAL in {1,0,-2,2}, ZWARNING mostly 0)
```

---

### Task 1: Delete the old Pillar-1 tree and its tests

**Files:**
- Delete: `oneuniverse/data/`, `oneuniverse/combine/`, `oneuniverse/measure/`, `oneuniverse/twin/`, `test/fixtures/{desi_dr1_like,healpix_maps,measure_ouf,pdf_catalog,tracer_sim}.py`, every `test/test_*.py` that imports any of those four packages (list them with `grep -l "oneuniverse\.\(data\|combine\|measure\|twin\)" test/test_*.py`), `scripts/{onboard_desi_dr1_qso,build_measure*,export_to_sql,twin_coupling_demo,mock_challenge_demo}.py`, `notebooks/`.
- Modify: `oneuniverse/__init__.py` → keep only `__version__` and a docstring; `test/test_guards.py` → remove the R0 allow-list entry.

- [ ] **Step 1: Delete with `git rm -r`** the paths above (keep `oneuniverse/simulation/`, `oneuniverse/_core/`, `test/core/`, `test/test_guards.py`, `test/test_sim*.py`, `test/test_oufsim*.py`, `test/test_lin*.py`, `test/test_pm*.py`, `test/test_resim*.py`, `test/test_execution*`, `test/test_registry_util.py`).
- [ ] **Step 2: Run the suite** — Run: `pytest -n 6`. Expected: only simulation + core + guard tests remain; all green. Record the count in STATUS.md.
- [ ] **Step 3: Commit** — `refactor!: remove old data/combine/measure/twin (rebuilt as OUF 3.0 in R1–R4)`

---

### Task 2: `data.schema`

**Files:**
- Create: `oneuniverse/data/__init__.py` (docstring + exports filled progressively), `oneuniverse/data/schema.py`
- Test: `test/data/__init__.py`, `test/data/test_schema.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_core_point_schema_types():
    s = core_schema(Geometry.POINT)
    assert s.field("ra").type == pa.float64() and s.field("hpix32").type == pa.int32()
    assert pa.types.is_dictionary(s.field("z_type").type) and s.field("z_type").type.index_type == pa.int8()

def test_validate_schema_missing_and_wrong_type():
    good = core_schema(Geometry.POINT); validate_schema(good, Geometry.POINT)
    with pytest.raises(SchemaError, match="missing.*dec"): validate_schema(good.remove(good.get_field_index("dec")), Geometry.POINT)
    bad = good.set(good.get_field_index("ra"), pa.field("ra", pa.string()))
    with pytest.raises(SchemaError, match="ra"): validate_schema(bad, Geometry.POINT)

def test_validate_schema_accepts_extra_and_optional_group_types():
    s = core_schema(Geometry.POINT).append(pa.field("mag", pa.float32())).append(pa.field("e1", pa.float32()))
    validate_schema(s, Geometry.POINT)
    with pytest.raises(SchemaError, match="e1"): validate_schema(s.set(s.get_field_index("e1"), pa.field("e1", pa.string())), Geometry.POINT)

def test_register_z_type_and_reject_unknown():
    register_z_type("grism"); assert "grism" in z_types()
    with pytest.raises(ValueError): register_z_type("spec")

def test_stable_object_id_is_deterministic_and_disjoint():
    a = stable_object_id("eboss_qso", np.arange(3)); b = stable_object_id("desi_qso", np.arange(3))
    assert a.dtype == np.int64 and np.array_equal(a, stable_object_id("eboss_qso", np.arange(3))) and not np.intersect1d(a, b).size
    with pytest.raises(ValueError): stable_object_id("x", np.array([2**32]))

def test_cast_to_core_dictionary_encodes():
    t = pa.table({"ra": [1.0], "dec": [2.0], "z": [0.5], "z_type": ["spec"], "z_err": [0.1], "object_id": [1], "survey": ["s"], "source_row": [0], "hpix32": [3], "mag": [20.0]})
    out = cast_to_core(t, Geometry.POINT)
    assert pa.types.is_dictionary(out.schema.field("survey").type) and out.schema.field("z_err").type == pa.float32() and out.schema.field("mag").type == pa.float64()
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement `schema.py`** per the interface block. Type-family check: `pa.types.is_floating` for `ra/dec/z/z_err/mag-like`, `is_integer` for ids, `is_dictionary or is_string` for `z_type/survey`, `is_list/is_fixed_size_list` for list columns. `OPTIONAL_COLUMNS` content: port the column names and types from `main:oneuniverse/data/schema.py` groups SPECTROSCOPIC/PHOTOMETRIC/PECULIAR_VELOCITY/SNIA/SHEAR/PROBABILISTIC_REDSHIFT (names only, arrow types; drop `z_pdf_kind`, `composite_id`).
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(data): OUF 3.0 schema module`

---

### Task 3: `data.manifest`

**Files:**
- Create: `oneuniverse/data/manifest.py`
- Test: `test/data/test_manifest.py`

- [ ] **Step 1: Write the failing tests**

```python
def _manifest(**kw) -> Manifest: ...  # helper building a valid Manifest with core_schema(POINT), Partitioning(32), Validity("2026-11-12T00:00:00Z", None, 1), extensions={"coordinate": {"version": 1, "frame": "icrs"}}

def test_manifest_round_trip_preserves_arrow_schema(tmp_path):
    m = _manifest(); write_manifest(tmp_path / "manifest.json", m); back = read_manifest(tmp_path / "manifest.json")
    assert back.schema.equals(m.schema) and back == m

def test_manifest_json_is_human_readable(tmp_path):
    write_manifest(tmp_path / "m.json", _manifest()); d = json.loads((tmp_path / "m.json").read_text())
    assert d["format_version"] == "3.0.0" and d["schema"]["columns"][0] == {"name": "ra", "type": "double"} and "ipc_base64" in d["schema"]

def test_read_manifest_rejects_other_major(tmp_path):
    d = _manifest().to_dict(); d["format_version"] = "2.6.0"; (tmp_path / "m.json").write_text(json.dumps(d))
    with pytest.raises(FormatError, match="2.6.0"): read_manifest(tmp_path / "m.json")

def test_read_manifest_rejects_missing_key(tmp_path):
    d = _manifest().to_dict(); del d["partitioning"]; (tmp_path / "m.json").write_text(json.dumps(d))
    with pytest.raises(FormatError, match="partitioning"): read_manifest(tmp_path / "m.json")

def test_extension_version_required():
    with pytest.raises(FormatError, match="version"): _manifest(extensions={"coordinate": {"frame": "icrs"}})

def test_partition_index_round_trip(tmp_path):
    t = pa.table({"path": ["data/cell=1/part-0.parquet"], "cell": pa.array([1], pa.int32()), "n_rows": [10], "ra_min": [0.0], "ra_max": [1.0], "dec_min": [0.0], "dec_max": [1.0], "z_min": [0.1], "z_max": [0.2]})
    write_partition_index(tmp_path / "_index.parquet", t); assert read_partition_index(tmp_path / "_index.parquet").equals(t)
    with pytest.raises(FormatError): write_partition_index(tmp_path / "x.parquet", t.drop(["cell"]))
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement.** Schema serialisation: `schema.serialize().to_pybytes()` base64 for `ipc_base64`, `pa.ipc.read_schema(pa.py_buffer(...))` back; `columns` list from `str(field.type)`. `Manifest.__post_init__` validates every extension dict has an int `version` equal to `EXTENSION_VERSIONS[name]` when the name is known (unknown names allowed, must still carry `version`). Dataclass `eq` must compare schemas via `.equals` (custom `__eq__`).
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(data): OUF 3.0 manifest and partition index`

---

### Task 4: `data.writer` + synthetic fixture

**Files:**
- Create: `oneuniverse/data/writer.py`, `test/fixtures/__init__.py`, `test/fixtures/synthetic.py`
- Test: `test/data/test_writer.py`

- [ ] **Step 1: Write the fixtures** `synthetic_point_table` / `write_synthetic_dataset` / `write_synthetic_sightlines` per the interface block (row order shuffled with the seed; `hpix32` NOT included so the writer computes it; `source_row` = original index before shuffling).

- [ ] **Step 2: Write the failing tests**

```python
def test_layout_and_manifest(tmp_path):
    m = write_synthetic_dataset(tmp_path / "ds", 20_000, 0)
    assert (tmp_path / "ds/manifest.json").exists() and (tmp_path / "ds/_index.parquet").exists()
    files = sorted((tmp_path / "ds/data").rglob("*.parquet")); assert files and all(f.parent.name.startswith("cell=") for f in files)
    assert m.n_rows == 20_000 and m.partitioning.nside == auto_partition_nside(20_000) and m.format_version == "3.0.0"

def test_auto_partition_nside_values():
    assert auto_partition_nside(10) == 1 and auto_partition_nside(5_000 * 12 * 64) == 8 and auto_partition_nside(10**9) == 32

def test_index_stats_match_data(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 20_000, 1, partition_nside=4)
    idx = read_partition_index(tmp_path / "ds/_index.parquet")
    for row in idx.to_pylist():
        t = pq.read_table(tmp_path / "ds" / row["path"], columns=["ra", "dec", "z", "hpix32"])
        assert t.num_rows == row["n_rows"] and pc.min(t["z"]).as_py() == pytest.approx(row["z_min"]) and pc.max(t["ra"]).as_py() == pytest.approx(row["ra_max"])
        assert set(coarsen(t["hpix32"].to_numpy(), 32, 4)) == {row["cell"]}

def test_extra_stats_columns(tmp_path):
    tab = synthetic_point_table(5_000, 2); m = write_dataset(tab, tmp_path / "ds", geometry=Geometry.POINT, survey=SurveyMeta("s", "test"), loader=LoaderInfo("test", "1", {}), extra_stats_columns=["mag"])
    idx = read_partition_index(tmp_path / "ds/_index.parquet"); assert "mag_min" in idx.column_names and "mag_max" in idx.column_names

def test_streaming_batches_bounded_memory(tmp_path):
    def gen():  # 40 batches of 50k rows, never materialised together
        for i in range(40): yield synthetic_point_table(50_000, i).to_batches()[0]
    tracemalloc.start(); write_dataset(gen(), tmp_path / "ds", geometry=Geometry.POINT, survey=SurveyMeta("s", "test"), loader=LoaderInfo("t", "1", {})); peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    assert read_manifest(tmp_path / "ds/manifest.json").n_rows == 2_000_000 and peak < 400 * 2**20

def test_overwrite_is_atomic_and_previous_untouched_on_failure(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 1_000, 0); before = read_manifest(tmp_path / "ds/manifest.json")
    with pytest.raises(FormatError): write_dataset(synthetic_point_table(10, 0), tmp_path / "ds", geometry=Geometry.POINT, survey=SurveyMeta("s", "test"), loader=LoaderInfo("t", "1", {}))  # no overwrite
    def bad(): yield synthetic_point_table(10, 0).to_batches()[0]; raise RuntimeError("loader died")
    with pytest.raises(RuntimeError): write_dataset(bad(), tmp_path / "ds", geometry=Geometry.POINT, survey=SurveyMeta("s", "test"), loader=LoaderInfo("t", "1", {}), overwrite=True)
    assert read_manifest(tmp_path / "ds/manifest.json") == before and not (tmp_path / "ds.tmp").exists()

def test_rejects_unknown_z_type_and_missing_core(tmp_path):
    t = synthetic_point_table(100, 0); bad = t.set_column(t.schema.get_field_index("z_type"), "z_type", pa.array(["weird"] * 100))
    with pytest.raises(SchemaError, match="weird"): write_dataset(bad, tmp_path / "a", geometry=Geometry.POINT, survey=SurveyMeta("s", "test"), loader=LoaderInfo("t", "1", {}))
    with pytest.raises(SchemaError, match="dec"): write_dataset(t.drop(["dec"]), tmp_path / "b", geometry=Geometry.POINT, survey=SurveyMeta("s", "test"), loader=LoaderInfo("t", "1", {}))

def test_sightline_geometry_round_trip(tmp_path):
    n = 300; rng = np.random.default_rng(0); lam = [np.linspace(3600, 5000, 50).astype(np.float32)] * n
    t = pa.table({"sightline_id": np.arange(n), "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-30, 30, n), "z": rng.uniform(2, 3, n), "survey": ["lya"] * n, "source_row": np.arange(n),
                  "wavelength": pa.array(lam, pa.list_(pa.float32())), "delta": pa.array([rng.normal(0, 0.1, 50).astype(np.float32)] * n, pa.list_(pa.float32())), "weight": pa.array([np.ones(50, np.float32)] * n, pa.list_(pa.float32()))})
    m = write_dataset(t, tmp_path / "sl", geometry=Geometry.SIGHTLINE, survey=SurveyMeta("lya", "lyman_alpha"), loader=LoaderInfo("t", "1", {}))
    v = DatasetView.from_path(tmp_path / "sl"); assert m.geometry == Geometry.SIGHTLINE and v.n_rows == n and v.table(columns=["delta"])["delta"].type == pa.list_(pa.float32())
    with pytest.raises(SchemaError, match="wavelength"): write_dataset(t.drop(["wavelength"]), tmp_path / "bad", geometry=Geometry.SIGHTLINE, survey=SurveyMeta("lya", "lyman_alpha"), loader=LoaderInfo("t", "1", {}))

def test_validity_version_increments_on_overwrite(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 100, 0); m2 = write_synthetic_dataset(tmp_path / "ds", 100, 1, overwrite=True)
    assert m2.validity.version == 2 and (tmp_path / "ds/_history").exists() and len(list((tmp_path / "ds/_history").glob("*.manifest.json"))) == 1
```

- [ ] **Step 3: Run, expect failure**
- [ ] **Step 4: Implement `writer.py`.** Algorithm (the part the signature does not determine):
  1. `with atomic_dir(out) as tmp:` — if `out` exists and not `overwrite` → `FormatError` before entering.
  2. Stage: iterate batches; on the first batch `validate_schema`; per batch: compute `hpix32` if absent (`radec_to_pix(32, ra, dec)`), validate `z_type` values against `z_types()`, `cast_to_core`; write to `tmp/stage/` with `ds.write_dataset(batch_iterator_as_RecordBatchReader, format="parquet", max_rows_per_group=batch_size)`; count rows.
  3. `nside = partition_nside or auto_partition_nside(n_rows)`; project `cell = coarsen(hpix32, 32, nside)` with `pc.shift_right`; `ds.write_dataset(ds.dataset(tmp/"stage").scanner(columns={...all, "cell": expr}), tmp/"data", partitioning=ds.partitioning(pa.schema([("cell", pa.int32())]), flavor="hive"), basename_template="part-{i}.parquet", max_rows_per_file=MAX_ROWS_PER_FILE, existing_data_behavior="error")`; delete `tmp/stage`.
  4. Index: for each written file, `pq.ParquetFile(f).metadata` → sum row counts; per stats column min/max over row groups (`statistics.min/max`, converted to float); `path` relative to `out`; `cell` from the directory name.
  5. Manifest: `validity.version` = previous version + 1 if `out/manifest.json` existed (copy the previous manifest to `tmp/_history/<previous created_utc>.manifest.json`, preserving any existing `_history`); `created_utc` ISO-8601 `Z`.
  6. `write_partition_index`, then `write_manifest` last.
  Parquet options: `compression="zstd"`, `write_statistics=True`, dictionary encoding on.
- [ ] **Step 5: Run, expect PASS** (the memory test may need `batch_size` tuning; 400 MB is the gate).
- [ ] **Step 6: Commit** — `feat(data): streaming OUF 3.0 writer`

---

### Task 5: `data.view`

**Files:**
- Create: `oneuniverse/data/view.py`
- Test: `test/data/test_view.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.fixture(scope="module")
def ds(tmp_path_factory): p = tmp_path_factory.mktemp("v") / "ds"; write_synthetic_dataset(p, 100_000, 0, partition_nside=8); return DatasetView.from_path(p)

def test_properties(ds): assert ds.n_rows == 100_000 and ds.nside == 8 and ds.n_partitions == ds.index.num_rows and "mag" in ds.columns

def test_cone_scan_matches_bruteforce(ds):
    cone = Cone(185.0, 15.0, 10.0); full = ds.table(columns=["ra", "dec", "object_id"])
    exp = full.filter(pa.array(cone.contains(full["ra"].to_numpy(), full["dec"].to_numpy())))
    got = ds.table(cone=cone, columns=["ra", "dec", "object_id"])
    assert sorted(got["object_id"].to_pylist()) == sorted(exp["object_id"].to_pylist())
    assert len(ds.cells_for(cone=cone)) < ds.n_partitions / 4

def test_view_cone_at_ra_zero(ds):
    cone = Cone(0.2, 0.0, 2.0); full = ds.table(columns=["ra", "dec"]); n = int(cone.contains(full["ra"].to_numpy(), full["dec"].to_numpy()).sum())
    assert ds.table(cone=cone).num_rows == n and n > 0

def test_view_patch_wrapping(ds):
    p = SkyPatch(350.0, 10.0, -5.0, 5.0); full = ds.table(columns=["ra", "dec"]); n = int(p.contains(full["ra"].to_numpy(), full["dec"].to_numpy()).sum())
    assert ds.table(patch=p).num_rows == n and n > 0

def test_z_and_ranges_prune_and_filter(ds):
    t = ds.table(z=(0.5, 0.6), ranges={"mag": (18.0, 19.0)}, columns=["z", "mag"])
    assert t.num_rows > 0 and pc.min(t["z"]).as_py() >= 0.5 and pc.max(t["mag"]).as_py() <= 19.0

def test_empty_result_keeps_schema(ds):
    t = ds.table(z=(5.0, 6.0)); assert t.num_rows == 0 and t.schema.equals(ds.table(columns=None).schema)

def test_take_returns_rows_in_order(ds):
    pid, tab = next(ds.iter_partitions(columns=["object_id"])); ids = tab["object_id"].to_numpy()
    got = ds.take(np.array([pid, pid], np.int32), np.array([5, 2]), columns=["object_id"])
    assert got["object_id"].to_pylist() == [ids[5], ids[2]]

def test_read_returns_dataframe_with_categoricals(ds):
    df = ds.read(z=(0.1, 0.2), columns=["z", "survey", "z_type"]); assert isinstance(df, pd.DataFrame) and str(df["survey"].dtype) == "category"

def test_scan_is_lazy_batches(ds):
    it = ds.scan(columns=["ra"], batch_size=1000); b = next(it); assert isinstance(b, pa.RecordBatch) and b.num_rows <= 1000

def test_load_pdf_requires_extension(ds):
    with pytest.raises(FormatError, match="pdf"): ds.load_pdf()
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement.** `from_path` reads manifest + index once, builds `ds.dataset(path/"data", format="parquet", partitioning=hive int32 "cell")` once. `cells_for`: numpy over index columns — z/t/ranges by interval overlap; cone by `cone_pixels(nside, …)` intersected with index cells; patch by `patch_pixels`; if nothing requested → all cells. `scan` = `dataset.scanner(columns=..., filter=cell.isin(cells) & row-level expr, batch_size=...)`; row-level expr: `cone_expr` (haversine in `pc`), `patch_expr` (handles wrap with `|`), `range_expr` closed intervals, `t` via the `temporal` extension's `time_column` (FormatError if `t` given without the extension). `_empty_table`: `dataset.schema.empty_table()` projected. `take`: group by partition, `pq.read_table(file, columns).take(offsets)`, concat, then reorder to the input order. `read`: `table(...).to_pandas(types_mapper=None, categories=dictionary columns)`.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(data): DatasetView with vectorised pruning and take()`

---

### Task 6: `data.pdf`

**Files:**
- Create: `oneuniverse/data/pdf.py`
- Modify: `oneuniverse/data/view.py` (`load_pdf`), `test/fixtures/synthetic.py` (`with_pdf`)
- Test: `test/data/test_pdf.py`

- [ ] **Step 1: Write the failing tests**

```python
def _gauss_interp(n=200, ng=61):  # rows with known mu, sigma on grid 0..3
    grid = np.linspace(0, 3, ng); rng = np.random.default_rng(0); mu = rng.uniform(0.5, 2.5, n); sig = rng.uniform(0.05, 0.2, n)
    p = np.exp(-0.5 * ((grid[None] - mu[:, None]) / sig[:, None]) ** 2); p /= np.trapezoid(p, grid, axis=1)[:, None]
    return PdfSpec("interp", grid=tuple(grid)), p.astype(np.float32), mu, sig

def test_interp_moments_cdf_ppf():
    spec, p, mu, sig = _gauss_interp(); pz = ProbabilisticRedshift(spec, p)
    assert np.allclose(pz.mean(), mu, atol=2e-3) and np.allclose(pz.std(), sig, atol=5e-3)
    assert np.allclose(pz.cdf(mu), 0.5, atol=0.02) and np.allclose(pz.ppf(0.5), mu, atol=0.02)

def test_sample_kind_mean_std_and_unsupported():
    draws = [np.random.default_rng(i).normal(1.0, 0.1, 50).astype(np.float32) for i in range(10)]
    pz = ProbabilisticRedshift(PdfSpec("sample"), draws)
    assert np.allclose(pz.mean(), 1.0, atol=0.06) and np.allclose(pz.std(), 0.1, atol=0.05)
    with pytest.raises(NotImplementedError): pz.cdf(1.0)

def test_quant_ppf_and_sample_seeded():
    levels = (0.16, 0.5, 0.84); vals = np.array([[0.9, 1.0, 1.1], [1.9, 2.0, 2.1]], np.float32)
    pz = ProbabilisticRedshift(PdfSpec("quant", quant_levels=levels), vals)
    assert np.allclose(pz.ppf(0.5), [1.0, 2.0]) and pz.sample(5, seed=1).shape == (2, 5) and np.array_equal(pz.sample(5, seed=1), pz.sample(5, seed=1))

def test_pdf_round_trip_through_dataset(tmp_path):
    spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 31)))
    write_synthetic_dataset(tmp_path / "ds", 2_000, 0, with_pdf=spec); v = DatasetView.from_path(tmp_path / "ds")
    assert v.manifest.extension("pdf")["kind"] == "interp" and v.schema.field("z_pdf").type == pa.list_(pa.float32(), 31)
    pz = v.load_pdf(); assert len(pz) == 2_000 and pz.mean().shape == (2_000,)
    rows = v.table(z=(0.5, 1.0), columns=["z_pdf"]); assert len(v.load_pdf(rows)) == rows.num_rows
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement** vectorised: `interp` mean/std by trapezoid over the 2-D array; `cdf` by cumulative trapezoid + `np.interp` per row via `np.searchsorted` on a 2-D cumulative (vectorised with `take_along_axis`); `quant` ppf by interpolating levels; `sample`: interp → inverse-CDF on uniform draws, quant → inverse-CDF on quantile levels, sample → `rng.choice` per row vectorised with `take_along_axis`. `from_table` reads `spec.value_column` as fixed-size list (interp/quant) or list (sample). Writer: when the `pdf` extension is passed, cast the value column to `pa.list_(pa.float32(), n_components)` for interp/quant. `DatasetView.load_pdf(rows)` reads `[value_column]` only (via `self.table(columns=[...])` when `rows is None`).
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(data): photo-z PDFs (interp, sample, quant) as a manifest extension`

---

### Task 7: Loaders — base, registry, FITS fixtures, eBOSS, DESI

**Files:**
- Create: `oneuniverse/data/loaders/__init__.py`, `base.py`, `registry.py`, `eboss_qso.py`, `desi_qso.py`, `test/fixtures/fits_like.py`
- Test: `test/data/test_loaders.py`

- [ ] **Step 1: Write the FITS fixtures** per the interface block (astropy `fits.BinTableHDU.from_columns`; include at least one big-endian float column as astropy writes them and one 2-D column for eBOSS).

- [ ] **Step 2: Write the failing tests**

```python
def test_registry_lists_two_real_surveys():
    s = list_surveys(); assert set(s) >= {"eboss_qso", "desi_qso"} and s["eboss_qso"].kind == "spectroscopic"
    with pytest.raises(NotFoundError): get_loader("nope")

def test_desi_batches_core_complete_and_filtered(tmp_path):
    src = write_fake_desi_dr1_fits(tmp_path, 5_000, 0); loader = get_loader("desi_qso")
    t = pa.concat_tables([pa.Table.from_batches([b]) for b in loader.batches(src.parent, batch_size=1_000)])
    validate_schema(t.schema, Geometry.POINT)
    assert t.num_rows < 5_000 and set(t["z_type"].to_pylist()) == {"spec"} and "mag_g" in t.column_names and "targetid" in t.column_names

def test_source_row_survives_cut(tmp_path):
    src = write_fake_desi_dr1_fits(tmp_path, 3_000, 1); loader = get_loader("desi_qso")
    t = pa.concat_tables([pa.Table.from_batches([b]) for b in loader.batches(src.parent, batch_size=700)])
    raw = fits.getdata(src, 1); rows = t["source_row"].to_numpy()
    assert np.allclose(raw["RA"][rows], t["ra"].to_numpy()) and np.all(np.char.strip(raw["SPECTYPE"][rows]) == "QSO") and np.all(raw["ZWARN"][rows] == 0)
    assert np.array_equal(t["object_id"].to_numpy(), stable_object_id("desi_qso", rows))

def test_eboss_batches_and_dla_count(tmp_path):
    src = write_fake_eboss_dr16q_fits(tmp_path, 4_000, 0); loader = get_loader("eboss_qso")
    t = pa.concat_tables([pa.Table.from_batches([b]) for b in loader.batches(src.parent)])
    validate_schema(t.schema, Geometry.POINT); raw = fits.getdata(src, 1); rows = t["source_row"].to_numpy()
    assert np.all(raw["IS_QSO_FINAL"][rows] == 1) and np.all(np.isnan(t["z_err"].to_numpy())) and t["n_dla"].to_numpy().max() <= 5
    assert np.array_equal(t["n_dla"].to_numpy(), (raw["Z_DLA"][rows] > 0).sum(axis=1))

def test_loader_kwargs_z_cut(tmp_path):
    src = write_fake_desi_dr1_fits(tmp_path, 2_000, 2); t = pa.concat_tables([pa.Table.from_batches([b]) for b in get_loader("desi_qso").batches(src.parent, z_min=2.0)])
    assert pc.min(t["z"]).as_py() >= 2.0

def test_fits_batches_native_types(tmp_path):
    src = write_fake_eboss_dr16q_fits(tmp_path, 100, 0)
    start, t = next(read_fits_batches(src, 1, ["RA", "SDSS_NAME", "Z_DLA"], 100))
    assert start == 0 and t["RA"].type == pa.float64() and pa.types.is_string(t["SDSS_NAME"].type) and pa.types.is_fixed_size_list(t["Z_DLA"].type)
```

- [ ] **Step 3: Run, expect failure**
- [ ] **Step 4: Implement.**
  - `base.py` per the interface block. `read_fits_batches`: `fits.open(path, memmap=True)` once; for `start in range(0, nrows, batch_size)`: `hdu.data[start:stop]`, per column `np.ascontiguousarray(col).byteswap().view(newbyteorder)` when big-endian; strings `np.char.strip` → `pa.array(..., pa.string())`; 2-D → `pa.FixedSizeListArray.from_arrays`.
  - `FitsCatalogLoader.batches(source, *, batch_size=250_000, z_min=None, z_max=None, **options)` order per batch: read native → `mask = self.row_filter(native, **options)` → `source_row = start + flatnonzero(mask)` → select+rename `columns` → `derive` → z cut (`z_min`/`z_max` on the standard `z`, applied to `source_row` too) → stamp `survey` (dictionary), `z_type`, `z_err` (from `z_err_column` or NaN float32), `object_id`, `hpix32` → `cast_to_core`. Subclasses implement `row_filter(self, native, *, qso_only=True, ...)` with their own keyword options; unknown options raise `TypeError` naturally.
  - `eboss_qso.py`: `filename = "DR16Q_Superset_v3.fits"`, `columns` from `main:…/eboss_qso/loader.py` `_COLUMN_MAP` (keep `IS_QSO_FINAL` → `is_qso`), `row_filter(native, *, qso_only=True)` = `IS_QSO_FINAL == 1` when `qso_only`, `z_err_column = None`, `derive`: `n_dla = count(Z_DLA > 0)` (int8), `extensions = {"coordinate": {"version": 1, "frame": "icrs"}, "spectrum": {"version": 1, "wavelength_convention": "vacuum", "log_binned": True, "rest_frame": False}}`, `meta` reference "Lyke et al. 2020, ApJS 250 8".
  - `desi_qso.py`: `filename = "QSO_full.dat.fits"`, `columns` from the old `_COLUMN_MAP`, `row_filter(native, *, qso_only=True, good_zwarn=True)`: `strip(SPECTYPE) == "QSO"` & `ZWARN == 0`, `z_err_column = "ZERR"`, `derive`: `mag_{g,r,z} = 22.5 - 2.5 log10(flux)` (NaN for flux ≤ 0), same extensions, reference "DESI Collaboration 2025, DR1".
  - `registry.py`: module `Registry("survey loader", key=lambda cls: cls.meta.name)`; `get_loader` returns an instance; `NotFoundError` on unknown; `load_entry_points` called in `loaders/__init__.py` after importing the two built-ins.
- [ ] **Step 5: Run, expect PASS**
- [ ] **Step 6: Commit** — `feat(data): declarative batch-streaming FITS loaders (eBOSS DR16Q, DESI DR1 QSO)`

---

### Task 8: `data.convert` and the package surface

**Files:**
- Create: `oneuniverse/data/convert.py`
- Modify: `oneuniverse/data/__init__.py`, `oneuniverse/__init__.py`
- Test: `test/data/test_convert.py`, `test/data/test_public_api.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_convert_desi_end_to_end(tmp_path):
    src = write_fake_desi_dr1_fits(tmp_path / "raw", 6_000, 0)
    m = convert("desi_qso", source=src.parent, out=tmp_path / "db/desi_qso", batch_size=1_000)
    assert m.survey.name == "desi_qso" and m.loader.kwargs == {"batch_size": 1_000} and m.original_files[0].path == "QSO_full.dat.fits" and m.original_files[0].n_rows == 6_000
    v = DatasetView.from_path(tmp_path / "db/desi_qso"); assert v.n_rows == m.n_rows and v.manifest.extension("spectrum")["wavelength_convention"] == "vacuum"

def test_convert_accepts_loader_instance_and_overwrite(tmp_path):
    src = write_fake_eboss_dr16q_fits(tmp_path / "raw", 1_000, 0)
    convert(get_loader("eboss_qso"), source=src.parent, out=tmp_path / "e")
    with pytest.raises(FormatError): convert("eboss_qso", source=src.parent, out=tmp_path / "e")
    m = convert("eboss_qso", source=src.parent, out=tmp_path / "e", overwrite=True); assert m.validity.version == 2

def test_public_surface():
    import oneuniverse.data as d
    for name in ["Geometry", "SurveyMeta", "Manifest", "read_manifest", "DatasetView", "write_dataset", "convert", "list_surveys", "get_loader", "register", "SurveyLoader", "FitsCatalogLoader", "PdfSpec", "ProbabilisticRedshift", "Cone", "SkyPatch", "ZRange"]:
        assert hasattr(d, name)
    assert len(d.__all__) <= 24
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement `convert`** (resolve loader; `original_files = loader.original_files(source)` hashed by `hash_file`; `LoaderInfo(name, loader.version, kwargs)`; `write_dataset(loader.batches(source, batch_size=..., **kw), ...)` with `extensions=loader.extensions`, `geometry=loader.geometry`). Fill `__all__` in `data/__init__.py`; top-level `oneuniverse/__init__.py` re-exports `data`, `convert`, `DatasetView`, `Cone`, `SkyPatch`, `ZRange`.
- [ ] **Step 4: Run, expect PASS; run the whole suite**
- [ ] **Step 5: Commit** — `feat(data): convert() and the public surface`

---

### Task 9: Close-out

- [ ] **Step 1: Run** `pytest -n 6 --durations=10` and `pytest` (single process). Expected: green; single-process < 60 s.
- [ ] **Step 2: Update `plans/refactor-v1/STATUS.md`** (R1 row) and the memory file; commit + push claude_config.
- [ ] **Step 3: Commit** — `docs: R1 close-out`

**Exit criterion:** fake eBOSS + DESI convert through `convert()`; cone, patch, z and range queries prune partitions and match brute force; a 2 M-row streamed conversion peaks under 400 MB; guards green.
