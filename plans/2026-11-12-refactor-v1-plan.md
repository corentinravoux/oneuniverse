# oneuniverse 1.0 Refactor — Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite oneuniverse as version 1.0 under its existing three-pillar concepts: half the code, one identity per row, bounded memory, enforced boundaries, a DuckDB SQL face, and a reference physics engine separated from the simulation substrate.

**Architecture:** Nine sequential steps (R0–R8), each its own plan file in `plans/refactor-v1/`. Each step deletes the old subpackage it replaces at its start and rebuilds it test-first; the other old subpackages stay importable until their own step. Shared helpers live in `oneuniverse/_core`; parquet is the storage everywhere, pyarrow the internal data model, pandas the user-facing frame, DuckDB the optional SQL query engine.

**Tech Stack:** Python ≥ 3.11, numpy ≥ 1.26, pyarrow ≥ 15, pandas ≥ 2.1, healpy ≥ 1.16, scipy ≥ 1.11, astropy ≥ 6 (FITS I/O only). Optional: fitsio, duckdb ≥ 1.0, mpi4py. Tests: pytest + pytest-xdist. Docs: MkDocs Material + mkdocstrings. CI: GitHub Actions.

**Spec:** `plans/2026-11-12-refactor-v1-design.md` (approved 2026-11-12). Audit with verified bugs: `~/Software/claude_config/reports/code-analysis/2026-11-12_oneuniverse-refactor-audit/`.

## Global Constraints

- `requires-python = ">=3.11"`; CI matrix 3.11, 3.12, 3.13, 3.14.
- Hard dependencies exactly: numpy, pyarrow, pandas, healpy, scipy, astropy. `pyyaml` removed. Extras: `fits = [fitsio]`, `sql = [duckdb>=1.0]`, `mpi = [mpi4py]`, `dev`, `docs`.
- No cosmology in `oneuniverse/data`, `oneuniverse/combine`, `oneuniverse/measure` (identifiers `H0`, `Om0`, `omega_m`, `Omega_m`, `sigma8`, `sigma_8`, imports of `astropy.cosmology`, strings `comoving_distance`, `r_comoving`, `dist_mpc`); enforced by `test/test_guards.py` from R0.
- `oneuniverse/simulation` and `oneuniverse/_core` import nothing from `data`, `combine`, `measure`, `twin`, `reference`. `measure` imports nothing from `simulation`, `reference`, `twin`. `simulation` and `twin/engine.py` import nothing from `reference`.
- Every public simulation reader takes a selector; whole-box reads only via `scan_full`.
- OUF format version `3.0.0`, no reader for 2.x. OUF-Sim format `1.0.0`, no reader for 0.1.
- Core POINT columns: `ra f8, dec f8, z f8, z_type dict<int8,str>, z_err f4, object_id i8, survey dict<int16,str>, source_row i8, hpix32 i4`. `object_id = (survey_hash32 << 32) | source_row`.
- All on-disk publishes atomic: write to `<target>.tmp`, rename; previous artefact untouched until the new one is complete.
- No `except Exception: pass/warn` in library code; errors derive from `oneuniverse._core.errors.OneuniverseError`.
- Default `pytest` run (`-m "not slow"`) under 60 s single-process on the dev machine; `slow` tests run in the nightly CI job.
- Conventional commits (`feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`), no phase prefixes. Branch `refactor-v1` off `main`; never commit to `main` during the work.
- Old code is available for porting via `git show main:<path>` after it is deleted on the branch. Port by re-authoring against the new interfaces, never by copying files whole.
- No binary files committed (figures go to ignored `examples/output/`).

## Review Focus

Inputs the spec implies but no single task would naturally test; each line names the task that pins it.

1. **A catalog whose objects straddle RA = 0/360** must be found by a `Cone` centred at RA ≈ 0 and by a `SkyPatch(350, 10, …)`; partition pruning must not drop the wrapped cells. Pinned in R1 Task 5 (`test_view_cone_at_ra_zero`, `test_view_patch_wrapping`).
2. **A survey loader whose row filter removes rows** must give `source_row` values that index the original FITS, and `object_id` that is unchanged by the cut. Pinned in R1 Task 7 (`test_source_row_survives_cut`).
3. **Rebuilding an ONEUID index with the same name while a reader holds the old one** must leave the old version readable `as_of` its time and the new one live; a crash simulated between the two renames must leave the live pointer consistent. Pinned in R2 Task 1 (`test_sidecar_crash_between_steps`).
4. **A `MeasurementSet` product with zero rows after selection** must raise `ValidationError` at construction, not produce empty randoms silently. Pinned in R4 Task 2 (`test_product_rejects_empty_catalog`).
5. **A simulation cube that straddles the periodic box boundary** must return the same rows as the union of the two wrapped sub-cubes. Pinned in R5 Task 5 (`test_read_box_periodic_wrap`).

---

## Execution protocol

1. Steps run in order R0 → R8; a step starts only when the previous step's exit criterion is met and committed.
2. Inside a step, tasks run in order; every task ends with the default `pytest` run green and a commit.
3. Each step's plan is self-contained: it names the files, interfaces, tests and commands. Read the design spec §5–§6 once before starting any step.
4. At the end of each step, update `plans/refactor-v1/STATUS.md` (table: step, date, commit, tests count, suite time) and the memory file `project_oneuniverse_refactor_v1.md` in the claude_config memory directory in use on the machine (commit + push claude_config).
5. If a step reveals that the design is wrong, stop, write the deviation in `STATUS.md` with the reason, and ask the owner before continuing.

## Steps

| Step | Plan file | Deletes | Builds | Exit criterion |
|---|---|---|---|---|
| R0 | `plans/refactor-v1/R0-foundation.md` | nothing | branch, pyproject, `_core/`, guards, CI | guards + old suite green on the branch |
| R1 | `plans/refactor-v1/R1-data-format.md` | `data/`, `combine/`, `measure/`, `twin/` + their tests | `data/{schema,manifest,writer,view,pdf,convert}`, `data/loaders/`, fixtures | fake eBOSS + DESI convert, cone/z queries prune, conversion memory bounded |
| R2 | `plans/refactor-v1/R2-identity.md` | — | `data/{sidecar,crossmatch,oneuid,links,database}` | row-identity regression green; 2 × 1 M synthetic cross-match < 60 s |
| R3 | `plans/refactor-v1/R3-combine.md` | — | `combine/{weights,defaults,catalog}` | weights + `WeightedCatalog` on R2 output, vectorised |
| R4 | `plans/refactor-v1/R4-measure.md` | — | `measure/*` | clustering + shear presets; save/load drops nothing |
| R5 | `plans/refactor-v1/R5-simulation.md` | `simulation/` + tests | `simulation/{specs,manifest,execution,converter,db}`, `simulation/store/*` | store ⊥ reference guard; periodic read; manifest round-trip |
| R6 | `plans/refactor-v1/R6-reference-twin.md` | — | `reference/*`, `twin/*` | four physics fixes with ensemble tests; resim loop store→store |
| R7 | `plans/refactor-v1/R7-sql.md` | — | `data/sql.py`, `simulation/sql.py` | DuckDB views answer cone / join / n(z) on fixtures |
| R8 | `plans/refactor-v1/R8-release.md` | `notebooks/`, `scripts/*`, `REVIEW.md`, `docs_site/` old pages | tests layout, examples, docs, CLAUDE.md, README, 1.0.0 | suite < 60 s; `mkdocs build --strict`; tag `v1.0.0` on merge |

## Interface ledger (names every step relies on)

Defined once here so later steps do not drift. Each step's plan repeats the ones it consumes.

```
oneuniverse._core.errors:   OneuniverseError, FormatError, SchemaError, NotFoundError, ValidationError
oneuniverse._core.healpix:  NSIDE_FINE = 32; radec_to_pix(nside, ra, dec, *, nest=True) -> int32[]; coarsen(pix, nside_from, nside_to) -> int32[];
                            cone_pixels(nside, ra, dec, radius_deg, *, nest=True) -> int32[]; patch_pixels(nside, ra_min, ra_max, dec_min, dec_max, *, nest=True) -> int32[]
oneuniverse._core.geometry: Cone(ra, dec, radius) ; SkyPatch(ra_min, ra_max, dec_min, dec_max) [.wraps] ; ZRange(lo, hi) ; Cube(lo: tuple[float,float,float], hi: tuple[float,float,float])
                            each with .contains(...) -> bool[]; Selection = Cone | SkyPatch
oneuniverse._core.fs:       atomic_write_bytes(path, data) ; atomic_write_json(path, obj) ; atomic_dir(path) -> ContextManager[Path] ; hash_file(path) -> str ; hash_bytes(b) -> str
oneuniverse._core.registry: Registry(kind, key=None) .register(obj, name=None) .get(name) .names() .__contains__ .load_entry_points(group)

oneuniverse.data.schema:    Geometry(Enum: POINT, SIGHTLINE, LIGHTCURVE) ; Z_TYPES ; register_z_type(name) ; core_schema(geometry) -> pa.Schema ;
                            validate_schema(schema: pa.Schema, geometry) -> None ; stable_object_id(survey: str, source_row: int64[]) -> int64[]
oneuniverse.data.manifest:  FORMAT_VERSION = "3.0.0" ; SurveyMeta(name, kind, description="", reference="", url="") ; LoaderInfo(name, version, kwargs) ;
                            OriginalFile(path, sha256, n_rows) ; Validity(valid_from, valid_to, version) ; Partitioning(nside, nest=True) ;
                            Manifest(format_version, geometry, survey, loader, created_utc, schema, partitioning, original_files, n_rows, validity, extensions) ;
                            write_manifest(path, m) ; read_manifest(path) -> Manifest ; read_partition_index(path) -> pa.Table ; write_partition_index(path, table)
oneuniverse.data.writer:    write_dataset(batches, out, *, geometry, survey, loader, original_files=(), partition_nside=None, extra_stats_columns=(), extensions=None, overwrite=False) -> Manifest
oneuniverse.data.view:      DatasetView.from_path(path) ; .manifest .index .geometry .n_rows .n_partitions .columns .schema ;
                            .cells_for(*, cone=None, patch=None, z=None, t=None, ranges=None) -> int32[] ;
                            .scan(*, columns=None, cone=None, patch=None, z=None, t=None, ranges=None, filter=None, batch_size=65536) -> Iterator[pa.RecordBatch] ;
                            .table(**same) -> pa.Table ; .read(**same) -> pd.DataFrame ; .take(partition, offset, *, columns=None) -> pa.Table ; .load_pdf(rows=None) -> ProbabilisticRedshift
oneuniverse.data.pdf:       PdfSpec(kind: "interp"|"sample"|"quant", grid=None, quant_levels=None, value_column="z_pdf") ; ProbabilisticRedshift(spec, values) .mean() .std() .sample(n, seed) .cdf(z) .ppf(q)
oneuniverse.data.loaders:   SurveyLoader (meta, geometry, z_type, version, extensions; batches(source, *, batch_size, **kw)) ; FitsCatalogLoader (filename, hdu, columns, row_filter, derive) ;
                            register(cls) ; get_loader(name) ; list_surveys() -> dict[str, SurveyMeta]
oneuniverse.data.convert:   convert(survey, *, source, out, overwrite=False, partition_nside=None, **loader_kwargs) -> Manifest
oneuniverse.data.sidecar:   VersionedSidecar(root, kind, name) .write(table, meta) -> str ; .load(*, as_of=None) -> (pa.Table, dict) ; .versions() ; VersionedSidecar.names(root, kind)
oneuniverse.data.crossmatch: CrossMatchRules(sky_tol_arcsec=1.0, dz_tol=5e-3, dz_tol_by_pair=(), reject_pairs=(), allow_same_survey_merge=False) .hash() ;
                            Pairs(left, right, sep_arcsec, dz) ; match_pair(a, b, rules, *, block_nside=8) -> Pairs ; components(n, left, right, sep, *, survey_of, allow_same_survey_merge) -> int64[]
oneuniverse.data.oneuid:    build_oneuid(db, datasets, *, rules, name="default") -> OneuidIndex ; load_oneuid(db, name, *, as_of=None) ;
                            OneuidIndex(table, meta, name) .datasets .rules .n_objects .multi_only() .restrict_to(datasets) ;
                            OneuidQuery(index, db) .select(*, cone=None, patch=None, z=None, datasets=None) -> int64[] ; .rows(oneuids) -> pa.Table ; .hydrate(oneuids, *, columns=None, datasets=None) -> Iterator[pa.Table] ; .hydrate_table(...) -> pa.Table
oneuniverse.data.links:     SubobjectRules(parent_kind, child_kind, sky_tol_arcsec=1.0, dz_tol=5e-3, relation="contains", accept_ambiguous=False) ;
                            build_links(db, *, parents, children, rules, name="default", oneuid_name="default") -> SubobjectLinks ; load_links(db, name, *, as_of=None) ;
                            SubobjectLinks(table, meta, name) .children_of(oneuid) .parent_of(oneuid) ; chain(db, starts, names, *, as_of=None) -> int64[]
oneuniverse.data.database:  Database(root, *, as_of=None, strict=True) .names() [name] .manifest(name) .path(name) .as_of(when) ;
                            .build_oneuid(datasets, *, rules, name) .load_oneuid(name, as_of=None) .list_oneuids() .oneuid_query(name) ;
                            .build_links(...) .load_links(...) .list_links() .chain(...) ; .summary() -> dict
oneuniverse.combine:        Weight (.columns ; __call__(table) -> float64[] ; __mul__) ; ColumnWeight ; InverseVarianceWeight(err_column, *, missing="unit") ; FKPWeight(nbar, P0, z_column="z") ;
                            HealpixMapWeight(map, nside, *, nest=True, nan="raise") ; ShearWeight(kind, ...) ; QualityMaskWeight(column, op, threshold) ; ProductWeight ;
                            default_weight_for(survey_kind, z_type) ; register_default(survey_kind, z_type, factory) ;
                            WeightedCatalog.from_oneuid(index, db, *, columns=None) .add_weight(dataset, w) .fill_defaults(db) .weights() .combine(value, *, variance=None, strategy="ivar") -> pa.Table
oneuniverse.measure:        Nz, Window, Regions, Product, FieldMap, Sightline, MeasurementSpec, MeasurementSet ; build_pointset(view, *, z, weight=None, nz_edges, mask=None, regions=20, randoms="generate", random_factor=10, keep_columns=(), seed=0) -> Product ; build_sightlines(view, *, z, regions=20, mask=None) -> Sightline ; attach_photoz ; tomographic_nz ;
                            galaxy_clustering(...) ; cosmic_shear(...) ; three_by_two(lens, source) ; save(ms, path) ; load(path) -> MeasurementSet
oneuniverse.simulation:     CosmologySpec, UnitFrameSpec, ProvenanceSpec, ProductDecl, RegionSpec, SimulationRequest ; OUFSIM_FORMAT_VERSION = "1.0.0" ; OUFSimManifest ; ExecutionMode(STREAMED, MPI) ; ExecutionPlan(mode, memory_budget_bytes, mpi_comm=None) .batch_for(bytes_per_row) ;
                            SimConverter (code ; detect(path) ; manifest(path) ; products(path) -> list[NativeProduct] ; convert(path, out, *, plan, projection)) ; register ; get_converter ; detect_converter
oneuniverse.simulation.store: ChunkIndex helpers (build_catalog_index, overlapping_chunks) ; write_catalog_product ; write_field_product ; write_lightcone_product ; ingest_field ;
                            NativeReaderAdapter (read_rows ; read_field_region) ; register_adapter ; get_adapter ; NativeProduct ; build_store ;
                            SimStore(root) .manifest .index .read_box(product, z, cube, *, columns=None, plan=None) .iter_box(...) .read_field_box(name, z, cube) .read_cone(product, cone, z_range) .scan_full(product, z) .last_read_stats
oneuniverse.simulation.db:  SimDatabase(root) .scan() .stores() .lineage() .request_region(parent, region, physics) -> SimulationRequest ; .dispatch(request, engine) -> SimStore ; .save() .load()
oneuniverse.reference:      linear (eh98_power, growth_factor, growth_rate, gaussian_field, zeldovich, find_peaks) ; pm (deposit_cic, pm_force, run_pm, zeldovich_pm_ic) ; resim (run_buffered, run_treepm, buffer_convergence) ;
                            reconstruct (wiener_reconstruct, constrained_realization) ; fieldstats (k_grid, bin_power, power_spectrum, cross_r, transfer) ; mock (lognormal_tracers, mask_slab, mask_ball, run_mock_challenge) ;
                            native (generate_linear_sim, write_packed_native, LinearSimConverter, PackedSimConverter) ; engines (LinearForwardEngine, PMForwardEngine, WienerReconstruction, ConstrainedRealization)
oneuniverse.twin:           Observation ; ProductBundle ; ForwardEngine ; ReconstructionEngine ; register_engine ; get_engine ; registered_engines ; observe_from_view ; run_engine_to_store
oneuniverse.data.sql:       connect(root, *, as_of=None) -> duckdb connection ; cone_where(cone, nside) -> str
oneuniverse.simulation.sql: connect(store_root) -> duckdb connection
```
