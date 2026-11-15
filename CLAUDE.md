# oneuniverse

## Mission

Data and orchestration substrate for a three-pillar cosmology stack; it ships no estimators and no samplers. Pillar 1: every survey catalog goes into one lazily and partially queried on-disk format (OUF), with one cross-survey identity (ONEUID), sub-object links and weights (`data`, `combine`), with no cosmology. Pillar 1 to 2: `measure` builds the cosmology-free `MeasurementSet` that external estimators (flip, pycorr, picca) consume. Pillar 3: `simulation` is a partial-access store and request/lineage catalog for simulation products, and `twin` holds the engine contracts that couple data and simulation; the physics is pluggable and only `reference` ships.

## Hard rules

Enforced by `test/test_guards.py` (AST walk, not docstrings), so a violation fails CI.

| Rule | Enforcement |
|---|---|
| No cosmology in `data`, `combine`, `measure` | Identifiers (`H0`, `Om0`, `sigma8`, `w0`, ...), strings (`comoving_distance`, ...), `astropy.cosmology`, camb/classy/pyccl/colossus/cosmoprimo/jax_cosmo are flagged. The `ALLOW` dict is the only escape (`measure/product.py`: forbidden column names as strings). `MeasurementSet` rejects cosmology-derived column names. |
| `_core` and `simulation` import nothing from `data`, `combine`, `measure`, `twin`, `reference` | `_core` also imports no `simulation`. `simulation` imports no physics package. |
| `measure` imports nothing from `simulation`, `reference`, `twin` | AST guard. |
| `simulation` and `twin/engine.py` never import `reference` | AST guard. `reference` and `twin/observe.py` may import `simulation`. |
| `duckdb` only in `data/sql.py` and `simulation/sql.py` | AST guard; imported lazily, optional extra `[sql]`. |
| Partial access is the only read path for simulation products | `SimStore` reads take a selector (`Cube`, `Cone`); a whole-box read is the explicit `scan_full()`. |
| Errors derive from `oneuniverse._core.errors` | `ValidationError` (is a `ValueError`) for bad arguments, `NotFoundError` (is a `KeyError`), `FormatError`, `SchemaError`. No `except Exception: pass/warn`; the one exception is `Registry.load_entry_points`, which isolates broken third-party plugins. |
| Deps: numpy, pyarrow, pandas, healpy, scipy, astropy; extras `fits`, `sql`, `mpi`, `dev`, `docs` | Python >= 3.11. No binary files committed (figures go to the ignored `examples/output/`). |

## Layout

- `_core/`: errors, `Registry`, `fs` (atomic write/dir, staging suffixes, `is_staging`), `geometry` (`Cone`, `SkyPatch`, `ZRange`, `Cube`), `healpix`.
- `data/`: OUF 3.0. `schema` (core columns, `z_type` registry), `manifest`, `writer` (`write_dataset`), `view` (`DatasetView`), `convert`, `loaders/` (eBOSS DR16Q, DESI DR1 QSO), `pdf`, `sidecar`, `crossmatch`, `oneuid`, `links`, `database` (`Database`), `sql`.
- `combine/`: `Weight` primitives with `*` composition, `default_weight_for` registry, `WeightedCatalog` (ivar / best / mean / priority strategies).
- `measure/`: `Product`/`FieldMap`/`Sightline`, `MeasurementSet`, `build_pointset` (the one spine), presets (`galaxy_clustering`, `cosmic_shear`, `three_by_two`), `attach_photoz`, `save`/`load`.
- `simulation/`: OUF-Sim: `specs`, `manifest`, `execution` (`ExecutionPlan`), `store/` (`SimStore`, `build_store`, `ingest_field`, native adapters), `converter` registry, `db` (`SimDatabase`, `dispatch`), `sql`.
- `reference/`: the reference engine (EH98 linear theory, Zel'dovich, PM, TreePM resim, Wiener / constrained realisation, lognormal mock, linear fixture sim, four engines). Registers on import of `reference.engines` / `reference.native`; import submodules explicitly.
- `twin/`: `ForwardEngine`/`ReconstructionEngine` ABCs and registry, `observe_from_view` (data to twin), `run_engine_to_store` (engine to store).
- Top level re-exports only `data`, `convert`, `DatasetView`, `Cone`, `SkyPatch`, `ZRange`.

## Formats

OUF 3.0 (`format_version` "3.0.0"), one dataset directory:
```
<dataset>/manifest.json     identity: geometry (point|sightline|lightcurve), survey, loader, Arrow schema,
                            partitioning {nside, nest}, original_files (sha256), validity {version}, dataset_uuid, extensions
<dataset>/_index.parquet    one row per partition: path, n_rows, cell, ra/dec/z min-max, extra stats
<dataset>/data/cell=<n>/part-<i>.parquet   sorted by (hpix32, source_row); cell numbered at the manifest nside
<dataset>/_history/<created_utc with ':' → '-'>.manifest.json   kept on overwrite; validity.version increments
```
Sidecars (ONEUID, links): `<root>/_<kind>/<name>/<version>/{table.parquet,meta.json}` plus `current.json`; version id = UTC write time to the microsecond. Publishes go through `<path>.tmp` then rename (`atomic_dir`).

OUF-Sim 1.0 (`OUFSIM_FORMAT_VERSION` "1.0.0"): `<store>/manifest.json`, `layout.json` (lists committed products: the per-product commit point, unlisted product dirs are orphans a retry replaces; the store directory itself is published atomically), `products/<name>/z=<z:.3f>/{_index.parquet, chunk_<id>.parquet | tile_<id>.npy | pix_<id>.parquet}`. Catalogs are chunked on an `n_side`^3 periodic grid, fields in tiles, lightcones in NEST pixels. A `SimDatabase` root holds stores plus `_simdb/{requests.json, lineage.parquet}`. `measure.save` writes `MEASURE_FORMAT_VERSION` "1.0.0".

## Identity

- `source_row` is the true row in the original file; loaders preserve it through cuts. It is the only row identity and the recommended join key (also in SQL).
- `object_id = (hash32(survey) << 32) | source_row` (`schema.stable_object_id`; needs `source_row` < 2^32).
- ONEUID and links store `(dataset, source_row)` plus `(partition, offset)` computed at build time, so hydration is a `take` per partition, never a scan. They record each dataset's `DatasetView.version`; a changed dataset raises `FormatError` ("rebuild the index"). Links also record the ONEUID version.
- Sidecars are bitemporal: `Database(root, as_of=...)` (timezone-aware) resolves versions and is read-only. A view opened before an overwrite raises `FormatError` (`dataset_uuid`).

## Test conventions

- `test/` mirrors the package (`test/data/test_writer.py`; `test/core/` covers `_core`). No phase names in files or tests.
- Default run is `-m "not slow"`: under 60 s single-process on the dev machine (CI budget 120 s, enforced on the 3.12 job); `-n auto` for speed. `slow` = ensemble physics, memory, real-data and example tests (nightly CI). `full_stage` keeps the production writer staging; otherwise the autouse conftest fixture patches `writer._BUCKET_SHIFT` to 12 coarse buckets (output is invariant).
- Fixtures live in `test/fixtures/` and import as `fixtures.<module>` (no top-level `test/__init__.py`). Shared fixtures are in `test/conftest.py`: `two_survey_db`, `two_survey_db_large`, `synthetic_view`, `cosmo`; expensive databases are built once and cloned.
- No binaries: fixtures synthesize FITS and sims in `tmp_path`. Real-data tests skip unless `ONEUNIVERSE_DATA_ROOT` is set. Ensemble tests use fixed seed lists.

## Things that bite

- Partition `cell=<n>` is numbered at the manifest's `partitioning.nside` (not 32); `hpix32` is the fine NEST pixel, `cell = hpix32 >> shift`. Use the manifest nside when reading partitions yourself (`writer.py` partition pass).
- `z_type` is a registry: values outside `schema.z_types()` raise `SchemaError`; add one with `register_z_type`, which rejects duplicates and non-lowercase identifiers.
- `write_dataset` needs every core column (`SchemaError` names the missing ones), `ra`/`dec` above all; only `hpix32` is optional, and it is always recomputed from `ra`/`dec`, overwriting a supplied column; empty input and an existing `out` without `overwrite=True` are errors.
- `SimStore.read_box` is periodic: selection uses positions wrapped into `[0, box)`, a cube across the boundary is split, rows come back with their stored (unwrapped) positions.
- `reference.reconstruct.constrained_realization` requires a keyword `seed` (non-negative int, no default); the truth's seed does not reproduce the truth.
- `duckdb` is optional (`pip install oneuniverse[sql]`). DuckDB 2.0 relations have no `fetchone`: use `con.execute(q).fetchone()`. `partition` and `offset` are reserved words in SQL: quote them.

## Where the history is

`docs/history/` holds the archived pre-rewrite plans and review (not maintained). `plans/2026-11-12-refactor-v1-design.md` is the 2.0 design; `plans/refactor-v1/STATUS.md` carries the decisions log. `research/` holds survey-format research notes.
