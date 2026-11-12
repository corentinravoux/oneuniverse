# Refactor v1 — status

One row per step, written when the step's exit criterion is met. Deviations
from the design are recorded under the table with their reason. SHAs refer to
the history as rewritten by the owner on 2026-11-12 (author email change).

| step | date | commit | tests (passed/skipped) | default suite time | notes |
|---|---|---|---|---|---|
| R0 | 2026-11-12 | a0251e2 | 931 / 12 | 51 s (`-n 6`); 118 s single-process | `_core` (errors, registry, fs, geometry, healpix) + guards + CI; old suite still green. Single-process time is the old suite's; the < 60 s target is R8's exit criterion. CI has not yet run on the branch (not pushed). |
| R1 T1 | 2026-11-12 | 41bad19 | 273 / 5 | 15 s (`-n 6`) | old data/combine/measure/twin removed |
| R1 | 2026-11-12 | 5bc6b38 | 376 / 4 | 28 s (`-n 6`); 86 s single-process | OUF 3.0: schema, manifest, bucketed streaming writer, DatasetView (pruning, take, stale-view guard), PDFs, eBOSS/DESI loaders, convert(). Fake 2 M DESI convert peaks 213 MiB anon RSS. Exit criterion met. |

## Decisions taken during execution

- **R0 — error hierarchy.** `ValidationError(OneuniverseError, ValueError)` and
  `NotFoundError(OneuniverseError, KeyError)`: argument errors raise
  `ValidationError` everywhere, unknown names `NotFoundError`. Reconciles the
  global constraint (errors derive from `OneuniverseError`) with step plans that
  test `pytest.raises(ValueError)`.
- **R0 — staging directories.** `atomic_dir` keeps `<path>.tmp` / `<path>.old`;
  `_core.fs.is_staging()` / `STAGING_SUFFIXES` must be used by R2 `Database`
  discovery and R5 `SimDatabase.scan` so a crashed publish never appears as a
  dataset.
- **R0 — cosmology guard.** Also flags string constants equal to forbidden
  identifiers/imports, `astropy.cosmology` attribute chains, string-based
  imports, and `pyccl`, `colossus`, `cosmoprimo`, `jax_cosmo` (spec §9).
  R4's `MeasurementSet` validator needs the forbidden column names as strings:
  re-add an `ALLOW` entry for `measure/measurement_set.py` limited to them.
- **R0 — `patch_pixels`.** The plan's four-corner `query_polygon` is not an
  inclusive cover (great-circle edges bow poleward); the implementation slices
  each RA piece and pads the equatorward edge with the exact bulge
  `atan(tan δ · cos(w/2))`, falling back to a polar strip within one pixel of a
  pole. Randomised checks: 0 misses over ~23 000 patches, nside 1–256.
- **R1 — writer algorithm.** The plan's single `pyarrow.dataset.write_dataset`
  pass over an unsorted stage crashed at nside ≥ 16 (1024-partition limit),
  held the whole dataset in memory and wrote ~130-row row groups. Replaced by a
  two-pass bucketed writer: stage by NEST nside-8 bucket, then per bucket sort
  by `(hpix32, source_row)` and write cells sequentially. Memory is bounded by
  the densest nside-8 bucket (~54 deg²); row order is reproducible.
- **R1 — dataset identity.** `Manifest.dataset_uuid` (new per publish) is
  stamped into every parquet file; a `DatasetView` opened before an overwrite
  raises `FormatError` instead of returning another version's rows.
  `DatasetView.version` = `(validity.version, created_utc, dataset_uuid)` —
  R2 should record it with each ONEUID build and check it at hydration.
- **R1 — defaults.** Loader/`convert` `batch_size` default 100 000 (plan:
  250 000) to keep conversion under the 400 MB gate.
- **R1 — PDFs.** `quant` mean/std are truncated-tail estimates over the stored
  levels; `quant` sampling draws within the stored level range.
- **R1 — plugins.** `Registry.load_entry_points(on_error="warn", validate=…)`
  isolates a broken third-party loader (warning + `plugin_errors`): the single
  documented exception to the no-`except Exception` rule.
- **Open for R4.** `DatasetView.scan` cannot yet return `(partition, offset)`;
  `build_pointset` needs it (fix before R4 Task 3).
