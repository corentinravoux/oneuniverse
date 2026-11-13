# Refactor v1 — status

One row per step, written when the step's exit criterion is met. Deviations
from the design are recorded under the table with their reason. SHAs refer to
the history as rewritten by the owner on 2026-11-12 (author email change).

| step | date | commit | tests (passed/skipped) | default suite time | notes |
|---|---|---|---|---|---|
| R0 | 2026-11-12 | a0251e2 | 931 / 12 | 51 s (`-n 6`); 118 s single-process | `_core` (errors, registry, fs, geometry, healpix) + guards + CI; old suite still green. Single-process time is the old suite's; the < 60 s target is R8's exit criterion. CI has not yet run on the branch (not pushed). |
| R1 T1 | 2026-11-12 | 41bad19 | 273 / 5 | 15 s (`-n 6`) | old data/combine/measure/twin removed |
| R1 | 2026-11-12 | 5bc6b38 | 376 / 4 | 28 s (`-n 6`); 86 s single-process | OUF 3.0: schema, manifest, bucketed streaming writer, DatasetView (pruning, take, stale-view guard), PDFs, eBOSS/DESI loaders, convert(). Fake 2 M DESI convert peaks 213 MiB anon RSS. Exit criterion met. |
| R2 | 2026-11-12 | f1fefea | 459 / 4 | 40 s (`-n 6`); 140 s single-process | Versioned sidecar (crash-safe, µs version ids), pair-wise blocked cross-match (brute-force exact), ONEUID with position hydration + dataset-version guard, links with ONEUID-version guard, Database. Row-identity regression green; 2×1 M cross-match 13 s; sidecar crash test green. |
| R3 | 2026-11-12 | e6813ff | 500 / 3 | 40 s (`-n 6`) | Weight primitives (full-name describe()), default registry, WeightedCatalog with ivar / best / mean / priority strategies; combine on 60 k rows ≈ 0.02 s. |
| R4 | 2026-11-13 | 8565495 | 577 / 1 | 50 s (`-n 6`) | measure: atoms, Product/FieldMap/Sightline, MeasurementSet (shared-region check), build_pointset (with_position, uniform randoms truncated to the data z range, adaptive footprint), photo-z attachment by position, tomographic n(z), presets, versioned save/load. Exit criterion met. |

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
- **R2 — identity guards.** ONEUID meta records each dataset's
  `DatasetView.version`; hydration and `build_links` refuse a dataset changed
  since the build ("rebuild the index"). Links record the ONEUID version they
  were built on; `load_links`/`chain` refuse a rebuilt ONEUID. A `Database`
  pinned with `as_of` is read-only.
- **R2 — `survey.kind` (decision for the owner to confirm).** R1 and R3 use
  `survey.kind` as the survey type (`"spectroscopic"`, `"photometric"`), so the
  plan's links guard `SubobjectRules(parent_kind="galaxy", child_kind="snia")`
  could never match a real loader. The guard was removed; `parent_kind` /
  `child_kind` are labels recorded in meta, and the `parents` / `children`
  dataset lists are authoritative. If an object-class notion is wanted, it
  needs a new manifest field (e.g. `SurveyMeta.objects`).
- **R2 — keeping hosts and SNe apart (provisional answer to the owner's
  question).** `CrossMatchRules(separate=(("hosts", "sne"),))` stops the ONEUID
  from merging two datasets, transitively. `build_links` raises if the ONEUID
  merged any parent with a child, unless `allow_self_merged=True`.
- **Risk for R8.** The default suite takes 140 s single-process (40 s with
  `-n 6`); `test/data` alone is ~110 s. The < 60 s single-process target needs
  slow-marking of the heavier writer/ONEUID tests or smaller fixtures.
- **R3 — combining values across surveys.** `combine(value, variance=<1-σ
  column>, strategy=…)`. "ivar" uses only 1/σ² (catalog weights are survey
  weights for measurements, not for a per-object value); "mean" uses catalog
  weights; "best" takes the smallest σ; "priority" takes the first dataset in a
  given order (usual for multi-pipeline QSO redshifts, e.g. DESI before eBOSS).
  Objects without a valid σ (eBOSS has no redshift error) keep a value with
  variance NaN instead of being dropped. The plan's algorithm (Σ w·ivar)
  contradicted its own test and was replaced.
- **Open for R4.** `combine()` returns no positions, so a combined
  `WeightedCatalog` cannot yet feed `build_pointset(view, …)`.
- **R4 — randoms and window.** Randoms are uniform inside window pixels,
  follow the weighted data n(z) truncated to the requested z range, carry
  weight 1 (do not re-apply the data weights to them), and obey the same
  cone/patch as the data. Without a survey mask the window is a data
  footprint at an adaptive nside (≥ 10 points per pixel), recorded in
  provenance; pass a real mask for science. An integer `regions` request is
  clamped to the number of masked pixels (recorded).
- **R4 — reproducibility.** Provenance records dataset version, z, nz_edges,
  weight, cone/patch, mask hash, random_factor, seed, ingested-randoms dataset
  and version, package version. `attach_photoz` refuses a view of another
  dataset version. `MeasurementSet` rejects products whose `region_id`
  disagrees with its `Regions`.
- **Notes for R6.** A simulation-derived `Product` must fill `source_row`,
  `partition`, `offset`, `region_id`; box positions go in extras — catalog `z`
  is redshift.
