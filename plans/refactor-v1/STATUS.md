# Refactor v1 — status

One row per step, written when the step's exit criterion is met. Deviations
from the design are recorded under the table with their reason.

| step | date | commit | tests (passed/skipped) | default suite time | notes |
|---|---|---|---|---|---|
| R0 | 2026-11-12 | d058aeb | 931 / 12 | 51 s (`-n 6`); 118 s single-process | `_core` (errors, registry, fs, geometry, healpix) + guards + CI; old suite still green. Single-process time is the old suite's; the < 60 s target is R8's exit criterion. CI has not yet run on the branch (not pushed). |
| R1 T1 | 2026-11-12 | 0ba25b8 | 273 / 5 | 15 s (`-n 6`) | old data/combine/measure/twin removed |

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
