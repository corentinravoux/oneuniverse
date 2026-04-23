# Phase 9 — DESI DR1 QSO Onboarding Fragility Audit

Running punch-list of fragilities surfaced by end-to-end onboarding tests
(`test/test_desi_dr1_onboarding.py`, `test/test_visual_desi_dr1.py`,
`scripts/onboard_desi_dr1_qso.py`). Formalised during Task 9; each item
gets a fix + regression test in Task 10.

Severity: S1 blocks onboarding, S2 degrades it, S3 cosmetic.

## Open

### F3 (S3) — Over-partitioning on small catalogs inflates file count and kills compression

- **Where:** `oneuniverse/data/converter.py` `_write_partitions_by_healpix`; hit any time `len(df) / n_healpix_cells` is small.
- **Symptom:** Synthetic 17 618-row run produces **2 551** `part_*.parquet` files (~7 rows per file); on-disk size 91.8 MB for a 4.6 MB FITS input (0.1× compression, parquet header dominates).
- **Root cause:** every populated NSIDE=32 cell gets its own directory + part file regardless of population. Fine for DR1 in full (2.3 M rows) but embarrassing for small / down-sampled runs.
- **Fix direction:** optional coarsening — if `len(df) < min_rows_per_cell * n_cells`, fall back to a single-cell partitioning or a coarser NSIDE (16/8). Keep the default behaviour intact for full DR1 so existing callers don't see a change.
- **Surfaced by:** Task 8 (onboarding script run on 20 000-row fake catalog).
- **Priority:** deferred — not blocking Phase 9 closure. Tracked for Phase 11/12 housekeeping.

## Closed

### F1 (S1) — Converter rejects loader output: missing `z_err`, `_healpix32`

- **Where:** `convert_survey("desi_qso", …)` → `write_ouf_dataset` at `oneuniverse/data/converter.py:131`.
- **Symptom:** `ValueError: data_df missing required columns: ['z_err', '_healpix32']`.
- **Root cause:** DESI loader maps `ZERR`→`z_spec_err` but never materialises CORE `z_err`; partition key `_healpix32` was nobody's responsibility before `write_ouf_dataset`.
- **Fix:** both promoted to the converter itself so every POINT loader benefits — `convert_survey` now aliases `z_spec_err`/`z_phot_err`→`z_err` and computes `_healpix32` via `healpy.ang2pix(NSIDE=32, nest=True)` from `ra`/`dec` before validation.
- **Regression test:** `test/test_desi_dr1_onboarding.py::test_convert_and_reread`.
- **Surfaced by:** Task 3.

### F2 (S1) — DESI loader writes `z_type` as int8(0) instead of `"spec"`

- **Where:** `oneuniverse/data/surveys/spectroscopic/desi_qso/loader.py` L224 (pre-fix).
- **Symptom:** After conversion, `df["z_type"].unique() == np.array([0], dtype=int8)`; downstream `WeightedCatalog.fill_defaults` / z-type rules require the string `"spec"` (see `schema.Z_TYPE_VALUES`).
- **Fix:** loader now writes `np.full(n, "spec", dtype="<U4")`.
- **Regression test:** `test/test_desi_dr1_onboarding.py::test_convert_and_reread` asserts `set(df["z_type"].unique()) <= {"spec"}`.
- **Surfaced by:** Task 3 (second run, after F1 fix).
