# Phase 9 — DESI DR1 QSO Onboarding Fragility Audit

Running punch-list of fragilities surfaced by end-to-end onboarding tests
(`test/test_desi_dr1_onboarding.py`, `test/test_visual_desi_dr1.py`,
`scripts/onboard_desi_dr1_qso.py`). Formalised during Task 9; each item
gets a fix + regression test in Task 10.

Severity: S1 blocks onboarding, S2 degrades it, S3 cosmetic.

## Open

_(none currently — see Closed.)_

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
