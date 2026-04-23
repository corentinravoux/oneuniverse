# Phase 9 — DESI DR1 QSO Onboarding Fragility Audit

Running punch-list of fragilities surfaced by end-to-end onboarding tests
(`test/test_desi_dr1_onboarding.py`, `test/test_visual_desi_dr1.py`,
`scripts/onboard_desi_dr1_qso.py`). Formalised during Task 9; each item
gets a fix + regression test in Task 10.

Severity: S1 blocks onboarding, S2 degrades it, S3 cosmetic.

## Open

### F1 (S1) — Converter rejects DESIQSOLoader output: missing `z_err`, `_healpix32`

- **Where:** `convert_survey("desi_qso", raw_path=..., output_dir=..., overwrite=True)` → `write_ouf_dataset` at `oneuniverse/data/converter.py:131`.
- **Error:** `ValueError: data_df missing required columns: ['z_err', '_healpix32']`
- **Root cause:**
  1. `DESIQSOLoader._COLUMN_MAP` maps FITS `ZERR` → `z_spec_err`; CORE schema requires `z_err`. Loader never promotes spectroscopic `z_spec_err` into the generic `z_err` column.
  2. `_healpix32` is a partition key computed from `(ra, dec)`. Converter expects it to exist on the DataFrame before validation but neither the loader nor `write_ouf_dataset`'s pre-check derives it.
- **Surfaced by:** `test_convert_and_reread`.
- **Fix direction:** either (a) have `DESIQSOLoader._load_raw` alias `z_err = z_spec_err` when CORE group is active and let `write_ouf_dataset` auto-compute `_healpix32` from `ra`/`dec`, or (b) make the converter itself materialise both before validation. Prefer (b) — it closes the gap for every loader, not just DESI.

## Closed

_(none yet — populated as fixes land in Task 10.)_
