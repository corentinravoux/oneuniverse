# Phase 9 — DESI DR1 QSO Onboarding + Fragility Audit

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Onboard the DESI DR1 QSO catalog end-to-end through the full Pillar-1 stack (loader → converter → DatasetView → ONEUID → sub-object/temporal stubs → WeightedCatalog → diagnostic figures) with a single orchestration script and `pytest` harness, then *surface and fix* every fragility the run exposes.

**Architecture:** Treat DR1 QSO as the first **real** dataset flowing through OUF 2.1. A new `Packages/oneuniverse/scripts/onboard_desi_dr1_qso.py` drives the pipeline. Parallel `test/test_desi_dr1_onboarding.py` uses a **synthetic DR1-like fixture** (correct columns, realistic ZWARN/BAL values, ~5k rows) when the real file is absent, so the pipeline is testable on CI. Task 9 ("fragility audit") is deliberately open-ended: collect a list of issues during Tasks 1–8 and fix them one commit at a time in Task 10.

**Tech Stack:** oneuniverse (existing DESIQSOLoader, converter, DatasetView, ONEUID, combine), fitsio, pandas, pyarrow, matplotlib, healpy.

---

## File Structure

- Create: `Packages/oneuniverse/scripts/onboard_desi_dr1_qso.py` — standalone driver.
- Create: `Packages/oneuniverse/test/fixtures/desi_dr1_like.py` — synthetic DR1-shape fixture factory.
- Create: `Packages/oneuniverse/test/test_desi_dr1_onboarding.py` — end-to-end pytest.
- Create: `Packages/oneuniverse/test/test_output/phase9_desi_dr1_*.png` — diagnostic figures.
- Create: `Packages/oneuniverse/plans/2026-04-23-phase9-fragilities.md` — running punch-list (updated during Task 9).
- Modify: `oneuniverse/data/surveys/spectroscopic/desi_qso/loader.py` — already exists; audit for fragilities only.
- Modify (as needed from Task 10 audit): converter / manifest / schema / combine / database.

---

### Task 1: Synthetic DR1-like fixture

**Files:**
- Create: `test/fixtures/desi_dr1_like.py`
- Test: `test/test_desi_dr1_fixture.py`

**Why:** The onboarding pipeline must be runnable without the 2.33 M-row `QSO_full.dat.fits` present. The fixture matches every column the loader reads (see [`loader.py` `_FITS_COLUMNS`](../oneuniverse/data/surveys/spectroscopic/desi_qso/loader.py)) with realistic dtypes and distributions.

- [ ] **Step 1: Write failing test**

```python
# test/test_desi_dr1_fixture.py
from pathlib import Path
from test.fixtures.desi_dr1_like import write_fake_desi_dr1_fits

def test_fake_fits_has_all_loader_columns(tmp_path):
    from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import (
        _FITS_COLUMNS,
    )
    out = write_fake_desi_dr1_fits(tmp_path, n_rows=200, seed=0)
    import fitsio
    with fitsio.FITS(out) as f:
        cols = set(f[1].get_colnames())
    for c in _FITS_COLUMNS:
        assert c in cols, f"fixture missing {c!r}"
```

Run: `pytest test/test_desi_dr1_fixture.py -v` → FAIL (fixture module absent).

- [ ] **Step 2: Implement fixture**

```python
# test/fixtures/desi_dr1_like.py
"""Minimal DESI DR1 QSO-shaped FITS writer for tests."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import fitsio


def write_fake_desi_dr1_fits(
    out_dir: Path, n_rows: int = 5000, seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    ra = rng.uniform(120.0, 260.0, n_rows)
    dec = rng.uniform(-10.0, 60.0, n_rows)
    z = rng.uniform(0.8, 3.5, n_rows)
    zerr = rng.uniform(1e-4, 1e-3, n_rows).astype("f8")

    spectype = np.where(rng.uniform(size=n_rows) < 0.9, "QSO", "GALAXY")
    spectype = np.asarray(spectype, dtype="<U6")
    subtype = np.where(z > 2.1, "LYA", "HIZ").astype("<U8")
    zwarn = np.where(rng.uniform(size=n_rows) < 0.98, 0, 4).astype("i4")
    targetid = (39_627_000_000 + np.arange(n_rows)).astype("i8")

    arr = {
        "RA": ra, "DEC": dec, "Z_RR": z, "ZERR": zerr, "ZWARN": zwarn,
        "SPECTYPE": spectype, "SUBTYPE": subtype,
        "DELTACHI2": rng.uniform(10.0, 300.0, n_rows).astype("f8"),
        "TARGETID": targetid,
        "Z_QN": z + rng.normal(0, 1e-3, n_rows).astype("f8"),
        "Z_QN_CONF": rng.uniform(0.5, 1.0, n_rows).astype("f8"),
        "IS_QSO_QN": np.ones(n_rows, dtype="i2"),
        "NTILE": rng.integers(1, 4, n_rows).astype("i4"),
        "FIBER": rng.integers(0, 5000, n_rows).astype("i4"),
        "FLUX_G": rng.uniform(0.1, 20.0, n_rows).astype("f4"),
        "FLUX_R": rng.uniform(0.1, 20.0, n_rows).astype("f4"),
        "FLUX_Z": rng.uniform(0.1, 20.0, n_rows).astype("f4"),
        "FLUX_IVAR_G": rng.uniform(1.0, 500.0, n_rows).astype("f4"),
        "FLUX_IVAR_R": rng.uniform(1.0, 500.0, n_rows).astype("f4"),
        "FLUX_IVAR_Z": rng.uniform(1.0, 500.0, n_rows).astype("f4"),
        "FLUX_W1": rng.uniform(0.1, 20.0, n_rows).astype("f4"),
        "FLUX_W2": rng.uniform(0.1, 20.0, n_rows).astype("f4"),
        "FLUX_IVAR_W1": rng.uniform(1.0, 500.0, n_rows).astype("f4"),
        "FLUX_IVAR_W2": rng.uniform(1.0, 500.0, n_rows).astype("f4"),
        "MW_TRANSMISSION_G": rng.uniform(0.7, 1.0, n_rows).astype("f4"),
        "MW_TRANSMISSION_R": rng.uniform(0.7, 1.0, n_rows).astype("f4"),
        "MW_TRANSMISSION_Z": rng.uniform(0.7, 1.0, n_rows).astype("f4"),
        "MW_TRANSMISSION_W1": rng.uniform(0.9, 1.0, n_rows).astype("f4"),
        "MW_TRANSMISSION_W2": rng.uniform(0.9, 1.0, n_rows).astype("f4"),
        "EBV": rng.uniform(0.0, 0.3, n_rows).astype("f4"),
        "TSNR2_QSO": rng.uniform(1.0, 100.0, n_rows).astype("f4"),
        "TSNR2_LYA": rng.uniform(0.0, 50.0, n_rows).astype("f4"),
        "COADD_NUMEXP": rng.integers(1, 8, n_rows).astype("i4"),
        "COADD_EXPTIME": rng.uniform(1000.0, 4000.0, n_rows).astype("f4"),
        "COADD_NUMNIGHT": rng.integers(1, 3, n_rows).astype("i4"),
        "WEIGHT_ZFAIL": rng.uniform(0.95, 1.05, n_rows).astype("f8"),
        "COMP_TILE": rng.uniform(0.8, 1.0, n_rows).astype("f8"),
        "FRACZ_TILELOCID": rng.uniform(0.5, 1.0, n_rows).astype("f8"),
        "FRAC_TLOBS_TILES": rng.uniform(0.5, 1.0, n_rows).astype("f8"),
        "PROB_OBS": rng.uniform(0.5, 1.0, n_rows).astype("f8"),
        "DESI_TARGET": rng.integers(1, 1 << 20, n_rows).astype("i8"),
        "MORPHTYPE": np.full(n_rows, "PSF", dtype="<U4"),
        "PHOTSYS": np.where(dec > 32.0, "N", "S").astype("<U1"),
    }
    out = out_dir / "QSO_full.dat.fits"
    fitsio.write(str(out), arr, clobber=True)
    return out
```

Run: `pytest test/test_desi_dr1_fixture.py -v` → PASS.

- [ ] **Step 3: Commit**

```bash
git add test/fixtures/desi_dr1_like.py test/test_desi_dr1_fixture.py
git commit -m "test(desi_dr1): synthetic DR1-shaped QSO FITS fixture"
```

---

### Task 2: Loader→DataFrame smoke test

**Files:**
- Test: `test/test_desi_dr1_onboarding.py` (create; first test only)

- [ ] **Step 1: Write failing test**

```python
# test/test_desi_dr1_onboarding.py
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.desi_dr1_like import write_fake_desi_dr1_fits

from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import (
    DESIQSOLoader,
)


def test_loader_reads_fake_dr1(tmp_path):
    write_fake_desi_dr1_fits(tmp_path, n_rows=500, seed=1)
    df = DESIQSOLoader()._load_raw(data_path=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert {"ra", "dec", "z", "z_spec_err", "zwarning"}.issubset(df.columns)
    assert (df["zwarning"] == 0).all()      # good_zwarn default
    assert (df["z"] > 0).all()
```

Run: `pytest test/test_desi_dr1_onboarding.py::test_loader_reads_fake_dr1 -v` → decide: PASS or FAIL. **Any failure is recorded as fragility #F1 in `plans/2026-04-23-phase9-fragilities.md` and fixed in Task 10.**

- [ ] **Step 2: Commit (regardless of pass/fail — failures drive Task 10)**

```bash
git add test/test_desi_dr1_onboarding.py
git commit -m "test(desi_dr1): loader smoke test against fake DR1 FITS"
```

---

### Task 3: Convert to OUF POINT + re-read

**Files:**
- Modify: `test/test_desi_dr1_onboarding.py` (add one test)

- [ ] **Step 1: Write failing test**

```python
def test_convert_and_reread(tmp_path):
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    write_fake_desi_dr1_fits(raw_dir, n_rows=1000, seed=2)

    from oneuniverse.data.converter import convert_point_survey
    from oneuniverse.data.dataset_view import DatasetView
    from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import (
        DESIQSOLoader,
    )

    ou_dir = tmp_path / "db" / "desi_dr1_qso"
    convert_point_survey(
        loader=DESIQSOLoader(),
        survey_path=ou_dir,
        data_path=raw_dir,
    )

    view = DatasetView.from_path(ou_dir)
    df = view.read(columns=["ra", "dec", "z", "z_type"])
    assert len(df) > 0
    assert set(df["z_type"].unique()) <= {"spec"}
    assert df["ra"].between(0, 360).all()
```

Run → fragility notes for any failure.

- [ ] **Step 2: Commit**

```bash
git add test/test_desi_dr1_onboarding.py
git commit -m "test(desi_dr1): POINT conversion + DatasetView round-trip"
```

---

### Task 4: HEALPix partition + cone query

**Files:**
- Modify: `test/test_desi_dr1_onboarding.py`

- [ ] **Step 1: Add cone-query test**

```python
def test_cone_query_prunes_partitions(tmp_path):
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    write_fake_desi_dr1_fits(raw_dir, n_rows=5000, seed=3)

    from oneuniverse.data.converter import convert_point_survey
    from oneuniverse.data.dataset_view import DatasetView, Cone
    from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import DESIQSOLoader

    ou_dir = tmp_path / "db" / "desi_dr1_qso"
    convert_point_survey(DESIQSOLoader(), ou_dir, raw_dir)

    view = DatasetView.from_path(ou_dir)
    cone = Cone(ra=180.0, dec=20.0, radius_deg=5.0)
    df = view.read(cone=cone, columns=["ra", "dec"])
    assert len(df) > 0
    # all returned rows must be inside the cone
    from oneuniverse.data.dataset_view import _angular_sep_deg  # or rewrite inline
    seps = _angular_sep_deg(df["ra"].to_numpy(), df["dec"].to_numpy(), 180.0, 20.0)
    assert (seps <= 5.0 + 1e-6).all()
```

Run. If `_angular_sep_deg` isn't importable, replace with haversine inline. Record as fragility (private helper not re-used).

- [ ] **Step 2: Commit**

```bash
git add test/test_desi_dr1_onboarding.py
git commit -m "test(desi_dr1): cone query on HEALPix partitions"
```

---

### Task 5: ONEUID self-index (one-dataset degenerate)

**Files:**
- Modify: `test/test_desi_dr1_onboarding.py`

A single-dataset ONEUID is a degenerate but useful smoke test: every row gets a unique oneuid 0..N-1.

- [ ] **Step 1: Add test**

```python
def test_oneuid_single_dataset(tmp_path):
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    write_fake_desi_dr1_fits(raw_dir, n_rows=1200, seed=4)

    from oneuniverse.data.converter import convert_point_survey
    from oneuniverse.data.database import OneuniverseDatabase
    from oneuniverse.data.oneuid_rules import CrossMatchRules
    from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import DESIQSOLoader

    db_root = tmp_path / "db"; db_root.mkdir()
    convert_point_survey(DESIQSOLoader(), db_root / "desi_dr1_qso", raw_dir)

    db = OneuniverseDatabase(db_root)
    db.build_oneuid(
        datasets=["desi_dr1_qso"],
        rules=CrossMatchRules(sky_tol_arcsec=0.5),
        name="default",
    )
    idx = db.load_oneuid("default")
    assert idx.table["oneuid"].nunique() == len(idx.table)
    assert set(idx.table["dataset"].unique()) == {"desi_dr1_qso"}
```

Run. Record fragilities (categorical dtype round-trip, survey_type propagation).

- [ ] **Step 2: Commit**

```bash
git add test/test_desi_dr1_onboarding.py
git commit -m "test(desi_dr1): ONEUID self-index on converted dataset"
```

---

### Task 6: WeightedCatalog.fill_defaults

**Files:**
- Modify: `test/test_desi_dr1_onboarding.py`

- [ ] **Step 1: Add test**

```python
def test_weighted_catalog_defaults(tmp_path):
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    write_fake_desi_dr1_fits(raw_dir, n_rows=800, seed=5)

    from oneuniverse.data.converter import convert_point_survey
    from oneuniverse.data.database import OneuniverseDatabase
    from oneuniverse.data.oneuid_rules import CrossMatchRules
    from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import DESIQSOLoader
    from oneuniverse.combine import WeightedCatalog

    db_root = tmp_path / "db"; db_root.mkdir()
    convert_point_survey(DESIQSOLoader(), db_root / "desi_dr1_qso", raw_dir)
    db = OneuniverseDatabase(db_root)
    db.build_oneuid(
        datasets=["desi_dr1_qso"],
        rules=CrossMatchRules(sky_tol_arcsec=0.5),
        name="default",
    )
    idx = db.load_oneuid("default")
    wc = WeightedCatalog.from_oneuid(idx, db)
    wc.fill_defaults(db)

    summary = wc.summary()
    assert "desi_dr1_qso" in summary
    meas = wc.as_measurements()
    assert "desi_dr1_qso" in meas.weights
    import numpy as np
    w = meas.weights["desi_dr1_qso"]
    assert np.all(np.isfinite(w)) and np.all(w > 0)
```

Run. Record fragilities: missing `z_spec_err` handling, ambiguous default weight for QSO survey_type.

- [ ] **Step 2: Commit**

```bash
git add test/test_desi_dr1_onboarding.py
git commit -m "test(desi_dr1): WeightedCatalog.fill_defaults end-to-end"
```

---

### Task 7: Diagnostic figures

**Files:**
- Create: `test/test_visual_desi_dr1.py`

Produces three figures under `test/test_output/`: (a) sky density, (b) z-histogram, (c) FLUX_R vs z colored by DESI_TARGET bitmask.

- [ ] **Step 1: Write failing test**

```python
# test/test_visual_desi_dr1.py
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.desi_dr1_like import write_fake_desi_dr1_fits  # noqa: E402

from oneuniverse.data.converter import convert_point_survey  # noqa: E402
from oneuniverse.data.dataset_view import DatasetView  # noqa: E402
from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import (  # noqa: E402
    DESIQSOLoader,
)


OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_desi_dr1_sky_and_z(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    write_fake_desi_dr1_fits(raw, n_rows=5000, seed=9)
    ou = tmp_path / "db" / "desi_dr1_qso"
    convert_point_survey(DESIQSOLoader(), ou, raw)
    df = DatasetView.from_path(ou).read(columns=["ra", "dec", "z"])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5))
    axa.scatter(df["ra"], df["dec"], s=2, alpha=0.3)
    axa.set_xlabel("RA [deg]"); axa.set_ylabel("Dec [deg]")
    axa.set_title(f"DESI DR1 QSO (fake) sky — n={len(df)}")
    axb.hist(df["z"], bins=60, color="C3", alpha=0.8)
    axb.set_xlabel("z"); axb.set_ylabel("N")
    axb.set_title("redshift histogram")
    fig.tight_layout()
    out = OUT / "phase9_desi_dr1_sky_z.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    assert out.exists() and out.stat().st_size > 10_000
```

- [ ] **Step 2: Run**: `pytest test/test_visual_desi_dr1.py -v` → PASS.
- [ ] **Step 3: Commit**

```bash
git add test/test_visual_desi_dr1.py
git commit -m "test(desi_dr1): sky + z diagnostic figures"
```

---

### Task 8: Standalone onboarding script

**Files:**
- Create: `scripts/onboard_desi_dr1_qso.py`

- [ ] **Step 1: Write script**

```python
#!/usr/bin/env python
"""End-to-end onboarding driver for DESI DR1 QSO.

Usage:
    python scripts/onboard_desi_dr1_qso.py \
        --raw-root /path/to/raw \
        --db-root  /path/to/oneuniverse/db

When --raw-root points at real DESI DR1 data, uses QSO_full.dat.fits;
otherwise generates a synthetic DR1-shaped FITS under raw-root.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("onboard_desi_dr1_qso")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--db-root", type=Path, required=True)
    p.add_argument("--fake-if-missing", action="store_true")
    p.add_argument("--fake-n-rows", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from oneuniverse.data.converter import convert_point_survey
    from oneuniverse.data.database import OneuniverseDatabase
    from oneuniverse.data.oneuid_rules import CrossMatchRules
    from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import (
        DESIQSOLoader,
    )
    from oneuniverse.combine import WeightedCatalog

    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.db_root.mkdir(parents=True, exist_ok=True)

    fits_path = args.raw_root / DESIQSOLoader.config.data_filename
    if not fits_path.exists():
        if not args.fake_if_missing:
            LOG.error("FITS not found: %s", fits_path)
            return 2
        LOG.warning("Generating synthetic fixture at %s", fits_path)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))
        from fixtures.desi_dr1_like import write_fake_desi_dr1_fits
        write_fake_desi_dr1_fits(
            args.raw_root, n_rows=args.fake_n_rows, seed=args.seed,
        )

    ou = args.db_root / "desi_dr1_qso"
    LOG.info("Converting -> %s", ou)
    convert_point_survey(DESIQSOLoader(), ou, args.raw_root)

    db = OneuniverseDatabase(args.db_root)
    LOG.info("Building ONEUID index 'default' …")
    db.build_oneuid(
        datasets=["desi_dr1_qso"],
        rules=CrossMatchRules(sky_tol_arcsec=0.5),
        name="default",
    )
    idx = db.load_oneuid("default")

    wc = WeightedCatalog.from_oneuid(idx, db)
    wc.fill_defaults(db, skip_unknown=True)
    LOG.info("WeightedCatalog summary:\n%s", wc.summary())

    LOG.info("DONE — database root: %s", args.db_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run** (synthetic): `python scripts/onboard_desi_dr1_qso.py --raw-root /tmp/raw --db-root /tmp/db --fake-if-missing --fake-n-rows 20000`. Expected: exit 0; `/tmp/db/desi_dr1_qso/` populated; `/tmp/db/_oneuid/default.parquet` exists. Record every warning or non-fatal error as fragility.
- [ ] **Step 3: Commit**

```bash
git add scripts/onboard_desi_dr1_qso.py
git commit -m "scripts(desi_dr1): end-to-end onboarding driver"
```

---

### Task 9: Write the fragility audit

**Files:**
- Create: `plans/2026-04-23-phase9-fragilities.md`

During Tasks 1–8 you collected every error, warning, off-by-one, silent coercion, missing guard, deprecated API, or awkward ergonomic paper-cut. Turn that list into an ordered punch-list.

- [ ] **Step 1: Write the fragility doc**

Template:

```markdown
# Phase 9 Fragility Audit — DESI DR1 QSO Onboarding (2026-04-23)

Each item: ID | severity (S1/S2/S3) | file:line | description | fix sketch.

## S1 — blockers (onboarding did not succeed as specified)
- [ ] F1 …

## S2 — soft failures (completed but wrong, confusing, or silently lossy)
- [ ] F2 …

## S3 — ergonomic / cosmetic
- [ ] F3 …

## Decision log
- …
```

Known seed items (verify during execution, move to the lists above):
- `_load_raw` enforces `survey_type="spectroscopic"` but DESIQSOLoader hardcodes `z_type=0` (int8) rather than the schema's `"spec"` string — may clash with schema validation.
- `desi_qso` loader adds `z_spec` (f4) but schema expects `z_spec_err` to be present too; default-weight registry needs `z_spec_err` to build the IVar weight.
- Converter behaviour on empty-after-cut DataFrames (all rows fail ZWARN).
- `fill_defaults` on a QSO survey_type — which pair does `default_weight_for("spectroscopic", "spec")` resolve to? May need a QSO-aware default.

- [ ] **Step 2: Commit**

```bash
git add plans/2026-04-23-phase9-fragilities.md
git commit -m "docs(phase9): fragility audit from DR1 onboarding"
```

---

### Task 10: Fix fragilities, one commit per fix

For each unchecked item in the audit:

- [ ] **Step 1: Write a regression test that reproduces the fragility.**
- [ ] **Step 2: Implement the minimal fix.**
- [ ] **Step 3: Re-run the onboarding script — must still exit 0.**
- [ ] **Step 4: Tick the audit box and commit** with message `fix(phase9/F<N>): <short>`.

Do S1s first, then S2s, then S3s.

Definition of done: all boxes ticked, `pytest -q` green, onboarding script exits 0 on both synthetic and (if available) real DR1 data.

---

### Task 11: Close Phase 9 — update status table + memory

- [ ] **Step 1:** Add row in `plans/README.md` status table: `| 9 | DESI DR1 QSO onboarding + fragility audit | **complete (YYYY-MM-DD, N/N tests green)** |`.
- [ ] **Step 2:** Update memory `project_oneuniverse_stabilisation.md` with Phase 9 block — list fragilities fixed and new commits.
- [ ] **Step 3:** `git add plans/README.md && git commit -m "docs(plans): mark Phase 9 complete"`.

---

## Self-Review Notes (author of the plan)

Spec coverage: onboarding step (Tasks 2-8) covers loader → converter → DatasetView cone query → ONEUID → WeightedCatalog defaults → figures → standalone script. Fragility audit (Task 9) + fix loop (Task 10) is explicitly open-ended — this is intentional because we don't know what will break until we run.

Non-placeholders: every test body is given in full; commit messages prescribed; script is complete; synthetic fixture is complete.

Type consistency: `DESIQSOLoader` + `convert_point_survey` + `DatasetView.from_path` + `OneuniverseDatabase.build_oneuid` + `WeightedCatalog.from_oneuid` / `fill_defaults` names all match current code (see `oneuniverse/data/*.py`, `oneuniverse/combine/*.py`).
