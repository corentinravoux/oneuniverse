# Phase 21 — Cleanup of Deferred Sub-Object Items Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the three items deferred from Phase 20 by:
(1) widening `CrossMatchRules` with pluggable `attribute_filters`,
(2) adding a CORE-schema `composite_id: U64` column so surveys with
composite IDs (`PLATE-MJD-FIBERID`, `KIDS_TILE+SeqNr`,
GAIA `source_id`-decoded) can preserve the published form alongside
the canonical int64 `galaxy_id`, and (3) shipping a
`mocpy`-backed multi-order MOC → fixed-NSIDE HEALPix rasteriser so
GW LIGO/Virgo BAYESTAR / LALInference outputs can flow into the
existing `build_subobject_links_to_map` builder.

**Architecture:** Three small, independent extensions, each in its
own file. `CrossMatchRules` gains an `attribute_filters` tuple field
whose entries are picklable callables; the ONEUID cross-matcher loop
already iterates candidate pairs and applies dz / reject filters —
attribute filters slot in immediately after dz. `composite_id` is a
new optional CORE column (no required-set change). MOC support lives
in a new `oneuniverse.data.moc` module that imports `mocpy` lazily
with a clean error message if missing. No OUF bump.

**Tech Stack:** Python 3.9+, numpy, healpy (already in use), pandas,
dataclasses, pytest. Optional new dep: `mocpy` (added to `[dev]`
extras only).

---

## File Structure

**New files:**
- `oneuniverse/data/moc.py` — `rasterise_moc_to_healpix(...)`.
- `test/test_attribute_filters.py` — `CrossMatchRules.attribute_filters` round-trip + matcher integration.
- `test/test_composite_id_column.py` — schema integration.
- `test/test_moc_rasterise.py` — MOC → fixed-NSIDE rasterisation (skipped when `mocpy` is missing).
- `test/test_visual_phase21.py` — diagnostic figure.

**Modified files:**
- `oneuniverse/data/oneuid_rules.py` — `CrossMatchRules.attribute_filters` field + hash entry.
- `oneuniverse/data/oneuid_crossmatch.py` — apply `attribute_filters` after dz cut.
- `oneuniverse/data/schema.py` — add `composite_id` to CORE.
- `pyproject.toml` — `mocpy>=0.13` in `[dev]` extra (optional).
- `oneuniverse/CLAUDE.md`, `plans/README.md`,
  `research/schema_generalisation_audit.md` — Phase 21 close-out cross-refs.

---

## Pre-flight

- [ ] **Step 0: Baseline.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest -q 2>&1 | tail -3
```

Expected: `487 passed, 1 skipped` (Phase 20 baseline).

---

## Task 1: `CrossMatchRules.attribute_filters`

**Files:**
- Modify: `oneuniverse/data/oneuid_rules.py`
- Create: `test/test_attribute_filters.py`

- [ ] **Step 1: Failing test**

```python
# test/test_attribute_filters.py
"""Phase 21 T1 — CrossMatchRules.attribute_filters."""
import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.oneuid_rules import CrossMatchRules


def _color_filter(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    """Keep pair iff |Δ(g-r)| < 0.1."""
    dg = (left["psfmag_g"] - left["psfmag_r"]).to_numpy()
    dr = (right["psfmag_g"] - right["psfmag_r"]).to_numpy()
    return np.abs(dg - dr) < 0.1


def test_default_attribute_filters_is_empty():
    r = CrossMatchRules()
    assert r.attribute_filters == ()


def test_attribute_filters_tuple_stored():
    r = CrossMatchRules(attribute_filters=(_color_filter,))
    assert r.attribute_filters == (_color_filter,)


def test_hash_includes_attribute_filters():
    a = CrossMatchRules()
    b = CrossMatchRules(attribute_filters=(_color_filter,))
    assert a.hash() != b.hash()


def test_attribute_filters_must_be_tuple_of_callables():
    with pytest.raises(TypeError, match="callable"):
        CrossMatchRules(attribute_filters=("not_callable",))
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest test/test_attribute_filters.py -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'attribute_filters'`.

- [ ] **Step 3: Extend `CrossMatchRules`**

In `oneuniverse/data/oneuid_rules.py`, add the field + hash entry +
validation:

```python
@dataclass(frozen=True, eq=False)
class CrossMatchRules:
    sky_tol_arcsec: float = 1.0
    dz_tol_default: Optional[float] = 1e-3
    dz_tol_by_ztype: Mapping[ZtypePair, float] = field(default_factory=dict)
    reject_ztype: FrozenSet[ZtypePair] = field(default_factory=frozenset)
    # Phase 21: pluggable predicates evaluated on candidate pairs.
    # Each callable receives two pandas DataFrames of equal length
    # (one row per candidate pair, left + right) and returns a
    # length-N bool array — True = keep.
    attribute_filters: Tuple[Callable, ...] = ()

    def __post_init__(self) -> None:
        norm_dz = {self._key(*k): v for k, v in dict(self.dz_tol_by_ztype).items()}
        norm_rej = frozenset(self._key(*p) for p in self.reject_ztype)
        for f in self.attribute_filters:
            if not callable(f):
                raise TypeError(
                    f"CrossMatchRules.attribute_filters: expected "
                    f"callables, got {f!r}"
                )
        object.__setattr__(self, "dz_tol_by_ztype", norm_dz)
        object.__setattr__(self, "reject_ztype", norm_rej)
        object.__setattr__(
            self, "attribute_filters", tuple(self.attribute_filters),
        )
```

Add `Callable, Tuple` to the imports.

Update `_canonical` to include attribute-filter identities (use
qualified names; callables are not JSON-serialisable):

```python
    def _canonical(self) -> dict:
        return {
            "sky_tol_arcsec": self.sky_tol_arcsec,
            "dz_tol_default": self.dz_tol_default,
            "dz_tol_by_ztype": sorted(
                [list(self._key(*k)), v]
                for k, v in self.dz_tol_by_ztype.items()
            ),
            "reject_ztype": sorted(
                list(self._key(*k)) for k in self.reject_ztype
            ),
            "attribute_filters": [
                f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', repr(f))}"
                for f in self.attribute_filters
            ],
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_attribute_filters.py test/test_oneuid_rules.py -q
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/oneuid_rules.py test/test_attribute_filters.py
git commit -m "phase21/T1: CrossMatchRules.attribute_filters (pluggable per-pair predicates, hashed by qualname)"
```

---

## Task 2: ONEUID cross-matcher applies `attribute_filters`

**Files:**
- Modify: `oneuniverse/data/oneuid_crossmatch.py`
- Extend: `test/test_attribute_filters.py`

- [ ] **Step 1: Append matcher-integration tests**

Append to `test/test_attribute_filters.py`:

```python
from oneuniverse.data.oneuid_crossmatch import cross_match_surveys


def _build_table():
    """Two surveys, two objects each, all within 0.1 arcsec.

    Survey 'a' rows have g - r = 0.0 and 1.0.
    Survey 'b' rows have g - r = 0.05 and 0.9.

    With the colour filter (|Δ(g-r)| < 0.1) only matching colours
    cross-match.
    """
    return pd.DataFrame({
        "ra":  [10.0, 20.0, 10.000001, 20.000001],
        "dec": [0.0, 0.0, 0.0, 0.0],
        "z":   [0.5, 0.5, 0.5, 0.5],
        "z_type": ["spec", "spec", "spec", "spec"],
        "z_err": [0.001, 0.001, 0.001, 0.001],
        "galaxy_id": [0, 1, 2, 3],
        "survey": ["a", "a", "b", "b"],
        "_original_row_index": [0, 1, 0, 1],
        "psfmag_g": [22.0, 22.0, 22.0, 22.0],
        "psfmag_r": [22.0, 21.0, 21.95, 21.1],
    })


def test_attribute_filter_blocks_color_mismatch():
    table = _build_table()
    rules = CrossMatchRules(
        sky_tol_arcsec=2.0,
        attribute_filters=(_color_filter,),
    )
    out = cross_match_surveys(table.copy(), rules)
    # Match by inspecting universal_id groupings:
    multi_survey_groups = (
        out.groupby("universal_id")["survey"].nunique() > 1
    )
    n_multi = int(multi_survey_groups.sum())
    # Only 1 multi-survey group survives the color filter
    # (the matching pair at RA=10, color≈0).
    assert n_multi == 1


def test_no_filter_keeps_both_matches():
    table = _build_table()
    rules = CrossMatchRules(sky_tol_arcsec=2.0)  # no attribute filter
    out = cross_match_surveys(table.copy(), rules)
    multi_survey_groups = (
        out.groupby("universal_id")["survey"].nunique() > 1
    )
    assert int(multi_survey_groups.sum()) == 2
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest test/test_attribute_filters.py -v
```

Expected: filter rule does nothing — both pairs survive.

- [ ] **Step 3: Apply filters in the matcher loop**

In `oneuniverse/data/oneuid_crossmatch.py`, right after the existing
dz-tolerance block (after the line ``idx2 = idx2[keep]``), insert:

```python
    # Phase 21: pluggable attribute filters.
    if idx1.size and rules.attribute_filters:
        left = table.iloc[idx1].reset_index(drop=True)
        right = table.iloc[idx2].reset_index(drop=True)
        keep = np.ones(idx1.size, dtype=bool)
        for f in rules.attribute_filters:
            mask = np.asarray(f(left, right), dtype=bool)
            if mask.shape != (idx1.size,):
                raise ValueError(
                    f"attribute_filter {f!r} must return bool[{idx1.size}], "
                    f"got shape {mask.shape}"
                )
            keep &= mask
        idx1 = idx1[keep]
        idx2 = idx2[keep]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_attribute_filters.py test/test_oneuid.py -q
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/oneuid_crossmatch.py test/test_attribute_filters.py
git commit -m "phase21/T2: ONEUID cross-matcher applies attribute_filters after dz cut"
```

---

## Task 3: `composite_id` CORE column

**Files:**
- Modify: `oneuniverse/data/schema.py`
- Create: `test/test_composite_id_column.py`

- [ ] **Step 1: Failing test**

```python
# test/test_composite_id_column.py
"""Phase 21 T3 — composite_id optional CORE column."""
import numpy as np
import pandas as pd

from oneuniverse.data.schema import (
    CORE_COLUMNS,
    get_all_columns,
    validate_dataframe,
)


def test_composite_id_in_core_columns():
    names = {c.name for c in CORE_COLUMNS}
    assert "composite_id" in names


def test_composite_id_is_optional_string():
    by_name = {c.name: c for c in CORE_COLUMNS}
    col = by_name["composite_id"]
    assert col.required is False
    assert col.dtype.startswith("U")


def test_dataframe_without_composite_id_still_validates():
    df = pd.DataFrame({
        "ra": np.array([0.0], dtype="f8"),
        "dec": np.array([0.0], dtype="f8"),
        "z": np.array([0.5], dtype="f4"),
        "z_type": np.array(["spec"], dtype=object),
        "z_err": np.array([0.001], dtype="f4"),
        "galaxy_id": np.array([0], dtype="i8"),
        "survey_id": np.array(["x"], dtype=object),
        "_original_row_index": np.array([0], dtype="i8"),
        "_healpix32": np.array([0], dtype="i4"),
    })
    assert validate_dataframe(df, ["core"]) == []


def test_dataframe_with_composite_id_string_validates():
    df = pd.DataFrame({
        "ra": np.array([0.0], dtype="f8"),
        "dec": np.array([0.0], dtype="f8"),
        "z": np.array([0.5], dtype="f4"),
        "z_type": np.array(["spec"], dtype=object),
        "z_err": np.array([0.001], dtype="f4"),
        "galaxy_id": np.array([0], dtype="i8"),
        "survey_id": np.array(["x"], dtype=object),
        "_original_row_index": np.array([0], dtype="i8"),
        "_healpix32": np.array([0], dtype="i4"),
        "composite_id": np.array(["3551-55065-0010"], dtype=object),
    })
    assert validate_dataframe(df, ["core"]) == []
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest test/test_composite_id_column.py -v
```

Expected: `composite_id` not in CORE_COLUMNS.

- [ ] **Step 3: Add the column**

In `oneuniverse/data/schema.py`, in the `CORE_COLUMNS` tuple, after
the existing `survey_id` entry add:

```python
    ColumnDef("composite_id", "U64", "",
              "Survey-published composite ID (PLATE-MJD-FIBERID, "
              "KIDS_TILE+SeqNr, GAIA source_id, …); optional",
              required=False),
```

(Keep the bookkeeping columns `_original_row_index` and
`_healpix32` after the new entry to preserve current ordering.)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_composite_id_column.py test/test_pdf_schema.py test/test_manifest.py -q
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/schema.py test/test_composite_id_column.py
git commit -m "phase21/T3: composite_id optional CORE column (U64; PLATE-MJD-FIBERID, KIDS_TILE+SeqNr, …)"
```

---

## Task 4: MOC → fixed-NSIDE rasteriser

**Files:**
- Create: `oneuniverse/data/moc.py`
- Modify: `pyproject.toml` — add `mocpy` to `[dev]` extras.
- Create: `test/test_moc_rasterise.py`

- [ ] **Step 1: Failing test**

```python
# test/test_moc_rasterise.py
"""Phase 21 T4 — rasterise a multi-order MOC HEALPix file to fixed NSIDE."""
import healpy as hp
import numpy as np
import pytest

mocpy = pytest.importorskip("mocpy")

from oneuniverse.data.moc import rasterise_moc_to_healpix


def test_rasterise_fixed_nside_circle(tmp_path):
    """A 1-deg radius MOC around (RA=10, Dec=0) at order 7 should map
    onto >0 pixels at NSIDE=32 (NEST) and 0 pixels outside the cone.
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    moc = mocpy.MOC.from_cone(
        lon=10 * u.deg, lat=0 * u.deg, radius=1 * u.deg, max_depth=7,
    )
    moc_file = tmp_path / "circle.fits"
    moc.write(str(moc_file))

    nside = 32
    arr = rasterise_moc_to_healpix(moc_file, nside=nside, nest=True)
    npix = hp.nside2npix(nside)
    assert arr.shape == (npix,)
    assert arr.sum() > 0
    # Pixel at the cone centre must be ON, pixel at the antipode OFF.
    centre_pix = hp.ang2pix(nside, 10.0, 0.0, nest=True, lonlat=True)
    anti_pix = hp.ang2pix(nside, 190.0, 0.0, nest=True, lonlat=True)
    assert arr[centre_pix] > 0
    assert arr[anti_pix] == 0


def test_missing_mocpy_raises_actionable_error(monkeypatch):
    """When mocpy is not installed, the import error should explain
    how to add it.
    """
    import sys

    monkeypatch.setitem(sys.modules, "mocpy", None)
    # Force a fresh import of the function module so the lazy import
    # picks up the None sentinel.
    if "oneuniverse.data.moc" in sys.modules:
        del sys.modules["oneuniverse.data.moc"]
    from oneuniverse.data import moc as mocmod  # noqa: F401
    with pytest.raises(ImportError, match="mocpy"):
        mocmod.rasterise_moc_to_healpix("dummy.fits", nside=32)
```

- [ ] **Step 2: Implement the module**

```python
# oneuniverse/data/moc.py
"""Multi-order MOC HEALPix → fixed-NSIDE rasteriser.

GW LIGO/Virgo sky-localisation FITS files (BAYESTAR / LALInference)
ship as **multi-order** HEALPix (NUNIQ-indexed). The downstream
:func:`oneuniverse.data.subobject_map.build_subobject_links_to_map`
expects a **fixed-NSIDE** numpy array; this module bridges the two
formats.

`mocpy` is an optional dependency. Importing this module without
`mocpy` installed succeeds; calling :func:`rasterise_moc_to_healpix`
raises an actionable :class:`ImportError`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import healpy as hp
import numpy as np


def rasterise_moc_to_healpix(
    moc_path: Union[str, Path],
    *,
    nside: int,
    nest: bool = True,
) -> np.ndarray:
    """Read a multi-order MOC HEALPix file from ``moc_path`` and
    rasterise it to a fixed-NSIDE float32 array of length
    ``12 * nside²``.

    Cells inside the MOC are set to ``1.0`` (uniform within-MOC
    weight); cells outside are ``0.0``. For a probability-map MOC
    (NUNIQ → PROB), the caller should multiply the returned array
    elementwise by the underlying probability extracted via
    ``mocpy.MOC.serialize``; the helper here only honours the
    coverage geometry.

    Parameters
    ----------
    moc_path
        Path to a FITS file readable by :class:`mocpy.MOC`.
    nside
        Output HEALPix NSIDE (power of two).
    nest
        Output ordering; default ``True`` to match the
        :func:`build_subobject_links_to_map` convention.

    Raises
    ------
    ImportError
        If `mocpy` is not installed in the current environment.
    """
    try:
        import mocpy as _mocpy  # type: ignore[import]
    except (ImportError, TypeError):
        raise ImportError(
            "rasterise_moc_to_healpix requires the optional `mocpy` "
            "dependency. Install with `pip install mocpy>=0.13` or "
            "use the dev extra: `pip install .[dev]`."
        ) from None

    moc = _mocpy.MOC.from_fits(str(moc_path))
    npix = hp.nside2npix(nside)
    # mocpy's contains expects ICRS lon/lat in degrees.
    pix = np.arange(npix)
    lon, lat = hp.pix2ang(nside, pix, nest=nest, lonlat=True)
    from astropy import units as u

    inside = moc.contains_lonlat(lon * u.deg, lat * u.deg)
    return np.asarray(inside, dtype=np.float32)
```

- [ ] **Step 3: Add `mocpy` to dev extras**

Edit `pyproject.toml` to extend the existing `[project.optional-dependencies]`
`dev` list with:

```toml
    "mocpy>=0.13",
```

- [ ] **Step 4: Run test to verify pass / skip cleanly**

```bash
pytest test/test_moc_rasterise.py -v
```

Expected: either 2 passed (if mocpy is installed) or 2 skipped (the
`importorskip` clause handles the first; the second test injects a
None sentinel for the negative path).

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/moc.py test/test_moc_rasterise.py pyproject.toml
git commit -m "phase21/T4: rasterise_moc_to_healpix (mocpy optional; bridges multi-order MOC to fixed NSIDE)"
```

---

## Task 5: Visual diagnostic

**Files:**
- Create: `test/test_visual_phase21.py`

- [ ] **Step 1: Test**

```python
# test/test_visual_phase21.py
"""Phase 21 visual diagnostic — attribute filter + composite_id + MOC."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.oneuid_crossmatch import cross_match_surveys  # noqa: E402
from oneuniverse.data.oneuid_rules import CrossMatchRules  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def _color_filter(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    dg = (left["psfmag_g"] - left["psfmag_r"]).to_numpy()
    dr = (right["psfmag_g"] - right["psfmag_r"]).to_numpy()
    return np.abs(dg - dr) < 0.1


def test_phase21_visual(tmp_path):
    rng = np.random.default_rng(0)
    n_per = 200
    ra1 = rng.uniform(10.0, 20.0, n_per)
    ra2 = ra1 + rng.normal(0, 1e-6, n_per)
    col1 = rng.uniform(0.0, 1.0, n_per)
    col2 = col1 + rng.normal(0, 0.05, n_per)
    col2[: n_per // 4] += 0.5  # 25% with mismatched colour

    table = pd.DataFrame({
        "ra": np.concatenate([ra1, ra2]),
        "dec": np.zeros(2 * n_per),
        "z": np.full(2 * n_per, 0.5, dtype="f4"),
        "z_type": np.array(["spec"] * (2 * n_per), dtype=object),
        "z_err": np.full(2 * n_per, 0.001, dtype="f4"),
        "galaxy_id": np.arange(2 * n_per, dtype="i8"),
        "survey": np.concatenate([
            np.array(["a"] * n_per), np.array(["b"] * n_per),
        ]),
        "_original_row_index": np.concatenate([
            np.arange(n_per), np.arange(n_per),
        ]),
        "psfmag_g": np.full(2 * n_per, 22.0, dtype="f4"),
        "psfmag_r": np.concatenate([22.0 - col1, 22.0 - col2]).astype("f4"),
    })

    no_filter = cross_match_surveys(
        table.copy(), CrossMatchRules(sky_tol_arcsec=2.0),
    )
    with_filter = cross_match_surveys(
        table.copy(),
        CrossMatchRules(
            sky_tol_arcsec=2.0,
            attribute_filters=(_color_filter,),
        ),
    )

    def _multi_count(df):
        return int(
            (df.groupby("universal_id")["survey"].nunique() > 1).sum()
        )

    n_no = _multi_count(no_filter)
    n_yes = _multi_count(with_filter)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].scatter(col1, col2, s=8, alpha=0.6, label="all")
    ax[0].plot([0, 1.5], [0, 1.5], "k--", lw=0.8, label="identity")
    ax[0].plot([0, 1.5], [0.1, 1.6], "r--", lw=0.8, label="|Δ|=0.1")
    ax[0].plot([0, 1.5], [-0.1, 1.4], "r--", lw=0.8)
    ax[0].set_xlabel("(g - r)_a")
    ax[0].set_ylabel("(g - r)_b")
    ax[0].set_title("Per-pair colour distribution")
    ax[0].legend()

    ax[1].bar(
        ["no filter", "colour filter"],
        [n_no, n_yes],
        color=["tab:gray", "tab:blue"],
    )
    ax[1].set_ylabel("multi-survey groups")
    ax[1].set_title(
        "attribute_filters cut "
        f"({100 * (1 - n_yes / max(n_no, 1)):.0f}% of links)"
    )

    fig.tight_layout()
    out_png = OUT / "phase21_attribute_filters.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im:
        assert im.width >= 800 and im.height >= 200
```

- [ ] **Step 2: Run**

```bash
pytest test/test_visual_phase21.py -v
```

Expected: pass; PNG ≥ 30 kB.

- [ ] **Step 3: Commit**

```bash
git add test/test_visual_phase21.py \
        test/test_output/phase21_attribute_filters.png
git commit -m "phase21/T5: visual diagnostic — attribute_filters cut multi-survey links by colour"
```

---

## Task 6: Docs

**Files:**
- Modify: `oneuniverse/CLAUDE.md`, `plans/README.md`,
  `research/schema_generalisation_audit.md`.

- [ ] **Step 1: CLAUDE.md**

Under "Bitemporal ONEUID / sub-object" append:

```
- `CrossMatchRules.attribute_filters: Tuple[Callable, ...]`
  (Phase 21) — pluggable predicates evaluated on candidate
  (left, right) DataFrames; return a bool mask. The matcher applies
  them after the dz cut. Filters are hashed by qualname.
- CORE `composite_id: U64` (optional, Phase 21) preserves the
  survey-published composite ID (PLATE-MJD-FIBERID, KIDS_TILE+SeqNr,
  GAIA source_id, …) alongside the canonical `int64` `galaxy_id`.
- `oneuniverse.data.moc.rasterise_moc_to_healpix(moc_path, *, nside,
  nest=True)` (Phase 21) bridges GW LIGO/Virgo multi-order MOC FITS
  to the fixed-NSIDE numpy arrays consumed by
  `build_subobject_links_to_map`. `mocpy` is an optional dev extra.
```

- [ ] **Step 2: plans/README.md**

Update the Phase 21 row to:

```
| 21 | Cleanup of deferred sub-object items (`CrossMatchRules.attribute_filters`, CORE `composite_id`, `mocpy` MOC rasteriser) | **complete (2026-05-29, NNN/NNN tests green)** |
```

- [ ] **Step 3: research/schema_generalisation_audit.md**

Replace the existing Phase 21 bullet with:

```
- **Phase 21 — Cleanup of deferred sub-object items.** Landed
  2026-05-29. Adds `CrossMatchRules.attribute_filters`,
  optional CORE `composite_id` column,
  `oneuniverse.data.moc.rasterise_moc_to_healpix` (mocpy optional
  dev extra). See
  [`../plans/2026-05-29-phase21-cleanup-deferred-subobject-items.md`](../plans/2026-05-29-phase21-cleanup-deferred-subobject-items.md).
```

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/CLAUDE.md plans/README.md \
        research/schema_generalisation_audit.md
git commit -m "docs(phase21): attribute_filters, composite_id, MOC rasteriser"
```

---

## Task 7: Close-out

- [ ] **Step 1: Full suite**

```bash
pytest -q 2>&1 | tail -3
```

Expected: green; record the count.

- [ ] **Step 2: Replace `NNN/NNN`** in plans/README.md.

- [ ] **Step 3: Memory update**

Append to
`/home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`:

```markdown
## Phase 21 — Cleanup of deferred Phase 20 items (complete 2026-05-29)

- `CrossMatchRules.attribute_filters: Tuple[Callable, ...]`
  — pluggable per-pair predicates applied after the dz cut;
  filters identified by qualname in the hash.
- CORE `composite_id: U64` (optional) — preserves the
  survey-published composite ID alongside the int64 `galaxy_id`
  (PLATE-MJD-FIBERID, KIDS_TILE+SeqNr, GAIA source_id, …).
- `oneuniverse.data.moc.rasterise_moc_to_healpix(moc_path, *, nside,
  nest=True)` — bridges multi-order MOC HEALPix FITS to fixed-NSIDE
  numpy arrays for `build_subobject_links_to_map`. `mocpy` is an
  optional dev extra.
- No OUF bump.
- Tests: NNN/NNN green.
- Per-phase plan:
  `plans/2026-05-29-phase21-cleanup-deferred-subobject-items.md`.
```

- [ ] **Step 4: Final commit**

```bash
git add plans/README.md \
        /home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md
git commit -m "phase21: close-out — deferred subobject items, NNN tests green"
```

---

## Self-review checklist

- [ ] No cosmology metadata added.
- [ ] `CrossMatchRules()` still works with no args.
- [ ] `composite_id` column is `required=False`.
- [ ] `mocpy` is not imported at package import time.
- [ ] `rasterise_moc_to_healpix` raises actionable `ImportError`
      when `mocpy` is missing.
- [ ] Visual PNG ≥ 30 kB.

## Spec-coverage map

| Requirement | Task |
|---|---|
| `CrossMatchRules.attribute_filters` | T1, T2 |
| Composite-ID preservation | T3 |
| MOC HEALPix support | T4 |
| Visual diagnostic | T5 |
| Docs | T6 |
| Close-out + memory | T7 |
