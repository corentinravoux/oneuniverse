# Phase 11 — Generic Selection / Completeness Weight Family

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `oneuniverse/combine/weights/` with a **generic selection / completeness weight family** so any user-supplied HEALPix mask, completeness map, imaging-systematic map, or per-object quality column can become a `Weight` without survey-specific code. Reproduces the *computational contract* of BOSS/eBOSS (`WEIGHT_SYSTOT`, `WEIGHT_CP`, `WEIGHT_NOZ`) and DESI DR1 (`WEIGHT_COMP`, `WEIGHT_SYS` via SYSNet, `WEIGHT_ZFAIL`) — but stays a library of primitives. **No dataset loaders ship with this phase.** User supplies the map or column; we read and apply.

**Architecture:** Three families land side-by-side in `oneuniverse/combine/weights/`:

1. **Angular-mask / HEALPix-map weights** (`hpmap.py`): `HealpixMapWeight(nside, map_array, nest=True)` — returns `map[pix_of(ra, dec)]` per object. Built on `healpy.ang2pix`. Used for completeness maps, imaging-systematic maps (SYSNet/Regressis output), stellar-density regressors, per-band depth maps, footprint masks.
2. **Fiber-assignment / redshift-failure weights** (`selection.py`): `FiberCollisionWeight(column)` and `ZFailureWeight(column)` — thin named wrappers around `ColumnWeight` that document the BOSS-style combination rule `w = w_sys * (w_cp + w_noz - 1)`.
3. **BOSS-style combination helper** (`selection.py`): `boss_total_weight(w_sys, w_cp, w_noz, w_fkp=None)` → a composite `Weight` applying the industry-standard formula. The library exposes both the primitives *and* the canonical combiner so users do not re-derive it.

The whole family composes through the existing `Weight` ABC / `ProductWeight` so chains like `FKP × HealpixMap(SYSNet) × (CP + NOZ − 1)` work out of the box.

**Tech Stack:** `healpy` (hard dep — already an oneuniverse dep for partitioning), numpy, pandas, the existing `Weight` ABC. No survey-specific assumptions; no default dataset loaders.

**Explicitly out of scope:**
- No BOSS/DESI loader writes. Real-survey adoption lives in a future phase.
- No regression framework (SYSNet/Regressis) ships here — we only consume its *output* maps.
- No spatial-window / pair-weighting math — those belong to estimators.

---

## File Structure

- Create: `oneuniverse/combine/weights/hpmap.py` — `HealpixMapWeight` + I/O helpers for `.fits` / ndarray maps.
- Create: `oneuniverse/combine/weights/selection.py` — `FiberCollisionWeight`, `ZFailureWeight`, `CompletenessWeight`, and `boss_total_weight` combiner.
- Modify: `oneuniverse/combine/weights/__init__.py` — export new classes.
- Modify: `oneuniverse/combine/weights/registry.py` — add `("spectroscopic", "spec_boss_like")` default chain + public `register_default(...)` hook so user code can register survey-specific recipes without editing the library.
- Create: `test/fixtures/healpix_maps.py` — fixture factory for synthetic HEALPix maps (footprint, SYSNet-like, completeness).
- Create: `test/test_hpmap_weight.py`, `test/test_selection_weights.py`, `test/test_boss_combiner.py`, `test/test_weights_registry_public.py`.
- Create: `test/test_visual_selection_weights.py` — diagnostic mollview of (map × points).
- Create: `test/test_output/phase11_*.png` — diagnostic figures.

---

### Task 1: HEALPix map fixture factory

**Files:**
- Create: `test/fixtures/healpix_maps.py`
- Test: `test/test_hpmap_fixture.py`

**Why:** Every weight test needs a reproducible HEALPix map with known structure so assertions can reference analytic values. Covers four flavours: a binary footprint, a smooth completeness gradient, a SYSNet-like Gaussian systematic, and a NaN-sprinkled map (to test missing-pixel handling).

- [ ] **Step 1: Failing test**

```python
# test/test_hpmap_fixture.py
import numpy as np
from test.fixtures.healpix_maps import (
    make_footprint_mask, make_smooth_completeness, make_systematic_map,
)


def test_footprint_binary_and_shape():
    m = make_footprint_mask(nside=32, seed=0)
    assert set(np.unique(m)) <= {0.0, 1.0}
    import healpy as hp
    assert m.shape == (hp.nside2npix(32),)


def test_completeness_in_unit_interval():
    m = make_smooth_completeness(nside=32, seed=0)
    assert (m >= 0).all() and (m <= 1).all()


def test_systematic_map_finite_positive():
    m = make_systematic_map(nside=32, seed=0)
    assert np.all(np.isfinite(m))
    assert (m > 0).all()
```

Run: `pytest test/test_hpmap_fixture.py -v` → FAIL.

- [ ] **Step 2: Implement fixtures**

```python
# test/fixtures/healpix_maps.py
"""Synthetic HEALPix maps used by Phase 11 weight tests."""
from __future__ import annotations

import healpy as hp
import numpy as np


def make_footprint_mask(nside: int = 32, fsky: float = 0.2, seed: int = 0) -> np.ndarray:
    """Contiguous-ish footprint: every pixel within a dec band is 1."""
    npix = hp.nside2npix(nside)
    theta, _phi = hp.pix2ang(nside, np.arange(npix), nest=True)
    dec = 90.0 - np.degrees(theta)
    band = (dec > -20) & (dec < 40)
    m = np.zeros(npix, dtype=np.float64)
    m[band] = 1.0
    return m


def make_smooth_completeness(nside: int = 32, seed: int = 0) -> np.ndarray:
    """Per-pixel completeness in [0, 1]: mean 0.9, Gaussian dimples."""
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    base = np.full(npix, 0.9)
    ripple = 0.1 * rng.standard_normal(npix)
    m = np.clip(base + ripple, 0.0, 1.0)
    return m


def make_systematic_map(nside: int = 32, seed: int = 0) -> np.ndarray:
    """SYSNet-like positive weight map with mean 1.0."""
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)
    m = np.exp(0.05 * rng.standard_normal(npix))
    return m
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_hpmap_fixture.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add test/fixtures/healpix_maps.py test/test_hpmap_fixture.py
git commit -m "phase11: HEALPix map fixture factory (footprint, completeness, systematic)"
```

---

### Task 2: `HealpixMapWeight` — core primitive

**Files:**
- Create: `oneuniverse/combine/weights/hpmap.py`
- Test: `test/test_hpmap_weight.py`

**Why:** Single ring/nest-aware path from (ra, dec, map) → per-object weight. Every DESI/SYSNet/Regressis map slots in here.

- [ ] **Step 1: Failing tests**

```python
# test/test_hpmap_weight.py
import healpy as hp
import numpy as np
import pandas as pd
import pytest

from oneuniverse.combine.weights.hpmap import HealpixMapWeight
from test.fixtures.healpix_maps import (
    make_footprint_mask, make_smooth_completeness, make_systematic_map,
)


def _df(ra, dec):
    return pd.DataFrame({"ra": np.asarray(ra), "dec": np.asarray(dec)})


def test_systematic_map_weight_matches_direct_lookup():
    nside = 32
    m = make_systematic_map(nside, seed=0)
    w = HealpixMapWeight(nside=nside, map_array=m, nest=True)
    ra = np.array([12.0, 180.0, 350.0])
    dec = np.array([0.0, 10.0, -25.0])
    got = w(_df(ra, dec))
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    expected = m[hp.ang2pix(nside, theta, phi, nest=True)]
    np.testing.assert_allclose(got, expected)


def test_footprint_mask_zeroes_outside_band():
    nside = 32
    m = make_footprint_mask(nside, seed=0)
    w = HealpixMapWeight(nside=nside, map_array=m, nest=True)
    df = _df([100.0, 100.0], [-70.0, 20.0])
    got = w(df)
    assert got[0] == 0.0 and got[1] == 1.0


def test_ring_order_map_accepted():
    nside = 32
    m_nest = make_systematic_map(nside, seed=1)
    # Convert to ring order and confirm equivalence when nest=False.
    m_ring = hp.reorder(m_nest, n2r=True)
    w_ring = HealpixMapWeight(nside=nside, map_array=m_ring, nest=False)
    w_nest = HealpixMapWeight(nside=nside, map_array=m_nest, nest=True)
    df = _df([10, 100, 200, 300], [0, 20, -10, 30])
    np.testing.assert_allclose(w_ring(df), w_nest(df))


def test_rejects_wrong_length_map():
    with pytest.raises(ValueError, match="length"):
        HealpixMapWeight(nside=32, map_array=np.ones(5), nest=True)


def test_nan_pixels_raise_unless_fill():
    nside = 32
    m = make_systematic_map(nside, seed=2)
    m[0] = np.nan
    # Row that lands in pixel 0 (pole)
    df = _df([0.0], [90.0])
    with pytest.raises(ValueError, match="NaN"):
        HealpixMapWeight(nside=nside, map_array=m, nest=True)(df)

    # Opt-in fill value
    w2 = HealpixMapWeight(nside=nside, map_array=m, nest=True, nan_fill=0.0)
    assert w2(df)[0] == 0.0


def test_from_fits_roundtrip(tmp_path):
    import healpy as hp
    nside = 32
    m = make_systematic_map(nside, seed=3)
    path = tmp_path / "sysmap.fits"
    hp.write_map(path, m, nest=True, overwrite=True)

    w = HealpixMapWeight.from_fits(path, nest=True)
    df = _df([10, 20, 30], [0, 5, -5])
    np.testing.assert_allclose(w(df), HealpixMapWeight(nside, m, nest=True)(df))
```

Run: `pytest test/test_hpmap_weight.py -v` → FAIL (module absent).

- [ ] **Step 2: Implement `hpmap.py`**

```python
# oneuniverse/combine/weights/hpmap.py
"""HEALPix map-backed weight primitive.

Given a full-sky HEALPix array (any valid NSIDE, ring or nest), returns
the map value at the pixel containing each object's (ra, dec).  Covers
the entire class of survey weights that are *stored as a map*:

* completeness / angular-footprint masks
* imaging-systematic weights (SYSNet, Regressis, linear regressors)
* stellar-density / extinction regressors
* per-band depth maps

No survey-specific knowledge lives here — user provides the map.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight


class HealpixMapWeight(Weight):
    """Per-object weight ``w_i = map[pix(ra_i, dec_i)]``.

    Parameters
    ----------
    nside
        HEALPix NSIDE of the map.  Must satisfy ``len(map_array) ==
        12 * nside**2``.
    map_array
        The full-sky map as a 1-D array of per-pixel values.
    nest
        ``True`` if the map is in NESTED ordering, ``False`` for RING.
    ra_column, dec_column
        DataFrame column names carrying ICRS RA/Dec in degrees.
    nan_fill
        If not ``None``, any NaN pixel hit returns this value instead of
        raising.  Use e.g. ``0.0`` to zero out objects falling in
        unsurveyed / masked cells.
    """

    def __init__(
        self,
        nside: int,
        map_array: np.ndarray,
        nest: bool = True,
        ra_column: str = "ra",
        dec_column: str = "dec",
        nan_fill: Optional[float] = None,
        name: str = "hpmap",
    ) -> None:
        map_array = np.asarray(map_array, dtype=np.float64)
        expected = hp.nside2npix(nside)
        if map_array.ndim != 1 or map_array.size != expected:
            raise ValueError(
                f"HealpixMapWeight: map length {map_array.size} does not match "
                f"NSIDE={nside} ({expected} pixels)"
            )
        self.nside = int(nside)
        self.map_array = map_array
        self.nest = bool(nest)
        self.ra_column = ra_column
        self.dec_column = dec_column
        self.nan_fill = nan_fill
        self.name = name

    @classmethod
    def from_fits(
        cls, path: Union[str, Path], nest: bool = True, **kwargs
    ) -> "HealpixMapWeight":
        """Read a HEALPix FITS map (``hp.read_map``) and return the weight."""
        arr = hp.read_map(path, nest=nest, verbose=False)
        nside = hp.npix2nside(arr.size)
        return cls(nside=nside, map_array=arr, nest=nest, **kwargs)

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        ra = df[self.ra_column].to_numpy(dtype=np.float64)
        dec = df[self.dec_column].to_numpy(dtype=np.float64)
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        pix = hp.ang2pix(self.nside, theta, phi, nest=self.nest)
        vals = self.map_array[pix]
        if np.any(~np.isfinite(vals)):
            if self.nan_fill is None:
                raise ValueError(
                    f"HealpixMapWeight({self.name}): NaN/inf in map at "
                    f"{(~np.isfinite(vals)).sum()} object(s); pass nan_fill=… "
                    f"to silently mask them."
                )
            vals = np.where(np.isfinite(vals), vals, self.nan_fill)
        return vals
```

- [ ] **Step 3: Run tests**

Run: `pytest test/test_hpmap_weight.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/combine/weights/hpmap.py test/test_hpmap_weight.py
git commit -m "phase11: HealpixMapWeight primitive + from_fits loader"
```

---

### Task 3: Named selection wrappers + BOSS combiner

**Files:**
- Create: `oneuniverse/combine/weights/selection.py`
- Test: `test/test_selection_weights.py`, `test/test_boss_combiner.py`

**Why:** Most surveys compute three per-object *numbers* (`w_cp`, `w_noz`, `w_sys`) and one *map* (footprint/systematic). The named wrappers exist so call-sites are self-documenting, and `boss_total_weight` hardcodes the industry-standard formula so users do not re-derive it.

- [ ] **Step 1: Failing tests**

```python
# test/test_selection_weights.py
import numpy as np
import pandas as pd

from oneuniverse.combine.weights.selection import (
    CompletenessWeight, FiberCollisionWeight, ZFailureWeight,
)


def _df():
    return pd.DataFrame({
        "w_cp": [1.0, 1.5, 2.0],
        "w_noz": [1.0, 1.2, 0.8],
        "w_comp": [0.95, 0.9, 1.0],
    })


def test_fiber_collision_weight_passthrough():
    np.testing.assert_allclose(FiberCollisionWeight("w_cp")(_df()),
                               _df()["w_cp"].to_numpy(dtype=np.float64))


def test_z_failure_weight_passthrough():
    np.testing.assert_allclose(ZFailureWeight("w_noz")(_df()),
                               _df()["w_noz"].to_numpy(dtype=np.float64))


def test_completeness_weight_passthrough():
    np.testing.assert_allclose(CompletenessWeight("w_comp")(_df()),
                               _df()["w_comp"].to_numpy(dtype=np.float64))
```

```python
# test/test_boss_combiner.py
import numpy as np
import pandas as pd

from oneuniverse.combine.weights.selection import (
    FiberCollisionWeight, ZFailureWeight, boss_total_weight,
)
from oneuniverse.combine.weights.quality import ColumnWeight


def _df():
    return pd.DataFrame({
        "w_cp": [1.0, 1.5, 2.0],
        "w_noz": [1.0, 1.2, 0.8],
        "w_sys": [1.0, 0.9, 1.1],
        "w_fkp": [0.3, 0.2, 0.4],
    })


def test_boss_total_formula_no_fkp():
    w = boss_total_weight(
        w_sys=ColumnWeight("w_sys"),
        w_cp=FiberCollisionWeight("w_cp"),
        w_noz=ZFailureWeight("w_noz"),
    )
    got = w(_df())
    expected = _df()["w_sys"].to_numpy() * (
        _df()["w_cp"].to_numpy() + _df()["w_noz"].to_numpy() - 1.0
    )
    np.testing.assert_allclose(got, expected)


def test_boss_total_formula_with_fkp():
    w = boss_total_weight(
        w_sys=ColumnWeight("w_sys"),
        w_cp=FiberCollisionWeight("w_cp"),
        w_noz=ZFailureWeight("w_noz"),
        w_fkp=ColumnWeight("w_fkp"),
    )
    got = w(_df())
    expected = (
        _df()["w_sys"].to_numpy()
        * (_df()["w_cp"].to_numpy() + _df()["w_noz"].to_numpy() - 1.0)
        * _df()["w_fkp"].to_numpy()
    )
    np.testing.assert_allclose(got, expected)
```

Run: `pytest test/test_selection_weights.py test/test_boss_combiner.py -v` → FAIL.

- [ ] **Step 2: Implement `selection.py`**

```python
# oneuniverse/combine/weights/selection.py
"""Named selection-weight primitives + BOSS/eBOSS combiner.

These are thin, self-documenting wrappers around
:class:`ColumnWeight` so a call-site like::

    w_cp = FiberCollisionWeight("w_cp")

reads the same as the BOSS-DR12 catalog column name it mirrors.  The
``boss_total_weight`` helper packages the industry-standard composition
formula ``w = w_sys * (w_cp + w_noz - 1) * w_fkp`` (Reid et al. 2016)
as one callable.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight
from oneuniverse.combine.weights.quality import ColumnWeight


class FiberCollisionWeight(ColumnWeight):
    """Close-pair / fiber-collision weight ``w_cp``."""

    def __init__(self, column: str = "w_cp", name: Optional[str] = None) -> None:
        super().__init__(column=column, name=name or "w_cp")


class ZFailureWeight(ColumnWeight):
    """Redshift-failure weight ``w_noz`` (BOSS) / ``w_zfail`` (DESI)."""

    def __init__(self, column: str = "w_noz", name: Optional[str] = None) -> None:
        super().__init__(column=column, name=name or "w_zfail")


class CompletenessWeight(ColumnWeight):
    """Per-object completeness weight ``w_comp``."""

    def __init__(self, column: str = "w_comp", name: Optional[str] = None) -> None:
        super().__init__(column=column, name=name or "w_comp")


class _BossCompositeWeight(Weight):
    """Implements ``w = w_sys * (w_cp + w_noz - 1) * [w_fkp]``."""

    def __init__(
        self,
        w_sys: Weight,
        w_cp: Weight,
        w_noz: Weight,
        w_fkp: Optional[Weight],
    ) -> None:
        self.w_sys = w_sys
        self.w_cp = w_cp
        self.w_noz = w_noz
        self.w_fkp = w_fkp
        tag = "sys*(cp+noz-1)"
        if w_fkp is not None:
            tag += "*fkp"
        self.name = tag

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        sys_ = self.w_sys(df)
        cp = self.w_cp(df)
        noz = self.w_noz(df)
        total = sys_ * (cp + noz - 1.0)
        if self.w_fkp is not None:
            total = total * self.w_fkp(df)
        return total


def boss_total_weight(
    w_sys: Weight,
    w_cp: Weight,
    w_noz: Weight,
    w_fkp: Optional[Weight] = None,
) -> Weight:
    """BOSS/eBOSS canonical composition.

    Formula (Reid et al. 2016): ``WEIGHT = WEIGHT_SYSTOT * (WEIGHT_NOZ +
    WEIGHT_CP - 1)`` with optional FKP factor ``WEIGHT_FKP``.
    """
    return _BossCompositeWeight(w_sys=w_sys, w_cp=w_cp, w_noz=w_noz, w_fkp=w_fkp)
```

- [ ] **Step 3: Run tests**

Run: `pytest test/test_selection_weights.py test/test_boss_combiner.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/combine/weights/selection.py \
        test/test_selection_weights.py test/test_boss_combiner.py
git commit -m "phase11: FiberCollision/ZFailure/Completeness wrappers + BOSS combiner"
```

---

### Task 4: Public `register_default` extension hook

**Files:**
- Modify: `oneuniverse/combine/weights/registry.py`
- Test: `test/test_weights_registry_public.py`

**Why:** The registry currently hard-codes its factories in a frozen `MappingProxyType`. Survey packages and user scripts need a way to register new `(survey_type, z_type)` pairs — e.g. `("spectroscopic", "spec_boss")` → `boss_total_weight` chain — without editing the library. Mirrors how Phase 6 named loaders.

- [ ] **Step 1: Failing test**

```python
# test/test_weights_registry_public.py
import pandas as pd
import pytest

from oneuniverse.combine.weights import ColumnWeight, default_weight_for
from oneuniverse.combine.weights.registry import (
    register_default, unregister_default,
)


def test_register_new_default():
    def _factory():
        return ColumnWeight("special_w", name="special")

    register_default("custom", "x", _factory)
    try:
        w = default_weight_for("custom", "x")
        df = pd.DataFrame({"special_w": [1.0, 2.0, 3.0]})
        got = w(df)
        assert list(got) == [1.0, 2.0, 3.0]
    finally:
        unregister_default("custom", "x")


def test_register_default_rejects_duplicate():
    def _factory():
        return ColumnWeight("a")
    register_default("custom2", "x", _factory)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_default("custom2", "x", _factory)
    finally:
        unregister_default("custom2", "x")


def test_unregister_missing_raises():
    with pytest.raises(KeyError):
        unregister_default("does_not_exist", "anything")
```

Run: `pytest test/test_weights_registry_public.py -v` → FAIL.

- [ ] **Step 2: Mutate `registry.py` to support registration**

```python
# oneuniverse/combine/weights/registry.py — replace MappingProxyType
# with a plain mutable dict + public helpers
from typing import Callable, Dict, Tuple
# …
_DEFAULTS: Dict[Key, Factory] = {
    ("spectroscopic", "spec"): _ivar_spec,
    ("photometric", "phot"): _ivar_phot,
    ("peculiar_velocity", "pec"): _ivar_pec,
}


def register_default(survey_type: str, z_type: str, factory: Factory) -> None:
    """Register a default weight factory for ``(survey_type, z_type)``.

    Raises ``ValueError`` if the key already has a registration — callers
    must explicitly ``unregister_default`` first to avoid silent clobber.
    """
    key = (survey_type, z_type)
    if key in _DEFAULTS:
        raise ValueError(
            f"register_default: {key!r} is already registered "
            f"(call unregister_default first if you intend to replace it)"
        )
    _DEFAULTS[key] = factory


def unregister_default(survey_type: str, z_type: str) -> None:
    key = (survey_type, z_type)
    del _DEFAULTS[key]
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_weights_registry_public.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/combine/weights/registry.py test/test_weights_registry_public.py
git commit -m "phase11: register_default / unregister_default public hooks"
```

---

### Task 5: Export new classes + smoke-test composition

**Files:**
- Modify: `oneuniverse/combine/weights/__init__.py`
- Test: `test/test_weights_composition_full.py`

**Why:** End-to-end smoke test: a realistic weight chain — HEALPix systematic × BOSS composite × FKP — returns the right numeric vector when fed a mini DataFrame + toy maps. Catches any future break where `ProductWeight` chaining stops flowing.

- [ ] **Step 1: Failing test**

```python
# test/test_weights_composition_full.py
import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.combine.weights import (
    FKPWeight, HealpixMapWeight,
    FiberCollisionWeight, ZFailureWeight, boss_total_weight,
    ColumnWeight,
)
from test.fixtures.healpix_maps import make_systematic_map


def test_full_chain_sysmap_times_boss_times_fkp():
    nside = 32
    m = make_systematic_map(nside, seed=0)
    hpw = HealpixMapWeight(nside=nside, map_array=m, nest=True, name="w_sys_map")

    boss = boss_total_weight(
        w_sys=hpw,
        w_cp=FiberCollisionWeight("w_cp"),
        w_noz=ZFailureWeight("w_noz"),
    )

    def _nbar(z):
        return np.full_like(z, 1e-4, dtype=np.float64)

    fkp = FKPWeight(nbar=_nbar, P0=1e4, z_column="z")

    composed = boss * fkp
    df = pd.DataFrame({
        "ra": [10.0, 100.0, 250.0],
        "dec": [0.0, 20.0, -10.0],
        "z": [0.5, 0.7, 0.3],
        "w_cp": [1.0, 1.4, 1.1],
        "w_noz": [1.0, 1.1, 0.9],
    })
    got = composed(df)
    # Manual reconstruction:
    theta = np.radians(90.0 - df["dec"].to_numpy())
    phi = np.radians(df["ra"].to_numpy())
    sys_vals = m[hp.ang2pix(nside, theta, phi, nest=True)]
    boss_vals = sys_vals * (df["w_cp"].to_numpy() + df["w_noz"].to_numpy() - 1.0)
    fkp_vals = 1.0 / (1.0 + 1e-4 * 1e4)
    np.testing.assert_allclose(got, boss_vals * fkp_vals)
```

Run: `pytest test/test_weights_composition_full.py -v` → FAIL.

- [ ] **Step 2: Export new classes**

```python
# oneuniverse/combine/weights/__init__.py — extend imports
from oneuniverse.combine.weights.hpmap import HealpixMapWeight
from oneuniverse.combine.weights.selection import (
    CompletenessWeight, FiberCollisionWeight, ZFailureWeight,
    boss_total_weight,
)

__all__ = [
    "Weight", "ProductWeight", "ConstantWeight", "ColumnWeight",
    "InverseVarianceWeight", "FKPWeight", "QualityMaskWeight",
    "HealpixMapWeight",
    "CompletenessWeight", "FiberCollisionWeight", "ZFailureWeight",
    "boss_total_weight",
    "default_weight_for",
]
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_weights_composition_full.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/combine/weights/__init__.py test/test_weights_composition_full.py
git commit -m "phase11: export HealpixMap + selection weights + full-chain smoke test"
```

---

### Task 6: Diagnostic visual test — map × catalog overlay

**Files:**
- Create: `test/test_visual_selection_weights.py`
- Create: `test/test_output/phase11_hpmap_overlay.png`

**Why:** User's standing rule: data-infrastructure work must produce diagnostic figures. A mollview of the systematic map with the DataFrame points over-plotted, coloured by their weight value, reveals off-by-one pixel errors instantly.

- [ ] **Step 1: Failing visual test**

```python
# test/test_visual_selection_weights.py
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.combine.weights.hpmap import HealpixMapWeight
from test.fixtures.healpix_maps import make_systematic_map


OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase11_hpmap_overlay():
    nside = 32
    m = make_systematic_map(nside, seed=7)
    n = 2000
    rng = np.random.default_rng(0)
    # uniform sphere sampling
    u = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2 * np.pi, size=n)
    dec = np.degrees(np.arcsin(u))
    ra = np.degrees(phi)
    df = pd.DataFrame({"ra": ra, "dec": dec})

    w = HealpixMapWeight(nside=nside, map_array=m, nest=True)
    vals = w(df)

    fig = plt.figure(figsize=(10, 5))
    hp.mollview(m, nest=True, title="Systematic map (synthetic)", fig=fig,
                sub=121, cmap="viridis")
    ax = fig.add_subplot(122)
    sc = ax.scatter(df["ra"], df["dec"], c=vals, s=2, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="HealpixMapWeight value")
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
    ax.set_title("Points coloured by pixel weight")
    fig.tight_layout()
    fig.savefig(OUT / "phase11_hpmap_overlay.png", dpi=110)
    plt.close(fig)
```

Run: `pytest test/test_visual_selection_weights.py -v` → PASS. Inspect figure.

- [ ] **Step 2: Commit**

```bash
git add test/test_visual_selection_weights.py
git commit -m "phase11: visual test — systematic map × catalog overlay"
```

---

### Task 7: Close Phase 11

**Files:**
- Modify: `plans/README.md`
- Modify: `~/.claude/projects/-home-ravoux-Documents-Python/memory/reference_selection_weights.md` (status pointer)

**Why:** Mirrors every prior phase closeout.

- [ ] **Step 1: Run full suite**

Run: `pytest` from `Packages/oneuniverse` → all green; record count.

- [ ] **Step 2: Update `plans/README.md`**

Add row:

```
| 11 | Generic selection / completeness weights (HealpixMapWeight, FiberCollision/ZFailure/Completeness wrappers, boss_total_weight, public register_default) | **complete (YYYY-MM-DD, N/N tests green)** |
```

- [ ] **Step 3: Update memory file `project_oneuniverse_stabilisation.md`**

Append Phase 11 summary: primitives, public registration hook, no dataset loaders shipped, BOSS combiner documented.

- [ ] **Step 4: Final commit**

```bash
git add plans/README.md
git commit -m "phase11: close-out — update plans/README"
```
