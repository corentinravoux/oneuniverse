# Phase 19 — Shear Column Group + ShearWeight + PipBitweightWeight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make weak lensing a first-class probe in OUF. Add a `SHEAR_COLUMNS` schema group (`e1, e2, R11, R22, R12, R21, R_S, m_bias, c1_bias, c2_bias, shear_weight, e1_err, e2_err`), a `ShearWeight` primitive that propagates shape noise + calibration responses for both metacalibration and lensfit pipelines, and a `PipBitweightWeight` primitive that expands DESI `BITWEIGHTS: i8[64]` into per-row fractional weights. Extend `default_weight_for` registration to optionally key on `sub_kind` so sub-species of the same `(survey_type, z_type)` (e.g. DESI BGS bright vs faint) can register distinct defaults.

**Architecture:** Three weight modules under `oneuniverse/combine/weights/`. `shear.py` exports `ShearWeight(kind="metacal" | "lensfit")` returning `shear_weight / (R_eff² + σ_e²)`. `pip.py` exports `PipBitweightWeight(mode="fraction" | "realisations")` using `numpy.unpackbits` over the int64 bit field. `registry.py` widens the registration key from `(survey_type, z_type)` to `(survey_type, sub_kind, z_type)` with `sub_kind=None` default — existing two-key registrations keep working. Schema additions are pure forward-compatible (all `required=False`); no OUF version bump.

**Tech Stack:** Python 3.9+, numpy, pandas, dataclasses, pytest. No new dependencies.

---

## File Structure

**New files:**
- `oneuniverse/combine/weights/shear.py` — `ShearWeight`.
- `oneuniverse/combine/weights/pip.py` — `PipBitweightWeight`.
- `test/test_shear_columns.py` — `SHEAR_COLUMNS` group + schema validation.
- `test/test_shear_weight.py` — `ShearWeight` for metacal + lensfit.
- `test/test_pip_bitweight.py` — `PipBitweightWeight` for both modes.
- `test/test_registry_sub_kind.py` — sub-species registry key extension.
- `test/test_visual_phase19.py` — shear + bitweight diagnostic.

**Modified files:**
- `oneuniverse/data/schema.py` — add `SHEAR_COLUMNS` tuple + entry in `COLUMN_GROUPS`.
- `oneuniverse/combine/weights/__init__.py` — re-export `ShearWeight` + `PipBitweightWeight`.
- `oneuniverse/combine/__init__.py` — re-export both names too (matches existing pattern).
- `oneuniverse/combine/weights/registry.py` — accept optional `sub_kind`.
- `oneuniverse/CLAUDE.md` — mention shear group + new weight primitives.
- `plans/README.md` — Phase 19 status row.
- `research/schema_generalisation_audit.md` — Phase 19 close-out cross-ref.

---

## Pre-flight

- [ ] **Step 0: Confirm baseline.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest -q 2>&1 | tail -3
```

Expected: `450 passed, 1 skipped` (Phase 18 baseline).

---

## Task 1: `SHEAR_COLUMNS` group

**Files:**
- Modify: `oneuniverse/data/schema.py` — add tuple + group entry
- Create: `test/test_shear_columns.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_shear_columns.py
"""Phase 19 T1 — SHEAR_COLUMNS group + schema integration."""
import numpy as np
import pandas as pd

from oneuniverse.data.schema import (
    COLUMN_GROUPS,
    SHEAR_COLUMNS,
    get_all_columns,
    validate_dataframe,
)


def test_shear_columns_registered():
    assert "shear" in COLUMN_GROUPS
    assert COLUMN_GROUPS["shear"] is SHEAR_COLUMNS


def test_shear_columns_contents():
    names = {c.name for c in SHEAR_COLUMNS}
    expected = {
        "e1", "e2", "e1_err", "e2_err",
        "R11", "R22", "R12", "R21", "R_S",
        "m_bias", "c1_bias", "c2_bias",
        "shear_weight",
    }
    assert expected <= names


def test_no_shear_column_required_by_default():
    for c in SHEAR_COLUMNS:
        assert c.required is False


def test_validate_dataframe_accepts_shear_subset():
    df = pd.DataFrame({
        "e1": np.array([0.1, 0.0], dtype="f4"),
        "e2": np.array([0.0, -0.1], dtype="f4"),
        "shear_weight": np.array([1.0, 1.0], dtype="f4"),
    })
    warnings = validate_dataframe(df, ["shear"])
    assert warnings == []


def test_get_all_columns_includes_shear():
    cols = get_all_columns(["shear"])
    assert "e1" in cols and "R11" in cols
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_shear_columns.py -v
```

Expected: `ImportError: cannot import name 'SHEAR_COLUMNS'`.

- [ ] **Step 3: Add the group**

In `oneuniverse/data/schema.py`, just before `COLUMN_GROUPS`, add:

```python
SHEAR_COLUMNS: Tuple[ColumnDef, ...] = (
    ColumnDef("e1", "f4", "", "First shear component", required=False),
    ColumnDef("e2", "f4", "", "Second shear component", required=False),
    ColumnDef("e1_err", "f4", "", "1σ on e1", required=False),
    ColumnDef("e2_err", "f4", "", "1σ on e2", required=False),
    ColumnDef("R11", "f4", "", "Metacal response ∂e1/∂γ1", required=False),
    ColumnDef("R22", "f4", "", "Metacal response ∂e2/∂γ2", required=False),
    ColumnDef("R12", "f4", "", "Metacal off-diagonal response", required=False),
    ColumnDef("R21", "f4", "", "Metacal off-diagonal response", required=False),
    ColumnDef("R_S", "f4", "", "Selection response", required=False),
    ColumnDef("m_bias", "f4", "", "Multiplicative bias (lensfit)", required=False),
    ColumnDef("c1_bias", "f4", "", "Additive bias e1 (lensfit)", required=False),
    ColumnDef("c2_bias", "f4", "", "Additive bias e2 (lensfit)", required=False),
    ColumnDef("shear_weight", "f4", "", "Per-object shape weight", required=False),
)
```

Extend `COLUMN_GROUPS` with the new entry:

```python
COLUMN_GROUPS: Dict[str, Tuple[ColumnDef, ...]] = {
    "core": CORE_COLUMNS,
    "spectroscopic": SPECTROSCOPIC_COLUMNS,
    "photometric": PHOTOMETRIC_COLUMNS,
    "peculiar_velocity": PV_COLUMNS,
    "qso": QSO_COLUMNS,
    "snia": SNIA_COLUMNS,
    "probabilistic_redshift": PROBABILISTIC_REDSHIFT_COLUMNS,
    "shear": SHEAR_COLUMNS,
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_shear_columns.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/schema.py test/test_shear_columns.py
git commit -m "phase19/T1: SHEAR_COLUMNS schema group (e1/e2/R*/m/c*/shear_weight; all optional)"
```

---

## Task 2: `ShearWeight` primitive (metacal + lensfit)

**Files:**
- Create: `oneuniverse/combine/weights/shear.py`
- Create: `test/test_shear_weight.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_shear_weight.py
"""Phase 19 T2 — ShearWeight (metacal + lensfit) propagates shape noise
and calibration responses.
"""
import numpy as np
import pandas as pd
import pytest

from oneuniverse.combine.weights.shear import ShearWeight


def test_metacal_default_response_one_and_zero_sigma():
    df = pd.DataFrame({
        "shear_weight": np.array([1.0, 1.0], dtype="f4"),
        "R11": np.array([1.0, 1.0], dtype="f4"),
        "R22": np.array([1.0, 1.0], dtype="f4"),
    })
    w = ShearWeight(kind="metacal").compute(df)
    # shear_weight / R_eff^2 with R_eff = (1+1)/2 = 1 and σ_e=0 → 1.0
    np.testing.assert_allclose(w, np.array([1.0, 1.0]), rtol=1e-6)


def test_metacal_with_response_below_one_amplifies():
    df = pd.DataFrame({
        "shear_weight": np.array([1.0], dtype="f4"),
        "R11": np.array([0.7], dtype="f4"),
        "R22": np.array([0.7], dtype="f4"),
    })
    w = ShearWeight(kind="metacal").compute(df)
    # R_eff = 0.7; w = 1 / 0.49 ≈ 2.04
    np.testing.assert_allclose(w, np.array([1.0 / 0.49]), rtol=1e-6)


def test_metacal_with_selection_response_added():
    df = pd.DataFrame({
        "shear_weight": np.array([1.0], dtype="f4"),
        "R11": np.array([0.6], dtype="f4"),
        "R22": np.array([0.6], dtype="f4"),
        "R_S": np.array([0.1], dtype="f4"),
    })
    w = ShearWeight(kind="metacal").compute(df)
    # R_eff = 0.6 + 0.1 = 0.7
    np.testing.assert_allclose(w, np.array([1.0 / 0.49]), rtol=1e-6)


def test_metacal_sigma_e_in_denominator():
    df = pd.DataFrame({
        "shear_weight": np.array([1.0], dtype="f4"),
        "R11": np.array([1.0], dtype="f4"),
        "R22": np.array([1.0], dtype="f4"),
        "e1_err": np.array([0.5], dtype="f4"),
        "e2_err": np.array([0.5], dtype="f4"),
    })
    w = ShearWeight(kind="metacal", sigma_e_cols=("e1_err", "e2_err")).compute(df)
    # σ_e^2 = 0.5*0.5 + 0.5*0.5 = 0.5 → denom = 1 + 0.5 = 1.5
    np.testing.assert_allclose(w, np.array([1.0 / 1.5]), rtol=1e-6)


def test_lensfit_uses_one_plus_m():
    df = pd.DataFrame({
        "shear_weight": np.array([1.0], dtype="f4"),
        "m_bias": np.array([0.05], dtype="f4"),
    })
    w = ShearWeight(kind="lensfit").compute(df)
    # R_eff = 1 + 0.05 = 1.05 ; w = 1 / 1.1025
    np.testing.assert_allclose(w, np.array([1.0 / (1.05 ** 2)]), rtol=1e-6)


def test_invalid_kind_rejected():
    with pytest.raises(ValueError, match="kind"):
        ShearWeight(kind="unknown")


def test_missing_response_columns_raise():
    df = pd.DataFrame({"shear_weight": np.array([1.0], dtype="f4")})
    with pytest.raises(KeyError):
        ShearWeight(kind="metacal").compute(df)
    with pytest.raises(KeyError):
        ShearWeight(kind="lensfit").compute(df)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_shear_weight.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the module**

```python
# oneuniverse/combine/weights/shear.py
"""
oneuniverse.combine.weights.shear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shear-catalogue weights for both metacalibration and lensfit
pipelines.

For metacalibration (DES Y3, HSC-Y3 metadetect, Rubin), the effective
shear response is

    R_eff = (R11 + R22) / 2 + R_S    (R_S optional)

For lensfit (KiDS-1000, KiDS-450, CFHTLenS), the effective response is

    R_eff = 1 + m_bias

The output weight per row is

    w = shear_weight / (R_eff² + σ_e²)

where σ_e² is optional; if ``sigma_e_cols`` is given, it is computed as
``e1_err² + e2_err²``, matching the standard shape-noise convention.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight

_ALLOWED_KINDS = frozenset({"metacal", "lensfit"})


class ShearWeight(Weight):
    """Per-object shear weight for metacal or lensfit pipelines.

    Parameters
    ----------
    kind : str
        ``"metacal"`` (DES Y3, HSC metadetect, Rubin) or
        ``"lensfit"`` (KiDS-1000, CFHTLenS).
    shape_weight_col : str
        Column carrying the catalog-published per-object shape weight.
        Default ``"shear_weight"``.
    R11_col, R22_col : str
        Metacal response columns. Used only when ``kind == "metacal"``.
    R_S_col : str or None
        Optional selection-response column added to ``R_eff`` when
        ``kind == "metacal"``. ``None`` to skip.
    m_col : str
        Lensfit multiplicative-bias column. Used only when
        ``kind == "lensfit"``.
    sigma_e_cols : (str, str) or None
        Per-component shape-noise columns. When given, the row-level
        ``σ_e²`` is added in quadrature to ``R_eff²`` in the
        denominator. Default ``None`` (no shape-noise floor).
    name : str or None
        Override for ``repr``.
    """

    def __init__(
        self,
        kind: str,
        *,
        shape_weight_col: str = "shear_weight",
        R11_col: str = "R11",
        R22_col: str = "R22",
        R_S_col: Optional[str] = "R_S",
        m_col: str = "m_bias",
        sigma_e_cols: Optional[Tuple[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        if kind not in _ALLOWED_KINDS:
            raise ValueError(
                f"unknown ShearWeight kind {kind!r}; "
                f"allowed: {sorted(_ALLOWED_KINDS)}"
            )
        self.kind = kind
        self.shape_weight_col = shape_weight_col
        self.R11_col = R11_col
        self.R22_col = R22_col
        self.R_S_col = R_S_col
        self.m_col = m_col
        self.sigma_e_cols = sigma_e_cols
        self.name = name or f"shear_weight({kind})"

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.shape_weight_col not in df.columns:
            raise KeyError(
                f"ShearWeight: missing shape-weight column "
                f"{self.shape_weight_col!r}"
            )
        w = df[self.shape_weight_col].to_numpy(dtype=np.float64)
        if self.kind == "metacal":
            for c in (self.R11_col, self.R22_col):
                if c not in df.columns:
                    raise KeyError(
                        f"ShearWeight(metacal): missing response column {c!r}"
                    )
            r11 = df[self.R11_col].to_numpy(dtype=np.float64)
            r22 = df[self.R22_col].to_numpy(dtype=np.float64)
            r_eff = 0.5 * (r11 + r22)
            if (
                self.R_S_col is not None
                and self.R_S_col in df.columns
            ):
                r_eff = r_eff + df[self.R_S_col].to_numpy(dtype=np.float64)
        else:  # lensfit
            if self.m_col not in df.columns:
                raise KeyError(
                    f"ShearWeight(lensfit): missing bias column {self.m_col!r}"
                )
            m = df[self.m_col].to_numpy(dtype=np.float64)
            r_eff = 1.0 + m
        denom = r_eff * r_eff
        if self.sigma_e_cols is not None:
            for c in self.sigma_e_cols:
                if c not in df.columns:
                    raise KeyError(
                        f"ShearWeight: missing sigma_e column {c!r}"
                    )
            s1 = df[self.sigma_e_cols[0]].to_numpy(dtype=np.float64)
            s2 = df[self.sigma_e_cols[1]].to_numpy(dtype=np.float64)
            denom = denom + s1 * s1 + s2 * s2
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(denom > 0, w / denom, 0.0)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_shear_weight.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/combine/weights/shear.py test/test_shear_weight.py
git commit -m "phase19/T2: ShearWeight (metacal + lensfit) with optional sigma_e quadrature"
```

---

## Task 3: `PipBitweightWeight`

**Files:**
- Create: `oneuniverse/combine/weights/pip.py`
- Create: `test/test_pip_bitweight.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_pip_bitweight.py
"""Phase 19 T3 — PipBitweightWeight expands BITWEIGHTS: i8[64]."""
import numpy as np
import pandas as pd
import pytest

from oneuniverse.combine.weights.pip import PipBitweightWeight


def _bitcount(arr: np.ndarray) -> np.ndarray:
    return np.unpackbits(arr.view(np.uint8), axis=-1).sum(axis=-1)


def test_fraction_mode_counts_set_bits():
    rows = np.array([
        np.zeros(1, dtype="i8"),
        np.array([-1], dtype="i8"),    # 64 set bits
    ], dtype=object)
    df = pd.DataFrame({"BITWEIGHTS": rows})
    w = PipBitweightWeight(bitweights_col="BITWEIGHTS").compute(df)
    np.testing.assert_allclose(w, np.array([0.0, 1.0]), rtol=1e-6)


def test_fraction_intermediate_value():
    # 32 set bits → 0.5
    val = np.int64(0x00000000FFFFFFFF)
    df = pd.DataFrame({"BITWEIGHTS": [np.array([val], dtype="i8")]})
    w = PipBitweightWeight().compute(df)
    np.testing.assert_allclose(w, np.array([32.0 / 64.0]), rtol=1e-6)


def test_realisations_mode_returns_per_row_array():
    rows = [np.array([0], dtype="i8"), np.array([-1], dtype="i8")]
    df = pd.DataFrame({"BITWEIGHTS": rows})
    w = PipBitweightWeight(mode="realisations").compute(df)
    assert w.shape == (2, 64)
    assert (w[0] == 0).all()
    assert (w[1] == 1).all()


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        PipBitweightWeight(mode="bogus")


def test_missing_column_raises():
    with pytest.raises(KeyError):
        PipBitweightWeight().compute(pd.DataFrame({"x": [1, 2]}))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_pip_bitweight.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/combine/weights/pip.py
"""
oneuniverse.combine.weights.pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pairwise-inverse-probability (PIP) bitweight expansion for DESI
clustering.

DESI ships ``BITWEIGHTS: i8[64]`` per object: bit ``k`` is 1 iff the
object passed fiber assignment in realisation ``k``. Two output modes:

* ``"fraction"`` (default): per-row fractional weight
  ``count_set_bits / 64`` — a scalar between 0 and 1 suitable as a
  drop-in object weight.
* ``"realisations"``: per-row ``(64,)`` array of 0/1 floats, one per
  PIP realisation, for jackknife-style accumulators.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight

_ALLOWED_MODES = frozenset({"fraction", "realisations"})


class PipBitweightWeight(Weight):
    """PIP bitweight expansion of ``i8[N]`` arrays.

    Parameters
    ----------
    bitweights_col : str
        Column carrying the per-row ``i8[N]`` BITWEIGHTS payload.
        Default ``"BITWEIGHTS"``.
    mode : str
        ``"fraction"`` (default) or ``"realisations"``.
    name : str or None
        Override for ``repr``.
    """

    def __init__(
        self,
        bitweights_col: str = "BITWEIGHTS",
        mode: str = "fraction",
        name: Optional[str] = None,
    ) -> None:
        if mode not in _ALLOWED_MODES:
            raise ValueError(
                f"unknown PipBitweightWeight mode {mode!r}; "
                f"allowed: {sorted(_ALLOWED_MODES)}"
            )
        self.bitweights_col = bitweights_col
        self.mode = mode
        self.name = name or f"pip({mode})"

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.bitweights_col not in df.columns:
            raise KeyError(
                f"PipBitweightWeight: missing column "
                f"{self.bitweights_col!r}"
            )
        rows = df[self.bitweights_col].to_numpy()
        n_rows = len(rows)
        # Each row is an array of int64s; flatten to 1-D int64 buffer.
        first = np.asarray(rows[0], dtype="i8")
        n_ints = first.size
        n_bits = 64 * n_ints
        stacked = np.empty((n_rows, n_ints), dtype="i8")
        for i, r in enumerate(rows):
            stacked[i, :] = np.asarray(r, dtype="i8").reshape(-1)
        bits = np.unpackbits(
            stacked.view(np.uint8).reshape(n_rows, -1),
            axis=1,
        ).astype(np.float64)
        # bits has shape (n_rows, 8 * n_ints * 8) — but unpackbits already
        # returns one bit per byte, so reshape gives n_rows × (64 * n_ints).
        bits = bits.reshape(n_rows, n_bits)
        if self.mode == "fraction":
            return bits.sum(axis=1) / float(n_bits)
        return bits
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_pip_bitweight.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/combine/weights/pip.py test/test_pip_bitweight.py
git commit -m "phase19/T3: PipBitweightWeight (fraction + realisations) over i8[N] BITWEIGHTS"
```

---

## Task 4: Re-export new weights through the package API

**Files:**
- Modify: `oneuniverse/combine/weights/__init__.py`
- Modify: `oneuniverse/combine/__init__.py`

- [ ] **Step 1: Inspect the current export pattern**

```bash
grep -n "^from \|__all__" oneuniverse/combine/weights/__init__.py | head
grep -n "^from \|__all__" oneuniverse/combine/__init__.py | head
```

- [ ] **Step 2: Extend `combine/weights/__init__.py`**

Append the new imports to the existing import block, e.g.:

```python
from oneuniverse.combine.weights.pip import PipBitweightWeight
from oneuniverse.combine.weights.shear import ShearWeight
```

Extend `__all__` (if defined) with the same two names.

- [ ] **Step 3: Extend `combine/__init__.py`**

Add to the existing chain:

```python
from oneuniverse.combine.weights import (
    ...,
    PipBitweightWeight,
    ShearWeight,
)
```

Extend the package-level `__all__` accordingly.

- [ ] **Step 4: Smoke test the imports**

```bash
python3 -c "from oneuniverse.combine import ShearWeight, PipBitweightWeight; print('OK', ShearWeight, PipBitweightWeight)"
```

Expected: `OK <class 'ShearWeight'> <class 'PipBitweightWeight'>`.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/combine/__init__.py oneuniverse/combine/weights/__init__.py
git commit -m "phase19/T4: re-export ShearWeight + PipBitweightWeight at combine top-level"
```

---

## Task 5: `default_weight_for` accepts `sub_kind`

**Files:**
- Modify: `oneuniverse/combine/weights/registry.py`
- Create: `test/test_registry_sub_kind.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_registry_sub_kind.py
"""Phase 19 T5 — registry key widens to (survey_type, sub_kind, z_type)."""
import pytest

from oneuniverse.combine.weights import default_weight_for
from oneuniverse.combine.weights.registry import (
    register_default,
    unregister_default,
)


def test_backward_compat_two_key_call_still_works():
    w = default_weight_for("spectroscopic", "spec")
    assert w is not None


def test_explicit_none_sub_kind_matches_two_key_default():
    w = default_weight_for("spectroscopic", "spec", sub_kind=None)
    assert w is not None


def test_register_sub_kind_specific_default():
    from oneuniverse.combine.weights.ivar import InverseVarianceWeight

    register_default(
        "spectroscopic", "spec",
        lambda: InverseVarianceWeight("z_spec_err", floor=0.01,
                                      name="ivar(z_spec,BGS_BRIGHT)"),
        sub_kind="BGS_BRIGHT",
    )
    try:
        w = default_weight_for("spectroscopic", "spec",
                               sub_kind="BGS_BRIGHT")
        assert "BGS_BRIGHT" in repr(w)
        # Fallback to default when sub_kind is unknown.
        fallback = default_weight_for("spectroscopic", "spec",
                                      sub_kind="BGS_FAINT")
        assert "BGS_BRIGHT" not in repr(fallback)
    finally:
        unregister_default("spectroscopic", "spec", sub_kind="BGS_BRIGHT")


def test_register_rejects_duplicate_sub_kind():
    from oneuniverse.combine.weights.ivar import InverseVarianceWeight

    register_default(
        "spectroscopic", "spec",
        lambda: InverseVarianceWeight("z_spec_err"),
        sub_kind="DUP",
    )
    try:
        with pytest.raises(ValueError, match="already"):
            register_default(
                "spectroscopic", "spec",
                lambda: InverseVarianceWeight("z_spec_err"),
                sub_kind="DUP",
            )
    finally:
        unregister_default("spectroscopic", "spec", sub_kind="DUP")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_registry_sub_kind.py -v
```

Expected: `TypeError: default_weight_for() got an unexpected keyword argument 'sub_kind'`.

- [ ] **Step 3: Extend the registry signature**

Replace the body of `oneuniverse/combine/weights/registry.py` with
the following — keeping the original two-key entries intact:

```python
"""
oneuniverse.combine.weights.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Opinionated default-weight factory keyed on
``(survey_type, sub_kind, z_type)``. ``sub_kind=None`` is the original
two-key behaviour and stays the fallback when no sub-species match
is registered. Sub-kind keys let surveys split a single
``(survey_type, z_type)`` into species like DESI ``BGS_BRIGHT`` vs
``BGS_FAINT`` or DES Y3 ``METACAL`` vs ``MCAL2`` while keeping the
top-level default intact.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from oneuniverse.combine.weights.base import Weight
from oneuniverse.combine.weights.ivar import InverseVarianceWeight

Key = Tuple[str, Optional[str], str]
Factory = Callable[[], Weight]


def _ivar_spec() -> Weight:
    return InverseVarianceWeight("z_spec_err", name="ivar(z_spec)")


def _ivar_phot() -> Weight:
    return InverseVarianceWeight("z_phot_err", name="ivar(z_phot)")


def _ivar_pec() -> Weight:
    return InverseVarianceWeight("velocity_error", name="ivar(vpec)")


def _ivar_pdf_width() -> Weight:
    from oneuniverse.combine.weights.pdf import PdfWidthIVarWeight
    return PdfWidthIVarWeight(std_column="z_pdf_std")


_DEFAULTS: Dict[Key, Factory] = {
    ("spectroscopic", None, "spec"): _ivar_spec,
    ("photometric", None, "phot"): _ivar_phot,
    ("peculiar_velocity", None, "pec"): _ivar_pec,
    ("photometric", None, "phot_pdf"): _ivar_pdf_width,
}


def default_weight_for(
    survey_type: str,
    z_type: str,
    *,
    sub_kind: Optional[str] = None,
) -> Weight:
    """Return the recommended default :class:`Weight`.

    Resolution order:

    1. ``(survey_type, sub_kind, z_type)`` if ``sub_kind`` is not None.
    2. ``(survey_type, None, z_type)`` (the canonical default).
    """
    if sub_kind is not None:
        key = (survey_type, sub_kind, z_type)
        if key in _DEFAULTS:
            return _DEFAULTS[key]()
    key = (survey_type, None, z_type)
    try:
        return _DEFAULTS[key]()
    except KeyError:
        raise KeyError(
            f"No default weight registered for "
            f"(survey_type={survey_type!r}, sub_kind={sub_kind!r}, "
            f"z_type={z_type!r}). Known keys: {sorted(_DEFAULTS)}"
        ) from None


def register_default(
    survey_type: str,
    z_type: str,
    factory: Factory,
    *,
    sub_kind: Optional[str] = None,
) -> None:
    """Register a default :class:`Weight` factory for
    ``(survey_type, sub_kind, z_type)``. Default ``sub_kind=None``
    matches the canonical pre-Phase-19 contract.
    """
    key = (survey_type, sub_kind, z_type)
    if key in _DEFAULTS:
        raise ValueError(
            f"register_default: {key!r} is already registered "
            f"(call unregister_default first if you intend to replace it)"
        )
    _DEFAULTS[key] = factory


def unregister_default(
    survey_type: str,
    z_type: str,
    *,
    sub_kind: Optional[str] = None,
) -> None:
    """Remove the default factory for
    ``(survey_type, sub_kind, z_type)``.
    """
    key = (survey_type, sub_kind, z_type)
    del _DEFAULTS[key]
```

- [ ] **Step 4: Run the new tests + pre-existing registry tests**

```bash
pytest test/test_registry_sub_kind.py test/test_weights_registry_public.py test/test_weight.py test/test_pdf_weights.py -q
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/combine/weights/registry.py \
        test/test_registry_sub_kind.py
git commit -m "phase19/T5: default_weight_for keys on (survey_type, sub_kind, z_type); sub_kind=None is the canonical fallback"
```

---

## Task 6: Visual diagnostic

**Files:**
- Create: `test/test_visual_phase19.py`

- [ ] **Step 1: Write the test**

```python
# test/test_visual_phase19.py
"""Phase 19 visual diagnostic — ShearWeight vs raw shape weight + PIP histogram."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.combine.weights.pip import PipBitweightWeight  # noqa: E402
from oneuniverse.combine.weights.shear import ShearWeight  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase19_visual(tmp_path):
    rng = np.random.default_rng(0)
    n = 1000
    shape_w = rng.uniform(0.2, 1.0, n).astype("f4")
    R11 = rng.normal(0.7, 0.05, n).astype("f4")
    R22 = rng.normal(0.7, 0.05, n).astype("f4")
    R_S = rng.normal(0.05, 0.01, n).astype("f4")
    df = pd.DataFrame({
        "shear_weight": shape_w,
        "R11": R11, "R22": R22, "R_S": R_S,
    })
    metacal = ShearWeight(kind="metacal").compute(df)

    rng2 = np.random.default_rng(1)
    bits = rng2.integers(0, 2**63 - 1, size=n, dtype="i8")
    pip_df = pd.DataFrame({
        "BITWEIGHTS": [np.array([b], dtype="i8") for b in bits],
    })
    pip = PipBitweightWeight().compute(pip_df)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].hist(shape_w, bins=40, color="tab:gray", alpha=0.7,
               label="shape_weight")
    ax[0].hist(metacal, bins=40, color="tab:blue", alpha=0.7,
               label="metacal ShearWeight")
    ax[0].set_xlabel("weight")
    ax[0].set_ylabel("count")
    ax[0].legend()
    ax[0].set_title("Shape weight vs metacal-calibrated weight")

    R_eff = 0.5 * (R11 + R22) + R_S
    ax[1].scatter(R_eff, metacal, s=4, alpha=0.4)
    ax[1].set_xlabel("R_eff")
    ax[1].set_ylabel("metacal ShearWeight")
    ax[1].set_title("Weight inversely scales as R_eff²")

    ax[2].hist(pip, bins=40, color="tab:red", alpha=0.8)
    ax[2].set_xlabel("PIP fraction (set bits / 64)")
    ax[2].set_ylabel("count")
    ax[2].set_title("PipBitweightWeight (fraction mode)")

    fig.tight_layout()
    out_png = OUT / "phase19_shear_and_pip_weights.png"
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

- [ ] **Step 2: Run the test**

```bash
pytest test/test_visual_phase19.py -v
```

Expected: pass; PNG ≥ 30 kB.

- [ ] **Step 3: Commit**

```bash
git add test/test_visual_phase19.py \
        test/test_output/phase19_shear_and_pip_weights.png
git commit -m "phase19/T6: visual diagnostic — shear + PIP weights"
```

---

## Task 7: Docs + plan-README + audit cross-ref

**Files:**
- Modify: `oneuniverse/CLAUDE.md`
- Modify: `plans/README.md`
- Modify: `research/schema_generalisation_audit.md`

- [ ] **Step 1: CLAUDE.md**

Under the existing weight-primitive list, append:

```
- `ShearWeight(kind="metacal" | "lensfit", …)` — propagates shape
  noise + calibration response: `w = shear_weight / (R_eff² + σ_e²)`
  with `R_eff = (R11+R22)/2 + R_S` (metacal) or `1 + m_bias`
  (lensfit). DES Y3 / KiDS-1000 / HSC-Y3 / Rubin.
- `PipBitweightWeight(mode="fraction" | "realisations")` — expand
  DESI `BITWEIGHTS: i8[64]` into a per-row fraction (default) or a
  `(64,)` realisation array for jackknife accumulation.
- `default_weight_for(survey_type, z_type, *, sub_kind=None)` —
  registry key widens to `(survey_type, sub_kind, z_type)`;
  `sub_kind=None` matches the canonical pre-Phase-19 contract.
```

Update the schema-group list (if present) to mention `shear`.

- [ ] **Step 2: plans/README.md**

```
| 19 | Shear column group + `ShearWeight` + `PipBitweightWeight` + sub-species registry key | **complete (2026-05-29, NNN/NNN tests green)** |
```

- [ ] **Step 3: research/schema_generalisation_audit.md**

Replace the existing "Phase 19 —" bullet with:

```
- **Phase 19 — Shear column group + weight expansion.** Landed
  2026-05-29. Adds `SHEAR_COLUMNS` schema group,
  `ShearWeight(kind="metacal" | "lensfit", sigma_e_cols=…)`,
  `PipBitweightWeight(mode="fraction" | "realisations")`, and
  registry sub-species key
  `(survey_type, sub_kind, z_type)`. No OUF bump (all schema
  additions are optional). See
  [`../plans/2026-05-29-phase19-shear-and-pip-weights.md`](../plans/2026-05-29-phase19-shear-and-pip-weights.md).
```

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/CLAUDE.md plans/README.md \
        research/schema_generalisation_audit.md
git commit -m "docs(phase19): shear group, ShearWeight, PipBitweightWeight, sub_kind registry"
```

---

## Task 8: Close-out

- [ ] **Step 1: Run the full suite**

```bash
pytest -q 2>&1 | tail -3
```

Expected: green. Record the count (Phase 18 baseline 450 + ~24 new).

- [ ] **Step 2: Replace `NNN/NNN` in plans/README.md.**

- [ ] **Step 3: Update memory**

Append to
`/home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`:

```markdown
## Phase 19 — Shear column group + weight expansion (complete 2026-05-29)

- New `SHEAR_COLUMNS` schema group:
  `e1 / e2 / e1_err / e2_err / R11 / R22 / R12 / R21 / R_S / m_bias /
  c1_bias / c2_bias / shear_weight`. All optional.
- `ShearWeight(kind="metacal" | "lensfit", sigma_e_cols=…)` ships in
  `oneuniverse.combine.weights.shear`; propagates calibration
  response + optional shape-noise quadrature.
- `PipBitweightWeight(mode="fraction" | "realisations")` ships in
  `oneuniverse.combine.weights.pip`; expands DESI BITWEIGHTS via
  ``numpy.unpackbits``.
- `default_weight_for(...)` registry key widens to
  ``(survey_type, sub_kind, z_type)``; ``sub_kind=None`` retains
  the canonical pre-Phase-19 contract.
- No OUF format bump (additions are all optional schema columns).
- Tests: NNN/NNN green.
- Per-phase plan:
  `plans/2026-05-29-phase19-shear-and-pip-weights.md`.
```

- [ ] **Step 4: Final commit**

```bash
git add plans/README.md \
        /home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md
git commit -m "phase19: close-out — shear group + weights + sub_kind registry, NNN tests green"
```

---

## Self-review checklist

- [ ] No cosmology metadata added anywhere.
- [ ] `SHEAR_COLUMNS` entries are all `required=False`.
- [ ] `ShearWeight(kind="metacal")` recovers `w / R_eff²` with
      `R_eff = 0.7` for a synthetic `(R11=R22=0.7, R_S=0)` row.
- [ ] `ShearWeight(kind="lensfit")` uses `(1 + m_bias)`.
- [ ] `PipBitweightWeight(mode="fraction")` returns `0.0` for an
      all-zero BITWEIGHTS row and `1.0` for an all-ones row.
- [ ] `default_weight_for("spectroscopic", "spec")` still works
      without the new `sub_kind` kwarg.
- [ ] Visual PNG `phase19_shear_and_pip_weights.png` ≥ 30 kB.

## Spec-coverage map

| Requirement | Task |
|---|---|
| `SHEAR_COLUMNS` group | T1 |
| `ShearWeight` (metacal + lensfit + sigma_e) | T2 |
| `PipBitweightWeight` (fraction + realisations) | T3 |
| Top-level re-exports | T4 |
| Registry sub-species key | T5 |
| Visual diagnostic | T6 |
| Docs + plan README + audit | T7 |
| Close-out + memory | T8 |
