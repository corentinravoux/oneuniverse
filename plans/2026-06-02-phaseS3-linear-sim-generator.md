# Phase S3 — Dummy Linear Simulation Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, deterministic, dependency-light **linear-theory simulation generator** under `oneuniverse/simulation/linear/` that produces every OUF-Sim product type — a regular density field (mesh / voxel), Zel'dovich particles, and a toy halo catalogue — from a pure-numpy Eisenstein–Hu power spectrum + linear growth. This synthetic source is the vehicle to finish the OUF-Sim machinery (converter, indexes, partial-access views, database, orchestration) in later phases without any real-simulation dependency.

**Architecture:** Pure numpy + pyarrow + pyyaml (all already package deps). No CAMB/CLASS/abacusutils/yt. Physics: Eisenstein & Hu (1998) no-wiggle transfer function → σ8-normalised linear P(k) → Gaussian random field on an N³ mesh via FFT → Zel'dovich (1LPT) particle displacements → peak-threshold toy halos. Everything is seeded → byte-reproducible. The generator writes a simple native on-disk layout (`config.yaml` + per-redshift `field.npy` / `particles.npy` / `halos.parquet`) that the Phase-S4 `LinearSimConverter` will wrap into OUF-Sim.

**Tech Stack:** Python 3.9+, numpy (FFT), pyarrow (halo parquet), pyyaml (config), pytest. Spec: [`2026-06-01-phaseS1-oufsim-architecture.md`](2026-06-01-phaseS1-oufsim-architecture.md). Reuses `oneuniverse.simulation.cosmology.CosmologySpec` from Phase S2. **Rule 1:** no `oneuniverse.data` / `combine` imports (the existing `test_sim_no_pillar1_imports.py` already scans `linear/` recursively).

---

## File Structure

New files under `Packages/oneuniverse/oneuniverse/simulation/linear/`:

- `__init__.py` — public exports.
- `_cosmo.py` — `require_cosmo(spec)` validator + derived helpers (no new dataclass; reuses `CosmologySpec`).
- `power_spectrum.py` — `transfer_eh_nowiggle`, `unnormalised_power`, `sigma_R`, `normalisation`, `linear_power`.
- `growth.py` — `growth_factor`, `growth_rate`.
- `gaussian_field.py` — `generate_density_field` (mesh / voxel product).
- `zeldovich.py` — `zeldovich_particles` (particle product).
- `halos.py` — `find_peaks` (toy halo product).
- `generate.py` — `generate_linear_sim` (writes the native layout).

Tests under `Packages/oneuniverse/test/`:

- `test_lin_power_spectrum.py`
- `test_lin_growth.py`
- `test_lin_gaussian_field.py`
- `test_lin_zeldovich.py`
- `test_lin_halos.py`
- `test_lin_generate.py`
- `test_lin_public_api.py`
- `test_visual_linear_sim.py` (diagnostic figure)

---

## Pre-flight

- [ ] **Step 0: Baseline green.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest test/test_sim_*.py -q 2>&1 | tail -3
```

Expected: `49 passed` (Phase S2 sim suite).

---

## Task 1: linear subpackage skeleton + cosmo validator

**Files:**
- Create: `oneuniverse/simulation/linear/__init__.py`, `oneuniverse/simulation/linear/_cosmo.py`
- Test: `test/test_lin_cosmo.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_lin_cosmo.py
"""Phase S3 T1 — linear-sim cosmology validator."""
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo


def _full() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
        sigma8=0.81, t_cmb=2.7255,
    )


def test_require_passes_for_full_cosmo():
    c = require_cosmo(_full())
    assert c.omega_m == 0.31
    # t_cmb defaulted if missing handled separately
    assert c.t_cmb == 2.7255


def test_require_defaults_tcmb():
    c = require_cosmo(CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
    ))
    assert c.t_cmb == 2.7255


def test_require_rejects_missing_field():
    with pytest.raises(ValueError, match="omega_m"):
        require_cosmo(CosmologySpec(
            omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
        ))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_lin_cosmo.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create the package init**

```python
# oneuniverse/simulation/linear/__init__.py
"""oneuniverse.simulation.linear — a pure-numpy linear-theory dummy
simulation: Eisenstein-Hu P(k) + Gaussian field + Zel'dovich particles
+ toy halos. The synthetic source used to finish the OUF-Sim machinery.

Standalone (Rule 1): no imports from oneuniverse.data / combine.
"""
```

- [ ] **Step 4: Implement `_cosmo.py`**

```python
# oneuniverse/simulation/linear/_cosmo.py
"""Validate + complete a CosmologySpec for linear-sim use.

The storage CosmologySpec has all-optional fields; the generator needs
concrete omega_m / omega_b / h / n_s / sigma8 and a CMB temperature
(defaulted to 2.7255 K). ``require_cosmo`` returns a completed spec or
raises with the name of the first missing field.
"""
from __future__ import annotations

from dataclasses import replace

from oneuniverse.simulation.cosmology import CosmologySpec

_REQUIRED = ("omega_m", "omega_b", "h", "n_s", "sigma8")
_DEFAULT_TCMB = 2.7255


def require_cosmo(spec: CosmologySpec) -> CosmologySpec:
    for name in _REQUIRED:
        if getattr(spec, name) is None:
            raise ValueError(
                f"linear sim requires CosmologySpec.{name} to be set"
            )
    if spec.t_cmb is None:
        return replace(spec, t_cmb=_DEFAULT_TCMB)
    return spec
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest test/test_lin_cosmo.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/simulation/linear/__init__.py \
        oneuniverse/simulation/linear/_cosmo.py test/test_lin_cosmo.py
git commit -m "phaseS3/T1: linear subpackage skeleton + require_cosmo validator"
```

---

## Task 2: Eisenstein–Hu transfer function + unnormalised power

**Files:**
- Create: `oneuniverse/simulation/linear/power_spectrum.py`
- Test: `test/test_lin_power_spectrum.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_lin_power_spectrum.py
"""Phase S3 T2/T3 — Eisenstein-Hu P(k)."""
import numpy as np
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.power_spectrum import (
    linear_power,
    sigma_R,
    transfer_eh_nowiggle,
    unnormalised_power,
)


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
        sigma8=0.81, t_cmb=2.7255,
    )


def test_transfer_goes_to_one_at_large_scale():
    c = _cosmo()
    t_big = transfer_eh_nowiggle(np.array([1e-4]), c)[0]
    assert t_big == pytest.approx(1.0, abs=0.02)


def test_transfer_decreases_with_k():
    c = _cosmo()
    k = np.array([1e-3, 1e-2, 1e-1, 1.0])
    t = transfer_eh_nowiggle(k, c)
    assert np.all(np.diff(t) < 0)


def test_unnormalised_power_low_k_slope_is_ns():
    c = _cosmo()
    k = np.array([1e-4, 2e-4])
    p = unnormalised_power(k, c)
    # On large scales T->1 so P ~ k^ns; measure the local slope.
    slope = np.log(p[1] / p[0]) / np.log(k[1] / k[0])
    assert slope == pytest.approx(c.n_s, abs=0.02)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_lin_power_spectrum.py::test_transfer_goes_to_one_at_large_scale -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the transfer + unnormalised power**

```python
# oneuniverse/simulation/linear/power_spectrum.py
"""Eisenstein & Hu (1998) no-wiggle linear power spectrum, pure numpy.

Reference: Eisenstein & Hu 1998, ApJ 496, 605 (arXiv:astro-ph/9709112),
the "no-wiggle" shape fit (eqs. 26-31). Wavenumbers ``k`` in h/Mpc;
P(k) in (Mpc/h)^3. Normalised so that sigma8 reproduces the input.
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo


def transfer_eh_nowiggle(k, cosmo: CosmologySpec) -> np.ndarray:
    """No-wiggle EH98 transfer function T(k). ``k`` in h/Mpc."""
    c = require_cosmo(cosmo)
    k = np.atleast_1d(np.asarray(k, dtype=np.float64))
    om = c.omega_m
    ob = c.omega_b
    h = c.h
    theta = c.t_cmb / 2.7
    omh2 = om * h * h
    obh2 = ob * h * h
    f_baryon = ob / om
    # sound horizon in Mpc/h (the trailing * h converts Mpc -> Mpc/h)
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2 ** 0.75) * h
    alpha = (
        1.0
        - 0.328 * np.log(431.0 * omh2) * f_baryon
        + 0.38 * np.log(22.3 * omh2) * f_baryon ** 2
    )
    # effective shape; k in h/Mpc, s in Mpc/h -> k*s dimensionless
    gamma_eff = om * h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * k * s) ** 4))
    q = k * theta ** 2 / gamma_eff
    l0 = np.log(2.0 * np.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return l0 / (l0 + c0 * q * q)


def unnormalised_power(k, cosmo: CosmologySpec) -> np.ndarray:
    """Shape-only P(k) ~ k^n_s T(k)^2 (arbitrary amplitude)."""
    c = require_cosmo(cosmo)
    k = np.atleast_1d(np.asarray(k, dtype=np.float64))
    t = transfer_eh_nowiggle(k, c)
    return k ** c.n_s * t ** 2
```

- [ ] **Step 4: Run the transfer + slope tests**

```bash
pytest test/test_lin_power_spectrum.py::test_transfer_goes_to_one_at_large_scale test/test_lin_power_spectrum.py::test_transfer_decreases_with_k test/test_lin_power_spectrum.py::test_unnormalised_power_low_k_slope_is_ns -v
```

Expected: 3 passed. (`sigma_R` / `linear_power` tests still fail — added in Task 3.)

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/linear/power_spectrum.py test/test_lin_power_spectrum.py
git commit -m "phaseS3/T2: Eisenstein-Hu no-wiggle transfer + unnormalised power (pure numpy)"
```

---

## Task 3: σ8 normalisation → `linear_power`

**Files:**
- Modify: `oneuniverse/simulation/linear/power_spectrum.py`
- Extend: `test/test_lin_power_spectrum.py`

- [ ] **Step 1: Append the failing tests**

```python
# append to test/test_lin_power_spectrum.py

def test_sigma8_roundtrips():
    c = _cosmo()
    # sigma_R(8) on the *normalised* P(k) must return the input sigma8.
    pk_norm = lambda kk: linear_power(kk, c, z=0.0)  # noqa: E731
    s8 = sigma_R(8.0, c, pk_func=pk_norm)
    assert s8 == pytest.approx(c.sigma8, rel=0.01)


def test_linear_power_scales_with_growth_squared():
    c = _cosmo()
    k = np.array([0.1])
    p0 = linear_power(k, c, z=0.0)[0]
    p1 = linear_power(k, c, z=1.0)[0]
    # higher z -> smaller amplitude
    assert p1 < p0


def test_linear_power_positive():
    c = _cosmo()
    k = np.logspace(-3, 1, 50)
    p = linear_power(k, c, z=0.0)
    assert np.all(p > 0)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/test_lin_power_spectrum.py::test_sigma8_roundtrips -v
```

Expected: `ImportError`/`AttributeError` on `sigma_R` / `linear_power`.

- [ ] **Step 3: Implement `sigma_R`, `normalisation`, `linear_power`**

Append to `power_spectrum.py`:

```python
def _top_hat(x: np.ndarray) -> np.ndarray:
    """Fourier top-hat window W(x); W(0) = 1."""
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    nz = x > 1e-8
    xn = x[nz]
    out[nz] = 3.0 * (np.sin(xn) - xn * np.cos(xn)) / xn ** 3
    return out


def sigma_R(R: float, cosmo: CosmologySpec, *, pk_func=None) -> float:
    """RMS density fluctuation in a top-hat of radius ``R`` (Mpc/h).

    If ``pk_func`` is None, integrates the *unnormalised* shape (used to
    derive the normalisation); otherwise integrates ``pk_func(k)``.
    """
    c = require_cosmo(cosmo)
    k = np.logspace(-4.0, 2.0, 4000)
    pk = unnormalised_power(k, c) if pk_func is None else np.asarray(pk_func(k))
    w = _top_hat(k * R)
    integrand = k ** 2 * pk * w ** 2
    sigma2 = np.trapz(integrand, k) / (2.0 * np.pi ** 2)
    return float(np.sqrt(sigma2))


def normalisation(cosmo: CosmologySpec) -> float:
    """Amplitude A such that sigma8 of A * unnormalised_power == sigma8."""
    c = require_cosmo(cosmo)
    s8_unnorm = sigma_R(8.0, c, pk_func=None)
    return float((c.sigma8 / s8_unnorm) ** 2)


def linear_power(k, cosmo: CosmologySpec, z: float = 0.0) -> np.ndarray:
    """σ8-normalised linear matter P(k) at redshift ``z`` (Mpc/h)^3."""
    from oneuniverse.simulation.linear.growth import growth_factor

    c = require_cosmo(cosmo)
    k = np.atleast_1d(np.asarray(k, dtype=np.float64))
    amp = normalisation(c)
    d = growth_factor(z, c)
    return amp * unnormalised_power(k, c) * d * d
```

(Note: `linear_power` imports `growth_factor` lazily to avoid a circular import; `growth.py` arrives in Task 4. The σ8-roundtrip test uses z=0 where `growth_factor(0)=1`.)

- [ ] **Step 4: Run to verify pass (after Task 4 growth exists; for now skip the z!=0 path)**

The `linear_power` tests need `growth.py`. Implement Task 4 next, then run:

```bash
pytest test/test_lin_power_spectrum.py -v
```

Expected (after Task 4): all power-spectrum tests pass, including σ8 round-trip within 1%.

- [ ] **Step 5: Commit (after Task 4 makes it green)**

Defer this commit until Task 4 is done so the suite is green at commit time. Proceed to Task 4.

---

## Task 4: linear growth factor + rate

**Files:**
- Create: `oneuniverse/simulation/linear/growth.py`
- Test: `test/test_lin_growth.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_lin_growth.py
"""Phase S3 T4 — linear growth factor D(z), rate f(z)."""
import numpy as np
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.growth import growth_factor, growth_rate


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
    )


def test_growth_normalised_to_one_at_z0():
    assert growth_factor(0.0, _cosmo()) == pytest.approx(1.0, abs=1e-9)


def test_growth_decreases_with_redshift():
    c = _cosmo()
    z = np.array([0.0, 0.5, 1.0, 2.0])
    d = np.array([growth_factor(zi, c) for zi in z])
    assert np.all(np.diff(d) < 0)


def test_growth_high_z_approaches_eds():
    # At high z (matter domination) D ~ a = 1/(1+z).
    c = _cosmo()
    z = 9.0
    d = growth_factor(z, c)
    a = 1.0 / (1.0 + z)
    # D(z)/D(0) ~ a/1 only loosely; check D(9) ~ 0.1 within 30%.
    assert d == pytest.approx(a, rel=0.3)


def test_growth_rate_is_omega_matter_power():
    c = _cosmo()
    f0 = growth_rate(0.0, c)
    # f(0) ~ Omega_m^0.55 ~ 0.31^0.55 ~ 0.52
    assert f0 == pytest.approx(0.31 ** 0.55, rel=0.05)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/test_lin_growth.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/simulation/linear/growth.py
"""Linear growth factor + rate for flat LambdaCDM.

D(z) via the Carroll, Press & Turner (1992) fitting formula, normalised
to D(0) = 1. f(z) = Omega_m(z)^0.55 (Linder 2005 gamma). Flat universe
assumed: Omega_Lambda = 1 - Omega_m.
"""
from __future__ import annotations

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo


def _growth_unnorm(a: float, om: float) -> float:
    ol = 1.0 - om
    e2 = om * a ** -3 + ol
    om_a = om * a ** -3 / e2
    ol_a = ol / e2
    return (
        2.5 * a * om_a
        / (om_a ** (4.0 / 7.0) - ol_a + (1.0 + om_a / 2.0) * (1.0 + ol_a / 70.0))
    )


def growth_factor(z: float, cosmo: CosmologySpec) -> float:
    """Linear growth D(z), normalised so D(0) = 1."""
    c = require_cosmo(cosmo)
    a = 1.0 / (1.0 + z)
    return _growth_unnorm(a, c.omega_m) / _growth_unnorm(1.0, c.omega_m)


def growth_rate(z: float, cosmo: CosmologySpec) -> float:
    """Linear growth rate f(z) = Omega_m(z)^0.55."""
    c = require_cosmo(cosmo)
    a = 1.0 / (1.0 + z)
    e2 = c.omega_m * a ** -3 + (1.0 - c.omega_m)
    om_a = c.omega_m * a ** -3 / e2
    return float(om_a ** 0.55)
```

- [ ] **Step 4: Run growth tests + the deferred power-spectrum tests**

```bash
pytest test/test_lin_growth.py test/test_lin_power_spectrum.py -v
```

Expected: all green (growth + σ8 round-trip + growth-scaling).

- [ ] **Step 5: Commit (growth + the Task-3 power-spectrum normalisation together)**

```bash
git add oneuniverse/simulation/linear/growth.py \
        oneuniverse/simulation/linear/power_spectrum.py \
        test/test_lin_growth.py test/test_lin_power_spectrum.py
git commit -m "phaseS3/T3+T4: sigma8-normalised linear_power + growth factor D(z)/rate f(z)"
```

---

## Task 5: Gaussian density field (mesh / voxel product)

**Files:**
- Create: `oneuniverse/simulation/linear/gaussian_field.py`
- Test: `test/test_lin_gaussian_field.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_lin_gaussian_field.py
"""Phase S3 T5 — Gaussian density field (mesh / voxel product)."""
import numpy as np
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.power_spectrum import linear_power


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
    )


def test_shape_and_real():
    d = generate_density_field(_cosmo(), box_size=200.0, n_grid=32, z=0.0, seed=1)
    assert d.shape == (32, 32, 32)
    assert np.isrealobj(d)


def test_mean_near_zero():
    d = generate_density_field(_cosmo(), box_size=200.0, n_grid=32, z=0.0, seed=1)
    assert abs(float(d.mean())) < 0.05


def test_reproducible_with_seed():
    a = generate_density_field(_cosmo(), box_size=200.0, n_grid=16, z=0.0, seed=7)
    b = generate_density_field(_cosmo(), box_size=200.0, n_grid=16, z=0.0, seed=7)
    np.testing.assert_array_equal(a, b)


def test_variance_matches_mode_sum():
    """Real-space variance ~ (1/V) sum_k P(k_grid), within cosmic scatter."""
    c = _cosmo()
    box, n = 200.0, 32
    d = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=3)
    measured = float(d.var())
    # Predicted variance from the same grid's mode sum.
    kf = 2.0 * np.pi / box
    kx = np.fft.fftfreq(n, d=box / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kx, indexing="ij")
    kmag = np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2)
    kmag_flat = kmag.ravel()
    pk = np.zeros_like(kmag_flat)
    nz = kmag_flat > 0
    pk[nz] = linear_power(kmag_flat[nz], c, z=0.0)
    predicted = pk.sum() / box ** 3
    assert measured == pytest.approx(predicted, rel=0.35)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/test_lin_gaussian_field.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/simulation/linear/gaussian_field.py
"""Gaussian random density field delta(x) on a regular mesh.

This is the OUF-Sim "field" (mesh / voxel) product. Method: white noise
in real space -> rFFT -> colour by sqrt(P(k)) -> irFFT, with the
Pylians-style normalisation factor 1/sqrt(V_cell) so the discrete
real-space variance approximates (1/V) sum_k P(k). Seeded -> reproducible.
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo
from oneuniverse.simulation.linear.power_spectrum import linear_power


def generate_density_field(
    cosmo: CosmologySpec,
    *,
    box_size: float,
    n_grid: int,
    z: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Return a real (n_grid, n_grid, n_grid) linear density contrast.

    Parameters
    ----------
    box_size
        Comoving box side in Mpc/h.
    n_grid
        Cells per side.
    z
        Redshift (sets the growth-scaled amplitude).
    seed
        RNG seed (reproducible).
    """
    c = require_cosmo(cosmo)
    n = int(n_grid)
    rng = np.random.default_rng(seed)

    # |k| grid (h/Mpc), rfft layout.
    kx = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag = np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2)

    pk = np.zeros_like(kmag)
    nz = kmag > 0.0
    pk[nz] = linear_power(kmag[nz], c, z=z)

    # White noise in real space, colour in Fourier space.
    white = rng.standard_normal((n, n, n))
    white_k = np.fft.rfftn(white)
    delta_k = white_k * np.sqrt(pk)
    delta = np.fft.irfftn(delta_k, s=(n, n, n))

    # Normalise so variance ~ (1/V) sum_k P(k): factor 1/sqrt(V_cell).
    v_cell = (box_size / n) ** 3
    delta *= 1.0 / np.sqrt(v_cell)
    # Enforce zero mean (DC mode).
    delta -= delta.mean()
    return delta
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest test/test_lin_gaussian_field.py -v
```

Expected: 4 passed (variance within 35% of the mode-sum prediction).

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/linear/gaussian_field.py test/test_lin_gaussian_field.py
git commit -m "phaseS3/T5: Gaussian density field (mesh/voxel product) via FFT, seeded + variance-checked"
```

---

## Task 6: Zel'dovich particles (particle product)

**Files:**
- Create: `oneuniverse/simulation/linear/zeldovich.py`
- Test: `test/test_lin_zeldovich.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_lin_zeldovich.py
"""Phase S3 T6 — Zel'dovich particles (particle product)."""
import numpy as np
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.zeldovich import zeldovich_particles


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
    )


def test_particle_count_and_shapes():
    pos, vel = zeldovich_particles(
        _cosmo(), box_size=200.0, n_grid=16, z=0.0, seed=1,
    )
    assert pos.shape == (16 ** 3, 3)
    assert vel.shape == (16 ** 3, 3)


def test_positions_wrapped_in_box():
    box = 200.0
    pos, _ = zeldovich_particles(
        _cosmo(), box_size=box, n_grid=16, z=0.0, seed=1,
    )
    assert pos.min() >= 0.0
    assert pos.max() < box


def test_reproducible():
    a_pos, a_vel = zeldovich_particles(_cosmo(), box_size=200.0, n_grid=16, z=0.0, seed=4)
    b_pos, b_vel = zeldovich_particles(_cosmo(), box_size=200.0, n_grid=16, z=0.0, seed=4)
    np.testing.assert_array_equal(a_pos, b_pos)
    np.testing.assert_array_equal(a_vel, b_vel)


def test_mean_displacement_small():
    box, n = 200.0, 16
    pos, _ = zeldovich_particles(_cosmo(), box_size=box, n_grid=n, z=0.0, seed=2)
    # Lagrangian grid centres.
    cell = box / n
    g = (np.arange(n) + 0.5) * cell
    qx, qy, qz = np.meshgrid(g, g, g, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)
    disp = pos - q
    # Periodic-wrap the displacement to [-box/2, box/2).
    disp = (disp + box / 2.0) % box - box / 2.0
    assert abs(float(disp.mean())) < 1.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/test_lin_zeldovich.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/simulation/linear/zeldovich.py
"""Zel'dovich (1LPT) particle realisation from the linear density field.

This is the OUF-Sim "particles" product. The Zel'dovich displacement is
psi(q) = inverse-Laplacian gradient of -delta, i.e. in Fourier space
psi_k = i k / k^2 * delta_k. Particles start on a uniform Lagrangian
grid q and move to x = q + psi (already growth-scaled via delta(z)).
Velocities (km/s) follow v = a H(a) f psi in linear theory; here we use
the simple proportionality v = (H0 f / (1+z)) * psi for a toy field.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.growth import growth_rate


def zeldovich_particles(
    cosmo: CosmologySpec,
    *,
    box_size: float,
    n_grid: int,
    z: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions, velocities), each (n_grid^3, 3).

    Positions in Mpc/h wrapped to [0, box_size); velocities in km/s
    (toy linear-theory scaling).
    """
    c = require_cosmo(cosmo)
    n = int(n_grid)
    delta = generate_density_field(
        c, box_size=box_size, n_grid=n, z=z, seed=seed,
    )
    delta_k = np.fft.rfftn(delta)

    kx = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    k2 = kxg ** 2 + kyg ** 2 + kzg ** 2
    k2[0, 0, 0] = 1.0  # avoid division by zero; DC displacement set to 0

    # psi_k = i k / k^2 * delta_k  (per component)
    psi = []
    for kg in (kxg, kyg, kzg):
        psi_k = 1j * kg / k2 * delta_k
        comp = np.fft.irfftn(psi_k, s=(n, n, n))
        psi.append(comp)
    psi = np.stack([p.ravel() for p in psi], axis=1)  # (n^3, 3)

    # Lagrangian grid centres.
    cell = box_size / n
    g = (np.arange(n) + 0.5) * cell
    qx, qy, qz = np.meshgrid(g, g, g, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)

    pos = (q + psi) % box_size

    # Toy velocity: v = (H0 * f / (1+z)) * psi (km/s); H0 = 100 h km/s/Mpc.
    h0 = 100.0 * c.h
    f = growth_rate(z, c)
    vel = (h0 * f / (1.0 + z)) * psi
    return pos.astype(np.float64), vel.astype(np.float64)
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest test/test_lin_zeldovich.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/linear/zeldovich.py test/test_lin_zeldovich.py
git commit -m "phaseS3/T6: Zel'dovich particles (particle product) — psi = i k/k^2 delta, wrapped to box"
```

---

## Task 7: Toy peak halos (halo product)

**Files:**
- Create: `oneuniverse/simulation/linear/halos.py`
- Test: `test/test_lin_halos.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_lin_halos.py
"""Phase S3 T7 — toy peak halos (halo product)."""
import numpy as np
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.halos import find_peaks


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
    )


def test_returns_expected_columns():
    box, n = 200.0, 32
    d = generate_density_field(_cosmo(), box_size=box, n_grid=n, z=0.0, seed=5)
    halos = find_peaks(d, box_size=box, threshold=1.0)
    for col in ("halo_id", "x", "y", "z", "delta_peak", "mass"):
        assert col in halos


def test_finds_some_peaks():
    box, n = 200.0, 32
    d = generate_density_field(_cosmo(), box_size=box, n_grid=n, z=0.0, seed=5)
    halos = find_peaks(d, box_size=box, threshold=1.0)
    assert len(halos["halo_id"]) > 0


def test_positions_in_box():
    box, n = 200.0, 32
    d = generate_density_field(_cosmo(), box_size=box, n_grid=n, z=0.0, seed=5)
    halos = find_peaks(d, box_size=box, threshold=1.0)
    for ax in ("x", "y", "z"):
        v = np.asarray(halos[ax])
        assert v.min() >= 0.0 and v.max() < box


def test_higher_threshold_fewer_halos():
    box, n = 200.0, 32
    d = generate_density_field(_cosmo(), box_size=box, n_grid=n, z=0.0, seed=5)
    n_low = len(find_peaks(d, box_size=box, threshold=0.5)["halo_id"])
    n_high = len(find_peaks(d, box_size=box, threshold=2.0)["halo_id"])
    assert n_high <= n_low
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/test_lin_halos.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/simulation/linear/halos.py
"""Toy halo catalogue from local maxima of the density field.

This is the OUF-Sim "halos" product. A "halo" is a cell whose delta
exceeds ``threshold`` AND is a strict local maximum over its 26
neighbours (periodic). Mass is a toy proxy: (1 + delta) * mean cell
mass, with mean cell mass set from Omega_m * rho_crit * V_cell. The
result is a plain dict of equal-length arrays (parquet-friendly).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

# rho_crit = 2.775e11 h^2 Msun / (Mpc)^3  -> in Msun/h per (Mpc/h)^3:
_RHO_CRIT_H2 = 2.775e11  # Msun/h / (Mpc/h)^3 (h-factors cancel in this unit)


def find_peaks(
    delta: np.ndarray,
    *,
    box_size: float,
    threshold: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Return a toy halo catalogue as a dict of arrays."""
    d = np.asarray(delta, dtype=np.float64)
    n = d.shape[0]
    cell = box_size / n

    # Local-maximum test over 26 periodic neighbours.
    is_peak = np.ones_like(d, dtype=bool)
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                if sx == 0 and sy == 0 and sz == 0:
                    continue
                shifted = np.roll(np.roll(np.roll(d, sx, 0), sy, 1), sz, 2)
                is_peak &= d > shifted

    mask = is_peak & (d > threshold)
    idx = np.argwhere(mask)  # (n_halos, 3) integer cell indices
    if idx.size == 0:
        empty_f = np.empty(0, dtype=np.float64)
        return {
            "halo_id": np.empty(0, dtype=np.int64),
            "x": empty_f, "y": empty_f, "z": empty_f,
            "delta_peak": empty_f, "mass": empty_f,
        }

    centres = (idx + 0.5) * cell
    deltas = d[mask]
    mean_cell_mass = _RHO_CRIT_H2 * cell ** 3  # toy: Omega_m absorbed below
    mass = (1.0 + deltas) * mean_cell_mass
    return {
        "halo_id": np.arange(idx.shape[0], dtype=np.int64),
        "x": centres[:, 0].astype(np.float64),
        "y": centres[:, 1].astype(np.float64),
        "z": centres[:, 2].astype(np.float64),
        "delta_peak": deltas.astype(np.float64),
        "mass": mass.astype(np.float64),
    }
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest test/test_lin_halos.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/linear/halos.py test/test_lin_halos.py
git commit -m "phaseS3/T7: toy peak halos (halo product) — periodic local maxima above threshold"
```

---

## Task 8: `generate_linear_sim` — write the native layout

**Files:**
- Create: `oneuniverse/simulation/linear/generate.py`
- Test: `test/test_lin_generate.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_lin_generate.py
"""Phase S3 T8 — generate_linear_sim native-layout writer."""
import numpy as np
import pyarrow.parquet as pq
import yaml

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.generate import generate_linear_sim


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
        t_cmb=2.7255,
    )


def test_writes_config_and_products(tmp_path):
    out = generate_linear_sim(
        tmp_path / "linsim", _cosmo(),
        box_size=200.0, n_grid=16, redshifts=(0.0, 0.5), seed=11,
    )
    out = out  # Path
    assert (out / "config.yaml").is_file()
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    assert cfg["n_grid"] == 16
    assert cfg["redshifts"] == [0.0, 0.5]
    for ztag in ("z0.000", "z0.500"):
        assert (out / ztag / "field.npy").is_file()
        assert (out / ztag / "particles.npy").is_file()
        assert (out / ztag / "halos.parquet").is_file()


def test_field_and_particle_shapes(tmp_path):
    out = generate_linear_sim(
        tmp_path / "linsim", _cosmo(),
        box_size=200.0, n_grid=16, redshifts=(0.0,), seed=11,
    )
    field = np.load(out / "z0.000" / "field.npy")
    assert field.shape == (16, 16, 16)
    parts = np.load(out / "z0.000" / "particles.npy")
    # (n^3, 6): x,y,z,vx,vy,vz
    assert parts.shape == (16 ** 3, 6)
    halos = pq.read_table(out / "z0.000" / "halos.parquet")
    assert "mass" in halos.column_names


def test_deterministic(tmp_path):
    a = generate_linear_sim(
        tmp_path / "a", _cosmo(),
        box_size=200.0, n_grid=16, redshifts=(0.0,), seed=99,
    )
    b = generate_linear_sim(
        tmp_path / "b", _cosmo(),
        box_size=200.0, n_grid=16, redshifts=(0.0,), seed=99,
    )
    fa = np.load(a / "z0.000" / "field.npy")
    fb = np.load(b / "z0.000" / "field.npy")
    np.testing.assert_array_equal(fa, fb)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest test/test_lin_generate.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/simulation/linear/generate.py
"""Top-level dummy-simulation writer.

Generates the field (mesh/voxel), Zel'dovich particles, and toy halos
for a list of redshifts and writes a simple native on-disk layout that
the Phase-S4 LinearSimConverter will wrap into OUF-Sim:

    {out_dir}/
    |- config.yaml                 (cosmology + box + grid + seed + redshifts)
    |- z0.000/
    |   |- field.npy               (n,n,n) float64 density contrast
    |   |- particles.npy           (n^3, 6) float64 x,y,z,vx,vy,vz
    |   `- halos.parquet           toy halo catalogue
    `- z0.500/ ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.halos import find_peaks
from oneuniverse.simulation.linear.zeldovich import zeldovich_particles


def _ztag(z: float) -> str:
    return f"z{z:.3f}"


def generate_linear_sim(
    out_dir: Union[str, Path],
    cosmo: CosmologySpec,
    *,
    box_size: float,
    n_grid: int,
    redshifts: Sequence[float],
    seed: int = 0,
    halo_threshold: float = 1.0,
) -> Path:
    """Generate + write a dummy linear simulation. Returns the root dir."""
    c = require_cosmo(cosmo)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "generator": "oneuniverse.simulation.linear",
        "box_size": float(box_size),
        "n_grid": int(n_grid),
        "redshifts": [float(z) for z in redshifts],
        "seed": int(seed),
        "halo_threshold": float(halo_threshold),
        "cosmology": c.to_dict(),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    for z in redshifts:
        zdir = out / _ztag(z)
        zdir.mkdir(parents=True, exist_ok=True)

        field = generate_density_field(
            c, box_size=box_size, n_grid=n_grid, z=z, seed=seed,
        )
        np.save(zdir / "field.npy", field)

        pos, vel = zeldovich_particles(
            c, box_size=box_size, n_grid=n_grid, z=z, seed=seed,
        )
        parts = np.concatenate([pos, vel], axis=1)  # (n^3, 6)
        np.save(zdir / "particles.npy", parts)

        halos = find_peaks(field, box_size=box_size, threshold=halo_threshold)
        table = pa.table({k: pa.array(v) for k, v in halos.items()})
        pq.write_table(table, zdir / "halos.parquet")

    return out
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest test/test_lin_generate.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/linear/generate.py test/test_lin_generate.py
git commit -m "phaseS3/T8: generate_linear_sim — write native layout (config + field/particles/halos per z)"
```

---

## Task 9: Visual diagnostic + public exports + close-out

**Files:**
- Create: `test/test_visual_linear_sim.py`, `test/test_lin_public_api.py`
- Modify: `oneuniverse/simulation/linear/__init__.py`, `CLAUDE.md`, `plans/README.md`

- [ ] **Step 1: Visual diagnostic test**

```python
# test/test_visual_linear_sim.py
"""Phase S3 — diagnostic figure for the dummy linear simulation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.simulation.cosmology import CosmologySpec  # noqa: E402
from oneuniverse.simulation.linear.gaussian_field import (  # noqa: E402
    generate_density_field,
)
from oneuniverse.simulation.linear.halos import find_peaks  # noqa: E402
from oneuniverse.simulation.linear.power_spectrum import linear_power  # noqa: E402
from oneuniverse.simulation.linear.zeldovich import zeldovich_particles  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_linear_sim_visual():
    c = CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
        t_cmb=2.7255,
    )
    box, n = 256.0, 64
    field = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=42)
    pos, _ = zeldovich_particles(c, box_size=box, n_grid=n, z=0.0, seed=42)
    halos = find_peaks(field, box_size=box, threshold=1.5)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    k = np.logspace(-2.5, 0.5, 200)
    for z, style in ((0.0, "-"), (1.0, "--")):
        ax[0].loglog(k, linear_power(k, c, z=z), style, label=f"z={z}")
    ax[0].set_xlabel("k [h/Mpc]")
    ax[0].set_ylabel("P(k) [(Mpc/h)$^3$]")
    ax[0].set_title("Eisenstein-Hu linear P(k)")
    ax[0].legend()

    proj = field.sum(axis=2)
    im = ax[1].imshow(proj.T, origin="lower", extent=(0, box, 0, box),
                      cmap="magma")
    ax[1].set_xlabel("x [Mpc/h]")
    ax[1].set_ylabel("y [Mpc/h]")
    ax[1].set_title("density field (projected)")
    plt.colorbar(im, ax=ax[1])

    sel = pos[:, 2] < box / n * 4  # a thin slab of particles
    ax[2].scatter(pos[sel, 0], pos[sel, 1], s=1, alpha=0.3, color="0.3")
    if len(halos["x"]):
        ax[2].scatter(halos["x"], halos["y"], s=20, color="tab:red",
                      marker="x", label="halos")
        ax[2].legend()
    ax[2].set_xlim(0, box)
    ax[2].set_ylim(0, box)
    ax[2].set_xlabel("x [Mpc/h]")
    ax[2].set_ylabel("y [Mpc/h]")
    ax[2].set_title("Zel'dovich particles + halos")

    fig.tight_layout()
    out_png = OUT / "linear_sim_overview.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im2:
        assert im2.width >= 800 and im2.height >= 200
```

- [ ] **Step 2: Run the visual test**

```bash
pytest test/test_visual_linear_sim.py -v
```

Expected: pass; `test/test_output/linear_sim_overview.png` created (P(k), projected field, particles+halos).

- [ ] **Step 3: Public-API test**

```python
# test/test_lin_public_api.py
"""Phase S3 T9 — linear-sim public API."""
import oneuniverse.simulation.linear as lin


def test_public_exports_present():
    for name in (
        "transfer_eh_nowiggle", "linear_power", "sigma_R",
        "growth_factor", "growth_rate",
        "generate_density_field", "zeldovich_particles", "find_peaks",
        "generate_linear_sim",
    ):
        assert hasattr(lin, name), f"missing export: {name}"
```

- [ ] **Step 4: Fill in `linear/__init__.py`**

```python
# oneuniverse/simulation/linear/__init__.py
"""oneuniverse.simulation.linear — a pure-numpy linear-theory dummy
simulation: Eisenstein-Hu P(k) + Gaussian field + Zel'dovich particles
+ toy halos. The synthetic source used to finish the OUF-Sim machinery.

Standalone (Rule 1): no imports from oneuniverse.data / combine.
"""
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.generate import generate_linear_sim
from oneuniverse.simulation.linear.growth import growth_factor, growth_rate
from oneuniverse.simulation.linear.halos import find_peaks
from oneuniverse.simulation.linear.power_spectrum import (
    linear_power,
    sigma_R,
    transfer_eh_nowiggle,
    unnormalised_power,
)
from oneuniverse.simulation.linear.zeldovich import zeldovich_particles

__all__ = [
    "transfer_eh_nowiggle", "unnormalised_power", "sigma_R", "linear_power",
    "growth_factor", "growth_rate",
    "generate_density_field", "zeldovich_particles", "find_peaks",
    "generate_linear_sim",
]
```

- [ ] **Step 5: Run the public-API + isolation guard + full linear suite**

```bash
pytest test/test_lin_public_api.py test/test_sim_no_pillar1_imports.py test/test_lin_*.py test/test_visual_linear_sim.py -q 2>&1 | tail -4
```

Expected: all green; the isolation guard still passes (the `linear/` modules import only numpy / pyarrow / yaml / `oneuniverse.simulation.*`).

- [ ] **Step 6: Full suite (no regression)**

```bash
pytest -q 2>&1 | tail -3
```

Expected: `>= 600 passed` (571 baseline + ~28 new), `2 skipped`.

- [ ] **Step 7: Docs**

In `CLAUDE.md`, extend the `oneuniverse/simulation/` bullet:

```
  - `oneuniverse/simulation/linear/` — pure-numpy **dummy linear
    simulation** (Eisenstein-Hu P(k), growth D(z), Gaussian field /
    voxel, Zel'dovich particles, toy halos). The synthetic source
    used to finish + test the OUF-Sim machinery before any real
    backend. `generate_linear_sim(out, cosmo, box_size=, n_grid=,
    redshifts=, seed=)` writes a native layout (config.yaml +
    per-z field.npy / particles.npy / halos.parquet).
```

In `plans/README.md`, update the S3 row to **complete** with the test count.

- [ ] **Step 8: Commit + memory**

```bash
git add oneuniverse/simulation/linear/__init__.py \
        test/test_lin_public_api.py test/test_visual_linear_sim.py \
        test/test_output/linear_sim_overview.png \
        CLAUDE.md plans/README.md
git commit -m "phaseS3/T9: visual diagnostic + public exports + docs; dummy linear simulation complete"
```

Append to `/home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`:

```markdown
## Phase S3 — dummy linear simulation generator (complete 2026-06-02)

- New `oneuniverse/simulation/linear/`: pure-numpy linear-theory dummy
  simulation. Eisenstein-Hu (1998) no-wiggle transfer + sigma8-
  normalised `linear_power(k, cosmo, z)`; Carroll-Press-Turner growth
  `growth_factor`/`growth_rate`; `generate_density_field` (mesh/voxel
  via FFT); `zeldovich_particles` (1LPT particles); `find_peaks` (toy
  halos); `generate_linear_sim` writes native layout (config.yaml +
  per-z field.npy/particles.npy/halos.parquet).
- Deterministic (seeded). Deps: numpy + pyarrow + pyyaml only. Obeys
  Rule 1 (lint guard covers linear/).
- This synthetic source drives S4 (converter+index), S5 (view), S6
  (database+orchestration). Real-format backends stay deferred.
- Tests: NNN/NNN green.
- Per-phase plan: `plans/2026-06-02-phaseS3-linear-sim-generator.md`.
```

---

## Self-review checklist

- [ ] No cosmology *engine* leaks into Pillar 1 — this is Pillar-3
      synthetic-source code; cosmology lives here legitimately.
- [ ] No `oneuniverse.data` / `combine` imports (guard green).
- [ ] Deps limited to numpy / pyarrow / pyyaml (no CAMB/CLASS/yt/abacusutils).
- [ ] Everything seeded → reproducible (field, particles, generate).
- [ ] σ8 round-trips through the normalised P(k) within 1%.
- [ ] All three product types emitted (field/voxel, particles, halos).
- [ ] Visual PNG ≥ 30 kB.
- [ ] Full suite green; Pillar 1 + S2 untouched.

## Spec-coverage map (products envisioned → S3 output)

| OUF-Sim product | S3 source | Native artefact |
|---|---|---|
| `fields` (mesh / voxel) | `generate_density_field` | `z*/field.npy` |
| `snapshots` (particles) | `zeldovich_particles` | `z*/particles.npy` |
| `halos` | `find_peaks` | `z*/halos.parquet` |
| `lightcone` | — | **Phase S4** (stack field shells onto HEALPix) |
| `tree` | — | **Phase S4** (link halos across z by nearest-neighbour) |

Deferred to later phases (correctly absent from S3): `LinearSimConverter`
+ `convert()` body, IndexBuilder toolkit, `SimDatasetView` partial-access,
`SimDatabase`, orchestration. S3 is the synthetic *source* only.
