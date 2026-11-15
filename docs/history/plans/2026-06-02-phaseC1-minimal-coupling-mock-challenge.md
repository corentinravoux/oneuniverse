# Phase C1 — Minimal data↔sim coupling (the mock challenge)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans / subagent-driven-development. Checkbox steps. Pulled **ahead** of S5–S8 as the feasibility keystone (close the scientific loop on the dummy before further substrate polish).

**Goal:** Demonstrate, end to end on the dummy, the **data↔simulation
coupling**: inject a known field (truth), **mock-observe** it as biased
tracers with shot noise, **constrain** the underlying field from the mock
data (Wiener filter / constrained realisation), and **verify** recovery
against the truth via the cross-correlation `r(k)`. This is the mock
challenge from the ADR — the single artifact that turns "two databases"
into "a data-driven twin", with a measurable feasibility number.

**Architecture:** New `oneuniverse.twin` subpackage — the **coupling layer**
the ADR mandates (it may import both `simulation` and, later, `data`;
`oneuniverse.simulation` stays Rule-1 clean). MVP is all **linear theory +
FFT**, deterministic, dependency-light (numpy). "Data" is a synthetic
Poisson sampler for now; real Pillar-1 selection is the first *data*
complexification. The constrained field is, by construction, the IC the
fast-PM mini-sim (S8.1) will later resimulate — so this demo is the front
of the resimulation loop.

**Tech stack:** numpy (FFT), matplotlib (plots); imports
`oneuniverse.simulation.linear` for P(k) + the truth field. **No
`oneuniverse.data` import yet** (MVP). Tests: pytest.

**The physics (Wiener filter).** Observed tracer overdensity
`δ_g = b·δ_m + ε`, shot noise power `N = 1/n̄`. Minimum-variance estimate of
the matter field: `δ̂_m(k) = [b·P_m(k) / (b²·P_m(k) + N)] · δ_g(k)`. Low k
(signal-dominated) → `δ̂_m → δ_m`; high k (noise-dominated) → suppressed.
Feasibility metric: `r(k) = Re⟨δ̂_m δ_m^truth*⟩ / √(P̂ P_truth)` per |k| bin
→ 1 at low k, falling where shot noise kills information.

---

## File Structure

- Create: `oneuniverse/twin/__init__.py`
- Create: `oneuniverse/twin/mock_observe.py` — biased Poisson tracer sampler.
- Create: `oneuniverse/twin/wiener.py` — Wiener reconstruction (+ optional CR).
- Create: `oneuniverse/twin/verify.py` — `cross_correlation`, `power_ratio`.
- Create: `oneuniverse/twin/mock_challenge.py` — the loop driver + metrics.
- Create: `scripts/mock_challenge_demo.py` — run + plots → Science dir.
- Tests: `test/test_twin_mock_observe.py`, `test_twin_wiener.py`,
  `test_twin_verify.py`, `test_twin_mock_challenge.py`,
  `test_visual_mock_challenge.py`.

**Test-case axes (we build cases around the demo):** tracer density `n̄`,
bias `b`, grid `n_grid`, seed; later (complexify) a selection **mask**,
redshift-space, multiple tracers (data side), and feeding the constrained
field into the PM mini-sim (sim side).

## Pre-flight

- [ ] **Step 0:** `cd Packages/oneuniverse && pytest test/test_lin_*.py -q | tail -2` (linear sim green).

---

## Task 1 — Mock observation (biased Poisson tracers)

**Files:** `twin/__init__.py`, `twin/mock_observe.py`; test
`test/test_twin_mock_observe.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_twin_mock_observe.py
"""Phase C1 T1 — mock biased Poisson tracers."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.mock_observe import mock_tracer_field


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_mean_density_and_overdensity():
    c = _cosmo()
    d = generate_density_field(c, box_size=256.0, n_grid=64, z=0.0, seed=1)
    obs = mock_tracer_field(d, box_size=256.0, nbar=5e-2, bias=1.5, seed=2)
    # mean tracer overdensity ~ 0 (Poisson around 1+bδ, δ mean 0)
    assert abs(float(obs["delta_g"].mean())) < 0.1
    # more tracers land in overdense cells: positive correlation with δ
    corr = np.corrcoef(obs["counts"].ravel(), d.ravel())[0, 1]
    assert corr > 0.2


def test_reproducible_and_nonnegative_counts():
    c = _cosmo()
    d = generate_density_field(c, box_size=256.0, n_grid=32, z=0.0, seed=1)
    a = mock_tracer_field(d, box_size=256.0, nbar=1e-2, bias=2.0, seed=7)
    b = mock_tracer_field(d, box_size=256.0, nbar=1e-2, bias=2.0, seed=7)
    np.testing.assert_array_equal(a["counts"], b["counts"])
    assert a["counts"].min() >= 0
```

- [ ] **Step 2:** Run → FAIL (import).

- [ ] **Step 3: Implement**

```python
# oneuniverse/twin/mock_observe.py
"""Mock 'observation': sample biased tracers from a truth density field.

A stand-in for the Pillar-1 data side. Expected count per cell
λ = n̄_cell · max(0, 1 + b·δ); counts ~ Poisson(λ). Returns the counts and
the observed tracer overdensity δ_g = counts/n̄_cell − 1. Linear-bias +
clip (simplest; lognormal/HOD are later complexifications).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def mock_tracer_field(delta, *, box_size, nbar, bias=1.0, seed=0,
                      mask=None) -> Dict[str, np.ndarray]:
    d = np.asarray(delta, dtype=np.float64)
    n = d.shape[0]
    v_cell = (box_size / n) ** 3
    nbar_cell = nbar * v_cell
    rng = np.random.default_rng(seed)
    lam = nbar_cell * np.clip(1.0 + bias * d, 0.0, None)
    if mask is not None:
        lam = lam * np.asarray(mask, dtype=np.float64)
    counts = rng.poisson(lam).astype(np.float64)
    # observed overdensity (within the mask if given)
    delta_g = counts / nbar_cell - 1.0
    return {"counts": counts, "delta_g": delta_g,
            "nbar": float(nbar), "bias": float(bias),
            "nbar_cell": float(nbar_cell)}
```

- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `phaseC1/T1: oneuniverse.twin + mock biased Poisson tracer field`.

---

## Task 2 — Wiener-filter reconstruction

**Files:** `twin/wiener.py`; test `test/test_twin_wiener.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_twin_wiener.py
"""Phase C1 T2 — Wiener reconstruction of the matter field."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.wiener import wiener_reconstruct


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_noise_free_recovers_truth():
    c = _cosmo()
    box, n, b = 256.0, 64, 1.5
    truth = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=3)
    # noise-free observed field: delta_g = b * truth (tests the operator)
    delta_g = b * truth
    rec = wiener_reconstruct(delta_g, c, box_size=box, nbar=1e9, bias=b)
    # with negligible shot noise the matter field is recovered
    assert np.corrcoef(rec.ravel(), truth.ravel())[0, 1] > 0.98


def test_high_noise_suppresses_small_scales():
    c = _cosmo()
    box, n, b = 256.0, 64, 1.5
    truth = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=3)
    rng = np.random.default_rng(0)
    delta_g = b * truth + rng.standard_normal(truth.shape) * 2.0
    rec = wiener_reconstruct(delta_g, c, box_size=box, nbar=1e-3, bias=b)
    # reconstruction has less small-scale power than the noisy input
    assert rec.var() < delta_g.var()
```

- [ ] **Step 2:** Run → FAIL.

- [ ] **Step 3: Implement**

```python
# oneuniverse/twin/wiener.py
"""Wiener-filter reconstruction of the matter field from a tracer field.

δ̂_m(k) = [b·P_m(k) / (b²·P_m(k) + N)] · δ_g(k), N = 1/n̄ shot noise.
Full-box (periodic) → diagonal in Fourier space. Masked / non-periodic
reconstruction (messy, mode-coupling) is a later complexification.
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.power_spectrum import linear_power


def _kgrid(n, box):
    kx = np.fft.fftfreq(n, d=box / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    return np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2)


def wiener_reconstruct(delta_g, cosmo: CosmologySpec, *, box_size, nbar,
                       bias=1.0, z=0.0) -> np.ndarray:
    d = np.asarray(delta_g, dtype=np.float64)
    n = d.shape[0]
    kmag = _kgrid(n, box_size)
    Pm = np.zeros_like(kmag)
    nz = kmag > 0
    Pm[nz] = linear_power(kmag[nz], cosmo, z=z)
    N = 1.0 / nbar
    gain = np.zeros_like(Pm)
    denom = bias * bias * Pm + N
    gain[denom > 0] = (bias * Pm[denom > 0]) / denom[denom > 0]
    dk = np.fft.rfftn(d)
    rec = np.fft.irfftn(gain * dk, s=(n, n, n))
    return rec
```

- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `phaseC1/T2: Wiener-filter matter-field reconstruction (diagonal, full-box)`.

---

## Task 3 — Verification (cross-correlation r(k), power ratio)

**Files:** `twin/verify.py`; test `test/test_twin_verify.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_twin_verify.py
"""Phase C1 T3 — cross-correlation r(k)."""
import numpy as np

from oneuniverse.twin.verify import cross_correlation, power_ratio


def test_self_correlation_is_one():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((32, 32, 32))
    k, r = cross_correlation(f, f, box_size=200.0)
    assert np.all(r[np.isfinite(r)] > 0.999)


def test_uncorrelated_fields_near_zero():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 32, 32))
    b = rng.standard_normal((32, 32, 32))
    k, r = cross_correlation(a, b, box_size=200.0)
    # independent fields: |r| small in most bins
    assert np.nanmedian(np.abs(r)) < 0.2
```

- [ ] **Step 2:** Run → FAIL.

- [ ] **Step 3: Implement**

```python
# oneuniverse/twin/verify.py
"""Fourier-space verification: cross-correlation r(k) and power ratio."""
from __future__ import annotations

import numpy as np


def _bin_kgrid(n, box):
    kx = np.fft.fftfreq(n, d=box / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    return np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2)


def _bins(kmag, box, n):
    kf = 2.0 * np.pi / box
    kny = np.pi * n / box
    edges = np.arange(kf / 2, kny, kf)
    idx = np.digitize(kmag.ravel(), edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return idx, edges, centres


def cross_correlation(a, b, *, box_size):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = a.shape[0]
    ak = np.fft.rfftn(a); bk = np.fft.rfftn(b)
    kmag = _bin_kgrid(n, box_size)
    idx, edges, centres = _bins(kmag, box_size, n)
    cross = np.real(ak * np.conj(bk)).ravel()
    pa = (np.abs(ak) ** 2).ravel(); pb = (np.abs(bk) ** 2).ravel()
    r = np.full(len(centres), np.nan)
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() == 0:
            continue
        denom = np.sqrt(pa[m].sum() * pb[m].sum())
        if denom > 0:
            r[i - 1] = cross[m].sum() / denom
    return centres, r


def power_ratio(a, b, *, box_size):
    """Binned P_a(k)/P_b(k)."""
    n = a.shape[0]
    ak = np.fft.rfftn(a); bk = np.fft.rfftn(b)
    kmag = _bin_kgrid(n, box_size)
    idx, edges, centres = _bins(kmag, box_size, n)
    pa = (np.abs(ak) ** 2).ravel(); pb = (np.abs(bk) ** 2).ravel()
    ratio = np.full(len(centres), np.nan)
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() and pb[m].sum() > 0:
            ratio[i - 1] = pa[m].sum() / pb[m].sum()
    return centres, ratio
```

- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `phaseC1/T3: verification — cross-correlation r(k) + power ratio`.

---

## Task 4 — Mock-challenge loop driver

**Files:** `twin/mock_challenge.py`; test `test/test_twin_mock_challenge.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_twin_mock_challenge.py
"""Phase C1 T4 — end-to-end mock challenge."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.twin.mock_challenge import run_mock_challenge


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_recovers_large_scales():
    res = run_mock_challenge(_cosmo(), box_size=256.0, n_grid=64,
                             nbar=5e-2, bias=1.5, seed=11)
    k, r = res["k"], res["r"]
    # large scales recovered (r high), small scales lost to shot noise
    low = r[k < 0.05]
    high = r[k > 0.3]
    assert np.nanmedian(low) > 0.8
    assert np.nanmedian(high) < np.nanmedian(low)


def test_denser_survey_recovers_more():
    lo = run_mock_challenge(_cosmo(), box_size=256.0, n_grid=64,
                            nbar=5e-3, bias=1.5, seed=11)
    hi = run_mock_challenge(_cosmo(), box_size=256.0, n_grid=64,
                            nbar=1e-1, bias=1.5, seed=11)
    # higher number density → better mid-scale recovery
    band = (lo["k"] > 0.1) & (lo["k"] < 0.3)
    assert np.nanmedian(hi["r"][band]) > np.nanmedian(lo["r"][band])
```

- [ ] **Step 2:** Run → FAIL.

- [ ] **Step 3: Implement**

```python
# oneuniverse/twin/mock_challenge.py
"""End-to-end mock challenge: truth → mock-observe → constrain → verify.

The minimal data↔sim coupling loop on the dummy, where the truth is known
so recovery is measurable. Returns the cross-correlation r(k) (the
feasibility number) plus the fields for plotting.
"""
from __future__ import annotations

from typing import Dict

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.mock_observe import mock_tracer_field
from oneuniverse.twin.verify import cross_correlation, power_ratio
from oneuniverse.twin.wiener import wiener_reconstruct


def run_mock_challenge(cosmo: CosmologySpec, *, box_size, n_grid, nbar,
                       bias=1.5, z=0.0, seed=0) -> Dict:
    truth = generate_density_field(cosmo, box_size=box_size, n_grid=n_grid,
                                   z=z, seed=seed)
    obs = mock_tracer_field(truth, box_size=box_size, nbar=nbar, bias=bias,
                            seed=seed + 1)
    rec = wiener_reconstruct(obs["delta_g"], cosmo, box_size=box_size,
                             nbar=nbar, bias=bias, z=z)
    k, r = cross_correlation(rec, truth, box_size=box_size)
    _, ratio = power_ratio(rec, truth, box_size=box_size)
    return {"truth": truth, "delta_g": obs["delta_g"], "rec": rec,
            "k": k, "r": r, "power_ratio": ratio,
            "nbar": nbar, "bias": bias, "box_size": box_size,
            "n_grid": n_grid}
```

- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Commit** `phaseC1/T4: mock-challenge loop driver (truth→observe→constrain→verify)`.

---

## Task 5 — Demo script + plots + close-out

**Files:** `scripts/mock_challenge_demo.py`,
`test/test_visual_mock_challenge.py`; `oneuniverse/twin/__init__.py`
exports; `CLAUDE.md`, `plans/README.md`, memory.

- [ ] **Step 1:** `mock_challenge_demo.py` runs the loop on a real grid
  (e.g. box 512, n_grid 128, sweep nbar ∈ {1e-3, 5e-3, 5e-2}), writes to
  `/home/ravoux/Documents/Science/Cosmography/oneuniverse_simulation/mock_challenge/`:
  - r(k) vs k for each nbar (the headline feasibility plot),
  - truth / observed / reconstruction slices (visual recovery),
  - δ_rec vs δ_truth scatter (large-scale cells),
  - `RESULTS.json` (r(k) tables + the k where r=0.5 per nbar).
- [ ] **Step 2:** Visual test asserts the plots exist + are non-trivial.
- [ ] **Step 3:** `twin/__init__.py` exports `mock_tracer_field`,
  `wiener_reconstruct`, `cross_correlation`, `power_ratio`,
  `run_mock_challenge`. Full suite green.
- [ ] **Step 4:** Docs — `CLAUDE.md` (new `oneuniverse/twin/` bullet:
  coupling layer, may import data+simulation, MVP mock challenge),
  `plans/README.md` (add Phase C1 row), memory append.
- [ ] **Step 5: Commit** `phaseC1/T5: mock-challenge demo + plots + docs; minimal data↔sim coupling closed`.

---

## Self-review checklist

- [ ] `oneuniverse.twin` imports `simulation` (and later `data`); the Rule-1
      guard over `simulation/` stays green (twin/ is not scanned).
- [ ] Loop is deterministic (seeded) and recovers large scales (r→1 low k).
- [ ] Feasibility number reported: the scale where r(k)=0.5 per survey n̄.
- [ ] Denser survey → better recovery (monotone, sanity).
- [ ] Demo plots non-trivial; results saved next to the data.

## Complexification roadmap (the test cases we grow around this)

**Data side:** selection **mask** (non-periodic WF / apodisation) → real
Pillar-1 survey geometry + n(z) (twin starts importing `oneuniverse.data`)
→ redshift-space distortions → multiple tracers (multi-tracer WF).
**Sim side:** constrained **realisation** (Hoffman–Ribak: add unconstrained
variance back) → feed the constrained field as the **IC into the fast-PM
mini-sim** (S8.1) → nonlinear forward model → Gate-2 post-run check.
**Inference side:** swap the Wiener step for an external engine / SBI via the
`ForwardEngine` contract (proves generality with ≥2 engines).
