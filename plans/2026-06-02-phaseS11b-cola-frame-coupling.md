# Phase S11b — COLA-frame coupling (the proper version)

> Replaces the deferred COLA part of S11. Linear (full-sim) + fast-PM
> (mini-sim) stand in for real codes; structure first. Implements the
> **LPT-subtraction frame** so the large scales (incl. the external tide) are
> carried analytically and the PM solves only the small-scale residual — no
> double-counting. This unlocks **smaller buffers** *and* **bounded-memory
> region-local resim** at once.

## Why S11's naive coupling failed (the point, restated)

The isolated sub-box already computes the large-scale modes that *fit inside
it*. The S11 experiment added the external large-scale displacement **on top,
at the end** → those modes got counted twice. Measured: +0.09 at small buffer
(little overlap) but **−0.08 at larger buffer** (more overlap). Inconsistent,
so it was not shipped.

## The fix: evolve the residual, subtract the LPT force

Write the trajectory as

    x(a) = q + D(a)·Ψ  +  s(a)
           └── x_LPT ──┘     └─ residual (PM)

- **x_LPT = q + D(a)·Ψ** is the analytic large-scale (Zel'dovich/1LPT) motion.
  Ψ is the **full-box** displacement (so it carries the super-buffer tide).
- **s(a)** is the small-scale residual, solved by the PM.

The residual equation of motion (peculiar, canonical momentum) is

    dp_s/da = (F_full − F_LPT)/(a²E) · (3/2)Ωₘ,     dx/da = (p_s)/(a³E) + dD/da·Ψ

where `F_full` = gravity from the actual particle positions (the existing PM
force) and **`F_LPT` = the linear force of the LPT density** (cheap — it is the
far-field, ∝ Ψ). Subtracting `F_LPT` is the no-double-count step: the residual
only ever feels the *non-LPT* (small-scale, mode-coupling) part of gravity, so
it stays small and few PM steps suffice, while the large-scale motion is added
once via the analytic `dD/da·Ψ` drift term.

This is the COLA method (Tassev, Zaldarriaga & Eisenstein 2013), 1LPT version.

---

## Tasks (TDD; linear+PM dummy)

### T1 — LPT background fields
**Files:** `resim/cola.py` (new); test `test/test_resim_cola.py`.
- `lpt_state(delta_z0, box, n_grid)` → the Zel'dovich displacement Ψ (reuse
  `pm.run._zeldovich_displacement`) and a callable for `x_LPT(a) = q + D(a)·Ψ`.
- `lpt_force(delta_z0, box, n_grid)` → `F_LPT`, the linear force of the LPT
  density (= the far-field force; reuse `pm.poisson.pm_force` on the LPT
  density `δ_LPT = D(a)·δ_z0`, scaled per step). At linear order
  `δ_LPT(a)=D(a)·δ_z0`, so `F_LPT(a) = D(a)·force_unit(δ_z0)`.
- **Test:** `F_LPT` reproduces the Zel'dovich growing-mode acceleration — i.e.
  a PM step driven by `F_LPT` alone advances a linear mode as `D(a)` (consistency
  with the analytic Ψ).

### T2 — COLA-framed PM integrator
**Files:** `resim/cola.py` `cola_run_pm`; test as above.
- KDK leapfrog on the **residual**: residual momentum `p_s` starts at 0; kick
  `p_s += (F_full − F_LPT)·∫da/(a²E)·(3/2)Ωₘ`; drift `x += p_s·∫da/(a³E) +
  ΔD·Ψ` (the analytic LPT increment).
- **Test (the headline COLA benefit):** `cola_run_pm` with **few steps (~5–10)**
  reproduces the full-PM large-scale field (low-k `r(k)` and `P(k)/P_full`
  within a few %) of a `run_pm` with many steps (~25). COLA's published
  property: large scales exact by construction, few steps needed.

### T3 — COLA coupling in the resim (smaller buffers)
**Files:** extend `resim/coupling.py` (`run_coupled(..., cola=True)`).
- The sub-box PM runs in the COLA frame using the **full-box** Ψ (super-buffer
  tide included) restricted to the buffer particles as the LPT background, and
  the local `F_LPT`.
- **Test:** COLA-coupled `run_coupled` reaches the **same inner-region accuracy
  as the S8.5 uncoupled run at a smaller buffer** (e.g. COLA buffer 16 ≈
  uncoupled buffer 32). Monotone: the buffer needed for a target `r_lowk` drops.

### T4 — Bounded-memory region-local resim (Rule 2/5)
**Files:** `resim/coupling.py`; test.
- Because Ψ/`F_LPT` are the *global, cheap* part (computed once from the
  full-sim far-field, partial-accessed per region) and the PM residual is
  *local*, the resim touches only the buffer region's arrays.
- **Test:** `tracemalloc` peak of the COLA region resim scales with the **buffer
  size, not the full box** (e.g. resimulating a small region of a large parent
  uses ≪ the full-box memory).

### T5 — Demo + close-out
- Extend `scripts/resim_feasibility.py`: overlay **uncoupled vs COLA** buffer-
  convergence curves (COLA reaches target accuracy at smaller buffer).
- Update the audit (`research/...physics-audit`) + roadmap; memory.
- Full suite green.

---

## Success criteria
- `cola_run_pm` few-step ≈ full-PM many-step on large scales (COLA property).
- `F_LPT` subtraction verified (no double-count: COLA result is *consistent*
  across buffer sizes, unlike the naive S11 injection).
- COLA coupling hits target accuracy at a **smaller buffer** than S8.5.
- Region resim peak memory bounded by the buffer, not the box.

## Physics honesty (carry forward from the audit)
- Even proper COLA drops modes **larger than the full-box** — irreducible, as
  always.
- 1LPT (Zel'dovich) COLA is implemented here; 2LPT improves accuracy further
  (a later refinement, not required for the structural demonstration).
- The external tide carried is the **full-box** Ψ; if the parent itself is a
  finite box, its own missing super-box tide is inherited (documented).
