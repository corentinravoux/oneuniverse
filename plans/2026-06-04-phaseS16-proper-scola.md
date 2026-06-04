# Phase S16 — Proper sCOLA (the proof of concept)

> **STATUS 2026-06-04 — T1 DONE; coupling (T3/T4) STILL NOT beating uncoupled.**
> ✅ **T1:** `pm_force_isolated` — Hockney zero-padded open Poisson, **no
> periodic images** (1/r², far-field 4× below periodic), verified.
> ⚠️ **Isolated Poisson alone is NOT sufficient.** An isolated-solver 1LPT-COLA
> tile still helped at buffer 16 (+0.05) yet **hurt at buffer 32 (−0.20)** —
> inconsistent, like S11b. Still missing: **(T2) 2LPT** far field; **(T3) full
> Dirichlet BC** (set the tile *potential* at the boundary to the far field,
> not merely an open force) + a **consistent open deposit**. The **toy PM mesh
> resolution** (N=64) may also cap any tile gain — a reportable outcome.
> **Conclusion:** T1 lands (reusable open Poisson solver); the validating
> coupling needs T2+T3 and is **deferred** rather than shipped worse-than-
> uncoupled.

> Implements correct sub-box sCOLA so selective resimulation **beats** the
> uncoupled run at a small buffer — validating the digital-twin core. Built on
> the focused research ([`research/2026-06-04-scola-focused-research.md`](../research/2026-06-04-scola-focused-research.md)).
> linear (full-sim, far field) + fast-PM (tile near field) stand in for real
> codes. Replaces the deferred S11b T3+.

## The decisive fix
S11b ran tiles as **periodic** boxes → worse than uncoupled. Proper sCOLA
(Tassev–Eisenstein 2015; Leclercq+ 2020) needs:
1. an **isolated (zero-padded) Poisson solve** with the tile boundary set to the
   far field — **Dirichlet BCs, not periodic**;
2. a **2LPT** far field (not 1LPT);
3. **buffer ≈ rms displacement** (~20–25 Mpc/h).

---

## Tasks (TDD; linear+PM dummy)

### T1 — Isolated (zero-padded) Poisson solver
**Files:** `pm/poisson.py` `pm_force_isolated`; test `test/test_pm_isolated.py`.
- Hockney/James zero-padding: deposit on the tile grid, embed in a 2× zero-padded
  grid, FFT-Poisson, crop back → an **open** (non-periodic) force with no
  periodic images.
- **Test:** a point mass in the tile gives a force ∝ 1/r² with **no wrap-around**
  (the periodic solver shows images at the box edge; the isolated one does not).

### T2 — 2LPT displacement
**Files:** `pm/lpt.py` `lpt_2nd` (Ψ₁, Ψ₂); test `test/test_pm_lpt.py`.
- Ψ₁ = Zel'dovich (have it); Ψ₂ from the 2LPT source
  `δ⁽²⁾ = Σ_{i<j}(Φ_,ii Φ_,jj − Φ_,ij²)`. Growth D₂ ≈ −3/7 D₁² Ωm^(−1/143).
- **Test:** 2LPT improves the low-k displacement vs Zel'dovich (cross-corr with a
  full-PM displacement higher at fixed early z); reduces to Zel'dovich as δ→0.

### T3 — Dirichlet far-field boundary on the tile
**Files:** `resim/scola.py`; test `test/test_resim_scola.py`.
- Solve the tile near field with `pm_force_isolated`, then **add the far-field
  force** (from the global 2LPT potential, evaluated on the tile) as the
  large-scale background — Dirichlet boundary = the far field at the tile edge.
- The COLA residual is kicked by `F_near_isolated − F_LPT_near`; the far-field
  2LPT carried in the drift.
- **Test:** the tile result is **insensitive to the tile being embedded
  anywhere** in the parent (translation invariance up to the parent field) —
  the periodic-image artifact is gone.

### T4 — sCOLA beats uncoupled at small buffer (the proof of concept)
**Files:** `resim/scola.py` `scola_run_coupled`; extend `test_resim_scola.py`.
- Full pipeline: global 2LPT (coarse) → tile (target+buffer) init at high z →
  COLA residual leapfrog with isolated Poisson + far-field boundary → trim.
- **Test (the headline):** at **buffer ≈ rms displacement**, the sCOLA inner
  region matches the full reference **and beats the uncoupled S8.5 run at the
  same buffer** (the opposite of the S11b failure). Convergence: error drops with
  buffer, plateauing near buffer≈displacement.

### T5 — Bounded-memory + multi-tile, and close-out
- The global 2LPT is coarse/cheap; the tile near-field PM is local → peak memory
  ≈ tile, not box (`tracemalloc` test). Tile the box → `scola_run_coupled` each →
  `merge_fields` (S12) → global field matches full reference.
- Demo: overlay **uncoupled vs sCOLA** convergence (sCOLA reaches accuracy at a
  smaller buffer) — the inverse of notebook 05's "not working" plot. Update the
  audit, notebooks, roadmap, memory. Full suite green.

---

## Success criteria
- Isolated Poisson: no periodic images (T1).
- 2LPT improves the far field (T2).
- **sCOLA inner region beats uncoupled at buffer ≈ rms displacement** (T4) — the
  decisive validation that selective resimulation works as designed.
- Tile peak memory ≈ tile size; multi-tile merge reproduces the full box.

## Honesty / scope
- Even correct sCOLA drops super-parent-box modes (irreducible).
- This is 1-level (no nested zoom); combine with S10 `refine_ic` for
  higher-resolution tiles.
- If T4 still fails to beat uncoupled after the isolated-Poisson + 2LPT fix,
  that is itself a reportable result (the dummy PM's force resolution may
  dominate) — document, don't force.
