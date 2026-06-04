# sCOLA focused research — what proper tile resimulation needs

Follows the 2026-06-02 feasibility study + the S11b failures. Goal: pin down
exactly what a *correct* sub-box sCOLA needs, since validating it is the proof
of concept for selective resimulation (the digital-twin core).

## 1. Why my S11b attempts failed (diagnosis)

All three S11b coupling attempts ran the tile as a **periodic** box (FFT
Poisson is periodic) and were worse than the plain uncoupled resim:
- naive end-of-run injection → double-counts;
- sub-grid-Ψ COLA tile (corr 0.09) → the sub-grid's own Zel'dovich Ψ is wrong
  near the tile edge (periodic FFT of a non-periodic patch);
- full-box-Ψ-override COLA tile (corr 0.34–0.42) → the tile PM force is still
  computed with **periodic** BCs, so the tile's own large-scale modes fight the
  injected far field.

**Root cause:** a periodic tile is the wrong boundary. The literature is
explicit on the fix.

## 2. What the literature says (the missing ingredients)

**Tassev & Eisenstein 2015** (arXiv:1502.07751) — sCOLA: the far field
(large-scale, perturbative LPT) and the near field (N-body) are **completely
decoupled**; the tile N-body solves only the near field. Accuracy is optimised
by **(a) a buffer region around each tile** and **(b) appropriate Dirichlet
boundary conditions** on the tile.

**Leclercq et al. 2020** (arXiv:2003.04925; A&A 2020) — perfectly-parallel
sCOLA: each tile is evolved **independently**; the key numbers/choices:
- **Dirichlet boundary conditions** on each sCOLA box (not periodic);
- **buffer ≈ 25 Mpc/h**, "a good compromise… roughly the maximum distance a
  particle travels from initial to final position" (i.e. ≈ the rms displacement
  — matching our feasibility estimate);
- the far field carried with **2LPT** (we used 1LPT/Zel'dovich);
- particles initialised with 2LPT to high z (z≈19 in their tests).

## 3. The three ingredients I was missing

1. **Dirichlet (non-periodic) tile Poisson solve.** The tile's gravity must be
   solved with the potential **fixed at the boundary to the far-field (LPT)
   value**, not wrapped periodically. Implementation: an **isolated / open**
   Poisson solver — zero-padded FFT (double the grid, the standard "James
   method" / Hockney trick) so the box is non-periodic, with the boundary set by
   the far-field. This removes the spurious periodic images that wrecked my
   tiles.
2. **2LPT far field** (not just Zel'dovich): `x_2LPT = q + D₁Ψ₁ + D₂Ψ₂`. The
   COLA residual is then genuinely small (2LPT captures more of the large-scale
   flow), so few PM steps + a small tile suffice.
3. **Buffer ≈ rms displacement (~20–25 Mpc/h here).** Our own measured rms
   displacement was ~5 Mpc/h/axis (~8–9 Mpc/h magnitude); Leclercq's 25 Mpc/h is
   for a z=0 box with larger displacements — the rule "buffer ≈ max displacement"
   holds and is what our feasibility study predicted.

## 4. The correct algorithm (sketch)

For a target tile + buffer:
1. Compute the **global 2LPT** displacement Ψ₁,Ψ₂ on the full (coarse) box once
   (cheap). This is the far field; it carries the external tide exactly.
2. Initialise the tile particles (target+buffer) at high z with the 2LPT
   trajectory restricted to the tile.
3. Evolve the tile with the COLA residual leapfrog, but solve the near-field
   force with an **isolated (zero-padded) Poisson solver** whose boundary is set
   to the **far-field potential** (Dirichlet) — so the tile gravity = local
   near field, with the correct large-scale boundary, no periodic images.
4. The residual is kicked by `F_near − F_LPT_near` (subtract only the LPT force
   of the tile-representable near field — consistent now because the solve is
   isolated, not periodic).
5. Trim the buffer → inner target.

The decisive change vs S11b: **step 3 (isolated/Dirichlet Poisson)**. My tiles
used periodic FFT Poisson → wrong boundary → worse-than-uncoupled. With the
isolated solver + far-field boundary, the tile near field is correct and the
buffer can be small.

## 5. Verification target (the proof of concept)
- The tile sCOLA inner region matches the full-box reference **at a buffer ≈ rms
  displacement** — and **beats** the uncoupled run at that buffer (the opposite
  of the S11b result).
- Convergence: error decreases with buffer, plateauing near the
  buffer≈displacement scale (Leclercq's 25 Mpc/h compromise).
- Bounded memory: the global 2LPT is cheap/coarse; the tile near-field PM is
  local → peak memory ≈ tile, not box.

## 6. References
- Tassev & Eisenstein 2015, *sCOLA* — arXiv:1502.07751.
- Leclercq, Faure, Lavaux, Wandelt, Jaffe, Heavens, Percival 2020, *Perfectly
  parallel cosmological simulations using spatial COLA* — arXiv:2003.04925,
  A&A 639, A91.
- Tassev, Zaldarriaga & Eisenstein 2013, *COLA* — arXiv:1301.0322.
- (Hockney & Eastwood; James 1977 — isolated/zero-padded FFT Poisson.)
