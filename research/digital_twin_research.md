# Research synthesis: from unified survey database to digital twin

**Date:** 2026-04-10
**Purpose:** Cross-referenced research report connecting the current
`oneuniverse` infrastructure to the three-pillar vision (unified data access,
cross-survey science, digital twin). Based on deep literature searches across
five domains: constrained realisations, multi-tracer statistics,
cross-matching, field-level inference, and zoom-in simulations.

---

## 1. Constrained realisations of the local Universe

### State of the art

The field has converged on a single dominant paradigm: **Bayesian forward
modelling with Hamiltonian Monte Carlo (HMC) sampling of initial conditions**.
The leading implementation is the **BORG** algorithm (Bayesian Origin
Reconstruction from Galaxies; Jasche & Wandelt 2013, arXiv:1306.1821), now
maintained by the Aquila Consortium.

**Manticore** (McAlpine, Jasche, Ata, Lavaux et al. 2025, arXiv:2505.10682)
is the current state of the art. It ingests ~69,000 galaxies from the
**2M++ catalogue** (K-band, z < 0.1), uses a COLA-enhanced forward model
during inference at z_init = 69, then resimulates 50 posterior realisations
with full N-body via SWIFT at 1024^3 resolution in a 1000 Mpc box (~3.9 Mpc
grid spacing). Meaningful constraints extend to ~350 Mpc, with S/N dropping
beyond ~200 Mpc.

The **Velocity Field Olympics** (Stiskalek et al. 2026, arXiv:2502.00121)
benchmarked all major reconstruction methods against direct distance tracers
(TF, FP, SNIa). Result: non-linear BORG reconstructions consistently
outperform all linear/Wiener-filter alternatives.

Other major projects:

| Project | Data | Method | Volume | Resolution | Reference |
|---------|------|--------|--------|------------|-----------|
| **BORG/Manticore** | 2M++ (69K galaxies) | HMC + COLA forward model | 1000 Mpc box | 3.9 Mpc | arXiv:2505.10682 |
| **CSiBORG/CSiBORG2** | BORG posterior | 101 RAMSES N-body runs | 677 Mpc/h | 2.6 Mpc/h | arXiv:2203.14724, 2304.09193 |
| **ELUCID** | SDSS DR7 groups | HMC + PM, then N-body | 500 Mpc/h | 0.16 Mpc/h (resim) | arXiv:1407.3451, 1608.01763 |
| **Hamlet-PM** | CosmicFlows-4 PV | Forward model on PV | 500 Mpc/h | ~4 Mpc/h | arXiv:2602.03699 |
| **SIBELIUS-DARK** | BORG ICs | SWIFT DM-only | 200 Mpc | multi-res | arXiv:2202.04099 |
| **SLOW** | CF2 PV | Magneticum hydro | 500 Mpc/h | ~1 Mpc/h | arXiv:2302.10960 |

### What data types matter most?

1. **Galaxy redshift surveys** (2M++, SDSS) — the backbone. Galaxy positions
   constrain the density field on scales > 4 Mpc. Minimum viable input.

2. **Peculiar velocities** (CosmicFlows-4: ~56K galaxies from TF, FP, SNIa,
   SBF, TRGB) — add critical dynamical information. Break the density-velocity
   degeneracy. Hamlet-PM shows PV-only reconstructions are competitive.

3. **CMB lensing** — projected mass map, breaks galaxy bias degeneracy.
   Multi-probe FLI combining CMB lensing + galaxy density is an active
   frontier (arXiv:2304.01387) but not yet in production constrained
   realisations.

4. **Lyman-alpha forest** — probes high-z (z = 2-5) density at small scales.
   Complementary, not directly useful for local (z < 0.1) reconstructions.
   cosmosTNG (arXiv:2409.19047) demonstrated constrained Lya simulations for
   specific sky patches.

### Implications for oneuniverse

The BORG/Manticore pipeline requires: galaxy positions + redshifts with a
well-characterised **selection function** (angular completeness + radial n(z)),
plus **survey geometry** (mask). These are the two critical missing pieces in
the current package (see gap analysis below).

---

## 2. Multi-tracer power spectrum estimation

### Theoretical foundations

**Seljak (2009, arXiv:0807.1770)** showed that cosmic variance cancels in the
ratio of two differently-biased tracers. Constraints on scale-dependent
quantities (f_NL, f*sigma_8) improve without bound as tracer density increases.

**Abramo & Leonard (2013, arXiv:1302.5444)** provided the N-tracer Fisher
matrix, showing that splitting a sample into sub-populations with different
biases always improves constraints on bias-sensitive parameters.

**Generalised FKP** (Abramo et al. 2016, arXiv:1505.04106): the multi-tracer
optimal quadratic estimator simultaneously fits P_mm(k) and per-tracer biases.
Each tracer needs its own n_alpha(r) and b_alpha.

### Current implementations

- **pypower** (cosmodesi): FFT-based FKP with multiple tracers, window-function
  convolution. Baseline for DESI.
- **nbodykit** (Hand et al. 2017, arXiv:1704.02357): `ConvolvedFFTPower` with
  multi-tracer support.
- **SACC** (LSST-DESC): format for summary statistics with per-tracer n(z),
  window functions, joint covariance.

### Recent results

- DESI combined-tracer (2025, arXiv:2508.05467): merging LRGs + ELGs at
  0.8 < z < 1.1 delivers 11% improvement on isotropic BAO (0.86% precision).
- DESI LRG × CMB lensing (arXiv:2407.04607): S8 = 0.765 ± 0.023 using
  Planck PR4 + ACT DR6.
- Euclid multi-tracer (arXiv:2404.12157): ratio estimator from spectro/photo
  samples over shared footprint.
- Euclid 6×2pt (arXiv:2409.18882): joint galaxy clustering + weak lensing +
  all cross-correlations.

### What multi-tracer estimators need from oneuniverse

The current package provides per-object weights and cross-matching. What is
**missing** for full multi-tracer science:

1. **Per-tracer n(z) / selection functions** — every estimator requires the
   smooth radial selection. This is the single most critical gap.
2. **Tracer classification labels** — LRG, ELG, QSO, BGS within a single
   survey. Beyond `survey_id`.
3. **Linear bias estimates** — per-class lookup table indexed by z-bin.
4. **Random catalogs** — no infrastructure exists. pypower/nbodykit require
   randoms matching each tracer's mask + n(z).
5. **Angular masks** — HEALPix or polygon survey footprints with per-pixel
   completeness.
6. **Survey overlap flags** — which surveys cover a given point, for
   effective volume computation.

---

## 3. Cross-matching: current approach vs. literature

### What oneuniverse does right

The current implementation — `SkyCoord.search_around_sky` + scipy
`connected_components` — is well-aligned with the literature:

- **Ball-tree matching** is the standard for catalogs up to ~10^6 objects
  (Astropy documentation, Gaia pipeline uses similar approaches).
- **Connected components for transitivity** matches the graph-theoretic
  approach recommended by Budavari & Szalay (2008, arXiv:0707.1611).
- **Cross-survey-only linking** (same-survey pairs excluded) is a sound design
  choice that avoids merging close pairs within a single survey.

### Literature-identified improvements

From the cross-matching agent's findings (with references):

1. **Survey-pair-specific tolerances**: optimal match radius should be
   3-5× sqrt(sigma_1^2 + sigma_2^2). For optical-optical: 0.5-1". For
   optical-WISE: 2-3". For X-ray-optical: 6-15" + NWAY priors
   (Salvato et al. 2018, arXiv:1705.10711).

2. **Redshift-dependent dz_tol**: DESI QSO redshifts carry z-dependent
   systematic offsets of ~200 km/s before template correction (Bault et al.
   2025, arXiv:2402.18009). QSOs may need dz_tol ~ 5e-3 vs galaxies at 1e-3.

3. **Match quality metadata**: store pairwise angular separation and delta-z
   in the ONEUID index for downstream filtering.

4. **Transitivity safeguards**: flag ONEUID groups where not all pairwise
   separations satisfy the tolerance (potential blends or close pairs).
   Maximum group diameter check.

5. **Probabilistic matching (long-term)**: for surveys with very different
   positional uncertainties (eROSITA, WISE), NWAY (arXiv:1705.10711) provides
   Bayesian multi-catalog matching with magnitude priors.

### Photometric cross-calibration

When combining photometry across surveys:
- SDSS and DESI Legacy Surveys have ~0.02-0.04 mag zero-point differences.
- Store E(B-V) separately (SFD98 with Schlafly & Finkbeiner 2011 recalibration,
  arXiv:1012.4804). Apply per-band R_lambda for each filter curve.
- Never average magnitudes across different filter systems.

---

## 4. Field-level inference and the flip connection

### flip as a special case of FLI

The flip package (Ravoux et al. 2025, arXiv:2501.16852) performs **Gaussian
field-level inference**: a multivariate Gaussian likelihood over the velocity
and/or density field with a covariance matrix C(fs8, bs8, sigv) assembled from
Hankel-transformed power spectra. This is formally equivalent to FLI with a
linear forward model and Gaussian field assumption.

The analytic covariance terms xi_ij(r) are the two-point functions of the
forward-modeled field, and the assembly step C = sum_i alpha_i(params) * xi_i
is the linearised parameter dependence. The Gaussian assumption is well-
justified for velocities (the velocity field remains closer to Gaussian than
the density field; it is a spatial integral of delta).

### Information gain from full FLI

Stadler et al. (arXiv:2509.09673) found that FLI error bars on {A_s, omega_cdm,
H_0} are ~20% tighter than joint P(k)+bispectrum at the perturbative level.
The gain increases in the non-linear regime. For f*sigma_8 from PV, Nguyen et
al. (arXiv:2407.14289) estimate factors of 2-3 improvement for DESI-sized
catalogs when moving from Gaussian to full forward-model FLI.

### What full FLI needs from oneuniverse

The transition from flip's Gaussian FLI to full forward-model FLI requires
oneuniverse to expose not just data + errors, but the **complete generative
observation model**:

1. **Selection function n(z, ra, dec)** — probability of observing a galaxy at
   any position given the survey design. The current per-object weights are
   ingredients, but FLI needs the generative model.

2. **Angular mask** — HEALPix/MOC with per-pixel completeness, so the forward
   model can mask simulated fields identically to real data.

3. **Fiber/targeting metadata** — plate/tile assignments, fiber collision
   groups, for forward-modelling incompleteness.

4. **Photometric columns for bias modelling** — luminosity proxies determine
   the luminosity-dependent galaxy bias.

5. **Comoving distance grids** — pre-computed r_com(z) under fiducial
   cosmology, since FLI forward models operate in comoving space.

6. **Cosmological parameter provenance** — fiducial cosmology must be stored
   and swappable.

7. **Random catalog support** — BORG-type FLI requires randoms encoding the
   selection function.

### Existing FLI frameworks

| Framework | Forward model | Inference | Key reference |
|-----------|--------------|-----------|---------------|
| **BORG** | LPT → PM → COLA | HMC | arXiv:1306.1821, 1806.11117 |
| **PORQUERES** | LPT → Lya flux | HMC | arXiv:2005.12928 |
| **FlowPM** | TensorFlow PM | HMC/MAP | arXiv:2010.11847 |
| **pmwd** | JAX PM (adjoint) | HMC/MAP | arXiv:2211.09958 |
| **SELFI** | Implicit likelihood | Score expansion | Leclercq+2019 |
| **V-net emulator** | Neural N-body | HMC (in BORG) | arXiv:2312.09271 |

Computational costs: BORG on SDSS (256^3) ~ 10^5 CPU-hours. FlowPM/pmwd on
GPU: single forward+backward pass ~ 1-5 seconds on A100. MCLMC (arXiv:
2504.20130) outperforms HMC by 40-80× in effective samples per gradient.

---

## 5. Zoom-in and multi-scale re-simulation

### The pipeline: observations → ICs → zoom → hydro

```
oneuniverse database (surveys)
  → BORG/Manticore (constrained ICs on ~4 Mpc grid)
  → MUSIC/monofonIC (multi-scale ICs with zoom region)
  → Nyx / AREPO / RAMSES (high-res hydro in zoom volume)
  → synthetic observables (Lya spectra, galaxy properties)
```

### IC generators

**MUSIC** (Hahn & Abel 2011, arXiv:1103.6031): nested-grid convolution with
multigrid Poisson solver. Already has output plugins for Nyx, RAMSES, AREPO,
Gadget, Enzo. **monofonIC/MUSIC2** (arXiv:2008.09124): extends to 3LPT with
10^-4 rms displacement accuracy.

**cosmICweb** (Stucker et al. 2024, arXiv:2406.02693): cloud service for
selecting zoom regions and generating MUSIC ICs.

**GenetIC** (Stopyra, Pontzen et al. 2021, arXiv:2006.01841): genetic
modifications — controlled alterations to a halo's merger history while
preserving the statistical ensemble.

### Major constrained zoom projects

| Project | Code | Data | Volume/res | Reference |
|---------|------|------|------------|-----------|
| **HESTIA** | AREPO (Auriga) | CLUES ICs | 3-5 Mpc zoom, ~kpc | arXiv:2008.04926 |
| **SIBELIUS** | SWIFT/GALFORM | BORG ICs | 200 Mpc, multi-res | arXiv:2202.04099 |
| **SLOW** | OpenGadget3 | CF2 PV | 500 Mpc/h, hydro | arXiv:2302.10960 |
| **CSiBORG2** | RAMSES | BORG + zoom | Local Group | A&A 2024 |
| **cosmosTNG** | IllustrisTNG | TARDIS + CLAMATO Lya | COSMOS field, z~2 | arXiv:2409.19047 |

### Nyx for constrained zoom-ins

Nyx (Almgren et al. 2013, arXiv:1301.4498) is AMReX-based, GPU-native, and
optimised for the Lya forest. MUSIC already has a Nyx output plugin.
However, Nyx does **not** natively support multi-resolution zoom-in ICs.

Required modifications for the oneuniverse pipeline:
1. Preprocessing to convert BORG-inferred fields → MUSIC inputs → Nyx ICs
   (the MUSIC plugin exists, but the BORG → MUSIC bridge needs development).
2. AMR refinement criteria configured to refine the zoom region while keeping
   the buffer at coarse resolution.
3. Nyx lacks sub-grid galaxy formation physics (no star formation, no AGN
   feedback). Well-suited for Lya forest; would need coupling to a sub-grid
   model for galaxy-formation zooms.

### Resolution requirements

| Science case | Spatial resolution | Physics | Box context |
|-------------|-------------------|---------|-------------|
| LSS / peculiar velocities | 1-2 Mpc/h | DM-only | 200-500 Mpc/h |
| Lya forest (P1D) | 20-50 kpc/h | Hydro | 40-100 Mpc/h |
| Lya forest (individual) | 10-20 kpc/h | Hydro | 40 Mpc/h |
| Galaxy formation | 0.1-1 kpc | Full sub-grid | 50-100 Mpc/h |

### Incremental re-simulation

No production framework currently exists. Closest analogues:
- BORG posterior sampling: each sample changes only affected modes.
- GenetIC: modify one halo without re-running the full box.
- CSiBORG2 multi-resolution: refine the posterior in one sub-region.

Challenge: long-range mode coupling means changing constraints locally
propagates globally at low-k. Hydrodynamic state is path-dependent (cannot
restart mid-run from modified ICs without losing thermal/chemical history).

### What oneuniverse must provide for zoom-ins

1. **Reconstructed 3D fields** on a grid (BORG posterior or equivalent).
2. **White-noise realisation** from the parent run (MUSIC needs this).
3. **Target region specification** (sky coords + z-range + comoving extent).
4. **Tidal-field boundary data** at the zoom boundary.
5. **Cosmological parameters** consistent between reconstruction and IC
   generation.

---

## 6. Gap analysis: what oneuniverse has vs. what it needs

### Already implemented (confirmed by cross-check)

| Feature | Status | Validated by |
|---------|--------|-------------|
| Standardised schema (core, spectro, photo, PV, QSO, SNIa) | Done | All agents |
| Three data geometries (POINT, SIGHTLINE, HEALPIX) | Done | Zoom-in, FLI |
| ONEUID cross-match (ball-tree + connected components) | Done | Cross-match agent |
| Cross-survey-only linking | Done | Cross-match agent (correct design choice) |
| Per-survey weights (FKP, ivar, quality mask) | Done | Multi-tracer agent |
| Combination strategies (best_only, ivar_average, hyperparameter) | Done | Multi-tracer agent |
| Tiered query API (selectors × hydration) | Done | FLI agent (right interface for FLI) |
| Column pushdown (Parquet) | Done | All agents (I/O efficiency) |

### Critical gaps (identified by ≥ 3 agents)

| Gap | Needed by | Priority |
|-----|-----------|----------|
| **Per-tracer n(z) / selection functions** | Multi-tracer, FLI, constrained realisations | P0 |
| **Angular masks / survey geometry** | Multi-tracer, FLI, constrained realisations, zoom-in | P0 |
| **Random catalog generation** | Multi-tracer (pypower), FLI (BORG) | P1 |

### Important gaps (identified by 1-2 agents)

| Gap | Needed by | Priority |
|-----|-----------|----------|
| Tracer classification labels (LRG/ELG/QSO/BGS) | Multi-tracer | P1 |
| Survey-pair-specific cross-match tolerances | Cross-match | P1 |
| Match quality metadata (sep, dz in index) | Cross-match | P1 |
| Transitivity safeguards (max group diameter) | Cross-match | P2 |
| Linear bias estimates per tracer class | Multi-tracer | P2 |
| Fiber/targeting metadata | FLI | P2 |
| Comoving distance grid under fiducial cosmology | FLI, zoom-in | P2 |
| Reconstructed 3D field storage | Digital twin | P3 (Phase 3) |
| White-noise realisation storage | Zoom-in | P3 (Phase 3) |

---

## 7. References

### Constrained realisations
- Jasche & Wandelt 2013 — BORG: arXiv:1306.1821
- Jasche & Lavaux 2019 — BORG on SDSS: arXiv:1806.11117, 1909.06396
- McAlpine et al. 2025 — Manticore: arXiv:2505.10682
- Stiskalek et al. 2026 — Velocity Field Olympics: arXiv:2502.00121
- Stopyra et al. 2022 — CSiBORG: arXiv:2203.14724
- Stopyra et al. 2024 — CSiBORG2: arXiv:2304.09193
- Wang et al. 2014, 2016 — ELUCID: arXiv:1407.3451, 1608.01763
- Valade et al. 2026 — Hamlet-PM: arXiv:2602.03699
- Hoffman et al. 2024 — CF4 Wiener filter: arXiv:2311.01340
- Doeser et al. 2024 — V-net emulator: arXiv:2312.09271
- Wempe et al. 2024 — BORG Local Group: arXiv:2406.02228

### Multi-tracer statistics
- Seljak 2009 — cosmic variance cancellation: arXiv:0807.1770
- Abramo & Leonard 2013 — N-tracer Fisher: arXiv:1302.5444
- Abramo et al. 2016 — multi-tracer FKP: arXiv:1505.04106
- Hand et al. 2017 — nbodykit: arXiv:1704.02357
- DESI 2025 — combined tracer BAO: arXiv:2508.05467
- DESI 2024 — LRG × CMB lensing: arXiv:2407.04607
- Euclid 2024 — multi-tracer XLVII: arXiv:2404.12157
- Euclid 2024 — 6×2pt: arXiv:2409.18882
- Hadzhiyska et al. 2024 — HOD systematics: arXiv:2404.03008

### Cross-matching and calibration
- Budavari & Szalay 2008 — Bayesian matching: arXiv:0707.1611
- Salvato et al. 2018 — NWAY: arXiv:1705.10711
- Bault et al. 2025 — DESI QSO z-systematic: arXiv:2402.18009
- Schlafly & Finkbeiner 2011 — extinction recalibration: arXiv:1012.4804
- Marrese et al. 2019 — Gaia cross-match (A&A 621, A144)

### Field-level inference
- Ravoux et al. 2025 — flip: arXiv:2501.16852
- Porqueres et al. 2020 — PORQUERES Lya: arXiv:2005.12928
- Modi et al. 2021 — FlowPM: arXiv:2010.11847
- Li et al. 2022 — pmwd: arXiv:2211.09958
- Stadler et al. 2025 — FLI vs P(k)+bispec: arXiv:2509.09673
- FLI benchmarks 2025: arXiv:2504.20130

### Zoom-in simulations
- Libeskind et al. 2020 — HESTIA: arXiv:2008.04926
- McAlpine et al. 2022 — SIBELIUS-DARK: arXiv:2202.04099
- Dolag et al. 2023 — SLOW: arXiv:2302.10960
- Buehlmann et al. 2025 — cosmosTNG: arXiv:2409.19047
- Almgren et al. 2013 — Nyx: arXiv:1301.4498
- Hahn & Abel 2011 — MUSIC: arXiv:1103.6031
- Hahn et al. 2020 — monofonIC: arXiv:2008.09124
- Stucker et al. 2024 — cosmICweb: arXiv:2406.02693
- Stopyra, Pontzen et al. 2021 — GenetIC: arXiv:2006.01841
