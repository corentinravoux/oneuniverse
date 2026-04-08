# Weighting Schemes for Cross-Survey Cosmological Measurements

Research notes for implementing `oneuniverse` weight classes. Focus: objective
weighting schemes used to combine redshift, peculiar-velocity, and SN Ia
measurements across eBOSS, DESI, SDSS MGS, 6dFGS, CosmicFlows-4, DESI-PV.

Author: research compiled for C. Ravoux, 2026-04-07.

---

## 1. Inverse-variance weighting (canonical)

The maximum-likelihood combination of independent Gaussian measurements
`x_i ± σ_i` with identical expectation is

    x_hat = (Σ_i w_i x_i) / (Σ_i w_i),   w_i = 1/σ_i^2,
    σ_hat^2 = 1 / (Σ_i w_i).

**When to use:** combining independent measurements of the same quantity
(redshift, distance modulus μ, log-distance ratio η, line-of-sight velocity
v_LOS) when errors are well characterised, approximately Gaussian, and
uncorrelated. Baseline default for cross-survey combination in `oneuniverse`.

**Gotchas:**
- Assumes uncorrelated errors. Surveys sharing calibration (e.g. SDSS/BOSS/eBOSS
  pipelines) may have correlated systematics; underestimates variance.
- Breaks if one σ is catastrophically wrong (e.g. ZWARNING redshift).
  Pre-filter on quality flags before applying.
- For η vs v_LOS errors: never mix units; propagate to a common estimator first
  (see §3).

### 1a. FKP weights — Feldman, Kaiser & Peacock (1994)

For power-spectrum / 2-point function estimation of a density field with
spatially varying number density n̄(z):

    w_FKP(z) = 1 / (1 + n̄(z) · P_0),

with `P_0 ≈ 10^4 (Mpc/h)^3` for BOSS/eBOSS LRG clustering and `P_0 ≈ 4·10^3` for
ELG/QSO-like tracers. Minimises the variance of the estimated P(k) at the
fiducial scale k such that `P(k) ≈ P_0`.

**Reference:** Feldman, Kaiser & Peacock 1994, ApJ 426, 23 (arXiv:astro-ph/9304022).

**Gotcha:** FKP is a *clustering* weight, not a per-object measurement weight.
It should be stored separately from inverse-variance weights and applied only
in 2-point estimators.

### 1b. PVW — pairwise / peculiar-velocity-field weights

For peculiar-velocity 2-point statistics (ψ1/ψ2, v-v correlation, momentum
P(k)), the analogue of FKP is

    w_PV(z) = 1 / (σ_v^2 + σ_*^2 + n̄_v(z) · P_vv),

where σ_* ≈ 150–300 km/s is a non-linear velocity dispersion and P_vv is
evaluated at the scale of interest.

**References:** Howlett 2019, MNRAS 487, 5209; Adams & Blake 2020, MNRAS 494,
3275; Carreres et al. 2023, MNRAS 518, 2253.

### 1c. Optimal quadratic estimator weights

Tegmark 1997 (PRD 55, 5895) / Tegmark, Taylor & Heavens 1997 (ApJ 480, 22):
weights are the rows of `F^{-1} C,_θ C^{-1}` where C is the full covariance and
F the Fisher matrix. Reduces to FKP in the diagonal, shot-noise-dominated
limit. Use in full-field likelihood; not practical as a per-object weight for a
catalogue merger but documented here for completeness.

---

## 2. Completeness / selection-function corrections (eBOSS/DESI convention)

eBOSS (Ross et al. 2017, 2020) and DESI (Mohammad et al. 2020; Ross et al.
2025) attach four multiplicative corrections to every target:

    w_tot = w_systot · w_cp · w_noz · w_FKP,       (clustering)
    w_tot = w_systot · w_cp · w_noz                 (non-clustering / N(z))

- **w_systot** — imaging systematics (stellar density, seeing, Galactic
  extinction, depth). Fitted by linear/neural regression of target density vs
  template maps. Ross+2020 (DR16 LRG/QSO) use linear multivariate regression.
- **w_cp** — close-pair / fiber-collision correction. Object that lost a fiber
  to a collided neighbour is up-weighted onto its nearest resolved neighbour.
  Standard: w_cp = 1 + N_missed.
- **w_noz** — redshift-failure correction. Successful redshifts are up-weighted
  by the inverse of the local (plate, fiber, S/N) success rate. DESI uses a
  per-petal model (Mohammad+2020).
- **w_FKP** — as in §1a.

**References:**
- Ross et al. 2017, MNRAS 464, 1168 (arXiv:1607.03145) — eBOSS QSO selection.
- Ross et al. 2020, MNRAS 498, 2354 (arXiv:2007.09000) — DR16 LRG/QSO clustering catalogues.
- Bautista et al. 2018, ApJ 863, 110 — DR14 LRG.
- Mohammad et al. 2020, MNRAS 498, 128 (arXiv:2007.09005) — fiber collisions (PIP).
- Raichoor et al. 2021 (eBOSS ELG); DESI 2024 BAO+FS papers (Ross et al. 2025, DESI Collaboration 2024).

**Gotcha:** w_cp and w_noz double-count if PIP (pairwise-inverse-probability,
Bianchi & Percival 2017) weights are used; pick one scheme.

---

## 3. Peculiar-velocity-specific weighting

PV surveys report either distance modulus μ, log-distance ratio η = log10(d_z/d_H),
or velocity v_LOS, each with its own error model.

**Estimator choice (Watkins & Feldman 2015, MNRAS 450, 1868):**

    v_est = c · ln(10) · η · (factor depending on z),

with σ_v ≈ c · ln(10) · σ_η at low z. The log-distance estimator is unbiased
under Gaussian distance errors while v = cz − H0 d is biased (Malmquist).

**Malmquist bias:**
- *Homogeneous* (inhomogeneous sampling along LOS) and *inhomogeneous* (density
  gradients) Malmquist corrections are standard in Tully-Fisher / Fundamental
  Plane catalogues. Strauss & Willick 1995 (Phys. Rep. 261, 271) is the
  classical reference; see also Hoffman et al. 2021 for CF4.
- Use **forward likelihood** (condition on observed magnitude, model distance)
  rather than inverse distance estimators where possible — this sidesteps bias
  and the weighting becomes simply `1/σ_μ^2` in the likelihood.

**Catalogue weights in practice:**
- CosmicFlows-4 (Tully et al. 2023, ApJ 944, 94) provides per-object σ_μ; users
  apply inverse-variance weighting in η-space plus a `σ_*` ≈ 250 km/s
  non-linear velocity-dispersion floor.
- 6dFGSv (Springob et al. 2014) and SDSS FP (Howlett et al. 2022) same
  treatment.
- DESI-PV (Saulder et al. 2023) uses TF+FP with per-object σ_η; inverse-variance
  weights in η plus grouping for clusters.

**Grouping:** for objects in the same host group (galaxy cluster), virial
motions must be removed. Standard: replace individual velocities with the
group-mean velocity, σ_v,group = σ_v / √N_group (Tully 2015 group catalogue).

**References:**
- Watkins & Feldman 2015, MNRAS 450, 1868 (arXiv:1411.6665).
- Strauss & Willick 1995, Phys. Rep. 261, 271.
- Tully et al. 2023 (CF4), ApJ 944, 94 (arXiv:2209.11238).
- Howlett et al. 2022 (SDSS-FP), MNRAS 515, 953.
- Said et al. 2020 (6dF+SDSS FP), MNRAS 497, 1275.

---

## 4. Redshift-quality weighting

Discrete quality flags are usually converted to a binary mask, not a
continuous weight:

- **SDSS/BOSS/eBOSS ZWARNING:** keep `ZWARNING == 0` (or `ZWARNING_NOQSO == 0`
  for LRG). Bolton et al. 2012, AJ 144, 144.
- **DESI Redrock ZWARN + DELTACHI2:** `ZWARN == 0 & DELTACHI2 > 25` for BGS/LRG,
  `> 40` for QSO (DESI Collaboration 2024 EDR VAC paper; Lan et al. 2023).
- **6dFGS Q-flag:** keep Q ≥ 3 (Jones et al. 2009).
- **CLASS confidence (VIPERS/VVDS convention):** confidence ≥ 2 or 3.

Continuous S/N weighting is rarely used for spectroscopic redshifts because
σ_z is already dominated by the template fit uncertainty, not S/N alone. When
used, the weight is simply `1/σ_z^2` from the pipeline.

**Gotcha:** redshift failures are *not* random — they correlate with colour,
magnitude, fiber position. Always couple quality cuts with a w_noz correction.

---

## 5. Cross-survey combination (same object, multiple surveys)

When a single physical object (matched by sky position ± redshift) is observed
by N surveys, practice splits along two lines:

### 5a. Best-only ("priority ladder")

Pick the highest-quality measurement and discard the rest. Used in:
- SDSS-IV value-added catalogues (SpecObj "bestObjID" convention).
- CF4 merged distance catalogue (Tully et al. 2023): when a galaxy has both TF
  and FP distances, the one with smaller σ_μ is kept; ties broken by survey
  priority (CF3 > 6dF-FP > SDSS-FP).
- DESI EDR QSO VAC: prefers DESI over SDSS/eBOSS when both exist.

**Pro:** avoids correlated-systematics double counting. **Con:** throws away
signal.

### 5b. Inverse-variance mean

    x̂ = (Σ x_i/σ_i²) / (Σ 1/σ_i²),   σ̂² = 1 / (Σ 1/σ_i²).

Used in:
- Pantheon+ SN compilation (Scolnic et al. 2022, ApJ 938, 113) — when a SN has
  multiple photometric measurements across samples, they are combined with
  inverse-variance weighting *after* cross-calibration of zero-points.
- CF4 velocity averaging for galaxies with multiple independent TF
  measurements (different bands, different observers).

**Pro:** uses all information. **Con:** requires trust in the error bars and
independence assumption.

### 5c. Hybrid — full covariance combination

When correlations are non-negligible (shared zero-points, shared photometry):

    x̂ = (1ᵀ C⁻¹)x / (1ᵀ C⁻¹ 1),   σ̂² = 1 / (1ᵀ C⁻¹ 1),

with C the full N×N covariance including off-diagonal calibration terms.
Used by Pantheon+ for SN subsample combination and by Planck+ACT for
cosmological parameter averaging.

**Standard reference for catalogue-level practice:** there is no single
canonical paper. The de facto standard is Pantheon+ (Scolnic+2022, Brout+2022)
for SNe; for PV, CF4 (Tully+2023) and Howlett et al. 2022; for redshifts,
simply ZWARNING-filter + IVW per DESI/eBOSS clustering-catalogue papers.

### Recommendation for `oneuniverse`

1. Default: **best-only** priority ladder per science case (clustering vs PV
   vs SN), configurable.
2. Optional: **inverse-variance mean** with explicit independence assumption
   flag, warning when shared-pipeline surveys are combined.
3. Advanced: **covariance-aware combination** once inter-survey covariances
   are characterised (future work).

Store every weight component separately (`w_ivar`, `w_fkp`, `w_cp`, `w_noz`,
`w_systot`, `w_qual`) so downstream code can reconstruct any combination.

---

## Paper citation list (full)

- Feldman, Kaiser & Peacock 1994, ApJ 426, 23 — FKP weights.
- Tegmark 1997, PRD 55, 5895 — optimal quadratic estimators.
- Tegmark, Taylor & Heavens 1997, ApJ 480, 22 — Karhunen-Loève.
- Strauss & Willick 1995, Phys. Rep. 261, 271 — PV estimators and Malmquist.
- Watkins & Feldman 2015, MNRAS 450, 1868 — η vs v_LOS estimators.
- Ross et al. 2017, MNRAS 464, 1168 — eBOSS QSO selection.
- Ross et al. 2020, MNRAS 498, 2354 — DR16 clustering catalogues.
- Bautista et al. 2018, ApJ 863, 110 — DR14 LRG.
- Mohammad et al. 2020, MNRAS 498, 128 — fiber collisions, PIP.
- Bianchi & Percival 2017, MNRAS 472, 1106 — PIP weights.
- Raichoor et al. 2021, MNRAS 500, 3254 — eBOSS ELG.
- DESI Collaboration 2024 (BAO + Full-Shape series).
- Bolton et al. 2012, AJ 144, 144 — BOSS spectroscopic pipeline + ZWARNING.
- Jones et al. 2009, MNRAS 399, 683 — 6dFGS redshift quality.
- Tully et al. 2023, ApJ 944, 94 — CosmicFlows-4.
- Howlett et al. 2022, MNRAS 515, 953 — SDSS Fundamental Plane PV.
- Said et al. 2020, MNRAS 497, 1275 — 6dFGSv + SDSS FP combined.
- Saulder et al. 2023 — DESI-PV TF/FP.
- Howlett 2019, MNRAS 487, 5209 — PV P(k) weighting.
- Adams & Blake 2020, MNRAS 494, 3275 — density+velocity covariance.
- Carreres et al. 2023, MNRAS 518, 2253 — velocity covariance model.
- Scolnic et al. 2022, ApJ 938, 113 — Pantheon+ compilation.
- Brout et al. 2022, ApJ 938, 110 — Pantheon+ cosmology + systematics.
