# Cosmography Reference for the `oneuniverse` Package

**Date:** 2026-04-01
**Purpose:** Reference document for designing `oneuniverse` — a Python package that stores and unifies galaxy survey catalogs (spectroscopic, photometric, peculiar velocity, SNe Ia, radio/HI) with consistent cosmographic quantities and weighting schemes.

---

## Table of Contents

1. [Cosmographic Quantities and Observables](#1-cosmographic-quantities-and-observables)
2. [Survey Types for Low-Redshift Cosmology](#2-survey-types-for-low-redshift-cosmology)
3. [Weighting Schemes](#3-weighting-schemes)
4. [Cross-Survey Consistency Challenges](#4-cross-survey-consistency-challenges)
5. [Standard Data Formats and Schemas](#5-standard-data-formats-and-schemas)
6. [Recommended Schema for `oneuniverse`](#6-recommended-schema-for-oneuniverse)
7. [Appendix: Key Formulae](#appendix-key-formulae)
8. [References](#references)

---

## 1. Cosmographic Quantities and Observables

### 1.1 Redshift

Redshift is the primary distance proxy in extragalactic cosmology. The observed redshift `z_obs` combines the cosmological (Hubble flow) redshift `z_cos` and the line-of-sight peculiar velocity `v_pec`:

```
1 + z_obs = (1 + z_cos)(1 + v_pec / c)
```

In the low-z limit (`z << 1`):

```
z_obs ≈ z_cos + v_pec / c
```

#### 1.1.1 Spectroscopic Redshift (z_spec)

- **Definition:** Measured from the Doppler shift of spectral features (H-alpha, [OII], Ca H&K, Mg, etc.) relative to rest-frame wavelengths.
- **Precision:** sigma_z ~ 0.0001–0.001, corresponding to 30–300 km/s.
- **Storage columns:** `Z_SPEC`, `Z_SPEC_ERR`, `Z_SPEC_QUAL` (quality flag: e.g., nQ >= 3 in GAMA; zconf >= 3 in 6dFGS; ZWARNING = 0 in SDSS/DESI).

#### 1.1.2 Photometric Redshift (z_phot)

- **Definition:** Estimated from multi-band photometry by template fitting or machine learning.
- **Precision:** sigma_z / (1+z) ~ 0.02–0.05 for template fitting; ~0.01–0.03 for well-trained ML.
- **Biases:** Catastrophic outliers (eta_out ~ 1–5%), color-redshift degeneracies (especially around z ~ 0.3).
- **Storage columns:** `Z_PHOT`, `Z_PHOT_ERR`, `Z_PHOT_L68`, `Z_PHOT_U68`, `ODDS` (quality scalar 0–1), `Z_PHOT_CODE` (method: BPZ, EAZY, LePhare, DNF, FlexZBoost, etc.).
- **PDFs:** Full P(z) PDFs increasingly stored as compressed arrays; essential for correct propagation into clustering statistics.

#### 1.1.3 Peculiar Velocity and Redshift Correction

The **peculiar velocity** `v_pec` of a galaxy relative to the Hubble flow:

```
v_pec = c * (z_obs - z_cos) / (1 + z_cos)    [exact to first order in v/c]
```

The cosmological redshift is obtained from the heliocentric redshift after subtracting the Local Group motion (CMB dipole correction):

```
cz_CMB = cz_helio + v_LG * cos(theta)
```

where `v_LG = 620 km/s` toward `(l, b) = (271.4°, 29.3°)` (WMAP/Planck).

**Key storage columns:**

| Column | Unit | Description |
|---|---|---|
| `Z_HELIO` | — | Heliocentric observed redshift |
| `Z_CMB` | — | CMB-frame redshift (dipole corrected) |
| `CZ_HELIO` | km/s | Heliocentric recession velocity |
| `CZ_CMB` | km/s | CMB-frame recession velocity |
| `V_PEC` | km/s | Inferred peculiar velocity (PV surveys) |
| `V_PEC_ERR` | km/s | Peculiar velocity uncertainty |

**Peculiar velocity estimators:**
- **Tully-Fisher (TF):** Baryonic/stellar TF relation; spiral galaxies. Yields distance modulus mu.
- **Fundamental Plane (FP):** Elliptical/early-type galaxies. Combines effective radius, velocity dispersion, surface brightness.
- **Type Ia Supernovae:** Standardizable candles giving high-precision mu.
- **Surface Brightness Fluctuations (SBF):** Most precise single-object method at short distances.

All estimators yield a **distance modulus** `mu = 5 log10(d_L / 10 pc)`. The peculiar velocity:

```
v_pec = cz_CMB - H0 * d    [km/s, low-z approximation]
```

with propagated uncertainty:

```
sigma_v = (ln(10)/5) * sigma_mu * cz_CMB    [km/s]
```

This uncertainty grows **linearly with redshift**, which intrinsically limits PV surveys to `z < 0.1` for TF/FP methods.

### 1.2 Angular Position and Coordinate Systems

#### 1.2.1 Equatorial Coordinates (ICRS)

- **RA (Right Ascension):** 0–360°, measured eastward.
- **DEC (Declination):** −90° to +90°.
- **Standard:** ICRS (International Celestial Reference System). Use float64 for sub-arcsecond precision.
- **Storage columns:** `RA` (float64, deg), `DEC` (float64, deg).

#### 1.2.2 Galactic Coordinates

- `l` (Galactic longitude): 0–360°, from the Galactic center.
- `b` (Galactic latitude): −90° to +90°, from the Galactic plane.
- **Use in cosmology:** Galactic extinction (E(B-V) from Schlafly & Finkbeiner 2011), Zone of Avoidance cuts (|b| < 10°–20°).
- **Storage columns:** `L_GAL`, `B_GAL` (deg), `EBV` (mag), `A_X` (extinction in band X).

#### 1.2.3 Supergalactic Coordinates

- `SGL` (Supergalactic longitude): 0–360°.
- `SGB` (Supergalactic latitude): −90° to +90°.
- **Reference:** Defined by the Supergalactic Plane (Virgo, Hydra-Centaurus, Perseus-Pisces superclusters).
- **Use:** Natural frame for the local cosmic web, CosmicFlows reconstructions.
- **Storage columns:** `SGL`, `SGB` (deg).

### 1.3 Distance Measures (Cosmology-Dependent)

`oneuniverse` should:
1. Store **cosmological parameters** as catalog-level metadata.
2. Store **comoving distance** as the canonical quantity.
3. Cache luminosity and angular diameter distances optionally.

#### 1.3.1 Comoving Distance

```
d_C(z) = (c / H0) * integral_0^z  dz' / E(z')
E(z) = sqrt(Omega_m (1+z)^3 + Omega_Lambda)    [flat LCDM]
```

Units: Mpc (with explicit H0) or Mpc/h. Column: `D_C`.

#### 1.3.2 Luminosity Distance

```
d_L(z) = (1 + z) * d_C(z)    [flat universe]
```

Used in the distance modulus: `mu = 5 log10(d_L / 10 pc)`. Column: `D_L`.

#### 1.3.3 Angular Diameter Distance

```
d_A(z) = d_C(z) / (1 + z)    [flat universe]
```

Important for BAO measurements. Column: `D_A`.

#### 1.3.4 Distance Modulus

```
mu(z) = 5 log10(d_L(z) / 10 pc) = 5 log10(d_L / Mpc) + 25
```

Storage columns: `MU`, `MU_ERR`.

### 1.4 Redshift-Space Distortions

The redshift-space position `s` is displaced from the real-space position `r` by:

```
s = r + (v_pec · r_hat) / (aH) * r_hat    [comoving coordinates]
```

**Two regimes:**
- **Kaiser effect (large scales):** Coherent infall toward overdensities compresses the radial clustering signal. Linear RSD parameter `beta = f / b` where `f = d ln D / d ln a ≈ Omega_m^0.55` (growth rate) and `b` is galaxy bias.
- **Fingers of God (small scales):** Virialized random motions elongate the clustering signal along the line of sight.

### 1.5 Hubble Flow Corrections

Standard corrections (decreasing importance):
1. **Solar motion toward the CMB dipole:** ~370 km/s — dominant term.
2. **Local Group infall toward Virgo / Great Attractor:** ~300 km/s.
3. **Mould et al. (2000) three-attractor model:** adds Shapley Supercluster.

**Recommendation:** Always store `Z_HELIO` (raw observable) and `Z_CMB` (dipole-corrected). Optionally store `Z_COSMO` (full PV model-corrected) when the survey provides it.

### 1.6 Photometric Quantities

| Column | Description |
|---|---|
| `MAG_X` | Apparent AB magnitude in band X (`MAG_G`, `MAG_R`, `MAG_I`, etc.) |
| `MAG_ERR_X` | Uncertainty |
| `K_X` | K-correction in band X |
| `MAG_ABS_X` | Absolute magnitude: `M_X = m_X - DM(z) - K_X(z) - A_X` |
| `EBV` | E(B-V) Galactic extinction |
| `A_X` | Extinction in band X: `A_X = R_X * EBV` |
| `LOG_MSTAR` | log10(stellar mass / M_sun) |
| `LOG_SFR` | log10(SFR / (M_sun/yr)) |

---

## 2. Survey Types for Low-Redshift Cosmology

### 2.1 Spectroscopic Galaxy Redshift Surveys

#### SDSS (Sloan Digital Sky Survey)

- **Main Galaxy Sample (MGS):** r < 17.77, z ~ 0.01–0.25, ~700K galaxies, ~10,000 deg².
- **BOSS LOWZ/CMASS:** z ~ 0.15–0.7, ~1.5M LRGs.
- **eBOSS:** ELGs, LRGs, QSOs at z ~ 0.6–2.2.
- **Key observables:** Z_SPEC (ZWARNING flag), ugriz photometry.
- **Fiber collisions:** ~6% of targets; corrected via nearest-neighbor upweighting (w_cp).
- **Reference:** York et al. (2000); Eisenstein et al. (2011); Dawson et al. (2013).

#### 2dFGRS (Two-degree Field Galaxy Redshift Survey)

- **Depth:** b_J < 19.45, z ~ 0–0.3, ~221K galaxies, ~1,500 deg².
- **Reference:** Colless et al. (2001, 2003).

#### 6dFGS (Six-degree Field Galaxy Survey)

- **Depth:** K < 12.65, median z ~ 0.053, ~136K galaxies, ~17,000 deg² (southern).
- **6dFGSv subsample:** ~9,000 early-type galaxies with Fundamental Plane measurements — largest single homogeneous PV sample until DESI PV.
- **Reference:** Jones et al. (2004, 2009); Springob et al. (2014).

#### GAMA (Galaxy And Mass Assembly)

- **Depth:** r < 19.8, z ~ 0.01–0.35, ~300K galaxies, ~250 deg².
- **Completeness:** > 98% by design — **the gold standard for completeness modeling**.
- **Key observables:** Multi-wavelength photometry (UV to far-IR), stellar masses, SFRs, group catalog.
- **Reference:** Driver et al. (2011, 2022); Liske et al. (2015).

#### DESI BGS (Bright Galaxy Survey)

- **Depth:** r < 19.5 (BGS Bright) / 20.175 (BGS Faint), z ~ 0.01–0.6, ~10–15M galaxies, ~14,000 deg².
- **Weights:** `WEIGHT_ZFAIL`, `WEIGHT_FKP`, `WEIGHT_COMP` (bit-masked weight system).
- **Reference:** DESI Collaboration (2016, 2023 EDR, 2024 DR1).

#### 2MRS (2MASS Redshift Survey)

- **Depth:** K_s < 11.75, z ~ 0–0.1, ~45K galaxies, ~91% of sky.
- **Use:** Near-IR selection minimizes stellar mass bias; density field reconstruction and PV predictions.
- **Reference:** Huchra et al. (2012).

### 2.2 Photometric Surveys

| Survey | z range | Sky (deg²) | Depth | Notes |
|---|---|---|---|---|
| DES DR2 | 0.1–1.5 | ~5,000 | i < 24 | g/r/i/z/Y; photo-z from BPZ/DNF |
| KiDS | 0.1–1.5 | ~1,350 | r < 25 | u/g/r/i + VIKING near-IR |
| HSC Wide | 0.1–4 | ~1,400 | i < 26.4 | Excellent PSF (0.6'' FWHM) |
| Euclid | 0–2 | ~15,000 | VIS < 24.5 | + NISP Y/J/H; ~35M grism spec-z |
| LSST/Rubin | 0–3 | ~18,000 | r < 27.5 | u/g/r/i/z/y; ~20B sources (10 yr) |

### 2.3 Peculiar Velocity Surveys

#### CosmicFlows (CF1/CF2/CF3/CF4)

- **CF4 (2023):** ~56,000 galaxies, z < 0.07. Compilation of TF (HI, optical, WISE), FP, SBF, SNe Ia.
- **Use:** Large-scale velocity field reconstruction, H0 measurement, bulk flow.
- **Reference:** Tully et al. (2013, 2016, 2023).

#### 2MTF (2MASS Tully-Fisher Survey)

- ~2,000 spiral galaxies, z < 0.03. HI line widths from Parkes + GBT.
- **Key observables:** W50/W20 (inclination-corrected HI width), J/H/K magnitudes, TF distance modulus.
- **Reference:** Masters et al. (2008); Said et al. (2020).

#### 6dFGSv

- ~9,000 early-type galaxies, z < 0.055. Fundamental Plane distances.
- **Reference:** Springob et al. (2014); Scrimgeour et al. (2016).

#### DESI PV

- ~150,000 PV measurements, z < 0.1. Combination of FP (early-type) + TF (spirals + HI from ALFALFA/WALLABY).
- **Reference:** DESI PV papers (2024, 2025).

#### WALLABY PV

- HI 21-cm with ASKAP. Target: ~200,000 HI detections for PV science, z < 0.08.
- **Reference:** Koribalski et al. (2020).

#### SFI++ (Spiral Field I-band)

- ~4,000 field + ~1,300 cluster spirals, z < 0.05, I-band TF.
- **Reference:** Masters et al. (2006); Springob et al. (2007).

### 2.4 Type Ia Supernova Surveys

SNe Ia provide the most precise individual PV measurements at low z (sigma_mu ~ 0.15 mag → sigma_v ~ 750 km/s at cz = 10,000 km/s).

| Survey | N_SN | z range | Notes |
|---|---|---|---|
| Pantheon+ | ~1,700 | 0.01–2.3 | Gold standard low-z anchor |
| ZTF SNe | ~3,000 | z < 0.1 | Large low-z sample; bulk flow |
| DES-SN | ~1,500 spec + ~20K photo | 0.1–1.3 | Deep photometric classification |

**Key observables:** `RA`, `DEC`, `Z_CMB`, `MU`, `MU_ERR`, `x1` (SALT2 stretch), `c` (SALT2 color).

### 2.5 Radio / HI Surveys

| Survey | Sky | z range | N_HI | Notes |
|---|---|---|---|---|
| ALFALFA alpha.100 | ~7,000 deg² | z < 0.06 | ~31,500 | Standard TF widths |
| WALLABY | ~75% sky | z < 0.26 | ~500,000 | ASKAP; TF for PV |
| HIPASS | Southern sky | z < 0.04 | ~5,000 | Parkes legacy |

### 2.6 CMB Lensing Maps

CMB lensing convergence maps (kappa maps) from Planck, SPT, ACT provide projected mass maps for:
- Galaxy–CMB lensing cross-correlation.
- Galaxy bias constraints.

For `oneuniverse`, CMB lensing is best handled as external HEALPix maps cross-correlated with the galaxy catalog via `NaMaster` or `healpy`.

### 2.7 Summary Table

| Survey | Type | z range | Sky (deg²) | N_gal |
|---|---|---|---|---|
| SDSS MGS | Spectro | 0.01–0.25 | ~10,000 | ~700K |
| SDSS BOSS | Spectro | 0.15–0.7 | ~10,000 | ~1.5M |
| 2dFGRS | Spectro | 0.01–0.3 | ~1,500 | ~221K |
| 6dFGS | Spectro | 0.001–0.2 | ~17,000 | ~136K |
| GAMA | Spectro | 0.01–0.35 | ~250 | ~300K |
| 2MRS | Spectro | 0.001–0.1 | ~41,000 | ~45K |
| DESI BGS | Spectro | 0.01–0.6 | ~14,000 | ~15M |
| DES | Photo | 0.1–1.5 | ~5,000 | ~700M |
| KiDS | Photo | 0.1–1.5 | ~1,350 | ~100M |
| HSC Wide | Photo | 0.1–4.0 | ~1,400 | ~800M |
| Euclid | Photo+Spectro | 0–2 | ~15,000 | ~2B |
| LSST | Photo | 0–3 | ~18,000 | ~20B |
| CF4 | PV (compilation) | 0–0.07 | all-sky | ~56K |
| 2MTF | PV (TF/HI) | 0–0.03 | all-sky | ~2K |
| 6dFGSv | PV (FP) | 0–0.055 | ~17,000 | ~9K |
| DESI PV | PV (FP+TF) | 0–0.1 | ~14,000 | ~150K |
| ALFALFA | HI | 0–0.06 | ~7,000 | ~31K |
| WALLABY | HI | 0–0.26 | ~30,000 | ~500K |
| Pantheon+ | SNe Ia | 0.01–2.3 | all-sky | ~1,700 |
| ZTF SNe | SNe Ia | 0.01–0.1 | ~15,000 | ~3,000 |

---

## 3. Weighting Schemes

### 3.1 FKP Weights (Feldman-Kaiser-Peacock 1994)

The FKP weight minimizes variance of the power spectrum estimator at a given wavenumber k:

```
w_FKP(r) = 1 / (1 + n_bar(r) * P_FKP)
```

- `n_bar(r)` = expected mean galaxy number density from the radial selection function.
- `P_FKP` = reference power spectrum amplitude.

**Standard values:**
- `P_FKP = 4000 h^{-3} Mpc^3`: low-z surveys (SDSS MGS, 6dFGS, DESI BGS at z < 0.2).
- `P_FKP = 10000 h^{-3} Mpc^3`: intermediate-z surveys (BOSS CMASS, eBOSS).

**Properties:** Upweights galaxies in underdense/high-z regions (shot noise dominated); downweights galaxies in overdense/low-z shells (cosmic variance dominated).

**Computation recipe:**
1. Estimate `n(z)` from the survey's observed redshift distribution, corrected for completeness.
2. Convert to `n_bar(r)` in 3D comoving coordinates via the cosmological model.
3. Compute `w_FKP = 1 / (1 + n_bar * P_FKP)` for each galaxy.

**Storage column:** `W_FKP` (float32).

### 3.2 Angular Completeness Weights

```
w_comp = 1 / C_spec(theta, phi)
C_spec = N_spec(good) / N_target(parent)
```

- **SDSS/BOSS:** Per survey **sector** (Mangle polygons).
- **DESI:** Per **HEALPix pixel** (nside=256 or 512) per fiber-assignment pass.
- **GAMA:** > 98% everywhere by design; completeness weight is near 1 almost universally.

**Storage columns:** `W_COMP` (float32, inverse completeness), `C_SPEC` (float32, forward completeness 0–1).

### 3.3 Fiber Collision (Close-Pair) Weight

Two targets within the minimum fiber separation (55'' for SDSS; ~30'' effective for DESI) cannot both receive a fiber in a single pass.

**Nearest-neighbor (NN) upweighting (SDSS/BOSS standard):**
```
w_cp = 1 + (N_no_spec_neighbors / N_spec_neighbors)
```

The **pairwise inverse probability (PIP) weight** (Percival et al. 2017) provides a more principled but computationally expensive alternative. For DESI, multiple passes drastically reduce fiber collision incompleteness.

**Storage column:** `W_CP` (float32).

### 3.4 Redshift Failure Weight

For galaxies with a failed redshift measurement (ZWARNING != 0), the nearest successfully-observed neighbor is upweighted.

For DESI: `WEIGHT_ZFAIL = 1 / P(good z | observed)`.

**Storage column:** `W_NOZ` (float32).

### 3.5 Total Spectroscopic Weight

**BOSS/eBOSS convention:**
```
w_tot = (w_cp + w_noz - 1) * w_sys * w_FKP
```

**DESI convention:**
```
w_tot = WEIGHT_SYS * WEIGHT_COMP * WEIGHT_ZFAIL * WEIGHT_FKP
```

**Storage column:** `W_TOT` (float32).

### 3.6 Photo-z Quality Weights

- **ODDS threshold:** `w_pz = 1` if `ODDS > threshold` (e.g., 0.7 or 0.9), else 0; or `w_pz = ODDS` directly.
- **Inverse variance:** `w_pz = 1 / sigma_z^2` for cross-correlation analyses.
- **PDF-based (correct approach):** Full `P_i(z)` PDF propagated through the statistical estimator. Mandatory for tomographic weak lensing.

### 3.7 Radial Selection Function n(z)

The expected mean galaxy number density as a function of redshift for a flux-limited survey:

```
n(z) = integral_{L_min(z)}^{infty}  Phi(L) dL
```

**Construction methods:**
1. **Schechter function fit** to the observed luminosity function.
2. **V_max method:** `n(z_bin) dV = sum_{i in bin} 1 / V_max,i`.
3. **Smoothed histogram** of the observed N(z) after completeness correction.

**Storage:** Tabulated arrays `Z_BIN_EDGES`, `N_BAR` (Mpc^{-3}) in survey metadata.

### 3.8 Survey Masks (Angular Footprints)

**Common veto conditions:**
- Bright star masks (Tycho-2/Gaia stars brighter than r ~ 12).
- Bad CCD / bad column masks.
- Galactic plane cuts (|b| < b_min, typically 10°–20°).
- Bright galaxy masks (Virgo cluster, Magellanic Clouds, M31).
- Satellite trail masks.

**Fractional coverage per HEALPix pixel:**
```
w_frac = A_unmasked / A_pixel   (0 to 1)
```

**HEALPix pixel sizes:**

| NSIDE | N_pix | Pixel area | Typical use |
|---|---|---|---|
| 64 | ~50K | 0.84 deg² | Coarse n(z) maps |
| 256 | ~786K | 13.4 arcmin² | Survey footprint, FKP |
| 512 | ~3.1M | 3.4 arcmin² | Bright star masks |
| 4096 | ~200M | 0.77 arcsec² | Fine satellite trail masks |

**Mangle polygons:** For SDSS/BOSS, masks stored as Mangle polygon files (`.ply`). The `pymangle` library provides Python access.

**Random catalogs:** Essential for power spectrum / correlation function estimation (Landy-Szalay estimator). Randoms must have the same angular footprint, same n(z), be 10–50× larger than the data, and carry the same weights.

### 3.9 Imaging Systematic Weights

Survey depth, seeing, sky brightness, Galactic extinction, and stellar density modulate the effective survey volume.

**Correction methods:**
1. **Linear regression:** Fit `delta_g = sum_j alpha_j * s_j` and subtract.
2. **Non-linear (neural network):** `SYSNET` (Rezaie et al. 2020), used in DESI EDR/DR1.
3. **Weight assignment:** `w_sys = 1 / (1 + sum_j alpha_j * s_j)` per galaxy.

**Storage column:** `W_SYS` (float32).

### 3.10 Inverse-Variance Weighting for Peculiar Velocities

Optimal weight for PV surveys:

```
w_PV = 1 / (sigma_v^2 + sigma_*^2)
```

- `sigma_v = (ln(10)/5) * sigma_mu * cz_CMB` km/s: propagated velocity uncertainty.
- `sigma_* ~ 150–300 km/s`: intrinsic "thermal" velocity dispersion.

This weight **strongly downweights high-z galaxies** because `sigma_v` grows linearly with `cz_CMB` — explaining why PV surveys are inherently low-z.

**Bulk flow estimator:**
```
V_bulk = sum_i w_i * v_i / sum_i w_i
```

**Storage columns:** `V_PEC`, `V_PEC_ERR`, `W_PV`, `SIGMA_V_THERMAL`.

---

## 4. Cross-Survey Consistency Challenges

### 4.1 Different Selection Functions and Magnitude Limits

Each survey's targeting defines a unique selection function:
- Flux limit and band: optical r (SDSS, DESI BGS), near-IR K (6dFGS, 2MRS), HI flux (ALFALFA).
- Color selection: BOSS CMASS uses g-r-i color cuts to select z ~ 0.5 LRGs.
- Surface brightness limit: low-SB galaxies may be in photometric catalogs but missed by spectroscopic targeting.

**Implication:** Cross-catalog matching yields only the **intersection** of both selection functions. Store each object's survey membership and selection function explicitly.

**Key metadata:** `SURVEY_ID`, `IN_SURVEY_{X}` (boolean per survey), `NTILE_PASS` (DESI fiber-assignment passes).

### 4.2 Overlapping Sky Areas with Different Depths

- Two surveys may provide independent spec-z (useful for redshift quality validation).
- Different angular resolutions require cross-matching within a tolerance (1''–2'' for optical; 10''–30'' for radio).

**Approach:**
- Designate a **master photometric catalog** (deepest/most reliable; e.g., DESI Legacy Surveys DR10 or Euclid in overlap regions).
- **Priority scheme for Z_BEST:** z_spec (quality >= threshold) > z_phot. Record all available redshifts in separate columns.
- **De-duplication:** Positional cross-match → unique `GALAXY_ID` → deterministic priority rule.

### 4.3 Photometric vs. Spectroscopic Redshifts

Mixing photo-z and spec-z requires:
- Propagating the full P(z) PDF through statistical estimators.
- Using **projected statistics** [w(theta), w_p(r_p)] rather than full 3D when photo-z scatter is large.
- **Clustering-based redshift calibration** (cross-correlating photometric sample with spec-z reference, Menard et al. 2013).

**For `oneuniverse`:** Store `Z_SPEC` and `Z_PHOT` separately. Provide `Z_BEST` (with documented priority rule) and `Z_TYPE` (0=spec, 1=phot, 2=other). Do not silently merge into a single Z column.

### 4.4 Converting Between Peculiar Velocity Estimators

| Estimator | sigma_mu (mag) | sigma_v at cz = 10,000 km/s |
|---|---|---|
| TF (I-band, field spirals) | 0.40–0.50 | 2000–2500 km/s |
| FP (ellipticals) | 0.38–0.45 | 1900–2200 km/s |
| SBF | 0.10–0.20 | 500–1000 km/s |
| SNe Ia | 0.13–0.16 | 650–800 km/s |

**Malmquist bias:** An apparent-magnitude-limited sample preferentially observes galaxies with brighter apparent magnitudes = closer true distances = positive bias in distance moduli.
- **Homogeneous:** Correction `<mu_obs> ≈ mu_true + (ln(10)/5)^2 * sigma_mu^2 * d ln n / d mu`.
- **Inhomogeneous:** Requires a model of the density field.

**Homogenization strategy (CosmicFlows):**
1. Calibrate zero-points of each method to a common distance scale (Cepheids, TRGB, Megamaser).
2. Apply Malmquist bias corrections consistently.
3. Use a Bayesian hierarchical model to jointly infer distances.

**Storage:** `MU_OBS`, `MU_ERR`, `MU_METHOD`, `MU_CORR` (bias-corrected), `BIAS_FLAG`, `ZERO_POINT_VERSION`.

### 4.5 Photometric Calibration Zero-Points

- **AB vs. Vega:** SDSS uses approximately AB; 2MASS uses Vega. Convert all to AB (e.g., K_Vega → K_AB: +1.85 mag).
- **Filter differences:** SDSS r, DES r (DECam), PanSTARRS r, HSC r have different effective wavelengths. Differences < 0.05 mag but matter for K-corrections.
- **Internal calibration:** Each survey has internal photometric errors of 1–5 mmag (DESI Legacy) to 10–20 mmag (older surveys).

**Approach:** Store the **filter name** and **photometric system** for each magnitude column as column-level metadata.

---

## 5. Standard Data Formats and Schemas

### 5.1 HEALPix Sky Pixelization

```python
import healpy as hp
import numpy as np

NSIDE = 256
# RA/Dec -> HEALPix pixel (RING ordering)
ipix = hp.ang2pix(NSIDE, np.radians(90.0 - dec), np.radians(ra), nest=False)

# Cone query: all pixels within 1 deg of (ra0, dec0)
vec  = hp.ang2vec(np.radians(90 - dec0), np.radians(ra0))
pixs = hp.query_disc(NSIDE, vec, np.radians(1.0))
```

**Ordering conventions:**
- `RING`: Default; required for spherical harmonic transforms.
- `NESTED`: Hierarchical; adjacent pixels share a prefix. Better for neighbor lookup and multi-scale analysis. **Use NESTED for spatial indexing.**

**COORDSYS in FITS header:** `C` = Celestial (ICRS), `G` = Galactic, `E` = Ecliptic.

**Recommended mask format for `oneuniverse`:**
```
{SURVEY_ID}_mask_nside{NSIDE}.fits
HDU 1: FRACGOOD [float32]: fraction of pixel area unmasked (0=masked, 1=open)
       NSIDE = 256, ORDERING = RING, COORDSYS = C
```

### 5.2 Catalog Header Metadata (FITS / HDF5 attrs / Parquet sidecar)

```yaml
survey:              "DESI_BGS_DR1"
survey_version:      "1.0"
oneuniverse_version: "0.1.0"
area_deg2:           14000.0
z_min:               0.01
z_max:               0.60
mag_band:            "r_DECAM"
mag_lim:             19.5
cosmo_name:          "Planck18"
cosmo_H0:            67.36
cosmo_Om0:           0.3153
cosmo_Ode0:          0.6847
cosmo_flat:          true
P_FKP:               4000.0       # h^-3 Mpc^3
v_LG:                620.0        # km/s
apex_l:              271.4        # Galactic longitude (deg)
apex_b:              29.3         # Galactic latitude (deg)
ebv_map:             "SF11"       # Schlafly & Finkbeiner 2011
reference:           "DESI Collaboration 2024"
```

---

## 6. Recommended Schema for `oneuniverse`

### 6.1 Design Principles

1. **Cosmology-aware:** All distances, FKP weights, and n(z) tagged with the cosmological model. Changing cosmology recomputes derivatives without touching raw observables.
2. **Survey-agnostic core:** Minimal core schema every survey must provide; optional extension columns per survey type.
3. **Lazy computation:** Cosmology-dependent quantities (D_C, D_L, W_FKP) computed on demand and cached.
4. **Randoms as first-class citizens:** Stored alongside data with the same column structure.
5. **Weight transparency:** All constituent weights stored separately (W_FKP, W_COMP, W_CP, W_SYS, W_PV) in addition to W_TOT.
6. **Provenance tracking:** Every row traceable to its source survey and processing version.

### 6.2 Core Schema (All Survey Types)

```python
CORE_COLUMNS = {
    # Position (float64 for sub-arcsecond precision)
    'RA':           ('f8', 'deg',  'Right ascension ICRS J2000'),
    'DEC':          ('f8', 'deg',  'Declination ICRS J2000'),
    # Redshift
    'Z_BEST':       ('f4', '',     'Best available redshift'),
    'Z_TYPE':       ('i1', '',     '0=spec, 1=phot, 2=other'),
    # Identifiers
    'GALAXY_ID':    ('i8', '',     'Unique ID in oneuniverse'),
    'SURVEY_ID':    ('U32','',     'Source survey name'),
    # Weights
    'W_COMP':       ('f4', '',     'Angular completeness weight'),
    'W_TOT':        ('f4', '',     'Total analysis weight'),
    # Mask
    'IN_FOOTPRINT': ('i1', '',     '1 if in survey footprint'),
    'HPIX256':      ('i4', '',     'HEALPix pixel NSIDE=256 RING ICRS'),
}
```

### 6.3 Spectroscopic Extension

```python
SPEC_COLUMNS = {
    'Z_SPEC':       ('f4', '',     'Spectroscopic redshift'),
    'Z_SPEC_ERR':   ('f4', '',     'Spectroscopic redshift uncertainty'),
    'Z_SPEC_QUAL':  ('i1', '',     'Quality flag (survey-native scale)'),
    'Z_HELIO':      ('f4', '',     'Heliocentric redshift'),
    'Z_CMB':        ('f4', '',     'CMB-frame redshift'),
    'CZ_CMB':       ('f4', 'km/s','CMB-frame recession velocity'),
    'W_FKP':        ('f4', '',     'FKP weight'),
    'W_CP':         ('f4', '',     'Fiber collision weight'),
    'W_NOZ':        ('f4', '',     'Redshift failure weight'),
    'W_SYS':        ('f4', '',     'Imaging systematic weight'),
    'MAG_R':        ('f4', 'ABmag','r-band apparent magnitude'),
    'EBV':          ('f4', 'mag',  'E(B-V) Milky Way extinction'),
    'A_R':          ('f4', 'mag',  'r-band Galactic extinction'),
}
```

### 6.4 Photometric Extension

```python
PHOTO_COLUMNS = {
    'Z_PHOT':       ('f4', '',     'Photometric redshift'),
    'Z_PHOT_ERR':   ('f4', '',     'Photo-z 1-sigma uncertainty'),
    'Z_PHOT_L68':   ('f4', '',     'Photo-z 68% lower bound'),
    'Z_PHOT_U68':   ('f4', '',     'Photo-z 68% upper bound'),
    'ODDS':         ('f4', '',     'Photo-z quality (0-1)'),
    'MAG_G':        ('f4', 'ABmag','g-band magnitude'),
    'MAG_R':        ('f4', 'ABmag','r-band magnitude'),
    'MAG_I':        ('f4', 'ABmag','i-band magnitude'),
    'MAG_Z':        ('f4', 'ABmag','z-band magnitude'),
    'W_PHOTO':      ('f4', '',     'Photo-z quality weight'),
}
```

### 6.5 Peculiar Velocity Extension

```python
PV_COLUMNS = {
    'V_PEC':           ('f4', 'km/s','Peculiar velocity'),
    'V_PEC_ERR':       ('f4', 'km/s','Peculiar velocity uncertainty'),
    'MU':              ('f4', 'mag', 'Distance modulus'),
    'MU_ERR':          ('f4', 'mag', 'Distance modulus uncertainty'),
    'MU_METHOD':       ('U8', '',    'TF/FP/SBF/SNIa'),
    'MU_CORR':         ('f4', 'mag', 'Bias-corrected distance modulus'),
    'W_PV':            ('f4', '',    'Inverse-variance PV weight'),
    'SIGMA_V_THERMAL': ('f4', 'km/s','Thermal velocity dispersion'),
    'Z_CMB':           ('f4', '',    'CMB-frame redshift'),
    'CZ_CMB':          ('f4', 'km/s','CMB-frame recession velocity'),
    'BIAS_FLAG':       ('i1', '',    '1 if Malmquist correction applied'),
    'ZERO_POINT_VER':  ('U16','',    'Zero-point calibration version'),
}
```

### 6.6 Suggested Package Structure

```
oneuniverse/
├── __init__.py
├── catalog.py              # UniverseCatalog class: load, merge, query
├── schema.py               # Column definitions, units, dtypes
├── surveys/
│   ├── base.py             # BaseSurvey abstract class
│   ├── spectroscopic/
│   │   ├── sdss.py
│   │   ├── desi_bgs.py
│   │   ├── sixdfgs.py
│   │   ├── gama.py
│   │   └── twomrs.py
│   ├── peculiar_velocity/
│   │   ├── cosmicflows.py
│   │   ├── sixdfgsv.py
│   │   ├── twomtf.py
│   │   └── desi_pv.py
│   ├── photometric/
│   │   ├── des.py
│   │   ├── kids.py
│   │   └── lsst.py
│   └── radio/
│       ├── alfalfa.py
│       └── wallaby.py
├── weights/
│   ├── fkp.py              # FKP weight computation
│   ├── completeness.py     # Angular completeness, mask handling
│   ├── systematic.py       # Imaging systematic weights
│   └── pv.py               # PV inverse-variance weights
├── cosmology/
│   ├── distances.py        # Wrappers around astropy.cosmology
│   └── corrections.py      # CMB frame correction, K-correction
├── masks/
│   ├── healpix.py          # HEALPix mask operations
│   ├── mangle.py           # Mangle polygon interface (via pymangle)
│   └── footprint.py        # Survey footprint union/intersection
├── io/
│   ├── fits.py             # FITS I/O
│   ├── hdf5.py             # HDF5 I/O
│   └── parquet.py          # Parquet I/O
├── crossmatch/
│   └── positional.py       # Positional cross-matching
└── randoms/
    └── generate.py         # Random catalog generation
```

---

## Appendix: Key Formulae

### A.1 FKP Weight

```
w_FKP(z) = 1 / (1 + n_bar(z) * P_FKP)
P_FKP = 4000 h^{-3} Mpc^3 (low-z),  10000 h^{-3} Mpc^3 (high-z)
```

### A.2 CMB Frame Correction

```
cz_CMB = cz_helio + v_LG * cos(theta_LG)
v_LG = 620 km/s,  apex at (l, b) = (271.4°, 29.3°)
```

```python
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

apex  = coord.Galactic(l=271.4*u.deg, b=29.3*u.deg)
gals  = coord.ICRS(ra=ra*u.deg, dec=dec*u.deg).galactic
cos_t = (np.sin(np.radians(gals.b.deg)) * np.sin(np.radians(apex.b.deg)) +
         np.cos(np.radians(gals.b.deg)) * np.cos(np.radians(apex.b.deg)) *
         np.cos(np.radians(gals.l.deg - apex.l.deg)))
cz_cmb = cz_helio + 620.0 * cos_t
```

### A.3 Peculiar Velocity from Distance Modulus

```
v_pec = cz_CMB - H0 * d_true    [km/s, low-z approximation]
sigma_v = (ln(10)/5) * sigma_mu * cz_CMB    [km/s]
```

Exact (relativistic):
```
v_pec = c * [(1 + z_obs)^2 - (1 + z_cos)^2] / [(1 + z_obs)^2 + (1 + z_cos)^2]
```

### A.4 Comoving Distance (Flat LCDM)

```python
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.36, Om0=0.3153)
d_c = cosmo.comoving_distance(z).to('Mpc').value
d_l = cosmo.luminosity_distance(z).to('Mpc').value
d_a = cosmo.angular_diameter_distance(z).to('Mpc').value
```

### A.5 sigma_mu to sigma_v Conversion

```
sigma_v [km/s] = (ln(10)/5) * sigma_mu [mag] * cz_CMB [km/s]
```

Typical values at cz_CMB = 10,000 km/s:
- TF: 0.45 mag → 2250 km/s
- FP: 0.40 mag → 2000 km/s
- SNe Ia: 0.15 mag → 750 km/s
- SBF: 0.15 mag → 750 km/s

### A.6 Inverse-Variance PV Weight

```
w_PV = 1 / (sigma_v^2 + sigma_*^2)
     = 1 / ([(ln10/5) * sigma_mu * cz_CMB]^2 + sigma_*^2)
```

Typical `sigma_* = 150–300 km/s`.

---

## References

**Foundational:**
- Feldman, Kaiser & Peacock (1994), ApJ 426, 23 — FKP weights.
- Hamilton (1997), ASSL 231, 185 — Linear RSD (review).
- Mould et al. (2000), ApJ 529, 786 — Velocity frame corrections.
- Schlafly & Finkbeiner (2011), ApJ 737, 103 — Galactic dust calibration.
- Gorski et al. (2005), ApJ 622, 759 — HEALPix.

**Spectroscopic Surveys:**
- York et al. (2000), AJ 120, 1579 — SDSS.
- Eisenstein et al. (2011), AJ 142, 72 — BOSS.
- Colless et al. (2001), MNRAS 328, 1039 — 2dFGRS.
- Jones et al. (2009), MNRAS 399, 683 — 6dFGS.
- Driver et al. (2011), MNRAS 413, 971 — GAMA.
- Huchra et al. (2012), ApJS 199, 26 — 2MRS.
- DESI Collaboration (2016), arXiv:1611.00036; (2023) arXiv:2306.06308; (2024) DR1.

**Peculiar Velocity Surveys:**
- Tully et al. (2013), AJ 146, 86 — CF2.
- Tully et al. (2016), AJ 152, 50 — CF3.
- Tully et al. (2023), ApJ 944, 94 — CF4.
- Springob et al. (2014), MNRAS 445, 2677 — 6dFGSv.
- Haynes et al. (2018), ApJ 861, 49 — ALFALFA alpha.100.
- Koribalski et al. (2020), Ap&SS 365, 118 — WALLABY.

**Supernovae:**
- Scolnic et al. (2018), ApJ 859, 101 — Pantheon.
- Scolnic et al. (2022), ApJ 938, 113 — Pantheon+.

**Methods:**
- Percival et al. (2017), MNRAS 464, 1168 — PIP fiber collision weights.
- Mohammad et al. (2020), MNRAS 498, 1 — DESI fiber assignment weights.
- Rezaie et al. (2020), MNRAS 495, 1613 — SYSNET imaging systematics.
- Lavaux & Hudson (2011), MNRAS 416, 2840 — 2M++ velocity field.
- Menard et al. (2013), arXiv:1303.4722 — Clustering-based photo-z calibration.
