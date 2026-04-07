# eBOSS DR16Q Superset v3 — Technical Reference

**Date:** 2026-04-03  
**Purpose:** Technical reference for the eBOSS DR16Q quasar catalog loader in `oneuniverse`.

---

## 1. Catalog Overview

**File:** `DR16Q_Superset_v3.fits` (2.48 GB)  
**Reference:** Lyke et al. 2020, ApJS 250, 8  
**URL:** https://www.sdss.org/dr16/algorithms/qso-catalog/

The DR16Q Superset contains **all objects** that were targeted as quasar candidates across SDSS-I/II, BOSS, and eBOSS. It is the superset of the clean DR16Q catalog — meaning it includes confirmed QSOs, stars, galaxies, and unclassified targets.

- **Total rows:** 1,440,615
- **Confirmed QSOs (IS_QSO_FINAL=1):** 919,374 (63.8%)
- **Non-QSOs (IS_QSO_FINAL=0):** 517,410 (35.9%)
- **Bad spectra (IS_QSO_FINAL=-2):** 3,095 (0.2%)
- **Ambiguous (IS_QSO_FINAL=2):** 736 (0.1%)
- **Columns:** 97 total (scalar + multi-element arrays)

The clean DR16Q catalog (750K QSOs) is a subset of IS_QSO_FINAL=1 objects.

---

## 2. Key Columns

### 2.1 Classification

| Column | Dtype | Description |
|--------|-------|-------------|
| IS_QSO_FINAL | i2 | Final QSO classification: 1=QSO, 0=not QSO, -2=bad spectrum, 2=ambiguous |
| IS_QSO_QN | i2 | QuasarNET classification (0/1) |
| IS_QSO_10K | i2 | 10K visual inspection classification (-1=not inspected) |
| IS_QSO_DR12Q | i2 | Was in DR12Q catalog |
| IS_QSO_DR7Q | i2 | Was in DR7Q catalog |
| AUTOCLASS_PQN | U6 | Automated class: "QSO", "STAR", "GALAXY" |
| AUTOCLASS_DR14Q | U6 | DR14Q automated class |

### 2.2 Redshifts

| Column | Dtype | Description |
|--------|-------|-------------|
| Z | f8 | **Best available redshift** (selected by SOURCE_Z priority) |
| SOURCE_Z | U12 | Source of Z: "VI" (47%), "PIPE" (45%), "DR6Q_HW" (7%), "DR7QV_SCH" (1%), "DR12QV" (<1%) |
| Z_PIPE | f8 | SDSS pipeline (idlspec2d) redshift |
| Z_PCA | f8 | PCA-based redshift |
| Z_QN | f8 | QuasarNET deep learning redshift |
| Z_VI | f8 | Visual inspection redshift (-1 = not inspected) |
| Z_10K | f8 | 10K visual inspection redshift (-1 = not inspected) |
| Z_DR12Q | f8 | DR12Q redshift (-1 = not in DR12Q) |
| Z_DR7Q_SCH | f8 | Schneider DR7Q redshift |
| Z_DR6Q_HW | f8 | Hewett & Wild DR6Q redshift |
| Z_DR7Q_HW | f8 | Hewett & Wild DR7Q redshift |
| ZWARNING | i4 | Pipeline quality flag (0=good, 4=small delta-chi2, 64=unplugged fiber) |
| Z_CONF | i2 | Visual inspection confidence: -1=not inspected, 0=from pipeline, 1-3=increasing confidence |
| Z_CONF_10K | i2 | 10K inspection confidence |

### 2.3 Emission Line Redshifts

| Column | Dtype | Description |
|--------|-------|-------------|
| Z_HALPHA | f8 | H-alpha emission line fit |
| Z_HBETA | f8 | H-beta emission line fit |
| Z_MGII | f8 | MgII emission line (-1 = not measured) |
| Z_CIII | f8 | CIII] emission line |
| Z_CIV | f8 | CIV emission line |
| Z_LYA | f8 | Lyman-alpha emission line |
| ZWARN_* | i8 | Warning flags for each line (7682 = line not covered) |
| DELTACHI2_* | f8 | Delta-chi2 for each line fit |

### 2.4 Broad Absorption Line (BAL) Properties

| Column | Dtype | Description |
|--------|-------|-------------|
| BAL_PROB | f4 | CNN BAL probability: 0-1 (>0.5 = BAL), -1 = not assessed |
| BI_CIV | f8 | CIV balnicity index [km/s] (>0 = traditional BAL definition) |
| ERR_BI_CIV | f8 | BI uncertainty |
| AI_CIV | f8 | CIV absorption index |
| BI_SIIV | f8 | SiIV balnicity index |
| AI_SIIV | f8 | SiIV absorption index |

**Statistics (QSOs only):**
- BAL_PROB > 0.5: ~100K (10.8%)
- BI_CIV > 0 (traditional BAL): ~24K (2.6%)

### 2.5 DLA (Damped Lyman-alpha) Systems

| Column | Shape | Description |
|--------|-------|-------------|
| Z_DLA | (N, 5) | DLA redshifts (up to 5 per sightline, -1 = no DLA) |
| NHI_DLA | (N, 5) | log10(N_HI / cm^-2) |
| CONF_DLA | (N, 5) | DLA detection confidence |

**Statistics:** 48,748 sightlines with ≥1 DLA.

### 2.6 Photometry (ugriz)

| Column | Shape | Description |
|--------|-------|-------------|
| PSFMAG | (N, 5) | PSF magnitudes [u, g, r, i, z] |
| PSFMAGERR | (N, 5) | PSF magnitude errors |
| PSFFLUX | (N, 5) | PSF fluxes [nanomaggies] |
| PSFFLUX_IVAR | (N, 5) | Inverse variance of PSF flux |
| EXTINCTION | (N, 5) | Galactic extinction [u, g, r, i, z] (Schlafly & Finkbeiner 2011) |

**r-band median:** ~20.5 mag (PSF)

### 2.7 Targeting Bits

| Column | Dtype | Description |
|--------|-------|-------------|
| BOSS_TARGET1 | i8 | BOSS targeting bits |
| EBOSS_TARGET0 | i8 | eBOSS commissioning targets |
| EBOSS_TARGET1 | i8 | eBOSS QSO targeting bits (bit 10 = CORE QSO target) |
| EBOSS_TARGET2 | i8 | eBOSS ancillary targets |
| ANCILLARY_TARGET1 | i8 | SDSS ancillary |
| ANCILLARY_TARGET2 | i8 | SDSS ancillary |

### 2.8 Observation Metadata

| Column | Dtype | Description |
|--------|-------|-------------|
| RA | f8 | Right ascension (ICRS, J2000, degrees) |
| DEC | f8 | Declination (ICRS, J2000, degrees) |
| PLATE | i4 | SDSS plate number |
| MJD | i4 | Modified Julian Date of observation |
| FIBERID | i2 | Fiber ID (1-1000) |
| THING_ID | i8 | Unique SDSS object identifier |
| SDSS_NAME | U18 | SDSS J2000 name (HHMMSS.ss+DDMMSS.s) |
| NSPEC | i4 | Number of spectra (-1 = eBOSS-only) |
| LAMBDA_EFF | f8 | Effective wavelength of fiber [Å] (4000 or 5400) |
| SN_MEDIAN_ALL | f8 | Median S/N per pixel over full spectrum |

---

## 3. Sentinel Values

| Value | Meaning |
|-------|---------|
| -1 | Not measured / not applicable (most common sentinel) |
| -999 | Catastrophic failure or missing data (rare, 4 QSOs) |
| 7682 | ZWARN for emission lines: line not covered by spectrum |

---

## 4. Quality Cuts

### 4.1 Clean QSO Sample
```
IS_QSO_FINAL == 1  →  919K objects
```

### 4.2 Pipeline-confident QSOs
```
IS_QSO_FINAL == 1  AND  ZWARNING == 0  →  ~800K objects (87%)
```

### 4.3 Visually-inspected confident QSOs
```
IS_QSO_FINAL == 1  AND  Z_CONF == 3  →  ~428K objects (47%)
```

### 4.4 Lyman-alpha forest QSOs
```
IS_QSO_FINAL == 1  AND  Z > 2.1  →  ~312K objects
```

### 4.5 Non-BAL QSOs (for clustering)
```
IS_QSO_FINAL == 1  AND  BAL_PROB < 0.5  →  ~820K objects
```

---

## 5. I/O Performance

File is 2.48 GB FITS binary table. Benchmarks on this system:

| Method | Time | Notes |
|--------|------|-------|
| fitsio full read (all 97 cols, 1.44M rows) | 4.7s | |
| fitsio selective (4 scalar cols) | 3.4s | Column I/O still reads full rows |
| fitsio two-pass (filter + selective) | 33s | Our approach: read IS_QSO_FINAL first, then rows |
| **Parquet (zstd)** | **0.07s** (4 cols) | **50x faster**, 146 MB vs 2482 MB |
| Parquet with predicate pushdown | 0.05s | Filters at the storage level |

**Recommendation:** Convert to Parquet on first load for repeat access.

---

## 6. Coordinate System

- **Frame:** ICRS (International Celestial Reference System)
- **Epoch:** J2000
- **RA range:** [0°, 360°]  
- **Dec range:** [-17.5°, 84.4°] (eBOSS footprint is northern galactic cap + south galactic cap)
- **Sky coverage:** ~10,000 deg² (SDSS imaging footprint), ~24% of sky

---

## 7. Redshift Statistics (QSOs only)

| Statistic | Value |
|-----------|-------|
| Min z (valid) | 0.001 |
| Max z | 7.024 |
| Median z | 1.748 |
| Mean z | 1.777 |
| z < 0 (sentinel) | 7 objects |
| 0 < z < 0.5 | ~50K |
| 0.5 < z < 1.0 | ~75K |
| 1.0 < z < 2.0 | ~330K |
| 2.0 < z < 3.0 | ~360K |
| 3.0 < z < 4.0 | ~90K |
| z > 4.0 | ~14K |

The bimodal redshift distribution reflects the two targeting strategies:
- **Low-z peak (~1.5):** CORE eBOSS QSO targets for BAO at 0.8 < z < 2.2
- **High-z peak (~2.5):** Lyman-alpha forest QSOs for BAO at z > 2.1

---

## 8. Key References

- **DR16Q catalog:** Lyke et al. 2020, ApJS, 250, 8
- **DR14Q catalog:** Pâris et al. 2018, A&A, 613, A51
- **eBOSS QSO targeting:** Myers et al. 2015, ApJS, 221, 27
- **QuasarNET:** Busca & Balland 2018
- **SDSS-IV/eBOSS overview:** Dawson et al. 2016, AJ, 151, 44
- **BAL catalog:** Guo & Martini 2019

---

## 9. Column Mapping (FITS → oneuniverse)

```
RA              → ra           (f8, deg, ICRS)
DEC             → dec          (f8, deg, ICRS)
Z               → z            (f8, best redshift)
IS_QSO_FINAL    → is_qso       (i1, classification)
SOURCE_Z        → source_z     (U16, redshift source)
Z_PIPE          → z_pipe       (f8)
Z_PCA           → z_pca        (f8)
Z_VI            → z_vi         (f8)
Z_QN            → z_qn         (f8)
Z_CONF          → z_conf       (i1)
ZWARNING        → zwarning     (i4)
CLASS_PERSON    → class_person  (i2)
BAL_PROB        → bal_prob     (f4)
BI_CIV          → bi_civ       (f8, km/s)
Z_MGII          → z_mgii       (f8)
Z_CIII          → z_ciii       (f8)
Z_CIV           → z_civ        (f8)
Z_LYA           → z_lya        (f8)
Z_HALPHA        → z_halpha     (f8)
Z_HBETA         → z_hbeta      (f8)
PLATE           → plate        (i4)
MJD             → mjd          (i4)
FIBERID         → fiberid      (i2)
THING_ID        → thing_id     (i8)
SDSS_NAME       → sdss_name    (U18)
BOSS_TARGET1    → boss_target1 (i8)
EBOSS_TARGET0   → eboss_target0 (i8)
EBOSS_TARGET1   → eboss_target1 (i8)
EBOSS_TARGET2   → eboss_target2 (i8)
SN_MEDIAN_ALL   → sn_median    (f4)
LAMBDA_EFF      → lambda_eff   (f8)
NSPEC           → nspec        (i4)
PSFMAG[5]       → psfmag_u/g/r/i/z  (f4, mag)
EXTINCTION[5]   → extinction_u/g/r/i/z  (f4, mag)
Z_DLA[5]        → n_dla        (i1, count of valid DLAs)
```
