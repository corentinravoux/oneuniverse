# oneuniverse

**A unified observational foundation for the digital twin of our Universe.**

`oneuniverse` is a Python package that ingests, standardises, cross-matches,
and weights astronomical survey catalogs so that every observation of the sky
— spectroscopic redshifts, photometric maps, peculiar velocities, supernovae,
Lyman-alpha forests, CMB lensing — lives in a single, queryable database.

There is one Universe. Every telescope observes the same cosmic web, the same
voids, the same filaments, the same clusters — through different instruments,
different wavelengths, different systematic distortions, different epochs. This
package exists because **no single survey sees the whole picture**, but
together they constrain the underlying matter and velocity fields far better
than any one of them alone.

---

## Vision

`oneuniverse` is built around three goals of increasing ambition:

### 1. Unified data access

Every survey has its own file format, column naming convention, coordinate
system, quality flags, and magnitude system. A researcher who wants to
combine eBOSS quasars with DESI spectra, DES photometry, and CosmicFlows-4
peculiar velocities currently has to write hundreds of lines of survey-specific
I/O code before doing any science.

`oneuniverse` solves this at the source: each survey gets a **loader** that
maps native columns into a standardised schema, applies quality cuts, and
writes the result into a partitioned Parquet format (the _oneuniverse file
format_, OUF). A `OneuniverseDatabase` object then discovers all converted
datasets in a directory tree and exposes them through a uniform Python API —
column pushdown, spatial selections, redshift shells — regardless of whether
the original data was FITS, HDF5, CSV, or Parquet.

```python
from oneuniverse.data import OneuniverseDatabase, Cone, Shell

db = OneuniverseDatabase("/data/oneuniverse_database")
print(db.summary())

# Load only the columns you need, from the region you care about
df = db.load("spectroscopic_desi_dr1_qso",
             selection=[Cone(ra=185, dec=15, radius=5), Shell(z_min=1.0, z_max=2.5)],
             columns=["ra", "dec", "z", "z_spec_err"])
```

### 2. Cross-survey science

Once the data is in a common format, the natural next step is to link it
across surveys. The **ONEUID** (one-universe identifier) system does exactly
this: it cross-matches objects by angular separation and redshift consistency
using an Astropy ball-tree + scipy connected-components algorithm, and assigns
a single integer identifier to each physical object regardless of how many
surveys observed it.

This makes cross-correlation studies straightforward. With ONEUID in hand:

- **Redshift calibration**: compare spectroscopic z from DESI with eBOSS for
  the ~200k quasars observed by both, quantify systematic offsets per pipeline.
- **Multi-tracer power spectra**: the optimal estimator for the matter power
  spectrum P(k) uses all available tracers simultaneously, with survey-specific
  bias and shot noise — but requires knowing which objects are shared.
- **Peculiar velocity + density joint analysis**: `flip` needs both a density
  field (from spectroscopic galaxy positions) and a velocity field (from
  distance indicators); `oneuniverse` provides both in a single weighted
  catalog, with the cross-match handling overlapping footprints.
- **Photo-z training and validation**: photometric surveys calibrate their
  redshifts against spectroscopic truth tables — ONEUID automates the
  spectro-photo match.

The `weight` subpackage handles the statistical subtleties: per-survey weights
(FKP, inverse-variance, completeness, quality masks), and combination
strategies (`best_only`, `ivar_average`, hyperparameter weighting) that
preserve the inverse-variance interpretation downstream codes like `flip`
rely on.

### 3. Digital twin of the Universe

This is the most ambitious goal, and the reason the package is called
_oneuniverse_.

The idea: given enough observations from enough surveys, reconstruct the
**actual three-dimensional matter density field and velocity field of our
Universe** — not a statistical average, not a random realisation from a
cosmological prior, but the specific cosmic web that we inhabit.

#### Why is this possible?

The matter density contrast delta(x, z) is a single scalar field. Every
observable is a (biased, noisy, projected) function of it:

| Tracer | Relation to delta | What it constrains |
|--------|------------------|--------------------|
| Spectroscopic galaxies | n_g(x) = n_bar (1 + b_g delta) | Biased density field in 3D |
| Peculiar velocities | v(x) = -f a H integral[delta] / (4 pi r^2) | Velocity field, growth rate f*sigma_8 |
| SNe Ia distance moduli | mu = mu_cos + 5/(ln 10) * v_pec / (c z) | Velocity field along individual lines of sight |
| Lyman-alpha forest | delta_F = b_F delta + b_eta eta + noise | Density field in the diffuse IGM, z > 2 |
| CMB lensing kappa | kappa = integral[W(chi) delta] d chi | Projected matter density (line-of-sight integral) |
| Photometric galaxies | angular n(theta) ~ integral[b_g delta phi(z)] dz | 2D projected density with photo-z smearing |
| Weak lensing shear | gamma ~ integral[W(chi) delta] d chi | Projected matter density, complementary to kappa |

Because all of these trace the same underlying field, **combining them breaks
degeneracies** that no single probe can resolve alone. The galaxy bias b_g is
degenerate with sigma_8 in galaxy clustering, but peculiar velocities measure
f*sigma_8 independently. Photo-z uncertainties blur the radial information, but
spectroscopic cross-calibration sharpens it. CMB lensing and galaxy lensing
have different redshift kernels, so their cross-correlation constrains the
growth of structure D(z) as a function of time.

#### The reconstruction pipeline

The digital twin is built through **constrained Bayesian forward modelling**:

1. **Prior**: a Gaussian random field with the theoretical power spectrum
   P(k, z) from LCDM (or extensions). This encodes our knowledge of
   cosmological initial conditions.

2. **Forward model**: evolve initial conditions through gravitational
   dynamics — either full N-body, approximate methods (2LPT, COLA,
   particle-mesh), or learned surrogates — to predict the present-day
   density and velocity fields.

3. **Observation model**: for each survey in the `oneuniverse` database,
   apply the appropriate bias model, selection function, survey geometry
   (mask, completeness), and noise model to produce mock observables that
   can be compared to the real data.

4. **Likelihood**: compare mock and real data. For Gaussian fields this is
   the standard multivariate Gaussian (what `flip` already implements);
   for the full non-linear field, simulation-based inference or
   Hamiltonian Monte Carlo on the initial conditions (a la BORG, ELUCID,
   LEX).

5. **Posterior**: the constrained realisation that maximises the posterior
   is the digital twin — a 3D map of delta(x) and v(x) that is
   simultaneously consistent with all survey data.

#### Multi-scale re-simulation

Once the large-scale structure is fixed by the global reconstruction, specific
regions can be **re-simulated at higher resolution** without rerunning the
entire volume. This is the "mini-simulation" concept:

- **Boundary conditions from the global field**: the large-scale tidal field
  and bulk flow at the boundary of the zoom region come from the
  reconstruction.
- **Hydrodynamic physics inside**: baryonic processes (cooling, star
  formation, AGN feedback, chemical enrichment) are modelled with
  SPH/AMR codes (e.g. Nyx, Arepo, RAMSES) only within the zoom volume.
- **Incremental updates**: when new survey data arrives for a specific sky
  region (e.g. a new DESI tile), only the affected local volume needs to be
  re-constrained and re-simulated — not the full 1 Gpc/h box.

This hierarchical approach makes the digital twin **maintainable**: as the
`oneuniverse` database grows (new surveys, new data releases), the
reconstruction improves locally without requiring a monolithic re-analysis.

---

## Physics in the code

The package design is driven by the physics of what each survey measures.

### Data geometries

Not all observations are point catalogs. The format specification
(`format_spec.py`) supports three geometries:

- **POINT** — one row per object (galaxy, QSO, SN). Standard catalog with
  scalar properties. This is the default for spectroscopic and peculiar
  velocity surveys.
- **SIGHTLINE** — one row per spectral pixel, grouped by sightline. Needed for
  Lyman-alpha forest delta_F fields, where the observable is a 1D field along
  each QSO line of sight, not a single number per object.
- **HEALPIX** — one row per HEALPix pixel. Used for CMB lensing convergence
  maps, survey depth maps, photometric source density maps, and any quantity
  defined on the sphere rather than at discrete object positions.

These three geometries cover all the observables listed in the table above.
The manifest file declares the geometry, so any reader can handle the data
correctly without prior knowledge of the survey.

### Schema and column standardisation

The schema (`schema.py`) defines column groups: **core** (ra, dec, z,
galaxy_id, survey_id — mandatory for all surveys), **spectroscopic** (z_spec,
z_spec_err, FKP/completeness weights), **photometric** (z_phot, photo-z
uncertainties, odds), **peculiar_velocity** (v_pec, distance modulus, TF/FP
method), **qso** (classification, BAL properties, emission line redshifts,
DLA counts), and **snia** (SALT2 light-curve parameters).

Each loader declares which groups it provides. The schema validation ensures
that required columns are present with correct dtypes, while allowing surveys
to carry arbitrary additional (characteristic) fields that are
survey-specific.

### Cross-match physics

The ONEUID cross-match (`oneuid.py`) is physically motivated:

- **Sky tolerance** (default 2 arcsec): accounts for astrometric differences
  between surveys. SDSS and DESI have ~0.1 arcsec precision, so 2 arcsec is
  conservative and handles extended sources.
- **Redshift tolerance** (default dz = 5e-3): accounts for pipeline
  differences, catastrophic redshift failures, and the physical velocity
  dispersion of cluster members (~1000 km/s ~ dz ~ 0.003).
- **Transitivity**: the connected-components algorithm makes the match
  transitive — if A~B in survey 1 and B~C in survey 2, then A, B, C share
  one ONEUID. This is correct for linking heterogeneous surveys through an
  intermediate.

### Weighting and combination

The weight system (`weight/`) mirrors the statistical framework used in
large-scale structure analyses:

- **FKP weights** (Feldman, Kaiser & Peacock 1994):
  w_FKP(z) = 1 / (1 + n_bar(z) P_0), where n_bar(z) is the survey number
  density and P_0 is the reference power. This minimises the variance of
  the power spectrum estimator.
- **Inverse-variance weights** with an optional non-linear velocity dispersion
  floor sigma_* (Howlett 2019): w = 1 / (sigma^2 + sigma_*^2). For peculiar
  velocities, the floor absorbs small-scale non-linear motions that the
  linear theory covariance model cannot capture.
- **Completeness and systematic weights**: per-survey columns
  (w_cp, w_noz, w_systot) that correct for fiber collisions, redshift
  failures, and imaging systematics.
- **Combination strategies**: when an object is observed by multiple surveys,
  the measurements must be combined. `best_only` keeps the lowest-variance
  measurement (standard practice for DESI/eBOSS). `ivar_average` computes
  the BLUE inverse-variance weighted mean (valid if errors are independent).
  `hyperparameter` (Lahav 2000) applies per-survey multipliers alpha_s that
  encode subjective trust without corrupting individual error bars.

---

## Architecture

```
oneuniverse/
├── data/                          # Data layer
│   ├── schema.py                  # Column definitions (core, spectro, photo, PV, QSO, SNIa)
│   ├── format_spec.py             # OUF format spec (POINT, SIGHTLINE, HEALPIX)
│   ├── _base_loader.py            # BaseSurveyLoader ABC + SurveyConfig dataclass
│   ├── _registry.py               # @register decorator, list_surveys(), get_loader()
│   ├── _config.py                 # ONEUNIVERSE_DATA_ROOT, resolve_survey_path()
│   ├── _io.py                     # read_fits() with fitsio, byte-order fixes
│   ├── converter.py               # Survey -> partitioned Parquet (zstd), manifest.json
│   ├── database.py                # OneuniverseDatabase: folder scan, dynamic loaders
│   ├── config_loader.py           # INI-file driven database construction
│   ├── oneuid.py                  # ONEUID index, OneuidQuery (selectors x hydration)
│   ├── selection.py               # Cone, Shell, SkyPatch composable selectors
│   └── surveys/                   # One sub-package per survey
│       ├── spectroscopic/
│       │   ├── eboss_qso/         # eBOSS DR16Q (Lyke+2020), 920K QSOs
│       │   ├── desi_qso/          # DESI DR1 QSO (DESI 2024), 1.27M QSOs
│       │   ├── sdss_mgs/          # SDSS Main Galaxy Sample (skeleton)
│       │   ├── desi_bgs/          # DESI Bright Galaxy Survey (skeleton)
│       │   └── sixdfgs/           # 6dF Galaxy Survey (skeleton)
│       ├── photometric/
│       │   └── des_dr2/           # DES Year 6 (skeleton)
│       ├── peculiar_velocity/
│       │   ├── cosmicflows4/      # CosmicFlows-4 (Tully+2023) (skeleton)
│       │   └── desi_pv/           # DESI peculiar velocities (skeleton)
│       └── snia/
│           └── pantheonplus/      # Pantheon+ SNe Ia (Scolnic+2022) (skeleton)
│
├── weight/                        # Weighting and cross-match layer
│   ├── base.py                    # Weight ABC, InverseVariance, FKP, QualityMask, Column, Product
│   ├── crossmatch.py              # Ball-tree sky matching + connected components
│   ├── combine.py                 # best_only, ivar_average, hyperparameter, unit_mean
│   └── catalog.py                 # WeightedCatalog facade
│
├── notebooks/                     # Tutorial notebooks
│   ├── 01_open_database.ipynb     # Opening and inspecting a database
│   ├── 02_select_objects.ipynb    # Selectors + tiered ONEUID queries
│   └── 03_eboss_desi_crossmatch.ipynb  # Full cross-match + weighting demo
│
└── test/                          # 108 tests (core, format, database, weight, oneuid)
```

---

## Installation

```bash
pip install .                      # minimal (numpy, pandas, astropy)
pip install ".[all]"               # + fitsio, pyarrow, healpy, matplotlib
pip install -e ".[dev]"            # editable + test dependencies
```

Requires Python >= 3.9.

---

## Quick start

### Build a database from a config file

```ini
# oneuniverse.ini
[database]
root = /data/oneuniverse_database

[dataset eboss_qso]
loader = eboss_qso
raw_path = /data/raw/spectroscopic/eboss/qso
output_subpath = spectroscopic/eboss/qso
qso_only = true

[dataset desi_qso]
loader = desi_qso
raw_path = /data/raw/spectroscopic/desi/dr1/qso
output_subpath = spectroscopic/desi/dr1/qso
qso_only = true
good_zwarn = true
```

```python
from oneuniverse.data import OneuniverseDatabase

db = OneuniverseDatabase.from_config("oneuniverse.ini")
print(db.summary())
# OneuniverseDatabase @ /data/oneuniverse_database
#   2 dataset(s):
#   - spectroscopic_desi_dr1_qso    [spectroscopic  ] point   n_rows= 1,265,294
#   - spectroscopic_eboss_qso       [spectroscopic  ] point   n_rows=   919,374
```

### Cross-match and query

```python
# Build the ONEUID index (sky=2", dz=5e-3)
index = db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=5e-3)
print(f"{index.n_unique:,} unique objects, {index.n_multi:,} multi-survey")
# 1,918,486 unique objects, 234,034 multi-survey

# Tiered queries
q = db.oneuid_query(index=index)
uids = q.from_cone(ra=185, dec=15, radius=2)       # selector: sky cone
df   = q.partial_for(uids, columns=["z_spec_err"])  # hydration: selected columns
```

### Per-survey weighting

```python
from oneuniverse.weight import WeightedCatalog, InverseVarianceWeight, FKPWeight

wc = WeightedCatalog({"eboss": eboss_df, "desi": desi_df})
wc.add_weight("eboss", InverseVarianceWeight("z_spec_err"))
wc.add_weight("desi",  InverseVarianceWeight("z_spec_err"))
match = wc.crossmatch(sky_tol_arcsec=2.0, dz_tol=5e-3)
combined = wc.combine(value_col="z", variance_col="z_var", strategy="best_only")
```

---

## Implemented surveys

| Survey | Type | Objects | Redshift range | Reference |
|--------|------|---------|---------------|-----------|
| eBOSS DR16Q | spectroscopic / QSO | 919,374 | 0 - 7 | Lyke+2020, ApJS 250, 8 |
| DESI DR1 QSO | spectroscopic / QSO | 1,265,294 | 0 - 6 | DESI Collaboration 2024, AJ 168, 58 |

Skeleton loaders (ready for data): SDSS MGS, DESI BGS, 6dFGS, DES DR2,
CosmicFlows-4, DESI PV, Pantheon+.

### Adding a new survey

1. Create `oneuniverse/data/surveys/<type>/<name>/loader.py`
2. Subclass `BaseSurveyLoader`, define a `SurveyConfig` and `_load_raw()`
3. Decorate with `@register`
4. Add a `[dataset <name>]` section to the INI config
5. Run `OneuniverseDatabase.from_config(...)` to convert and scan

---

## Roadmap

### Phase 1 — Data layer (current)

- [x] Standardised schema with column groups
- [x] OUF Parquet format (POINT, SIGHTLINE, HEALPIX geometries)
- [x] Loader framework with @register discovery
- [x] eBOSS DR16Q and DESI DR1 QSO loaders
- [x] INI-config-driven database construction
- [x] ONEUID cross-match (ball-tree + connected components)
- [x] Tiered query API (selectors x hydration levels)
- [x] Weight primitives (FKP, inverse-variance, quality mask, column, product)
- [x] Combination strategies (best_only, ivar_average, hyperparameter)
- [ ] SDSS MGS, DESI BGS, 6dFGS loaders
- [ ] CosmicFlows-4 and DESI PV (peculiar velocity) loaders
- [ ] Pantheon+ SNe Ia loader
- [ ] DES DR2 photometric loader
- [ ] Lyman-alpha forest sightline loader (SIGHTLINE geometry)
- [ ] CMB lensing kappa map loader (HEALPIX geometry)

### Phase 2 — Cross-correlation infrastructure

- [ ] Survey window functions and angular selection functions
- [ ] Angular cross-power spectra C_l between any pair of tracers
- [ ] 3D cross-correlation functions for overlapping spectroscopic surveys
- [ ] Photo-z calibration module (spectroscopic truth table matching)
- [ ] Direct interface to `flip` DataVector and CovMatrix
- [ ] Multi-tracer optimal power spectrum estimator

### Phase 3 — Digital twin

- [ ] Initial conditions parametrisation (Fourier modes, wavelet basis)
- [ ] Forward model interface (N-body / 2LPT / COLA / learned surrogates)
- [ ] Per-survey observation model (bias, selection, noise)
- [ ] Constrained realisation sampler (HMC on initial conditions)
- [ ] Zoom-in re-simulation boundary condition extractor
- [ ] Incremental update: re-constrain local volumes from new data

---

## Connection to `flip`

[`flip`](https://github.com/corentinravoux/flip) (Field Level Inference
Package) implements the statistical inference layer for measuring f*sigma_8
from velocity and density fields (Ravoux et al. 2025, arXiv:2501.16852). The
data pipeline is:

```
Raw survey files
    -> oneuniverse loaders (standardise, quality-cut, write Parquet)
    -> OneuniverseDatabase (discover, scan, expose uniform API)
    -> ONEUID cross-match (link objects across surveys)
    -> weight + combine (per-survey weights, multi-survey combination)
    -> flip DataVector (DirectVel, VelFromHDres, DensVel, ...)
    -> flip CovMatrix (analytical covariance via Hankel transforms)
    -> flip FitMinuit / FitMCMC (maximum likelihood / posterior sampling)
    -> f*sigma_8, b*sigma_8, sigma_v, ...
```

The digital twin extends this chain: the constrained density and velocity
fields become the **prior** for a forward-modelled likelihood, replacing the
Gaussian random field assumption with the actual reconstructed cosmic web.

---

## License

MIT

## Author

Corentin Ravoux
