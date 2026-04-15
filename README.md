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

`oneuniverse` is built around three goals of increasing ambition. Across
all three, the package is a **data, interface and orchestration layer**:
it does not implement cross-correlation estimators, forward models,
samplers, or simulation codes. Those live in existing mature packages
(pypower, NaMaster, `flip`, BORG, FlowPM, pmwd, MUSIC, Nyx, AREPO, RAMSES,
SWIFT), chosen and wired in by the user.

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

`oneuniverse` does **not** implement cross-correlation estimators or power
spectrum codes — that is the job of dedicated packages (pypower, nbodykit,
NaMaster, `flip`). Instead, it prepares the **data products** those codes
need so that cross-survey analyses become straightforward:

- **Cross-matched catalogs with ONEUID**: know which objects are shared
  between surveys, enabling multi-tracer analyses and duplicate removal.
- **Selection functions and angular masks**: n(z) per tracer class and
  HEALPix completeness maps — the inputs every estimator requires but that
  are painful to build from heterogeneous survey files.
- **Random catalogs**: drawn from each tracer's mask + n(z), ready to feed
  into pypower or nbodykit.
- **Weighted, combined catalogs**: per-survey weights (FKP, inverse-variance,
  completeness, quality masks) and combination strategies (`best_only`,
  `ivar_average`, hyperparameter) that preserve the inverse-variance
  interpretation downstream codes rely on.

With these data products in hand, cross-correlation studies become one-liners:

- **Redshift calibration**: compare spectroscopic z from DESI with eBOSS for
  the ~200k quasars observed by both, quantify systematic offsets per pipeline.
- **Multi-tracer power spectra**: feed the weighted catalogs + randoms into
  pypower with per-tracer bias and shot noise.
- **Peculiar velocity + density joint analysis**: `flip` needs both a density
  field and a velocity field; `oneuniverse` provides both in a single
  weighted catalog, with the cross-match handling overlapping footprints.
- **Photo-z training and validation**: ONEUID automates the spectro-photo
  match needed for calibration truth tables.

### 3. Digital twin of the Universe — the microsimulation method

This is the most ambitious goal, and the reason the package is called
_oneuniverse_.

The idea: given enough observations from enough surveys, reconstruct the
**actual three-dimensional matter density field and velocity field of our
Universe** — not a statistical average, not a random realisation from a
cosmological prior, but the specific cosmic web that we inhabit.

`oneuniverse` does **not** implement the forward model, the HMC sampler,
or the hydrodynamic code. BORG, FlowPM, pmwd, MUSIC, Nyx, AREPO, and
SWIFT already exist and are mature. Which simulation code is used — for
both the global reconstruction and the local re-simulations — is a
**user choice**, selected by whoever interfaces `oneuniverse` with their
simulation stack through the package's input/output adapters.

The distinctive scientific contribution sits on top of this: the
**microsimulation method** — a two-tier forward-modelling strategy for
field-level inference.

**Tier A — global reconstruction.** A single large-scale run (e.g.
BORG/Manticore-style, roughly annually or on major DR arrival) fixes the
**large-scale gravitational potential** and long-wavelength tidal /
density field. This run is expensive, but infrequent.

**Tier B — microsimulations.** For any local region of scientific
interest, a fast, high-resolution local simulation is run **conditioned
on the global field as a tidal + density boundary condition**. Local
initial conditions ("best seeds") are inferred from the local survey
data (DESI BGS, PV, local SNe, Lyα sightlines in the volume) through a
local field-level inference loop. Because the local volume is small and
the long-wavelength modes are fixed, each forward evaluation is orders
of magnitude cheaper than a global run — enabling iterative FLI at
resolutions that global reconstructions cannot afford.

The role of `oneuniverse` in this picture:

1. **Provide the observation-side inputs** the reconstruction codes need
   — cross-matched catalogs, selection functions, angular masks, survey
   geometry, per-tracer bias metadata — in a standardised format.
2. **Detect change**: monitor the survey database for new or modified
   data (new DR, reprocessing, new survey ingested) and identify which
   sub-volumes have stale inputs.
3. **Dispatch microsimulations**: extract the appropriate boundary
   conditions from the global reconstruction, assemble the local input
   catalog, and hand the job to whichever simulation code the user has
   configured. The simulation format is not imposed by `oneuniverse` —
   the user (or the person writing an interface for their group's
   preferred code) chooses.
4. **Store results** back into a **digital twin database** — OUF-style
   partitioned storage with 3D field data and metadata linking each
   volume to the survey data, global reconstruction, and simulation code
   that produced it.

The package is therefore a **data and orchestration layer** sitting
above the forward-modelling stack, not a reimplementation of it. The
scientific novelty is the microsimulation methodology; the engineering
contribution is the data infrastructure that makes it tractable.

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

#### External reconstruction tooling (user-selectable)

The reconstruction work itself — global or local — is done by external
codes chosen by the user. The mature global-reconstruction stack centres
on the **BORG** algorithm (Jasche & Wandelt 2013) and its successors; the
latest result, **Manticore** (McAlpine et al. 2025, arXiv:2505.10682),
ingests ~69,000 galaxies from the 2M++ catalogue and produces 50
posterior realisations of the local cosmic web out to 350 Mpc at ~4 Mpc
resolution. The **Velocity Field Olympics** (Stiskalek et al. 2026,
arXiv:2502.00121) benchmarked all major reconstruction methods and found
that non-linear Bayesian forward models consistently outperform
linear/Wiener-filter alternatives.

For microsimulations (the local Tier-B runs), the user chooses the
simulation code appropriate for their science: differentiable PM codes
(**FlowPM**, **pmwd**) are well-suited for local FLI loops; AMR / SPH
codes (**Nyx**, **AREPO**, **RAMSES**) for hydrodynamic local physics;
**SWIFT** for cosmological DM-only sub-volumes. `oneuniverse` provides
the input/output adapters; the choice of code is made by whoever writes
the interface for their group's workflow.

`oneuniverse` interfaces with this external stack at two points — inputs
and outputs — without reimplementing it:

**What `oneuniverse` feeds in:**

- Cross-matched, weighted catalogs (galaxies, PV, SNe, QSOs) with ONEUID
- Per-tracer selection functions n(z, ra, dec) and angular masks
- Random catalogs matching each tracer's mask + n(z)
- Fiducial cosmology and comoving-distance grids
- Tracer classification labels for bias modelling
- For microsimulations: global-field boundary conditions (long-wavelength
  tidal + density field) extracted from the current Tier-A reconstruction

**What the external stack does** (not implemented here):

- Prior over initial conditions (Gaussian random field, P(k,z) from LCDM)
- Forward model (2LPT, COLA, PM, neural surrogates)
- Observation model (bias, selection, noise) applied to mock observables
- Likelihood + HMC sampling of initial conditions, or SELFI-style SBI

**What `oneuniverse` stores back:**

- 3D posterior realisations (delta(x), v(x)) in a new OUF variant for
  field data, with metadata linking each volume to its input survey
  version.
- White-noise realisations needed by MUSIC/monofonIC for downstream zooms.
- Provenance: which survey DR, which ONEUID index version, which fiducial
  cosmology, which external code + version produced the reconstruction,
  and — for Tier-B outputs — which Tier-A reconstruction provided the
  boundary conditions.

#### What data types matter most?

Research across five domains converges on a clear hierarchy:

1. **Galaxy redshift surveys** — the backbone. Galaxy positions constrain the
   density field on scales > 4 Mpc. This is the minimum viable input (2M++
   with ~69K galaxies suffices for Manticore).
2. **Peculiar velocities** — add critical dynamical information. Break the
   density-velocity degeneracy. Hamlet-PM (arXiv:2602.03699) shows PV-only
   reconstructions from CosmicFlows-4 are already competitive.
3. **CMB lensing** — projected mass map, breaks galaxy bias degeneracy.
   Multi-probe FLI combining CMB lensing + galaxy density is an active
   frontier but not yet in production constrained realisations.
4. **Lyman-alpha forest** — probes high-z (z = 2-5) density. cosmosTNG
   (arXiv:2409.19047) demonstrated constrained Lya simulations for specific
   sky patches using TARDIS + CLAMATO tomography data.

#### Change-driven microsimulation: the orchestration layer

The engineering contribution of `oneuniverse` is the **orchestration**:
deciding what needs to be re-simulated when the database changes, and
maintaining a consistent digital twin as new survey data arrives — while
leaving all actual simulation work to user-selected external codes.

**The change-detection loop:**

1. **Inventory** — for every sub-volume of the digital twin database,
   record which surveys, which DRs, which ONEUID index version, which
   fiducial cosmology, and which Tier-A global reconstruction constrained
   it.
2. **Diff** — on database update (new DR, reprocessing, new survey, new
   ONEUID run), compute which sub-volumes have stale inputs. A new DESI
   tile in one patch of sky should not invalidate microsimulations on
   the opposite hemisphere.
3. **Dispatch microsimulation** — extract the long-wavelength tidal +
   density boundary conditions for the affected sub-volume from the
   current Tier-A reconstruction, assemble the relevant local survey
   data, and hand the job to whichever simulation code the user has
   configured through the interface. The microsimulation format and
   code are user-selectable; `oneuniverse` is agnostic.
4. **Ingest** — write the returned 3D fields back into the digital twin
   database with updated provenance, so the next diff round is correct.

Tier A (the global reconstruction itself) is typically refreshed on a
much slower cadence (roughly annually, or on major DR arrival). Tier B
microsimulations are where the frequent, change-driven dispatch happens.

**External tooling the dispatcher can call** (non-exhaustive; chosen by
the user):

- Global (Tier A) constrained ICs: BORG, Manticore (arXiv:2505.10682)
- Multi-scale IC generation: MUSIC/monofonIC with nested grids; output
  plugins exist for Nyx, AREPO, RAMSES, Gadget, Enzo
- Microsimulation (Tier B) candidates: FlowPM (arXiv:2010.11847) and
  pmwd (arXiv:2211.09958) for differentiable local FLI; Nyx for
  Lyα-optimised hydro; AREPO for galaxy formation; RAMSES for general
  AMR; SWIFT for cosmological DM-only
- Targeted field modifications: GenetIC (arXiv:2006.01841)

Proven end-to-end examples of observation → constrained ICs → high-res
simulation pipelines include HESTIA (arXiv:2008.04926, AREPO into CLUES
ICs) and SIBELIUS-DARK (arXiv:2202.04099, SWIFT into BORG ICs). Neither
is inferential at the local level, and neither automates change-driven
dispatch — those are the gaps the microsimulation method + `oneuniverse`
orchestration layer address.

Resolution targets that determine which external code is appropriate for
a given microsimulation: LSS/PV ~1-2 Mpc/h (DM-only), Lyα forest
20-50 kpc/h (hydro), galaxy formation < 1 kpc (full sub-grid).

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

The ONEUID cross-match (`oneuid.py`) is physically motivated, and the current
design choices are well-aligned with the literature (Budavari & Szalay 2008,
arXiv:0707.1611; Marrese et al. 2019, A&A 621, A144):

- **Sky tolerance** (default 2 arcsec): accounts for astrometric differences
  between surveys. SDSS and DESI have ~0.1 arcsec precision, so 2 arcsec is
  conservative and handles extended sources. The optimal match radius is
  3-5× sqrt(sigma_1^2 + sigma_2^2); for optical-optical this gives 0.5-1",
  for optical-WISE 2-3", for X-ray-optical 6-15" (Salvato et al. 2018,
  arXiv:1705.10711). Future versions should support **survey-pair-specific
  tolerances**.
- **Redshift tolerance** (default dz = 5e-3): accounts for pipeline
  differences, catastrophic redshift failures, and the physical velocity
  dispersion of cluster members (~1000 km/s ~ dz ~ 0.003). Note that DESI QSO
  redshifts carry z-dependent systematic offsets of ~200 km/s (Bault et al.
  2025, arXiv:2402.18009), motivating a looser dz_tol ~ 5e-3 for QSOs vs
  ~1e-3 for galaxies.
- **Transitivity**: the connected-components algorithm makes the match
  transitive — if A~B in survey 1 and B~C in survey 2, then A, B, C share
  one ONEUID. This is the graph-theoretic approach recommended by Budavari &
  Szalay (2008). A planned safeguard is to flag ONEUID groups where not all
  pairwise separations satisfy the tolerance (potential blends or close pairs),
  using a maximum group diameter check.
- **Cross-survey-only linking**: same-survey pairs are excluded, avoiding
  spurious merging of close pairs within a single catalog.
- **Match quality metadata** (planned): store pairwise angular separation and
  delta-z in the ONEUID index, enabling downstream quality filtering.

For surveys with very different positional uncertainties (eROSITA, WISE),
the long-term path is **NWAY** probabilistic matching (Salvato et al. 2018),
which provides Bayesian multi-catalog matching with magnitude priors.

### Weighting and combination

The weight system (`weight/`) mirrors the statistical framework used in
large-scale structure analyses:

- **FKP weights** (Feldman, Kaiser & Peacock 1994):
  w_FKP(z) = 1 / (1 + n_bar(z) P_0), where n_bar(z) is the survey number
  density and P_0 is the reference power. This minimises the variance of
  the power spectrum estimator. In the multi-tracer generalisation
  (Abramo & Leonard 2013, arXiv:1302.5444), the optimal weights depend on
  the bias ratio b_A/b_B and the per-tracer number densities — splitting a
  sample into sub-populations with different biases always improves
  constraints on bias-sensitive parameters like f_NL and f*sigma_8.
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

**What is still needed for multi-tracer science** (Seljak 2009,
arXiv:0807.1770; pypower/nbodykit implementations):

- Per-tracer n(z) selection functions (the single most critical gap)
- Tracer classification labels (LRG, ELG, QSO, BGS) beyond `survey_id`
- Angular masks / survey geometry (HEALPix completeness maps)
- Random catalog generation matching each tracer's mask + n(z)
- Per-class linear bias estimates b(z) for optimal multi-tracer weighting

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

Priority labels (P0-P3) are derived from a gap analysis cross-validated
across five independent research domains: constrained realisations,
multi-tracer statistics, cross-matching, field-level inference, and zoom-in
simulations. P0 gaps were flagged by >= 3 domains; P1 by 2; P2 by 1.

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

### Phase 2 — Data products for cross-correlation (no estimators here)

`oneuniverse` produces the inputs that external estimators (pypower,
nbodykit, NaMaster, TreeCorr, `flip`) need. Estimator implementations
stay in those packages.

*P0 critical gaps (needed by multi-tracer, FLI, and constrained realisations):*

- [ ] **Per-tracer n(z) selection functions** — smooth radial selection for
  each tracer class. The single most critical missing piece.
- [ ] **Angular masks / survey geometry** — HEALPix or MOC per-pixel
  completeness, exportable to NaMaster/pypower mask formats.
- [ ] **Random catalog generation** — draw randoms matching each tracer's
  mask + n(z); writeable in pypower/nbodykit input format.

*P1 gaps:*

- [ ] Tracer classification labels (LRG/ELG/QSO/BGS) within each survey
- [ ] Survey-pair-specific cross-match tolerances (optical-optical vs
  optical-IR vs X-ray-optical; Salvato et al. 2018)
- [ ] Match quality metadata in ONEUID index (pairwise sep, delta-z)
- [ ] Thin export adapters to `flip` DataVector, pypower catalog, SACC

*P2 gaps:*

- [ ] Per-class linear bias estimates b(z) metadata (informational, not
  used to fit)
- [ ] Fiber/targeting metadata for forward-modelling incompleteness
- [ ] Comoving distance grids under fiducial cosmology
- [ ] Transitivity safeguards (max ONEUID group diameter check)
- [ ] Cosmological parameter provenance (fiducial cosmology, swappable)
- [ ] Photo-z calibration data products (spectroscopic truth tables)

### Phase 3 — Digital twin interface and microsimulation orchestration

`oneuniverse` orchestrates, stores, and exposes interfaces; the external
simulation code (global and microsimulation) is chosen by the user.

*Storage and provenance:*

- [ ] Field-data OUF variant for 3D delta(x), v(x) grids (Tier A and B)
- [ ] White-noise realisation storage (MUSIC/monofonIC input)
- [ ] Reconstruction manifest: survey DR versions, ONEUID index hash,
  fiducial cosmology, external code + version
- [ ] Sub-volume index (spatial partitioning of the digital twin)
- [ ] Tier-B provenance link: microsimulation → Tier-A reconstruction
  that supplied its boundary conditions

*Generic interfaces (code-agnostic, user-configured):*

- [ ] Export adapter spec: catalogs + selection + mask → global-reconstruction input
- [ ] Export adapter spec: sub-volume boundary conditions + local catalog
  → microsimulation input
- [ ] Import adapter spec: external simulation output → field-data OUF
- [ ] Adapter registry: users declare which codes they have wired in

*Microsimulation method (the distinctive scientific contribution):*

- [ ] Boundary-condition extractor: long-wavelength tidal + density field
  from a Tier-A reconstruction on an arbitrary sub-volume
- [ ] Buffer-zone computation: account for long-range mode coupling
  between global and local modes
- [ ] Local-catalog assembler: gather all survey data intersecting a
  sub-volume through the ONEUID layer
- [ ] Consistency checks between Tier-A and Tier-B posteriors at sub-volume
  boundaries

*Change-driven orchestration:*

- [ ] Diff engine: detect which sub-volumes have stale inputs on DB update
- [ ] Dispatcher: submit affected sub-volumes to the user-configured
  microsimulation code
- [ ] Ingest: write returned fields back with updated provenance

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

### flip as a special case of field-level inference

`flip` performs **Gaussian field-level inference**: a multivariate Gaussian
likelihood over the velocity and/or density field with a covariance matrix
C(fs8, bs8, sigv) assembled from Hankel-transformed power spectra. This is
formally equivalent to FLI with a linear forward model and Gaussian field
assumption. The Gaussian approximation is well-justified for velocities —
the velocity field is a spatial integral of delta, which averages out
non-Gaussianity even at low redshift.

Full forward-model FLI (BORG, FlowPM, pmwd) would add: (i) non-Gaussian
information from the velocity divergence field; (ii) joint 3D velocity
field reconstruction for map-level cross-correlations; (iii) self-consistent
non-linear bias for joint density-velocity analyses. Nguyen et al.
(arXiv:2407.14289) estimate factors of 2-3 improvement on f*sigma_8 for
DESI-sized PV catalogs. Stadler et al. (arXiv:2509.09673) find ~20%
tighter constraints on {A_s, omega_cdm, H_0} from FLI vs. P(k)+bispectrum.

These full-FLI codes live outside `oneuniverse` — the package's role is to
supply them with the cross-matched, weighted catalogs, selection functions,
and masks they need, and to store their posterior realisations back into
the digital twin database.

---

## Scientific context and key references

The design of `oneuniverse` draws on established methodology and recent
advances across several fields. A detailed research synthesis is available in
[research/digital_twin_research.md](research/digital_twin_research.md).

| Domain | Key references |
|--------|---------------|
| Constrained realisations | BORG (Jasche & Wandelt 2013, arXiv:1306.1821), Manticore (McAlpine et al. 2025, arXiv:2505.10682), Velocity Field Olympics (Stiskalek et al. 2026, arXiv:2502.00121), CSiBORG (arXiv:2203.14724), Hamlet-PM (arXiv:2602.03699) |
| Multi-tracer statistics | Seljak 2009 (arXiv:0807.1770), Abramo & Leonard 2013 (arXiv:1302.5444), DESI combined-tracer BAO (arXiv:2508.05467) |
| Cross-matching | Budavari & Szalay 2008 (arXiv:0707.1611), NWAY (Salvato et al. 2018, arXiv:1705.10711) |
| Field-level inference | flip (Ravoux et al. 2025, arXiv:2501.16852), FlowPM (arXiv:2010.11847), pmwd (arXiv:2211.09958), SELFI (Leclercq et al. 2019) |
| Zoom-in simulations | MUSIC (arXiv:1103.6031), monofonIC (arXiv:2008.09124), HESTIA (arXiv:2008.04926), SIBELIUS (arXiv:2202.04099), GenetIC (arXiv:2006.01841), Nyx (arXiv:1301.4498) |

---

## License

MIT

## Author

Corentin Ravoux
