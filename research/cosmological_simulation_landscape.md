# Cosmological Simulation Landscape — Pillar 3 Reference

**Date:** 2026-06-01
**Purpose:** Catalogue every major class of cosmological simulation
the `oneuniverse` Pillar 3 (digital twin) layer must ingest +
orchestrate, and propose a storage architecture. Companion to
[`survey_landscape_review.md`](survey_landscape_review.md) (Pillar 1).

**Scope.** Three independent surveys synthesised here:
- Section 2 — simulation **codes** (Gadget, AREPO, GIZMO, RAMSES,
  Enzo, FLASH, Nyx, HACC, ABACUS, Cactus/ET, GRChombo, gevolution,
  FastPM, JaxPM, pmwd, BORG, ColDICE, GAMER-2, …).
- Section 3 — public **suites / projects** (AbacusSummit,
  MillenniumTNG, IllustrisTNG, EAGLE, BAHAMAS, Magneticum, FIRE,
  SIMBA, Horizon-AGN, FLAMINGO, Quijote, UNIT, Outer Rim, LastJourney,
  Aemulus, GLAM, PINOCCHIO, Buzzard, Flagship-2, Uchuu, CosmoDC2,
  CLUES, HESTIA, SIBELIUS-DARK, MANTICORE, ELUCID, BORG-DESI,
  CAMELS, …).
- Section 4 — on-disk **data representations** (particle snapshots
  DM-only + hydro; AMR mesh fields; halo catalogs; merger trees;
  lightcones; phase-space sheets; full-GR 4-tensors; IC posterior
  chains; differentiable forward-model checkpoints; smooth grid
  fields).

Section 5 distils all three into Pillar-3 architectural takeaways.
Section 6 is the proposed **OUF-Sim format** (manifest-of-manifests
model) that this doc recommends.

---

## 1. Pillar 3 framing

Pillar 3 ≡ digital twin **substrate**. Per the project owner
(2026-05-29 + 2026-06-01):

- Pillar 3 stores + orchestrates simulations in **any form**;
  it does **not** run simulations. Mini-simulation runs are
  deferred indefinitely.
- **Partial access is the load-bearing constraint.** Simulations
  are TB–PB per snapshot; the whole sim never fits in memory.
  Every public reader must take a mandatory selector
  (spatial / temporal / structural) and return either a lazy view
  or a chunked iterator.
- **Minimal cross-pillar coupling.** Pillar 3 is a standalone
  subpackage / package. It does **not** import from
  `oneuniverse.data` or `oneuniverse.combine`. Communication with
  Pillar 1 / Pillar 2 is through file artefacts (OUF parquet,
  MeasurementSet contracts), not Python imports.
- **MPI-collective + GPU-direct reads** are first-class
  optimisation targets, not afterthoughts.
- No mocks in Pillar 1. All mock-catalog ingestion (Buzzard,
  Flagship-2 galaxy catalog, CosmoDC2, AbacusSummit_HOD,
  UNIT-EZmock) is Pillar 3 territory.
- `PARTICLE` / mock geometries belong here, not in Pillar 1.

Pillar 3 deliverables — what we will build:
1. **OUF-Sim format** (Section 6): a manifest-of-manifests that wraps
   native simulation files + sidecar partial-access indexes +
   provenance.
2. **Selector API** with mandatory spatial / temporal / structural
   filters; `iter_*` / lazy-view returns; opt-in MPI + GPU paths.
3. **Adapters** to existing readers (`yt`, `halotools`, `ytree`,
   `abacusutils`, `illustris_python`, `swiftsimio`, `nbodykit`,
   `genericio`, `bigfile`, …) — never re-implement what they do.
4. **Cross-snapshot + cross-representation queries** (halo →
   particles, halo → progenitor chain, snapshot → lightcone shell).
5. **Suite-level orchestration** (AbacusSummit grid of cosmologies,
   CAMELS Latin-Hypercube, BORG posterior chain).

What Pillar 3 explicitly is **not**:
- Not a simulation code. We don't run anything.
- Not a re-encoding of native files. We index + manifest, do not
  duplicate the petabytes.
- Not a cosmology engine. Cosmological theory + comoving conversion
  remain Pillar 2.
- Not a sampler / forward modeller. BORG-like HMC, JaxPM gradient
  flows, IC posterior inference are downstream consumers, not
  Pillar 3.
- Not coupled to Pillar 1. Zero imports from `oneuniverse.data` or
  `oneuniverse.combine`.

---

## 2. Simulation codes — landscape

Eleven canonical code classes drive every public cosmological
product. Pillar 3 must accommodate the on-disk output of each.

### 2.1 Pure N-body — tree / TreePM / FMM / P3M

| Code | Method | Status | Native format | Lead refs |
|---|---|---|---|---|
| Gadget-2 | TreePM | frozen | Gadget binary fmt-1/-2 | Springel 2005 |
| Gadget-3 (collab) | TreePM | frozen | Gadget HDF5 | — |
| Gadget-4 | FMM + PM | active | Gadget HDF5 | Springel+ 2021 |
| PKDGRAV3 | FMM | active | tipsy / HDF5 / fio | Potter+ 2017 |
| ABACUS | P3M (near-field exact + global multipole) | active | ASDF + pack9 / pack14 | Garrison+ 2019, Maksimova+ 2021 |
| HACC | RCB-tree + PM | active | GenericIO | Habib+ 2016 |
| GreeM / μ-PPM | TreePM | active | Gadget binary | Ishiyama+ (Uchuu) |
| CUBE / CUBEP3M | classical P3M | frozen | Fortran binary | Yu+ |

Production scales: Euclid-Flagship PKDGRAV3 ≈ 4 trillion particles
(16000³) at 3.78 Gpc/h; LastJourney HACC 10752³ at 5.025 Gpc/h;
AbacusSummit base 6912³ × 97 cosmologies at 2 Gpc/h.

### 2.2 N-body + SPH

Gadget-2/-3/-4 SPH, GIZMO (MFM / MFV / meshless-finite-mass),
GASOLINE / GASOLINE2, ChaNGa, SWIFT (SPHENIX + GADGET2-SPH +
ANARCHY), PHANTOM. All except PHANTOM target cosmological volumes.
On-disk: Gadget HDF5 layout dominates (`/PartType{0..5}`); SWIFT adds
`/Cosmology`, `/HydroScheme`, `/SubgridScheme` groups but is
structurally compatible.

### 2.3 Moving-mesh hydro

AREPO. Used by IllustrisTNG (TNG50/100/300/Cluster), Illustris-1,
Auriga, MillenniumTNG-Hydro, FABLE. Writes per-cell data as
"particles" in Gadget HDF5; cell-specific fields are `Density`,
`Volume`, `CenterOfMass`, `InternalEnergy`, `MagneticField`,
`GFM_Metallicity`, `GFM_Metals[9]`, `GFM_StellarPhotometrics`.
Critical: `Coordinates` is the Voronoi generator point ≠ cell
barycenter.

### 2.4 AMR hydro

Enzo / Enzo-E, RAMSES, FLASH / FLASH-X, Athena++, AthenaPK, Nyx
(AMReX), CASTRO, CHARM, ART. Five family-distinct on-disk formats:

- **Enzo HDF5**: one file per AMR grid (10⁵ files per snapshot
  → Lustre metadata pain).
- **FLASH HDF5**: one file per checkpoint; mesh-block tree.
- **RAMSES Fortran-binary multi-file**: `info_NNNNN.txt` +
  `amr_*.outYYYYY` + `hydro_*.outYYYYY` + `part_*.outYYYYY` +
  `grav_*.outYYYYY` + `rt_*.outYYYYY` (RAMSES-RT). Often O(10⁴)
  files per snapshot.
- **AMReX plotfiles (Nyx, CASTRO, ERF)**: multi-level directory
  with per-FAB binary; Fortran column-major.
- **Athena++ HDF5** (`*.athdf`) + XDMF XML.

### 2.5 Particle-mesh / fast / forward-modelling

FastPM (BigFile + Gadget HDF5), pmwd (orbax checkpoints), JaxPM
(orbax), FlowPM (TF checkpoints), BORG-PM / BORG-LPT / BORG-COLA /
BORG-DESI (HDF5 per chain step), COLA / L-PICOLA, MP-Gadget, GLASS
(HEALPix FITS shells), PINOCCHIO (ASCII / FITS), EZmocks / GLAM-PM.

Distinct from Sections 2.1–2.4: short integration, low cost,
designed for **mock production at scale** or **differentiable
forward modelling**. BORG-PM additionally outputs an MCMC chain of
IC fields (constrained-realisation; see Section 2.8).

### 2.6 Full GR / relativistic

Einstein Toolkit (Cactus + Carpet + Kranc), GRChombo / GRTeclyn,
gevolution (Adamek+ 2016), CosmoGRaPH, GRAMSES (RAMSES fork).
Distinguish:

- **Full BSSN/Z4c on AMR** (ET, GRChombo) — 10 metric + 10 K_ij +
  α + β^i on Carpet / Chombo AMR.
- **Weak-field / post-Friedmannian PM** (gevolution): 4 metric
  perturbations (φ, χ, B_i, h_ij) on Cartesian grid + particles
  with 4-velocity.

Gauge conventions vary widely (synchronous, conformal Newtonian,
longitudinal, Poisson) — Pillar 3 must record gauge explicitly.

### 2.7 Hybrid / radiative-transfer

RAMSES-RT (SPHINX, OBELISK, AURORA), AREPO-RT (THESAN), Enzo + Moray
(Renaissance), CROC (ART + RT), GIZMO-MHD-RT, RAMSES-CR. Extra
per-cell fields: `Nph_*` (photon density), `Fp_*` (photon flux,
3-vector per band), species abundances (`xHI`, `xHII`, `xHeI..III`,
`xH2`), optionally dust mass.

### 2.8 Constrained / forward-model

BORG family (LPT, PM, COLA, FORWARD, DESI, 2LPT), HESTIA (CLUES +
Auriga-physics), CLUES, SIBELIUS-DARK, MANTICORE, HAMLET, ELUCID.
Output: parent code's snapshot format **plus** an MCMC chain
of IC samples + Wiener-filtered observed field + bias parameters.
A constrained realisation is *one sample* — Pillar 3 must store the
chain / ensemble, not a single snapshot.

### 2.9 Phase-space / Vlasov / fuzzy DM

ColDICE (tetrahedral phase-sheet refinement; Sousbie+Colombi 2016),
GAMER-2 (Schrödinger-Poisson AMR, fuzzy DM), HypH, PyUltraLight,
SCALAR / SCALARX, CHPLULTRA. Output: simplex-mesh of phase-space
vertices (ColDICE), or complex wavefunction ψ on AMR (GAMER-2).
Neither reducible to a particle list.

### 2.10 Emulators / surrogates of full sims

CAMELS / CAMELS-LH / CAMELS-Multifield-Dataset (CMD), AbacusSummit
emulators, Quijote (44k sims for ML), Aemulus α/ν, MillenniumTNG
emulator, EuclidEmulator2, DarkEmulator, BACCO, FrankenEmu. Suites
of *full simulations* spanning a parameter grid; downstream a neural
network or GP emulates P(k), HMF, ξ, etc. Pillar 3 ingests these
suites as **ensemble bundles**, not individual snapshots.

### 2.11 Differentiable / inverse-problem

pmwd, JaxPM, FlowPM, JAX-Cosmo, Diffrax-cosmo, DESI-FORWARD. JAX /
TF state objects + orbax checkpoints. No canonical disk format; this
is a Pillar-3 gap to close (we'll define a sub-spec).

---

## 3. Public suites — landscape

Tabulated by primary scientific use. Bold = "Pillar 3 ingest must
support to be useful to flagship surveys."

### 3.1 Particle / N-body / hydro flagships

| Suite | Box | Method | Code | Cosmologies | Format | Volume |
|---|---|---|---|---|---|---|
| **AbacusSummit** | 2 Gpc/h, 6912³ | TreePM-P3M | ABACUS | 97 | ASDF + parquet | ~2 PB |
| **MillenniumTNG** | 500–3000 Mpc/h | TreePM (DM-only) + AREPO (hydro) | Gadget-4 + AREPO | 1 fiducial | Gadget HDF5 | ~3 PB |
| **IllustrisTNG** (TNG50/100/300) | 35–303 Mpc | AREPO MHD | AREPO | 1 (Planck15) | Gadget HDF5 | ~750 TB |
| Illustris-1 | 75 Mpc | AREPO (no MHD) | AREPO | 1 (WMAP-9) | Gadget HDF5 | ~230 TB |
| EAGLE | 25–100 Mpc | Gadget-3 SPH | Gadget-3 | 1 (Planck13) | Gadget HDF5 + SQL | ~70 TB |
| BAHAMAS | 400 Mpc/h | Gadget-3 SPH | Gadget-3 | WMAP-9 + Mν grid | Gadget HDF5 + FITS | ~90 TB |
| Magneticum Pathfinder | 128–2688 Mpc/h | Gadget-3 hydro | Gadget-3 | WMAP-7 | Gadget bin + HDF5 + FITS | ~500 TB |
| FIRE / FIRE-2 / FIRE-3 | MW-mass zooms | GIZMO meshless | GIZMO | Planck13/15 | Gadget HDF5 | ~150 TB |
| SIMBA | 100 Mpc/h | GIZMO MFM | GIZMO | Planck15 | Gadget HDF5 | ~60 TB |
| OBELISK / NewHorizon / Horizon-AGN | zooms + 100 Mpc/h | RAMSES AMR | RAMSES | WMAP-7 / Planck13 | RAMSES native + HDF5 | ~400 TB |
| **FLAMINGO** | 1–5.6 Gpc | SWIFT SPH + DM + ν | SWIFT | Planck18 + Σmν + DES-Y3 + AGN± | SWIFT HDF5 + SOAP + FITS HEALPix | ~5 PB |

### 3.2 N-body-only flagships

| Suite | Box | Particles | Code | Cosmologies | Format |
|---|---|---|---|---|---|
| **Quijote** | 1 Gpc/h | 512³ + 1024³ | Gadget | 44k (LHC + Σmν + fNL) | Gadget bin + HDF5 + npy |
| UNIT | 1 Gpc/h | 4096³ | Gadget-3 | 1 paired-fixed | Gadget binary |
| AbacusCosmos | 1.1 Gpc/h | 1440³ | ABACUS | 41 (LHC) | ASDF |
| **Outer Rim / Last Journey / Q-Continuum** | 3–5 Gpc/h | 8192³–10752³ | HACC | WMAP7 / Planck18 | GenericIO |
| Aemulus α / ν | 1.05 Gpc/h | 1400³ | Gadget-2 | 75 wCDM + Σmν | Gadget bin + HDF5 |
| GLAM | 1 Gpc/h | 2000³ PM | GLAM-PM | Planck13/15 | custom + HDF5 |
| PINOCCHIO mocks | up to 3.8 Gpc/h | LPT halos only | PINOCCHIO | configurable | FITS + ASCII |

### 3.3 Lightcone / mock galaxy catalogs

| Suite | Survey target | Code | Format |
|---|---|---|---|
| Buzzard | DES Y1/Y3 | L-Gadget2 + ADDGALS | FITS per HEALPix tile |
| **Flagship-2** | Euclid | PKDGRAV3 | HDF5 + parquet per tile |
| Uchuu | DESI/Euclid/LSST/Subaru-PFS | GreeM | Gadget HDF5 + parquet |
| **CosmoDC2 / Skysim** | LSST DESC | Outer Rim + Galacticus + GalSampler | parquet + HDF5 per tile |
| CMASS / eBOSS / DESI mock cones | spec surveys | UNIT-EZmock + GLAM + AbacusSummit-CompaSO | FITS + parquet |

### 3.4 Constrained-realisation / local universe

| Suite | Volume | Method | Format |
|---|---|---|---|
| CLUES | 64–160 Mpc/h zoom | Gadget + AHF | Gadget + HDF5 |
| HESTIA | Local Group zoom in 100 Mpc/h | AREPO Auriga-physics | Gadget HDF5 |
| SIBELIUS-DARK | 200 Mpc/h zoom | SWIFT (BORG IC) | SWIFT HDF5 + SOAP |
| **MANTICORE** | 1 Gpc/h | SWIFT (BORG + 2M++ + Cosmicflows IC) | SWIFT HDF5 + SOAP + FITS |
| ELUCID | 500 Mpc/h | Gadget-2 (HMC reconstruction) | Gadget |
| **BORG-DESI / BORG-PM family** | survey-dependent | HMC IC + LPT/COLA/PM forward | HDF5 per MCMC sample |

### 3.5 ML / surrogate

| Suite | Scope | Format |
|---|---|---|
| **CAMELS / CAMELS-LH** | ~7000 sims (TNG, SIMBA, ASTRID, magneticum variants); LHC over (Ωm, σ8, A_SN1, A_SN2, A_AGN1, A_AGN2) | HDF5 + parquet + npy |
| CAMELS-Multifield-Dataset (CMD) | 2D images for ML training | NumPy |
| CARPool | variance-reduction correlations | HDF5 + NumPy |

### 3.6 Reionisation / hydrogen / Lyα

THESAN (AREPO-RT, 95.5 Mpc, full RT to z>5), SPHINX (RAMSES-RT,
~20 Mpc/h zoom), CROC (ART+RT), EAGLE-XL / FLAMINGO Lyα. All carry
extra per-cell photon-density + ionisation-fraction arrays.

### 3.7 Specialty / phase-space / non-CDM

Quijote-PNG (fNL extension), PSdense, Schive ψDM, CoCo / CoCo-Cold,
EuclidEmulator2, Mira-Titan (HACC, 111 wCDM+ν cosmologies, ~200 TB).

---

## 4. Data representations — exhaustive list

Eleven primitive structures, each with at least one production
example. Pillar 3 must accommodate (wrap, not necessarily re-encode)
all of them.

### 4.1 Particle table — flat columnar

`N × {ra, dec, position[3], velocity[3], mass, ID, optional auxiliary}`.
Backings: Gadget HDF5 (universal), Gadget-1/-2 binary (legacy),
ASDF/pack9 (Abacus), GenericIO (HACC), TIPSY / NCHILADA, BigFile
(FastPM), Parquet (modern ML), NumPy memory-map.

### 4.2 Particle table — hydro auxiliary

As 4.1 plus per-gas-particle `Density, InternalEnergy, ElectronAbundance,
NeutralHydrogenAbundance, Metallicity[N_elem], SFR, MagneticField,
SmoothingLength, …`; per-star `InitialMass, StellarFormationTime,
Metallicity`; per-BH `BH_Mass, BH_Mdot`. Backings: same as 4.1.

### 4.3 AMR hierarchical mesh — block / oct tree

Per refinement level, a set of patches/octs carrying float arrays
(density, velocity, pressure, B-field, metallicity, photon density,
…) plus parent/child/sibling pointers + refinement criterion +
ghost-zone convention. Backings: Enzo HDF5, FLASH HDF5, RAMSES
Fortran multi-file, AthenaPK HDF5, AMReX plotfiles (Nyx), GAMER
HDF5, Chombo HDF5.

### 4.4 Regular Cartesian grid (smooth field)

`(Nx, Ny, Nz, N_components)` float arrays — PM density, FastPM δ,
JaxPM checkpoint, lensing-convergence cubes. Backings: NumPy `.npy`,
HDF5 chunked, Zarr (cloud-friendly, growing adoption), FITS cubes
(with WCS), VTK/XDMF (visualisation pipelines).

### 4.5 Halo catalog

Per-halo table: `id, parent_id, descendant_id, npart, position[3],
velocity[3], M{vir,200c,500c,200m,fof}, R{vir,200c,500c,200m}, Vmax,
Vrms, Rs, c_NFW, spin_{Peebles,Bullock}, shape{a,b,c}, last_major_merger_z,
acc_rate, start_index, length`. Centre definition varies (most-bound
particle / density peak / centre-of-mass / shrinking-sphere). Mass
definition varies (Bryan-Norman Δ_vir(z) vs 200ρ_crit vs 200ρ_mean
vs SO-X). Backings: ROCKSTAR ASCII / binary, ROCKSTAR HDF5, CompaSO
ASDF, Subfind HDF5, Subfind-HBT, HBT+ HDF5, VELOCIraptor / STF HDF5,
AHF ASCII, Consistent Trees ASCII, BGC2.

### 4.6 Merger-tree graph

Directed acyclic graph: per node `halo_id, descendant_id,
main_progenitor_id, next_progenitor_in_branch_id, first/last_progenitor_id,
snap_num`. Distinguish *forests* (sets of trees linked by halo
exchanges) from *trees* (single root). Depth-first ordering on disk
enables range-scan branch retrieval. Backings: Consistent Trees
ASCII, SubLink HDF5, HBT+ HDF5, LHaloTree binary, TreeBuilder
HDF5, TreeFrog HDF5.

### 4.7 Lightcone — HEALPix-tiled shell

Per source redshift z_s, a HEALPix map carrying κ / γ₁ / γ₂ /
deflection / ISW δT/T / mass density. NSIDE 1024–16384. ORDERING
NESTED or RING (header keyword). Backings: FITS (`healpy.write_map`),
HDF5 shells.

### 4.8 Lightcone — galaxy / halo catalog

Per-row `(z, ra, dec, mag_u, mag_g, …, shear γ₁, γ₂, lensed mag,
host_halo_id, is_central)`. Backings: parquet partitioned by
HEALPix pixel (CosmoDC2, Flagship-2), FITS per tile (Buzzard).

### 4.9 Phase-space tessellation / sheet

Refined 6-D tetrahedral mesh: 4 vertices per simplex, each a
`(q → x, v)` mapping; adjacency, refinement level, deformation-tensor
determinant. Backings: ColDICE custom binary, GAMER-2 complex
wavefunction on AMR. No standard.

### 4.10 Full-GR / relativistic 4-tensor on (3+1) mesh

Per cell: `γ_ij (6 ind.), K_ij (6 ind.), α, β^i (3), T_μν (10)`.
Or weak-field perturbations `φ, χ, B_i, h_ij` (gevolution). Per
time-slice HDF5 per refinement level (Carpet/Chombo) or per Cartesian
grid (gevolution).

### 4.11 IC posterior chain / constrained realisation

Per MCMC sample: `seed, IC_white_noise_field (Nx, Ny, Nz), log_posterior,
log_prior, log_likelihood, gradient, momentum, derived_lightcone_path,
derived_halo_catalog_path`. Backings: BORG HDF5 per chain step;
MANTICORE HDF5 + manifest TSV; SIBELIUS as a parent-code snapshot +
external IC constraint table.

### 4.12 Differentiable / autodiff checkpoint

JAX or TF state: `positions, velocities, cosmology_struct, IC_noise,
step, a_now, gradient`. Backings: orbax checkpoint (pmwd, JaxPM),
TF checkpoint (FlowPM), BigFile (FastPM).

---

## 5. Cross-cutting takeaways

### 5.1 Storage modalities Pillar 3 must accommodate

| Primitive | Examples | Storage choice |
|---|---|---|
| Particle table | Gadget HDF5 family, ASDF, GenericIO, parquet | Wrap native; expose unified API |
| Hydro particle | TNG, EAGLE, FLAMINGO, FIRE, SIMBA | Wrap; common Gadget HDF5 schema |
| AMR mesh | Enzo, RAMSES, FLASH, Nyx, AthenaPK | Wrap; delegate to `yt` |
| Regular grid | FastPM, JaxPM, PM cubes, lensing maps | Zarr-native for new products; wrap legacy |
| Halo catalog | ROCKSTAR, Subfind, CompaSO, AHF, VELOCIraptor | Wrap; emit unified column dictionary |
| Merger tree | SubLink, CT, HBT+, LHaloTree, TreeFrog | Wrap; expose tree-walk API |
| HEALPix lightcone shell | Flagship, FLAMINGO, GLASS, AbacusSummit | FITS native; Pillar 3 adds shell-index parquet |
| Galaxy lightcone catalog | CosmoDC2, Flagship, Buzzard | Parquet partitioned by HEALPix |
| Phase-space sheet | ColDICE | Wrap; no canonical Pillar-3 format |
| GR 4-tensor | ET, gevolution, GRChombo | Wrap; per-time-slice HDF5 |
| IC posterior chain | BORG, MANTICORE | Wrap; build chain manifest |
| Differentiable checkpoint | pmwd, JaxPM, FastPM | Wrap; orbax + BigFile native |

**The hard rule.** Only **particle table** and **regular Cartesian
grid** admit a single canonical re-encoding. Everything else must
be wrapped, not duplicated. Re-encoding AbacusSummit pack9 or HACC
GenericIO would cost tens of PB of storage for zero scientific value.

### 5.2 Hierarchy patterns (real-world layouts)

Five patterns dominate; Pillar 3 should expose all five as first-class
views:

1. **per-cosmology / per-phase / per-snapshot** — AbacusSummit,
   Quijote, CAMELS, Aemulus. Key: `(suite, cosmology_id, phase, redshift)`.
2. **per-simulation / per-snapshot / per-rank-chunk** — TNG, MTNG,
   FLAMINGO, EAGLE, BAHAMAS, Magneticum, Outer Rim. Key:
   `(simulation, snapshot, chunk)`.
3. **per-HEALPix tile / per-shell** — Buzzard, CosmoDC2, FLAMINGO
   maps, Flagship-2, AbacusSummit lightcones. Key:
   `(survey, tile_id, redshift_shell)`.
4. **per-realisation lightcone bundle** — EZmock / GLAM / PINOCCHIO,
   BORG posterior samples. Key: `(survey, realisation_index, tracer)`.
5. **per-zoom-region** — FIRE, NewHorizon, OBELISK, HESTIA, SIBELIUS,
   MANTICORE. Key: `(suite, target_halo, snapshot)`.

A single `oneuniverse.sim.LayoutSchema` enum capturing these covers
~95% of public products.

### 5.3 Public access endpoints

A Pillar-3 fetcher must speak: POSIX direct-read on Perlmutter
(`/global/cfs/cdirs/desi`, `/cfs/cdirs/lsst`, `/cfs/cdirs/m3058/abacus`),
Globus (NERSC DTN, Cosma DTN, ALCF), HTTPS (data.desi.lbl.gov,
www.tng-project.org with API token, FLAMINGO Leiden, SkiesAndUniverses
for Uchuu), S3 (CAMELS / Quijote mirrors), Rubin Science Platform
Butler (CosmoDC2, DC2 truth), MPA portal (TNG, MTNG), AIP Potsdam
(HESTIA), Aquila Consortium share (BORG).

Stage through cache: never read >100 GB without a local mirror.

### 5.4 Common unit / convention pitfalls

| Pitfall | Codes affected | Mitigation |
|---|---|---|
| Velocity = √a · v_pec (Gadget) vs a · dx/dt vs v_pec km/s | Gadget, GIZMO, TNG, AREPO | Pillar-3 unit-frame declaration; convert on read |
| Position [0,L] vs [-L/2,L/2] | AbacusSummit (latter); rest (former) | Per-suite override in manifest |
| Internal energy `u` vs temperature `T` | All hydro | Derive T from `u` + `ElectronAbundance` + μ |
| Mass M⊙/h vs M⊙ | Gadget /h; EAGLE no /h; TNG mixes | Track h_factor in manifest |
| Distance Mpc/h vs Mpc | Gadget /h; some EAGLE outputs no /h | Per-column unit declaration |
| Comoving vs proper | RAMSES `unit_l` proper; Gadget positions comoving | Per-column frame tag |
| Time = scale factor `a`, redshift `z`, or proper `t` | Gadget `Time` is `a` or `t` | Explicit `time_kind` field |
| Metallicity = mass fraction vs Z/Z⊙ | TNG/EAGLE mass fraction; some legacy Z/Z⊙ | Declared convention + reference solar |
| RAMSES code units via `unit_l, unit_d, unit_t` | RAMSES | Parse `info_NNNNN.txt` on ingest |
| Particle ID 32 vs 64 vs 96-bit composite | HACC composite | Always promote to int64; preserve original encoding |
| Particle bitfield Type 0–5 even for DM-only | Gadget | Convention preserved |
| Endianness (pre-2010 IBM/Cray Gadget) | Gadget binary | Sniff first record |
| Cell-centre vs Voronoi generator point | AREPO | Use `CenterOfMass` |
| HEALPix NESTED vs RING | All lightcones | Header keyword `ORDERING` |
| Solar metallicity normalisation | Asplund 0.0127 vs Anders-Grevesse 0.0134 vs Grevesse-Sauval 0.0169 | Explicit reference |
| Gauge convention (ET, gevolution) | full-GR codes | Declared per-snapshot |

### 5.5 Format families that share storage (treat code-agnostically)

- **Gadget HDF5 family**: Gadget-3, Gadget-4, GIZMO, SWIFT, AREPO,
  TNG, EAGLE, SIMBA, FIRE, MillenniumTNG → single reader with
  field-presence introspection.
- **Subfind HDF5 family**: TNG, EAGLE, MillenniumTNG, Auriga, FABLE
  → common halo schema.
- **AMReX plotfile family**: Nyx, CASTRO, ERF, CHARM → reuse AMReX I/O.
- **ConsistentTrees family**: ROCKSTAR + CT, Quijote, Aemulus,
  AbacusSummit legacy → standard 60+ column schema.
- **Particle-mesh family**: FastPM, COLA, pmwd, JaxPM → universal flat-table.

### 5.6 Format families that need fundamentally different storage

- **AMR** (Enzo, RAMSES, FLASH, Nyx) — hierarchical, level-aware;
  cannot be flattened without resolution loss.
- **AREPO Voronoi moving-mesh** — particles plus geometry
  (optional Voronoi connectivity).
- **AbacusSummit ASDF + bit-packed positions** — needs decompression.
- **HACC GenericIO** — rank-chunked, expects MPI; reading requires
  `genericio` Python.
- **RAMSES Fortran multi-file** — O(10⁴) files per snapshot,
  Fortran column-major, must be unified per snapshot.
- **BigFile (FastPM)** — directory-as-file with per-column sub-dirs.
- **Phase-space tessellation (ColDICE)** — simplex topology, not
  point cloud.
- **GR tensor fields** — 10-component metric perturbations + 10
  T_μν on a grid (or AMR).
- **Constrained-realisation chains (BORG, MANTICORE)** — ensemble of
  IC samples + forward models; *posterior*, not single object.

### 5.7 Pillar 1 ↔ Pillar 3 survey ↔ simulation matchups

This drives Pillar 3 ingest priorities:

- DESI Y1/Y3 (BGS/LRG/ELG/QSO) → AbacusSummit + AbacusHOD; UNIT/EZmock;
  GLAM. Lyα: Saclay + Nyx mocks. Cross-CMB: FLAMINGO.
- eBOSS / BOSS DR12 → UNIT + EZmock + GLAM; Outer Rim + Galacticus.
- DES Y1/Y3/Y6 → Buzzard; BAHAMAS (baryon nuisance); Quijote.
- KiDS-1000 → Magneticum; BAHAMAS; FLAMINGO.
- LSST DESC (Rubin) → CosmoDC2; SkyPyMock; FLAMINGO; CAMELS.
- Euclid → Flagship-2 (primary); PINOCCHIO covariance; MTNG IA +
  lensing; FLAMINGO baryon variants; EuclidEmulator2.
- Roman / WFIRST → AbacusSummit HLIS; MTNG; FLAMINGO.
- CMB-S4 / Simons Observatory / Planck × LSS → FLAMINGO (SZ, kSZ,
  X-ray, κ); BAHAMAS; Magneticum.
- eROSITA → Magneticum; FLAMINGO; THESAN (high-z).
- MeerKLASS / SKA-low → THESAN, SPHINX, CROC (reionisation);
  UNIT/AbacusSummit (low-z 21cm).
- 2M++ / Cosmicflows / ZTF / HI / Local-volume → MANTICORE,
  SIBELIUS-DARK, BORG posterior, HESTIA, ELUCID.

**Pillar 3 ingest waves** (sequencing for minimum viable digital
twin coverage):

- **Wave 0** (must-have for DESI/LSST/Euclid demo): AbacusSummit
  (ASDF + parquet), CAMELS (HDF5), Quijote (Gadget + HDF5),
  CosmoDC2 (HDF5 + parquet), DESI EZmocks (FITS + parquet).
- **Wave 1** (joint-probe + baryonic feedback): FLAMINGO (SWIFT +
  SOAP + FITS HEALPix), MTNG, Buzzard, Flagship-2.
- **Wave 2** (specialty + local universe + reionisation): Uchuu,
  UNIT/GLAM, MANTICORE/SIBELIUS, BORG, THESAN, SPHINX, Magneticum,
  BAHAMAS, EAGLE, TNG.
- **Wave 3** (emulator + surrogate): Mira-Titan/CosmicEmu,
  EuclidEmulator2, Aemulus, CARPool, Quijote-PNG.

Pillar 3 federates rather than replicates; the 14+ PB of raw
simulation data stays at NERSC / Cosma / ALCF / MPA.

---

## 6. Proposed OUF-Sim format

**Three design principles (all load-bearing):**

1. **Manifest of manifests.** Pillar 3 stores a *manifest* that
   points at native files + adds an indexing layer. Native readers
   stay authoritative; OUF adds cross-cutting structure. **No
   re-encoding.**
2. **Partial access first.** Every reader takes a mandatory
   selector. Whole-snapshot loads are an explicit opt-in escape
   hatch with a loud docstring warning. Indexes (HEALPix tile,
   octree node, halo→particle pointer, tree branch range) are
   mandatory, not optional.
3. **MPI + GPU read paths are first-class.** Backends declare
   `BackendCapabilities` up-front; reader API accepts
   `mpi_comm=` and `device="cuda:0"` and dispatches to native
   parallel-HDF5 / GenericIO / BigFile / `kvikIO` / Zarr-v3-sharded
   accordingly. Single-process reads remain functional but never
   block the parallel paths.

### 6.1 On-disk layout

```
oufsim_AbacusSummit_base_c000_ph000/
├── manifest.yaml                    # schema-versioned top-level
├── cosmology.yaml                   # Ω_m, σ_8, h, n_s, w_0, w_a, T_CMB
├── unit_frame.yaml                  # canonical unit-frame declaration
├── snapshots/
│   ├── index.parquet                # (snap_id, z, a, path, native_format)
│   ├── snap_000/
│   │   ├── particles_native/        # symlink/path to native (ASDF/pack9)
│   │   ├── halos_native/            # CompaSO halo_info ASDF
│   │   └── ouf_index.parquet        # Pillar-3 indexes (HEALPix, octree, KD)
│   └── …
├── merger_tree/
│   └── tree_native/                 # native CT/SubLink/HBT+
├── lightcone/
│   ├── shells.parquet               # (shell_id, z_min, z_max, NSIDE, path)
│   ├── halo_lightcone_native/
│   └── healpix_shells/
│       └── shell_z0.3_nside4096.fits
├── ic_posterior/                    # optional; BORG chain or MANTICORE
│   ├── chain_manifest.parquet
│   └── samples_native/
├── checkpoints/                     # optional; pmwd/JaxPM/FastPM
│   └── …
└── provenance.yaml                  # run history, code versions, job IDs
```

### 6.2 Schema choices

- **manifest.yaml** is the contract — schema-versioned (`oufsim_format_version: 1.0`).
  Anything not in the manifest is invisible to OUF tooling.
- **index.parquet** carries `(snap_id, z, a, native_path, native_format,
  unit_frame_override, sha256)`. Native files are the source of truth;
  OUF builds indexes alongside.
- **HEALPix index** at fixed NSIDE (NSIDE=64 snapshot-level, NSIDE=4096
  lightcone shells) → fast spatial queries without touching native files.
- **KD-tree / octree** indexes optional, built lazily.
- **Cross-snapshot tree pointer**: parquet `(snap_id, halo_id) →
  (next_snap_id, descendant_halo_id, native_tree_id)`.
- **Unit-frame override** at dataset level — for wrapping legacy
  files where the native header is wrong (a surprisingly common
  failure mode).

### 6.3 What OUF-Sim does NOT do

- **No re-encoding of native data.** Wrap, don't duplicate.
- **No new merger-tree format.** Wrap SubLink / CT / HBT+ and provide
  tree-walking APIs that delegate.
- **No mass-definition unification.** Provide native value + derived
  translation (M200c → Mvir) tagged as derived with conversion cited.
- **No papering over unit / gauge ambiguities.** Force the producer
  to declare them; refuse to load datasets that don't.

### 6.4 OUF-Sim added value

- **Cross-representation queries**: "give me particles in this halo
  at this snapshot" — joins particle table + halo catalog + per-halo
  particle pointer. Built once, stored as parquet sidecar.
- **Cross-snapshot queries**: "give me the main-progenitor mass
  history of this halo" — walks merger tree, pulls per-snapshot halo
  records. Pre-built flat per-halo mass-history table.
- **Cross-simulation queries**: "give me all (Ω_m, σ_8) pairs run
  at this box size + resolution" — top-level `cosmology.yaml` lookup
  across the suite.
- **Lightcone reconstruction**: stitching onion-shell snapshots
  → per-pixel z-vs-shell lookup as HEALPix-indexed parquet.
- **Constrained-realisation linkage**: BORG chain produces realised
  lightcones; OUF cross-links `chain_link_id ↔ realised_snapshot_path
  ↔ realised_lightcone_shell_path`.

### 6.5 Interoperability targets

- **yt** — register OUF as a yt frontend that dispatches to the
  underlying native frontend per dataset.
- **illustris_python** / **abacusutils** / **halotools** /
  **swiftsimio** / **ytree** / **bigfile** / **genericio** — OUF
  accessors return native handles so existing scripts work unmodified.
- **xarray / dask** — for regular-mesh + HEALPix-shell data, expose
  Zarr stores so `xarray.open_zarr(sim.lightcone_zarr_path)` works.
- **MPI** — native parallel HDF5 / GenericIO / BigFile reads remain
  available; OUF never serialises through a single proxy.

### 6.6 Sketch API

```python
from oneuniverse.sim import OUFSim

sim = OUFSim("AbacusSummit_base_c000_ph000")
sim.cosmology                        # cosmology.yaml as dataclass
sim.unit_frame                       # canonical unit-frame declaration
sim.snapshots                        # list of available snapshots
sim.snapshot(z=0.5).particles(type="DM")    # native handle (abacusutils)
sim.snapshot(z=0.5).halos("CompaSO_L1")
sim.merger_tree                      # → ytree-compatible handle
sim.lightcone(shell_z=0.3)           # → healpy / parquet handle
sim.ic_posterior                     # → BORG chain if attached
```

### 6.7 What to prototype first

1. Pick **AbacusSummit** for the first OUFSim wrapper — all
   representations (particles, halos, merger trees, lightcones,
   HOD mocks) exist.
2. Build the wrapper for one box (`AbacusSummit_base_c000_ph000`).
3. Run two cross-cutting analyses: (a) HOD on a halo lightcone;
   (b) trace particle-by-particle main-progenitor history.
4. Compare wall-clock + LOC against raw `abacusutils` calls.
5. If OUF version is within ≈ 1.5× native + noticeably simpler,
   the wrapper model is justified.

### 6.8 Forward-looking notes

- **Zarr v3 + sharding** is becoming the de-facto standard for
  cloud-native scientific arrays. Quijote-MG, CAMELS-Astrid, parts
  of LSST DESC are migrating. OUF should adopt Zarr for *new* mesh /
  cube products.
- **Apache Arrow Flight / Parquet over S3** is the right protocol
  for cross-institution OUF serving. DESI- and LSST-adjacent
  projects already use this stack.
- **WASM-based readers** for in-browser visualisation work well
  with Zarr + Parquet, poorly with HDF5 — argues for Zarr/Parquet
  over HDF5 for *new* OUF-native data.
- **Adoption strategy**: a wrapper format only succeeds if at least
  one major archive (FLAMINGO, MTNG, AbacusSummit, Uchuu, CosmoDC2)
  adopts it. Pursuing a pilot collaboration early is high-leverage.

---

## 7. References

- Springel 2005 (Gadget-2): arXiv:astro-ph/0505010
- Habib et al. 2016 (HACC): arXiv:1410.2805
- Maksimova et al. 2021 (AbacusSummit): arXiv:2110.11398
- Hadzhiyska et al. 2022 (CompaSO): arXiv:2110.11408
- Behroozi et al. 2013 (ROCKSTAR): arXiv:1110.4372
- Behroozi et al. 2013 (Consistent Trees): arXiv:1110.4370
- Rodriguez-Gomez et al. 2015 (SubLink): arXiv:1502.01339
- Han et al. 2018 (HBT+): arXiv:1708.03646
- Elahi et al. 2019 (VELOCIraptor): arXiv:1902.01010
- Knollmann & Knebe 2009 (AHF): arXiv:0904.3559
- Hearin et al. 2017 (halotools): arXiv:1606.04106
- Adamek et al. 2016 (gevolution): arXiv:1604.06065
- Clough et al. 2015 (GRChombo): arXiv:1503.03436
- Jasche & Wandelt 2013 (BORG): arXiv:1203.3639
- McAlpine et al. 2022 (SIBELIUS-DARK): arXiv:2202.05606
- Stopyra et al. 2024 (MANTICORE): arXiv:2410.20307
- Li et al. 2022 (pmwd): arXiv:2211.09958
- Modi et al. 2021 (FlowPM): arXiv:2010.11847
- Feng et al. 2016 (FastPM): arXiv:1603.00476
- Sousbie & Colombi 2016 (ColDICE): arXiv:1509.07720
- Hahn et al. 2013 (Sheet codes): arXiv:1304.2049
- Schive et al. 2014 (ψDM): arXiv:1406.6586
- Hernández-Aguayo et al. 2023 (MillenniumTNG): arXiv:2210.10059
- Schaye et al. 2023 (FLAMINGO): arXiv:2306.04024
- Villaescusa-Navarro et al. 2020 (Quijote): arXiv:1909.05273
- Villaescusa-Navarro et al. 2021 (CAMELS): arXiv:2010.00619
- Ishiyama et al. 2021 (Uchuu): arXiv:2007.14720
- Korytov et al. 2019 (CosmoDC2): arXiv:1907.06530
- DeRose et al. 2022 (Buzzard): arXiv:1901.02401
- Euclid Collaboration: Castander et al. 2024 (Flagship-2): arXiv:2405.13495
- Garaldi et al. 2022 (THESAN): arXiv:2110.01628
- Rosdahl et al. 2018 (SPHINX): arXiv:1801.07259
- Libeskind et al. 2020 (HESTIA): arXiv:2002.07391
- Heitmann et al. 2019 (Outer Rim): arXiv:1904.11970
- Heitmann et al. 2021 (Last Journey): arXiv:2109.01956
- DeRose et al. 2019 (Aemulus): arXiv:1804.05865
- Klypin & Prada 2018 (GLAM): arXiv:1701.05690
- Munari et al. 2017 (PINOCCHIO): arXiv:1605.04788
- Wang et al. 2014/2016 (ELUCID): arXiv:1407.3451, arXiv:1608.01763
- Malz & Marshall 2018 (qp / photo-z PDFs): arXiv:1806.00014
