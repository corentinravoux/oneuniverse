# Cosmology Survey Landscape v2 — Survey-Agnostic Reference (merged)

**Date:** 2026-06-05.
**Purpose:** A **second, clean, survey-agnostic** pass over the cosmology survey
landscape, **merged** with v1
([`survey_landscape_review.md`](survey_landscape_review.md)). v1 catalogues
surveys *by name* with exhaustive per-column detail (kept as the detailed
companion). v2 reorganises the same space **by observable / probe / data
geometry** so the `oneuniverse.measure` output format (the Universal
DataProduct + MeasurementSet) is general across *survey types*, not tuned to
named surveys — and refreshes currency to mid-2026.

Feeds: [`2026-06-05-p1-to-p2-measurement-requirements.md`](2026-06-05-p1-to-p2-measurement-requirements.md)
(atoms + probe matrix) and [`../plans/2026-06-05-pillar2-definition.md`](../plans/2026-06-05-pillar2-definition.md)
(the contract). **Scope:** P1 ingest + the P1→P2 measure format must accommodate
*all* of the classes below.

---

## 1. The agnostic axes (how to classify ANY survey)

A survey is a point in a 5-axis space. The format must span each axis, not
enumerate surveys. Any new/unseen survey is ingested by locating it here.

| Axis | Values |
|---|---|
| **A. Observable modality** | broadband flux · spectrum (1D) · shape/ellipticity · absorption (δ along LOS) · intensity field (map/cube) · time series (light curve) · event posterior · distance indicator · time delay |
| **B. Data geometry** | point catalog · sightline · HEALPix/pixel map · N-D cube · light-curve series · lens-system / multi-image · region/mask |
| **C. Redshift knowledge** | spec-z · photo-z PDF (kernel) · tomographic n(z) · grism-z · none (flux-only) · derived (Lyα, distance) · multi-z (helio/CMB/VI/QN/PCA) |
| **D. Probe(s) enabled** | clustering (3D/angular) · RSD/f σ₈ · BAO · cosmic shear · GGL · CMBκ×g · SZ×g · peculiar velocity · SN distances · Lyα P1D/P3D · cluster counts · voids · ISW · LIM P(k) · time-delay H₀ · sirens · cosmic chronometer H(z) |
| **E. Selection / weight character** | fiber-assignment + collisions (PIP) · imaging systematics · shear calibration (metacal/lensfit) · flux-limit + completeness · veto masks · survey-published randoms · depth/PSF maps |

The three DataProduct geometry subtypes (PointSet / Sightline / FieldMap)
collapse axis B: point/light-curve/lens-system/distance → **PointSet** (with
attribute atoms); sightline → **Sightline**; map/cube/IM → **FieldMap**. v2
confirms no 4th subtype is required (below).

---

## 2. Survey-type catalog (by observable/probe class)

Each class: representative surveys (v1 detail + 2026 refresh), geometry, the
distinguishing atoms, the probes, and the DataProduct subtype.

### I. Spectroscopic galaxy / QSO redshift surveys
- **Now:** SDSS/BOSS/eBOSS, **DESI DR2 (2025-03-19; BGS/LRG/ELG/QSO ~30M + Lyα)**, GAMA, 2dF/6dF/WiggleZ, VIPERS/zCOSMOS/DEEP2-3, Euclid NISP grism (Q1 2025), Roman HLSS.
- **Coming:** **4MOST (ops ~2026)**, WEAVE, MOONS, PFS (started), **DESI-II (2029–35)**, MUST/Spec-S5/Wide-field Stage-V.
- **Geometry:** PointSet + 1D spectrum sidecar. **z:** spec-z (+ multi-z for QSO). **Probes:** 3D clustering, RSD/f σ₈, BAO. **Weights:** CP/NOZ/SYSTOT/FKP + PIP bitweights. **Randoms:** survey-published (DESI/eBOSS ship them) → *prefer ingest*.

### II. Photometric (imaging) + photo-z surveys
- **Now:** KiDS(+VIKING), DES (Y6 2024-25), HSC SSP (PDR4), **Rubin/LSST (commissioning → DR1 ~2027)**, Euclid VIS+NISP (Q1 2025), UNIONS, J-PAS/J-PLUS, COSMOS2020/-Web, SPHEREx (also LIM, class X).
- **Geometry:** PointSet + **photo-z kernel (`qp`)** + tomographic n(z). **Probes:** angular/tomographic clustering, photometric BAO, magnification, lens samples for GGL. **Weights:** imaging-systematics (linear/SYSNet/NN), depth/PSF maps.

### III. Weak-lensing shape catalogs
- **Now:** **KiDS-Legacy (DR5, 2025 cosmic shear)**, DES Y3/Y6, **HSC Y3 (2025, DESI clustering-z, S₈≈0.805)**, Euclid (forthcoming), UNIONS/ShapePipe, Rubin metadetect. Stage-III combined S₈≈0.813.
- **Geometry:** PointSet(source) with **shapes e1,e2 + calibration** (metacal `R11..R_S` / lensfit `m,c`) + shear_weight + PSF + per-bin n(z); or **FieldMap** κ mass map. **Probes:** cosmic shear ξ±/Cℓ, GGL γ_t/ΔΣ, 3×2pt.

### IV. Lyα forest (sightlines)
- **Now:** eBOSS DR16Q, **DESI Lyα DR2 (~2026, ~1.2M)**. picca `delta-{HPX}.fits.gz`, NSIDE=16 NEST. **Geometry:** Sightline (δ_F(λ), mask, continuum, resolution). **Probes:** P_1D, P_3D/ξ_3D BAO, Lyα×QSO/void. **Sub-objects:** DLA/BAL catalogs. **Blinding** keyword.

### V. Peculiar-velocity surveys
- **Now:** 6dFGSv, 2MTF, **CosmicFlows-4 → CF4++ (CF4 + WALLABY/FAST/DESI PV, ~65k)**, SDSS-PV, **DESI PV (Y5 ~200k)**, ZTF/Foundation low-z SNe. **Geometry:** PointSet + **μ / η / log-distance / v_pec + σ_v** + indicator type (TF/FP/SBF/SNIa). **Probes:** velocity ξ, density×velocity, f σ₈ (flip inputs, built P1-side).

### VI. Supernovae / standard candles
- **Now:** **Pantheon+ (1701, full cov)**, **Union3/UNITY (~2000, hierarchical HDF5)**, **DES-SN5YR (1635 phot + 194 spec)**, ZTF BTS; **LSST/Roman SN (per-SN SALT3 + light curves)**. **Geometry:** PointSet + **light curves (flux(t,band))** or μ + SALT(x0,x1,c) + **row-correlated covariance** (`cov_id`). **Probes:** Hubble diagram μ(z); PV.

### VII. Galaxy clusters (optical / X-ray / SZ)
- **Now:** redMaPPer (DES), **eROSITA eRASS1 (~12k clusters)**, ACT DR5/6 (~4k SZ), SPT-3G, Planck PSZ2, Euclid clusters. **Geometry:** PointSet + **mass proxy (λ/T_X/Y_SZ/L_X)** + **member sub-object** + **selection function**. **Probes:** counts N(obs,z), cluster clustering, cluster lensing (×shear).

### VIII. CMB primary + secondaries (maps)
- **Now:** Planck NPIPE, ACT DR6 (κ NSIDE=4096 + sims), SPT-3G, **Simons Observatory (final construction; lensing + SZ, ~10⁵ SZ clusters goal)**. **CMB-S4 cancelled (DOE/NSF, 2025-07-09).** **Geometry:** FieldMap (κ_CMB, y, T) + mask + noise sims. **Probes:** CMBκ×g, tSZ/kSZ×g, ISW; cluster catalogs (→ class VII).

### IX. HI 21cm emission / radio continuum
- **Now:** HIPASS, ALFALFA, **WALLABY (ASKAP, HI catalogs + cubes)**, MIGHTEE; continuum **LoTSS-DR2 (2025 angular clustering, counts-in-cells, ISW)**, **EMU/RACS (ASKAP)**, NVSS/FIRST; **SKA pathfinders → SKA**. **Geometry:** PointSet (HI/continuum sources; continuum often **flux-only, no z** → photo-z/none) + FieldMap (HI cubes). **Probes:** HI clustering, radio angular clustering, ISW, magnification, cosmic dipole.

### X. Line-intensity mapping (LIM) — *expanded in v2*
- **Now/Coming:** **SPHEREx (launched 2025; UV/optical lines Lyα/[OII]/[OIII]/Balmer)**, **COMAP (CO)**, **CONCERTO ([CII])**, **CCAT-FYST/EoR-Spec ([CII], first light 2026)**, TIME, CHIME/HERA (21cm), SKA-IM. **Geometry:** FieldMap cube (RA,Dec,ν|z) + foreground/interloper mask + beam. **Probes:** LIM auto-P(k), LIM×galaxies, BAO, fNL. **New atom:** interloper/foreground model handle; beam + spectral-response metadata.

### XI. Gravitational-wave standard sirens
- **Now:** GWTC-3/4 (LVK O4 growing); future ET/CE. **Geometry:** PointSet(event) + **FieldMap (MOC HEALPix skymap prob + DISTMU/SIGMA/NORM)** + host-galaxy sub-object + **distance posterior samples**. **Probes:** bright/dark-siren H₀, ×galaxy density.

### XII. Strong lensing / time-delay cosmography — *new in v2*
- **Now:** **TDCOSMO-2025 (8 time-delay lensed quasars + SLACS/SL2S kinematics, JWST/Keck/VLT σ_v)**, **Euclid Q1 strong-lens discovery engine (galaxy + cluster lenses)**, LSST (~10⁵ lenses forecast). **Geometry:** **lens-system object** (→ PointSet with a multi-image/system sub-object): image positions, **time delays Δt_ij**, lens & source z, deflector **kinematics σ_v**, **external convergence κ_ext**. **Probes:** time-delay H₀, lensing mass profiles. **New atom:** time-delay + image-configuration + κ_ext payload (a sub-object hierarchy: system → images).

### XIII. Cosmic chronometers — *new in v2 (niche)*
- Passive galaxies → differential age → **H(z)**. **Geometry:** PointSet + spectral age indicator (D4000) / SED-age posterior. **Probes:** H(z) direct. Small but a distinct cosmological probe; carried as PointSet attribute.

### XIV. Astrometric / proper-motion cosmology (niche)
- **Gaia DR3/DR4** QSO proper motions → secular aberration / cosmic dipole; also the reference astrometric frame. **Geometry:** PointSet + PM + parallax + epoch 2016.0. **Probes:** dipole, frame ties (mostly a cross/reference layer).

### XV. Time-domain / transient streams
- ZTF/Rubin alerts (Avro + cutouts), brokers (Alerce/Fink/Lasair classifier probs), TNS. **Geometry:** light-curve PointSet + alert payload. **Probes:** SN cosmology feeders, kilonova sirens. *(lower priority for the static measure format)*.

### XVI. Reference imaging VACs
- Gaia DR3, Legacy LS DR9/10, Pan-STARRS, (un)WISE/CatWISE, 2MASS, UKIDSS/VIKING. Not probes themselves — cross-ID + photometry + masks feeding I–III.

### XVII. Simulated mocks
- UNIT, **AbacusSummit**, Outer Rim/LastJourney (HACC), Quijote, MillenniumTNG, Buzzard, **Euclid Flagship2**, Uchuu. Same DataProduct shapes (so a mock MeasurementSet is byte-compatible with data for covariance + validation).

---

## 3. Currency refresh (mid-2026 deltas vs v1)

| Item | v1 (2026-05-28) | v2 update |
|---|---|---|
| DESI | "DR2 ~2026" | **DR2 released 2025-03-19** (BAO incl. Lyα; full BGS/LRG/ELG/QSO) |
| Euclid | "DR1 ~2026" | **Q1 released 2025-03** (strong-lens engine, cluster lenses); WL forthcoming |
| Cosmic shear | KiDS-1000/DES-Y3/HSC-Y3 | **KiDS-Legacy (DR5) + HSC-Y3 w/ DESI clustering-z (S₈≈0.805); combined S₈≈0.813** |
| Peculiar velocity | CF4 (~56k) | **CF4++ (~65k; + WALLABY/FAST/DESI PV)** |
| SN | Pantheon+/Union3/DES-SN5YR | confirmed; DES-SN5YR vs Pantheon+ dark-energy debate live |
| LIM | (absent as a class) | **SPHEREx launched 2025; FYST first light 2026; COMAP/CONCERTO ongoing** |
| Radio continuum | (folded in HI) | **LoTSS-DR2 cosmology (clustering, CiC, ISW); EMU/RACS** — own class |
| Strong lensing | (absent) | **TDCOSMO-2025 time-delay H₀; Euclid Q1 lens catalogs** — own class |
| Next-gen CMB | (Planck/ACT/SPT) | **Simons Observatory ongoing; CMB-S4 cancelled 2025-07-09** |
| Spectroscopic | DESI/4MOST/PFS | **4MOST ops ~2026, WEAVE/MOONS/PFS active, DESI-II 2029–35** |

---

## 4. What v2 adds for the `measure` format (agnostic implications)

v1's modality inventory (§13) already drove the OUF schema. v2's agnostic pass
surfaces these **additional format requirements** for the P1→P2 measure layer:

1. **Flux-only point sets (no redshift).** Radio continuum (LoTSS/EMU), some IM
   tracers: angular clustering with **no z column** — n(z) comes from external
   redshift distribution / cross-match, not per-object. PointSet must allow
   `z`-absent with an attached external `dndz`.
2. **Lens-system geometry = sub-object hierarchy, not a new subtype.** Strong
   lenses (TDCOSMO) → a PointSet of *systems* + a `system→image` sub-object link
   carrying time delays Δt, image positions, κ_ext, deflector σ_v. Confirms the
   3-subtype model holds (no 4th geometry) — it is a *hierarchy*, reusing P1
   sub-object links.
3. **LIM/IM cubes = FieldMap with a spectral axis + interloper handle.** Already
   covered by FieldMap (WCS axes); add `beam` + `spectral_response` +
   `interloper_model` metadata slots.
4. **Distance-indicator + correlated covariance as a first-class PointSet
   attribute** (PV + SN): μ/η/v_pec + `cov_id` → external covariance store (v1
   modality #15) is load-bearing for the format, not optional.
5. **Randoms: ingest OR generate (owner decision).** The format carries a
   `randoms` slot with provenance `{source: ingested|generated, recipe}`;
   survey-published randoms (DESI/eBOSS) preferred, generation available when
   absent — both first-class.
6. **External n(z) provenance** (clustering-z vs photo-z stack vs spec-z): the
   measure layer must record *which* method produced n(z) (it changes the
   covariance) — newly emphasised by HSC-Y3's DESI clustering-z recalibration.
7. **Multi-tracer same-object** (e.g. a galaxy that is both a clustering tracer
   and a shear source): one ONEUID feeding two DataProducts in a MeasurementSet
   with a shared region_map — the format must not duplicate the object.

**Net:** the Universal DataProduct (PointSet / Sightline / FieldMap) + sub-object
hierarchies + the atom inventory **cover all 17 classes**. The only additions are
*attribute/metadata slots* (flux-only z-absent, time-delay sub-object, beam +
interloper, cov_id, randoms provenance, n(z) provenance), not new geometries.

---

## 5. Build-order confirmation

The agnostic pass does not change the priority: **galaxy clustering
(spectroscopic)** is class I, the most-used probe, the only one fully backed by
real P1 data (DESI + eBOSS — [[project_p1_real_ingestion_status]]). It exercises
the PointSet core + randoms(ingest+generate) + n(z) + window + weights + region,
which classes II/III/V/VI then reuse with added attribute atoms.

---

## 6. Sources (second search, 2026-06)

- DESI DR2 / Euclid Q1: [DESI DR2 papers](https://data.desi.lbl.gov/doc/papers/dr2/), [DESI DR2 extended-cosmology](https://iopscience.iop.org/article/10.3847/2041-8213/ade1cc), [Euclid Q1 strong-lens engine](https://arxiv.org/pdf/2503.15324)
- Cosmic shear: [KiDS-Legacy cosmic shear (A&A 2025)](https://www.aanda.org/articles/aa/full_html/2025/11/aa54908-25/aa54908-25.html), [HSC-Y3 + DESI clustering-z](https://arxiv.org/html/2511.18134v1)
- LIM: [SPHEREx IM case study](https://arxiv.org/html/2509.02414v1), [CCAT/FYST](https://arxiv.org/html/2511.01707v1)
- Radio continuum: [LoTSS-DR2 counts-in-cells (A&A 2025)](https://www.aanda.org/articles/aa/full_html/2025/06/aa52734-24/aa52734-24.html), [future radio continuum cosmology (MNRAS)](https://academic.oup.com/mnras/article/506/3/4121/6317625)
- CMB next-gen: [CMB-S4 revised plan / shutdown (2025-06)](https://indico.global/event/14611/contributions/130157/attachments/60314/116145/Revised_CMB_S4_Project_Plan_Report.pdf), [Simons Observatory](https://en.wikipedia.org/wiki/Simons_Observatory)
- Strong lensing: [TDCOSMO 2025 (A&A)](https://www.aanda.org/articles/aa/full_html/2025/12/aa55801-25/aa55801-25.html)
- PV/SN: [Cosmicflows-4](https://arxiv.org/pdf/2209.11238), [DES-SN5YR vs Pantheon+ (MNRAS)](https://academic.oup.com/mnras/article/541/3/2585/8191262)
- Spectroscopic future: [Cosmology with wide spectroscopy surveys (EAS 2026)](https://eas.unige.ch/EAS2026/session.jsp?id=S1), [MUST](https://arxiv.org/html/2605.10102v1)

## 7. Relationship to other docs

- v1 detailed per-survey columns: [`survey_landscape_review.md`](survey_landscape_review.md) (companion; not superseded — v2 is the agnostic index over it).
- Atoms + probe matrix: [`2026-06-05-p1-to-p2-measurement-requirements.md`](2026-06-05-p1-to-p2-measurement-requirements.md).
- Contract: [`../plans/2026-06-05-pillar2-definition.md`](../plans/2026-06-05-pillar2-definition.md).
- P1 schema mapping: [`schema_generalisation_audit.md`](schema_generalisation_audit.md).
