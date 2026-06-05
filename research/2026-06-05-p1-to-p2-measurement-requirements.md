# P1 → P2 Measurement Requirements — probe-by-probe deep research

**Date:** 2026-06-05.
**Purpose:** Define, across (almost) all cosmological probes, **what a
`MeasurementSet` must carry** and **what the P1→P2 connection must do** to build
it from Pillar-1 OUF data — so the Universal DataProduct
([`plans/2026-06-05-pillar2-definition.md`](../plans/2026-06-05-pillar2-definition.md))
is general enough to match the data coverage Pillar 1 already targets.
**Axis:** the *measurement* axis (probe → required components → output shape),
complementary to the *data-modality* axis already in
[`survey_landscape_review.md`](survey_landscape_review.md) §13 and the column
coverage in [`schema_generalisation_audit.md`](schema_generalisation_audit.md).
**Scope note (owner, 2026-06-05):** ignore adapters to external packages for
now. The goal is the P1→P2 connection that *produces* a complete MeasurementSet.
Build order: **galaxy clustering first**, then one probe at a time.

---

## 1. What "the P1→P2 connection" is

Pillar 1 ingests *everything* (catalogs, sightlines, maps, cubes, PDFs, weights,
sub-object links) and stores it verbatim with observational metadata only. The
P1→P2 connection is the **analysis-preparation pipeline** that turns that raw,
universal store into a **measurement-ready** object. The owner's verbs:
*select / weight / clean / decide which dataset / define the aimed
cross-correlation + measurement.* Decomposed into nine transform steps:

| # | Step | Input (P1) | Output (toward MeasurementSet) |
|---|---|---|---|
| 1 | **Select** | OUF datasets + ONEUID | tracer definition: which dataset(s), z-range, target/quality cuts |
| 2 | **Clean** | rows + veto/quality flags + ONEUID | dedup, veto-masked, z-conflict-resolved sample |
| 3 | **Weight** | `combine` weight primitives | one total `weight` per object (+ named components kept) |
| 4 | **Randoms** | footprint + n(z) | random catalog matching angular × radial selection |
| 5 | **Window/Footprint** | depth/veto/systematics maps | angular mask (HEALPix/mangle) + completeness |
| 6 | **n(z)** | z_spec / photo-z / clustering-z | radial selection (per-bin or per-row) |
| 7 | **Photo-z kernel** | OUF PdfSpec (`qp`) | per-object p(z) attached (photometric probes) |
| 8 | **Region** | sky positions | shared HEALPix `region_id` (jackknife/bootstrap) |
| 9 | **Measurement spec** | user intent | tracer pairs + statistic + binning + estimator family (cosmology deferred to P2) |

**End state:** everything an external estimator needs is present in the
MeasurementSet; cosmology is the only thing still chosen later (at the P2 call).

---

## 2. DataProduct atom inventory (the union the contract must cover)

Every probe is some combination of these **atoms**. The MeasurementSet contract
must be able to carry any subset. Grouped by kind:

### A. Geometry carriers (the DataProduct subtype)
- **PointSet** — `ra, dec, z, weight, region_id` (+ probe attributes below).
- **Sightline** — per-LOS `δ_F(λ|z)`, `mask`, `continuum`, `resolution`, LOS `ra,dec`, `z_qso`.
- **FieldMap** — pixel/voxel field `values` + `mask` + axes (HEALPix NSIDE/ordering, or WCS for cubes).

### B. Redshift representations
- `z_spec` point + `z_err` + `z_type`.
- **Photo-z kernel** — per-object p(z) as `qp` (interp grid / quantiles / Gaussian-mixture / samples). *(the "photoz kernel")*
- **Tomographic n(z)** — per-bin ensemble n(z) on a shared grid + per-row bin assignment.
- **Multiple-z columns** — `z_helio, z_CMB, z_VI, z_QN, z_PCA, z_LYA, …` (disagreement structure for QSO/Lyα).

### C. Per-object physical attributes (probe-specific)
- **Shapes** — `e1, e2` + calibration (metacal `R11..R_S`, lensfit `m, c1, c2`), `shear_weight`, PSF size/ellipticity.
- **Distances / velocities** — distance modulus `μ`, log-distance-ratio `η`, peculiar velocity `v_pec`, distance-indicator type (TF/FP/SNIa), `σ_v`.
- **Light curves** — per-epoch `flux(t, band)` + errors + zeropoints (SN/transient); SALT params `x0,x1,c`.
- **Photometry** — multi-band fluxes/mags, **variable filter set** (5 ugriz … 56 J-PAS).
- **Mass proxies** — cluster richness `λ`, `T_X`, `Y_SZ`, member lists (sub-object).
- **Spectral features** — emission-line fluxes, equivalent widths (ELG, line-intensity).

### D. Selection products
- **Randoms** — matched to footprint × n(z); carry compatible `region_id` + `weight`.
- **Footprint / angular mask** — HEALPix completeness or mangle polygons; veto masks (bright stars, bad regions).
- **Depth / systematics maps** — depth, seeing, PSF, extinction, stellar density (window + weighting input).
- **Selection function** — radial (n(z)) × angular (mask) completeness.

### E. Weight families (named, composable)
- FKP, completeness, fiber-collision (CP), redshift-failure (NOZ), systematics (SYSTOT), imaging-systematics (SYSNet/linear).
- **PIP / bitwise** — `BITWEIGHTS: i8[N]` realisations (DESI), fiber-collision exact.
- Shear weights + responses (metacal/lensfit).
- Inverse-variance (PV/SN), per-pixel Lyα weights.

### F. Fields / maps (gridded)
- WL convergence `κ` (mass map), shear `γ1,γ2` maps.
- CMB lensing `κ_CMB`, tSZ `y`, kSZ temperature.
- Reconstructed **density** / **velocity** fields (the flip inputs; also BAO recon shifts).
- HI / 21 cm brightness-temperature cubes; line-intensity maps.

### G. Region & covariance scaffolding
- Shared HEALPix `region_map` (one scheme across all tracers in a set).
- Mock-suite reference (EZmock/GLAM/Abacus) handle for sample covariance.
- Analytic-covariance ingredients: `n̄(z)`, shot noise, pair counts, window multipoles.

### H. Metadata (no cosmology)
- `frame` (icrs/galactic/ecliptic), `epoch`, units, magnitude system, `wavelength_convention`, rest-frame state.

---

## 3. Probe-by-probe requirement matrix

For each probe: the measurement statistic, the atoms it needs, the P1 source
geometry, its natural cross-correlation partners, and a build priority.
**P** = PointSet, **S** = Sightline, **M** = FieldMap.

| Probe | Statistic(s) | DataProduct + key atoms | P1 source | Cross partners | Priority |
|---|---|---|---|---|---|
| **3D galaxy clustering (spec)** | ξ(s,μ)/P(k,μ) multipoles; BAO; RSD f σ₈ | **P**: pos + z_spec + weights(FKP,CP,NOZ,SYSTOT) + **randoms** + **n(z)** + footprint + region | OUF POINT (eBOSS/DESI) | self; ×Lyα; ×CMBκ; ×voids | **1** |
| **BAO reconstruction** | shifted ξ/P; recon randoms | **P** + displacement **field M** (Zel'dovich recon) + smoothing scale | POINT + derived field | — | 2 |
| **Angular / tomographic clustering (photo)** | w(θ), C_ℓ^{gg} | **P**: pos + **photo-z kernel** + tomo-bin + **per-bin n(z)** + systematics weights + footprint + randoms | OUF POINT + PdfSpec | ×shear (GGL); ×CMBκ | 3 |
| **Cosmic shear (WL)** | ξ±(θ), C_ℓ^{κκ} | **P**(source): **shapes** e1,e2 + metacal/lensfit calib + shear_weight + **photo-z n(z)** + footprint; or **M**: κ map | POINT + shear cols (DES/KiDS/HSC) | ×clustering (GGL); ×CMBκ | 2 |
| **Galaxy–galaxy lensing** | γ_t(θ), ΔΣ(R) | **P**(lens) × **P**(source shapes) sharing region | two POINT sets | the 3×2pt vector | 2 |
| **3×2pt** | {gg, gγ, γγ} joint | bundle of the above + joint covariance | multiple POINT | — | 3 |
| **CMB lensing × galaxies** | C_ℓ^{gκ}, C_ℓ^{κκ} | **P**(galaxies) × **M**(`κ_CMB` HEALPix + mask) | POINT × external map | ×shear; ×Lyα | 3 |
| **tSZ / kSZ × galaxies** | C_ℓ^{gy}; kSZ stacking + **v_rec** | **P** × **M**(`y` / CMB T map); kSZ needs reconstructed velocities | POINT × map | — | 4 |
| **Peculiar velocities (PV)** | velocity ξ; density–velocity cross; f σ₈ | **P**: pos + **μ / η / v_pec** + σ_v + IVar weight; **velocity+density fields** | POINT (CF4/DESI-PV/SN) | ×galaxy density | 2 |
| **SN Ia (Hubble diagram)** | μ(z); also PV | **P**: z + **μ**(or **light curves** + SALT x0,x1,c) + cov(syst); host props | POINT + lightcurve sidecar (Pantheon+/DES-SN) | PV; ×density | 3 |
| **Lyα forest P_1D** | P_1D(k,z) | **S**: δ_F(λ) + mask + continuum + **resolution** + noise | OUF SIGHTLINE (DESI/eBOSS Lyα) | — | 3 |
| **Lyα P_3D / ξ_3D** | P_3D, ξ(r_∥,r_⊥); BAO | **S** + sightline footprint + **randoms** (LOS) | SIGHTLINE | ×QSO; ×voids | 3 |
| **Lyα × QSO / void** | cross ξ | **S** × **P**(QSO/void) sharing region | SIGHTLINE × POINT | — | 4 |
| **Galaxy clusters** | counts N(λ,z); cluster clustering; cluster lensing | **P**: pos + z + **mass proxy** λ/T_X/Y + **member sub-object** + **selection function** | POINT + sub-object (redmapper/eROSITA) | ×shear (lensing) | 4 |
| **Voids** | void-galaxy ξ; RSD; ISW | **P**(voids, derived) × **P**(galaxies) or **S**(Lyα) | derived from POINT/SIGHTLINE | ×galaxy; ×Lyα; ×CMB(ISW) | 4 |
| **HI / 21 cm intensity mapping** | P(k); ×galaxies | **M**: T_b cube (RA,Dec,ν) + foreground mask + beam | OUF CUBE | ×galaxy clustering | 5 |
| **GW standard sirens** | H₀ via host z; ×LSS | **P**(events) + **M**(skymap prob) + host-galaxy sub-object + **distance posterior** | OUF GW_SKYMAP + POINT hosts | ×galaxy density | 5 |

**Coverage check:** every atom in §2 is exercised by ≥1 probe; PointSet covers
the majority (clustering, lensing, PV, SN, clusters, voids), Sightline covers
Lyα, FieldMap covers map-based and IM probes + derived κ/density/velocity. The
three-subtype split is sufficient — no probe needs a 4th geometry (cubes fold
into FieldMap with WCS axes).

---

## 4. The "aimed cross-correlation / measurement" spec

Step 9 (the user's "define which cross-corr type and measurement") is a small
declarative object the connection emits alongside the data — it does **not**
compute anything (cosmology + estimator math stay in P2):

```
MeasurementSpec
  tracers:   [name, ...]                      # which DataProducts
  pairs:     [(a, b), ...]                     # auto (a,a) and cross (a,b)
  statistic: xi_smu | pk_multipole | w_theta | cl |
             xi_pm | gamma_t | delta_sigma | p1d | p3d | counts | hubble
  binning:   edges + (ell|s|theta|k|r) convention
  coords:    on-sky | comoving(placeholder)    # z->r deferred to P2 fiducial
  covariance: jackknife(region_map) | mocks(handle) | analytic(ingredients)
  estimator_family: clustering | field_level | lensing | lya | sn | cross
```

This is the bridge between "we have the data" and "P2 runs the estimator": it
fixes *what* is measured and *how it is binned/resampled*, leaving only the
fiducial cosmology to P2.

---

## 5. Generalised MeasurementSet shape (synthesis)

Combining §2–§4, the contract a complete P1→P2 connection emits:

```
MeasurementSet
  products: {name -> DataProduct}              # PointSet | Sightline | FieldMap
      each carries only the atoms its probe needs (§2):
        PointSet : catalog(pos,z,weight,region) + randoms + nz
                   + optional {photoz_kernel, shapes+calib, mu/eta/v_pec,
                               lightcurves, photometry, mass_proxy}
        Sightline: los + delta + mask + continuum + resolution [+ randoms]
        FieldMap : values + mask + axes                # kappa, y, density, velocity, T_b
  region_map: HEALPix (shared by all products)         # joint jackknife
  spec:       MeasurementSpec                           # §4
  window:     {footprint mask, veto, depth/systematics maps}
  covariance: jackknife | mocks | analytic ingredients
  metadata:   frame, epoch, units, wavelength_convention   # NO cosmology
  provenance: ONEUID/dataset ids + weight recipe per product
```

**Invariant set** (generalises the old catalog-only invariants):
1. No cosmology anywhere in the object.
2. All products share `region_map` + `metadata.frame/epoch`.
3. Randoms (where present) drawn from the product's own window + n(z); compatible region.
4. Photo-z kernel and tomographic n(z) are first-class (photometric probes).
5. Weights are named + composable; the total `weight` is derived, components kept for audit.
6. Every product declares its `.kind` so an estimator accepts/rejects cleanly.

---

## 6. Build order (one probe at a time)

Per owner: most-used probes first. Each step is later turned into a
writing-plans TDD plan; this doc fixes the target.

1. **Galaxy clustering (spec)** — PointSet + randoms + n(z) + footprint + FKP/CP/NOZ/SYSTOT weights + region + `MeasurementSpec(pk_multipole/xi_smu)`. Exercises atoms A·P, B(z_spec), D, E(core), G, H. **Proven on DESI + eBOSS** (the only real P1 surveys today — [[project_p1_real_ingestion_status]]).
2. **Cosmic shear + GGL** — adds shapes+calibration, photo-z n(z), tomographic bins, the 3×2pt bundle.
3. **Peculiar velocities + SN** — adds μ/η/v_pec, light curves, velocity/density fields (the flip-input shape, but built P1-side, not flip-side).
4. **Lyα (P_1D/P_3D)** — the Sightline subtype end-to-end.
5. **Map × catalog (CMBκ, tSZ, HI) + voids/clusters/GW** — FieldMap subtype + cross partners.

Rationale for order: (1) is the single most-used LSS measurement and the only
one fully backed by real P1 data now; (2)–(3) reuse PointSet with new attribute
atoms; (4)–(5) introduce the other two geometry subtypes.

---

## 7. Open design questions (resolve when building each probe)

1. **Randoms: generate vs ingest.** P1 may already hold survey-published
   randoms (DESI/eBOSS ship them). Connection should *prefer ingesting* the
   official randoms (ONEUID-linked) and only *generate* when absent. Decide the
   precedence + how generated randoms reproduce the angular×radial selection.
2. **Window representation.** HEALPix completeness map (simple, lossy at edges)
   vs mangle polygons (exact, heavier). Start HEALPix; keep a polygon escape.
3. **n(z) provenance.** spec-z histogram vs photo-z stack vs clustering-z. The
   connection should record *which* method produced n(z) (it changes the
   covariance), not just the array.
4. **Per-row vs per-bin photo-z.** Tomographic probes need per-bin n(z); some
   estimators want per-object p(z). Carry both when available; the `qp` kernel
   is the per-object source, the per-bin n(z) its stacked summary.
5. **Weight composition policy.** Order of multiplication + which components are
   multiplicative vs additive (PIP bitwise is special). Mirror BOSS/DESI
   conventions; keep components, expose the product.
6. **Region map granularity.** One NSIDE for all tracers in a set (cross-cov
   consistency) vs per-tracer. Lock to one shared scheme (invariant 2).
7. **Field provenance (velocity/density/κ).** Some "fields" are *derived* (BAO
   recon displacement, mass maps, velocity reconstruction). Is that derivation
   P1→P2 connection work or a P2 estimator step? Provisional: *reconstruction
   that defines the measurement* (recon randoms, mass map) is connection-side;
   *theory-model fields* are P2.
8. **Covariance ownership.** jackknife (from region_map) is connection-side;
   mock-suite + analytic covariance need cosmology/theory → lean P2, but the
   connection supplies the ingredients (n̄, window, pair counts).

---

## 8. Relationship to existing docs

- Data-modality coverage + survey specifics: [`survey_landscape_review.md`](survey_landscape_review.md) (esp. §13).
- P1 column/format gaps: [`schema_generalisation_audit.md`](schema_generalisation_audit.md).
- The contract this feeds: [`plans/2026-06-05-pillar2-definition.md`](../plans/2026-06-05-pillar2-definition.md) (Universal DataProduct).
- Cosmology rule (why no cosmology in the object): [`plans/2026-05-28-pillar1-data-combine-measure.md`](../plans/2026-05-28-pillar1-data-combine-measure.md).

**Next:** turn §6 step 1 (galaxy clustering spec) into the first executable
P1→P2 connection plan.
