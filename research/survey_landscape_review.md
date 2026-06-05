# Cosmology Survey Data Landscape — Technical Reference

> **Companion:** [`2026-06-05-survey-landscape-v2-agnostic.md`](2026-06-05-survey-landscape-v2-agnostic.md)
> is the survey-**agnostic** index over this doc (organised by observable/probe,
> + mid-2026 currency refresh + new classes: LIM, radio continuum, strong-lens
> time-delay, cosmic chronometers). This doc remains the detailed per-survey
> column reference.

**Date:** 2026-05-28
**Purpose:** Catalogue every major class of cosmology survey dataset that
`oneuniverse` should be able to ingest, with enough technical detail to
audit the schema, manifest, converter, and loader subsystems against
real-world data shapes. Companion document to
[`schema_generalisation_audit.md`](schema_generalisation_audit.md).

---

## 1. Galaxy Spectroscopic Redshift Surveys

### 1.1 SDSS Legacy / BOSS / eBOSS

- **Generations / volume.** SDSS-I/II Legacy (DR7, 2000–2008) ~930k
  galaxies + 120k QSOs; BOSS (SDSS-III, DR12, 2009–2014) ~1.5M LRGs
  + ~250k QSOs; eBOSS (SDSS-IV, DR16, 2014–2019) ~340k LRG/ELG
  + 540k QSOs + Lyα. Total ~4–5M catalog rows; spectra ~10 TB.
- **Geometry.** Point catalogue + per-fiber 1D spectrum
  (`spec-PLATE-MJD-FIBERID.fits`): flux, ivar, log-λ, masks, sky,
  model on common log-λ grid (~3600–10400 Å).
- **Row IDs.** `PLATE-MJD-FIBERID` triplet; `SPECOBJID`, `OBJID`
  (photometric), `THING_ID` (cross-DR unique).
- **Redshifts.** `Z`, `Z_ERR`, `ZWARNING`, `CLASS ∈ {GALAXY,QSO,STAR}`,
  `SUBCLASS`. Visual-inspection `Z_VI` for some QSOs. Lyα QSO adds
  `Z_PCA`, `Z_PIPE`, `Z_MGII`.
- **Spatial.** ICRS RA/Dec deg J2000. Mangle polygon footprint masks.
- **Temporal.** Single epoch per fiber; multi-MJD coadds in `SPALL`.
- **Weights.** `WEIGHT_CP`, `WEIGHT_NOZ`, `WEIGHT_SYSTOT`,
  `WEIGHT_FKP`, `WEIGHT_SEEING`, `WEIGHT_STAR`. Combined
  `w_tot = (CP + NOZ − 1)·SYSTOT·FKP`.
- **PDFs.** None for spec-z; QSO line-property posteriors in
  Redmonster/Redrock χ² grids (some VACs only).
- **Cross-IDs.** `BESTOBJID` (SDSS photometric), GAIA, AllWISE.
- **Gotchas.** Vacuum λ in BOSS+; air for Legacy. AB mags with SDSS
  u/z offsets. Fiber collisions at 62″.
- **Sub-objects.** Portsmouth/Wisconsin per-line VACs (line fluxes,
  EWs, kinematics).
- **Format/access.** FITS via SAS rsync; `specObj-dr16.fits`.

### 1.2 DESI DR1 / DR2

- **Generations.** EDR (Jun 2023), DR1 (Apr 2025) ~14M extragalactic
  z; DR2 (~2026) ~30M; Y3 PV survey appended.
- **Geometry.** Catalogue + per-target `coadd-*.fits` and
  `redrock-*.fits`. Three arms B/R/Z over 3600–9800 Å.
- **Row IDs.** `TARGETID` (int64; encodes survey+program), `RELEASE`,
  `BRICKID`, `BRICK_OBJID`, `LOCATION`, `FIBER`, `TILEID`, `PETAL_LOC`.
- **Redshifts.** Redrock `Z`, `ZERR`, `ZWARN`, `SPECTYPE`, `SUBTYPE`,
  `DELTACHI2`, `COEFF`. QSO adds `Z_MGII`, `Z_QN` (QuasarNET),
  `Z_LYA`.
- **Spatial.** ICRS deg; 7.4° tiles, 10 petals/tile, ~500 fibers/petal.
- **Temporal.** `MJD`, `NIGHT`, `EXPID`; coadded into
  `coadd-{TILEID}-{PETAL}.fits`.
- **Weights.** `WEIGHT_COMP` (fiber-assignment completeness),
  `WEIGHT_ZFAIL`, `WEIGHT_SYS` (imaging regression), `WEIGHT_FKP`.
  `BITWEIGHTS` (64 PIP realisations as int64 array).
- **PDFs.** No spec-z PDFs; parent target selection used
  DECaLS/BASS/MzLS photo-z; per-template Redrock χ² surface.
- **Cross-IDs.** `LS_ID` (Legacy bit-packed brickid+objid), GAIA
  `REF_ID`, `MORPHTYPE`.
- **Gotchas.** Vacuum λ; `MASKBITS` propagated from Legacy; `EBV`
  SFD; fiber assignment imprints angular pattern → PIP/altMTL.
- **Sub-objects.** FastSpecFit / EmLine VACs with [OII], [OIII], Hα
  fluxes, EWs, kinematics.
- **Format/access.** FITS at NERSC; partitioned
  `SURVEY/PROGRAM/TILEID/NIGHT`; `zall-pix-*.fits` is HEALPix
  NSIDE=64 nested.

### 1.3 GAMA

- DR4 (2022) ~300k r<19.8, ~98% z completeness, 286 deg² in five
  fields (G09/G12/G15/G02/G23). AAOmega 3700–8800 Å.
- IDs `CATAID`. Redshifts `Z_HELIO`, `Z_CMB`, `Z_TONRY`; `NQ≥3`
  reliable.
- Weights: per-region fiber-collision completeness maps.
- VACs: ProSpect SED fits with posterior samples; GroupFinder
  (`GroupID` + member `CATAID`s).
- Cross-IDs: SDSS, UKIDSS, VIKING, GALEX, WISE.

### 1.4 2dFGRS / 6dFGS / WiggleZ

- **2dFGRS** Final (2003): ~221k z, b_J<19.45, NGP+SGP strips. `Z`,
  `QUALITY` (Q≥3 reliable).
- **6dFGS** DR3: ~125k K-selected z + 9k 6dFGSv FP peculiar
  velocities. IDs `TARGETNAME`, `_6dFGSID`; PV: `cz`, `sigma_eta`,
  `logDist`.
- **WiggleZ**: 225k UV+optical ELGs, 1000 deg² in 7 narrow regions,
  IDs `WIGZ_ID`.

### 1.5 VIPERS / zCOSMOS / DEEP2-3

- **VIPERS PDR2**: 90k i<22.5, 0.5<z<1.2, VIMOS. `id_IAU`, `zspec`,
  `zflg` (2–4 reliable).
- **zCOSMOS bright DR3 + deep**: 20k bright + 10k deep in COSMOS.
  `OBJ_ID`, 4-digit `zflag` (Lilly scheme).
- **DEEP2 DR4** (Keck/DEIMOS): 50k z<1.4, 6500–9100 Å,
  `ZQUALITY≥3`. DEEP3 extends EGS.

### 1.6 PFS / 4MOST

- **PFS** Subaru (cosmology runs 2025+): 2400 fibers, 380–1260 nm,
  expected ~4M cosmology z. IDs follow HSC `objId`.
- **4MOST** (2025+): BG/LRG/QSO/Lyα CRS + WAVES + S8; ~30M z over
  5 yr. ESO archive FITS; per-object `OBSID`, `FILEID`.

### 1.7 Euclid NISP grism / Roman HLSS grism

- **Euclid** (launched 2023, DR1 ~2026): NISP-S slitless 1.25–1.85 µm,
  ~25M Hα/[OIII] at 0.9<z<1.8 over 14000 deg². `OBJECT_ID`,
  `SOURCE_ID`; `Z_SPE`, `Z_SPE_ERR`, `FLAG_SPE`. 2D + 1D spectra.
- **Roman** HLSS (~2027 launch): slitless 1.0–1.93 µm, ~10M Hα at
  1<z<2 over 2000 deg². IRSA/MAST archive aligned with Euclid.

---

## 2. Photometric Redshift Surveys

### 2.1 KiDS

- KiDS-1000 (DR4, 2020) ~21M over 1006 deg²; DR5 (2023) ~32M over
  1347 deg² with VIKING NIR (9-band ugri+ZYJHK).
- IDs `KIDS_ID` = `KIDS_TILE+SeqNr`; `THELI_NAME` for shear.
- Redshifts: BPZ `Z_B`, `ODDS`; SOM-z per tomographic bin; n(z) as
  binned histogram per bin (DR4) or per SOM-cell (DR5).
- Spatial: ICRS deg; ~1 deg² tiles `KIDS_RAxx.x_DECyy.y`.
- **Weights / shear.** lensfit `weight`, c-terms `c1,c2`,
  multiplicative bias `m` per tomographic bin. e1/e2 in pixel
  coords → rotate to celestial.
- Cross-IDs: GAIA, 2MASS, AllWISE.
- Gotchas: `MAG_GAAP_*` for photo-z, `MAG_AUTO` for selection.
- Format: FITS via ESO archive + KiDS site.

### 2.2 DES

- DR2 (2021) ~691M over 5000 deg²; Y3 ~100M shape-selected; Y6
  (2024–25) final. grizY.
- IDs `COADD_OBJECT_ID` int64, `TILENAME = DESxxxx-yyyy`.
- Redshifts: Y3 BPZ + DNF point + SOM-based n(z); Y6 `SOMPZ`.
- **Shear.** Metacalibration responses `R11/R22/R12/R21` + selection
  responses `R_S`; `e1,e2` in TAN. Angular systematics maps
  (depth, seeing, airmass, sky brightness) HEALPix NSIDE=4096.
- PDFs: stacked BPZ in some VACs; SOMPZ uses cell assignment +
  per-cell n(z).
- Cross-IDs: GAIA DR3, VHS NIR, LS DR9/10.
- Gotchas: Metacal `e1/e2` already mean-zero corrected;
  multiplicative bias residual <5e-3; `FLAGS_GOLD` bitmask.
- Sub-objects: Redmapper VAC `MEM_MATCH_ID` → `ID_member`,
  `P_MEMBER`.
- Format: FITS + DB (NOIRLab Astro Data Lab + DES Release Server).

### 2.3 HSC SSP

- PDR3 (2022), PDR4 (2024); Wide ~1400 deg² to i~26.
- IDs `object_id` int64.
- Redshifts: Mizuki, DEmP, FRANKEN-Z, NNPZ, Ephor; per-object
  gridded p(z) on z grid (~101 bins).
- Tracts (1.7° × 1.7°) and patches (4096² pix) per LSST stack.
- Shear: HSM `ishape_hsm_regauss_e1/e2` + weight; Y3 metadetect VACs.
- Cross-IDs: GAIA, LS, Pan-STARRS.
- Format: FITS + parquet; SQL via HSC DB.

### 2.4 LSST / Rubin

- Commissioning 2025; DP0.2 available; DR1 ~2027; 10-yr ~30B objects.
- Tables: `Object`, `Source`, `ForcedSource`, `DiaObject`,
  `DiaSource`. IDs `objectId`, `sourceId`, `diaObjectId`,
  `diaSourceId`, `ccdVisitId` (all int64).
- Redshifts: RAIL/PZ photo-z; PDFs via `qp` (interp/quant/mixmod/
  sample); per-bin n(z) from SOM.
- Spatial: sphgeom partitioning into HTM + HEALPix (Butler NSIDE=32
  nest tiles). Tract/Patch hierarchy.
- Temporal: per-visit `ForcedSource`/`Source` with `expMidptMjd`;
  alerts in Avro.
- Weights: per-band metadetect `e1/e2` + responses; depth/PSF
  size/ellipticity maps HEALPix NSIDE≥4096.
- Cross-IDs: GAIA `gaia_dr3_source_id`; `objectId` stable across DRs.
- Gotchas: fluxes in nJy AB (`mag = -2.5 log10(F) + 31.4`);
  `detect_isPrimary`; deblender tree via `parentObjectId`.
- Sub-objects: deblender `parentObjectId → objectId`;
  `DiaObject → DiaSource` time series.
- Format: Parquet via Butler; alerts Avro; tract/patch partitioning.

### 2.5 Euclid VIS+NISP photometric

- DR1 (~2026): 14000 deg² over 6 yr; VIS (single broad I, 530–920 nm)
  + NISP Y, J, H; ~1.5B objects.
- IDs `OBJECT_ID`. Photo-z: NNPZ per-object PDF (~601-bin gridded).
- Shear: `gamma1, gamma2` with KSB+ and lensfit; weights,
  sensitivities. Per-pointing PSF + depth maps as HEALPix.

### 2.6 UNIONS / J-PAS / J-PLUS / SHARKS

- **UNIONS**: CFIS u + Pan-STARRS r/i + WISHES z over ~4800 deg² N;
  ~100M shapes (ShapePipe); per-tile FITS, `unique_id`.
- **J-PAS**: 56 narrow filters + 4 broad over 8000 deg² (observing
  2023+); ~300M photo-z from per-object pseudo-spectrum of 56 flux
  points. IDs `TILE_ID + NUMBER`.
- **J-PLUS DR3**: 12-band, 3000 deg², ~50M sources.
- **SHARKS**: VISTA Ks survey over ~1300 deg² to Ks~22.7;
  complements KiDS/VIKING NIR.

### 2.7 COSMOS2020 / COSMOS-Web

- COSMOS2020 (Weaver+ 2022): 1.7M sources, 2 deg², CLASSIC +
  FARMER catalogs, 35 bands. `ID`, `lp_zBEST`, `lp_zPDF_l68/u68`,
  `ez_z_phot`, `ez_z_phot_chi2`; per-object 1000-bin gridded `P(z)`
  in separate FITS extension.
- COSMOS-Web (JWST, 2023–24): NIRCam 4-band, 0.5M sources, 0.54
  deg².

---

## 3. Lyman-α Forest Surveys

### 3.1 eBOSS DR16Q

- 750k QSOs z>2.1 (BOSS+eBOSS reanalysed); ~500k usable for Lyα.
- **Geometry.** Per-LOS pixel arrays: `loglam`, `delta = F/<F> − 1`,
  `weight` (or `ivar`), `cont`, `mask` over 3600–7235 Å subset.
  picca outputs `delta-*.fits.gz` partitioned by HEALPix NSIDE=16
  NEST, multiple HDUs (one per LOS) or unified binary table.
- IDs `THING_ID`, `PLATE-MJD-FIBERID`.
- Redshifts: `Z_PCA`, `Z_PIPE`, `Z_VI`, `Z_LYAWG`.
- Weights: pixel `IVAR` + continuum-fitting `WEIGHT`;
  `WEIGHT_FKP`, `WEIGHT_NOZ` for clustering.
- Gotchas: vacuum log-binned λ; pixel mask bitfield (sky/BAL/DLA).
- Sub-objects: DLA VAC (`Z_DLA`, `NHI`, `DLA_CONFIDENCE`); BAL VAC
  (`BI_CIV`, `BALPROB`).

### 3.2 DESI Lyα DR1 / DR2

- DR1 (Apr 2025) ~420k z>2.1 QSOs for Lyα BAO; DR2 (~2026) ~1.2M.
- Same picca format: `delta-{HPX}.fits.gz`, HEALPix NSIDE=16 NEST,
  per-LOS `LOGLAM, DELTA, WEIGHT, CONT, MEANSNR, BLINDING`.
- IDs `TARGETID`; `LAST_NIGHT`, `FIRST_NIGHT`.
- Redshifts: `Z`, `Z_QN`, `Z_LYA` (QSO catalog).
- Sub-objects: DLA catalog (`Z_DLA`, `NHI`, `DLA_ID`), BAL catalog
  (`AI_CIV`, `BI_CIV`); per-LOS pixel masks.
- Gotchas: blinding factor on deltas; `BLINDING` header keyword
  `∈ {none, desi_y1, desi_y3}`.
- Companion: `attributes.fits.gz` with mean continuum + noise
  corrections.

---

## 4. Peculiar Velocity Surveys

- **6dFGSv** (~8900 FP distances, z<0.055): `cz_helio`, `cz_CMB`,
  `eta = log10(Re) − a·log(σ) − b·<µ>`, `eta_err`, `logDist`,
  `logDist_err`. IDs `_6dFGS`. Group catalog (Magoulas+).
- **2MTF** (~2000 TF distances): `PGC`, `2MASXJ`; `cz`, `logW`,
  `m_T,J,H,K`.
- **CosmicFlows-3 / CF4** (CF3 ~18k; CF4 ~56k): `DM`, `eDM`, `Vhel`,
  `Vcmb`, `Vmod`, `Vls`; `DM_method` enum (TF, FP, SBF, TRGB, Cepheid,
  SN); mixed group-averaged + singles. EDD database.
- **SDSS PV** (Howlett+ 2022) ~34k FP, z<0.1.
- **DESI FP / TF PV** (Y5 target ~200k).
- **ZTF SN Ia / Foundation** ~3000 low-z SNe Ia (SALT2 `x1`, `c`,
  `mu`, `mu_err`).
- **JWST PV** (SH0ES-style anchors; ultra-precise `mu, mu_err`).

---

## 5. Supernova Surveys

- **Pantheon+ / SH0ES**: ~1700 SNe Ia, 0.001<z<2.3. Columns `zHD`,
  `zHEL`, `mB_corr`, `x1`, `c`, `mB_err`, `MU_SH0ES`. Full
  1701×1701 covariance matrix in `.cov` file. IDs `CID`. Format
  ASCII + .cov.
- **Union3 / UNITY**: ~2087 SNe; UNITY3 hierarchical posterior; HDF5
  with per-SN chain samples.
- **DES SN5YR**: 1635 photometric + 194 spec Ia. `CID`, `IAUC`,
  `zHD`, `mB`, full systematic covariance.
- **ZTF BTS** ~3000 spec-confirmed; alerts Avro; final FITS.
- **LSST SN / Roman SN** (future): per-SN SALT3 fit posteriors;
  per-SN light curve in `ForcedSource` style (per band per visit).

---

## 6. CMB & Secondary Anisotropies (Ancillary Tracers)

- **Planck PSZ2** (1653 SZ): `NAME`, `GLON`, `GLAT`, `SNR`, `MSZ`,
  `Z`, `Z_SOURCE`, `VALIDATION`.
- **ACT DR6**: lensing κ HEALPix NSIDE=4096 + mask + noise sims;
  DR5/6 cluster catalog ~4000 SZ (`name`, `RADeg`, `decDeg`,
  `redshift`, `redshiftErr`, `M500c`, `SNR`).
- **SPT-3G**: SPT-SZ + SPTpol + SPT-3G catalogs (`xi`, `M500c`, `z`).
- **eROSITA-DE DR1 clusters**: eRASS1 ~12k (`NAME`, `RA`, `DEC`,
  `Z_BEST`, `Z_ERR`, `M500c`, `LX`, `T_X`).
- **Planck NPIPE PCCS2**: per-frequency compact-source catalog (30–
  857 GHz).
- CMB pixel maps (Planck NPIPE T/Q/U, ACT, SPT) HEALPix NSIDE
  2048–4096 RING — not catalog data; needed only as cross-correlation
  tracers (clusters + κ).

---

## 7. HI & Radio Surveys

- **HIPASS BGC** (2004) ~4315 HI: `HIPASS`, `RAdeg`, `Decdeg`,
  `RVmom`, `W50`, `SHI`, `D_Mpc`.
- **ALFALFA α.100** (2018) ~31500 HI: `AGCNr`, `Vhelio`, `W50`,
  `HIflux`, `SNR`, `OCcode`.
- **WALLABY** PDR1 (ASKAP, 2022) + DR (2024+): 600–2000 HI per pilot
  field + HI cubes + moment maps. `name`, `freq`, `w20`, `w50`,
  `flux_int`, `kin_pa`. Catalog FITS + 3D HI cubes (RA, Dec, freq).
- **MeerKAT MIGHTEE**: continuum + HI emission, ~few ×10⁵ continuum,
  per-LOS HI spectra in deep fields.
- **SKA1-MID intensity mapping** (future): HI temperature cubes (RA,
  Dec, freq); no per-galaxy catalog.
- **CHIME 21cm / HERA**: intensity-mapping cubes + delay-spectrum
  visibility products. Not point catalogs.

---

## 8. Gravitational-Wave Standard Sirens

- **GWTC-3 / GWTC-4**: ~90 confident events (GWTC-3, O3); O4a
  growing.
- Per event: posterior samples HDF5 (`mass_1_source`, `mass_2_source`,
  `luminosity_distance`, `ra`, `dec`, `redshift`); HEALPix
  probability map (`*.fits.gz` multi-order MOC or fixed NSIDE RING)
  with `PROB`, `DISTMU`, `DISTSIGMA`, `DISTNORM` columns.
- IDs `event_name` (`GW230529_181500`).
- Cross-IDs: GraceDB ID, BAYESTAR vs LALInference labels.
- Gotchas: skymap is multi-order HEALPix (UNIQ column) under MOC;
  need NUNIQ ↔ NSIDE/IPIX conversion.

---

## 9. X-ray / Gamma-ray / Multi-wavelength

- **eROSITA-DE DR1**: ~1M soft X-ray, ~12k clusters, ~600k AGN.
  `IAUNAME`, `DETUID`, `ML_FLUX_0`, `EXT`, `EXT_LIKE`, `ML_BKG_0`.
  Catalog FITS + per-source spectra/lightcurves.
- **Chandra Source Catalog 2.1**: ~315k unique sources; aperture
  photometry + variability stats; `name`, `obsid`, `region_id`.
- **Fermi 4FGL-DR4**: ~7000 γ sources; spectral fit (PL, LP, PLEC);
  `Source_Name`, `RAJ2000`, `DEJ2000`, `Flux1000`,
  `Variability_Index`.

---

## 10. Time-Domain & Alert Streams

- **ZTF alerts**: ~10⁶ alerts/night; Avro schema with embedded image
  cutouts (science/template/diff as FITS bytes). Per-alert `candid`,
  `objectId`, `ra`, `dec`, `jd`, `magpsf`, `sigmapsf`, `diffmaglim`,
  `prv_candidates` history array.
- **LSST broker outputs** (Alerce, ANTARES, Fink, Lasair):
  classifier probabilities per object per class (`p_AGN`,
  `p_SN_Ia`, `p_KN`); stamp classifications; per-broker schemas.
- **TNS** ~150k transients: `objname`, `RA`, `Dec`, `redshift`,
  `discoverydate`, `type`, `internal_names`.

---

## 11. Reference / Value-Added Imaging Catalogs

- **GAIA DR3**: 1.8B sources. `source_id` int64 encodes HEALPix
  NSIDE=4096 in upper bits. Positions, parallax, PMRA/PMDec, BP/RP
  photometry, BP/RP low-res spectra (continuous representation),
  GSP-Phot/Spec stellar params, ~220M XP spectra, ~33M RVS spectra.
  Archive ADQL + bulk parquet/HDF5/CSV per HEALPix chunk. Positions
  ICRS epoch 2016.0 — propagate PM before joining to other epochs.
- **Legacy Imaging (LS DR9/DR10)**: BASS+MzLS (N) + DECaLS (S) +
  DES (S), 20000 deg² grz. `RELEASE`, `BRICKID`, `OBJID`, `LS_ID`
  (bit-packed), Tractor model photometry, `MASKBITS`, `FITBITS`,
  morphology `TYPE ∈ {PSF, REX, EXP, DEV, SER}`. Per-brick FITS +
  sweep files partitioned by sky region.
- **Pan-STARRS DR2**: ~3B sources; `objID`, `objName`; stack + mean
  catalogs + per-detection.
- **AllWISE / unWISE / CatWISE2020**: ~750M WISE; W1–W4;
  `designation`, `source_id`; unWISE deeper.
- **2MASS PSC/XSC**: 471M point + 1.6M extended; `j_m`, `h_m`, `k_m`.
- **UKIDSS / VIKING**: NIR Y/J/H/K via WSA; 64-bit `sourceID`
  encoding survey+frame+seqnum.

---

## 12. Simulated Mocks

- **UNIT**: N-body, 1 Gpc/h boxes, 4096³; ROCKSTAR ASCII halos +
  Gadget snapshots.
- **AbacusSummit**: 97 cosmologies; boxes 250 Mpc/h → 7.5 Gpc/h;
  CompaSO halos asdf+parquet; particle subsample h5/asdf; HOD
  mocks per z slice.
- **Outer Rim / LastJourney** (HACC): GenericIO particles; ASCII
  halos.
- **Quijote**: 44000 N-body for Fisher; halo + matter grids;
  FITS/HDF5/binary.
- **MillenniumTNG**: IllustrisTNG-style hydro; HDF5 snapshots +
  Subfind catalogs; subhalo trees.
- **Buzzard**: DES-like full-sky lightcone; HEALPix tiled; FITS with
  `RA`, `DEC`, `Z`, `MAG_*`, `SIZE`, `EPSILON1`, `EPSILON2`.
- **Euclid Flagship2**: 2T-particle full-sky lightcone galaxy
  catalog HDF5 + parquet; photometry, shear, host halo, line fluxes;
  HEALPix partitioned. Uchuu, Skybot, Skybot4 similar.

---

## 13. Cross-cutting Modality Inventory

Modalities the OUF schema must accommodate, ordered by how
disruptive they are to the current Phase 1–15 model:

1. **Variable-length per-row spectra.** SDSS/BOSS/DESI/Euclid/Roman
   1D spectra of varying length. Need `LargeList<float32>` + length
   column, or fixed-pad with mask. Lyα picca delta sidecars already
   partition per HEALPix LOS file → ingest as sub-object sidecar
   keyed by `galaxy_id`.
2. **Variable filter-set photometry.** 5 ugriz vs 9 KiDS+VIKING vs
   12 J-PLUS vs 56 J-PAS. Generic
   `List<struct{filter_id, flux, flux_err, zeropoint, unit}>` or
   filter-set registry per partition.
3. **PDF parameterisation polymorphism.** Gridded p(z) (KiDS-BPZ,
   HSC, COSMOS2020, Euclid NNPZ), quantile (LSST RAIL `qp`), mixture
   (`qp` mixmod), samples (DES SOMPZ stacks). Phase 10 PdfSpec
   covers three; alignment with `qp` (Malz+) is non-trivial.
4. **Tomographic n(z) as data.** Per-bin n(z) (KiDS-1000, DES-Y3,
   HSC-Y3) is bin-level, not row-level. New dataset kind: "ensemble
   n(z) per tomographic bin" + per-row bin assignment column.
5. **HEALPix probability map as a row payload.** GW skymaps are
   multi-order MOC HEALPix per event with per-pixel
   `DISTMU/DISTSIGMA/DISTNORM`. Need row-level map storage + NUNIQ
   decoding.
6. **Pixel / grid datasets alongside catalogs.** CMB κ, lensing,
   intensity-mapping cubes, systematics maps (depth, seeing, PSF
   size), DESI imaging weights. Map kind with optional ν/z axis,
   partition NSIDE may differ from row partition.
7. **Multi-axis cubes.** HI cubes (RA, Dec, freq), 21cm cubes, IFU
   cubes (MaNGA, SAMI, MUSE GTO). N-D arrays as cell payloads with
   WCS/axis metadata.
8. **Bitemporal at row granularity.** GAIA epoch 2016.0 vs DESI MJD;
   per-visit alert observation time + ingestion time; DR validity.
   TemporalSpec must distinguish `t_obs`, `epoch_of_position`,
   `valid_from/to`, `ingested_at`.
9. **Multiple redshift columns per row.** `z_helio`, `z_CMB`,
   `z_TONRY/Vmod`, `z_pipe`, `z_VI`, `z_QN`, `z_PCA`, `z_LYA`.
   Single `z, z_type, z_err` cannot represent the disagreement
   structure used in Lyα/QSO analyses. Use `z_alternatives`
   list-of-struct or sidecar.
10. **Expanded weight families.** Metacal `R11/R22/R12/R21` +
    `R_S`; lensfit `weight + c1/c2/m`; PIP `BITWEIGHTS` int64
    arrays (~64 bits); Roman/Rubin detection-prob HEALPix maps;
    fiber-collision PIP realisations; systematics-template weights.
    Phase 11 needs arbitrary named weights + a weight-kind enum +
    bitfield arrays.
11. **Diverse sub-object hierarchies.** Cluster→members
    (redmapper/eROSITA/ACT); QSO→DLAs/BALs; galaxy→emission lines
    (FastSpecFit/GAMA ProSpect); GW event→posterior samples;
    transient→per-visit detections; deblender parent→child (Rubin
    `parentObjectId`); halo→subhalo (TNG/Abacus). Phase 8 needs a
    generic "hierarchy edge with role" model, not just "cluster
    membership".
12. **Heterogeneous ID conventions.** Bit-packed spatial IDs
    (GAIA `source_id`, `LS_ID`, Rubin `objectId`), tuples
    (`PLATE-MJD-FIBERID`), composite strings (`KIDS_TILE+SeqNr`),
    MJD-derived (`candid` ZTF). ONEUID must accept arbitrary tuples
    or byte payloads — int64 only is too narrow.
13. **Coordinate epochs / frames.** ICRS J2000, ICRS epoch 2016.0
    (GAIA), galactic (Planck), ecliptic (Euclid focal plane).
    Manifest must record source frame + epoch + PM availability.
14. **λ conventions.** Vacuum (BOSS+/DESI/Euclid) vs air (Legacy
    SDSS, some VIPERS). Spectrum payload metadata needs
    `air_or_vacuum` + rest-frame correction state.
15. **Distance-modulus + correlated covariance.** Pantheon+ 1701×1701
    covariance is row-correlated. Row-level "distance estimator"
    payload + `cov_id` pointing to global covariance store
    (per-survey or per-VAC). `z_type=pv` is insufficient.
16. **Selection masks at multiple resolutions.** Mangle polygons
    (SDSS), HEALPix NSIDE=4096–32768 (DES, Rubin), per-tile/petal
    completeness (DESI), per-pointing PSF maps. Row NSIDE=32 OK for
    catalogs; masks routinely sit at NSIDE≥4096. Decouple mask
    resolution from row partition NSIDE.
17. **Per-row probability classifications beyond redshift.**
    Star/galaxy/QSO probabilities (DESI `SPECTYPE/SUBTYPE`,
    pipeline `P_*`), transient classifier outputs (`p_SN_Ia`,
    `p_AGN`, `p_KN`), cluster membership `P_MEMBER`. Generic per-row
    `probabilities` struct or list of class-probability pairs.
18. **Alert payloads as a first-class shape.** ZTF/Rubin alerts
    carry embedded image cutouts (FITS bytes) + history arrays.
    Alert-row kind (lower priority; defer until Pillar 1 stable for
    static catalogs).

These observations imply the OUF schema needs (a) variable / list /
N-D array column support beyond fixed-size float32, (b) decoupled
mask/map partitioning, (c) bitemporal columns at row granularity,
(d) richer redshift/distance payload, (e) hierarchical link sidecars
with role enums, (f) PDF representation polymorphism aligned with
`qp`, (g) explicit unit + frame + epoch + λ-convention metadata
per column.

See [`schema_generalisation_audit.md`](schema_generalisation_audit.md)
for the file-by-file mapping of these requirements onto the current
codebase and a prioritised set of API changes.
