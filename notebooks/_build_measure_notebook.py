#!/usr/bin/env python3
"""Build 08_measure_probes.ipynb — the P1->P2 measure layer across all probes."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
md = lambda t: nbf.v4.new_markdown_cell(t)
code = lambda t: nbf.v4.new_code_cell(t)

cells = [
    md("# Pillar 1 → Pillar 2: the `oneuniverse.measure` layer\n"
       "\n"
       "`oneuniverse.measure` turns Pillar-1 OUF data into a **cosmology-free "
       "`MeasurementSet`** — the general output format Pillar-2 estimators "
       "consume. One **Universal DataProduct** with three subtypes covers the "
       "whole probe space:\n"
       "\n"
       "| subtype | probes | from OUF geometry |\n"
       "|---|---|---|\n"
       "| **PointSet** | galaxy clustering, weak lensing, PV, SN | POINT |\n"
       "| **Sightline** | Lyα forest | SIGHTLINE |\n"
       "| **FieldMap** | CMBκ/tSZ/HI × galaxies | HEALPix/CUBE |\n"
       "\n"
       "Each connection runs the 9-step transform (select·clean·weight·"
       "randoms·window·n(z)·photo-z·region·spec). **Cosmology enters only at "
       "the Pillar-2 estimator call** — never in the data. All synthetic OUF."),
    code("%matplotlib inline\n"
         "import sys, tempfile\n"
         "from pathlib import Path\n"
         "import numpy as np, matplotlib.pyplot as plt, healpy as hp\n"
         "sys.path.insert(0, str(Path.cwd().parent / 'test'))\n"
         "from fixtures.measure_ouf import (synthetic_point_view, synthetic_shear_view,\n"
         "    synthetic_pv_view, synthetic_sn_view, synthetic_sightline_view,\n"
         "    synthetic_healpix_map)\n"
         "from oneuniverse.combine.weights import FKPWeight, ColumnWeight\n"
         "from oneuniverse import measure\n"
         "TMP = Path(tempfile.mkdtemp())\n"
         "print('measure exports:', [x for x in measure.__all__ if x.startswith('build')])"),

    md("## 1. Galaxy clustering (PointSet)\n"
       "Catalog + FKP/completeness weights + randoms + n(z) + footprint + region."),
    code("gview = synthetic_point_view(TMP, n=8000, seed=8, name='gal1')\n"
         "ms = measure.build_galaxy_clustering(gview, tracer='gal', z_range=(0.2,0.9),\n"
         "    weights=[FKPWeight(nbar=lambda z: np.full_like(z,1e-3), P0=1e4), ColumnWeight('weight_comp')],\n"
         "    nside_window=64, nside_region=8, nz_edges=np.linspace(0,1.2,30),\n"
         "    randoms='generate', n_randoms=40000, seed=1)\n"
         "ps = ms.products['gal']; cat, rnd, nz = ps.catalog, ps.randoms, ps.nz\n"
         "fig, ax = plt.subplots(1,2, figsize=(11,4))\n"
         "ax[0].scatter(rnd['ra'], rnd['dec'], s=1, c='.7'); ax[0].scatter(cat['ra'], cat['dec'], s=3)\n"
         "ax[0].set_title('footprint: data (blue) vs randoms'); ax[0].set_xlabel('RA'); ax[0].set_ylabel('Dec')\n"
         "rh,_ = np.histogram(rnd['z'], bins=nz.edges, density=True)\n"
         "ax[1].plot(nz.centers(), nz.pdf(), lw=2, label='data n(z)'); ax[1].plot(nz.centers(), rh, '--', label='randoms')\n"
         "ax[1].set_title('radial selection'); ax[1].legend(); plt.tight_layout(); plt.show()\n"
         "print('statistic:', ms.spec.statistic, '| randoms:', ps.provenance.randoms_source); ms.check_invariants()"),

    md("## 2. Weak lensing — cosmic shear (PointSet + shapes + photo-z kernel)\n"
       "Source shapes (`e1,e2` + metacal calibration) + per-object p(z) + tomographic n(z)."),
    code("sview = synthetic_shear_view(TMP, n=6000, seed=3, kind='metacal', with_pdf=True, n_tomo=3, name='src1')\n"
         "ms = measure.build_cosmic_shear(sview, tracer='src', kind='metacal',\n"
         "    tomo_column='tomo_bin', z_grid=np.linspace(0,2,61), nside_region=8)\n"
         "ps = ms.products['src']; cat = ps.catalog\n"
         "fig, ax = plt.subplots(1,2, figsize=(11,4))\n"
         "s = cat.iloc[::20]; emag=np.hypot(s['e1'],s['e2']); a=0.5*np.arctan2(s['e2'],s['e1'])\n"
         "ax[0].quiver(s['ra'], s['dec'], emag*np.cos(a), emag*np.sin(a), headwidth=0, headlength=0, pivot='mid', scale=8, width=0.003)\n"
         "ax[0].set_title('shear whiskers'); ax[0].set_xlabel('RA'); ax[0].set_ylabel('Dec')\n"
         "for b,n in sorted(ps.nz.items()): ax[1].plot(n.centers(), n.pdf(), lw=2, label=f'bin {b}')\n"
         "ax[1].set_title('tomographic n(z)'); ax[1].legend(); plt.tight_layout(); plt.show()\n"
         "print('photoz kernel:', type(ps.photoz).__name__, '| weight:', ps.provenance.weight_recipe[0]); ms.check_invariants()"),

    md("## 3. Peculiar velocities + SN Ia (PointSet + distance atoms)\n"
       "Distance indicators (v_pec, μ, σ_v) and a lazy row-correlated covariance handle."),
    code("pv = measure.build_peculiar_velocity(synthetic_pv_view(TMP, n=3000, seed=3, name='pv1'),\n"
         "    z_range=(0,0.1), nside_region=8)\n"
         "snv,_ = synthetic_sn_view(TMP, n=400, seed=4, name='sn1')\n"
         "sn = measure.build_sn_hubble(snv, z_range=(0,1.5), nside_region=4)\n"
         "pcat, scat = pv.products['pv'].catalog, sn.products['sn'].catalog\n"
         "fig, ax = plt.subplots(1,2, figsize=(12,4))\n"
         "sc=ax[0].scatter(pcat['ra'], pcat['dec'], s=5, c=pcat['v_pec'], cmap='coolwarm', vmin=-600, vmax=600)\n"
         "ax[0].set_title('peculiar velocities'); plt.colorbar(sc, ax=ax[0], label='v_pec [km/s]')\n"
         "ax[1].errorbar(scat['z'], scat['mu'], yerr=scat['mu_err'], fmt='.', ms=4, alpha=.5, elinewidth=.5)\n"
         "ax[1].set_title('SN Ia Hubble diagram'); ax[1].set_xlabel('z'); ax[1].set_ylabel('μ'); plt.tight_layout(); plt.show()\n"
         "print('PV family:', pv.spec.estimator_family, '| SN statistic:', sn.spec.statistic)"),

    md("## 4. Lyα forest (Sightline subtype)\n"
       "Per-line-of-sight δ_F(λ) + weights + continuum — the non-point geometry."),
    code("lview = synthetic_sightline_view(TMP, n_los=200, n_pix=60, seed=2, name='lya1')\n"
         "ms = measure.build_lya(lview, statistic='p1d', nside_region=8)\n"
         "sl = ms.products['lya']\n"
         "fig, ax = plt.subplots(1,2, figsize=(12,4))\n"
         "for i in range(6): ax[0].plot(sl.delta[i] + i*1.2, lw=.8)\n"
         "ax[0].set_title('6 Lyα sightlines (δ_F)'); ax[0].set_xlabel('pixel (λ)')\n"
         "sc=ax[1].scatter(sl.los['ra'], sl.los['dec'], s=10, c=sl.los['region_id'], cmap='tab20')\n"
         "ax[1].set_title(f'{sl.n_sightlines} LOS by region'); plt.colorbar(sc, ax=ax[1]); plt.tight_layout(); plt.show()\n"
         "print('subtype:', sl.kind, '| statistic:', ms.spec.statistic); ms.check_invariants()"),

    md("## 5. Map × catalog (FieldMap subtype)\n"
       "A HEALPix field (CMBκ / tSZ y) crossed with the galaxy catalog → C_ℓ."),
    code("from oneuniverse.measure.fieldmap import fieldmap_from_healpix\n"
         "vals, mask = synthetic_healpix_map(nside=64, seed=4)\n"
         "fmap = fieldmap_from_healpix(vals, mask=mask, nside=64, dataset_id='cmbk')\n"
         "ms = measure.build_map_cross(synthetic_point_view(TMP, n=6000, seed=3, name='gal2'),\n"
         "    fmap, nside_region=8, z_range=(0.1,1.0))\n"
         "gcat = ms.products['gal'].catalog\n"
         "masked = np.where(fmap.mask, fmap.values, hp.UNSEEN)\n"
         "fig = plt.figure(figsize=(10,5))\n"
         "hp.mollview(masked, nest=True, fig=fig.number, title='FieldMap κ + galaxy LOS', cmap='coolwarm', min=-3, max=3, hold=True)\n"
         "hp.projscatter(np.radians(90-gcat['dec'].to_numpy()), np.radians(gcat['ra'].to_numpy()), s=1, c='k', alpha=.4)\n"
         "plt.show()\n"
         "print('products:', set(ms.products), '| pairs:', ms.spec.pairs, '| statistic:', ms.spec.statistic); ms.check_invariants()"),

    md("## Summary\n"
       "Five probe connections, one cosmology-free contract:\n"
       "\n"
       "| probe | builder | subtype | statistic |\n"
       "|---|---|---|---|\n"
       "| galaxy clustering | `build_galaxy_clustering` | PointSet | pk_multipole |\n"
       "| cosmic shear / 3×2pt | `build_cosmic_shear`/`build_3x2pt` | PointSet | xi_pm / mixed |\n"
       "| peculiar velocity | `build_peculiar_velocity` | PointSet | velocity |\n"
       "| SN Ia | `build_sn_hubble` | PointSet | hubble |\n"
       "| Lyα forest | `build_lya` | Sightline | p1d / p3d |\n"
       "| map × catalog | `build_map_cross` | PointSet × FieldMap | cl |\n"
       "\n"
       "Every `MeasurementSet` carries data + randoms + n(z) + window + weights "
       "+ shared region map + provenance — **no cosmology**. Estimator-side "
       "adapters (flip / pycorr / picca) are a separate, later layer; the "
       "format is the standard they adopt.\n"
       "\n"
       "**Honest caveat:** all synthetic OUF — real DESI/eBOSS validation is the "
       "pending follow-up."),
]

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, str(HERE / "08_measure_probes.ipynb"))
print("wrote 08_measure_probes.ipynb", f"({len(cells)} cells)")
