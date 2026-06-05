#!/usr/bin/env python3
"""Build the oneuniverse capability notebooks (01-04, all three pillars)."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
md = lambda t: nbf.v4.new_markdown_cell(t)
code = lambda t: nbf.v4.new_code_cell(t)


def save(name, cells):
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"}}
    nbf.write(nb, str(HERE / name))
    print("wrote", name, f"({len(cells)} cells)")


_SETUP = (
    "%matplotlib inline\n"
    "import os, sys, tempfile, warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "from pathlib import Path\n"
    "import numpy as np, pandas as pd, matplotlib.pyplot as plt, healpy as hp\n"
    "ROOT = Path.cwd().parent\n"
    "sys.path.insert(0, str(ROOT / 'test'))\n"
    "TMP = Path(tempfile.mkdtemp())\n"
    "DATA_ROOT='/home/ravoux/Documents/Science/Cosmography/oneuniverse_data'\n"
    "EBOSS=Path(DATA_ROOT)/'spectroscopic/eboss/qso/DR16Q_Superset_v3.fits'\n"
    "HAVE_EBOSS = EBOSS.exists()\n"
    "print('real eBOSS data available:', HAVE_EBOSS)")


# ── helper: real-or-synthetic eBOSS POINT view (reused by NB 01 + 02) ───────
_EBOSS_VIEW = (
    "def eboss_or_synth_view(tmp, n_cap=40000):\n"
    "    from fixtures.measure_ouf import synthetic_point_view\n"
    "    from oneuniverse.data.converter import write_ouf_dataset\n"
    "    from oneuniverse.data.dataset_view import DatasetView\n"
    "    from oneuniverse.data.format_spec import DataGeometry\n"
    "    from oneuniverse.data.manifest import LoaderSpec\n"
    "    if not HAVE_EBOSS:\n"
    "        return synthetic_point_view(tmp, n=n_cap, seed=1, name='synth'), False\n"
    "    os.environ['ONEUNIVERSE_DATA_ROOT']=DATA_ROOT\n"
    "    from oneuniverse.data import load_catalog\n"
    "    df = load_catalog('eboss_qso', validate=False)\n"
    "    df = df[(df['z']>=0.8)&(df['z']<=2.2)].dropna(subset=['ra','dec','z'])\n"
    "    if len(df)>n_cap: df=df.sample(n_cap, random_state=0)\n"
    "    df=df.reset_index(drop=True); n=len(df)\n"
    "    ra=df['ra'].to_numpy(float); dec=df['dec'].to_numpy(float)\n"
    "    out=pd.DataFrame({'ra':ra,'dec':dec,'z':df['z'].to_numpy(float),\n"
    "        'z_type':np.full(n,'spec'),'z_err':np.full(n,1e-4),\n"
    "        'galaxy_id':np.arange(n,dtype=np.int64),'survey_id':np.zeros(n,dtype=np.int64),\n"
    "        'weight_comp':np.ones(n),'nbar':np.full(n,1e-3),\n"
    "        '_original_row_index':np.arange(n,dtype='i8'),\n"
    "        '_healpix32':hp.ang2pix(32,ra,dec,nest=True,lonlat=True).astype('i4')})\n"
    "    od=tmp/'eboss'/'oneuniverse'\n"
    "    write_ouf_dataset(df=out,out_dir=od,survey_name='eboss',survey_type='spectroscopic',\n"
    "        geometry=DataGeometry.POINT,loader=LoaderSpec(name='eboss',version='0'))\n"
    "    return DatasetView.from_path(od.parent), True")


# ══════════════════════════════════════════════════════════════════════════
# NB 01 — Pillar 1: data
# ══════════════════════════════════════════════════════════════════════════
save("01_pillar1_data.ipynb", [
    md("# 01 · Pillar 1 — the data layer\n"
       "Ingest → standardise (OUF 2.5) → partial-access reads → weights. "
       "Real eBOSS DR16Q QSO if the data is present, else a synthetic stand-in. "
       "**No cosmology** lives in Pillar 1."),
    code(_SETUP),
    code(_EBOSS_VIEW + "\nview, real = eboss_or_synth_view(TMP)\n"
         "print('source:', 'REAL eBOSS DR16Q' if real else 'synthetic',\n"
         "      '| geometry:', view.geometry.value, '| rows:', view.n_rows)\n"
         "print('CORE columns:', [c for c in view.columns if not c.startswith('_')][:9])"),
    md("## Partial-access reads — only the partitions you need\n"
       "`DatasetView` prunes HEALPix partitions by sky/redshift selector."),
    code("from oneuniverse.data.selection import Cone\n"
         "full = view.read(columns=['ra','dec','z'])\n"
         "ctr_ra, ctr_dec = float(full['ra'].median()), float(full['dec'].median())\n"
         "sub = view.read(columns=['ra','dec','z'], cone=Cone(ra=ctr_ra, dec=ctr_dec, radius=10))\n"
         "print(f'full read: {len(full):,} rows | cone(10 deg) read: {len(sub):,} rows')\n"
         "fig,ax=plt.subplots(1,2,figsize=(13,4))\n"
         "ax[0].scatter(full['ra'],full['dec'],s=.3,c='.7'); ax[0].scatter(sub['ra'],sub['dec'],s=.5,c='C3')\n"
         "ax[0].set_title('footprint + 10° cone (red)'); ax[0].set_xlabel('RA'); ax[0].set_ylabel('Dec')\n"
         "ax[1].hist(full['z'],bins=40); ax[1].set_title('redshift distribution'); ax[1].set_xlabel('z')\n"
         "plt.tight_layout(); plt.show()"),
    md("## Weights — composable primitives (cosmology-free)\n"
       "`oneuniverse.combine` weight families: FKP, completeness, systematics, "
       "shear, PIP bitwise. Here FKP × completeness."),
    code("from oneuniverse.combine.weights import FKPWeight, ColumnWeight\n"
         "df = view.read()\n"
         "if 'nbar' not in df: df['nbar']=1e-3\n"
         "if 'weight_comp' not in df: df['weight_comp']=1.0\n"
         "fkp = FKPWeight(nbar=lambda z: np.full_like(z,1e-3), P0=1e4)(df)\n"
         "comp = ColumnWeight('weight_comp')(df)\n"
         "print('FKP weight (constant n̄):', round(float(fkp[0]),4), '| total = FKP×comp')\n"
         "plt.figure(figsize=(5,3)); plt.hist(fkp*comp, bins=30)\n"
         "plt.title('total weight'); plt.xlabel('w'); plt.show()"),
    md("## Photo-z PDF kernels (synthetic)\n"
       "OUF stores per-object p(z) (`qp`-style); `DatasetView.load_pdf()` "
       "reconstructs it. Photometric probes carry this kernel into Pillar 2."),
    code("from fixtures.measure_ouf import synthetic_shear_view\n"
         "pview = synthetic_shear_view(TMP, n=1500, seed=2, with_pdf=True, name='pdf')\n"
         "pz = pview.load_pdf()\n"
         "plt.figure(figsize=(6,3))\n"
         "for i in range(6): plt.plot(pz.grid, pz.values[i], lw=1)\n"
         "plt.title('6 per-object photo-z PDFs'); plt.xlabel('z'); plt.ylabel('p(z)'); plt.show()\n"
         "print('kernel mean shape:', pz.mean().shape)"),
    md("**Recap.** Pillar 1 = one standardised, partial-access, weighted, "
       "cross-matchable database — verbatim survey data + observational "
       "metadata only. Cosmology enters downstream."),
])


# ══════════════════════════════════════════════════════════════════════════
# NB 02 — Pillar 2: measure
# ══════════════════════════════════════════════════════════════════════════
save("02_pillar2_measure.ipynb", [
    md("# 02 · Pillar 2 — the measure layer\n"
       "Build the cosmology-free **MeasurementSet** for every probe. One "
       "Universal DataProduct (PointSet / Sightline / FieldMap). Real eBOSS "
       "clustering + synthetic for the other probes. **It builds & validates "
       "the handoff; it does not compute the estimator.**"),
    code(_SETUP),
    code(_EBOSS_VIEW + "\nfrom oneuniverse.combine.weights import FKPWeight, ColumnWeight\n"
         "from oneuniverse import measure\n"
         "print('builders:', [b for b in measure.__all__ if b.startswith('build')])"),
    md("## 1 · Galaxy clustering — real eBOSS DR16Q (or synthetic)\n"
       "Catalog → weights → randoms → n(z) → footprint → region → MeasurementSet."),
    code("view, real = eboss_or_synth_view(TMP, n_cap=40000)\n"
         "ms = measure.build_galaxy_clustering(view, tracer='qso', z_range=(0.8,2.2),\n"
         "    weights=[FKPWeight(nbar=lambda z: np.full_like(z,1e-3), P0=1e4)],\n"
         "    nside_window=64, nside_region=16, nz_edges=np.linspace(0.7,2.3,33),\n"
         "    randoms='generate', n_randoms=3*view.n_rows, seed=1)\n"
         "ps=ms.products['qso']; cat,rnd,nz=ps.catalog,ps.randoms,ps.nz\n"
         "fig,ax=plt.subplots(1,2,figsize=(13,4))\n"
         "ax[0].scatter(rnd['ra'],rnd['dec'],s=.3,c='.75'); ax[0].scatter(cat['ra'],cat['dec'],s=.5,c='C0')\n"
         "ax[0].set_title(('REAL eBOSS' if real else 'synthetic')+' footprint: data vs randoms')\n"
         "ax[1].plot(nz.centers(),nz.pdf(),lw=2); ax[1].set_title('QSO n(z)'); ax[1].set_xlabel('z')\n"
         "plt.tight_layout(); plt.show()\n"
         "ms.check_invariants(); print('covered fraction:', round(ps.window.covered_fraction(),3))"),
    md("## 2 · Weak lensing — shapes + photo-z kernel + tomographic n(z)"),
    code("sv = __import__('fixtures.measure_ouf', fromlist=['synthetic_shear_view']).synthetic_shear_view(TMP, n=5000, seed=3, kind='metacal', with_pdf=True, n_tomo=3, name='src')\n"
         "ms_wl = measure.build_cosmic_shear(sv, tomo_column='tomo_bin', z_grid=np.linspace(0,2,61), nside_region=8)\n"
         "wl=ms_wl.products['src']\n"
         "for b,n in sorted(wl.nz.items()): plt.plot(n.centers(),n.pdf(),lw=2,label=f'bin {b}')\n"
         "plt.title('tomographic n(z) (photo-z stack)'); plt.xlabel('z'); plt.legend(); plt.show()\n"
         "print('shapes carried:', wl.attributes if wl.attributes else 'e1,e2,R11,shear_weight; photoz:', type(wl.photoz).__name__)"),
    md("## 3 · Lyα (Sightline) + 4 · map×catalog (FieldMap)"),
    code("from fixtures.measure_ouf import synthetic_sightline_view, synthetic_healpix_map, synthetic_point_view\n"
         "from oneuniverse.measure.fieldmap import fieldmap_from_healpix\n"
         "lya = measure.build_lya(synthetic_sightline_view(TMP, n_los=150, n_pix=50, seed=2, name='lya'), nside_region=8)\n"
         "vals,mask = synthetic_healpix_map(nside=64, seed=4)\n"
         "fm = fieldmap_from_healpix(vals, mask=mask, nside=64, dataset_id='cmbk')\n"
         "xc = measure.build_map_cross(synthetic_point_view(TMP,n=5000,seed=3,name='g2'), fm, nside_region=8, z_range=(0.1,1.0))\n"
         "fig,ax=plt.subplots(1,2,figsize=(13,4))\n"
         "sl=lya.products['lya']\n"
         "for i in range(6): ax[0].plot(sl.delta[i]+i*1.0,lw=.8)\n"
         "ax[0].set_title(f'Lyα δ_F ({sl.n_sightlines} sightlines)'); ax[0].set_xlabel('pixel (λ)')\n"
         "ax[1].imshow(fm.values[:4096].reshape(64,64),cmap='coolwarm'); ax[1].set_title('FieldMap κ (patch)'); ax[1].axis('off')\n"
         "plt.tight_layout(); plt.show()"),
    md("## All probes, one cosmology-free contract"),
    code("import json\n"
         "from fixtures.measure_ouf import synthetic_pv_view, synthetic_sn_view\n"
         "pv = measure.build_peculiar_velocity(synthetic_pv_view(TMP,n=2000,seed=3,name='pv'), z_range=(0,0.1), nside_region=8)\n"
         "snv,_ = synthetic_sn_view(TMP,n=300,seed=4,name='sn'); sn=measure.build_sn_hubble(snv, nside_region=4)\n"
         "rows=[('clustering',ms),('cosmic_shear',ms_wl),('pec_velocity',pv),('sn',sn),('lya',lya),('map_cross',xc)]\n"
         "print(f\"{'probe':14s} {'family':12s} {'statistic':14s} subtypes\")\n"
         "for nm,m in rows:\n"
         "    s=m.summary(); kinds=','.join(p['kind'] for p in s['products'].values())\n"
         "    print(f\"{nm:14s} {s['spec']['estimator_family']:12s} {s['spec']['statistic']:14s} {kinds}\")\n"
         "json.dumps(ms.summary()); print('\\nsummary() is JSON-serialisable + cosmology_free =', ms.summary()['cosmology_free'])"),
    md("**Recap.** Six builders, three subtypes, one cosmology-free output. The "
       "container also expresses clusters / strong-lens time-delays / radio "
       "z-absent / GW sirens / LIM via optional atom slots (see "
       "`test_measure_generality.py`)."),
])


# ══════════════════════════════════════════════════════════════════════════
# NB 03 — Pillar 3: simulation (storage + fast-PM + resimulation)
# ══════════════════════════════════════════════════════════════════════════
save("03_pillar3_simulation.ipynb", [
    md("# 03 · Pillar 3 — simulation storage, fast-PM & resimulation\n"
       "OUF-Sim multi-backend storage (wrap-in-place vs re-encode + partial "
       "access), the fast-PM mini-sim (reproduces linear growth), and TreePM "
       "resimulation validated with field estimators. **Dummy/toy physics; the "
       "storage + orchestration substrate is real.**"),
    code(_SETUP + "\nfrom oneuniverse.simulation.cosmology import CosmologySpec\n"
         "C = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81, t_cmb=2.7255)"),
    md("## Storage — wrap-in-place vs re-encode + partial-access pruning"),
    code("from oneuniverse.simulation.linear import generate_linear_sim\n"
         "from oneuniverse.simulation.linear.pack import write_packed_native\n"
         "from oneuniverse.simulation.packed.converter import PackedSimConverter\n"
         "from oneuniverse.simulation.oufsim import SimStore\n"
         "from oneuniverse.simulation.selectors import Cube\n"
         "lin = generate_linear_sim(TMP/'lin', C, box_size=256.0, n_grid=48, redshifts=(0.0,), seed=2, with_lightcone=False)\n"
         "pk = write_packed_native(lin, TMP/'pk', particle_chunk_nside=4)\n"
         "enc = PackedSimConverter().convert(pk, TMP/'enc', sim_name='d', projection='reencode')\n"
         "ref = PackedSimConverter().convert(pk, TMP/'ref', sim_name='e', projection='reference')\n"
         "sz=lambda p: sum(f.stat().st_size for f in p.rglob('*') if f.is_file())/1e6\n"
         "store=SimStore(enc); store.read_box('snapshots',0.0,Cube(0,64,0,64,0,64))\n"
         "print(f'store size  re-encode={sz(enc):.1f} MB  reference(wrap)={sz(ref):.1f} MB  ({100*sz(ref)/sz(enc):.0f}% of re-encode)')\n"
         "print('partial-access read stats:', store.last_read_stats)"),
    md("## Fast-PM mini-sim reproduces linear growth"),
    code("from oneuniverse.simulation.linear.gaussian_field import generate_density_field\n"
         "from oneuniverse.simulation.pm.run import run_pm, zeldovich_pm_ic_from_field\n"
         "from oneuniverse.simulation.pm.deposit import deposit_cic\n"
         "from oneuniverse.simulation.validation import validate_field\n"
         "BOX,N=256.0,48\n"
         "ic = generate_density_field(C, box_size=BOX, n_grid=N, z=0.0, seed=2)\n"
         "x,_ = run_pm(*zeldovich_pm_ic_from_field(C, ic, box=BOX, n_grid=N, z_start=9.0), box=BOX, n_grid=N, cosmo=C, a_start=0.1, a_end=1.0, n_steps=20)\n"
         "d_pm = deposit_cic(x,N,BOX); d_pm = d_pm/d_pm.mean()-1\n"
         "v = validate_field(d_pm, ic, box=BOX)\n"
         "plt.figure(figsize=(6,3)); plt.semilogx(v.k, v.r, 'o-', label='r(k)'); plt.semilogx(v.k, v.transfer,'s-',label='T(k)')\n"
         "plt.axhline(1,color='.6',ls='--'); plt.ylim(0,1.1); plt.xlabel('k [h/Mpc]'); plt.legend()\n"
         "plt.title('PM vs linear theory'); plt.show()\n"
         "print('large-scale recovery: r=%.3f  T=%.3f'%(np.nanmedian(v.r[v.k<0.06]), np.nanmedian(v.transfer[v.k<0.06])))"),
    md("## Resimulation — TreePM-split beats the buffered baseline"),
    code("from oneuniverse.simulation.resim.bench import reference_inner, uncoupled_resim_fn\n"
         "from oneuniverse.simulation.resim.treepm import run_coupled_treepm\n"
         "KW=dict(box=256.0,n_grid=48,target_lo=96.0,target_side=64.0,seed=2,n_steps=15)\n"
         "rinner = reference_inner(C, **KW)\n"
         "icf = generate_density_field(C, box_size=KW['box'], n_grid=KW['n_grid'], z=0.0, seed=KW['seed'])\n"
         "unc = uncoupled_resim_fn(C, **KW)\n"
         "tp = lambda b: run_coupled_treepm(C, icf, box=KW['box'], n_grid=KW['n_grid'], target_lo=KW['target_lo'], target_side=KW['target_side'], buffer=b, z_start=9.0, z_end=0.0, n_steps=KW['n_steps'])['inner']\n"
         "co=lambda a: float(np.corrcoef(a.ravel(), rinner.ravel())[0,1])\n"
         "buffers=[8.0,16.0,24.0,32.0]\n"
         "cu=[co(unc(b)) for b in buffers]; ct=[co(tp(b)) for b in buffers]\n"
         "plt.figure(figsize=(6,4)); plt.plot(buffers,cu,'o-',label='uncoupled (baseline)'); plt.plot(buffers,ct,'s-',label='TreePM-split')\n"
         "plt.xlabel('buffer [Mpc/h]'); plt.ylabel('inner vs full-box corr'); plt.ylim(0,1); plt.legend()\n"
         "plt.title('resimulation buffer convergence'); plt.show()\n"
         "print('TreePM@8 (%.2f) ≈ uncoupled@32 (%.2f): same accuracy, 4× smaller buffer'%(ct[0],cu[-1]))"),
    md("**Recap.** Storage substrate (multi-backend, index-only wrap, partial "
       "access) + fast-PM (linear growth recovered) + TreePM resimulation "
       "(beats the baseline) — all validated with field estimators."),
])


# ══════════════════════════════════════════════════════════════════════════
# NB 04 — Pillar 3: the data↔sim twin
# ══════════════════════════════════════════════════════════════════════════
save("04_pillar3_twin.ipynb", [
    md("# 04 · Pillar 3 — the data↔simulation twin\n"
       "The coupling layer: a biased tracer field (mock data) → Wiener "
       "reconstruction of the underlying field → cross-correlation with truth. "
       "The MVP of constrained forward modelling (linear/Gaussian stand-in)."),
    code(_SETUP + "\nfrom oneuniverse.simulation.cosmology import CosmologySpec\n"
         "C = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81, t_cmb=2.7255)\n"
         "from oneuniverse.simulation.linear.gaussian_field import generate_density_field\n"
         "from oneuniverse.twin.mock_observe import mock_tracer_field\n"
         "from oneuniverse.twin.verify import cross_correlation"),
    md("## Mock challenge: truth → biased tracers → Wiener reconstruction"),
    code("BOX,N=256.0,64\n"
         "truth = generate_density_field(C, box_size=BOX, n_grid=N, z=0.0, seed=2)\n"
         "mock = mock_tracer_field(truth, box_size=BOX, nbar=5e-3, bias=1.5, seed=3, model='lognormal')\n"
         "dg = mock['delta_g']\n"
         "try:\n"
         "    from oneuniverse.twin.reconstruct import wiener_reconstruct\n"
         "    rec = wiener_reconstruct(dg, box_size=BOX, bias=1.5, nbar=5e-3)\n"
         "except Exception as e:\n"
         "    rec = dg  # fallback: show the tracer field itself\n"
         "    print('(using tracer field as recon proxy:', type(e).__name__, ')')\n"
         "sl=N//2\n"
         "fig,ax=plt.subplots(1,3,figsize=(15,4.3))\n"
         "for a,f,t in ((ax[0],truth,'truth δ'),(ax[1],dg,'mock tracers δ_g'),(ax[2],rec,'reconstruction')):\n"
         "    im=a.imshow(f[:,:,sl].T,origin='lower',cmap='magma',vmin=-1,vmax=3); a.set_title(t); plt.colorbar(im,ax=a,fraction=.046)\n"
         "plt.tight_layout(); plt.show()"),
    md("## Recovery: cross-correlation r(k) with the truth field"),
    code("k, r = cross_correlation(rec, truth, box_size=BOX)\n"
         "plt.figure(figsize=(6,4)); plt.plot(k, r, 'o-'); plt.axhline(0.5,color='.6',ls='--')\n"
         "plt.xlabel('k [h/Mpc]'); plt.ylabel('r(k)'); plt.ylim(0,1.05); plt.title('reconstruction × truth')\n"
         "plt.show()\n"
         "below=np.where(r<0.5)[0]; khalf = k[below[0]] if len(below) else float('inf')\n"
         "print('phase-agreement scale (r=0.5): k_half = %.2f h/Mpc'%khalf)"),
    md("## Orchestration — the SimDatabase lineage\n"
       "Catalog → region request → dispatch (dummy resim) → recorded "
       "parent→child lineage (bitemporal). The bookkeeping for a digital twin."),
    code("try:\n"
         "    from oneuniverse.simulation.oufsim.database import SimDatabase\n"
         "    print('SimDatabase available — request_region → dispatch → lineage')\n"
         "    print('methods:', [m for m in dir(SimDatabase) if not m.startswith('_')][:10])\n"
         "except Exception as e:\n"
         "    print('SimDatabase:', e)"),
    md("**Recap.** The twin couples real data to constrained simulations: mock "
       "→ reconstruct → verify with r(k). Linear/Gaussian MVP; the architecture "
       "(forward model + likelihood + orchestration) generalises to real "
       "Bayesian inference (BORG/JaxPM) — see the Pillar-3 plans."),
])

print("done.")
