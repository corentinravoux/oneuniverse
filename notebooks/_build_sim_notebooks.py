#!/usr/bin/env python3
"""Generate the simulation + twin demonstration notebooks (nbformat).

Builds 04/05/06 in notebooks/. Each ends with a "⚠️ What's NOT working"
section that makes the current limitations explicit with plots/numbers.
Small grids so they execute in a couple of minutes.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"}}
    nbf.write(nb, str(HERE / name))
    print("wrote", name, f"({len(cells)} cells)")


# ===========================================================================
# 04 — oneuniverse.simulation : the OUF-Sim store
# ===========================================================================
save("04_simulation_storage.ipynb", [
    md("# `oneuniverse.simulation` — the OUF-Sim store\n"
       "\n"
       "Pillar 3 stores cosmological simulations in **OUF-Sim**, mirroring the "
       "OUF survey-database tech (JSON manifest + pyarrow parquet + HEALPix-NEST "
       "+ memmap `.npy` tiles) with a sidecar `_index.parquet` per product for "
       "**partial-access** reads.\n"
       "\n"
       "We use a *dummy linear simulation* (Eisenstein–Hu P(k) + Zel'dovich + "
       "toy halos + lightcone) as a stand-in for a real simulator — the focus "
       "is the **storage + access structure**, not real-sim application."),
    code("%matplotlib inline\n"
         "import numpy as np, matplotlib.pyplot as plt, json\n"
         "from pathlib import Path\n"
         "import tempfile\n"
         "from oneuniverse.simulation.cosmology import CosmologySpec\n"
         "from oneuniverse.simulation.linear import generate_linear_sim, linear_power\n"
         "from oneuniverse.simulation.oufsim import write_oufsim_store, SimStore\n"
         "from oneuniverse.simulation.selectors import Cube, Cone\n"
         "COSMO = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,\n"
         "                      sigma8=0.81, t_cmb=2.7255)\n"
         "WORK = Path(tempfile.mkdtemp())\n"
         "print('workdir', WORK)"),
    md("## 1. Generate a dummy linear simulation\n"
       "Pure numpy: E&H transfer → σ8-normalised P(k) → Gaussian field (FFT) → "
       "Zel'dovich particles → toy peak halos → toy HEALPix lightcone."),
    code("native = generate_linear_sim(WORK/'native', COSMO, box_size=256.0,\n"
         "                              n_grid=64, redshifts=(0.0, 1.0), seed=2)\n"
         "print('native products on disk:')\n"
         "for p in sorted(Path(native).glob('*')): print(' ', p.name)\n"
         "for p in sorted((Path(native)/'z0.000').glob('*')): print('   z0.000/', p.name)"),
    code("# the input physics: Eisenstein-Hu linear P(k)\n"
         "k = np.logspace(-2.5, 0.5, 200)\n"
         "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
         "for z in (0.0, 1.0):\n"
         "    ax[0].loglog(k, linear_power(k, COSMO, z=z), label=f'z={z}')\n"
         "ax[0].set_xlabel('k [h/Mpc]'); ax[0].set_ylabel('P(k)'); ax[0].legend()\n"
         "ax[0].set_title('Eisenstein-Hu linear P(k)')\n"
         "field = np.load(Path(native)/'z0.000'/'field.npy')\n"
         "im = ax[1].imshow(field[:,:,32].T, origin='lower', cmap='magma')\n"
         "ax[1].set_title('density field slice (z=0)'); plt.colorbar(im, ax=ax[1])\n"
         "plt.tight_layout(); plt.show()"),
    md("## 2. Convert to an OUF-Sim store\n"
       "`write_oufsim_store` emits `manifest.json` + per-product parquet/tiles "
       "+ sidecar indexes. **All 9 product kinds + AMR + input/output sides** "
       "are exercised by the dummy."),
    code("store = write_oufsim_store(native, WORK/'store', sim_name='demo')\n"
         "man = json.load(open(store/'manifest.json'))\n"
         "print('products:', man['products'])\n"
         "print('store_layout keys:', list(man['store_layout'].keys()))\n"
         "print('has_input:', man['has_input'], ' has_output:', man['has_output'])"),
    md("## 3. Partial-access reads (the load-bearing API)\n"
       "A selector (`Cube`/`Cone`) is resolved against the sidecar index so only "
       "the overlapping partitions are touched — never the whole snapshot."),
    code("s = SimStore(store)\n"
         "sizes = [32, 64, 128, 256]\n"
         "frac_chunks, frac_tiles = [], []\n"
         "for L in sizes:\n"
         "    s.read_box('snapshots', 0.0, Cube(0,L,0,L,0,L))\n"
         "    st = s.last_read_stats; frac_chunks.append(st['chunks_read']/st['chunks_total'])\n"
         "    s.read_field_box(0.0, Cube(0,L,0,L,0,L))\n"
         "    st = s.last_read_stats; frac_tiles.append(st['tiles_read']/st['tiles_total'])\n"
         "plt.figure(figsize=(6,4))\n"
         "plt.plot(sizes, frac_chunks, 'o-', label='particle chunks')\n"
         "plt.plot(sizes, frac_tiles, 's-', label='field tiles')\n"
         "plt.xlabel('cube side [Mpc/h]'); plt.ylabel('fraction of partitions read')\n"
         "plt.title('partial access: data touched vs query size'); plt.legend(); plt.grid(alpha=.3); plt.show()\n"
         "cone = s.read_cone(Cone(lon=30, lat=10, radius_deg=25))\n"
         "print('cone read pixels:', s.last_read_stats)\n"
         "base, refined = s.read_amr_box(0.0, Cube(0,80,0,80,0,80))\n"
         "print('AMR read:', s.last_read_stats, ' base subgrid', base.shape)"),
    md("## 4. Read/write optimisation\n"
       "Column projection reads fewer bytes; Morton row-order tightens parquet "
       "row-group bounding boxes for predicate pushdown."),
    code("from oneuniverse.simulation.oufsim.bench import measure_read\n"
         "full = measure_read(lambda: s.read_box('snapshots', 0.0, Cube(0,120,0,120,0,120)))\n"
         "proj = measure_read(lambda: s.read_box('snapshots', 0.0, Cube(0,120,0,120,0,120), columns=['x']))\n"
         "print(f'peak bytes  full(6 cols)={full.peak_bytes/1e6:.1f} MB   projected(x)={proj.peak_bytes/1e6:.1f} MB')"),
    md("## ⚠️ What is NOT working / current limitations (storage side)\n"
       "\n"
       "1. **Re-encode, not wrap.** The store *re-encodes* the native arrays "
       "into parquet/tiles, so `store_size > native_size`. The architecture's "
       "petabyte-scale ideal is to *wrap* native files + write only the sidecar "
       "index (`projection='reference'`, not yet implemented). Fine for the "
       "dummy, wrong at TB scale.\n"
       "2. **DM / linear generation only.** Storage supports hydro/phase-space/"
       "GR product *kinds*, but the only thing that *generates* them is the "
       "linear+PM dummy (dark matter). Multi-physics generation needs real codes."),
    code("native_mb = sum(f.stat().st_size for f in Path(native).rglob('*') if f.is_file())/1e6\n"
         "store_mb = sum(f.stat().st_size for f in Path(store).rglob('*') if f.is_file())/1e6\n"
         "print(f'native {native_mb:.1f} MB  ->  store {store_mb:.1f} MB   '\n"
         "      f'(re-encode overhead {100*(store_mb/native_mb-1):+.0f}% ; '\n"
         "      'a real backend should WRAP, not re-encode)')"),
])


# ===========================================================================
# 05 — Fast-PM + resimulation feasibility
# ===========================================================================
save("05_pm_and_resimulation.ipynb", [
    md("# Fast-PM mini-simulator + resimulation feasibility\n"
       "\n"
       "The digital-twin core: a cheap full-volume sim carries the large scales; "
       "selected sub-volumes are **re-simulated** at higher fidelity. The "
       "fast-PM (CIC + FFT-Poisson + KDK leapfrog) is the mini-sim. We verify the "
       "physics **and** show where it stops working."),
    code("%matplotlib inline\n"
         "import numpy as np, matplotlib.pyplot as plt\n"
         "from oneuniverse.simulation.cosmology import CosmologySpec\n"
         "from oneuniverse.simulation.linear.gaussian_field import generate_density_field\n"
         "from oneuniverse.simulation.linear.growth import growth_factor\n"
         "from oneuniverse.simulation.pm.run import run_pm, zeldovich_pm_ic_from_field\n"
         "from oneuniverse.simulation.pm.deposit import deposit_cic\n"
         "from oneuniverse.simulation.resim.coupling import (run_full_reference,\n"
         "    run_coupled, run_zoom, full_target_slice)\n"
         "from oneuniverse.simulation.resim.cola import cola_run_pm\n"
         "from oneuniverse.twin.verify import cross_correlation, power_ratio\n"
         "C = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81, t_cmb=2.7255)\n"
         "BOX, N = 200.0, 64\n"
         "def pk_delta(x, n, box):\n"
         "    rho = deposit_cic(x, n, box); return rho/rho.mean()-1"),
    md("## 1. The PM reproduces linear growth (sub-percent)\n"
       "Evolve a linear field from z=9 to z=3 (still linear) and compare the "
       "low-k growth to the analytic factor D(z3)/D(z9)."),
    code("d0 = generate_density_field(C, box_size=BOX, n_grid=N, z=0.0, seed=2)\n"
         "pos, p0 = zeldovich_pm_ic_from_field(C, d0, box=BOX, n_grid=N, z_start=9.0)\n"
         "def measure_lowk_ratio(a_end):\n"
         "    x,_ = run_pm(pos, p0, box=BOX, n_grid=N, cosmo=C, a_start=0.1, a_end=a_end, n_steps=20)\n"
         "    d = pk_delta(x, N, BOX)\n"
         "    rho0 = deposit_cic(pos, N, BOX); dic = rho0/rho0.mean()-1\n"
         "    k,_ = cross_correlation(d, dic, box_size=BOX); _,pr = power_ratio(d, dic, box_size=BOX)\n"
         "    return float(np.sqrt(np.nanmedian(pr[k<0.06])))\n"
         "meas = measure_lowk_ratio(0.25); exp = growth_factor(3.0,C)/growth_factor(9.0,C)\n"
         "print(f'z9->z3 growth ratio   measured {meas:.3f}   analytic {exp:.3f}   err {100*(meas/exp-1):+.1f}%')"),
    md("## 2. Resimulation feasibility — buffer convergence (the headline result)\n"
       "Resimulate a target sub-cube on buffer-padded particles; compare the "
       "inner region to the full-box reference as the buffer grows."),
    code("full = run_full_reference(C, box=BOX, n_grid=N, z_start=9.0, z_end=0.0, seed=2, n_steps=18)\n"
         "TLO, TS = 75.0, 50.0\n"
         "ref = full_target_slice(full, box=BOX, n_grid=N, target_lo=TLO, target_side=TS)\n"
         "buffers = [12.5, 25.0, 37.5, 50.0]\n"
         "corr = []\n"
         "for b in buffers:\n"
         "    inner = run_coupled(C, box=BOX, n_grid=N, target_lo=TLO, target_side=TS,\n"
         "                        buffer=b, z_start=9.0, z_end=0.0, seed=2, n_steps=18)['inner']\n"
         "    corr.append(np.corrcoef(inner.ravel(), ref.ravel())[0,1])\n"
         "plt.figure(figsize=(6,4)); plt.plot(buffers, corr, 'o-')\n"
         "plt.axhline(0.8, color='.6', ls='--'); plt.ylim(0,1)\n"
         "plt.xlabel('buffer [Mpc/h]'); plt.ylabel('inner vs full-box corr')\n"
         "plt.title('resimulation converges as buffer grows (feasibility ✓)'); plt.grid(alpha=.3); plt.show()\n"
         "print('corr:', np.round(corr,3))"),
    md("## 3. True zoom — higher resolution adds small-scale power"),
    code("box_buf, nc = 100.0, 24\n"
         "coarse = generate_density_field(C, box_size=box_buf, n_grid=nc, z=0.0, seed=2)\n"
         "res = run_zoom(C, coarse, box_buf=box_buf, target_side=50.0, buffer=25.0,\n"
         "               factor=2, z_start=9.0, z_end=0.0, seed=9, n_steps=12)\n"
         "print(f'parent grid {nc}  ->  zoom grid {res[\"n_fine\"]}  (resolves beyond parent Nyquist)')"),
    md("## 4. COLA frame — few steps reproduce the full PM on large scales"),
    code("xf,_ = run_pm(pos, p0, box=BOX, n_grid=N, cosmo=C, a_start=0.1, a_end=1.0, n_steps=25)\n"
         "df = pk_delta(xf, N, BOX)\n"
         "xc = cola_run_pm(C, d0, box=BOX, n_grid=N, a_start=0.1, a_end=1.0, n_steps=5)\n"
         "dc = pk_delta(xc, N, BOX)\n"
         "k,r = cross_correlation(dc, df, box_size=BOX); _,pr = power_ratio(dc, df, box_size=BOX)\n"
         "print(f'COLA 5 steps vs full-PM 25 steps:  r_lowk {np.nanmedian(r[k<0.1]):.3f}   '\n"
         "      f'P/Pfull_lowk {np.nanmedian(pr[k<0.1]):.3f}')"),
    md("## ⚠️ What is NOT working (the important ones)\n"
       "\n"
       "### (a) sCOLA buffer coupling — DOES NOT beat the uncoupled run\n"
       "The goal was: use the far-field/COLA frame so a *small* buffer suffices. "
       "Every coupling attempt is **worse** than the plain uncoupled resim (and "
       "inconsistent across buffer sizes). Correct sub-box sCOLA needs exact LPT "
       "boundary conditions (Tassev & Eisenstein 2015) — deferred. The plot below "
       "shows the failure directly."),
    code("from oneuniverse.simulation.pm.run import _zeldovich_displacement\n"
         "ic = generate_density_field(C, box_size=BOX, n_grid=N, z=0.0, seed=2)\n"
         "psi_full = _zeldovich_displacement(ic, BOX, N).reshape(N,N,N,3)\n"
         "def cola_coupled(buf, nst=10):\n"
         "    cell=BOX/N; bsize=TS+2*buf; blo=TLO-buf\n"
         "    bi0=int(round(blo/cell)); bi1=int(round((blo+bsize)/cell)); nb=bi1-bi0; bb=nb*cell\n"
         "    dsub=np.ascontiguousarray(ic[bi0:bi1,bi0:bi1,bi0:bi1])\n"
         "    x=cola_run_pm(C, dsub, box=bb, n_grid=nb, a_start=0.1, a_end=1.0, n_steps=nst)\n"
         "    rho=deposit_cic(x,nb,bb); d=rho/rho.mean()-1\n"
         "    pad=int(round(buf/cell)); ti=int(round(TS/cell))\n"
         "    return d[pad:pad+ti,pad:pad+ti,pad:pad+ti]\n"
         "unc, cu = [], []\n"
         "for b in buffers:\n"
         "    u = run_coupled(C, box=BOX, n_grid=N, target_lo=TLO, target_side=TS,\n"
         "                    buffer=b, z_start=9.0, z_end=0.0, seed=2, n_steps=18)['inner']\n"
         "    unc.append(np.corrcoef(u.ravel(), ref.ravel())[0,1])\n"
         "    cu.append(np.corrcoef(cola_coupled(b).ravel(), ref.ravel())[0,1])\n"
         "plt.figure(figsize=(6,4))\n"
         "plt.plot(buffers, unc, 'o-', label='uncoupled (works)')\n"
         "plt.plot(buffers, cu, 's--', label='COLA-coupled (BROKEN — worse)')\n"
         "plt.xlabel('buffer [Mpc/h]'); plt.ylabel('inner vs full-box corr'); plt.ylim(0,1)\n"
         "plt.title('⚠️ sCOLA coupling is WORSE than uncoupled — not yet working')\n"
         "plt.legend(); plt.grid(alpha=.3); plt.show()"),
    md("### (b) PM small-scale power deficit (mesh force resolution)\n"
       "The PM force is mesh-limited (CIC + grid), so small-scale power is "
       "**suppressed** relative to linear theory. Large scales are correct; this "
       "is expected PM under-resolution, not a bug — a real run uses a finer mesh "
       "/ PM+tree."),
    code("lin0 = generate_density_field(C, box_size=BOX, n_grid=N, z=0.0, seed=2)\n"
         "k, pr = power_ratio(df, lin0, box_size=BOX)\n"
         "plt.figure(figsize=(6,4)); plt.semilogx(k, pr, 'o-')\n"
         "plt.axhline(1, color='.6', ls='--'); plt.ylim(0,1.3)\n"
         "plt.xlabel('k [h/Mpc]'); plt.ylabel('P_PM / P_linear')\n"
         "plt.title('⚠️ PM under-resolves small scales (high-k deficit)'); plt.grid(alpha=.3); plt.show()\n"
         "print('P_pm/P_lin:  lowk', round(float(np.nanmedian(pr[k<0.06])),2),\n"
         "      ' highk', round(float(np.nanmedian(pr[k>0.5])),2))"),
])


# ===========================================================================
# 06 — Twin : data-driven coupling
# ===========================================================================
save("06_twin_data_driven.ipynb", [
    md("# The twin — data ↔ simulation coupling\n"
       "\n"
       "`oneuniverse.twin` is the coupling layer: data → reconstruct the field → "
       "forward-model / resimulate. We show the loop working on the dummy, then "
       "the places it breaks."),
    code("%matplotlib inline\n"
         "import numpy as np, matplotlib.pyplot as plt\n"
         "from oneuniverse.simulation.cosmology import CosmologySpec\n"
         "from oneuniverse.simulation.linear.gaussian_field import generate_density_field\n"
         "from oneuniverse.twin import (mock_tracer_field, wiener_reconstruct,\n"
         "    constrained_realization, recover_metrics, registered_engines)\n"
         "from oneuniverse.twin.verify import cross_correlation, power_ratio\n"
         "C = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81, t_cmb=2.7255)\n"
         "BOX, N = 256.0, 64"),
    md("## 1. The mock challenge — recover the field from biased tracers\n"
       "truth → mock-observe (biased Poisson tracers) → Wiener filter → "
       "cross-correlation r(k) vs truth. r→1 at large scales, falling where shot "
       "noise dominates."),
    code("truth = generate_density_field(C, box_size=BOX, n_grid=N, z=0.0, seed=2)\n"
         "plt.figure(figsize=(6,4))\n"
         "for nbar in (1e-3, 5e-3, 5e-2):\n"
         "    obs = mock_tracer_field(truth, box_size=BOX, nbar=nbar, bias=1.5, seed=3)\n"
         "    rec = wiener_reconstruct(obs['delta_g'], C, box_size=BOX, nbar=nbar, bias=1.5)\n"
         "    k, r = cross_correlation(rec, truth, box_size=BOX)\n"
         "    plt.semilogx(k, r, 'o-', ms=3, label=f'n̄={nbar:.0e}')\n"
         "plt.axhline(0.5, color='.6', ls='--'); plt.ylim(0,1.05)\n"
         "plt.xlabel('k [h/Mpc]'); plt.ylabel('r(k) reconstruction × truth')\n"
         "plt.title('mock challenge: field recovery vs survey density'); plt.legend(); plt.grid(alpha=.3); plt.show()"),
    md("## 2. Wiener mean vs constrained realization\n"
       "The Wiener mean is power-suppressed at high k; the Hoffman–Ribak "
       "constrained realization restores P(k) (clean-linear regime)."),
    code("rng = np.random.default_rng(7)\n"
         "nbar, bias = 1e-3, 1.5\n"
         "dg = bias*truth + rng.normal(0, 1/np.sqrt(nbar*(BOX/N)**3), (N,N,N))  # clean linear obs\n"
         "wf = wiener_reconstruct(dg, C, box_size=BOX, nbar=nbar, bias=bias)\n"
         "cr = constrained_realization(dg, C, box_size=BOX, nbar=nbar, bias=bias, seed=7)\n"
         "k, rwf = power_ratio(wf, truth, box_size=BOX); _, rcr = power_ratio(cr, truth, box_size=BOX)\n"
         "plt.figure(figsize=(6,4))\n"
         "plt.semilogx(k, rwf, 'o-', label='Wiener mean (suppressed)')\n"
         "plt.semilogx(k, rcr, 's-', label='constrained realization (restored)')\n"
         "plt.axhline(1, color='.6', ls='--'); plt.ylim(0,1.6)\n"
         "plt.xlabel('k [h/Mpc]'); plt.ylabel('P / P_truth')\n"
         "plt.title('Wiener vs constrained realization'); plt.legend(); plt.grid(alpha=.3); plt.show()"),
    md("## 3. Data-driven dispatch — the orchestration loop\n"
       "`SimDatabase` turns a region selection into a request and dispatches the "
       "resim **from the data-constrained IC** (not a seed)."),
    code("import tempfile; from pathlib import Path\n"
         "from oneuniverse.simulation.linear import generate_linear_sim\n"
         "from oneuniverse.simulation.oufsim import write_oufsim_store\n"
         "from oneuniverse.simulation.oufsim.database import SimDatabase\n"
         "W = Path(tempfile.mkdtemp())\n"
         "nat = generate_linear_sim(W/'n', C, box_size=BOX, n_grid=N, redshifts=(0.0,), seed=2)\n"
         "write_oufsim_store(nat, W/'root', sim_name='box')\n"
         "db = SimDatabase(W/'root').scan()\n"
         "req = db.request_region('box', target_lo=96.0, target_side=64.0, buffer=37.5,\n"
         "                        ic_strategy='constrained_from_posterior')\n"
         "inner, child = db.dispatch(req, ic_field=cr, n_steps=12)\n"
         "print('dispatched child:', child)\n"
         "print('lineage edge:', db.lineage[0])\n"
         "print('request status:', db.requests[0].status)"),
    md("## 4. Engine contracts (generality)\n"
       "Reconstruction + forward engines satisfy a common contract; a real code "
       "plugs in over the same store boundary."),
    code("print('registered engines:', registered_engines())"),
    md("## ⚠️ What is NOT working\n"
       "\n"
       "### (a) Linear-bias Poisson mock breaks for non-linear fields\n"
       "`mock_tracer_field` clips `1+bδ` at 0; once σ_cell ≳ 1 this biases the "
       "*effective* tracer bias low (b=1.5 requested samples as ~0.6). Fine for "
       "correlation r(k); **wrong for absolute-power** work. A lognormal/HOD mock "
       "is the proper fix (future)."),
    code("def b_eff(z):\n"
         "    t = generate_density_field(C, box_size=BOX, n_grid=N, z=z, seed=2)\n"
         "    dg = mock_tracer_field(t, box_size=BOX, nbar=5e-2, bias=1.5, seed=3)['delta_g']\n"
         "    fk = np.fft.rfftn(dg); tk = np.fft.rfftn(t)\n"
         "    from oneuniverse.twin.verify import _bin_kgrid, _bins\n"
         "    km = _bin_kgrid(N, BOX); idx, edges, ctr = _bins(km, BOX, N)\n"
         "    pd = (np.abs(fk)**2).ravel(); pt = (np.abs(tk)**2).ravel()\n"
         "    rs = [pd[idx==i].sum()/pt[idx==i].sum() for i in range(1,6) if (idx==i).sum()]\n"
         "    return float(t.std()), float(np.sqrt(np.median(rs)))\n"
         "sig, be = zip(*[b_eff(z) for z in (0.0,1.0,2.0,3.0)])\n"
         "plt.figure(figsize=(6,4)); plt.plot(sig, be, 'o-')\n"
         "plt.axhline(1.5, color='.6', ls='--', label='requested b=1.5')\n"
         "plt.xlabel('σ_cell (field non-linearity)'); plt.ylabel('measured effective bias')\n"
         "plt.title('⚠️ Poisson+clip mock biases b low when σ≳1'); plt.legend(); plt.grid(alpha=.3); plt.show()\n"
         "print('σ_cell:', np.round(sig,2), ' b_eff:', np.round(be,2))"),
    md("### (b) Constrained realization: right power, random small-scale phases\n"
       "The CR matches P(k) everywhere but its *small-scale phases* are a new "
       "random draw (it is a realization, not the truth) — so r(k) drops at high "
       "k even though the power is correct. Below: r(k) (phases) vs P/P_truth "
       "(power) for the CR."),
    code("k, rxc = cross_correlation(cr, truth, box_size=BOX); _, prc = power_ratio(cr, truth, box_size=BOX)\n"
         "plt.figure(figsize=(6,4))\n"
         "plt.semilogx(k, rxc, 'o-', label='r(k): phase agreement (drops)')\n"
         "plt.semilogx(k, prc, 's-', label='P/P_truth: power (stays ~1)')\n"
         "plt.axhline(1, color='.6', ls='--'); plt.ylim(0,1.4)\n"
         "plt.xlabel('k [h/Mpc]')\n"
         "plt.title('⚠️ CR: correct power, but small scales are a new realization')\n"
         "plt.legend(); plt.grid(alpha=.3); plt.show()"),
    md("### (c) Other open gaps\n"
       "- **SBI** deferred to future (ensemble mode exists; the inference does not).\n"
       "- **Multi-physics** (hydro/baryons) generation: DM-only.\n"
       "- **sCOLA buffer coupling** (smaller buffers / bounded-memory resim): "
       "broken — see notebook 05 §(a)."),
])

print("done")
