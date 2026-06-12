#!/usr/bin/env python3
"""Build the oneuniverse capability notebooks (visual, full-surface tour).

Six notebooks, each strongly visual, together presenting everything the
package offers: data layer + OUF 2.6, SQL export, the measure layer +
Universal DataProduct, simulation storage, PM gravity + resimulation, and
the constrained twin. Executed via `jupyter nbconvert --execute`.
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
md = lambda t: nbf.v4.new_markdown_cell(t)
code = lambda t: nbf.v4.new_code_cell(t)


def save(name, cells):
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"}}
    nbf.write(nb, str(HERE / name)); print("wrote", name, f"({len(cells)} cells)")


SETUP = (
    "%matplotlib inline\n"
    "import os, sys, json, time, sqlite3, tempfile, warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "from pathlib import Path\n"
    "import numpy as np, pandas as pd, matplotlib.pyplot as plt, healpy as hp\n"
    "import matplotlib.patches as mpatches\n"
    "from matplotlib.patches import FancyArrow, Rectangle, FancyBboxPatch\n"
    "plt.rcParams.update({'figure.dpi':115,'font.size':10.5,'axes.titlesize':11.5,\n"
    "    'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,\n"
    "    'figure.facecolor':'white'})\n"
    "C0,C1,C2,C3,C4 = '#3a6ea5','#d1495b','#3c8d53','#edae49','#7d5ba6'\n"
    "ROOT = Path.cwd().parent; sys.path.insert(0, str(ROOT/'test'))\n"
    "TMP = Path(tempfile.mkdtemp())\n"
    "DATA_ROOT='/home/ravoux/Documents/Science/Cosmography/oneuniverse_data'\n"
    "EBOSS=Path(DATA_ROOT)/'spectroscopic/eboss/qso/DR16Q_Superset_v3.fits'\n"
    "DESI =Path(DATA_ROOT)/'spectroscopic/desi/dr1/qso/QSO_full.dat.fits'\n"
    "HAVE_REAL = EBOSS.exists() and DESI.exists()\n"
    "from oneuniverse.simulation.cosmology import CosmologySpec\n"
    "COSMO = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81, t_cmb=2.7255)\n"
    "def moll(ax, ra, dec, **kw):\n"
    "    lon = np.radians(((np.asarray(ra)+180)%360)-180)\n"
    "    ax.scatter(lon, np.radians(dec), **kw)\n"
    "print('real eBOSS+DESI available:', HAVE_REAL)")

PK = (
    "def measure_pk(field, box, edges):\n"
    "    '''Band-power P(k)=<|δ_k|²>V/N⁶ with Gaussian errors σ_P=P√(2/N_modes).'''\n"
    "    n=field.shape[0]; dk=np.fft.rfftn(field)\n"
    "    kf=np.fft.fftfreq(n,d=box/n)*2*np.pi; kz=np.fft.rfftfreq(n,d=box/n)*2*np.pi\n"
    "    KX,KY,KZ=np.meshgrid(kf,kf,kz,indexing='ij'); km=np.sqrt(KX**2+KY**2+KZ**2).ravel()\n"
    "    p=(np.abs(dk)**2*box**3/n**6).ravel(); idx=np.digitize(km,edges)\n"
    "    k=[];P=[];E=[]\n"
    "    for i in range(1,len(edges)):\n"
    "        m=idx==i\n"
    "        if m.sum()<2: continue\n"
    "        k.append(km[m].mean()); P.append(p[m].mean()); E.append(p[m].mean()*np.sqrt(2.0/m.sum()))\n"
    "    return np.array(k),np.array(P),np.array(E)")

LOADQSO = (
    "def load_qso(name, zlo=0.8, zhi=2.2, cap=60000, seed=0):\n"
    "    '''Real survey catalog via the registered P1 loader (synthetic fallback).'''\n"
    "    if not HAVE_REAL:\n"
    "        from fixtures.measure_ouf import synthetic_point_view\n"
    "        return synthetic_point_view(TMP, n=cap, seed=seed, name=name).read(), False\n"
    "    os.environ['ONEUNIVERSE_DATA_ROOT']=DATA_ROOT\n"
    "    from oneuniverse.data import load_catalog\n"
    "    df = load_catalog(name, validate=False)\n"
    "    df = df[(df['z']>=zlo)&(df['z']<=zhi)].dropna(subset=['ra','dec','z'])\n"
    "    if len(df)>cap: df=df.sample(cap, random_state=seed)\n"
    "    return df.reset_index(drop=True), True")

TO_OUF = (
    "def to_ouf_view(df, tmp, name, extra_cols=()):\n"
    "    '''Write a DataFrame as an OUF 2.6 POINT dataset; return its DatasetView.'''\n"
    "    from oneuniverse.data.converter import write_ouf_dataset\n"
    "    from oneuniverse.data.dataset_view import DatasetView\n"
    "    from oneuniverse.data.format_spec import DataGeometry\n"
    "    from oneuniverse.data.manifest import LoaderSpec\n"
    "    n=len(df); ra=df['ra'].to_numpy(float); dec=df['dec'].to_numpy(float)\n"
    "    out={'ra':ra,'dec':dec,'z':df['z'].to_numpy(float),'z_type':np.full(n,'spec'),\n"
    "         'z_err':np.full(n,1e-4),'galaxy_id':np.arange(n,dtype=np.int64),\n"
    "         'survey_id':np.zeros(n,dtype=np.int64),'weight_comp':np.ones(n),\n"
    "         '_original_row_index':np.arange(n,dtype='i8'),\n"
    "         '_healpix32':hp.ang2pix(32,ra,dec,nest=True,lonlat=True).astype('i4')}\n"
    "    for c in extra_cols:\n"
    "        out[c]=df[c].to_numpy()\n"
    "    od=tmp/name/'oneuniverse'\n"
    "    write_ouf_dataset(df=pd.DataFrame(out),out_dir=od,survey_name=name,\n"
    "        survey_type='spectroscopic',geometry=DataGeometry.POINT,\n"
    "        loader=LoaderSpec(name=name,version='0'))\n"
    "    return DatasetView.from_path(od.parent)")

print("(helpers ready)")
exec(open(HERE / "_nb_a.py").read())   # 01 data, 02 sql
exec(open(HERE / "_nb_b.py").read())   # 03 measure, 04 sim store
exec(open(HERE / "_nb_c.py").read())   # 05 gravity, 06 twin
print("all notebooks built.")
