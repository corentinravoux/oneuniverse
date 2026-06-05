#!/usr/bin/env python3
"""Build the oneuniverse scientific capability notebooks.

Each notebook states a claim and validates it against theory or known ground
truth. Executed with `jupyter nbconvert --execute`.
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
    "import os, sys, time, tempfile, warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "from pathlib import Path\n"
    "import numpy as np, pandas as pd, matplotlib.pyplot as plt, healpy as hp\n"
    "plt.rcParams.update({'figure.dpi':110,'font.size':11})\n"
    "ROOT = Path.cwd().parent; sys.path.insert(0, str(ROOT/'test'))\n"
    "TMP = Path(tempfile.mkdtemp())\n"
    "DATA_ROOT='/home/ravoux/Documents/Science/Cosmography/oneuniverse_data'\n"
    "EBOSS=Path(DATA_ROOT)/'spectroscopic/eboss/qso/DR16Q_Superset_v3.fits'\n"
    "DESI =Path(DATA_ROOT)/'spectroscopic/desi/dr1/qso/QSO_full.dat.fits'\n"
    "HAVE_EBOSS, HAVE_DESI = EBOSS.exists(), DESI.exists()\n"
    "from oneuniverse.simulation.cosmology import CosmologySpec\n"
    "COSMO = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81, t_cmb=2.7255)\n"
    "print('real data — eBOSS:',HAVE_EBOSS,'| DESI:',HAVE_DESI)")

PK = (
    "def measure_pk(field, box, edges):\n"
    "    '''Band-power P(k)=<|δ_k|²> V/N⁶ + Gaussian error σ_P=P√(2/N_modes).'''\n"
    "    n=field.shape[0]; dk=np.fft.rfftn(field)\n"
    "    kf=np.fft.fftfreq(n,d=box/n)*2*np.pi; kz=np.fft.rfftfreq(n,d=box/n)*2*np.pi\n"
    "    KX,KY,KZ=np.meshgrid(kf,kf,kz,indexing='ij'); km=np.sqrt(KX**2+KY**2+KZ**2).ravel()\n"
    "    p=(np.abs(dk)**2*box**3/n**6).ravel(); idx=np.digitize(km,edges)\n"
    "    k=[];P=[];E=[];Nm=[]\n"
    "    for i in range(1,len(edges)):\n"
    "        m=idx==i\n"
    "        if m.sum()<2: continue\n"
    "        k.append(km[m].mean()); P.append(p[m].mean()); Nm.append(int(m.sum())); E.append(p[m].mean()*np.sqrt(2.0/m.sum()))\n"
    "    return map(np.array,(k,P,E,Nm))\n"
    "def cross_pk(a,b,box,edges):\n"
    "    n=a.shape[0]; ak=np.fft.rfftn(a); bk=np.fft.rfftn(b)\n"
    "    kf=np.fft.fftfreq(n,d=box/n)*2*np.pi; kz=np.fft.rfftfreq(n,d=box/n)*2*np.pi\n"
    "    KX,KY,KZ=np.meshgrid(kf,kf,kz,indexing='ij'); km=np.sqrt(KX**2+KY**2+KZ**2).ravel()\n"
    "    p=(np.real(ak*np.conj(bk))*box**3/n**6).ravel(); idx=np.digitize(km,edges)\n"
    "    k=[];P=[]\n"
    "    for i in range(1,len(edges)):\n"
    "        m=idx==i\n"
    "        if m.sum()<2: continue\n"
    "        k.append(km[m].mean()); P.append(p[m].mean())\n"
    "    return np.array(k),np.array(P)")

EBOSSVIEW = (
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
    "    df=load_catalog('eboss_qso',validate=False)\n"
    "    df=df[(df['z']>=0.8)&(df['z']<=2.2)].dropna(subset=['ra','dec','z'])\n"
    "    if len(df)>n_cap: df=df.sample(n_cap,random_state=0)\n"
    "    df=df.reset_index(drop=True); n=len(df); ra=df['ra'].to_numpy(float); dec=df['dec'].to_numpy(float)\n"
    "    out=pd.DataFrame({'ra':ra,'dec':dec,'z':df['z'].to_numpy(float),'z_type':np.full(n,'spec'),\n"
    "        'z_err':np.full(n,1e-4),'galaxy_id':np.arange(n,dtype=np.int64),'survey_id':np.zeros(n,dtype=np.int64),\n"
    "        'weight_comp':np.ones(n),'nbar':np.full(n,1e-3),'_original_row_index':np.arange(n,dtype='i8'),\n"
    "        '_healpix32':hp.ang2pix(32,ra,dec,nest=True,lonlat=True).astype('i4')})\n"
    "    od=tmp/'eboss'/'oneuniverse'\n"
    "    write_ouf_dataset(df=out,out_dir=od,survey_name='eboss',survey_type='spectroscopic',geometry=DataGeometry.POINT,loader=LoaderSpec(name='eboss',version='0'))\n"
    "    return DatasetView.from_path(od.parent), True")

print("(helpers ready — building notebooks)")
exec(open(HERE / "_nb_part2.py").read())  # notebooks 01-03
exec(open(HERE / "_nb_part3.py").read())  # notebooks 04-07
print("all notebooks built.")
