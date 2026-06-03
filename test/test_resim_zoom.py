"""Phase S10 — true zoom: multi-resolution refined ICs + higher-res resim."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.resim.coupling import run_zoom
from oneuniverse.simulation.resim.zoom import refine_ic
from oneuniverse.twin.verify import _bin_kgrid, _bins, cross_correlation


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_refine_ic_preserves_large_scales_adds_small():
    c = _cosmo()
    box, nc = 128.0, 32
    coarse = generate_density_field(c, box_size=box, n_grid=nc, z=0.0, seed=2)
    fine = refine_ic(coarse, box_sub=box, cosmo=c, factor=2, seed=9)
    assert fine.shape == (64, 64, 64)
    # downsample fine -> coarse grid; large-scale modes preserved
    rd = fine.reshape(nc, 2, nc, 2, nc, 2).mean(axis=(1, 3, 5))
    k, r = cross_correlation(rd, coarse, box_size=box)
    assert np.nanmedian(r[k < 0.1]) > 0.95
    # small-scale power added (fine variance exceeds coarse)
    assert fine.var() > coarse.var()


def _pk_beyond(field, box, k_split):
    """Mean power in modes with |k| > k_split."""
    n = field.shape[0]
    fk = np.fft.rfftn(field)
    km = _bin_kgrid(n, box).ravel()
    p = (np.abs(fk) ** 2).ravel()
    return float(p[km > k_split].mean())


def test_zoom_resolves_higher_k():
    c = _cosmo()
    box_buf, nc = 100.0, 24
    coarse = generate_density_field(c, box_size=box_buf, n_grid=nc, z=0.0,
                                    seed=2)
    res = run_zoom(c, coarse, box_buf=box_buf, target_side=50.0, buffer=25.0,
                   factor=2, z_start=9.0, z_end=0.0, seed=9, n_steps=12)
    assert res["n_fine"] == 48                  # 2x the coarse grid
    # the parent Nyquist; the zoom must carry power beyond it
    k_nyq_parent = np.pi * nc / box_buf
    assert res["inner"].shape[0] == int(round(50.0 / (box_buf / 48)))
    assert _pk_beyond(res["inner"], 50.0, k_nyq_parent) > 0.0
