"""Phase S9 T4 — end-to-end data-driven resimulation loop.

truth → mock-observe → constrained realization → resim from the data-driven
IC → the resimulated region tracks the truth where the data constrained it.
This is the junction that makes the twin *data-driven*.
"""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.oufsim import write_oufsim_store
from oneuniverse.simulation.oufsim.database import SimDatabase
from oneuniverse.simulation.resim.coupling import run_coupled
from oneuniverse.simulation.resim.verify import gate2_dynamical
from oneuniverse.twin.constrained import constrained_realization


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


BOX, N, TLO, TSIDE, NBAR, BIAS = 200.0, 48, 75.0, 50.0, 5e-3, 1.5


def _constrained_ic(c, truth):
    # clean-linear mock observation (the regime the Wiener/HR model assumes)
    rng = np.random.default_rng(7)
    dg = BIAS * truth + rng.normal(0, 1 / np.sqrt(NBAR * (BOX / N) ** 3),
                                   (N, N, N))
    return constrained_realization(dg, c, box_size=BOX, nbar=NBAR, bias=BIAS,
                                   seed=3)


def test_data_driven_resim_tracks_truth():
    c = _cosmo()
    truth = generate_density_field(c, box_size=BOX, n_grid=N, z=0.0, seed=2)
    ic = _constrained_ic(c, truth)
    kw = dict(box=BOX, n_grid=N, target_lo=TLO, target_side=TSIDE,
              buffer=37.5, z_start=9.0, z_end=0.0, n_steps=12)
    ref = run_coupled(c, ic_field=truth, **kw)["inner"]       # truth resim
    dd = run_coupled(c, ic_field=ic, **kw)["inner"]           # data-driven
    g2 = gate2_dynamical(dd, ref, box_size=TSIDE)
    # the data-driven resim recovers the truth's large-scale structure
    assert g2["r_lowk"] > 0.7


def test_dispatch_data_driven_provenance(tmp_path):
    c = _cosmo()
    root = tmp_path / "root"
    native = generate_linear_sim(tmp_path / "n", c, box_size=BOX, n_grid=N,
                                 redshifts=(0.0,), seed=2)
    write_oufsim_store(native, root, sim_name="box")
    db = SimDatabase(root).scan()
    truth = generate_density_field(c, box_size=BOX, n_grid=N, z=0.0, seed=2)
    ic = _constrained_ic(c, truth)
    req = db.request_region("box", target_lo=TLO, target_side=TSIDE,
                            buffer=37.5, ic_strategy="constrained_from_posterior")
    inner, child = db.dispatch(req, ic_field=ic, n_steps=12)
    assert inner.ndim == 3
    assert db.lineage[0]["ic_source"] == "constrained_from_posterior"
    assert db.requests[0].status == "ingested"
