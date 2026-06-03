"""Phase S8.6 — SimDatabase orchestration control plane."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import write_oufsim_store
from oneuniverse.simulation.oufsim.database import SimDatabase
from oneuniverse.simulation.request import SimulationRequest


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _root(tmp_path):
    root = tmp_path / "root"
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=48, redshifts=(0.0,), seed=2)
    write_oufsim_store(native, root, sim_name="box")
    return root


def test_scan_catalogs_store(tmp_path):
    db = SimDatabase(_root(tmp_path)).scan()
    assert db.sim_names() == ("box",)
    rec = db.get("box")
    assert rec["box_size"] == 200.0 and rec["n_grid"] == 48
    assert rec["seed"] == 2                    # read from the checkpoint


def test_request_region_pending(tmp_path):
    db = SimDatabase(_root(tmp_path)).scan()
    req = db.request_region("box", target_lo=75.0, target_side=50.0,
                            buffer=25.0)
    assert isinstance(req, SimulationRequest)
    assert req.status == "pending" and req.parent_sim == "box"
    assert req.ic_strategy == "zoom_from_parent_ic"


def test_dispatch_runs_resim_and_records_lineage(tmp_path):
    db = SimDatabase(_root(tmp_path)).scan()
    req = db.request_region("box", target_lo=75.0, target_side=50.0,
                            buffer=37.5)
    inner, child = db.dispatch(req, n_steps=12)
    assert inner.ndim == 3 and np.isfinite(inner).all()
    assert db.children_of("box") == [child]
    assert db.parent_of(child) == "box"
    # request lifecycle advanced to ingested
    assert db.requests[0].status == "ingested"
