"""Phase S13 — SimDatabase persistence + ensemble mode."""
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import write_oufsim_store
from oneuniverse.simulation.oufsim.database import SimDatabase


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _db(tmp_path):
    root = tmp_path / "root"
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=32, redshifts=(0.0,), seed=2)
    write_oufsim_store(native, root, sim_name="box")
    return SimDatabase(root).scan()


def test_save_load_roundtrip(tmp_path):
    db = _db(tmp_path)
    db.request_region("box", target_lo=75.0, target_side=50.0, buffer=25.0)
    db.save()
    db2 = SimDatabase(db.root).load()
    assert db2.sim_names() == ("box",)
    assert db2.get("box")["box_size"] == 200.0
    assert len(db2.requests) == 1
    assert db2.requests[0].parent_sim == "box"
    assert db2.requests[0].region.lagrangian_patch is not None


def test_ensemble_mode(tmp_path):
    db = _db(tmp_path)
    reqs = db.request_ensemble("box", n_realisations=3, target_lo=75.0,
                               target_side=50.0, buffer=25.0)
    assert len(reqs) == 3
    inners = db.dispatch_ensemble(reqs, n_steps=8)
    assert len(inners) == 3
    assert all(f.ndim == 3 for f in inners)
    # three distinct children cataloged under one parent
    assert len(db.children_of("box")) == 3
    assert len(set(db.children_of("box"))) == 3
    # phases differ -> realisations differ
    assert not (inners[0] == inners[1]).all()


def test_lineage_records_valid_time(tmp_path):
    db = _db(tmp_path)
    req = db.request_region("box", target_lo=75.0, target_side=50.0,
                            buffer=25.0)
    db.dispatch(req, n_steps=8)
    assert "valid_time" in db.lineage[0]
