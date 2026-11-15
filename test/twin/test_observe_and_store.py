"""Data -> twin socket and the engine <-> store boundary."""
import numpy as np
import pyarrow as pa
import pytest
from fixtures.synthetic import synthetic_point_table
from fixtures.twin_mock import product_from_counts

import oneuniverse.reference.engines  # noqa: F401  (registers the engines at collection time)
from oneuniverse._core.errors import ValidationError
from oneuniverse.data import DatasetView, Geometry, SurveyMeta, write_dataset
from oneuniverse.data.manifest import LoaderInfo
from oneuniverse.reference.fieldstats import recover_metrics
from oneuniverse.reference.linear import gaussian_field
from oneuniverse.reference.mock import lognormal_tracers, mask_slab
from oneuniverse.simulation.specs import CosmologySpec
from oneuniverse.twin import get_engine, observe_from_view, run_engine_to_store

COSMO = CosmologySpec(0.6736, 0.3153, 0.0493, 0.8111, 0.9649)


def _view(tmp_path, n=20_000, hi=100.0):
    tab = synthetic_point_table(n, 0); rng = np.random.default_rng(0)
    for c in ("x", "y", "zb"):
        tab = tab.append_column(c, pa.array(rng.uniform(0, hi, n)))
    write_dataset(tab, tmp_path / "ds", geometry=Geometry.POINT, survey=SurveyMeta("s", "test"), loader=LoaderInfo("t", "1", {}))
    return DatasetView.from_path(tmp_path / "ds")


def test_observe_from_view_grid_and_mask(tmp_path):
    v = _view(tmp_path)
    obs = observe_from_view(v, box=100.0, n_grid=16, position_columns=("x", "y", "zb"), bias=1.2); assert obs.delta_g.shape == (16,) * 3 and abs(obs.delta_g.mean()) < 1e-10 and obs.nbar == pytest.approx(20_000 / 100.0**3)
    mask = mask_slab(16, 0, 0.0, 0.5); obs_m = observe_from_view(v, box=100.0, n_grid=16, position_columns=("x", "y", "zb"), mask=mask); assert abs(obs_m.delta_g[mask].mean()) < 1e-10 and np.all(obs_m.delta_g[~mask] == 0)
    assert obs.box == 100.0 and obs.meta["columns"] == ["x", "y", "zb"] and "dataset_version" in obs.meta


def test_observe_streaming_matches_single_batch(tmp_path):
    v = _view(tmp_path)
    a = observe_from_view(v, box=100.0, n_grid=8, position_columns=("x", "y", "zb"))
    cols = v.table(columns=["x", "y", "zb"])
    from oneuniverse.reference.pm import deposit_cic
    n = deposit_cic(np.stack([cols[c].to_numpy() for c in ("x", "y", "zb")], 1), 8, 100.0)
    assert np.allclose(a.delta_g, n / n.mean() - 1)


def test_observe_rejects_out_of_box(tmp_path):
    v = _view(tmp_path, hi=120.0)
    with pytest.raises(ValidationError):
        observe_from_view(v, box=100.0, n_grid=8, position_columns=("x", "y", "zb"))
    with pytest.raises(ValidationError):
        observe_from_view(_view(tmp_path / "b"), box=100.0, n_grid=8, position_columns=("x", "y", "nope"))


def test_closed_loop_recovers_truth(tmp_path):
    # Regime: the plan's box 128 / z=0 gave sigma_cell 1.82, outside the lognormal validity regime
    # (sigma_cell <~ 1); box 256 at z=1 gives 0.77 (controller ruling 2026-11-15).
    from oneuniverse.reference.linear import growth_factor
    truth = gaussian_field(COSMO, box=256.0, n_grid=32, seed=0) * growth_factor(1.0, COSMO); counts = lognormal_tracers(truth, box=256.0, nbar=2e-2, bias=1.5, seed=1)
    prod = product_from_counts(counts, box=256.0, seed=2); obs = observe_from_view(prod, box=256.0, n_grid=32, position_columns=("x", "y", "z"), bias=1.5)
    rec = get_engine("wiener")().reconstruct(obs, cosmo=COSMO, z=1.0); m = recover_metrics(rec, truth, box=256.0); print("closed-loop r", np.mean(m.r[m.k < 0.1])); assert np.mean(m.r[m.k < 0.1]) > 0.6


def test_run_engine_to_store_store_to_store(tmp_path):
    st = run_engine_to_store(get_engine("linear")(), store_root=tmp_path / "s", name="lin", cosmo=COSMO, box=64.0, n_grid=16, z=0.0, seed=0)
    st2 = run_engine_to_store(get_engine("fastpm")(), store_root=tmp_path / "s", name="pm", cosmo=COSMO, box=64.0, n_grid=16, z=0.0, ic_from=(st, "delta", 0.0))
    assert "delta" in st2.products and st2.manifest.parent == "lin"
    assert st2.manifest.lineage["ic_from"] == {"store": "lin", "product": "delta", "z": 0.0}
    assert st.manifest.provenance.code == "linear" and st.manifest.provenance.seed == 0


def test_run_engine_to_store_refuses_existing_and_wrong_engine(tmp_path):
    kw = {"store_root": tmp_path / "s", "cosmo": COSMO, "box": 64.0, "n_grid": 8, "z": 0.0, "seed": 0}
    run_engine_to_store(get_engine("linear")(), name="a", **kw)
    with pytest.raises(ValidationError):
        run_engine_to_store(get_engine("linear")(), name="a", **kw)
    with pytest.raises(ValidationError):
        run_engine_to_store(get_engine("wiener")(), name="b", **kw)


def test_default_nbar_with_mask_is_in_mask_density(tmp_path):
    from oneuniverse.reference.pm import deposit_cic
    v = _view(tmp_path)
    mask = mask_slab(16, 0, 0.0, 0.5)
    obs = observe_from_view(v, box=100.0, n_grid=16, position_columns=("x", "y", "zb"), mask=mask)
    cols = v.table(columns=["x", "y", "zb"])
    n = deposit_cic(np.stack([cols[c].to_numpy() for c in ("x", "y", "zb")], 1), 16, 100.0)
    expect = n[mask].sum() / (mask.sum() * (100.0 / 16) ** 3)
    assert obs.nbar == pytest.approx(expect, rel=1e-12) and obs.nbar == pytest.approx(20_000 / 100.0**3, rel=0.05)
    given = observe_from_view(v, box=100.0, n_grid=16, position_columns=("x", "y", "zb"), mask=mask, nbar=3e-3)
    assert given.nbar == 3e-3


def test_ic_from_nonzero_z_refused(tmp_path):
    st = run_engine_to_store(get_engine("linear")(), store_root=tmp_path / "s", name="lin", cosmo=COSMO, box=64.0, n_grid=8, z=1.0, seed=0)
    with pytest.raises(ValidationError, match="z=0 linear"):
        run_engine_to_store(get_engine("linear")(), store_root=tmp_path / "s", name="child", cosmo=COSMO, box=64.0, n_grid=8, z=0.0, ic_from=(st, "delta", 1.0))
    assert not (tmp_path / "s" / "child").exists()


def test_product_positions_all_from_extras(tmp_path):
    p = product_from_counts(np.ones((4, 4, 4), dtype=int), box=64.0, seed=0)
    partial = type(p)(p.catalog, extras={"x": p.extras["x"], "y": p.extras["y"]}, provenance=dict(p.provenance))
    with pytest.raises(ValidationError, match=r"\['z'\]"):
        observe_from_view(partial, box=64.0, n_grid=4)
    assert observe_from_view(p, box=64.0, n_grid=4).delta_g.shape == (4,) * 3


def test_observe_mask_validated(tmp_path):
    p = product_from_counts(np.ones((4, 4, 4), dtype=int), box=64.0, seed=0)
    with pytest.raises(ValidationError):
        observe_from_view(p, box=64.0, n_grid=4, mask=np.full((4,) * 3, 0.5))
    with pytest.raises(ValidationError):
        observe_from_view(p, box=64.0, n_grid=4, mask=np.ones((3,) * 3, dtype=bool))
    ok = observe_from_view(p, box=64.0, n_grid=4, mask=np.ones((4,) * 3))      # 0/1 floats accepted
    assert ok.mask.dtype == bool


def test_observation_identity_semantics():
    from oneuniverse.twin.engine import Observation
    a = Observation(np.zeros((2, 2, 2)), 1e-3, 1.0)
    b = Observation(np.zeros((2, 2, 2)), 1e-3, 1.0)
    assert a != b and a is not b and isinstance(hash(a), int) and {a: 1}[a] == 1
