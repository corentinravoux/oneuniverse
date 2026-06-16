"""Review cosmetic items S4 (strict scan), S6 (typed atom slots),
S8 (typed covariance), S11 (store_layout sidecar)."""
import json

import pytest


# ── S6: PointSet/Sightline atom slots are typed, not bare `object` ──────────
def test_s6_pointset_slots_typed():
    from oneuniverse.measure.dataproduct import DataProduct, PointSet, Sightline
    # __future__ annotations => annotations are strings (quote style normalised).
    assert PointSet.__annotations__["nz"] != "object"
    assert "Nz" in PointSet.__annotations__["nz"]
    assert "Window" in PointSet.__annotations__["window"]
    assert "ProbabilisticRedshift" in PointSet.__annotations__["photoz"]
    assert "NamedWeights" in PointSet.__annotations__["weights"]
    assert "NamedWeights" in Sightline.__annotations__["weights"]
    assert "CovariancePlan" in DataProduct.__annotations__["covariance"]
    for slot in ("nz", "window", "photoz", "weights"):
        assert PointSet.__annotations__[slot] != "object"


# ── S8: MeasurementSpec.covariance is Union[str, CovariancePlan] ────────────
def test_s8_covariance_typed():
    from oneuniverse.measure.spec import MeasurementSpec
    ann = MeasurementSpec.__annotations__["covariance"]
    assert ann != "object"
    assert "str" in ann and "CovariancePlan" in ann
    # default still works
    spec = MeasurementSpec(tracers=("g",), pairs=(("g", "g"),),
                           statistic="pk_multipole", estimator_family="clustering")
    assert spec.covariance == "jackknife"


# ── S4: Database.scan(strict=True) re-raises on a corrupt manifest ──────────
def test_s4_strict_scan_raises_on_corrupt(tmp_path):
    from oneuniverse.data.database import OneuniverseDatabase
    bad = tmp_path / "broken" / "oneuniverse"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{ this is not valid json ")
    db = OneuniverseDatabase(tmp_path)
    # lenient (default): corrupt dir skipped, no raise
    db.scan()
    assert "broken" not in db.list()
    # strict: re-raises
    with pytest.raises(Exception):
        db.scan(strict=True)


# ── S11: store_layout lives in a sidecar, not the manifest (with fallback) ──
def test_s11_layout_in_sidecar(tmp_path):
    from oneuniverse.simulation.cosmology import CosmologySpec
    from oneuniverse.simulation.linear import generate_linear_sim
    from oneuniverse.simulation.oufsim import write_oufsim_store, SimStore
    from oneuniverse.simulation.oufsim._layout import (
        STORE_LAYOUT_FILENAME, read_store_layout,
    )
    cosmo = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67,
                          n_s=0.96, sigma8=0.81, t_cmb=2.7255)
    native = generate_linear_sim(tmp_path / "lin", cosmo, box_size=100.0,
                                 n_grid=16, redshifts=(0.0,), seed=1)
    store = write_oufsim_store(native, tmp_path / "store", sim_name="d")
    # sidecar exists; manifest no longer carries the layout
    assert (store / STORE_LAYOUT_FILENAME).exists()
    manifest = json.loads((store / "manifest.json").read_text())
    assert "store_layout" not in manifest
    # SimStore still resolves the layout
    assert SimStore(store).layout == read_store_layout(store)
    assert SimStore(store).layout  # non-empty


def test_s11_back_compat_legacy_manifest(tmp_path):
    """A pre-S11 store (layout embedded in manifest, no sidecar) still reads."""
    from oneuniverse.simulation.oufsim._layout import read_store_layout
    store = tmp_path / "legacy"
    store.mkdir()
    (store / "manifest.json").write_text(json.dumps(
        {"store_layout": {"snapshots": {"0.0": "chunks"}}, "products": []}))
    assert read_store_layout(store) == {"snapshots": {"0.0": "chunks"}}
