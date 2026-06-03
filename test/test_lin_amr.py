"""Phase S7 T1 — toy AMR refinement + IC product."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.amr import refine_field
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.ic import white_noise_ic


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_refines_only_above_threshold():
    c = _cosmo()
    d = generate_density_field(c, box_size=200.0, n_grid=32, z=0.0, seed=4)
    amr = refine_field(d, threshold=1.5)
    assert amr["n_refined"] == int((d > 1.5).sum())
    assert amr["subcells"].shape == (amr["n_refined"], 8)
    assert amr["parent_ix"].max() < 32
    assert len(amr["node_id"]) == amr["n_refined"]


def test_empty_below_threshold():
    c = _cosmo()
    d = generate_density_field(c, box_size=200.0, n_grid=16, z=0.0, seed=4)
    amr = refine_field(d, threshold=1e6)        # nothing above
    assert amr["n_refined"] == 0
    assert amr["subcells"].shape == (0, 8)


def test_ic_reproducible_and_described():
    field, desc = white_noise_ic(_cosmo(), box_size=200.0, n_grid=32, seed=7)
    field2, _ = white_noise_ic(_cosmo(), box_size=200.0, n_grid=32, seed=7)
    np.testing.assert_array_equal(field, field2)
    assert desc["seed"] == 7 and desc["n_grid"] == 32
    assert desc["pk_model"] == "eisenstein_hu_nowiggle"
    assert abs(float(field.mean())) < 0.1      # ~zero-mean white noise


def test_amr_and_ic_in_store(tmp_path):
    import json
    from oneuniverse.simulation.linear import generate_linear_sim
    from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=16, redshifts=(0.0,), seed=1)
    assert (native / "ic_field.npy").is_file()
    store = write_oufsim_store(native, tmp_path / "s", sim_name="d")
    s = SimStore(store)
    assert "fields_amr" in s.layout
    assert "ic_posterior" in s.products
    man = json.load(open(store / "manifest.json"))
    assert man["has_input"] is True
