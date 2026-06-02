"""Phase S5 T5 — toy merger tree."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.halos import find_peaks
from oneuniverse.simulation.linear.tree import build_merger_tree


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _halos():
    c = _cosmo()
    out = {}
    for z in (0.0, 1.0):
        d = generate_density_field(c, box_size=200.0, n_grid=32, z=z, seed=4)
        out[z] = find_peaks(d, box_size=200.0, threshold=1.0)
    return out


def test_edges_link_adjacent_snapshots():
    tree = build_merger_tree(_halos(), box_size=200.0)
    assert {"descendant_id", "progenitor_id", "z_desc", "z_prog"} <= set(tree)
    assert len(tree["descendant_id"]) > 0
    assert np.all(tree["z_prog"] > tree["z_desc"])   # progenitor at higher z


def test_one_edge_per_descendant():
    halos = _halos()
    tree = build_merger_tree(halos, box_size=200.0)
    # every z=0 halo gets exactly one progenitor edge
    assert len(tree["descendant_id"]) == len(halos[0.0]["halo_id"])


def test_empty_when_single_snapshot():
    halos = _halos()
    tree = build_merger_tree({0.0: halos[0.0]}, box_size=200.0)
    assert len(tree["descendant_id"]) == 0


def test_tree_product_in_store(tmp_path):
    import pyarrow.parquet as pq
    from oneuniverse.simulation.linear import generate_linear_sim
    from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=16, redshifts=(0.0, 1.0), seed=1)
    assert (native / "tree.parquet").is_file()
    store = write_oufsim_store(native, tmp_path / "s", sim_name="d")
    s = SimStore(store)
    assert "tree" in s.products
    t = pq.read_table(store / s.layout["tree"]["dir"] / "part_0000.parquet")
    assert "progenitor_id" in t.column_names
