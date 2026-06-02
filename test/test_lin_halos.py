"""Phase S3 T7 — toy peak halos (halo product)."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.halos import find_peaks


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
    )


def test_returns_expected_columns():
    box, n = 200.0, 32
    d = generate_density_field(_cosmo(), box_size=box, n_grid=n, z=0.0, seed=5)
    halos = find_peaks(d, box_size=box, threshold=1.0)
    for col in ("halo_id", "x", "y", "z", "delta_peak", "mass"):
        assert col in halos


def test_finds_some_peaks():
    box, n = 200.0, 32
    d = generate_density_field(_cosmo(), box_size=box, n_grid=n, z=0.0, seed=5)
    halos = find_peaks(d, box_size=box, threshold=1.0)
    assert len(halos["halo_id"]) > 0


def test_positions_in_box():
    box, n = 200.0, 32
    d = generate_density_field(_cosmo(), box_size=box, n_grid=n, z=0.0, seed=5)
    halos = find_peaks(d, box_size=box, threshold=1.0)
    for ax in ("x", "y", "z"):
        v = np.asarray(halos[ax])
        assert v.min() >= 0.0 and v.max() < box


def test_higher_threshold_fewer_halos():
    box, n = 200.0, 32
    d = generate_density_field(_cosmo(), box_size=box, n_grid=n, z=0.0, seed=5)
    n_low = len(find_peaks(d, box_size=box, threshold=0.5)["halo_id"])
    n_high = len(find_peaks(d, box_size=box, threshold=2.0)["halo_id"])
    assert n_high <= n_low
