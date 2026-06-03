"""Phase S8.3 — region IC extraction + Gate 1 (pre-run consistency)."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.resim import extract_region, gate1_consistency
from oneuniverse.simulation.selectors import Cube


def _parent():
    c = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                      sigma8=0.81, t_cmb=2.7255)
    return generate_density_field(c, box_size=256.0, n_grid=64, z=0.0, seed=3)


def test_extract_region_is_subgrid():
    f = _parent()
    cube = Cube(0, 64, 0, 64, 0, 64)
    sub, origin = extract_region(f, cube, box_size=256.0)
    nx, ny, nz = sub.shape
    np.testing.assert_array_equal(
        sub, f[origin[0]:origin[0] + nx, origin[1]:origin[1] + ny,
               origin[2]:origin[2] + nz])


def test_gate1_passes_for_consistent_ic():
    f = _parent()
    cube = Cube(40, 160, 40, 160, 40, 160)
    sub, _ = extract_region(f, cube, box_size=256.0)
    # the mini IC IS the extracted parent sub-region -> consistent
    res = gate1_consistency(sub, sub, box_size=256.0 * sub.shape[0] / 64)
    assert res["passed"] and res["cell_corr"] > 0.999


def test_gate1_fails_for_scrambled_ic():
    f = _parent()
    cube = Cube(40, 160, 40, 160, 40, 160)
    sub, _ = extract_region(f, cube, box_size=256.0)
    rng = np.random.default_rng(0)
    scrambled = sub.ravel().copy()
    rng.shuffle(scrambled)
    scrambled = scrambled.reshape(sub.shape)
    box_sub = 256.0 * sub.shape[0] / 64
    res = gate1_consistency(scrambled, sub, box_size=box_sub)
    assert not res["passed"]
