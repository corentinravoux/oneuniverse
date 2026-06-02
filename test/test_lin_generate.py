"""Phase S3 T8 — generate_linear_sim native-layout writer."""
import numpy as np
import pyarrow.parquet as pq
import yaml

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.generate import generate_linear_sim


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
        t_cmb=2.7255,
    )


def test_writes_config_and_products(tmp_path):
    out = generate_linear_sim(
        tmp_path / "linsim", _cosmo(),
        box_size=200.0, n_grid=16, redshifts=(0.0, 0.5), seed=11,
    )
    assert (out / "config.yaml").is_file()
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    assert cfg["n_grid"] == 16
    assert cfg["redshifts"] == [0.0, 0.5]
    for ztag in ("z0.000", "z0.500"):
        assert (out / ztag / "field.npy").is_file()
        assert (out / ztag / "particles.npy").is_file()
        assert (out / ztag / "halos.parquet").is_file()


def test_field_and_particle_shapes(tmp_path):
    out = generate_linear_sim(
        tmp_path / "linsim", _cosmo(),
        box_size=200.0, n_grid=16, redshifts=(0.0,), seed=11,
    )
    field = np.load(out / "z0.000" / "field.npy")
    assert field.shape == (16, 16, 16)
    parts = np.load(out / "z0.000" / "particles.npy")
    # (n^3, 6): x,y,z,vx,vy,vz
    assert parts.shape == (16 ** 3, 6)
    halos = pq.read_table(out / "z0.000" / "halos.parquet")
    assert "mass" in halos.column_names


def test_deterministic(tmp_path):
    a = generate_linear_sim(
        tmp_path / "a", _cosmo(),
        box_size=200.0, n_grid=16, redshifts=(0.0,), seed=99,
    )
    b = generate_linear_sim(
        tmp_path / "b", _cosmo(),
        box_size=200.0, n_grid=16, redshifts=(0.0,), seed=99,
    )
    fa = np.load(a / "z0.000" / "field.npy")
    fb = np.load(b / "z0.000" / "field.npy")
    np.testing.assert_array_equal(fa, fb)
