"""Periodic buffered resimulation and TreePM coupling (``oneuniverse.reference.resim``)."""
import numpy as np
import pytest

from oneuniverse._core.errors import ValidationError
from oneuniverse._core.geometry import Cube
from oneuniverse.reference.fieldstats import cell_corr
from oneuniverse.reference.linear import gaussian_field
from oneuniverse.reference.resim import (
    buffer_convergence,
    extract_buffered,
    inner_reference,
    run_buffered,
    run_full_reference,
    run_full_reference_from_ic,
    run_treepm,
)
from oneuniverse.simulation.specs import CosmologySpec

COSMO = CosmologySpec(0.6736, 0.3153, 0.0493, 0.8111, 0.9649)


# ---- brief tests (verbatim) ------------------------------------------------

def test_extract_buffered_wraps_periodically():
    ic = np.arange(64**3, dtype=float).reshape(64, 64, 64); sub, sub_box = extract_buffered(ic, box=128.0, cube=Cube((0, 0, 0), (32, 32, 32)), buffer=16.0)
    assert sub.shape == (32, 32, 32) and sub_box == 64.0 and sub[0, 0, 0] == ic[-8, -8, -8] and sub[8, 8, 8] == ic[0, 0, 0]


@pytest.mark.slow
def test_run_buffered_periodic_shift_invariance_ensemble():
    # Owner decision 2026-11-15 (deviation from design §9b / the brief's test): the
    # buffered accuracy is region-dependent (cell_corr 0.20-0.61 at buffer 16), so
    # the original edge-vs-different-centre comparison measured region variance, not
    # periodicity. Here the SAME region is resimulated at the box edge and, after
    # np.roll by +24 cells on every axis, at the box centre; each run is compared
    # with the full reference of its own (unrolled / rolled) IC field.
    box, n = 128.0, 64; diffs = []
    edge_cube, centre_cube = Cube((0, 0, 0), (32, 32, 32)), Cube((48, 48, 48), (80, 80, 80))
    for seed in range(5):
        ic = gaussian_field(COSMO, box=box, n_grid=n, seed=seed)
        ic_rolled = np.roll(ic, (24, 24, 24), axis=(0, 1, 2))     # edge region -> [48, 80) Mpc/h
        edge = run_buffered(COSMO, ic, box=box, cube=edge_cube, buffer=16.0, z_start=9.0, z_end=0.0, n_steps=10)
        cen = run_buffered(COSMO, ic_rolled, box=box, cube=centre_cube, buffer=16.0, z_start=9.0, z_end=0.0, n_steps=10)
        assert edge["n_particles"] == 32**3 and cen["n_particles"] == 32**3
        full, _ = run_full_reference_from_ic(COSMO, ic, box=box, n_grid=n, z_start=9.0, z_end=0.0, n_steps=10)
        full_rolled, _ = run_full_reference_from_ic(COSMO, ic_rolled, box=box, n_grid=n, z_start=9.0, z_end=0.0, n_steps=10)
        c_e = cell_corr(edge["inner"], inner_reference(full, box=box, cube=edge_cube))
        c_c = cell_corr(cen["inner"], inner_reference(full_rolled, box=box, cube=centre_cube))
        diffs.append(abs(c_e - c_c))
        print(f"seed={seed} c_edge={c_e:.6f} c_shifted_centre={c_c:.6f} |diff|={diffs[-1]:.2e}")
    print(f"mean |diff| = {np.mean(diffs):.2e}")
    assert np.mean(diffs) < 0.03


@pytest.mark.slow
def test_treepm_beats_buffered_at_small_buffer():
    box, n = 128.0, 64; gains = []
    for seed in range(3):
        ic = gaussian_field(COSMO, box=box, n_grid=n, seed=seed); cube = Cube((32, 32, 32), (64, 64, 64)); kw = dict(box=box, cube=cube, buffer=8.0, z_start=9.0, z_end=0.0, n_steps=10)  # noqa: C408
        full, _ = run_full_reference_from_ic(COSMO, ic, box=box, n_grid=n, z_start=9.0, z_end=0.0, n_steps=10); ref = inner_reference(full, box=box, cube=cube)
        gains.append(cell_corr(run_treepm(COSMO, ic, **kw)["inner"], ref) - cell_corr(run_buffered(COSMO, ic, **kw)["inner"], ref))
        print(f"seed={seed} gain={gains[-1]:.4f}")
    print(f"mean gain = {np.mean(gains):.4f}")
    assert np.mean(gains) >= 0.1


def test_buffer_convergence_table_shape():
    ic = gaussian_field(COSMO, box=64.0, n_grid=32, seed=0); t = buffer_convergence(COSMO, ic, box=64.0, cube=Cube((16, 16, 16), (32, 32, 32)), buffers=[4.0, 8.0], method="buffered", z_start=9.0, z_end=0.0, n_steps=4)
    assert t.column_names == ["buffer", "cell_corr", "r_lowk"] and t.num_rows == 2


# ---- extra unit tests ------------------------------------------------------

def _small_ic(seed=0):
    return gaussian_field(COSMO, box=64.0, n_grid=16, seed=seed)       # cell 4 Mpc/h


RUN_KW = {"z_start": 9.0, "z_end": 0.0, "n_steps": 3}


def test_extract_buffered_matches_rolled_slice_and_centre():
    ic = np.random.default_rng(1).standard_normal((16, 16, 16))
    sub, sub_box = extract_buffered(ic, box=64.0, cube=Cube((56, 0, 24), (72, 16, 40)), buffer=8.0)
    # start cells (56-8)/4 = 12, (0-8)/4 = -2, (24-8)/4 = 4; n_sub = (16+16)/4 = 8
    expect = np.roll(ic, (-12, 2, -4), axis=(0, 1, 2))[:8, :8, :8]
    assert sub_box == 32.0 and np.array_equal(sub, expect)
    centre, _ = extract_buffered(ic, box=64.0, cube=Cube((16, 16, 16), (32, 32, 32)), buffer=0.0)
    assert np.array_equal(centre, ic[4:8, 4:8, 4:8])


def test_inner_reference_is_periodic_cube_slice():
    d = np.random.default_rng(2).standard_normal((16, 16, 16))
    got = inner_reference(d, box=64.0, cube=Cube((-8, 56, 0), (8, 72, 16)))
    assert np.array_equal(got, np.roll(d, (2, -14, 0), axis=(0, 1, 2))[:4, :4, :4])
    assert np.array_equal(inner_reference(d, box=64.0, cube=Cube((8, 8, 8), (24, 24, 24))),
                          extract_buffered(d, box=64.0, cube=Cube((8, 8, 8), (24, 24, 24)), buffer=0.0)[0])


@pytest.mark.parametrize("cube, buffer", [
    (Cube((1, 0, 0), (17, 16, 16)), 8.0),      # lo off the cell grid
    (Cube((0, 0, 0), (16, 16, 16)), 3.0),      # buffer off the cell grid
    (Cube((0, 0, 0), (16, 16, 20)), 8.0),      # not cubic
    (Cube((0, 0, 0), (48, 48, 48)), 12.0),     # buffered tile larger than the box
    (Cube((0, 0, 0), (16, 16, 16)), -4.0),     # negative buffer
])
def test_extract_buffered_rejects_bad_geometry(cube, buffer):
    with pytest.raises(ValidationError):
        extract_buffered(_small_ic(), box=64.0, cube=cube, buffer=buffer)


def test_run_buffered_is_translation_covariant():
    """Rolling the parent field by s cells and moving the cube by s·cell gives the same tile run."""
    ic = _small_ic()
    a = run_buffered(COSMO, ic, box=64.0, cube=Cube((0, 0, 0), (16, 16, 16)), buffer=8.0, **RUN_KW)
    b = run_buffered(COSMO, np.roll(ic, (5, 7, 3), axis=(0, 1, 2)), box=64.0,
                     cube=Cube((20, 28, 12), (36, 44, 28)), buffer=8.0, **RUN_KW)
    assert a["n_particles"] == b["n_particles"] == 8 ** 3 and a["sub_box"] == 32.0
    assert a["inner"].shape == (4, 4, 4)
    np.testing.assert_allclose(a["inner"], b["inner"], atol=1e-10)


def test_run_buffered_resolution_factor_refines_particles():
    ic = _small_ic()
    out = run_buffered(COSMO, ic, box=64.0, cube=Cube((16, 16, 16), (32, 32, 32)), buffer=8.0,
                       resolution_factor=2, **RUN_KW)
    assert out["n_particles"] == (8 * 2) ** 3 and out["inner"].shape == (4, 4, 4)
    base = run_buffered(COSMO, ic, box=64.0, cube=Cube((16, 16, 16), (32, 32, 32)), buffer=8.0, **RUN_KW)
    assert cell_corr(out["inner"], base["inner"]) > 0.9
    for bad in (0, 1.5, -1, True):
        with pytest.raises(ValidationError):
            run_buffered(COSMO, ic, box=64.0, cube=Cube((16, 16, 16), (32, 32, 32)), buffer=8.0,
                         resolution_factor=bad, **RUN_KW)


def test_fourier_refine_is_exact_for_band_limited_field():
    from oneuniverse.reference.resim import _fourier_refine
    n, box, r = 8, 8.0, 3
    c = (np.arange(n) + 0.5) * (box / n)
    x = c[:, None, None]; y = c[None, :, None]; z = c[None, None, :]
    f = lambda x, y, z: np.cos(2 * np.pi * x / box + 0.3) * np.sin(4 * np.pi * y / box) + np.cos(2 * np.pi * 3 * z / box)
    fine_c = (np.arange(n * r) + 0.5) * (box / (n * r))
    fx = fine_c[:, None, None]; fy = fine_c[None, :, None]; fz = fine_c[None, None, :]
    np.testing.assert_allclose(_fourier_refine(f(x, y, z), r), f(fx, fy, fz), atol=1e-12)
    g = np.random.default_rng(0).standard_normal((n, n, n))
    assert np.array_equal(_fourier_refine(g, 1), g)
    refined = _fourier_refine(g, 2)
    assert refined.dtype == np.float64 and refined.shape == (16, 16, 16)
    assert refined.mean() == pytest.approx(g.mean(), abs=1e-12)


def test_run_treepm_keys_default_split_and_translation_covariance():
    ic = _small_ic()
    a = run_treepm(COSMO, ic, box=64.0, cube=Cube((0, 0, 0), (16, 16, 16)), buffer=8.0, **RUN_KW)
    assert set(a) == {"inner", "n_particles", "sub_box", "k_split"}
    assert a["k_split"] == pytest.approx(2 * 2 * np.pi / 32.0) and a["n_particles"] == 8 ** 3
    b = run_treepm(COSMO, np.roll(ic, (5, 7, 3), axis=(0, 1, 2)), box=64.0,
                   cube=Cube((20, 28, 12), (36, 44, 28)), buffer=8.0, **RUN_KW)
    np.testing.assert_allclose(a["inner"], b["inner"], atol=1e-10)
    c = run_treepm(COSMO, ic, box=64.0, cube=Cube((0, 0, 0), (16, 16, 16)), buffer=8.0, k_split=0.5, **RUN_KW)
    assert c["k_split"] == 0.5
    for bad in (0.0, -1.0, np.inf):
        with pytest.raises(ValidationError):
            run_treepm(COSMO, ic, box=64.0, cube=Cube((0, 0, 0), (16, 16, 16)), buffer=8.0, k_split=bad, **RUN_KW)


def test_run_treepm_whole_box_tile_tracks_full_reference():
    """A tile with buffer reaching the whole box couples to the full linear force: close to the full PM."""
    ic = _small_ic(3)
    full, _ = run_full_reference_from_ic(COSMO, ic, box=64.0, n_grid=16, **RUN_KW)
    cube = Cube((16, 16, 16), (48, 48, 48))
    ref = inner_reference(full, box=64.0, cube=cube)
    t = run_treepm(COSMO, ic, box=64.0, cube=cube, buffer=16.0, **RUN_KW)
    assert t["n_particles"] == 16 ** 3 and cell_corr(t["inner"], ref) > 0.9


def test_run_full_reference_seed_matches_from_ic():
    d1, ic = run_full_reference(COSMO, box=64.0, n_grid=16, seed=4, **RUN_KW)
    d2, ic2 = run_full_reference_from_ic(COSMO, ic, box=64.0, n_grid=16, **RUN_KW)
    assert np.array_equal(ic, gaussian_field(COSMO, box=64.0, n_grid=16, seed=4)) and ic2 is ic
    np.testing.assert_array_equal(d1, d2)
    assert d1.shape == (16, 16, 16) and abs(d1.mean()) < 1e-12
    with pytest.raises(ValidationError):
        run_full_reference_from_ic(COSMO, ic, box=64.0, n_grid=32, **RUN_KW)


def test_buffer_convergence_values_and_validation():
    ic = _small_ic()
    cube = Cube((16, 16, 16), (32, 32, 32))
    t = buffer_convergence(COSMO, ic, box=64.0, cube=cube, buffers=[4.0, 8.0], method="treepm",
                           k_split=0.6, **RUN_KW)
    assert t.column_names == ["buffer", "cell_corr", "r_lowk"]
    assert t.column("buffer").to_pylist() == [4.0, 8.0]
    full, _ = run_full_reference_from_ic(COSMO, ic, box=64.0, n_grid=16, **RUN_KW)
    direct = run_treepm(COSMO, ic, box=64.0, cube=cube, buffer=8.0, k_split=0.6, **RUN_KW)["inner"]
    assert t.column("cell_corr")[1].as_py() == pytest.approx(
        cell_corr(direct, inner_reference(full, box=64.0, cube=cube)))
    assert all(-1.0 <= v <= 1.0 for v in t.column("r_lowk").to_pylist())
    with pytest.raises(ValidationError):
        buffer_convergence(COSMO, ic, box=64.0, cube=cube, buffers=[4.0], method="zoom", **RUN_KW)
    with pytest.raises(ValidationError):
        buffer_convergence(COSMO, ic, box=64.0, cube=cube, buffers=[], method="buffered", **RUN_KW)
    with pytest.raises(ValidationError):
        buffer_convergence(COSMO, ic, box=64.0, cube=cube, buffers=[4.0], method="buffered", n_steps=3)
