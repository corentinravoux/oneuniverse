"""Phase S12 — merge resimulated tiles into a global field."""
import numpy as np

from oneuniverse.simulation.resim.merge import feather_window, merge_fields


def test_feather_window_partition_of_unity_in_overlap():
    # two 1D-equivalent windows offset by (n - overlap) sum to ~1 in overlap
    w = feather_window((16, 16, 16), feather=4)
    assert w.max() <= 1.0 and w.min() >= 0.0
    assert w[8, 8, 8] == 1.0                       # interior unweighted


def test_split_then_merge_recovers_field():
    rng = np.random.default_rng(0)
    g = rng.standard_normal((32, 32, 32))
    # two overlapping tiles (slices of the same field)
    a = {"field": g[0:20], "origin": (0, 0, 0)}
    b = {"field": g[12:32], "origin": (12, 0, 0)}
    merged = merge_fields((32, 32, 32), [a, b], feather=4)
    # weighted average of identical values = the field, everywhere covered
    np.testing.assert_allclose(merged, g, atol=1e-9)


def test_merge_smooths_disagreeing_tiles():
    # two tiles disagreeing by a constant offset in the overlap -> no sharp seam
    base = np.zeros((40, 8, 8))
    a = {"field": base[0:24] + 1.0, "origin": (0, 0, 0)}
    b = {"field": base[16:40] - 1.0, "origin": (16, 0, 0)}
    hard = merge_fields((40, 8, 8), [a, b], feather=0)
    soft = merge_fields((40, 8, 8), [a, b], feather=8)
    # feathering reduces the maximum gradient along the seam axis
    gh = np.abs(np.diff(hard[:, 4, 4])).max()
    gs = np.abs(np.diff(soft[:, 4, 4])).max()
    assert gs < gh


def test_mass_conservation_on_agreeing_tiles():
    rng = np.random.default_rng(1)
    g = rng.standard_normal((24, 24, 24))
    tiles = [{"field": g[0:16], "origin": (0, 0, 0)},
             {"field": g[8:24], "origin": (8, 0, 0)}]
    merged = merge_fields((24, 24, 24), tiles, feather=3)
    assert abs(merged.sum() - g.sum()) < 1e-6
