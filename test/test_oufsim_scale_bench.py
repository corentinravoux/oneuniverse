"""S17 T8 — convert peak memory grows sub-linearly; reference << reencode."""
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.oufsim.scale_bench import sweep


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_reference_store_is_tiny_vs_reencode(tmp_path):
    rows = sweep(tmp_path, _cosmo(), grids=(16, 32))
    for r in rows:
        assert r["store_reference_mb"] < 0.5 * r["store_reencode_mb"]
    # peak memory grows slower than particle count's 8x (16->32)
    ratio_mem = rows[1]["convert_peak_mb"] / max(rows[0]["convert_peak_mb"], 1e-6)
    assert ratio_mem < 8.0
