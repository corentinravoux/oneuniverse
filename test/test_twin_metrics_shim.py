"""S10: twin metrics consolidated into twin.metrics; old paths still re-export."""
import numpy as np


def test_new_module_has_the_metrics():
    from oneuniverse.twin import metrics
    for name in ("cross_correlation", "power_ratio", "recover_metrics",
                 "RecoveryMetrics", "_bin_kgrid", "_bins", "_kf_edges"):
        assert hasattr(metrics, name), name


def test_old_paths_still_import():
    # Compat re-exports — implementations live in twin.metrics now.
    from oneuniverse.twin.verify import (  # noqa: F401
        cross_correlation, power_ratio, _bin_kgrid, _bins,
    )
    from oneuniverse.twin.validation import (  # noqa: F401
        recover_metrics, RecoveryMetrics,
    )
    # same object across the old and new paths (true re-export, not a copy)
    from oneuniverse.twin import metrics
    assert metrics.cross_correlation is cross_correlation
    assert metrics.recover_metrics is recover_metrics


def test_numerics_unchanged():
    from oneuniverse.twin import metrics
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 16, 16))
    k, r = metrics.cross_correlation(a, a, box_size=100.0)
    assert np.allclose(r[np.isfinite(r)], 1.0, atol=1e-6)  # self-correlation = 1
