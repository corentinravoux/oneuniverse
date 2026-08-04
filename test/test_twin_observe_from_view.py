import numpy as np
import pandas as pd
from oneuniverse.twin.observe_from_view import observe_from_view
from oneuniverse.twin.engine import Observation


def test_observe_from_dataframe_positions():
    box, n = 100.0, 16
    rng = np.random.default_rng(0)
    # 3000 uniform-random galaxies -> near-zero overdensity, right shape
    xyz = rng.uniform(0, box, size=(3000, 3))
    df = pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
    obs = observe_from_view(df, box_size=box, n_grid=n, bias=1.4)
    assert isinstance(obs, Observation)
    assert obs.delta_g.shape == (n, n, n)
    assert abs(float(obs.delta_g.mean())) < 1e-9   # delta defined rel. to realised mean
    assert obs.bias == 1.4
    assert obs.nbar > 0


def test_tracer_view_clusters_like_truth(tmp_path):
    from fixtures.tracer_sim import synthetic_tracer_view
    from oneuniverse.twin.metrics import cross_correlation
    box, n = 200.0, 32
    view, truth = synthetic_tracer_view(tmp_path, box_size=box, n_grid=n,
                                        nbar=5e-3, bias=1.5, seed=3)
    obs = observe_from_view(view, box_size=box, n_grid=n, bias=1.5,
                            position_cols=("x", "y", "z_box"))
    # gridded tracers must correlate with the truth field on large scales
    k, r = cross_correlation(obs.delta_g, truth, box_size=box)
    lo = k < 0.15
    assert np.nanmedian(r[lo]) > 0.5
