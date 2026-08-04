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
