"""measure WL-T3 — tomographic n(z)."""
import numpy as np
import pandas as pd

from oneuniverse.measure.nz import Nz
from oneuniverse.measure.tomography import tomographic_nz


def test_tomographic_nz_stacks_per_bin():
    n = 600
    z_grid = np.linspace(0.0, 2.0, 41)
    means = np.where(np.arange(n) < n // 2, 0.4, 1.0)
    cat = pd.DataFrame({"tomo_bin": (np.arange(n) >= n // 2).astype(int),
                        "z": means})

    class _K:                       # minimal kernel stand-in: point masses
        def sample(self, n_per, seed=None):
            return np.repeat(means[:, None], n_per, axis=1)

    nzs = tomographic_nz(cat, _K(), bin_column="tomo_bin", z_grid=z_grid)
    assert set(nzs) == {0, 1}
    assert all(isinstance(v, Nz) for v in nzs.values())
    c0, c1 = nzs[0].centers(), nzs[1].centers()
    assert c0[np.argmax(nzs[0].counts)] < 0.7      # bin 0 low-z
    assert c1[np.argmax(nzs[1].counts)] > 0.7      # bin 1 high-z
