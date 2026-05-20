"""End-to-end smoke: HEALPix sys × BOSS composite × FKP composition."""
from __future__ import annotations

import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.healpix_maps import make_systematic_map  # noqa: E402

from oneuniverse.combine.weights import (  # noqa: E402
    FKPWeight, FiberCollisionWeight, HealpixMapWeight, ZFailureWeight,
    boss_total_weight,
)


def test_full_chain_sysmap_times_boss_times_fkp():
    nside = 32
    m = make_systematic_map(nside, seed=0)
    hpw = HealpixMapWeight(
        nside=nside, map_array=m, nest=True, name="w_sys_map",
    )
    boss = boss_total_weight(
        w_sys=hpw,
        w_cp=FiberCollisionWeight("w_cp"),
        w_noz=ZFailureWeight("w_noz"),
    )

    def _nbar(z):
        return np.full_like(z, 1e-4, dtype=np.float64)

    fkp = FKPWeight(nbar=_nbar, P0=1e4, z_column="z")
    composed = boss * fkp

    df = pd.DataFrame({
        "ra": [10.0, 100.0, 250.0],
        "dec": [0.0, 20.0, -10.0],
        "z": [0.5, 0.7, 0.3],
        "w_cp": [1.0, 1.4, 1.1],
        "w_noz": [1.0, 1.1, 0.9],
    })
    got = composed(df)

    theta = np.radians(90.0 - df["dec"].to_numpy())
    phi = np.radians(df["ra"].to_numpy())
    sys_vals = m[hp.ang2pix(nside, theta, phi, nest=True)]
    boss_vals = sys_vals * (
        df["w_cp"].to_numpy() + df["w_noz"].to_numpy() - 1.0
    )
    fkp_vals = 1.0 / (1.0 + 1e-4 * 1e4)
    np.testing.assert_allclose(got, boss_vals * fkp_vals)
