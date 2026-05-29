"""Phase 19 visual diagnostic — ShearWeight vs raw shape weight + PIP histogram."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.combine.weights.pip import PipBitweightWeight  # noqa: E402
from oneuniverse.combine.weights.shear import ShearWeight  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase19_visual(tmp_path):
    rng = np.random.default_rng(0)
    n = 1000
    shape_w = rng.uniform(0.2, 1.0, n).astype("f4")
    R11 = rng.normal(0.7, 0.05, n).astype("f4")
    R22 = rng.normal(0.7, 0.05, n).astype("f4")
    R_S = rng.normal(0.05, 0.01, n).astype("f4")
    df = pd.DataFrame({
        "shear_weight": shape_w,
        "R11": R11, "R22": R22, "R_S": R_S,
    })
    metacal = ShearWeight(kind="metacal").compute(df)

    rng2 = np.random.default_rng(1)
    bits = rng2.integers(
        np.int64(-(2 ** 62)), np.int64(2 ** 62 - 1), size=n, dtype="i8",
    )
    pip_df = pd.DataFrame({
        "BITWEIGHTS": [np.array([b], dtype="i8") for b in bits],
    })
    pip = PipBitweightWeight().compute(pip_df)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].hist(shape_w, bins=40, color="tab:gray", alpha=0.7,
               label="shape_weight")
    ax[0].hist(metacal, bins=40, color="tab:blue", alpha=0.7,
               label="metacal ShearWeight")
    ax[0].set_xlabel("weight")
    ax[0].set_ylabel("count")
    ax[0].legend()
    ax[0].set_title("Shape weight vs metacal-calibrated weight")

    R_eff = 0.5 * (R11 + R22) + R_S
    ax[1].scatter(R_eff, metacal, s=4, alpha=0.4)
    ax[1].set_xlabel("R_eff")
    ax[1].set_ylabel("metacal ShearWeight")
    ax[1].set_title("Weight inversely scales as R_eff²")

    ax[2].hist(pip, bins=40, color="tab:red", alpha=0.8)
    ax[2].set_xlabel("PIP fraction (set bits / 64)")
    ax[2].set_ylabel("count")
    ax[2].set_title("PipBitweightWeight (fraction mode)")

    fig.tight_layout()
    out_png = OUT / "phase19_shear_and_pip_weights.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im:
        assert im.width >= 800 and im.height >= 200
