"""Phase 21 visual diagnostic — attribute filter cuts colour-mismatched links."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.oneuid_crossmatch import cross_match_surveys  # noqa: E402
from oneuniverse.data.oneuid_rules import CrossMatchRules  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def _color_filter(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    dg = (left["psfmag_g"] - left["psfmag_r"]).to_numpy()
    dr = (right["psfmag_g"] - right["psfmag_r"]).to_numpy()
    return np.abs(dg - dr) < 0.1


def test_phase21_visual(tmp_path):
    rng = np.random.default_rng(0)
    n_per = 200
    ra1 = rng.uniform(10.0, 20.0, n_per)
    ra2 = ra1 + rng.normal(0, 1e-6, n_per)
    col1 = rng.uniform(0.0, 1.0, n_per)
    col2 = col1 + rng.normal(0, 0.05, n_per)
    col2[: n_per // 4] += 0.5  # 25% mismatched colour

    a = pd.DataFrame({
        "ra": ra1, "dec": np.zeros(n_per),
        "z": np.full(n_per, 0.5, dtype="f4"),
        "z_type": np.array(["spec"] * n_per, dtype=object),
        "z_err": np.full(n_per, 0.001, dtype="f4"),
        "galaxy_id": np.arange(n_per, dtype="i8"),
        "_original_row_index": np.arange(n_per, dtype="i8"),
        "psfmag_g": np.full(n_per, 22.0, dtype="f4"),
        "psfmag_r": (22.0 - col1).astype("f4"),
    })
    b = pd.DataFrame({
        "ra": ra2, "dec": np.zeros(n_per),
        "z": np.full(n_per, 0.5, dtype="f4"),
        "z_type": np.array(["spec"] * n_per, dtype=object),
        "z_err": np.full(n_per, 0.001, dtype="f4"),
        "galaxy_id": np.arange(n_per, 2 * n_per, dtype="i8"),
        "_original_row_index": np.arange(n_per, dtype="i8"),
        "psfmag_g": np.full(n_per, 22.0, dtype="f4"),
        "psfmag_r": (22.0 - col2).astype("f4"),
    })
    catalogs = {"a": a, "b": b}

    no_filter = cross_match_surveys(
        catalogs, CrossMatchRules(sky_tol_arcsec=2.0),
    )
    with_filter = cross_match_surveys(
        catalogs,
        CrossMatchRules(
            sky_tol_arcsec=2.0,
            attribute_filters=(_color_filter,),
        ),
    )

    n_no = no_filter.n_multi
    n_yes = with_filter.n_multi

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].scatter(col1, col2, s=8, alpha=0.6, label="all pairs")
    ax[0].plot([0, 1.5], [0, 1.5], "k--", lw=0.8, label="identity")
    ax[0].plot([0, 1.5], [0.1, 1.6], "r--", lw=0.8, label="|Δ|=0.1")
    ax[0].plot([0, 1.5], [-0.1, 1.4], "r--", lw=0.8)
    ax[0].set_xlabel("(g − r)$_a$")
    ax[0].set_ylabel("(g − r)$_b$")
    ax[0].set_title("Per-pair colour distribution")
    ax[0].legend()

    ax[1].bar(
        ["no filter", "colour filter"],
        [n_no, n_yes],
        color=["tab:gray", "tab:blue"],
    )
    ax[1].set_ylabel("multi-survey groups")
    ax[1].set_title(
        f"attribute_filters cut "
        f"{100 * (1 - n_yes / max(n_no, 1)):.0f}% of links"
    )

    fig.tight_layout()
    out_png = OUT / "phase21_attribute_filters.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im:
        assert im.width >= 800 and im.height >= 200
