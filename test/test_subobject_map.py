"""Phase 20 T3 — match a point catalog to per-row HEALPix probability maps."""
import healpy as hp
import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.subobject_map import build_subobject_links_to_map


def _gaussian_map(nside: int, ra: float, dec: float, sigma_deg: float):
    npix = hp.nside2npix(nside)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    target = hp.ang2vec(theta, phi)
    pix = np.arange(npix)
    vecs = np.array(hp.pix2vec(nside, pix, nest=True))
    cos_sep = vecs.T @ target
    sep_rad = np.arccos(np.clip(cos_sep, -1.0, 1.0))
    sigma_rad = np.radians(sigma_deg)
    m = np.exp(-0.5 * (sep_rad / sigma_rad) ** 2)
    m /= m.sum()
    return m.astype("f4")


def test_match_at_map_peak_has_high_confidence():
    nside = 32
    parents = pd.DataFrame({
        "oneuid": np.array([0, 1, 2], dtype="i8"),
        "ra":  np.array([10.0, 20.0, 50.0], dtype="f8"),
        "dec": np.array([0.0, 0.0, 0.0], dtype="f8"),
    })
    map_at_p0 = _gaussian_map(nside, 10.0, 0.0, sigma_deg=2.0)
    events = pd.DataFrame({
        "oneuid": np.array([100], dtype="i8"),
        "skymap": [map_at_p0],
    })
    links = build_subobject_links_to_map(
        parents=parents, events=events,
        map_column="skymap", map_nside=nside, map_nest=True,
        threshold=0.0,
    )
    df = links.table
    peak_rows = df[df["parent_oneuid"] == 0]
    other_rows = df[df["parent_oneuid"] != 0]
    assert len(peak_rows) == 1
    assert (peak_rows["confidence"].iloc[0]
            > other_rows["confidence"].max())


def test_threshold_drops_rows_below_cut():
    nside = 32
    parents = pd.DataFrame({
        "oneuid": np.array([0, 1], dtype="i8"),
        "ra":  np.array([0.0, 180.0], dtype="f8"),
        "dec": np.array([0.0, 0.0], dtype="f8"),
    })
    map_at_p0 = _gaussian_map(nside, 0.0, 0.0, sigma_deg=2.0)
    events = pd.DataFrame({
        "oneuid": np.array([100], dtype="i8"),
        "skymap": [map_at_p0],
    })
    above = build_subobject_links_to_map(
        parents=parents, events=events,
        map_column="skymap", map_nside=nside, map_nest=True,
        threshold=1e-3,
    )
    parent_ids = set(above.table["parent_oneuid"].tolist())
    assert 0 in parent_ids
    assert 1 not in parent_ids


def test_rejects_wrong_map_length():
    parents = pd.DataFrame({
        "oneuid": np.array([0], dtype="i8"),
        "ra": np.array([0.0], dtype="f8"),
        "dec": np.array([0.0], dtype="f8"),
    })
    events = pd.DataFrame({
        "oneuid": np.array([100], dtype="i8"),
        "skymap": [np.zeros(7, dtype="f4")],
    })
    with pytest.raises(ValueError, match="length"):
        build_subobject_links_to_map(
            parents=parents, events=events,
            map_column="skymap", map_nside=32,
            threshold=0.0,
        )
