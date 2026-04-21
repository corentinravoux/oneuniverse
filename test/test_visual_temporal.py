"""Diagnostic figures for Phase 7. Run locally; skipped in headless CI
if matplotlib isn't importable. Outputs land under test/test_output/."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.converter import write_ouf_dataset  # noqa: E402
from oneuniverse.data.database import OneuniverseDatabase  # noqa: E402
from oneuniverse.data.dataset_view import DatasetView  # noqa: E402
from oneuniverse.data.format_spec import DataGeometry  # noqa: E402
from oneuniverse.data.manifest import LoaderSpec  # noqa: E402
from oneuniverse.data._converter_lightcurve import (  # noqa: E402
    write_ouf_lightcurve_dataset,
)
from oneuniverse.data.validity import DatasetValidity  # noqa: E402


OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _healpix32(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    import healpy as hp
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    return hp.ang2pix(32, theta, phi, nest=True).astype(np.int64)


def test_visual_transient_point(tmp_path):
    rng = np.random.default_rng(1)
    n = 5000
    ra = rng.uniform(0.0, 360.0, n)
    dec = rng.uniform(-60.0, 60.0, n)
    df = pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "z":     rng.uniform(0.01, 0.5, n),
        "z_type": ["spec"] * n,
        "z_err":  rng.uniform(1e-4, 1e-3, n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": [f"syn{i:06d}" for i in range(n)],
        "_original_row_index": np.arange(n, dtype=np.int64),
        "_healpix32": _healpix32(ra, dec),
        "t_obs": rng.uniform(58000.0, 60000.0, n),
    })
    survey_dir = tmp_path / "viz_pt"
    ou_dir = survey_dir / "oneuniverse"
    ou_dir.mkdir(parents=True, exist_ok=True)
    write_ouf_dataset(
        df, ou_dir,
        survey_name="viz_pt", survey_type="transient",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="syn", version="0"),
    )
    view = DatasetView.from_path(survey_dir)
    window = view.read(columns=["ra", "dec", "t_obs"],
                       t_range=(59000.0, 59500.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ra_full = np.where(df["ra"] > 180, df["ra"] - 360, df["ra"])
    axes[0].scatter(np.radians(ra_full), np.radians(df["dec"]),
                    s=1.5, alpha=0.35, color="#888", label="all")
    ra_win = np.where(window["ra"] > 180, window["ra"] - 360, window["ra"])
    axes[0].scatter(np.radians(ra_win), np.radians(window["dec"]),
                    s=3, color="#d62728",
                    label=f"t_range window ({len(window):,})")
    axes[0].set_xlabel("RA (rad)")
    axes[0].set_ylabel("Dec (rad)")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Transient events — window vs full")

    axes[1].hist(df["t_obs"], bins=80, color="#3a7bd5", alpha=0.6, label="all")
    axes[1].axvspan(59000.0, 59500.0, color="#d62728", alpha=0.2,
                    label="window")
    axes[1].set_xlabel("MJD")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].set_title(f"{len(window):,}/{len(df):,} events in window")

    fig.tight_layout()
    out = OUTPUT_DIR / "phase7_temporal_point.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 10_000


def test_visual_lightcurve(tmp_path):
    rng = np.random.default_rng(2)
    n_obj, n_epochs = 10, 30
    objects = pd.DataFrame({
        "object_id": np.arange(n_obj, dtype=np.int64),
        "ra":    rng.uniform(0.0, 360.0, n_obj),
        "dec":   rng.uniform(-60.0, 60.0, n_obj),
        "z":     rng.uniform(0.01, 0.2, n_obj),
        "z_type": ["spec"] * n_obj,
        "z_err":  rng.uniform(1e-4, 1e-3, n_obj),
    })
    rows = []
    for oid in objects["object_id"]:
        mjd = np.sort(rng.uniform(58000.0, 60000.0, n_epochs))
        for i, t in enumerate(mjd):
            rows.append({
                "object_id": int(oid), "mjd": float(t),
                "filter": ("g", "r", "i")[i % 3],
                "flux": float(100 + 20 * np.sin(0.01 * t)
                              + rng.normal(0, 2)),
                "flux_err": 1.0, "flag": 0,
            })
    epochs = pd.DataFrame(rows)
    survey_dir = tmp_path / "viz_lc"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="viz_lc", survey_type="transient",
        loader_name="syn", loader_version="0",
    )
    view = DatasetView.from_path(survey_dir)
    obj = view.objects_table().to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(obj["ra"], obj["dec"], s=50, c="#3a7bd5", edgecolor="k")
    for _, row in obj.iterrows():
        axes[0].annotate(str(row["object_id"]), (row["ra"], row["dec"]))
    axes[0].set_xlabel("RA (deg)")
    axes[0].set_ylabel("Dec (deg)")
    axes[0].set_title(f"{len(obj)} lightcurve objects")

    import pyarrow.compute as pc
    pick = int(obj["object_id"].iloc[0])
    lc = view.scan(
        columns=["mjd", "flux", "filter"],
        filter=pc.field("object_id") == pick,
    ).to_pandas()
    for band, sub in lc.groupby("filter"):
        axes[1].errorbar(sub["mjd"], sub["flux"], yerr=1.0,
                         fmt="o", label=str(band), ms=3)
    axes[1].set_xlabel("MJD")
    axes[1].set_ylabel("flux")
    axes[1].legend()
    axes[1].set_title(f"Lightcurve of object_id={pick}")

    fig.tight_layout()
    out = OUTPUT_DIR / "phase7_temporal_lightcurve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 10_000


def _syn_point(root, sub, name, validity, seed=0, n=200):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 30.0, n)
    dec = rng.uniform(-5.0, 5.0, n)
    df = pd.DataFrame({
        "ra":  ra,
        "dec": dec,
        "z":   rng.uniform(0.1, 0.3, n),
        "z_type": ["spec"] * n,
        "z_err":  rng.uniform(1e-4, 1e-3, n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": [f"{name}_{i:04d}" for i in range(n)],
        "_original_row_index": np.arange(n, dtype=np.int64),
        "_healpix32": _healpix32(ra, dec),
    })
    survey_dir = root / sub
    ou_dir = survey_dir / "oneuniverse"
    ou_dir.mkdir(parents=True, exist_ok=True)
    write_ouf_dataset(
        df, ou_dir, survey_name=name, survey_type=sub.split("/")[0],
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="syn", version="0"),
        validity=validity,
    )


def test_visual_database_snapshot(tmp_path):
    """Two versions of the same survey: show how as_of() changes the
    database contents between two probe timestamps."""
    v1 = DatasetValidity(
        valid_from_utc="2026-01-01T00:00:00+00:00",
        valid_to_utc="2026-06-01T00:00:00+00:00",
        version="dr16",
    )
    v2 = DatasetValidity(
        valid_from_utc="2026-06-01T00:00:00+00:00",
        version="dr17",
        supersedes=("spec_eb_qso_v_dr16",),
    )

    _syn_point(tmp_path, "spec/eb_qso/v_dr16",
               "spec_eb_qso_v_dr16", v1, seed=1, n=150)
    _syn_point(tmp_path, "spec/eb_qso/v_dr17",
               "spec_eb_qso_v_dr17", v2, seed=2, n=300)

    db = OneuniverseDatabase.from_root(tmp_path)

    probes = {
        "March 2026 (v_dr16)":
            dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc),
        "September 2026 (v_dr17)":
            dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc),
    }
    fig, axes = plt.subplots(1, len(probes), figsize=(12, 4), sharey=True)
    for (label, t), ax in zip(probes.items(), axes):
        snap = db.as_of(t)
        for name in snap.list():
            view = DatasetView.from_path(snap.get_path(name))
            d = view.read(columns=["ra", "dec"])
            ax.scatter(d["ra"], d["dec"], s=4, alpha=0.6, label=name)
        ax.set_title(label)
        ax.set_xlabel("RA (deg)")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("Dec (deg)")

    fig.tight_layout()
    out = OUTPUT_DIR / "phase7_database_snapshot.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 10_000
