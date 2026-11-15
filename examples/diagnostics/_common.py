"""Shared helpers of the diagnostic scripts: CLI, matplotlib setup, synthetic inputs.

Synthetic inputs are generated through the public API only (``data.write_dataset``
with synthetic columns, ``reference`` generators); nothing is imported from ``test/``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa

from oneuniverse.data import (
    Database,
    Geometry,
    SurveyMeta,
    stable_object_id,
    write_dataset,
)
from oneuniverse.data.manifest import LoaderInfo

DPI = 130


def parse_args(description: str, *, data_root: bool = True) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", type=Path, default=Path("examples/output"),
                   help="output directory of the PNG (default: examples/output)")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Database root of real OUF datasets (default: synthetic inputs)"
                   if data_root else "unused: this figure is always synthetic")
    p.add_argument("--dataset", action="append", default=None,
                   help="dataset name(s) to use under --data-root (default: first ones)")
    return p.parse_args()


def setup_style() -> None:
    if os.environ.get("PLT_STYLE"):
        plt.style.use(os.environ["PLT_STYLE"])


def suptitle(fig, text: str, args: argparse.Namespace) -> None:
    """Figure title, marked "(synthetic)" when no ``--data-root`` was given."""
    fig.suptitle(text if getattr(args, "data_root", None) else f"{text} (synthetic)")


def save(fig, out: Path, name: str) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path)
    return path


def point_table(n: int, seed: int, *, survey: str = "synthetic", ra_range=(0.0, 360.0),
                dec_range=(-90.0, 90.0), z_range=(0.01, 2.0)) -> pa.Table:
    """Core POINT columns plus ``mag``; dec uniform in sin(dec); ra clustered in blobs."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(*ra_range, n)
    s0, s1 = np.sin(np.radians(dec_range[0])), np.sin(np.radians(dec_range[1]))
    dec = np.degrees(np.arcsin(rng.uniform(s0, s1, n)))
    z = rng.uniform(*z_range, n)
    src = np.arange(n, dtype=np.int64)
    perm = rng.permutation(n)
    ra, dec, z, src = ra[perm], dec[perm], z[perm], src[perm]
    return pa.table({
        "ra": ra, "dec": dec, "z": z,
        "z_type": pa.array(["spec"] * n, pa.string()),
        "z_err": rng.uniform(1e-4, 1e-3, n).astype(np.float32),
        "object_id": stable_object_id(survey, src),
        "survey": pa.array([survey] * n, pa.string()),
        "source_row": src,
        "mag": rng.uniform(16.0, 22.0, n).astype(np.float32),
    })


def write_points(root: Path, name: str, table: pa.Table, *, partition_nside: int = 4):
    return write_dataset(table, Path(root) / name, geometry=Geometry.POINT,
                         survey=SurveyMeta(name, "spectroscopic"),
                         loader=LoaderInfo("examples", "1", {}), partition_nside=partition_nside,
                         extra_stats_columns=["mag"])


def synthetic_database(root: Path, *, n: int = 20_000, n_common: int = 7_000, seed: int = 0,
                       ra_range=(100.0, 220.0), dec_range=(-20.0, 50.0)) -> Database:
    """Surveys "a" and "b" over one footprint; the first *n_common* source rows of b are
    the same objects as rows of a, jittered by 0.4 arcsec (per axis) and 1.5e-3 in z (Gaussian)."""
    kw = {"ra_range": ra_range, "dec_range": dec_range}
    ta = point_table(n, seed, survey="a", **kw)
    tb = point_table(n, seed + 1, survey="b", **kw)
    pos_a = np.argsort(ta["source_row"].to_numpy())
    src_b = tb["source_row"].to_numpy()
    common = src_b < n_common
    ia = pos_a[src_b[common]]
    rng = np.random.default_rng(seed + 2)
    k = int(common.sum())
    cols = {c: tb[c].to_numpy().copy() for c in ("ra", "dec", "z")}
    cols["ra"][common] = ta["ra"].to_numpy()[ia] + rng.normal(0, 0.4, k) / 3600 / np.cos(
        np.radians(ta["dec"].to_numpy()[ia]))
    cols["dec"][common] = ta["dec"].to_numpy()[ia] + rng.normal(0, 0.4, k) / 3600
    cols["z"][common] = ta["z"].to_numpy()[ia] + rng.normal(0, 1.5e-3, k)
    for c, v in cols.items():
        tb = tb.set_column(tb.schema.get_field_index(c), c, pa.array(v))
    write_points(root, "a", ta)
    write_points(root, "b", tb)
    return Database(root)


def pick(db: Database, wanted, count: int) -> list[str]:
    names = list(wanted) if wanted else db.names()[:count]
    if len(names) < count:
        raise SystemExit(f"need {count} dataset(s), found {names}")
    return names[:count]
