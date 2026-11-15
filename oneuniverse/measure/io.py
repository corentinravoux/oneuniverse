"""Versioned, atomic save/load of a MeasurementSet.

Layout (``MEASURE_FORMAT_VERSION`` 1.0.0)::

    path/measurement_set.json
    path/regions.npy
    path/products/<name>/catalog.parquet [randoms.parquet] [window.npz] [nz.npz]
                         [values.npy mask.npy]   (FieldMap)
                         extras/<key>.npy | <key>.parquet

The whole tree is built in ``<path>.tmp`` and published by rename, so a
failed save never leaves a partial set. Nothing is dropped: all product
kinds, both extras types, window systematics, n(z) and provenance round-trip.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse._core.fs import atomic_dir, atomic_write_json
from oneuniverse.measure.atoms import Nz, Regions, Window
from oneuniverse.measure.measurement_set import MeasurementSet, MeasurementSpec
from oneuniverse.measure.product import FieldMap, Product, Sightline

MEASURE_FORMAT_VERSION = "1.0.0"

_JSON = "measurement_set.json"
_SYS = "sys__"
_KINDS = {"pointset": Product, "fieldmap": FieldMap, "sightline": Sightline}


def _check_name(name: str) -> None:
    if not name or name.startswith(".") or "/" in name or "\\" in name or "\0" in name:
        raise ValidationError(f"product name {name!r} is not usable as a directory name")


def save(ms: MeasurementSet, path, *, overwrite: bool = False) -> None:
    """Write *ms* to the directory *path* (FormatError if it exists and not *overwrite*)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FormatError(f"{path} already exists; pass overwrite=True to replace it")
    for name in ms.products:
        _check_name(name)
    with atomic_dir(path) as root:
        reg = ms.regions
        np.save(root / "regions.npy", reg.pixel_region, allow_pickle=False)
        products = {}
        for name, p in ms.products.items():
            products[name] = _save_product(p, root / "products" / name)
        atomic_write_json(root / _JSON, {
            "format_version": MEASURE_FORMAT_VERSION,
            "spec": {"name": ms.spec.name, "tracers": list(ms.spec.tracers),
                     "statistics": list(ms.spec.statistics),
                     "pairs": [list(q) for q in ms.spec.pairs],
                     "covariance": ms.spec.covariance, "notes": ms.spec.notes},
            "regions": {"nside": int(reg.nside), "nest": bool(reg.nest),
                        "n_regions": int(reg.n_regions), "method": reg.method},
            "products": products,
        })


def _save_product(p: Product, d: Path) -> dict:
    (d / "extras").mkdir(parents=True)
    pq.write_table(p.catalog, d / "catalog.parquet")
    meta: dict = {"kind": p.kind, "has_randoms": p.randoms is not None,
                  "has_window": p.window is not None, "has_nz": p.nz is not None,
                  "provenance": p.provenance}
    if p.randoms is not None:
        pq.write_table(p.randoms, d / "randoms.parquet")
    if p.window is not None:
        w = p.window
        np.savez(d / "window.npz", mask=w.mask,
                 **{_SYS + k: v for k, v in w.systematics.items()})
        meta["window"] = {"nside": int(w.nside), "nest": bool(w.nest), "source": w.source,
                          "systematics": list(w.systematics)}
    if p.nz is not None:
        np.savez(d / "nz.npz", edges=p.nz.edges, counts=p.nz.counts)
        meta["nz"] = {"method": p.nz.method,
                      "bins": None if p.nz.bins is None else [int(b) for b in p.nz.bins]}
    extras = {}
    for k, v in p.extras.items():
        if isinstance(v, np.ndarray):
            try:
                np.save(d / "extras" / f"{k}.npy", v, allow_pickle=False)
            except ValueError as e:
                raise TypeError(f"extras[{k!r}] array cannot be saved without pickle: {e}") from e
            extras[k] = "npy"
        elif isinstance(v, pa.Table):
            pq.write_table(v, d / "extras" / f"{k}.parquet")
            extras[k] = "parquet"
        else:
            raise TypeError(f"extras[{k!r}] must be a numpy array or pyarrow Table, "
                            f"got {type(v).__name__}")
    meta["extras"] = extras
    if isinstance(p, FieldMap):
        np.save(d / "values.npy", p.values, allow_pickle=False)
        np.save(d / "mask.npy", p.mask, allow_pickle=False)
        meta["fieldmap"] = {"nside": int(p.nside), "nest": bool(p.nest)}
    return meta


def _need(path: Path) -> Path:
    if not path.is_file():
        raise FormatError(f"declared file {path} is missing")
    return path


def load(path) -> MeasurementSet:
    """Read a MeasurementSet written by :func:`save`."""
    path = Path(path)
    jpath = path / _JSON
    if not jpath.is_file():
        raise FormatError(f"{path} is not a measurement set (no {_JSON})")
    try:
        d = json.loads(jpath.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise FormatError(f"{jpath} is not valid JSON: {e}") from e
    try:
        version = str(d["format_version"])
    except (KeyError, TypeError) as e:
        raise FormatError(f"{jpath} has no format_version") from e
    if version.split(".")[0] != MEASURE_FORMAT_VERSION.split(".")[0]:
        raise FormatError(f"{path} has format {version}; this reader handles "
                          f"{MEASURE_FORMAT_VERSION.split('.')[0]}.x")
    try:
        s, r = d["spec"], d["regions"]
        spec = MeasurementSpec(s["name"], tuple(s["tracers"]), tuple(s["statistics"]),
                               tuple(tuple(q) for q in s["pairs"]), s["covariance"],
                               dict(s["notes"]))
        pixel_region = np.load(_need(path / "regions.npy"), allow_pickle=False)
        regions = Regions(r["nside"], r["nest"], r["n_regions"], pixel_region, r["method"])
        products = {name: _load_product(path / "products" / name, m)
                    for name, m in d["products"].items()}
    except (KeyError, TypeError) as e:
        raise FormatError(f"{jpath} is malformed: {e!r}") from e
    return MeasurementSet(products, spec, regions)


def _load_product(d: Path, m: dict) -> Product:
    cls = _KINDS.get(m["kind"])
    if cls is None:
        raise FormatError(f"unknown product kind {m['kind']!r}")
    kw: dict = {"catalog": pq.read_table(_need(d / "catalog.parquet")),
                "provenance": m["provenance"]}
    if m["has_randoms"]:
        kw["randoms"] = pq.read_table(_need(d / "randoms.parquet"))
    if m["has_window"]:
        wm = m["window"]
        with np.load(_need(d / "window.npz"), allow_pickle=False) as z:
            kw["window"] = Window(wm["nside"], wm["nest"], z["mask"], wm["source"],
                                  {k: z[_SYS + k] for k in wm["systematics"]})
    if m["has_nz"]:
        nm = m["nz"]
        with np.load(_need(d / "nz.npz"), allow_pickle=False) as z:
            kw["nz"] = Nz(z["edges"], z["counts"], nm["method"],
                          None if nm["bins"] is None else tuple(nm["bins"]))
    extras = {}
    for k, fmt in m["extras"].items():
        if fmt == "npy":
            extras[k] = np.load(_need(d / "extras" / f"{k}.npy"), allow_pickle=False)
        elif fmt == "parquet":
            extras[k] = pq.read_table(_need(d / "extras" / f"{k}.parquet"))
        else:
            raise FormatError(f"extras[{k!r}] has unknown storage {fmt!r}")
    kw["extras"] = extras
    if cls is FieldMap:
        fm = m["fieldmap"]
        kw.update(values=np.load(_need(d / "values.npy"), allow_pickle=False),
                  mask=np.load(_need(d / "mask.npy"), allow_pickle=False),
                  nside=fm["nside"], nest=fm["nest"])
    return cls(**kw)
