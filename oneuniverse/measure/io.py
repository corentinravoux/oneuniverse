"""On-disk form for a MeasurementSet — the P1→P2 handoff across processes.

``save_measurement_set(ms, path)`` writes a self-describing directory:

    {path}/
      manifest.json                  # spec + metadata + per-product index
      {product}/
        catalog.parquet  randoms.parquet  region_map.npy
        nz.json | nz_tomo.json   window.npy + window.json
        weights.npz  photoz.npz  tomo_bin.npy  links.npz
        sightline.npz (Sightline)  | field.npy + field.json (FieldMap)

``load_measurement_set(path)`` reconstructs the object. Tables→parquet,
arrays→npy/npz, scalars/specs→JSON. **Cosmology-free by construction** (the
container carries none). Round-trips the atoms the builders produce; the photo-z
kernel is restored as a light grid+values view (`PhotozArrays`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from oneuniverse.measure.covariance import CovarianceHandle, CovariancePlan
from oneuniverse.measure.dataproduct import FieldMap, PointSet, Sightline
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.nz import Nz
from oneuniverse.measure.photoz import PhotozArrays
from oneuniverse.measure.spec import MeasurementSpec
from oneuniverse.measure.links import SubObjectLinks
from oneuniverse.measure.weighting import NamedWeights
from oneuniverse.measure.window import Window

_FMT = "oneuniverse.measure/1.0"


# ── small helpers ──────────────────────────────────────────────────────────
def _nz_to_dict(nz: Nz) -> dict:
    return {"edges": nz.edges.tolist(), "counts": nz.counts.tolist(),
            "method": nz.method}


def _nz_from_dict(d: dict) -> Nz:
    return Nz(np.asarray(d["edges"], float), np.asarray(d["counts"], float),
              d["method"])


def _meta_to_dict(m: ProductMetadata) -> dict:
    return {"frame": m.frame, "epoch": m.epoch, "length_unit": m.length_unit,
            "nside_region": m.nside_region,
            "redshift_frames": list(m.redshift_frames),
            "wavelength_convention": m.wavelength_convention,
            "magnitude_system": m.magnitude_system}


def _meta_from_dict(d: dict) -> ProductMetadata:
    return ProductMetadata(
        frame=d["frame"], epoch=d["epoch"], length_unit=d["length_unit"],
        nside_region=d["nside_region"],
        redshift_frames=tuple(d.get("redshift_frames", ())),
        wavelength_convention=d.get("wavelength_convention"),
        magnitude_system=d.get("magnitude_system"))


def _prov_to_dict(p: Provenance) -> dict:
    return {"dataset_ids": list(p.dataset_ids),
            "weight_recipe": list(p.weight_recipe),
            "randoms_source": p.randoms_source, "nz_method": p.nz_method,
            "extra": p.extra}


def _prov_from_dict(d: dict) -> Provenance:
    return Provenance(dataset_ids=tuple(d["dataset_ids"]),
                      weight_recipe=tuple(d.get("weight_recipe", ())),
                      randoms_source=d.get("randoms_source"),
                      nz_method=d.get("nz_method"), extra=d.get("extra", {}))


def _cov_to_dict(c):
    if c is None:
        return None
    if isinstance(c, CovariancePlan):
        h = c.handle
        return {"kind": c.kind, "region_nside": c.region_nside,
                "mocks_handle": c.mocks_handle, "ingredients": c.ingredients,
                "handle": (None if h is None else
                           {"cov_id": h.cov_id, "path": h.path, "n": h.n})}
    return {"kind": "string", "value": str(c)}


def _cov_from_dict(d):
    if d is None:
        return None
    if d.get("kind") == "string":
        return d["value"]
    h = d.get("handle")
    return CovariancePlan(
        kind=d["kind"], region_nside=d.get("region_nside"),
        mocks_handle=d.get("mocks_handle"), ingredients=d.get("ingredients"),
        handle=(None if h is None else
                CovarianceHandle(h["cov_id"], h["path"], h["n"])))


def _spec_to_dict(s: MeasurementSpec) -> dict:
    ps = s.pair_statistics
    return {"tracers": list(s.tracers),
            "pairs": [list(p) for p in s.pairs],
            "statistic": s.statistic, "estimator_family": s.estimator_family,
            "binning": s.binning, "coords": s.coords,
            "covariance": _cov_to_dict(s.covariance),
            "pair_statistics": (None if not ps else
                                [[a, b, v] for (a, b), v in ps.items()])}


def _spec_from_dict(d: dict) -> MeasurementSpec:
    ps = d.get("pair_statistics")
    return MeasurementSpec(
        tracers=tuple(d["tracers"]),
        pairs=tuple(tuple(p) for p in d["pairs"]),
        statistic=d["statistic"], estimator_family=d["estimator_family"],
        binning=d.get("binning"), coords=d.get("coords", "on_sky"),
        covariance=_cov_from_dict(d.get("covariance")),
        pair_statistics=(None if not ps else
                         {(a, b): v for a, b, v in ps}))


# ── per-product save/load ──────────────────────────────────────────────────
def _save_product(p, pdir: Path) -> dict:
    pdir.mkdir(parents=True, exist_ok=True)
    np.save(pdir / "region_map.npy", np.asarray(p.region_map))
    info = {"kind": p.kind, "metadata": _meta_to_dict(p.metadata),
            "provenance": _prov_to_dict(p.provenance),
            "covariance": _cov_to_dict(p.covariance),
            "links": []}
    for lk in (p.links or []):
        np.savez(pdir / f"links_{lk.role}.npz", parent_ids=lk.parent_ids,
                 child_ids=lk.child_ids,
                 confidence=(np.array([]) if lk.confidence is None
                             else lk.confidence))
        info["links"].append({"role": lk.role,
                              "payload_keys": list((lk.payload or {}).keys())})
        if lk.payload:
            np.savez(pdir / f"linkpayload_{lk.role}.npz",
                     **{k: np.asarray(v) for k, v in lk.payload.items()})

    if p.kind == "pointset":
        p.catalog.to_parquet(pdir / "catalog.parquet")
        info["has_catalog"] = True
        if p.randoms is not None:
            p.randoms.to_parquet(pdir / "randoms.parquet")
            info["has_randoms"] = True
        if isinstance(p.nz, dict):
            (pdir / "nz_tomo.json").write_text(json.dumps(
                {str(b): _nz_to_dict(v) for b, v in p.nz.items()}))
            info["nz"] = "tomo"
        elif p.nz is not None:
            (pdir / "nz.json").write_text(json.dumps(_nz_to_dict(p.nz)))
            info["nz"] = "single"
        if p.window is not None:
            np.save(pdir / "window.npy", p.window.mask)
            wj = {"nside": p.window.nside, "polygon_path": p.window.polygon_path,
                  "systematics": list((p.window.systematics or {}).keys())}
            (pdir / "window.json").write_text(json.dumps(wj))
            if p.window.systematics:
                np.savez(pdir / "window_sys.npz", **p.window.systematics)
            info["has_window"] = True
        if p.tomo_bin is not None:
            np.save(pdir / "tomo_bin.npy", np.asarray(p.tomo_bin))
            info["has_tomo_bin"] = True
        if p.weights is not None:
            # components namespaced (`comp_<name>`) so a component named
            # "total" cannot collide with the total= kwarg (B2).
            np.savez(pdir / "weights.npz", total=p.weights.total,
                     **{f"comp_{k}": v for k, v in p.weights.components.items()})
            (pdir / "weights.json").write_text(json.dumps(
                {"recipe": list(p.weights.recipe),
                 "components": list(p.weights.components)}))
            info["has_weights"] = True
        if p.photoz is not None and hasattr(p.photoz, "grid"):
            np.savez(pdir / "photoz.npz", grid=np.asarray(p.photoz.grid),
                     values=np.asarray(p.photoz.values))
            info["has_photoz"] = True
        if p.attributes is not None:
            info["attributes"] = p.attributes
    elif p.kind == "sightline":
        p.los.to_parquet(pdir / "los.parquet")
        def _flat(lst):
            if lst is None:
                return None, None
            lens = np.array([len(a) for a in lst])
            return np.concatenate([np.asarray(a) for a in lst]), lens
        d, dl = _flat(p.delta); m, ml = _flat(p.mask); c, cl = _flat(p.continuum)
        np.savez(pdir / "sightline.npz",
                 delta=d if d is not None else np.array([]), dlen=dl,
                 mask=m if m is not None else np.array([]),
                 cont=(c if c is not None else np.array([])),
                 has_cont=np.array([p.continuum is not None]))
        info["n_sightlines"] = int(p.n_sightlines)
    elif p.kind == "fieldmap":
        np.save(pdir / "field.npy", p.values)
        np.save(pdir / "field_mask.npy", p.mask)
        fj = {"nside": p.nside, "nest": p.nest,
              "has_axes": p.axes is not None, "has_beam": p.beam is not None,
              "axes": _to_jsonable(p.axes, "FieldMap.axes"),
              "beam": _to_jsonable(p.beam, "FieldMap.beam"),
              "interloper": _to_jsonable(p.interloper, "FieldMap.interloper")}
        (pdir / "field.json").write_text(json.dumps(fj))
        if p.distance_extras is not None:
            np.savez(pdir / "field_dist.npz",
                     **{k: np.asarray(v) for k, v in p.distance_extras.items()})
            info["has_distance_extras"] = True
    return info


def _to_jsonable(x, what: str):
    """Convert to a JSON-serialisable structure, or raise — never drop (B1).

    numpy arrays/scalars become lists/python scalars (so they round-trip as
    lists). Anything else unserialisable raises rather than silently saving
    ``None`` for an atom the user attached.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_jsonable(v, f"{what}[{k!r}]") for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v, what) for v in x]
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        raise TypeError(
            f"save_measurement_set: {what} contains a non-serialisable "
            f"{type(x).__name__}; convert it to arrays/scalars/strings")


def _load_links(pdir: Path, info: dict):
    out = []
    for lk in info.get("links", []):
        role = lk["role"]
        z = np.load(pdir / f"links_{role}.npz")
        conf = z["confidence"]
        payload = None
        ppath = pdir / f"linkpayload_{role}.npz"
        if ppath.exists():
            pz = np.load(ppath)
            payload = {k: pz[k] for k in pz.files}
        out.append(SubObjectLinks(role=role, parent_ids=z["parent_ids"],
                                  child_ids=z["child_ids"],
                                  confidence=(None if conf.size == 0 else conf),
                                  payload=payload))
    return out or None


def _load_product(pdir: Path, info: dict):
    region = np.load(pdir / "region_map.npy")
    meta = _meta_from_dict(info["metadata"])
    prov = _prov_from_dict(info["provenance"])
    cov = _cov_from_dict(info.get("covariance"))
    links = _load_links(pdir, info)
    common = dict(region_map=region, metadata=meta, provenance=prov,
                  covariance=cov, links=links)
    kind = info["kind"]
    if kind == "pointset":
        cat = pd.read_parquet(pdir / "catalog.parquet")
        rnd = (pd.read_parquet(pdir / "randoms.parquet")
               if info.get("has_randoms") else None)
        nz = None
        if info.get("nz") == "tomo":
            raw = json.loads((pdir / "nz_tomo.json").read_text())
            nz = {int(b): _nz_from_dict(v) for b, v in raw.items()}
        elif info.get("nz") == "single":
            nz = _nz_from_dict(json.loads((pdir / "nz.json").read_text()))
        window = None
        if info.get("has_window"):
            wj = json.loads((pdir / "window.json").read_text())
            sysmaps = None
            if (pdir / "window_sys.npz").exists():
                z = np.load(pdir / "window_sys.npz")
                sysmaps = {k: z[k] for k in z.files}
            window = Window(nside=wj["nside"], mask=np.load(pdir / "window.npy"),
                            systematics=sysmaps, polygon_path=wj["polygon_path"])
        weights = None
        if info.get("has_weights"):
            z = np.load(pdir / "weights.npz")
            wj = json.loads((pdir / "weights.json").read_text())
            weights = NamedWeights(
                total=z["total"],
                components={k: z[f"comp_{k}"] for k in wj["components"]},
                recipe=tuple(wj["recipe"]))
        photoz = None
        if info.get("has_photoz"):
            z = np.load(pdir / "photoz.npz")
            photoz = PhotozArrays(grid=z["grid"], values=z["values"])
        tomo = (np.load(pdir / "tomo_bin.npy") if info.get("has_tomo_bin")
                else None)
        return PointSet(catalog=cat, randoms=rnd, nz=nz, window=window,
                        photoz=photoz, tomo_bin=tomo, weights=weights,
                        attributes=info.get("attributes"), **common)
    if kind == "sightline":
        los = pd.read_parquet(pdir / "los.parquet")
        z = np.load(pdir / "sightline.npz")
        def _split(flat, lens):
            out, i = [], 0
            for n in lens:
                out.append(flat[i:i + n]); i += n
            return out
        delta = _split(z["delta"], z["dlen"]); mask = _split(z["mask"], z["dlen"])
        cont = (_split(z["cont"], z["dlen"]) if bool(z["has_cont"][0]) else None)
        return Sightline(los=los, delta=delta, mask=mask, continuum=cont,
                         **common)
    if kind == "fieldmap":
        fj = json.loads((pdir / "field.json").read_text())
        dist = None
        if info.get("has_distance_extras"):
            z = np.load(pdir / "field_dist.npz")
            dist = {k: z[k] for k in z.files}
        return FieldMap(values=np.load(pdir / "field.npy"),
                        mask=np.load(pdir / "field_mask.npy"),
                        nside=fj["nside"], nest=fj["nest"], axes=fj.get("axes"),
                        beam=fj.get("beam"), interloper=fj.get("interloper"),
                        distance_extras=dist, **common)
    raise ValueError(f"unknown product kind {kind!r}")


# ── top-level API ──────────────────────────────────────────────────────────
def save_measurement_set(ms: MeasurementSet, path: Union[str, Path]) -> Path:
    """Persist a MeasurementSet to a self-describing directory. Returns it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    products = {}
    for name, p in ms.products.items():
        products[name] = _save_product(p, path / name)
    manifest = {"format": _FMT, "cosmology_free": True,
                "spec": _spec_to_dict(ms.spec),
                "metadata": _meta_to_dict(ms.metadata),
                "products": products}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


def load_measurement_set(path: Union[str, Path]) -> MeasurementSet:
    """Reconstruct a MeasurementSet written by :func:`save_measurement_set`."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    products = {name: _load_product(path / name, info)
                for name, info in manifest["products"].items()}
    return MeasurementSet(products=products,
                          spec=_spec_from_dict(manifest["spec"]),
                          metadata=_meta_from_dict(manifest["metadata"]))
