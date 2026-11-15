"""Linear fixture simulation on disk, its packed layout and the two converters.

``generate_linear_sim`` writes a small *native* simulation directory from linear
theory (:mod:`oneuniverse.reference.linear`): Zel'dovich particles, the linear
density field, peak "halos" and a toy sky catalogue, per redshift. It exists to
exercise the simulation substrate (converters, adapters, stores); it is not a
physical simulation.

Layout of a linear directory (``<tag>`` = ``f"{z:.3f}"``)::

    config.json              generator, cosmology, box, n_grid, redshifts, seed
    particles_z<tag>.npy     structured float32 (x, y, z, vx, vy, vz), x in [0, box)
    field_z<tag>.npy         float32 (n, n, n): delta(z=0) * D(z)
    halos_z<tag>.parquet     linear.find_peaks of that field
    lightcone.parquet        columns ra, dec, z_shell, radius

The lightcone is **not** a physical lightcone: every redshift's halos are
projected on the sky about the box centre and tagged with the snapshot
redshift as ``z_shell``. It has no ``z`` column, so ``read_cone(z_range=...)``
on it raises ``SchemaError``.

``write_packed_native`` re-sorts the particle slabs by store chunk id and adds
``header.json`` (``"layout": "packed_npy"``), the format read by
``PackedNpyAdapter``.

Detection markers (exclusive): ``config.json`` with
``"generator": "oneuniverse.reference.linear"`` and no ``header.json`` is
``linear``; additionally ``header.json`` with ``"layout": "packed_npy"`` is
``packed_npy``.
"""

from __future__ import annotations

import json
import numbers
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse._core.fs import atomic_dir
from oneuniverse.reference.linear import (
    find_peaks,
    gaussian_field,
    growth_factor,
    zeldovich,
)
from oneuniverse.simulation.converter import SimConverter, register
from oneuniverse.simulation.manifest import OUFSIM_FORMAT_VERSION, OUFSimManifest
from oneuniverse.simulation.specs import (
    CosmologySpec,
    ProductDecl,
    ProvenanceSpec,
    UnitFrameSpec,
)
from oneuniverse.simulation.store.build import NativeProduct
from oneuniverse.simulation.store.index import chunk_ids

GENERATOR = "oneuniverse.reference.linear"
PARTICLE_COLUMNS = ("x", "y", "z", "vx", "vy", "vz")
LIGHTCONE_COLUMNS = ("ra", "dec", "z_shell", "radius")
_BATCH = 50_000
_CREATED = "2026-11-15T00:00:00Z"
_CONFIG_KEYS = ("box_size", "n_grid", "redshifts", "seed", "cosmology")


def _tag(z: float) -> str:
    return f"{float(z):.3f}"


def _particles(pos: np.ndarray, vel: np.ndarray, box: float) -> np.ndarray:
    arr = np.empty(len(pos), dtype=[(c, np.float32) for c in PARTICLE_COLUMNS])
    for i, c in enumerate("xyz"):
        col = pos[:, i].astype(np.float32)
        col[col >= box] = 0.0  # float32 rounding of values just below box
        arr[c] = col
    for i, c in enumerate(("vx", "vy", "vz")):
        arr[c] = vel[:, i].astype(np.float32)
    return arr


def _sky_table(halos_by_z: dict[float, pa.Table], box: float) -> pa.Table:
    centre = np.full(3, box / 2.0)
    cols: dict[str, list[np.ndarray]] = {c: [] for c in LIGHTCONE_COLUMNS}
    for z, halos in halos_by_z.items():
        if halos.num_rows == 0:
            continue
        d = np.stack([halos[c].to_numpy() for c in "xyz"], axis=1) - centre
        r = np.sqrt((d ** 2).sum(axis=1))
        keep = r > 0.0
        d, r = d[keep], r[keep]
        cols["ra"].append(np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0)
        cols["dec"].append(np.degrees(np.arcsin(np.clip(d[:, 2] / r, -1.0, 1.0))))
        cols["z_shell"].append(np.full(len(r), float(z)))
        cols["radius"].append(r)
    return pa.table({c: (np.concatenate(v) if v else np.empty(0)) for c, v in cols.items()},
                    schema=pa.schema([(c, pa.float64()) for c in LIGHTCONE_COLUMNS]))


def generate_linear_sim(out, cosmo: CosmologySpec, *, box: float, n_grid: int,
                        redshifts, seed: int) -> Path:
    """Write a linear-theory native simulation directory at ``out`` and return it.

    Arguments are validated before anything is written; the directory is published
    atomically and an existing ``out`` is refused (``ValidationError``).
    """
    out = Path(out)
    zs = tuple(float(z) for z in redshifts)
    if not zs or any(z < 0 for z in zs) or len({_tag(z) for z in zs}) != len(zs):
        raise ValidationError(f"redshifts must be non-empty, >= 0 and distinct at 3 decimals, got {zs}")
    if not box > 0:
        raise ValidationError(f"box must be positive, got {box}")
    if int(n_grid) != n_grid or n_grid < 2:
        raise ValidationError(f"n_grid must be an integer >= 2, got {n_grid}")
    n_grid = int(n_grid)
    if out.exists():
        raise ValidationError(f"destination {out} already exists")
    delta0 = gaussian_field(cosmo, box=box, n_grid=n_grid, z=0.0, seed=seed)
    with atomic_dir(out) as tmp:
        halos_by_z: dict[float, pa.Table] = {}
        for z in zs:
            tag = _tag(z)
            pos, vel = zeldovich(delta0, cosmo, box=box, z=z)
            np.save(tmp / f"particles_z{tag}.npy", _particles(pos, vel, box))
            field = delta0 * growth_factor(z, cosmo)
            np.save(tmp / f"field_z{tag}.npy", field.astype(np.float32))
            halos = find_peaks(field, box=box)
            pq.write_table(halos, tmp / f"halos_z{tag}.parquet")
            halos_by_z[z] = halos
        pq.write_table(_sky_table(halos_by_z, box), tmp / "lightcone.parquet")
        (tmp / "config.json").write_text(json.dumps({
            "generator": GENERATOR, "box_size": float(box), "n_grid": n_grid,
            "redshifts": list(zs), "seed": int(seed), "cosmology": cosmo.to_dict()}))
    return out


def _is_number(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _is_int(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _read_config(path: Path) -> dict:
    """Parsed and checked ``config.json``; ``FormatError`` if unreadable or malformed."""
    try:
        cfg = json.loads((Path(path) / "config.json").read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"cannot read config.json in {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise FormatError(f"config.json in {path} must be a JSON object, got {type(cfg).__name__}")
    missing = [k for k in _CONFIG_KEYS if k not in cfg]
    if missing:
        raise FormatError(f"config.json in {path} lacks key(s) {missing}")
    zs = cfg["redshifts"]
    if not (_is_number(cfg["box_size"]) and cfg["box_size"] > 0
            and _is_int(cfg["n_grid"]) and cfg["n_grid"] >= 2
            and isinstance(zs, list) and zs and all(_is_number(z) for z in zs)
            and isinstance(cfg["cosmology"], dict)):
        raise FormatError(
            f"config.json in {path}: box_size must be a positive number, n_grid an integer >= 2, "
            "redshifts a non-empty list of numbers and cosmology an object")
    return cfg


def _load_json_object(file: Path) -> dict | None:
    """The JSON object in ``file``, or ``None`` if it is missing, unreadable or not an object."""
    try:
        data = json.loads(file.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_linear_config(path: Path) -> bool:
    cfg = _load_json_object(path / "config.json")
    return cfg is not None and cfg.get("generator") == GENERATOR


def _header_layout(path: Path) -> str | None:
    hdr = _load_json_object(path / "header.json")
    return None if hdr is None else hdr.get("layout")


def write_packed_native(linear_dir, out_dir, *, n_side: int) -> Path:
    """Chunk-sort the particles of ``linear_dir`` into a packed native dir ``out_dir``.

    Published atomically; ``ValidationError`` if ``out_dir`` exists or is ``linear_dir``.
    """
    src, dst = Path(linear_dir), Path(out_dir)
    if int(n_side) != n_side or n_side < 1:
        raise ValidationError(f"n_side must be a positive integer, got {n_side}")
    if not _is_linear_config(src):
        raise ValidationError(f"{src} is not a linear fixture simulation (config.json generator)")
    cfg = _read_config(src)
    if dst.resolve() == src.resolve():
        raise ValidationError(f"destination {dst} is the source directory")
    if dst.exists():
        raise ValidationError(f"destination {dst} already exists")
    box = float(cfg["box_size"])
    with atomic_dir(dst) as tmp:
        chunks = {}
        for z in cfg["redshifts"]:
            tag = _tag(z)
            arr = np.load(src / f"particles_z{tag}.npy")
            pos = np.stack([arr[c] for c in "xyz"], axis=1).astype(np.float64)
            ids = chunk_ids(pos, box, int(n_side))
            order = np.argsort(ids, kind="stable")
            np.save(tmp / f"particles_z{tag}.npy", arr[order])
            uniq, start, count = np.unique(ids[order], return_index=True, return_counts=True)
            chunks[tag] = [[int(c), int(a), int(a + n)] for c, a, n in zip(uniq, start, count)]
            for name in (f"field_z{tag}.npy", f"halos_z{tag}.parquet"):
                shutil.copy2(src / name, tmp / name)
        for name in ("lightcone.parquet", "config.json"):
            shutil.copy2(src / name, tmp / name)
        (tmp / "header.json").write_text(json.dumps({
            "layout": "packed_npy", "box_size": box, "n_side": int(n_side),
            "redshifts": cfg["redshifts"], "chunks": chunks}))
    return dst


def _batches_parquet(file: Path):
    yield from pq.ParquetFile(file).iter_batches(batch_size=_BATCH)


def _manifest(path: Path, code: str, projection: str) -> OUFSimManifest:
    cfg = _read_config(path)
    return OUFSimManifest(
        OUFSIM_FORMAT_VERSION, "native_sim", float(cfg["box_size"]),
        CosmologySpec.from_dict(cfg["cosmology"]), UnitFrameSpec(),
        ProvenanceSpec(code, "1", cfg["seed"], GENERATOR),
        (), {"format": code, "root": str(path), "projection": projection}, _CREATED)


def _products(path: Path, adapter_format: str | None) -> list[NativeProduct]:
    cfg = _read_config(path)
    zs = tuple(float(z) for z in cfg["redshifts"])
    tags = {z: _tag(z) for z in zs}
    part_src = {z: path / f"particles_z{t}.npy" for z, t in tags.items()}
    halo_src = {z: path / f"halos_z{t}.parquet" for z, t in tags.items()}
    halo_cols = tuple(pq.read_schema(halo_src[zs[0]]).names)
    lc_file = path / "lightcone.parquet"
    particles = NativeProduct(
        ProductDecl("particles", "particles", zs, PARTICLE_COLUMNS), "catalog", part_src, adapter_format)
    return [
        particles,
        NativeProduct(ProductDecl("halos", "halos", zs, halo_cols), "catalog", halo_src, None,
                      lambda z, s=halo_src: _batches_parquet(s[float(z)])),
        NativeProduct(ProductDecl("field", "field", zs, n_grid=cfg["n_grid"]), "field",
                      {z: path / f"field_z{t}.npy" for z, t in tags.items()}),
        NativeProduct(ProductDecl("lightcone", "lightcone", (0.0,), LIGHTCONE_COLUMNS,
                                  extra={"physical": False}),
                      "lightcone", {0.0: lc_file}, None, lambda z: _batches_parquet(lc_file)),
    ]


@register
class LinearSimConverter(SimConverter):
    """Linear fixture simulation (marker: ``config.json`` generator, no ``header.json``)."""

    code = "linear"

    @classmethod
    def detect(cls, path) -> bool:
        path = Path(path)
        return _is_linear_config(path) and not (path / "header.json").exists()

    def manifest(self, path) -> OUFSimManifest:
        return _manifest(Path(path), "linear", "reencode")

    def products(self, path) -> list[NativeProduct]:
        return _products(Path(path), None)


@register
class PackedSimConverter(SimConverter):
    """Packed-npy layout of a linear simulation (marker: ``header.json`` layout ``packed_npy``)."""

    code = "packed_npy"

    @classmethod
    def detect(cls, path) -> bool:
        path = Path(path)
        return _header_layout(path) == "packed_npy" and _is_linear_config(path)

    def manifest(self, path) -> OUFSimManifest:
        return _manifest(Path(path), "packed_npy", "reencode")

    def products(self, path) -> list[NativeProduct]:
        return _products(Path(path), "packed_npy")
