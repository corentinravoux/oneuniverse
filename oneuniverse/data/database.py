"""
oneuniverse.data.database
~~~~~~~~~~~~~~~~~~~~~~~~~
Dynamic survey database built from a folder tree.

Given a root directory whose subdirectories contain ``oneuniverse/manifest.json``
files (the canonical OUF layout), :class:`OneuniverseDatabase` walks the tree,
reads each manifest, and *dynamically* builds a :class:`BaseSurveyLoader`
subclass per discovered dataset.

This makes the package adapt to any directory architecture of the form::

    <root>/<arbitrary>/<nesting>/<survey>/oneuniverse/manifest.json

The survey name is derived from the relative path (path components joined
with ``_``) and the survey type from the first path component, but both can
be overridden with a user-supplied ``name_from_path`` callable.

Example
-------
>>> db = OneuniverseDatabase.from_root("/data/oneuniverse_dataset")
>>> db.list()
{'spectroscopic_eboss_qso': 'eBOSS QSO (point, 919,000 rows)'}
>>> loader = db.get_loader("spectroscopic_eboss_qso")
>>> df = loader.load()
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._dataset_entry import DatasetEntry
from oneuniverse.data._registry import _REGISTRY, register
from oneuniverse.data.converter import (
    get_manifest,
    is_converted,
    read_oneuniverse_parquet,
)
from oneuniverse.data.format_spec import (
    MANIFEST_FILENAME,
    ONEUNIVERSE_SUBDIR,
    DataGeometry,
)
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.manifest import Manifest

logger = logging.getLogger(__name__)


# ── Default naming ───────────────────────────────────────────────────────


def _default_name_from_path(relpath: Path) -> Tuple[str, str]:
    """Derive ``(survey_name, survey_type)`` from a path relative to the root.

    - ``survey_name`` = path parts joined by ``_`` (lowercased)
    - ``survey_type`` = first path part, or ``"unknown"`` for flat layouts
    """
    parts = [p.lower() for p in relpath.parts]
    if not parts:
        return "root", "unknown"
    name = "_".join(parts)
    survey_type = parts[0] if len(parts) > 1 else "unknown"
    return name, survey_type


# ── Dynamic loader factory ───────────────────────────────────────────────


def _make_loader_class(
    name: str,
    config: SurveyConfig,
    data_path: Path,
) -> type:
    """Build a :class:`BaseSurveyLoader` subclass bound to a concrete path.

    The subclass' ``_load_raw`` reads the Parquet partitions directly from
    *data_path*.  ``BaseSurveyLoader.load()`` auto-detects the ``oneuniverse/``
    subdirectory and uses the fast Parquet reader, but we still provide
    ``_load_raw`` as a fallback (and to satisfy the ABC contract).
    """

    cls_name = "".join(part.capitalize() for part in name.split("_")) + "Loader"

    def _load_raw(self, data_path=None, **kwargs) -> pd.DataFrame:
        path = Path(data_path) if data_path is not None else self._bound_path
        return read_oneuniverse_parquet(path)

    attrs = {
        "config": config,
        "_bound_path": data_path,
        "_load_raw": _load_raw,
        "__doc__": f"Dynamically generated loader for '{name}' at {data_path}.",
        "__module__": __name__,
    }

    # Override `load()` to honor the bound path without requiring the global
    # data root to be configured.
    base_load = BaseSurveyLoader.load

    def load(self, *args, **kwargs):
        kwargs.setdefault("data_path", self._bound_path)
        return base_load(self, *args, **kwargs)

    attrs["load"] = load

    return type(cls_name, (BaseSurveyLoader,), attrs)


# ── Database class ───────────────────────────────────────────────────────


class OneuniverseDatabase:
    """Dynamic registry of OUF datasets discovered under a root directory.

    Parameters
    ----------
    root : str or Path
        Root of the dataset tree to scan.
    name_from_path : callable, optional
        ``(relpath: Path) -> (name, survey_type)``.  Defaults to joining
        path parts with ``_`` and using the first component as the type.
    max_depth : int, optional
        Maximum directory depth to search (default: 6).
    register_global : bool
        If True, also register each discovered loader in the package-wide
        registry so it becomes available via :func:`load_catalog`.
        Names already registered by hand-coded loaders are skipped.
    """

    def __init__(
        self,
        root: Union[str, Path],
        name_from_path: Optional[Callable[[Path], Tuple[str, str]]] = None,
        max_depth: int = 6,
        register_global: bool = False,
        data_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Root directory does not exist: {self.root}")
        from oneuniverse.data._config import env_data_root
        self.data_root: Optional[Path] = (
            Path(data_root).expanduser().resolve() if data_root is not None
            else env_data_root()
        )
        self._name_from_path = name_from_path or _default_name_from_path
        self._max_depth = max_depth
        self._register_global = register_global
        self._entries: Dict[str, DatasetEntry] = {}
        self.scan()

    # ── Construction helpers ─────────────────────────────────────────────

    @classmethod
    def from_root(cls, root: Union[str, Path], **kwargs) -> "OneuniverseDatabase":
        """Shortcut for ``OneuniverseDatabase(root, **kwargs)``."""
        return cls(root, **kwargs)

    @classmethod
    def from_config(cls, config_path: Union[str, Path], **kwargs) -> "OneuniverseDatabase":
        """Build a database from an INI config file.

        See :mod:`oneuniverse.data.config_loader` for the file format.
        """
        from oneuniverse.data.config_loader import build_from_config
        return build_from_config(config_path, **kwargs)

    @classmethod
    def build(
        cls,
        raw_root: Union[str, Path],
        database_root: Union[str, Path],
        overwrite: bool = False,
        skip_existing: bool = True,
        **kwargs,
    ) -> "OneuniverseDatabase":
        """Convert every registered survey whose raw file lives under *raw_root*
        into the oneuniverse format under *database_root*, mirroring the raw
        tree, then return a :class:`OneuniverseDatabase` bound to
        *database_root*.

        Parameters
        ----------
        raw_root : path
            Root directory containing the native survey files. Each registered
            loader is checked by looking for
            ``raw_root / config.data_subpath / config.data_filename``.
        database_root : path
            Where to write the converted OUF tree. Structure mirrors
            *raw_root*: ``database_root / <data_subpath> / oneuniverse/``.
        overwrite : bool
            If True, re-convert datasets whose output already exists.
        skip_existing : bool
            If True (default), skip datasets whose output already exists
            (ignored when ``overwrite=True``).
        **kwargs
            Forwarded to :class:`OneuniverseDatabase` (e.g. ``name_from_path``).
        """
        from oneuniverse.data._config import set_data_root
        from oneuniverse.data._registry import _REGISTRY

        raw_root = Path(raw_root).expanduser().resolve()
        database_root = Path(database_root).expanduser().resolve()
        if not raw_root.is_dir():
            raise FileNotFoundError(f"Raw root does not exist: {raw_root}")
        database_root.mkdir(parents=True, exist_ok=True)

        # Point the global data-root at the raw tree so loaders can find files.
        set_data_root(raw_root)

        from oneuniverse.data.converter import convert_survey  # local import

        n_converted = n_skipped = 0
        for name, loader_cls in sorted(_REGISTRY.items()):
            cfg = loader_cls.config
            if not cfg.data_filename:
                logger.debug("Skipping %s: no data_filename (in-memory loader)", name)
                continue
            subpath = cfg.data_subpath or f"{cfg.survey_type}/{cfg.name}"
            raw_file = raw_root / subpath / cfg.data_filename
            if not raw_file.exists():
                logger.debug("Skipping %s: no raw file at %s", name, raw_file)
                continue

            out_base = database_root / subpath
            out_dir = out_base / ONEUNIVERSE_SUBDIR
            if out_dir.exists() and not overwrite:
                if skip_existing:
                    logger.info("Already converted, skipping: %s", name)
                    n_skipped += 1
                    continue

            logger.info("Converting %s  →  %s", name, out_dir)
            convert_survey(
                survey_name=name,
                data_root=raw_root,
                output_dir=out_base,
                overwrite=overwrite,
            )
            n_converted += 1

        logger.info(
            "OneuniverseDatabase.build: %d converted, %d skipped",
            n_converted,
            n_skipped,
        )
        return cls(database_root, **kwargs)

    # ── Scanning ─────────────────────────────────────────────────────────

    def scan(self) -> None:
        """Walk :attr:`root` and (re)build the in-memory registry."""
        self._entries.clear()

        for manifest_path in self._find_manifests():
            survey_dir = manifest_path.parent.parent  # …/<survey_dir>/oneuniverse/manifest.json
            relpath = survey_dir.relative_to(self.root)
            try:
                name, survey_type = self._name_from_path(relpath)
            except Exception as exc:
                logger.warning("name_from_path failed for %s: %s", relpath, exc)
                continue

            try:
                manifest = get_manifest(survey_dir)
            except Exception as exc:
                logger.warning("Skipping %s: cannot read manifest (%s)", relpath, exc)
                continue

            config = self._config_from_manifest(
                name=name,
                survey_type=survey_type,
                relpath=relpath,
                manifest=manifest,
            )
            loader_cls = _make_loader_class(name, config, survey_dir)
            self._entries[name] = DatasetEntry(
                loader=loader_cls,
                manifest=manifest,
                path=survey_dir,
            )

            if self._register_global and name not in _REGISTRY:
                try:
                    register(loader_cls)
                except ValueError:
                    pass

        logger.info(
            "OneuniverseDatabase: discovered %d dataset(s) under %s",
            len(self._entries),
            self.root,
        )

    def _find_manifests(self) -> List[Path]:
        """Return all ``oneuniverse/manifest.json`` files under root, bounded
        by ``max_depth``."""
        found: List[Path] = []
        root_depth = len(self.root.parts)

        def walk(d: Path):
            if len(d.parts) - root_depth > self._max_depth:
                return
            try:
                entries = list(d.iterdir())
            except (PermissionError, OSError):
                return
            # If this dir is itself an `oneuniverse/` dir, record its manifest.
            if d.name == ONEUNIVERSE_SUBDIR:
                mf = d / MANIFEST_FILENAME
                if mf.is_file():
                    found.append(mf)
                return  # do not descend further
            for entry in entries:
                if entry.is_dir():
                    walk(entry)

        walk(self.root)
        return sorted(found)

    @staticmethod
    def _config_from_manifest(
        name: str,
        survey_type: str,
        relpath: Path,
        manifest: Manifest,
    ) -> SurveyConfig:
        """Build a :class:`SurveyConfig` from a typed :class:`Manifest`."""
        geometry = manifest.geometry.value
        n_rows = manifest.n_rows
        survey_name_m = manifest.survey_name or name
        survey_type_m = manifest.survey_type or survey_type
        description = (
            f"{survey_name_m} ({geometry}, {n_rows:,} rows)"
            if n_rows
            else f"{survey_name_m} ({geometry})"
        )
        return SurveyConfig(
            name=name,
            survey_type=survey_type_m,
            description=description,
            data_subpath=str(relpath),
            data_format="parquet",
            n_objects_approx=n_rows,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def list(self, survey_type: Optional[str] = None) -> Dict[str, str]:
        """Return ``{name: description}`` for discovered datasets."""
        out = {}
        for n, entry in sorted(self._entries.items()):
            cfg = entry.loader.config
            if survey_type is not None and cfg.survey_type != survey_type:
                continue
            out[n] = cfg.description
        return out

    def types(self) -> List[str]:
        return sorted({e.loader.config.survey_type for e in self._entries.values()})

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, name: str) -> DatasetView:
        return self.view(name)

    def entry(self, name: str) -> DatasetEntry:
        """Return the full :class:`DatasetEntry` for *name*."""
        if name not in self._entries:
            available = ", ".join(sorted(self._entries)) or "(none)"
            raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
        return self._entries[name]

    def view(self, name: str) -> DatasetView:
        """Return a lazy :class:`DatasetView` for *name*."""
        entry = self.entry(name)
        ou_dir = entry.path / ONEUNIVERSE_SUBDIR
        return DatasetView(ou_dir=ou_dir, manifest=entry.manifest)

    def get_loader(self, name: str) -> BaseSurveyLoader:
        """Instantiate the dynamic loader for *name* (legacy path)."""
        return self.entry(name).loader()

    def get_manifest(self, name: str) -> Manifest:
        return self.entry(name).manifest

    def get_path(self, name: str) -> Path:
        return self.entry(name).path

    def get_config(self, name: str) -> SurveyConfig:
        return self.entry(name).loader.config

    def load(self, name: str, **kwargs) -> pd.DataFrame:
        """Shortcut: instantiate loader and call ``.load(**kwargs)``."""
        return self.get_loader(name).load(**kwargs)

    # ── ONEUID — universal cross-survey identifier ───────────────────────

    def build_oneuid(
        self,
        sky_tol_arcsec: Optional[float] = None,
        dz_tol: Optional[float] = None,
        *,
        datasets: Optional[Sequence[str]] = None,
        rules=None,
        name: str = "default",
        persist: bool = True,
    ):
        """Build the ONEUID index across every discovered dataset.

        Returns the :class:`oneuniverse.data.oneuid.OneuidIndex` and, by
        default, persists it to ``{root}/_oneuid/<name>.parquet``.

        Pass ``rules=CrossMatchRules(...)`` for the full policy; the
        ``sky_tol_arcsec`` / ``dz_tol`` args are kept for back-compat.
        """
        from oneuniverse.data.oneuid import build_oneuid_index
        return build_oneuid_index(
            self,
            datasets=datasets, rules=rules, name=name, persist=persist,
            sky_tol_arcsec=sky_tol_arcsec, dz_tol=dz_tol,
        )

    def load_oneuid(self, name: str = "default"):
        """Load a previously persisted ONEUID index by *name*."""
        from oneuniverse.data.oneuid import load_oneuid_index
        return load_oneuid_index(self, name=name)

    def list_oneuids(self) -> List[str]:
        """Return the names of persisted ONEUID indices."""
        from oneuniverse.data.oneuid import list_oneuids
        return list_oneuids(self)

    def oneuid_query(self, index=None, *, name: str = "default") -> "OneuidQuery":
        """Return a tiered :class:`OneuidQuery` over the ONEUID index.

        Pass *index* to reuse an in-memory :class:`OneuidIndex`; otherwise
        the persisted sidecar *name* is loaded from disk.
        """
        from oneuniverse.data.oneuid import OneuidQuery
        return OneuidQuery(self, index=index, name=name)

    def summary(self) -> str:
        """Human-readable summary of all discovered datasets."""
        if not self._entries:
            return f"OneuniverseDatabase @ {self.root}\n  (no datasets found)"
        lines = [
            f"OneuniverseDatabase @ {self.root}",
            f"  {len(self._entries)} dataset(s):",
        ]
        for name in sorted(self._entries):
            e = self._entries[name]
            cfg = e.loader.config
            mf = e.manifest
            lines.append(
                f"  - {name:40s}  [{cfg.survey_type:15s}] "
                f"{mf.geometry.value:9s} "
                f"n_rows={mf.n_rows:>10,}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<OneuniverseDatabase root={self.root} n_datasets={len(self._loaders)}>"
