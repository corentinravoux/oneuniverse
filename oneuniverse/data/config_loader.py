"""
oneuniverse.data.config_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build a :class:`OneuniverseDatabase` from an INI-style config file.

The config lists one ``[dataset …]`` section per input dataset.  Each
section points to a native file (or pre-converted directory) and tells
the builder which registered loader to use and where to write the OUF
tree.  This is the entry point for running on a supercluster where
many heterogeneous datasets live in arbitrary locations.

Example
-------
::

    # oneuniverse.ini
    [database]
    root = /scratch/ravoux/oneuniverse_database
    overwrite = false

    [dataset eboss_qso]
    loader   = eboss_qso
    raw_path = /data/sdss/dr16/DR16Q_Superset_v3.fits
    qso_only = true
    z_min    = 0.8

    [dataset cosmicflows4]
    loader   = cosmicflows4
    raw_path = /data/pv/CF4.csv

Usage
-----
>>> from oneuniverse.data import OneuniverseDatabase
>>> db = OneuniverseDatabase.from_config("oneuniverse.ini")
>>> db.list()
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ── Parsing helpers ──────────────────────────────────────────────────────


_RESERVED_KEYS = {"loader", "raw_path", "output_subpath", "skip", "description"}


def _coerce(value: str) -> Any:
    """Lightweight type coercion for INI string values."""
    v = value.strip()
    low = v.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    # int / float
    try:
        if "." not in v and "e" not in low:
            return int(v)
        return float(v)
    except ValueError:
        pass
    return v


def _section_to_spec(name: str, section: configparser.SectionProxy) -> Dict[str, Any]:
    """Convert a ``[dataset …]`` section into a spec dict."""
    spec: Dict[str, Any] = {
        "section": name,
        "loader": section.get("loader"),
        "raw_path": section.get("raw_path"),
        "output_subpath": section.get("output_subpath"),
        "skip": section.getboolean("skip", fallback=False),
        "description": section.get("description", ""),
        "kwargs": {},
    }
    for key, val in section.items():
        if key in _RESERVED_KEYS:
            continue
        spec["kwargs"][key] = _coerce(val)
    return spec


def parse_config(path: Path | str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read an oneuniverse INI config.

    Returns
    -------
    db_settings : dict
        Options from the ``[database]`` section (``root``, ``overwrite``, …).
    datasets : list of dict
        One spec per ``[dataset …]`` section.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(path)

    if "database" not in parser:
        raise ValueError(f"{path}: missing required [database] section")
    db = parser["database"]
    db_settings = {
        "root": db.get("root"),
        "overwrite": db.getboolean("overwrite", fallback=False),
        "skip_existing": db.getboolean("skip_existing", fallback=True),
    }
    if not db_settings["root"]:
        raise ValueError(f"{path}: [database] section must set 'root'")

    datasets: List[Dict[str, Any]] = []
    for sec_name in parser.sections():
        if not sec_name.lower().startswith("dataset"):
            continue
        datasets.append(_section_to_spec(sec_name, parser[sec_name]))

    if not datasets:
        logger.warning("%s: no [dataset …] sections found", path)
    return db_settings, datasets


# ── Builder ──────────────────────────────────────────────────────────────


def build_from_config(config_path: Path | str, **db_kwargs):
    """Build (or refresh) an :class:`OneuniverseDatabase` from a config file.

    For each ``[dataset …]`` section, the corresponding registered loader is
    invoked against ``raw_path`` and the result is written under
    ``<database.root>/<output_subpath>/oneuniverse/``.  If ``output_subpath``
    is not given it defaults to the loader's ``data_subpath`` (or
    ``<survey_type>/<survey_name>``).

    Extra keys in a dataset section are forwarded to the loader as kwargs.

    Returns
    -------
    OneuniverseDatabase bound to ``database.root``.
    """
    from oneuniverse.data._registry import _REGISTRY
    from oneuniverse.data.converter import convert_survey
    from oneuniverse.data.database import OneuniverseDatabase
    from oneuniverse.data.format_spec import ONEUNIVERSE_SUBDIR

    db_settings, datasets = parse_config(config_path)
    database_root = Path(db_settings["root"]).expanduser().resolve()
    overwrite = db_settings["overwrite"]
    skip_existing = db_settings["skip_existing"]
    database_root.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_fail = 0
    for spec in datasets:
        tag = spec["section"]
        if spec["skip"]:
            logger.info("[%s] skip=true, ignoring", tag)
            n_skip += 1
            continue

        loader_name = spec["loader"]
        if loader_name is None:
            logger.error("[%s] missing 'loader' key", tag)
            n_fail += 1
            continue
        if loader_name not in _REGISTRY:
            logger.error(
                "[%s] unknown loader '%s' (registered: %s)",
                tag, loader_name, ", ".join(sorted(_REGISTRY)),
            )
            n_fail += 1
            continue

        cfg = _REGISTRY[loader_name].config
        raw_path = spec["raw_path"]
        if raw_path is None:
            logger.error("[%s] missing 'raw_path'", tag)
            n_fail += 1
            continue
        raw_path = Path(raw_path).expanduser().resolve()
        if not raw_path.exists():
            logger.error("[%s] raw_path not found: %s", tag, raw_path)
            n_fail += 1
            continue

        # Derive output location inside database_root
        output_subpath = spec["output_subpath"] or cfg.data_subpath or \
            f"{cfg.survey_type}/{cfg.name}"
        out_base = database_root / output_subpath
        out_dir = out_base / ONEUNIVERSE_SUBDIR

        if out_dir.exists() and not overwrite and skip_existing:
            logger.info("[%s] already converted at %s, skipping", tag, out_dir)
            n_skip += 1
            continue

        logger.info("[%s] converting %s → %s", tag, raw_path, out_dir)
        try:
            convert_survey(
                survey_name=loader_name,
                raw_path=raw_path,
                output_dir=out_base,
                overwrite=overwrite,
                **spec["kwargs"],
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] conversion failed: %s", tag, exc)
            n_fail += 1

    logger.info(
        "build_from_config: %d ok, %d skipped, %d failed", n_ok, n_skip, n_fail,
    )
    return OneuniverseDatabase(database_root, **db_kwargs)
