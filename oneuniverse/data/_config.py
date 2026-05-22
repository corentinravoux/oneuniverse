"""
oneuniverse.data._config
~~~~~~~~~~~~~~~~~~~~~~~~~
Per-call data-root resolution. **No module-level mutable state.**

A "data root" is the filesystem prefix under which raw survey catalogs
live. It can be supplied three ways, in order of precedence:

1. Explicit ``data_root=`` keyword argument on the consuming call
   (:func:`resolve_survey_path`, :class:`OneuniverseDatabase`,
   :func:`convert_survey`).
2. The ``ONEUNIVERSE_DATA_ROOT`` environment variable.
3. ``None`` (only in-memory / test surveys available).

Surveys then resolve their data path as::

    {data_root}/{survey_type}/{survey_name}/

The module-level ``set_data_root``/``get_data_root`` wrappers from
earlier releases were removed in Phase 12 — they bled config between
processes and tests. Pass ``data_root=`` explicitly or set the env var.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_VAR = "ONEUNIVERSE_DATA_ROOT"


def env_data_root() -> Optional[Path]:
    """Return the data root from :envvar:`ONEUNIVERSE_DATA_ROOT`, or None."""
    env = os.environ.get(_ENV_VAR)
    return Path(env) if env else None


def resolve_survey_path(
    survey_type: str,
    survey_name: str,
    data_subpath: str = "",
    *,
    data_root: Optional[Path] = None,
) -> Optional[Path]:
    """Return the survey data directory or None.

    Resolution: ``data_root`` kwarg → ``ONEUNIVERSE_DATA_ROOT`` env →
    None. If *data_subpath* is set (e.g. ``"spectroscopic/eboss/qso"``)
    it is used directly under the data root; otherwise the path falls
    back to ``{data_root}/{survey_type}/{survey_name}/``.
    """
    root = data_root if data_root is not None else env_data_root()
    if root is None:
        return None
    root = Path(root)
    if data_subpath:
        return root / data_subpath
    return root / survey_type / survey_name
