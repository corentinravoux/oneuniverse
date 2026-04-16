"""
oneuniverse.data._config
~~~~~~~~~~~~~~~~~~~~~~~~~
Global configuration for data loading — principally the root path where
survey data lives on a cluster filesystem.

Configuration hierarchy (first match wins):
1. Explicit ``set_data_root(path)`` call in user code
2. Environment variable ``ONEUNIVERSE_DATA_ROOT``
3. Default ``None`` (only in-memory / test surveys available)

On a cluster the admin typically sets::

    export ONEUNIVERSE_DATA_ROOT=/data/cosmology/surveys

Surveys then resolve their data path as::

    {data_root}/{survey_type}/{survey_name}/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_VAR = "ONEUNIVERSE_DATA_ROOT"
_data_root: Optional[Path] = None


def env_data_root() -> Optional[Path]:
    """Return the data root from :envvar:`ONEUNIVERSE_DATA_ROOT`, or None.

    Prefer this over :func:`get_data_root` in new code — it is the per-
    database kwarg's natural default and has no mutable module state.
    """
    env = os.environ.get(_ENV_VAR)
    return Path(env) if env else None


def get_data_root() -> Optional[Path]:
    """Return the current data root, or None if unset.

    .. deprecated::
        Pass ``data_root=`` to :class:`OneuniverseDatabase` instead.
        Module-level state is retained only for backward compatibility
        and will be removed in a future major release.
    """
    if _data_root is not None:
        return _data_root
    return env_data_root()


def set_data_root(path: str | Path) -> None:
    """Override the data root for this session.

    .. deprecated::
        Pass ``data_root=`` to :class:`OneuniverseDatabase` instead.
    """
    global _data_root
    _data_root = Path(path)


def resolve_survey_path(
    survey_type: str,
    survey_name: str,
    data_subpath: str = "",
) -> Optional[Path]:
    """Return the survey data directory or None.

    If *data_subpath* is set (e.g. ``"spectroscopic/eboss/qso"``), it is used
    directly under the data root.  Otherwise falls back to
    ``{data_root}/{survey_type}/{survey_name}/``.
    """
    root = get_data_root()
    if root is None:
        return None
    if data_subpath:
        return root / data_subpath
    return root / survey_type / survey_name
