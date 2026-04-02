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


def get_data_root() -> Optional[Path]:
    """Return the current data root, or None if unset."""
    if _data_root is not None:
        return _data_root
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env)
    return None


def set_data_root(path: str | Path) -> None:
    """Override the data root for this session."""
    global _data_root
    _data_root = Path(path)


def resolve_survey_path(survey_type: str, survey_name: str) -> Optional[Path]:
    """Return ``{data_root}/{survey_type}/{survey_name}/`` or None."""
    root = get_data_root()
    if root is None:
        return None
    return root / survey_type / survey_name
