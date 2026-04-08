"""
oneuniverse.data
~~~~~~~~~~~~~~~~
Fast loading of galaxy survey catalogs with standardized schemas and
composable spatial / redshift selections.

Quick start
-----------
>>> from oneuniverse.data import load_catalog, list_surveys, Cone, Shell
>>> list_surveys()
{'dummy': ..., 'sdss_mgs': ..., 'cosmicflows4': ..., ...}
>>> list_surveys(survey_type="peculiar_velocity")
{'cosmicflows4': ..., 'desi_pv': ...}
>>> df = load_catalog("dummy", selection=Cone(ra=185, dec=15, radius=10))
>>> df = load_catalog("dummy", selection=[Cone(ra=0, dec=0, radius=30), Shell(0.02, 0.1)])
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import pandas as pd

# ── Selection objects (user-facing) ──────────────────────────────────────
from oneuniverse.data.selection import Cone, Shell, SkyPatch, Selection  # noqa: F401

# ── Import all survey sub-packages (triggers @register for each loader) ──
import oneuniverse.data.surveys  # noqa: F401

# ── Private machinery ────────────────────────────────────────────────────
from oneuniverse.data._config import get_data_root, set_data_root  # noqa: F401
from oneuniverse.data._registry import (  # noqa: F401
    get_loader,
    get_survey_config,
    list_surveys,
    list_survey_types,
)
from oneuniverse.data.converter import (  # noqa: F401
    convert_survey,
    convert_sightlines,
    convert_healpix_map,
    fetch_original_columns,
    get_manifest,
    get_geometry,
    is_converted,
    read_objects_table,
)
from oneuniverse.data.format_spec import DataGeometry  # noqa: F401
from oneuniverse.data.database import OneuniverseDatabase  # noqa: F401
from oneuniverse.data.oneuid import (  # noqa: F401
    OneuidIndex,
    OneuidQuery,
    build_oneuid_index,
    load_oneuid_index,
    load_universal,
)


def load_catalog(
    survey: str,
    selection: Optional[Union[Selection, Sequence[Selection]]] = None,
    columns: Optional[List[str]] = None,
    validate: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Load a survey catalog as a standardized pandas DataFrame.

    Parameters
    ----------
    survey : str
        Registered survey name (e.g. ``"dummy"``, ``"sdss_mgs"``).
        See ``list_surveys()`` for all available names.
    selection : Selection or list[Selection] or None
        Spatial / redshift filters (AND logic).
        Examples: ``Cone(ra=185, dec=15, radius=5)``, ``Shell(0.02, 0.08)``.
    columns : list[str] or None
        Subset of columns to return.  None = all available.
    validate : bool
        Run schema validation (default True).
    **kwargs
        Forwarded to the survey loader (e.g. ``data_path``, ``n_galaxies``).

    Returns
    -------
    pd.DataFrame with lowercase oneuniverse column names.
    """
    loader = get_loader(survey)
    return loader.load(selection=selection, columns=columns, validate=validate, **kwargs)


__all__ = [
    # Main API
    "load_catalog",
    "list_surveys",
    "list_survey_types",
    "get_survey_config",
    # Selections
    "Cone",
    "Shell",
    "SkyPatch",
    # Configuration
    "get_data_root",
    "set_data_root",
    # Conversion & format
    "convert_survey",
    "convert_sightlines",
    "convert_healpix_map",
    "fetch_original_columns",
    "get_manifest",
    "get_geometry",
    "is_converted",
    "read_objects_table",
    "DataGeometry",
    "OneuniverseDatabase",
    "OneuidIndex",
    "OneuidQuery",
    "build_oneuid_index",
    "load_oneuid_index",
    "load_universal",
]
