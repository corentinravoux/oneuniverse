"""
oneuniverse.data._base_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for survey loaders.

Every concrete survey loader inherits from ``BaseSurveyLoader`` and lives
inside a survey sub-sub-package under ``data/surveys/{type}/{name}/``.

A loader declares its identity through class-level attributes and
implements ``_load_raw()`` to read native data.  The base class handles
selection filtering, schema validation, column subsetting, and data-path
resolution so that individual loaders stay simple.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from oneuniverse.data import schema
from oneuniverse.data._config import resolve_survey_path
from oneuniverse.data.selection import Selection, apply_selections

logger = logging.getLogger(__name__)


# ── Survey metadata descriptor ───────────────────────────────────────────


@dataclass(frozen=True)
class SurveyConfig:
    """Declarative metadata for a survey sub-sub-package.

    Every survey loader exposes one ``SurveyConfig`` as a class attribute.
    The registry and the public API use it for discovery and documentation.
    """

    # Identity
    name: str               # unique key, e.g. "sdss_mgs", "cosmicflows4"
    survey_type: str        # category: "spectroscopic", "photometric",
                            # "peculiar_velocity", "snia", "radio", "test"
    description: str        # one-line human-readable summary

    # Schema
    column_groups: Tuple[str, ...] = ("core",)

    # Characteristic fields — survey-specific observables beyond the schema
    # {field_name: (dtype, unit, description)}
    # These are the columns that make this survey unique.
    characteristic_fields: Dict[str, Tuple[str, str, str]] = field(
        default_factory=dict
    )

    # Data on disk
    data_subpath: str = ""        # path relative to data root, e.g. "spectroscopic/eboss/qso"
    data_filename: str = ""       # expected filename inside the survey dir
    data_format: str = ""         # "fits", "parquet", "csv", "hdf5", …

    # Provenance
    reference: str = ""           # e.g. "York+2000, AJ 120 1579"
    url: str = ""                 # data release URL

    # Sky coverage (rough, for documentation / quick filtering)
    sky_fraction: float = 0.0     # fraction of 4π covered  (0–1)
    z_range: Tuple[float, float] = (0.0, 0.0)
    n_objects_approx: int = 0     # approximate catalog size


# ── Abstract loader ──────────────────────────────────────────────────────


class BaseSurveyLoader(abc.ABC):
    """Abstract base for loading a survey catalog.

    Subclasses must define
    ----------------------
    config : SurveyConfig (class attribute)
        All survey metadata in one place.
    _load_raw(data_path, **kwargs) → pd.DataFrame
        Read the catalog from *data_path* (a ``pathlib.Path`` or ``None``
        for in-memory surveys like ``dummy``).
    """

    config: SurveyConfig  # to be set on the class by each survey

    @abc.abstractmethod
    def _load_raw(self, data_path: Optional[Path] = None, **kwargs) -> pd.DataFrame:
        """Load the full catalog (no filtering) as a DataFrame.

        Parameters
        ----------
        data_path : Path or None
            Resolved path to the survey data directory.
            ``None`` if no data root is configured (test/in-memory surveys
            must handle this case).
        """
        ...

    # ── Convenience properties (delegate to config) ──────────────────────

    @property
    def survey_name(self) -> str:
        return self.config.name

    @property
    def survey_type(self) -> str:
        return self.config.survey_type

    @property
    def survey_description(self) -> str:
        return self.config.description

    @property
    def column_groups(self) -> List[str]:
        return list(self.config.column_groups)

    # ── Public interface ─────────────────────────────────────────────────

    def load(
        self,
        selection: Optional[Union[Selection, Sequence[Selection]]] = None,
        columns: Optional[List[str]] = None,
        validate: bool = True,
        data_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Load catalog, apply selections, return standardized DataFrame.

        Parameters
        ----------
        selection : Selection or list[Selection] or None
            Spatial / redshift filters applied *after* loading (AND logic).
        columns : list[str] or None
            Subset of columns to return.  ``None`` = all available.
        validate : bool
            Run schema validation (default True).
        data_path : str or Path or None
            Explicit path override.  If ``None``, resolved from the global
            data root + survey type + survey name.
        **kwargs
            Forwarded to ``_load_raw()`` (e.g. options specific to a survey).

        Returns
        -------
        pd.DataFrame with lowercase oneuniverse column names.
        """
        # Resolve data path
        if data_path is not None:
            resolved = Path(data_path)
        else:
            resolved = resolve_survey_path(
                self.config.survey_type,
                self.config.name,
                self.config.data_subpath,
            )

        # Prefer oneuniverse Parquet format if available
        from oneuniverse.data.converter import is_converted, read_oneuniverse_parquet
        force_native = kwargs.pop("force_native", False)
        if (
            resolved is not None
            and is_converted(resolved)
            and not force_native
        ):
            logger.info(
                "[%s] Loading from oneuniverse Parquet (use force_native=True for original)",
                self.survey_name,
            )
            # Read all columns if validating; else pushdown the subset.
            read_cols = None if validate else columns
            df = read_oneuniverse_parquet(resolved, columns=read_cols)
        else:
            df = self._load_raw(data_path=resolved, **kwargs)

        # Validate schema
        if validate:
            warnings = schema.validate_dataframe(df, self.column_groups)
            for w in warnings:
                logger.warning("Schema warning [%s]: %s", self.survey_name, w)

        # Apply selections
        if selection is not None:
            mask = apply_selections(
                df["ra"].to_numpy(),
                df["dec"].to_numpy(),
                df["z"].to_numpy(),
                selection,
            )
            n_before = len(df)
            df = df.loc[mask].reset_index(drop=True)
            logger.info(
                "[%s] Selection: %d / %d galaxies kept (%.1f%%)",
                self.survey_name,
                len(df),
                n_before,
                100 * len(df) / max(n_before, 1),
            )

        # Column subsetting
        if columns is not None:
            missing = set(columns) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Requested columns not in catalog: {sorted(missing)}. "
                    f"Available: {sorted(df.columns)}"
                )
            df = df[list(columns)]

        return df

    # ── Introspection helpers ────────────────────────────────────────────

    def available_columns(self) -> Dict[str, schema.ColumnDef]:
        """Return all column definitions this loader can provide."""
        return schema.get_all_columns(self.column_groups)

    def info(self) -> str:
        """Human-readable summary of this survey."""
        c = self.config
        cols = self.available_columns()
        n_req = sum(1 for v in cols.values() if v.required)
        n_opt = sum(1 for v in cols.values() if not v.required)
        data_path = resolve_survey_path(c.survey_type, c.name, c.data_subpath)

        lines = [
            f"Survey:      {c.name}",
            f"Type:        {c.survey_type}",
            f"Description: {c.description}",
            f"Reference:   {c.reference or '—'}",
            f"z range:     [{c.z_range[0]:.3f}, {c.z_range[1]:.3f}]",
            f"Sky:         {c.sky_fraction*100:.0f}%  (~{c.n_objects_approx:,} objects)",
            f"Groups:      {', '.join(c.column_groups)}",
            f"Columns:     {n_req} required + {n_opt} optional = {len(cols)} total",
            f"Data path:   {data_path or '(not configured)'}",
        ]
        if c.characteristic_fields:
            lines.append("Characteristic fields:")
            for fname, (dt, unit, desc) in c.characteristic_fields.items():
                u = f" [{unit}]" if unit else ""
                lines.append(f"  {fname:20s} {dt:4s}{u:10s}  {desc}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}('{self.survey_name}', type='{self.survey_type}')>"
