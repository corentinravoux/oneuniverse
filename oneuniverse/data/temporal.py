"""Temporal (observation-time) metadata for OUF 2.1 manifests.

:class:`TemporalSpec` describes the *physical* time axis of a temporal
dataset — the axis astronomers plot on a lightcurve. It is not to be
confused with :class:`oneuniverse.data.validity.DatasetValidity`, which
tracks *database* (transaction) time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

_ALLOWED_TIME_REFERENCES = frozenset({"TDB", "TAI", "UTC", "TT"})


@dataclass(frozen=True)
class TemporalSpec:
    """Row-level observation-time axis metadata.

    Attributes
    ----------
    t_min, t_max
        Inclusive range covered by the dataset. Units set by
        :attr:`time_unit` (default ``"MJD"``).
    time_column
        Parquet column carrying the per-row time stamp. Default
        ``"t_obs"``.
    time_unit
        Default ``"MJD"``. Surveys quoting other units should convert.
    time_reference
        One of ``"TDB"`` (default), ``"TAI"``, ``"UTC"``, ``"TT"``.
    cadence
        Optional typical sampling interval, in the same units as
        ``t_min``/``t_max``. Informational only.
    """

    t_min: float
    t_max: float
    time_column: str = "t_obs"
    time_unit: str = "MJD"
    time_reference: str = "TDB"
    cadence: Optional[float] = None

    def __post_init__(self) -> None:
        if self.t_max < self.t_min:
            raise ValueError(
                f"TemporalSpec: t_min={self.t_min} > t_max={self.t_max}"
            )
        if self.time_reference not in _ALLOWED_TIME_REFERENCES:
            raise ValueError(
                f"TemporalSpec: time_reference={self.time_reference!r} "
                f"not in {sorted(_ALLOWED_TIME_REFERENCES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TemporalSpec":
        return cls(**raw)
