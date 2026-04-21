"""Transaction-time metadata: which datasets were valid when.

:class:`DatasetValidity` carries the *database* time axis: the interval
during which the entry was the authoritative answer. Together with
:class:`oneuniverse.data.temporal.TemporalSpec` (physical observation
time), this implements the bitemporal model underlying
``OneuniverseDatabase.as_of(T)``.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Optional, Tuple


def _parse(ts: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        raise ValueError(
            f"DatasetValidity: timestamp {ts!r} has no timezone; "
            f"require ISO-8601 UTC (e.g. '2026-01-01T00:00:00+00:00')"
        )
    return parsed.astimezone(dt.timezone.utc)


@dataclass(frozen=True)
class DatasetValidity:
    """When this dataset was the authoritative answer.

    Attributes
    ----------
    valid_from_utc
        ISO-8601 UTC timestamp. Required.
    valid_to_utc
        ISO-8601 UTC timestamp. ``None`` means still current.
    version
        Free-form version label (``"dr16"``, ``"2.0"``,
        ``"2026-04-15-rebuild"``). Default ``"1.0"``.
    supersedes
        Zero or more dataset names whose validity this entry closes
        out.
    """

    valid_from_utc: str
    valid_to_utc: Optional[str] = None
    version: str = "1.0"
    supersedes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        t0 = _parse(self.valid_from_utc)
        if self.valid_to_utc is not None:
            t1 = _parse(self.valid_to_utc)
            if t1 <= t0:
                raise ValueError(
                    f"DatasetValidity: valid_from={self.valid_from_utc} "
                    f">= valid_to={self.valid_to_utc}"
                )
        object.__setattr__(self, "valid_from_utc", t0.isoformat())
        if self.valid_to_utc is not None:
            object.__setattr__(
                self, "valid_to_utc", _parse(self.valid_to_utc).isoformat(),
            )
        object.__setattr__(self, "supersedes", tuple(self.supersedes))

    def contains(self, when: dt.datetime) -> bool:
        if when.tzinfo is None:
            raise ValueError("DatasetValidity.contains: when must be tz-aware")
        t0 = _parse(self.valid_from_utc)
        if when < t0:
            return False
        if self.valid_to_utc is None:
            return True
        return when < _parse(self.valid_to_utc)

    def is_current(self, now: Optional[dt.datetime] = None) -> bool:
        now = now or dt.datetime.now(dt.timezone.utc)
        return self.contains(now)

    def closed_at(self, ts: str) -> "DatasetValidity":
        return replace(self, valid_to_utc=ts)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["supersedes"] = list(self.supersedes)
        return d

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DatasetValidity":
        return cls(
            valid_from_utc=raw["valid_from_utc"],
            valid_to_utc=raw.get("valid_to_utc"),
            version=raw.get("version", "1.0"),
            supersedes=tuple(raw.get("supersedes", ())),
        )
