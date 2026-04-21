"""
oneuniverse.data.subobject_rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`SubobjectRules` — policy for building sub-object links between
a *parent* survey (galaxies, clusters) and a *child* survey (SNe,
TDEs). Symmetric in spirit to :class:`CrossMatchRules`, but records a
relation of *containment*, not identity, and is directional
(parent -> child).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, eq=False)
class SubobjectRules:
    parent_survey_type: str
    child_survey_type: str
    sky_tol_arcsec: float = 1.0
    dz_tol: Optional[float] = 5e-3
    relation: str = "contains"
    accept_ambiguous: bool = False

    def __post_init__(self) -> None:
        if not self.parent_survey_type:
            raise ValueError("SubobjectRules: parent_survey_type must be non-empty")
        if not self.child_survey_type:
            raise ValueError("SubobjectRules: child_survey_type must be non-empty")
        if self.sky_tol_arcsec <= 0.0:
            raise ValueError(
                f"SubobjectRules: sky_tol_arcsec must be positive, "
                f"got {self.sky_tol_arcsec!r}"
            )
        if self.dz_tol is not None and self.dz_tol < 0.0:
            raise ValueError(
                f"SubobjectRules: dz_tol must be non-negative or None, "
                f"got {self.dz_tol!r}"
            )
        if not self.relation:
            raise ValueError("SubobjectRules: relation must be non-empty")

    def _canonical(self) -> dict:
        return {
            "parent_survey_type": self.parent_survey_type,
            "child_survey_type": self.child_survey_type,
            "sky_tol_arcsec": float(self.sky_tol_arcsec),
            "dz_tol": None if self.dz_tol is None else float(self.dz_tol),
            "relation": self.relation,
            "accept_ambiguous": bool(self.accept_ambiguous),
        }

    def hash(self) -> str:
        payload = json.dumps(self._canonical(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubobjectRules):
            return NotImplemented
        return self.hash() == other.hash()
