"""
oneuniverse.data.oneuid_rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`CrossMatchRules` — the z-type-aware cross-match policy.

The ONEUID build consumes a :class:`CrossMatchRules` instance. Rules are
*symmetric in z-type pairs* (``("spec", "phot")`` and ``("phot", "spec")``
refer to the same bucket) and produce a stable short hash so built
indices can record the exact policy that generated them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import FrozenSet, Mapping, Optional, Tuple


ZtypePair = Tuple[str, str]


@dataclass(frozen=True, eq=False)
class CrossMatchRules:
    """Pairwise z-type rules for ONEUID cross-matching.

    Parameters
    ----------
    sky_tol_arcsec
        Sky-separation tolerance used by the cross-matcher (arcsec).
    dz_tol_default
        Baseline ``|Δz|`` tolerance applied when no pair-specific
        override is set. ``None`` disables the redshift cut entirely.
    dz_tol_by_ztype
        Per-pair ``|Δz|`` tolerance. Keys are ``(ztype_a, ztype_b)``
        tuples; lookup is order-insensitive.
    reject_ztype
        Pairs that must never match regardless of sky/z agreement.
        Order-insensitive — ``{("phot","phot")}`` blocks phot×phot links.
    """

    sky_tol_arcsec: float = 1.0
    dz_tol_default: Optional[float] = 1e-3
    dz_tol_by_ztype: Mapping[ZtypePair, float] = field(default_factory=dict)
    reject_ztype: FrozenSet[ZtypePair] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Normalise pair keys to sorted form so lookups are symmetric
        # and so two semantically-equal rule objects hash identically.
        norm_dz = {self._key(*k): v for k, v in dict(self.dz_tol_by_ztype).items()}
        norm_rej = frozenset(self._key(*p) for p in self.reject_ztype)
        object.__setattr__(self, "dz_tol_by_ztype", norm_dz)
        object.__setattr__(self, "reject_ztype", norm_rej)

    # ── Lookups ──────────────────────────────────────────────────────

    @staticmethod
    def _key(a: str, b: str) -> ZtypePair:
        return tuple(sorted((a, b)))  # type: ignore[return-value]

    def dz_tol_for(self, ztype_a: str, ztype_b: str) -> Optional[float]:
        """Return the ``|Δz|`` tolerance for the (a, b) pair."""
        k = self._key(ztype_a, ztype_b)
        if k in self.dz_tol_by_ztype:
            return self.dz_tol_by_ztype[k]
        return self.dz_tol_default

    def accepts(self, ztype_a: str, ztype_b: str) -> bool:
        """``False`` iff (a, b) — in any order — is in ``reject_ztype``."""
        return self._key(ztype_a, ztype_b) not in self.reject_ztype

    # ── Hashing / serialisation ─────────────────────────────────────

    def _canonical(self) -> dict:
        """Order-invariant canonical form used for hashing."""
        return {
            "sky_tol_arcsec": self.sky_tol_arcsec,
            "dz_tol_default": self.dz_tol_default,
            "dz_tol_by_ztype": sorted(
                [list(self._key(*k)), v]
                for k, v in self.dz_tol_by_ztype.items()
            ),
            "reject_ztype": sorted(
                list(self._key(*k)) for k in self.reject_ztype
            ),
        }

    def hash(self) -> str:
        """Short sha256 (16 hex chars) over the canonical form."""
        payload = json.dumps(self._canonical(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self.hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CrossMatchRules):
            return NotImplemented
        return self.hash() == other.hash()
