"""Sim-side cosmology declaration.

Duplicated (not imported) from Pillar 1 by design — Pillar 3 must not
depend on ``oneuniverse.data``. This records the cosmology a simulation
was *run with*; it is not a cosmology engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CosmologySpec:
    omega_m: Optional[float] = None
    omega_b: Optional[float] = None
    h: Optional[float] = None
    n_s: Optional[float] = None
    sigma8: Optional[float] = None
    w0: Optional[float] = None
    wa: Optional[float] = None
    t_cmb: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "omega_m": self.omega_m,
            "omega_b": self.omega_b,
            "h": self.h,
            "n_s": self.n_s,
            "sigma8": self.sigma8,
            "w0": self.w0,
            "wa": self.wa,
            "t_cmb": self.t_cmb,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CosmologySpec":
        return cls(
            omega_m=d.get("omega_m"),
            omega_b=d.get("omega_b"),
            h=d.get("h"),
            n_s=d.get("n_s"),
            sigma8=d.get("sigma8"),
            w0=d.get("w0"),
            wa=d.get("wa"),
            t_cmb=d.get("t_cmb"),
            extra=dict(d.get("extra", {})),
        )
