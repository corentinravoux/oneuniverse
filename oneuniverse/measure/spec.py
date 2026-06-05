"""Step 9: the aimed-measurement declaration. Cosmology deferred to P2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class MeasurementSpec:
    tracers: Tuple[str, ...]
    pairs: Tuple[Tuple[str, str], ...]
    statistic: str                    # "pk_multipole" | "xi_smu" | "w_theta" | ...
    estimator_family: str             # "clustering" | "field_level" | ...
    binning: Optional[dict] = None
    coords: str = "on_sky"            # comoving conversion happens in P2
    covariance: str = "jackknife"
