"""Sub-object hierarchy links — a generic parent→child relation on a product.

Covers cluster→members, strong-lens system→images (with time-delay payload),
QSO→DLA/BAL, GW event→host, deblender parent→child. Mirrors P1's sub-object
sidecars; carried on a DataProduct so multi-level probes are expressible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SubObjectLinks:
    role: str                              # cluster_member | lens_image | dla | host | ...
    parent_ids: np.ndarray
    child_ids: np.ndarray
    confidence: Optional[np.ndarray] = None
    payload: Optional[dict] = None         # role-specific (Δt, N_HI, κ_ext, ...)

    def __len__(self) -> int:
        return int(len(self.parent_ids))
