"""Tomographic n(z) sub-spec for OUF 2.4.

:class:`TomographicNzSpec` is a **dataset-level** sidecar declaring a
per-bin n(z) plus the row-level column name carrying each row's
tomographic-bin assignment. It does not store probabilities per row
— that is what :class:`PdfSpec` is for. Used by weak-lensing surveys
(KiDS-1000, DES-Y3, HSC-Y3) and any pipeline that delivers stacked
n(z) per tomographic bin via SOM cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class TomographicNzSpec:
    """Per-bin n(z) on a shared z grid.

    Parameters
    ----------
    bin_edges
        ``[(z_lo_1, z_hi_1), (z_lo_2, z_hi_2), ...]`` — one tuple
        per tomographic bin.
    grid
        Shared z grid (length ``n_grid``) over which every bin's
        n(z) is evaluated.
    values
        Sequence of length ``len(bin_edges)``, each element a sequence
        of length ``len(grid)`` carrying that bin's n(z).
    bin_assignment_column
        Name of the integer row-level column that records which bin
        each object belongs to. Defaults to ``"tomo_bin"``.
    """

    bin_edges: List[Tuple[float, float]]
    grid: List[float]
    values: List[List[float]]
    bin_assignment_column: str = "tomo_bin"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_bins = len(self.bin_edges)
        n_grid = len(self.grid)
        if len(self.values) != n_bins:
            raise ValueError(
                f"values length ({len(self.values)}) must match number "
                f"of bin_edges ({n_bins})"
            )
        for i, row in enumerate(self.values):
            if len(row) != n_grid:
                raise ValueError(
                    f"values[{i}] length ({len(row)}) must match grid "
                    f"length ({n_grid})"
                )
        # Normalise to plain floats so JSON round-trips cleanly.
        object.__setattr__(
            self, "bin_edges",
            [(float(a), float(b)) for a, b in self.bin_edges],
        )
        object.__setattr__(self, "grid", [float(x) for x in self.grid])
        object.__setattr__(
            self, "values",
            [[float(x) for x in row] for row in self.values],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_edges": [list(e) for e in self.bin_edges],
            "grid": list(self.grid),
            "values": [list(row) for row in self.values],
            "bin_assignment_column": self.bin_assignment_column,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TomographicNzSpec":
        return cls(
            bin_edges=[tuple(e) for e in d["bin_edges"]],
            grid=list(d["grid"]),
            values=[list(row) for row in d["values"]],
            bin_assignment_column=d.get(
                "bin_assignment_column", "tomo_bin",
            ),
            extra=dict(d.get("extra", {})),
        )
