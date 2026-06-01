"""Per-backend execution capability declaration.

A backend (native-format reader) declares up-front which execution
modes it can deliver per heavy step. The reader / converter consults
this and refuses a mode the backend cannot honour, rather than
silently degrading to an unbounded in-memory path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

from oneuniverse.simulation.execution import ExecutionMode


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    native_format: str
    supports_mpi: bool = False
    supports_gpu_direct: bool = False
    supports_random_access: bool = False     # KD-tree / Hilbert key range
    supports_streaming: bool = True          # bounded-memory chunked iterator
    requires_extra: Tuple[str, ...] = ()     # ("abacusutils",), ("genericio",)
    # Per-heavy-step execution capability. Steps absent from this map
    # default to SEQUENTIAL-only.
    heavy_step_modes: Mapping[str, Tuple[ExecutionMode, ...]] = field(
        default_factory=dict
    )

    def modes_for(self, step: str) -> Tuple[ExecutionMode, ...]:
        """Modes available for ``step``; SEQUENTIAL-only if undeclared."""
        return tuple(
            self.heavy_step_modes.get(step, (ExecutionMode.SEQUENTIAL,))
        )

    def supports(self, step: str, mode: ExecutionMode) -> bool:
        return mode in self.modes_for(step)
