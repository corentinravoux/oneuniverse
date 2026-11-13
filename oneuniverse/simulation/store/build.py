"""Store construction.  ``build_store`` arrives in a later R5 task; this module
currently holds only the :class:`NativeProduct` description."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow as pa

from oneuniverse.simulation.specs import ProductDecl


@dataclass
class NativeProduct:
    """One product of a native simulation, ready to be written into a store.

    ``sources`` maps redshift to the native file; ``adapter_format`` names the
    native reader adapter (``None``: default for the layout); ``batches`` yields
    record batches of a catalog or lightcone product at a redshift.
    """

    decl: ProductDecl
    kind_layout: Literal["catalog", "field", "lightcone"]
    sources: dict[float, Path]
    adapter_format: str | None = None
    batches: Callable[[float], Iterable[pa.RecordBatch]] | None = None
