"""Read side of the store.  Minimal for now: Task 4 adds the selector reads."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse._core.errors import FormatError, NotFoundError
from oneuniverse.simulation.manifest import (
    LAYOUT_FILENAME,
    MANIFEST_FILENAME,
    OUFSimManifest,
    read_manifest,
)

INDEX_FILENAME = "_index.parquet"


def _tag(z: float) -> str:
    return f"{float(z):.3f}"


class SimStore:
    """A partial-access simulation store rooted at ``root``."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        if not (self.root / MANIFEST_FILENAME).exists():
            raise NotFoundError(f"no {MANIFEST_FILENAME} in {self.root}")
        self.manifest: OUFSimManifest = read_manifest(self.root / MANIFEST_FILENAME)
        try:
            self.layout: dict = json.loads((self.root / LAYOUT_FILENAME).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FormatError(f"cannot read {LAYOUT_FILENAME} in {self.root}: {exc}") from exc
        self._index_cache: dict[tuple[str, str], pa.Table] = {}

    @property
    def products(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.manifest.products)

    def _entry(self, product: str, z: float) -> dict:
        try:
            return self.layout["products"][product][_tag(z)]
        except KeyError:
            raise NotFoundError(f"no product {product!r} at z={_tag(z)} in {self.root}") from None

    def redshifts(self, product: str) -> tuple[float, ...]:
        zs = self.manifest.product(product).redshifts
        have = self.layout["products"].get(product, {})
        return tuple(z for z in zs if _tag(z) in have)

    def index(self, product: str, z: float) -> pa.Table:
        key = (product, _tag(z))
        if key not in self._index_cache:
            entry = self._entry(product, z)
            self._index_cache[key] = pq.read_table(self.root / entry["dir"] / INDEX_FILENAME)
        return self._index_cache[key]
