"""ProductDecl — a converter declares each product it found + which
Layer-1 indexers to run + which canonical fields it exposes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from oneuniverse.simulation._version import PRODUCT_KINDS


@dataclass(frozen=True)
class ProductDecl:
    product: str               # one of PRODUCT_KINDS
    native_format: str         # "Gadget HDF5" | "ASDF/pack9" | …
    indexes: Tuple[str, ...]   # Layer-1 indexer names to run
    fields: Tuple[str, ...]    # canonical field names exposed

    def __post_init__(self) -> None:
        if self.product not in PRODUCT_KINDS:
            raise ValueError(
                f"ProductDecl: unknown product {self.product!r}; "
                f"allowed: {list(PRODUCT_KINDS)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product": self.product,
            "native_format": self.native_format,
            "indexes": list(self.indexes),
            "fields": list(self.fields),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductDecl":
        return cls(
            product=d["product"],
            native_format=d["native_format"],
            indexes=tuple(d.get("indexes", ())),
            fields=tuple(d.get("fields", ())),
        )
