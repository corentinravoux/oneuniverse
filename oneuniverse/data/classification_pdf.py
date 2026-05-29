"""Per-row classification PDF sub-spec for OUF 2.4.

:class:`ClassificationPdfSpec` declares an ordered class label tuple
and the column on disk that stores the per-row probability vector.
Use cases: DESI ``SPECTYPE`` posteriors, ZTF / Fink classifier
outputs, AGN-vs-galaxy probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

_ALLOWED = frozenset({"categorical", "mixture"})


@dataclass(frozen=True)
class ClassificationPdfSpec:
    """Per-row class probability metadata.

    Parameters
    ----------
    classes
        Ordered tuple of class labels.
    parameterisation
        ``"categorical"`` (default — probabilities sum to ~1) or
        ``"mixture"`` (probabilities + component widths declared via
        ``extra``; not yet exercised by the reader).
    value_column
        Per-row column name on disk; stored as ``f4[n_classes]``.
        Default ``"class_pdf_values"``.
    """

    classes: Tuple[str, ...]
    parameterisation: str = "categorical"
    value_column: str = "class_pdf_values"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("classes must not be empty")
        if self.parameterisation not in _ALLOWED:
            raise ValueError(
                f"unknown parameterisation {self.parameterisation!r}; "
                f"allowed: {sorted(_ALLOWED)}"
            )
        object.__setattr__(
            self, "classes", tuple(str(c) for c in self.classes),
        )

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classes": list(self.classes),
            "parameterisation": self.parameterisation,
            "value_column": self.value_column,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassificationPdfSpec":
        return cls(
            classes=tuple(d["classes"]),
            parameterisation=d.get("parameterisation", "categorical"),
            value_column=d.get("value_column", "class_pdf_values"),
            extra=dict(d.get("extra", {})),
        )
