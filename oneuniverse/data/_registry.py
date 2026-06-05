"""
oneuniverse.data._registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Registry mapping survey names to loader classes.

Survey loaders register themselves at import time via the ``@register``
decorator.  The public API (``load_catalog``, ``list_surveys``) delegates
to this registry.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Type

_REGISTRY: Dict[str, Type] = {}

#: Read-only public view of the registry. Writes go through
#: :func:`register` (or directly to ``_REGISTRY`` in tests only).
REGISTRY: Mapping[str, Type] = MappingProxyType(_REGISTRY)


def register(cls):
    """Class decorator: register a BaseSurveyLoader subclass by its config.name."""
    name = cls.config.name
    if name in _REGISTRY:
        raise ValueError(
            f"Survey '{name}' is already registered by {_REGISTRY[name].__name__}. "
            f"Cannot register {cls.__name__}."
        )
    _REGISTRY[name] = cls
    return cls


def get_loader(name: str):
    """Return an instance of the loader registered under *name*.

    Raises KeyError if not found.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown survey '{name}'. Available: {available}"
        )
    return _REGISTRY[name]()


def list_surveys(
    survey_type: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, str]:
    """Return ``{name: description}`` for registered surveys.

    Parameters
    ----------
    survey_type : str or None
        If given, filter to surveys of this type
        (e.g. ``"spectroscopic"``, ``"peculiar_velocity"``).
    status : str or None
        If given (``"ready"`` / ``"planned"``), filter to that implementation
        status. ``"ready"`` loaders return data; ``"planned"`` loaders are
        registered scaffolds whose ``load()`` raises ``NotImplementedError``.
        Default ``None`` returns all, with ``" [planned …]"`` appended to the
        description of non-ready loaders so discovery is never silent.
    """
    out = {}
    for name, cls in sorted(_REGISTRY.items()):
        cfg = cls.config
        if survey_type is not None and cfg.survey_type != survey_type:
            continue
        st = getattr(cfg, "status", "ready")
        if status is not None and st != status:
            continue
        out[name] = cfg.description if st == "ready" \
            else f"{cfg.description} [planned — not yet implemented]"
    return out


def survey_status(name: str) -> str:
    """Return the implementation status (``"ready"`` / ``"planned"``) of *name*."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown survey '{name}'. Available: {available}")
    return getattr(_REGISTRY[name].config, "status", "ready")


def list_survey_types() -> List[str]:
    """Return sorted list of distinct survey types that have registered loaders."""
    return sorted({cls.config.survey_type for cls in _REGISTRY.values()})


def get_survey_config(name: str):
    """Return the SurveyConfig for *name* without loading data."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown survey '{name}'. Available: {available}")
    return _REGISTRY[name].config
