"""
oneuniverse.data._registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Registry mapping survey names to loader classes.

Survey loaders register themselves at import time via the ``@register``
decorator.  The public API (``load_catalog``, ``list_surveys``) delegates
to this registry.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

_REGISTRY: Dict[str, Type] = {}


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
) -> Dict[str, str]:
    """Return ``{name: description}`` for registered surveys.

    Parameters
    ----------
    survey_type : str or None
        If given, filter to surveys of this type
        (e.g. ``"spectroscopic"``, ``"peculiar_velocity"``).
    """
    out = {}
    for name, cls in sorted(_REGISTRY.items()):
        cfg = cls.config
        if survey_type is not None and cfg.survey_type != survey_type:
            continue
        out[name] = cfg.description
    return out


def list_survey_types() -> List[str]:
    """Return sorted list of distinct survey types that have registered loaders."""
    return sorted({cls.config.survey_type for cls in _REGISTRY.values()})


def get_survey_config(name: str):
    """Return the SurveyConfig for *name* without loading data."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown survey '{name}'. Available: {available}")
    return _REGISTRY[name].config
