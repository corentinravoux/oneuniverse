"""oneuniverse.data._registry — survey-name → loader-class registry.

Backed by the shared :class:`oneuniverse._registry.Registry`. Loaders register
at import time via the ``@register`` decorator (keyed on ``cls.config.name``).
``_REGISTRY`` remains the live internal dict for back-compat with tests that
mutate it directly; ``REGISTRY`` is the read-only proxy.
"""
from __future__ import annotations

from typing import Dict, List, Mapping

from oneuniverse._registry import Registry

_REG: "Registry[type]" = Registry("survey loader", key=lambda cls: cls.config.name)

#: Live internal dict (tests mutate this directly; production goes via register()).
_REGISTRY: Dict[str, type] = _REG.items_dict
#: Read-only public view.
REGISTRY: Mapping[str, type] = _REG.mapping


def register(cls):
    """Class decorator: register a BaseSurveyLoader subclass by ``config.name``."""
    return _REG.register(cls)


def get_loader(name: str):
    """Return an *instance* of the loader registered under *name*."""
    return _REG.get(name)()  # registry stores the class; callers want an instance


def list_surveys(survey_type=None, status=None) -> Dict[str, str]:
    """Return ``{name: description}`` for registered surveys.

    ``survey_type`` filters by type; ``status`` filters by implementation status
    (``"ready"`` / ``"planned"``). With ``status=None`` (default) all are
    returned, with ``" [planned …]"`` appended to non-ready descriptions so
    discovery is never silent.
    """
    out = {}
    for name in _REG.names():
        cfg = _REG.get(name).config
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
    return getattr(_REG.get(name).config, "status", "ready")


def list_survey_types() -> List[str]:
    """Return sorted distinct survey types that have registered loaders."""
    return sorted({_REG.get(n).config.survey_type for n in _REG.names()})


def get_survey_config(name: str):
    """Return the SurveyConfig for *name* without loading data."""
    return _REG.get(name).config
