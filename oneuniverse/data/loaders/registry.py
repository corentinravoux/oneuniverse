"""Survey-loader registry (built-ins plus the ``oneuniverse.survey_loaders`` entry points)."""
from __future__ import annotations

import warnings
from importlib.metadata import entry_points

from oneuniverse._core.registry import Registry
from oneuniverse.data.loaders.base import SurveyLoader
from oneuniverse.data.manifest import SurveyMeta

ENTRY_POINT_GROUP = "oneuniverse.survey_loaders"
_registry: Registry[type[SurveyLoader]] = Registry("survey loader", key=lambda cls: cls.meta.name)


def register(cls):
    return _registry.register(cls)


def get_loader(name: str) -> SurveyLoader:
    return _registry.get(name)()


def list_surveys() -> dict[str, SurveyMeta]:
    return {n: _registry.get(n).meta for n in _registry.names()}


def load_entry_points() -> list[str]:
    """Register third-party loaders; a broken plugin is skipped with a warning.

    ``Registry.load_entry_points`` lets a failing ``ep.load()`` propagate, which would make
    ``oneuniverse.data`` unimportable, so the loop is repeated here with narrow handling.
    """
    added: list[str] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in _registry:
            continue
        try:
            cls = ep.load()
        except (ImportError, AttributeError) as err:
            warnings.warn(f"survey loader plugin {ep.name!r} failed to load: {err!r}",
                          RuntimeWarning, stacklevel=2)
            continue
        _registry.register(cls, name=ep.name)
        added.append(ep.name)
    return added
