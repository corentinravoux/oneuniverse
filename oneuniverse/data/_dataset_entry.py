"""
oneuniverse.data._dataset_entry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One row of :class:`OneuniverseDatabase`'s in-memory registry.

Before Phase 6 the database kept three parallel dicts
(``_loaders`` / ``_manifests`` / ``_paths``) keyed by dataset name.
Keeping them in lockstep was fragile and meant any accessor had to
triple-check membership. :class:`DatasetEntry` collapses them into a
single frozen record — one authoritative slot per dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:  # pragma: no cover
    from oneuniverse.data._base_loader import BaseSurveyLoader
    from oneuniverse.data.manifest import Manifest


@dataclass(frozen=True)
class DatasetEntry:
    """In-memory record for one dataset discovered under the database root.

    Attributes
    ----------
    loader : type[BaseSurveyLoader]
        Dynamically generated loader class.
    manifest : Manifest
        The typed manifest loaded from disk.
    path : Path
        The survey directory (parent of ``oneuniverse/``).
    """

    loader: "Type[BaseSurveyLoader]"
    manifest: "Manifest"
    path: Path
