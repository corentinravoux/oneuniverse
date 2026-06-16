"""Store-layout sidecar (review S11).

The per-product ``store_layout`` map (which products / redshifts / chunk files a
store holds) used to live inside ``manifest.json``, bloating the identity
manifest. It now lives in its own ``_store_layout.json`` sidecar — the OUF-Sim
analogue of OUF 2.6's ``_index.parquet``. Readers fall back to the manifest for
stores written before this change, so old stores keep working.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from oneuniverse.simulation.oufsim._io import read_json, write_json

STORE_LAYOUT_FILENAME = "_store_layout.json"


def write_store_layout(store: Union[str, Path], layout: dict) -> None:
    """Write the layout map to the sidecar beside ``manifest.json``."""
    write_json(Path(store) / STORE_LAYOUT_FILENAME, layout)


def read_store_layout(store: Union[str, Path]) -> dict:
    """Read the layout map: sidecar if present, else the legacy manifest field."""
    store = Path(store)
    sidecar = store / STORE_LAYOUT_FILENAME
    if sidecar.exists():
        return read_json(sidecar)
    manifest = store / "manifest.json"
    if manifest.exists():  # back-compat: pre-S11 stores embedded it
        return read_json(manifest).get("store_layout", {})
    return {}
