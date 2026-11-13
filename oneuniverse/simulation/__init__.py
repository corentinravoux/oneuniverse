"""Simulation layer: typed specs, OUF-Sim manifest and execution planning.

Imports only ``oneuniverse._core``.  Store, converter and database layers are
added by later R5 tasks.
"""

from oneuniverse._core.geometry import Cone, Cube, SkyPatch
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.manifest import (
    LAYOUT_FILENAME,
    MANIFEST_FILENAME,
    OUFSIM_FORMAT_VERSION,
    OUFSimManifest,
    read_manifest,
    write_manifest,
)
from oneuniverse.simulation.specs import (
    PRODUCT_KINDS,
    CosmologySpec,
    ProductDecl,
    ProvenanceSpec,
    RegionSpec,
    SimulationRequest,
    UnitFrameSpec,
)

__all__ = [
    "LAYOUT_FILENAME",
    "MANIFEST_FILENAME",
    "OUFSIM_FORMAT_VERSION",
    "PRODUCT_KINDS",
    "Cone",
    "CosmologySpec",
    "Cube",
    "ExecutionMode",
    "ExecutionPlan",
    "OUFSimManifest",
    "ProductDecl",
    "ProvenanceSpec",
    "RegionSpec",
    "SimulationRequest",
    "SkyPatch",
    "UnitFrameSpec",
    "read_manifest",
    "write_manifest",
]
