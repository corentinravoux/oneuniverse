"""oneuniverse.simulation — OUF-Sim: storage + orchestration of cosmological
simulations (Pillar 3, digital-twin substrate).

Standalone subpackage. **Must not import** from ``oneuniverse.data`` or
``oneuniverse.combine`` (enforced by test_sim_no_pillar1_imports).
"""
from oneuniverse.simulation._version import (
    LAYOUT_SCHEMAS,
    OUFSIM_FORMAT_VERSION,
    PRODUCT_KINDS,
    SIM_KINDS,
)
from oneuniverse.simulation.capabilities import BackendCapabilities
from oneuniverse.simulation.converter import (
    SimConverter,
    detect_converter,
    get_converter,
    register,
    registered_codes,
)
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.manifest import (
    OUFSimManifest,
    OUFSimManifestError,
    read_manifest,
    write_manifest,
)
from oneuniverse.simulation.product import ProductDecl
from oneuniverse.simulation.provenance import ProvenanceSpec
from oneuniverse.simulation.region import RegionSpec
from oneuniverse.simulation.request import SimulationRequest
from oneuniverse.simulation.selectors import Cone, Cube, SkyPatch
from oneuniverse.simulation.unit_frame import UnitFrameSpec

__all__ = [
    "OUFSIM_FORMAT_VERSION", "SIM_KINDS", "PRODUCT_KINDS", "LAYOUT_SCHEMAS",
    "ExecutionMode", "ExecutionPlan",
    "BackendCapabilities",
    "Cube", "Cone", "SkyPatch",
    "CosmologySpec", "UnitFrameSpec", "ProvenanceSpec",
    "ProductDecl",
    "OUFSimManifest", "read_manifest", "write_manifest", "OUFSimManifestError",
    "RegionSpec", "SimulationRequest",
    "SimConverter", "register", "get_converter", "detect_converter",
    "registered_codes",
]
