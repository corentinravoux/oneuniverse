# Phase S2 — OUF-Sim Package Skeleton + Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the standalone `oneuniverse.sim` subpackage with every OUF-Sim dataclass type (manifest, cosmology / unit-frame / provenance sidecars, ProductDecl, ExecutionMode / ExecutionPlan, BackendCapabilities, spatial selectors, RegionSpec, SimulationRequest, SimConverter ABC + registry) plus a lint guard that fails if any Pillar-1 import sneaks in. No backend, no real-data ingest.

**Architecture:** Pure-Python types + YAML serialisation, mirroring the OUF data `Manifest` discipline (typed dataclass, pinned `oufsim_format_version`, `to_dict`/`from_dict`, read/write helpers). The subpackage lives under `oneuniverse/sim/` but is a **standalone island**: zero imports from `oneuniverse.data` / `oneuniverse.combine`, enforced by a source-scanning test. Build order follows the architecture's central-hub structure (validated by the graphify knowledge graph): the `OUFSimManifest` format hub first, then the execution / capability substrate, then the converter ABC + registry, then the orchestration types.

**Tech Stack:** Python 3.9+, dataclasses, enum, abc, PyYAML (new dependency), pytest. Spec: [`2026-06-01-phaseS1-oufsim-architecture.md`](2026-06-01-phaseS1-oufsim-architecture.md).

---

## File Structure

All new files under `Packages/oneuniverse/oneuniverse/sim/`:

- `__init__.py` — public exports.
- `_version.py` — `OUFSIM_FORMAT_VERSION`, `SIM_KINDS`, `PRODUCT_KINDS`, `LAYOUT_SCHEMAS`.
- `execution.py` — `ExecutionMode` (enum), `ExecutionPlan` (dataclass).
- `capabilities.py` — `BackendCapabilities` (dataclass + helpers).
- `selectors.py` — `Cube`, `Cone`, `SkyPatch` (spatial selector dataclasses).
- `cosmology.py` — `CosmologySpec` (sim-side; duplicated, no Pillar-1 import).
- `unit_frame.py` — `UnitFrameSpec`.
- `provenance.py` — `ProvenanceSpec`.
- `product.py` — `ProductDecl`.
- `manifest.py` — `OUFSimManifest`, `read_manifest`, `write_manifest`, `OUFSimManifestError`.
- `region.py` — `RegionSpec`.
- `request.py` — `SimulationRequest`.
- `converter.py` — `SimConverter` (ABC), `register`, `get_converter`, `detect_converter`, `registered_codes`.

Tests under `Packages/oneuniverse/test/`:

- `test_sim_execution.py`
- `test_sim_capabilities.py`
- `test_sim_selectors.py`
- `test_sim_sidecar_specs.py` (cosmology + unit_frame + provenance)
- `test_sim_product.py`
- `test_sim_manifest.py`
- `test_sim_region.py`
- `test_sim_request.py`
- `test_sim_converter_registry.py`
- `test_sim_no_pillar1_imports.py` (the lint guard)

---

## Pre-flight

- [ ] **Step 0a: Confirm baseline is green.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest -q 2>&1 | tail -3
```

Expected: `522 passed, 2 skipped` (the post-Phase-22 baseline). If not, stop.

- [ ] **Step 0b: Worktree.**

Per `superpowers:using-git-worktrees`, work on a branch `phaseS2-oufsim-skeleton` (or continue on the working branch if that is the established repo workflow — prior phases committed directly to `main`).

---

## Task 1: Add PyYAML dependency + package directory

**Files:**
- Modify: `pyproject.toml`
- Create: `oneuniverse/sim/__init__.py`

- [ ] **Step 1: Add PyYAML to core dependencies**

Open `pyproject.toml`, find the `[project]` `dependencies = [...]` list, and add `"pyyaml>=6.0"`. If a `dependencies` array does not exist under `[project]`, add it:

```toml
dependencies = [
    "numpy",
    "pyyaml>=6.0",
]
```

(Keep whatever entries already exist; just ensure `pyyaml>=6.0` is present.)

- [ ] **Step 2: Create the empty subpackage init**

Create `oneuniverse/sim/__init__.py` with a docstring and a version re-export only (fuller exports come in later tasks):

```python
"""oneuniverse.sim — OUF-Sim: storage + orchestration of cosmological
simulations (Pillar 3, digital-twin substrate).

Standalone subpackage. **Must not import** from ``oneuniverse.data`` or
``oneuniverse.combine`` (enforced by test_sim_no_pillar1_imports).
"""
from oneuniverse.sim._version import OUFSIM_FORMAT_VERSION

__all__ = ["OUFSIM_FORMAT_VERSION"]
```

- [ ] **Step 3: Create the version module**

Create `oneuniverse/sim/_version.py`:

```python
"""OUF-Sim format version + the controlled vocabularies it validates."""
from __future__ import annotations

OUFSIM_FORMAT_VERSION: str = "0.1.0"

# Simulation kinds (manifest.sim_kind).
SIM_KINDS = (
    "nbody", "sph", "amr", "pm", "gr",
    "phase_space", "constrained", "differentiable",
)

# Product subdirectories (manifest.products + ProductDecl.product).
PRODUCT_KINDS = (
    "snapshots", "halos", "tree", "lightcone", "fields",
    "phase_space", "gr_fields", "checkpoints", "ic_posterior",
)

# Hierarchy patterns (manifest.layout_schema), from the research landscape §5.2.
LAYOUT_SCHEMAS = (
    "per_cosmology_phase_snapshot",
    "per_simulation_snapshot_chunk",
    "per_healpix_tile",
    "per_realisation_lightcone",
    "per_zoom_region",
)
```

- [ ] **Step 4: Verify importable**

```bash
python3 -c "from oneuniverse.sim import OUFSIM_FORMAT_VERSION; print(OUFSIM_FORMAT_VERSION)"
```

Expected: `0.1.0`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml oneuniverse/sim/__init__.py oneuniverse/sim/_version.py
git commit -m "phaseS2/T1: oneuniverse.sim package skeleton + OUFSIM_FORMAT_VERSION + vocabularies; add pyyaml dep"
```

---

## Task 2: `ExecutionMode` + `ExecutionPlan`

**Files:**
- Create: `oneuniverse/sim/execution.py`
- Test: `test/test_sim_execution.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_sim_execution.py
"""Phase S2 T2 — ExecutionMode + ExecutionPlan."""
import pytest

from oneuniverse.sim.execution import ExecutionMode, ExecutionPlan


def test_execution_modes():
    assert ExecutionMode.SEQUENTIAL.value == "sequential"
    assert ExecutionMode.MPI.value == "mpi"
    assert ExecutionMode.GPU.value == "gpu"


def test_plan_defaults():
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL, memory_budget_bytes=4 * 1024**3)
    assert plan.batch_rows is None
    assert plan.device is None
    assert plan.n_chunks_estimate == 0


def test_plan_rejects_nonpositive_budget():
    with pytest.raises(ValueError, match="memory_budget_bytes"):
        ExecutionPlan(mode=ExecutionMode.SEQUENTIAL, memory_budget_bytes=0)


def test_plan_rejects_nonpositive_batch():
    with pytest.raises(ValueError, match="batch_rows"):
        ExecutionPlan(
            mode=ExecutionMode.GPU, memory_budget_bytes=1024, batch_rows=0,
        )


def test_plan_is_frozen():
    plan = ExecutionPlan(mode=ExecutionMode.MPI, memory_budget_bytes=1024)
    with pytest.raises(Exception):
        plan.mode = ExecutionMode.GPU  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_sim_execution.py -v
```

Expected: `ImportError: No module named 'oneuniverse.sim.execution'`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/sim/execution.py
"""Execution model for heavy OUF-Sim steps.

Optimisation is load-bearing (Pillar-3 Rule 5): every heavy-memory /
heavy-CPU-time step runs sequential-streamed (bounded memory),
MPI-collective, or GPU. An :class:`ExecutionPlan` declares the mode +
a hard memory budget; the chunk size derives from the budget, never
"the whole snapshot". The MPI communicator is intentionally NOT stored
on the (frozen, serialisable) plan — it is passed at call time.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"   # streamed, bounded working set
    MPI = "mpi"                 # collective, per-rank-local
    GPU = "gpu"                 # device-resident, GPUDirect where possible


@dataclass(frozen=True)
class ExecutionPlan:
    """How a heavy step will run + its memory budget.

    Parameters
    ----------
    mode
        One of :class:`ExecutionMode`.
    memory_budget_bytes
        Hard cap on the per-process working set. Must be > 0.
    batch_rows
        Chunk size for SEQUENTIAL / GPU streaming. ``None`` = derive
        from ``memory_budget_bytes`` at call time. If given, must be > 0.
    device
        e.g. ``"cuda:0"`` for GPU mode; ``None`` otherwise.
    n_chunks_estimate
        Estimated number of chunks (for progress / scheduling).
    """

    mode: ExecutionMode
    memory_budget_bytes: int
    batch_rows: Optional[int] = None
    device: Optional[str] = None
    n_chunks_estimate: int = 0

    def __post_init__(self) -> None:
        if self.memory_budget_bytes <= 0:
            raise ValueError(
                f"ExecutionPlan.memory_budget_bytes must be > 0, "
                f"got {self.memory_budget_bytes!r}"
            )
        if self.batch_rows is not None and self.batch_rows <= 0:
            raise ValueError(
                f"ExecutionPlan.batch_rows must be > 0 or None, "
                f"got {self.batch_rows!r}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_sim_execution.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/sim/execution.py test/test_sim_execution.py
git commit -m "phaseS2/T2: ExecutionMode + ExecutionPlan (memory-budget-bounded, MPI comm not stored)"
```

---

## Task 3: `BackendCapabilities`

**Files:**
- Create: `oneuniverse/sim/capabilities.py`
- Test: `test/test_sim_capabilities.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_sim_capabilities.py
"""Phase S2 T3 — BackendCapabilities."""
from oneuniverse.sim.capabilities import BackendCapabilities
from oneuniverse.sim.execution import ExecutionMode


def test_defaults():
    cap = BackendCapabilities(name="dummy", native_format="Gadget HDF5")
    assert cap.supports_mpi is False
    assert cap.supports_gpu_direct is False
    assert cap.supports_random_access is False
    assert cap.supports_streaming is True
    assert cap.requires_extra == ()
    assert cap.heavy_step_modes == {}


def test_modes_for_default_is_sequential():
    cap = BackendCapabilities(name="x", native_format="f")
    assert cap.modes_for("region_extract") == (ExecutionMode.SEQUENTIAL,)


def test_modes_for_declared_step():
    cap = BackendCapabilities(
        name="x", native_format="f",
        heavy_step_modes={
            "region_extract": (ExecutionMode.SEQUENTIAL, ExecutionMode.MPI),
        },
    )
    assert cap.modes_for("region_extract") == (
        ExecutionMode.SEQUENTIAL, ExecutionMode.MPI,
    )


def test_supports_mode():
    cap = BackendCapabilities(
        name="x", native_format="f",
        heavy_step_modes={"index_build": (ExecutionMode.MPI,)},
    )
    assert cap.supports("index_build", ExecutionMode.MPI) is True
    assert cap.supports("index_build", ExecutionMode.GPU) is False
    # Undeclared step defaults to SEQUENTIAL-only.
    assert cap.supports("foo", ExecutionMode.SEQUENTIAL) is True
    assert cap.supports("foo", ExecutionMode.MPI) is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_sim_capabilities.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/sim/capabilities.py
"""Per-backend execution capability declaration.

A backend (native-format reader) declares up-front which execution
modes it can deliver per heavy step. The reader / converter consults
this and refuses a mode the backend cannot honour, rather than
silently degrading to an unbounded in-memory path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

from oneuniverse.sim.execution import ExecutionMode


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    native_format: str
    supports_mpi: bool = False
    supports_gpu_direct: bool = False
    supports_random_access: bool = False     # KD-tree / Hilbert key range
    supports_streaming: bool = True          # bounded-memory chunked iterator
    requires_extra: Tuple[str, ...] = ()     # ("abacusutils",), ("genericio",)
    # Per-heavy-step execution capability. Steps absent from this map
    # default to SEQUENTIAL-only.
    heavy_step_modes: Mapping[str, Tuple[ExecutionMode, ...]] = field(
        default_factory=dict
    )

    def modes_for(self, step: str) -> Tuple[ExecutionMode, ...]:
        """Modes available for ``step``; SEQUENTIAL-only if undeclared."""
        return tuple(
            self.heavy_step_modes.get(step, (ExecutionMode.SEQUENTIAL,))
        )

    def supports(self, step: str, mode: ExecutionMode) -> bool:
        return mode in self.modes_for(step)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_sim_capabilities.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/sim/capabilities.py test/test_sim_capabilities.py
git commit -m "phaseS2/T3: BackendCapabilities + heavy_step_modes (refuse unsupported mode, no silent unbounded fallback)"
```

---

## Task 4: Spatial selectors (`Cube`, `Cone`, `SkyPatch`)

**Files:**
- Create: `oneuniverse/sim/selectors.py`
- Test: `test/test_sim_selectors.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_sim_selectors.py
"""Phase S2 T4 — spatial selectors."""
import pytest

from oneuniverse.sim.selectors import Cone, Cube, SkyPatch


def test_cube_ok():
    c = Cube(0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
    assert c.xhi == 1.0


def test_cube_rejects_inverted():
    with pytest.raises(ValueError, match="xlo"):
        Cube(1.0, 0.0, 0.0, 1.0, 0.0, 1.0)


def test_cone_ok():
    c = Cone(lon=120.0, lat=0.0, radius_deg=5.0)
    assert c.radius_deg == 5.0


def test_cone_rejects_nonpositive_radius():
    with pytest.raises(ValueError, match="radius_deg"):
        Cone(lon=0.0, lat=0.0, radius_deg=0.0)


def test_skypatch_ok():
    p = SkyPatch(0.0, 30.0, -10.0, 10.0)
    assert p.lon_max == 30.0


def test_skypatch_rejects_inverted_lat():
    with pytest.raises(ValueError, match="lat"):
        SkyPatch(0.0, 30.0, 10.0, -10.0)


def test_selectors_frozen():
    c = Cube(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    with pytest.raises(Exception):
        c.xlo = 5.0  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_sim_selectors.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/sim/selectors.py
"""Spatial selectors for partial-access reads.

These are the spatial members of the OUF-Sim selector taxonomy. The
view's reader takes one of these (or a HEALPix tile list / octree node
id, added with their backends) to materialise only a sub-region —
never the whole snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cube:
    """Axis-aligned comoving bounding box (unit-frame length units)."""

    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float

    def __post_init__(self) -> None:
        for lo, hi, ax in (
            (self.xlo, self.xhi, "x"),
            (self.ylo, self.yhi, "y"),
            (self.zlo, self.zhi, "z"),
        ):
            if lo > hi:
                raise ValueError(
                    f"Cube: {ax}lo ({lo}) must be <= {ax}hi ({hi})"
                )


@dataclass(frozen=True)
class Cone:
    """Angular cone: centre (lon, lat) in degrees + radius in degrees."""

    lon: float
    lat: float
    radius_deg: float

    def __post_init__(self) -> None:
        if self.radius_deg <= 0.0:
            raise ValueError(
                f"Cone.radius_deg must be > 0, got {self.radius_deg!r}"
            )


@dataclass(frozen=True)
class SkyPatch:
    """Angular rectangle in degrees (lon/lat min/max)."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def __post_init__(self) -> None:
        if self.lon_min > self.lon_max:
            raise ValueError(
                f"SkyPatch: lon_min ({self.lon_min}) must be <= "
                f"lon_max ({self.lon_max})"
            )
        if self.lat_min > self.lat_max:
            raise ValueError(
                f"SkyPatch: lat_min ({self.lat_min}) must be <= "
                f"lat_max ({self.lat_max})"
            )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_sim_selectors.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/sim/selectors.py test/test_sim_selectors.py
git commit -m "phaseS2/T4: Cube / Cone / SkyPatch spatial selectors (validated, frozen)"
```

---

## Task 5: Sidecar specs — `CosmologySpec`, `UnitFrameSpec`, `ProvenanceSpec`

**Files:**
- Create: `oneuniverse/sim/cosmology.py`, `oneuniverse/sim/unit_frame.py`, `oneuniverse/sim/provenance.py`
- Test: `test/test_sim_sidecar_specs.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_sim_sidecar_specs.py
"""Phase S2 T5 — cosmology / unit-frame / provenance sidecar specs."""
import pytest

from oneuniverse.sim.cosmology import CosmologySpec
from oneuniverse.sim.provenance import ProvenanceSpec
from oneuniverse.sim.unit_frame import UnitFrameSpec


def test_cosmology_roundtrip():
    spec = CosmologySpec(
        omega_m=0.3089, omega_b=0.0486, h=0.6774, n_s=0.9667,
        sigma8=0.8159, w0=-1.0, wa=0.0, t_cmb=2.7255,
    )
    assert CosmologySpec.from_dict(spec.to_dict()) == spec


def test_cosmology_all_optional():
    spec = CosmologySpec()
    assert spec.omega_m is None
    assert CosmologySpec.from_dict(spec.to_dict()) == spec


def test_unit_frame_defaults_and_roundtrip():
    spec = UnitFrameSpec(
        length_unit="Mpc/h", mass_unit="Msun/h",
        velocity_unit="km/s peculiar",
    )
    assert spec.time_unit == "Gyr"
    assert spec.h_factor is True
    assert spec.comoving is True
    assert spec.frame == "icrs"
    assert spec.endianness == "native"
    assert UnitFrameSpec.from_dict(spec.to_dict()) == spec


def test_unit_frame_rejects_unknown_velocity():
    with pytest.raises(ValueError, match="velocity_unit"):
        UnitFrameSpec(
            length_unit="Mpc/h", mass_unit="Msun/h",
            velocity_unit="furlongs/fortnight",
        )


def test_provenance_roundtrip():
    spec = ProvenanceSpec(
        code="ABACUS", code_version="2.0", git_hash="deadbeef",
        original_paths=("/data/abacus/slab0",),
        ingested_utc="2026-06-01T00:00:00+00:00",
        converter="AbacusSummitOutputConverter",
    )
    assert ProvenanceSpec.from_dict(spec.to_dict()) == spec
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_sim_sidecar_specs.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `cosmology.py`**

```python
# oneuniverse/sim/cosmology.py
"""Sim-side cosmology declaration.

Duplicated (not imported) from Pillar 1 by design — Pillar 3 must not
depend on ``oneuniverse.data``. This records the cosmology a simulation
was *run with*; it is not a cosmology engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CosmologySpec:
    omega_m: Optional[float] = None
    omega_b: Optional[float] = None
    h: Optional[float] = None
    n_s: Optional[float] = None
    sigma8: Optional[float] = None
    w0: Optional[float] = None
    wa: Optional[float] = None
    t_cmb: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "omega_m": self.omega_m,
            "omega_b": self.omega_b,
            "h": self.h,
            "n_s": self.n_s,
            "sigma8": self.sigma8,
            "w0": self.w0,
            "wa": self.wa,
            "t_cmb": self.t_cmb,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CosmologySpec":
        return cls(
            omega_m=d.get("omega_m"),
            omega_b=d.get("omega_b"),
            h=d.get("h"),
            n_s=d.get("n_s"),
            sigma8=d.get("sigma8"),
            w0=d.get("w0"),
            wa=d.get("wa"),
            t_cmb=d.get("t_cmb"),
            extra=dict(d.get("extra", {})),
        )
```

- [ ] **Step 4: Implement `unit_frame.py`**

```python
# oneuniverse/sim/unit_frame.py
"""Sim-side unit + frame declaration.

The single most important metadata for cross-code comparison: every
simulation declares its length / mass / velocity units, h-factor,
comoving-vs-proper, frame, and endianness. Explicit attribution wins —
Gadget vs SWIFT vs CompaSO vs FLAMINGO each have different defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

_ALLOWED_VELOCITY = frozenset({
    "km/s peculiar",     # physical peculiar velocity
    "km/s a",            # a * dx/dt
    "km/s sqrt_a",       # Gadget: v_pec / sqrt(a)
    "code",              # code units; conversion via length/time
})


@dataclass(frozen=True)
class UnitFrameSpec:
    length_unit: str                 # "Mpc/h", "kpc/h", "Mpc"
    mass_unit: str                   # "Msun/h", "Msun", "1e10 Msun/h"
    velocity_unit: str               # one of _ALLOWED_VELOCITY
    time_unit: str = "Gyr"
    h_factor: bool = True            # quantities carry /h
    comoving: bool = True            # positions comoving (vs proper)
    frame: str = "icrs"              # "icrs" | "galactic" | "ecliptic" | "box"
    endianness: str = "native"       # "native" | "little" | "big"

    def __post_init__(self) -> None:
        if self.velocity_unit not in _ALLOWED_VELOCITY:
            raise ValueError(
                f"UnitFrameSpec: unknown velocity_unit "
                f"{self.velocity_unit!r}; allowed: {sorted(_ALLOWED_VELOCITY)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length_unit": self.length_unit,
            "mass_unit": self.mass_unit,
            "velocity_unit": self.velocity_unit,
            "time_unit": self.time_unit,
            "h_factor": bool(self.h_factor),
            "comoving": bool(self.comoving),
            "frame": self.frame,
            "endianness": self.endianness,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnitFrameSpec":
        return cls(
            length_unit=d["length_unit"],
            mass_unit=d["mass_unit"],
            velocity_unit=d["velocity_unit"],
            time_unit=d.get("time_unit", "Gyr"),
            h_factor=bool(d.get("h_factor", True)),
            comoving=bool(d.get("comoving", True)),
            frame=d.get("frame", "icrs"),
            endianness=d.get("endianness", "native"),
        )
```

- [ ] **Step 5: Implement `provenance.py`**

```python
# oneuniverse/sim/provenance.py
"""Sim-side provenance declaration — run history + ingest trail."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ProvenanceSpec:
    code: str
    code_version: Optional[str]
    git_hash: Optional[str]
    original_paths: Tuple[str, ...]
    ingested_utc: str
    converter: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "code_version": self.code_version,
            "git_hash": self.git_hash,
            "original_paths": list(self.original_paths),
            "ingested_utc": self.ingested_utc,
            "converter": self.converter,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProvenanceSpec":
        return cls(
            code=d["code"],
            code_version=d.get("code_version"),
            git_hash=d.get("git_hash"),
            original_paths=tuple(d.get("original_paths", ())),
            ingested_utc=d["ingested_utc"],
            converter=d["converter"],
        )
```

- [ ] **Step 6: Run test to verify it passes**

```bash
pytest test/test_sim_sidecar_specs.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add oneuniverse/sim/cosmology.py oneuniverse/sim/unit_frame.py \
        oneuniverse/sim/provenance.py test/test_sim_sidecar_specs.py
git commit -m "phaseS2/T5: CosmologySpec + UnitFrameSpec + ProvenanceSpec sidecars (sim-side, no Pillar-1 import)"
```

---

## Task 6: `ProductDecl`

**Files:**
- Create: `oneuniverse/sim/product.py`
- Test: `test/test_sim_product.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_sim_product.py
"""Phase S2 T6 — ProductDecl."""
import pytest

from oneuniverse.sim.product import ProductDecl


def test_ok():
    p = ProductDecl(
        product="snapshots", native_format="ASDF/pack9",
        indexes=("healpix_tiles", "halo_particle_ptr"),
        fields=("Coordinates", "Velocities"),
    )
    assert p.product == "snapshots"


def test_rejects_unknown_product():
    with pytest.raises(ValueError, match="product"):
        ProductDecl(
            product="not_a_product", native_format="x",
            indexes=(), fields=(),
        )


def test_roundtrip():
    p = ProductDecl(
        product="lightcone", native_format="FITS HEALPix",
        indexes=("lightcone_shell",), fields=("kappa", "gamma1", "gamma2"),
    )
    assert ProductDecl.from_dict(p.to_dict()) == p
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_sim_product.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/sim/product.py
"""ProductDecl — a converter declares each product it found + which
Layer-1 indexers to run + which canonical fields it exposes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from oneuniverse.sim._version import PRODUCT_KINDS


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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_sim_product.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/sim/product.py test/test_sim_product.py
git commit -m "phaseS2/T6: ProductDecl (validated against PRODUCT_KINDS)"
```

---

## Task 7: `OUFSimManifest` + YAML read/write

**Files:**
- Create: `oneuniverse/sim/manifest.py`
- Test: `test/test_sim_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_sim_manifest.py
"""Phase S2 T7 — OUFSimManifest + YAML round-trip."""
import pytest

from oneuniverse.sim._version import OUFSIM_FORMAT_VERSION
from oneuniverse.sim.cosmology import CosmologySpec
from oneuniverse.sim.manifest import (
    OUFSimManifest,
    OUFSimManifestError,
    read_manifest,
    write_manifest,
)
from oneuniverse.sim.product import ProductDecl
from oneuniverse.sim.provenance import ProvenanceSpec
from oneuniverse.sim.unit_frame import UnitFrameSpec


def _minimal(**overrides) -> OUFSimManifest:
    defaults = dict(
        oufsim_format_version=OUFSIM_FORMAT_VERSION,
        sim_name="AbacusSummit_base_c000_ph000",
        sim_kind="nbody",
        code="ABACUS",
        code_version="2.0",
        layout_schema="per_cosmology_phase_snapshot",
        backends=("ASDF/pack9", "CompaSO ASDF"),
        has_input=False,
        has_output=True,
        products=("snapshots", "halos"),
        n_snapshots=12,
        redshifts=(0.1, 0.2, 0.5),
        box_size=2000.0,
        n_particles=6912 ** 3,
        cosmology=CosmologySpec(omega_m=0.3137, sigma8=0.8076, h=0.6736),
        unit_frame=UnitFrameSpec(
            length_unit="Mpc/h", mass_unit="Msun/h",
            velocity_unit="km/s peculiar",
        ),
        provenance=ProvenanceSpec(
            code="ABACUS", code_version="2.0", git_hash=None,
            original_paths=("/cfs/abacus/base_c000_ph000",),
            ingested_utc="2026-06-01T00:00:00+00:00",
            converter="AbacusSummitOutputConverter",
        ),
        product_decls=(
            ProductDecl(
                product="snapshots", native_format="ASDF/pack9",
                indexes=("healpix_tiles",), fields=("Coordinates",),
            ),
        ),
    )
    defaults.update(overrides)
    return OUFSimManifest(**defaults)


def test_version_constant():
    assert OUFSIM_FORMAT_VERSION == "0.1.0"


def test_rejects_unknown_sim_kind():
    with pytest.raises(ValueError, match="sim_kind"):
        _minimal(sim_kind="quantum_foam")


def test_rejects_unknown_product():
    with pytest.raises(ValueError, match="products"):
        _minimal(products=("snapshots", "bogus"))


def test_rejects_unknown_layout_schema():
    with pytest.raises(ValueError, match="layout_schema"):
        _minimal(layout_schema="spaghetti")


def test_yaml_roundtrip(tmp_path):
    m = _minimal()
    path = tmp_path / "manifest.yaml"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read == m


def test_read_rejects_incompatible_major(tmp_path):
    import yaml
    payload = {
        "oufsim_format_version": "9.9.9",
        "sim_name": "x", "sim_kind": "nbody", "code": "X",
        "code_version": None, "layout_schema": "per_cosmology_phase_snapshot",
        "backends": [], "has_input": False, "has_output": True,
        "products": [], "n_snapshots": 0, "redshifts": [],
        "box_size": None, "n_particles": None,
        "cosmology": None, "unit_frame": None, "provenance": None,
        "product_decls": [],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(OUFSimManifestError, match="version"):
        read_manifest(path)


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(OUFSimManifestError, match="not found"):
        read_manifest(tmp_path / "nope.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_sim_manifest.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/sim/manifest.py
"""OUFSimManifest — the typed contract for one OUF-Sim record.

Mirrors the OUF data ``Manifest`` discipline: pinned format version,
typed sub-specs, ``to_dict`` / ``from_dict``, YAML read/write with a
hard version-compat check. The manifest points at native files +
sidecar indexes; it never holds bulk particle data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

from oneuniverse.sim._version import (
    LAYOUT_SCHEMAS,
    OUFSIM_FORMAT_VERSION,
    PRODUCT_KINDS,
    SIM_KINDS,
)
from oneuniverse.sim.cosmology import CosmologySpec
from oneuniverse.sim.product import ProductDecl
from oneuniverse.sim.provenance import ProvenanceSpec
from oneuniverse.sim.unit_frame import UnitFrameSpec


class OUFSimManifestError(ValueError):
    """Raised on a malformed or version-incompatible OUF-Sim manifest."""


@dataclass(frozen=True)
class OUFSimManifest:
    oufsim_format_version: str
    sim_name: str
    sim_kind: str
    code: str
    code_version: Optional[str]
    layout_schema: str
    backends: Tuple[str, ...]
    has_input: bool
    has_output: bool
    products: Tuple[str, ...]
    n_snapshots: int
    redshifts: Tuple[float, ...]
    box_size: Optional[float]
    n_particles: Optional[int]
    cosmology: Optional[CosmologySpec]
    unit_frame: Optional[UnitFrameSpec]
    provenance: Optional[ProvenanceSpec]
    product_decls: Tuple[ProductDecl, ...] = ()

    def __post_init__(self) -> None:
        if self.sim_kind not in SIM_KINDS:
            raise ValueError(
                f"OUFSimManifest: unknown sim_kind {self.sim_kind!r}; "
                f"allowed: {list(SIM_KINDS)}"
            )
        if self.layout_schema not in LAYOUT_SCHEMAS:
            raise ValueError(
                f"OUFSimManifest: unknown layout_schema "
                f"{self.layout_schema!r}; allowed: {list(LAYOUT_SCHEMAS)}"
            )
        bad = [p for p in self.products if p not in PRODUCT_KINDS]
        if bad:
            raise ValueError(
                f"OUFSimManifest: unknown products {bad!r}; "
                f"allowed: {list(PRODUCT_KINDS)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "oufsim_format_version": self.oufsim_format_version,
            "sim_name": self.sim_name,
            "sim_kind": self.sim_kind,
            "code": self.code,
            "code_version": self.code_version,
            "layout_schema": self.layout_schema,
            "backends": list(self.backends),
            "has_input": bool(self.has_input),
            "has_output": bool(self.has_output),
            "products": list(self.products),
            "n_snapshots": int(self.n_snapshots),
            "redshifts": [float(z) for z in self.redshifts],
            "box_size": self.box_size,
            "n_particles": self.n_particles,
            "cosmology": (
                self.cosmology.to_dict() if self.cosmology is not None else None
            ),
            "unit_frame": (
                self.unit_frame.to_dict()
                if self.unit_frame is not None else None
            ),
            "provenance": (
                self.provenance.to_dict()
                if self.provenance is not None else None
            ),
            "product_decls": [p.to_dict() for p in self.product_decls],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OUFSimManifest":
        cosmo = d.get("cosmology")
        uf = d.get("unit_frame")
        prov = d.get("provenance")
        return cls(
            oufsim_format_version=d["oufsim_format_version"],
            sim_name=d["sim_name"],
            sim_kind=d["sim_kind"],
            code=d["code"],
            code_version=d.get("code_version"),
            layout_schema=d["layout_schema"],
            backends=tuple(d.get("backends", ())),
            has_input=bool(d.get("has_input", False)),
            has_output=bool(d.get("has_output", False)),
            products=tuple(d.get("products", ())),
            n_snapshots=int(d.get("n_snapshots", 0)),
            redshifts=tuple(float(z) for z in d.get("redshifts", ())),
            box_size=d.get("box_size"),
            n_particles=d.get("n_particles"),
            cosmology=CosmologySpec.from_dict(cosmo) if cosmo else None,
            unit_frame=UnitFrameSpec.from_dict(uf) if uf else None,
            provenance=ProvenanceSpec.from_dict(prov) if prov else None,
            product_decls=tuple(
                ProductDecl.from_dict(p) for p in d.get("product_decls", ())
            ),
        )


def write_manifest(path: Union[str, Path], manifest: OUFSimManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))


def read_manifest(path: Union[str, Path]) -> OUFSimManifest:
    path = Path(path)
    if not path.is_file():
        raise OUFSimManifestError(f"OUF-Sim manifest not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise OUFSimManifestError(f"{path}: invalid YAML ({e})") from e
    if not isinstance(raw, dict):
        raise OUFSimManifestError(f"{path}: top-level must be a mapping")
    fmt = raw.get("oufsim_format_version")
    if not (isinstance(fmt, str) and fmt.startswith("0.1")):
        raise OUFSimManifestError(
            f"{path}: oufsim_format_version={fmt!r} is not compatible "
            f"with this library (expected 0.1.x)."
        )
    try:
        return OUFSimManifest.from_dict(raw)
    except (KeyError, ValueError) as e:
        raise OUFSimManifestError(f"{path}: {e}") from e
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_sim_manifest.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/sim/manifest.py test/test_sim_manifest.py
git commit -m "phaseS2/T7: OUFSimManifest + YAML read/write + version-compat (0.1.x); validates sim_kind/products/layout_schema"
```

---

## Task 8: `RegionSpec`

**Files:**
- Create: `oneuniverse/sim/region.py`
- Test: `test/test_sim_region.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_sim_region.py
"""Phase S2 T8 — RegionSpec."""
import pytest

from oneuniverse.sim.region import RegionSpec
from oneuniverse.sim.selectors import Cone


def test_eulerian_bbox_ok():
    r = RegionSpec(
        region_id="coma", kind="cluster",
        eulerian_bbox=(100.0, 110.0, 100.0, 110.0, 100.0, 110.0),
    )
    assert r.eulerian_bbox[1] == 110.0


def test_cone_region_ok():
    r = RegionSpec(
        region_id="patch1", kind="observed",
        cone=Cone(lon=120.0, lat=0.0, radius_deg=2.0),
        refs=("/data/oneuniverse/clusters/redmapper.parquet",),
    )
    assert r.cone.radius_deg == 2.0
    assert r.refs[0].endswith("redmapper.parquet")


def test_requires_at_least_one_geometry():
    with pytest.raises(ValueError, match="geometry"):
        RegionSpec(region_id="x", kind="void")


def test_roundtrip_with_cone():
    r = RegionSpec(
        region_id="patch1", kind="observed",
        cone=Cone(lon=120.0, lat=0.0, radius_deg=2.0),
        z=0.3, mass=1e14, refs=("/a.parquet",),
    )
    assert RegionSpec.from_dict(r.to_dict()) == r


def test_roundtrip_with_bbox_and_lagrangian():
    r = RegionSpec(
        region_id="zoom1", kind="lagrangian",
        eulerian_bbox=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        lagrangian_patch=(0.0, 0.5, 0.0, 0.5, 0.0, 0.5),
    )
    assert RegionSpec.from_dict(r.to_dict()) == r
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_sim_region.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/sim/region.py
"""RegionSpec — a region of interest in the region catalog.

Carries both an Eulerian geometry (bbox / cone — for observed-structure
pinning) and an optional Lagrangian patch (for zoom-IC re-simulation).
``refs`` are file paths to Pillar-1 artefacts (cluster / void / PV
reconstructions) — paths, NOT Python imports (Rule 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from oneuniverse.sim.selectors import Cone

# (xlo, xhi, ylo, yhi, zlo, zhi)
Bbox6 = Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    kind: str                                  # cluster|void|filament|observed|lagrangian
    eulerian_bbox: Optional[Bbox6] = None
    lagrangian_patch: Optional[Bbox6] = None
    cone: Optional[Cone] = None
    z: Optional[float] = None
    mass: Optional[float] = None
    refs: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if (
            self.eulerian_bbox is None
            and self.lagrangian_patch is None
            and self.cone is None
        ):
            raise ValueError(
                "RegionSpec: at least one geometry "
                "(eulerian_bbox / lagrangian_patch / cone) is required"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "kind": self.kind,
            "eulerian_bbox": (
                list(self.eulerian_bbox)
                if self.eulerian_bbox is not None else None
            ),
            "lagrangian_patch": (
                list(self.lagrangian_patch)
                if self.lagrangian_patch is not None else None
            ),
            "cone": (
                {"lon": self.cone.lon, "lat": self.cone.lat,
                 "radius_deg": self.cone.radius_deg}
                if self.cone is not None else None
            ),
            "z": self.z,
            "mass": self.mass,
            "refs": list(self.refs),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegionSpec":
        cone_raw = d.get("cone")
        bbox = d.get("eulerian_bbox")
        lag = d.get("lagrangian_patch")
        return cls(
            region_id=d["region_id"],
            kind=d["kind"],
            eulerian_bbox=tuple(bbox) if bbox is not None else None,
            lagrangian_patch=tuple(lag) if lag is not None else None,
            cone=(
                Cone(lon=cone_raw["lon"], lat=cone_raw["lat"],
                     radius_deg=cone_raw["radius_deg"])
                if cone_raw is not None else None
            ),
            z=d.get("z"),
            mass=d.get("mass"),
            refs=tuple(d.get("refs", ())),
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_sim_region.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/sim/region.py test/test_sim_region.py
git commit -m "phaseS2/T8: RegionSpec (Eulerian bbox + cone + Lagrangian patch; refs are file paths not imports)"
```

---

## Task 9: `SimulationRequest`

**Files:**
- Create: `oneuniverse/sim/request.py`
- Test: `test/test_sim_request.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_sim_request.py
"""Phase S2 T9 — SimulationRequest."""
import pytest

from oneuniverse.sim.cosmology import CosmologySpec
from oneuniverse.sim.region import RegionSpec
from oneuniverse.sim.request import SimulationRequest


def _region() -> RegionSpec:
    return RegionSpec(
        region_id="coma", kind="cluster",
        eulerian_bbox=(100.0, 110.0, 100.0, 110.0, 100.0, 110.0),
    )


def test_ok():
    req = SimulationRequest(
        request_id="req-001", parent_sim="AbacusSummit_base_c000_ph000",
        region=_region(), target_resolution=1e7,
        physics=("dm", "hydro"),
        cosmology=CosmologySpec(omega_m=0.31, sigma8=0.81, h=0.67),
        ic_strategy="zoom_from_parent_ic", code_hint="AREPO",
    )
    assert req.status == "pending"


def test_rejects_unknown_ic_strategy():
    with pytest.raises(ValueError, match="ic_strategy"):
        SimulationRequest(
            request_id="x", parent_sim=None, region=_region(),
            target_resolution=1.0, physics=("dm",),
            cosmology=CosmologySpec(), ic_strategy="teleport",
        )


def test_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        SimulationRequest(
            request_id="x", parent_sim=None, region=_region(),
            target_resolution=1.0, physics=("dm",),
            cosmology=CosmologySpec(), ic_strategy="fresh",
            status="exploded",
        )


def test_rejects_unknown_physics():
    with pytest.raises(ValueError, match="physics"):
        SimulationRequest(
            request_id="x", parent_sim=None, region=_region(),
            target_resolution=1.0, physics=("dm", "magic"),
            cosmology=CosmologySpec(), ic_strategy="fresh",
        )


def test_roundtrip():
    req = SimulationRequest(
        request_id="req-001", parent_sim="parent",
        region=_region(), target_resolution=1e7,
        physics=("dm", "hydro", "mhd"),
        cosmology=CosmologySpec(omega_m=0.31),
        ic_strategy="constrained_from_posterior", code_hint=None,
        status="dispatched", provenance={"submitted_by": "tester"},
    )
    assert SimulationRequest.from_dict(req.to_dict()) == req
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_sim_request.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/sim/request.py
"""SimulationRequest — the orchestration output artefact.

Region selection emits a SimulationRequest describing what to
(re-)simulate. Pillar 3 stores it + tracks its lifecycle; it never
runs the simulation (Rule 4). The external runner updates ``status``
out-of-band and re-ingests output, closing the lineage loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from oneuniverse.sim.cosmology import CosmologySpec
from oneuniverse.sim.region import RegionSpec

_IC_STRATEGIES = frozenset({
    "zoom_from_parent_ic",
    "constrained_from_posterior",
    "fresh",
})
_STATUSES = frozenset({"pending", "dispatched", "running", "ingested"})
_PHYSICS = frozenset({"dm", "hydro", "mhd", "rt", "cr"})


@dataclass(frozen=True)
class SimulationRequest:
    request_id: str
    parent_sim: Optional[str]
    region: RegionSpec
    target_resolution: float           # mass or spatial resolution
    physics: Tuple[str, ...]
    cosmology: CosmologySpec
    ic_strategy: str
    code_hint: Optional[str] = None
    status: str = "pending"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ic_strategy not in _IC_STRATEGIES:
            raise ValueError(
                f"SimulationRequest: unknown ic_strategy "
                f"{self.ic_strategy!r}; allowed: {sorted(_IC_STRATEGIES)}"
            )
        if self.status not in _STATUSES:
            raise ValueError(
                f"SimulationRequest: unknown status {self.status!r}; "
                f"allowed: {sorted(_STATUSES)}"
            )
        bad = [p for p in self.physics if p not in _PHYSICS]
        if bad:
            raise ValueError(
                f"SimulationRequest: unknown physics {bad!r}; "
                f"allowed: {sorted(_PHYSICS)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "parent_sim": self.parent_sim,
            "region": self.region.to_dict(),
            "target_resolution": float(self.target_resolution),
            "physics": list(self.physics),
            "cosmology": self.cosmology.to_dict(),
            "ic_strategy": self.ic_strategy,
            "code_hint": self.code_hint,
            "status": self.status,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationRequest":
        return cls(
            request_id=d["request_id"],
            parent_sim=d.get("parent_sim"),
            region=RegionSpec.from_dict(d["region"]),
            target_resolution=float(d["target_resolution"]),
            physics=tuple(d.get("physics", ())),
            cosmology=CosmologySpec.from_dict(d["cosmology"]),
            ic_strategy=d["ic_strategy"],
            code_hint=d.get("code_hint"),
            status=d.get("status", "pending"),
            provenance=dict(d.get("provenance", {})),
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_sim_request.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/sim/request.py test/test_sim_request.py
git commit -m "phaseS2/T9: SimulationRequest (validated ic_strategy/status/physics; lifecycle status field)"
```

---

## Task 10: `SimConverter` ABC + registry

**Files:**
- Create: `oneuniverse/sim/converter.py`
- Test: `test/test_sim_converter_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_sim_converter_registry.py
"""Phase S2 T10 — SimConverter ABC + registry."""
from pathlib import Path

import pytest

from oneuniverse.sim.capabilities import BackendCapabilities
from oneuniverse.sim.cosmology import CosmologySpec
from oneuniverse.sim.converter import (
    SimConverter,
    detect_converter,
    get_converter,
    register,
    registered_codes,
)
from oneuniverse.sim.product import ProductDecl
from oneuniverse.sim.unit_frame import UnitFrameSpec


@register
class _DummyConverter(SimConverter):
    code = "DUMMY"
    sim_kind = "nbody"
    capabilities = BackendCapabilities(name="dummy", native_format="dummy-fmt")

    def detect(self, path: Path) -> bool:
        return Path(path).name == "dummy_sim"

    def declare_products(self, src: Path):
        return (
            ProductDecl(
                product="snapshots", native_format="dummy-fmt",
                indexes=(), fields=("Coordinates",),
            ),
        )

    def read_cosmology(self, src: Path) -> CosmologySpec:
        return CosmologySpec(omega_m=0.3)

    def read_unit_frame(self, src: Path) -> UnitFrameSpec:
        return UnitFrameSpec(
            length_unit="Mpc/h", mass_unit="Msun/h",
            velocity_unit="km/s peculiar",
        )


def test_registered():
    assert "DUMMY" in registered_codes()
    assert get_converter("DUMMY") is _DummyConverter


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="UNKNOWN"):
        get_converter("UNKNOWN")


def test_register_rejects_duplicate():
    with pytest.raises(ValueError, match="already"):
        register(_DummyConverter)


def test_register_rejects_missing_code():
    with pytest.raises(ValueError, match="code"):
        @register
        class _NoCode(SimConverter):  # noqa: N801
            sim_kind = "nbody"
            capabilities = BackendCapabilities(name="n", native_format="f")

            def detect(self, path): return False
            def declare_products(self, src): return ()
            def read_cosmology(self, src): return CosmologySpec()
            def read_unit_frame(self, src):
                return UnitFrameSpec(
                    length_unit="Mpc/h", mass_unit="Msun/h",
                    velocity_unit="km/s peculiar",
                )


def test_detect_converter(tmp_path):
    target = tmp_path / "dummy_sim"
    target.mkdir()
    assert detect_converter(target) is _DummyConverter
    other = tmp_path / "other_sim"
    other.mkdir()
    assert detect_converter(other) is None


def test_convert_not_implemented_in_s2(tmp_path):
    """convert() lands in Phase S3 — S2 ABC raises NotImplementedError."""
    conv = _DummyConverter()
    with pytest.raises(NotImplementedError, match="S3"):
        conv.convert(tmp_path, tmp_path / "out")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_sim_converter_registry.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/sim/converter.py
"""SimConverter ABC + registry — the Layer-3 extensibility surface.

A new simulation code is added by subclassing SimConverter and
implementing four small methods (detect, declare_products,
read_cosmology, read_unit_frame), then @register. The concrete
``convert()`` orchestration (wrap native files + run Layer-1 index
builders + emit manifest) lands in Phase S3; here it raises
NotImplementedError so the contract is testable now.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple, Type

from oneuniverse.sim.capabilities import BackendCapabilities
from oneuniverse.sim.cosmology import CosmologySpec
from oneuniverse.sim.product import ProductDecl
from oneuniverse.sim.unit_frame import UnitFrameSpec


class SimConverter(abc.ABC):
    """Per-code converter (Layer 3). Subclasses set ``code`` /
    ``sim_kind`` / ``capabilities`` and implement four methods."""

    code: ClassVar[str]
    sim_kind: ClassVar[str]
    capabilities: ClassVar[BackendCapabilities]

    @abc.abstractmethod
    def detect(self, path: Path) -> bool:
        """Return True if this converter handles the dataset at ``path``."""

    @abc.abstractmethod
    def declare_products(self, src: Path) -> Tuple[ProductDecl, ...]:
        """List products found at ``src`` + which Layer-1 indexers each needs."""

    @abc.abstractmethod
    def read_cosmology(self, src: Path) -> CosmologySpec:
        """Parse the run cosmology from ``src``."""

    @abc.abstractmethod
    def read_unit_frame(self, src: Path) -> UnitFrameSpec:
        """Parse the unit/frame declaration from ``src``."""

    def convert(self, src: Path, out: Path, *, projection: str = "native",
                build_indexes: bool = True):
        """Wrap native files + build indexes + emit manifest.

        Concrete implementation lands in Phase S3 (needs the Layer-1
        IndexBuilder toolkit + ManifestWriter). Until then this raises.
        """
        raise NotImplementedError(
            "SimConverter.convert is implemented in Phase S3 "
            "(needs the Layer-1 IndexBuilder toolkit)."
        )


_REGISTRY: Dict[str, Type[SimConverter]] = {}


def register(cls: Type[SimConverter]) -> Type[SimConverter]:
    """Class decorator: register a converter by its ``code``."""
    code = getattr(cls, "code", None)
    if not code:
        raise ValueError(
            f"register: {cls.__name__} must set a non-empty class "
            f"attribute `code`"
        )
    if code in _REGISTRY:
        raise ValueError(
            f"register: code {code!r} is already registered "
            f"(by {_REGISTRY[code].__name__})"
        )
    _REGISTRY[code] = cls
    return cls


def get_converter(code: str) -> Type[SimConverter]:
    if code not in _REGISTRY:
        raise KeyError(
            f"no converter registered for code {code!r}; "
            f"known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[code]


def detect_converter(path: Path) -> Optional[Type[SimConverter]]:
    """Return the first registered converter whose ``detect`` matches."""
    for cls in _REGISTRY.values():
        if cls().detect(Path(path)):
            return cls
    return None


def registered_codes() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_sim_converter_registry.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/sim/converter.py test/test_sim_converter_registry.py
git commit -m "phaseS2/T10: SimConverter ABC (4-method contract) + @register registry; convert() stubbed for S3"
```

---

## Task 11: Lint guard — no Pillar-1 imports

**Files:**
- Create: `test/test_sim_no_pillar1_imports.py`

- [ ] **Step 1: Write the test (this is the deliverable — no implementation needed)**

```python
# test/test_sim_no_pillar1_imports.py
"""Phase S2 T11 — Pillar-3 isolation guard.

oneuniverse.sim must NOT import from oneuniverse.data or
oneuniverse.combine (Rule 1: minimal cross-pillar coupling). This test
scans every source file under oneuniverse/sim/ via the AST and fails
if a forbidden import appears.
"""
import ast
from pathlib import Path

import oneuniverse.sim as sim_pkg

_FORBIDDEN_ROOTS = ("oneuniverse.data", "oneuniverse.combine")


def _sim_source_files():
    root = Path(sim_pkg.__file__).parent
    return sorted(root.rglob("*.py"))


def _forbidden_imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == r or alias.name.startswith(r + ".")
                       for r in _FORBIDDEN_ROOTS):
                    bad.append((path.name, alias.name))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod == r or mod.startswith(r + ".")
                   for r in _FORBIDDEN_ROOTS):
                bad.append((path.name, mod))
    return bad


def test_no_pillar1_imports_anywhere_in_sim():
    offenders = []
    files = _sim_source_files()
    assert files, "no source files found under oneuniverse/sim/"
    for path in files:
        offenders.extend(_forbidden_imports(path))
    assert offenders == [], (
        "oneuniverse.sim must not import oneuniverse.data / "
        f"oneuniverse.combine; offenders: {offenders}"
    )
```

- [ ] **Step 2: Run test to verify it passes (guard is green from the start)**

```bash
pytest test/test_sim_no_pillar1_imports.py -v
```

Expected: 1 passed (no forbidden imports exist yet — the guard protects future work).

- [ ] **Step 3: Sanity-check the guard actually catches a violation**

Temporarily add `import oneuniverse.data` to `oneuniverse/sim/_version.py`, run the test, confirm it FAILS, then remove the line:

```bash
printf '\nimport oneuniverse.data  # TEMP\n' >> oneuniverse/sim/_version.py
pytest test/test_sim_no_pillar1_imports.py -q 2>&1 | tail -3   # expect FAIL
git checkout oneuniverse/sim/_version.py                       # revert
pytest test/test_sim_no_pillar1_imports.py -q 2>&1 | tail -3   # expect PASS
```

- [ ] **Step 4: Commit**

```bash
git add test/test_sim_no_pillar1_imports.py
git commit -m "phaseS2/T11: lint guard — fail if oneuniverse.sim imports oneuniverse.data/combine (Rule 1)"
```

---

## Task 12: Public exports + close-out

**Files:**
- Modify: `oneuniverse/sim/__init__.py`
- Modify: `plans/README.md`, `oneuniverse/CLAUDE.md`

- [ ] **Step 1: Write the failing test for the public surface**

```python
# test/test_sim_public_api.py
"""Phase S2 T12 — public API surface."""
import oneuniverse.sim as sim


def test_public_exports_present():
    for name in (
        "OUFSIM_FORMAT_VERSION",
        "ExecutionMode", "ExecutionPlan",
        "BackendCapabilities",
        "Cube", "Cone", "SkyPatch",
        "CosmologySpec", "UnitFrameSpec", "ProvenanceSpec",
        "ProductDecl",
        "OUFSimManifest", "read_manifest", "write_manifest",
        "OUFSimManifestError",
        "RegionSpec", "SimulationRequest",
        "SimConverter", "register", "get_converter",
        "detect_converter", "registered_codes",
    ):
        assert hasattr(sim, name), f"missing public export: {name}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_sim_public_api.py -v
```

Expected: FAIL on the first missing export.

- [ ] **Step 3: Fill in `__init__.py`**

Replace `oneuniverse/sim/__init__.py` with:

```python
"""oneuniverse.sim — OUF-Sim: storage + orchestration of cosmological
simulations (Pillar 3, digital-twin substrate).

Standalone subpackage. **Must not import** from ``oneuniverse.data`` or
``oneuniverse.combine`` (enforced by test_sim_no_pillar1_imports).
"""
from oneuniverse.sim._version import (
    LAYOUT_SCHEMAS,
    OUFSIM_FORMAT_VERSION,
    PRODUCT_KINDS,
    SIM_KINDS,
)
from oneuniverse.sim.capabilities import BackendCapabilities
from oneuniverse.sim.converter import (
    SimConverter,
    detect_converter,
    get_converter,
    register,
    registered_codes,
)
from oneuniverse.sim.cosmology import CosmologySpec
from oneuniverse.sim.execution import ExecutionMode, ExecutionPlan
from oneuniverse.sim.manifest import (
    OUFSimManifest,
    OUFSimManifestError,
    read_manifest,
    write_manifest,
)
from oneuniverse.sim.product import ProductDecl
from oneuniverse.sim.provenance import ProvenanceSpec
from oneuniverse.sim.region import RegionSpec
from oneuniverse.sim.request import SimulationRequest
from oneuniverse.sim.selectors import Cone, Cube, SkyPatch
from oneuniverse.sim.unit_frame import UnitFrameSpec

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
```

- [ ] **Step 4: Run the public-API test + the isolation guard together**

```bash
pytest test/test_sim_public_api.py test/test_sim_no_pillar1_imports.py -v
```

Expected: both pass (the new `__init__` imports only `oneuniverse.sim.*`, never Pillar 1).

- [ ] **Step 5: Run the full sim test set**

```bash
pytest test/test_sim_*.py -q
```

Expected: all green (~40 tests across the 11 modules).

- [ ] **Step 6: Run the whole suite (no regressions in Pillar 1)**

```bash
pytest -q 2>&1 | tail -3
```

Expected: `>= 562 passed` (522 baseline + ~40 new), `2 skipped`.

- [ ] **Step 7: Update docs**

In `oneuniverse/CLAUDE.md`, under the Package layout section, add:

```
- `oneuniverse/sim/` — **Pillar 3 (OUF-Sim)**, standalone. Types for
  the simulation storage + orchestration substrate: `OUFSimManifest`,
  `ExecutionPlan`/`BackendCapabilities` (optimisation substrate),
  `SimConverter` ABC + registry, `RegionSpec`, `SimulationRequest`.
  **Zero imports** from `oneuniverse.data` / `combine` (guarded by
  `test_sim_no_pillar1_imports.py`). Backends + partial-access reads
  land in Phase S3+.
```

In `plans/README.md`, update the Pillar-3 phase table row:

```
| S2 | `oneuniverse.sim` skeleton + types + no-Pillar-1-import lint guard | **complete (2026-06-01, NNN/NNN tests green)** |
```

- [ ] **Step 8: Commit + memory**

```bash
git add oneuniverse/sim/__init__.py test/test_sim_public_api.py \
        oneuniverse/CLAUDE.md plans/README.md
git commit -m "phaseS2/T12: public API exports + docs; oneuniverse.sim skeleton complete"
```

Append to `/home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`:

```markdown
## Phase S2 — oneuniverse.sim skeleton (complete 2026-06-01)

- New standalone subpackage `oneuniverse/sim/` (Pillar 3, OUF-Sim).
- Types: `ExecutionMode`/`ExecutionPlan`, `BackendCapabilities`
  (+`heavy_step_modes`), `Cube`/`Cone`/`SkyPatch`, `CosmologySpec`/
  `UnitFrameSpec`/`ProvenanceSpec`, `ProductDecl`, `OUFSimManifest`
  (YAML, `oufsim_format_version=0.1.0`), `RegionSpec`,
  `SimulationRequest`, `SimConverter` ABC + `@register` registry.
- `convert()` stubbed → Phase S3 (needs Layer-1 IndexBuilder).
- Lint guard `test_sim_no_pillar1_imports.py`: zero imports from
  `oneuniverse.data`/`combine` (Rule 1).
- New dep: pyyaml. No backend, no real-data ingest.
- Tests: NNN/NNN green.
- Per-phase plan: `plans/2026-06-01-phaseS2-oufsim-skeleton.md`.
```

---

## Self-review checklist

- [ ] Every type has `to_dict`/`from_dict` round-trip tested except the
      pure-selector + execution dataclasses (validated, frozen).
- [ ] `oufsim_format_version` pinned to `0.1.0`; reader rejects other
      majors.
- [ ] `SimConverter.convert` raises `NotImplementedError("...S3...")`
      — no premature backend code.
- [ ] Lint guard catches a real violation (Task 11 Step 3 proves it).
- [ ] No `from oneuniverse.data` / `oneuniverse.combine` anywhere
      under `oneuniverse/sim/`.
- [ ] MPI communicator is NOT stored on `ExecutionPlan` (not
      serialisable) — passed at call time in later phases.
- [ ] Full suite green; Pillar-1 (522) untouched.

## Spec-coverage map (S1 architecture → S2 tasks)

| S1 type / concept | S2 task |
|---|---|
| `OUFSimManifest` + format version | T1, T7 |
| `ExecutionPlan` / `ExecutionMode` (§6.4) | T2 |
| `BackendCapabilities` + `heavy_step_modes` | T3 |
| Spatial selectors (§6.1) | T4 |
| `cosmology.yaml` / `unit_frame.yaml` / `provenance.yaml` | T5 |
| `ProductDecl` (§5.3) | T6 |
| `RegionSpec` (§7.3) | T8 |
| `SimulationRequest` (§7.3) | T9 |
| `SimConverter` ABC + registry (§5.3) | T10 |
| Rule 1 isolation (minimal coupling) | T11 |
| Public API + close-out | T12 |

Deferred to later phases (correctly absent from S2):
- `SimDatasetView`, partial-access reads, `iter_*` — Phase S3.
- Layer-1 IndexBuilder toolkit, Layer-2 NativeReaderAdapter — Phase S3.
- `SimConverter.convert()` concrete body — Phase S3.
- `SimDatabase`, lineage graph — Phase S4.
- Orchestration logic (region selection algorithm) — Phase S5.
