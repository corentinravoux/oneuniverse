"""oneuniverse.simulation.resim — resimulation orchestration (S8).

Region IC extraction + the verification gates that make selective
resimulation feasible: Gate 1 (pre-run large-scale consistency, necessary)
and later Gate 2/3 (post-run, sufficient + error budget). See the
feasibility study (research/2026-06-02-resimulation-orchestration-feasibility).

Standalone (Rule 1): no imports from oneuniverse.data / combine.
"""
from oneuniverse.simulation.resim.coupling import (
    full_target_slice,
    run_coupled,
    run_full_reference,
)
from oneuniverse.simulation.resim.farfield import far_field_box, far_field_potential
from oneuniverse.simulation.resim.ic_extract import extract_region
from oneuniverse.simulation.resim.verify import gate1_consistency, gate2_dynamical

__all__ = ["extract_region", "gate1_consistency",
           "far_field_potential", "far_field_box",
           "run_full_reference", "run_coupled", "full_target_slice",
           "gate2_dynamical"]
