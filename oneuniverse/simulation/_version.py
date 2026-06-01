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
