"""LinearSimConverter — the first concrete SimConverter (Layer 3).

Wraps the native linear-sim layout (config.yaml + per-z field/particle/
halo files + lightcone.parquet) into an OUF-Sim store. It is the
reference converter that proves the Layer-1 index toolkit + store writer
end to end; real-format backends (AbacusSummit, Gadget, …) follow the
same four-method + ``convert`` contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import yaml

from oneuniverse.simulation.capabilities import BackendCapabilities
from oneuniverse.simulation.converter import SimConverter, register
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.execution import ExecutionMode
from oneuniverse.simulation.oufsim.write import write_oufsim_store
from oneuniverse.simulation.product import ProductDecl
from oneuniverse.simulation.unit_frame import UnitFrameSpec


@register
class LinearSimConverter(SimConverter):
    code = "oneuniverse.simulation.linear"
    sim_kind = "pm"
    capabilities = BackendCapabilities(
        name="linear",
        native_format="linear .npy/.parquet + config.yaml",
        supports_mpi=False,
        supports_gpu_direct=False,
        supports_random_access=True,   # tile / chunk index gives random access
        supports_streaming=True,
        heavy_step_modes={
            # these are the steps the optimisation work will parallelise
            "field_tiling": (ExecutionMode.SEQUENTIAL,),
            "particle_chunking": (ExecutionMode.SEQUENTIAL,),
        },
    )

    def detect(self, path: Path) -> bool:
        cfg = Path(path) / "config.yaml"
        if not cfg.is_file():
            return False
        try:
            raw = yaml.safe_load(cfg.read_text())
        except yaml.YAMLError:
            return False
        return isinstance(raw, dict) and \
            raw.get("generator") == "oneuniverse.simulation.linear"

    def declare_products(self, src: Path) -> Tuple[ProductDecl, ...]:
        decls = [
            ProductDecl("snapshots", "linear .npy", ("cartesian_chunk",),
                        ("x", "y", "z", "vx", "vy", "vz")),
            ProductDecl("fields", "linear .npy mesh", ("grid_tile",), ("delta",)),
            ProductDecl("halos", "linear parquet", ("cartesian_chunk",),
                        ("halo_id", "x", "y", "z", "delta_peak", "mass")),
        ]
        if (Path(src) / "tree.parquet").is_file():
            decls.append(ProductDecl(
                "tree", "linear parquet (edges)", ("single",),
                ("descendant_id", "progenitor_id", "z_desc", "z_prog"),
            ))
        if (Path(src) / "lightcone.parquet").is_file():
            decls.append(ProductDecl(
                "lightcone", "linear parquet (sky)", ("healpix_nest",),
                ("lon", "lat", "redshift", "comoving_radius", "mass",
                 "_healpix32"),
            ))
        return tuple(decls)

    def read_cosmology(self, src: Path) -> CosmologySpec:
        raw = yaml.safe_load((Path(src) / "config.yaml").read_text())
        return CosmologySpec.from_dict(raw["cosmology"])

    def read_unit_frame(self, src: Path) -> UnitFrameSpec:
        return UnitFrameSpec(
            length_unit="Mpc/h", mass_unit="Msun/h",
            velocity_unit="km/s peculiar", comoving=True, frame="box",
        )

    def convert(self, src: Path, out: Path, *, projection: str = "native",
                build_indexes: bool = True, sim_name: str = "linsim",
                overwrite: bool = False, **kwargs) -> Path:
        """Wrap the native linear sim at ``src`` into an OUF-Sim store."""
        return write_oufsim_store(
            src, out, sim_name=sim_name, sim_kind=self.sim_kind,
            overwrite=overwrite, **kwargs,
        )
