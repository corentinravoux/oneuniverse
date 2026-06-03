"""oneuniverse.simulation.oufsim — the OUF-Sim on-disk store.

Mirrors OUF's storage stack (JSON manifest + pyarrow parquet partitions
+ HEALPix NEST sky partitioning), specialised for simulation products:

- tabular point sets (particles, halos) -> parquet, spatially chunked by a
  coarse Cartesian grid, with a sidecar ``_index.parquet`` of per-chunk
  bounding boxes for Cube partial-access reads;
- regular grids / voxel fields -> memmap-able ``.npy`` tiles + a tile index
  (Cube partial-access without decoding the whole grid);
- sky catalogues (lightcone) -> parquet partitioned by HEALPix super-pixel
  (NEST), exactly like OUF, for Cone / SkyPatch reads.

Standalone (Rule 1): no imports from oneuniverse.data / combine.
"""
from oneuniverse.simulation.oufsim.database import SimDatabase
from oneuniverse.simulation.oufsim.read import SimStore
from oneuniverse.simulation.oufsim.view import SimDatasetView
from oneuniverse.simulation.oufsim.write import write_oufsim_store

__all__ = ["write_oufsim_store", "SimStore", "SimDatasetView", "SimDatabase"]
