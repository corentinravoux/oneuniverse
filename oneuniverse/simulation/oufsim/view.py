"""SimDatasetView — typed, ExecutionPlan-batched streaming reads.

Read analogue of the write streaming: resolve the partitions a selector
touches (via the sidecar index), then yield row batches capped at
``batch_rows`` (from an ``ExecutionPlan`` or explicit), reading one
partition at a time so the working set stays bounded. The S6 read-path
optimisation (projection, pushdown, cache, parallel, Morton) tightens what
each partition read costs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import pyarrow.parquet as pq

from oneuniverse.simulation.execution import ExecutionPlan
from oneuniverse.simulation.oufsim.index import cube_overlaps_bbox
from oneuniverse.simulation.oufsim.read import SimStore
from oneuniverse.simulation.selectors import Cube

_BYTES_PER_ROW = 6 * 8     # 6 float64 particle columns (x,y,z,vx,vy,vz)


class SimDatasetView:
    """Streaming partial-access view over an OUF-Sim store."""

    def __init__(self, store: Union[str, Path, SimStore]):
        self.store = store if isinstance(store, SimStore) else SimStore(store)

    def iter_box(self, product: str, z: float, cube: Cube, *,
                 batch_rows: int = 100_000,
                 plan: Optional[ExecutionPlan] = None) -> Iterator[dict]:
        """Yield dict-of-array batches of ``product`` rows inside ``cube``."""
        if plan is not None:
            batch_rows = (plan.batch_rows if plan.batch_rows is not None
                          else plan.batch_for(bytes_per_row=_BYTES_PER_ROW))
        zt = f"z{float(z):.3f}"
        info = self.store.layout[product][zt]
        rows = self.store._index_rows(info["index"])
        prod_dir = self.store.root / info["dir"]
        hit = [r for r in rows
               if cube_overlaps_bbox(cube, (r["xlo"], r["xhi"], r["ylo"],
                                            r["yhi"], r["zlo"], r["zhi"]))]
        buf: Optional[dict] = None
        for r in hit:
            table = pq.read_table(prod_dir / r["file"])
            cols = {name: table.column(name).to_numpy(zero_copy_only=False)
                    for name in table.column_names}
            m = ((cols["x"] >= cube.xlo) & (cols["x"] <= cube.xhi)
                 & (cols["y"] >= cube.ylo) & (cols["y"] <= cube.yhi)
                 & (cols["z"] >= cube.zlo) & (cols["z"] <= cube.zhi))
            cols = {k: v[m] for k, v in cols.items()}
            buf = cols if buf is None else {
                k: np.concatenate([buf[k], cols[k]]) for k in cols}
            while len(next(iter(buf.values()))) >= batch_rows:
                yield {k: v[:batch_rows] for k, v in buf.items()}
                buf = {k: v[batch_rows:] for k, v in buf.items()}
        if buf is not None and len(next(iter(buf.values()))) > 0:
            yield buf
