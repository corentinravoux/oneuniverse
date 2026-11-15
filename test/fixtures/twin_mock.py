"""Test fixture: a measure ``Product`` whose objects are placed from a counts grid."""
import numpy as np
import pyarrow as pa

from oneuniverse.measure.product import Product


def product_from_counts(counts, *, box: float, seed: int) -> Product:
    """``int(counts[i, j, k])`` objects placed uniformly inside each cell.

    Box coordinates go in ``extras['x'|'y'|'z']`` (the catalog ``z`` is the redshift
    placeholder, 0.0). The remaining core columns are placeholders the Product validators accept.
    """
    counts = np.asarray(counts)
    n = counts.shape[0]
    cell = box / n
    rng = np.random.default_rng(seed)
    idx = np.repeat(np.arange(counts.size), counts.ravel().astype(np.int64))
    i, j, k = np.unravel_index(idx, counts.shape)
    pos = (np.stack([i, j, k], axis=1) + rng.uniform(0.0, 1.0, (len(idx), 3))) * cell
    m = len(idx)
    cat = pa.table({
        "ra": np.zeros(m), "dec": np.zeros(m), "z": np.zeros(m), "weight": np.ones(m),
        "region_id": np.zeros(m, np.int32), "source_row": np.arange(m, dtype=np.int64),
        "partition": np.zeros(m, np.int32), "offset": np.arange(m, dtype=np.int64),
    })
    extras = {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]}
    return Product(cat, extras=extras, provenance={"fixture": "product_from_counts", "seed": seed})
