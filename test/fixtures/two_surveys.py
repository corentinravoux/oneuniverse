"""Two overlapping point surveys "a" and "b" (shared objects jittered), as a Database."""
import numpy as np
import pyarrow as pa

from fixtures.synthetic import synthetic_point_table
from oneuniverse.data.database import Database
from oneuniverse.data.manifest import LoaderInfo, SurveyMeta
from oneuniverse.data.schema import Geometry
from oneuniverse.data.writer import write_dataset


def write_survey(root, name, table, **kw):
    return write_dataset(table, root / name, geometry=Geometry.POINT,
                         survey=SurveyMeta(name, "spectroscopic"), loader=LoaderInfo("test", "1", {}),
                         partition_nside=4, extra_stats_columns=["mag"], **kw)


def pair_tables(n, n_common, seed=0):
    """a and b with n rows each; source rows < n_common are the same objects (jittered)."""
    """a and b with n rows each; source rows < n_common are the same objects (jittered)."""
    kw = {"dec_range": (-80.0, 80.0)}
    ta = synthetic_point_table(n, seed, survey="a", **kw)
    tb = synthetic_point_table(n, seed + 1, survey="b", **kw)
    pos_a = np.argsort(ta["source_row"].to_numpy())  # position of source row k
    src_b = tb["source_row"].to_numpy()
    common = src_b < n_common
    rng = np.random.default_rng(seed + 2)
    cols = {c: tb[c].to_numpy().copy() for c in ("ra", "dec", "z")}
    ia = pos_a[src_b[common]]
    k = int(common.sum())
    cols["ra"][common] = ta["ra"].to_numpy()[ia] + rng.uniform(-0.2, 0.2, k) / 3600 / np.cos(
        np.radians(ta["dec"].to_numpy()[ia]))
    cols["dec"][common] = ta["dec"].to_numpy()[ia] + rng.uniform(-0.2, 0.2, k) / 3600
    cols["z"][common] = ta["z"].to_numpy()[ia] + rng.uniform(-1e-4, 1e-4, k)
    for c, v in cols.items():
        tb = tb.set_column(tb.schema.get_field_index(c), c, pa.array(v))
    return ta, tb



def make_two_survey_db(tmp_path, n_each=30_000, n_common=10_000, seed=0) -> Database:
    ta, tb = pair_tables(n_each, n_common, seed)
    write_survey(tmp_path, "a", ta)
    write_survey(tmp_path, "b", tb)
    return Database(tmp_path)
