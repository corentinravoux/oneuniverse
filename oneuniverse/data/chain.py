"""Multi-level sub-object chain walker.

Given a list of ``SubobjectLinks.table`` frames and a starting set of
oneuids, walk the chain end-to-end and return the union of leaf-level
oneuids reachable from any starting row.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd


def chain_subobjects_tables(
    starts: Iterable[int],
    link_tables: Sequence[pd.DataFrame],
) -> List[int]:
    """Walk *link_tables* in order; return the sorted union of leaf
    ``child_oneuid`` values reachable from ``starts``.
    """
    current = {int(s) for s in starts}
    for table in link_tables:
        if table.empty or not current:
            current = set()
            continue
        mask = table["parent_oneuid"].isin(current)
        current = set(
            table.loc[mask, "child_oneuid"].astype("int64").tolist()
        )
    return sorted(current)
