"""S17 T7 — deterministic rank partitioning of chunk reads."""
from oneuniverse.simulation.oufsim._partition import partition_by_rank


def test_partition_is_disjoint_and_complete():
    rows = list(range(10))
    parts = [partition_by_rank(rows, rank=r, size=3) for r in range(3)]
    assert sorted(x for p in parts for x in p) == rows      # complete
    seen = set()
    for p in parts:
        assert not (set(p) & seen)                          # disjoint
        seen |= set(p)


def test_single_rank_gets_everything():
    rows = list(range(5))
    assert partition_by_rank(rows, rank=0, size=1) == rows
