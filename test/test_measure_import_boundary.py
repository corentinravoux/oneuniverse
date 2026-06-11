"""S7 — import-boundary guard for `oneuniverse.measure`.

`measure` is the P1→P2 layer: it may import `data` and `combine`, but must
NEVER import `simulation` or `twin` (mirror of the Rule-1 guard that keeps
`simulation` free of `data`). A static scan, like the Rule-1 test.
"""
import re
from pathlib import Path

MEASURE = Path(__file__).parent.parent / "oneuniverse" / "measure"
_FORBIDDEN = re.compile(
    r"^\s*(from|import)\s+oneuniverse\.(simulation|twin)\b", re.M)


def test_measure_never_imports_simulation_or_twin():
    offenders = {}
    for f in MEASURE.rglob("*.py"):
        hits = _FORBIDDEN.findall(f.read_text())
        if hits:
            offenders[str(f.relative_to(MEASURE))] = len(hits)
    assert not offenders, (
        f"measure/ must not import simulation/ or twin/: {offenders}")
