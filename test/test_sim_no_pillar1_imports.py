"""Phase S2 T11 — Pillar-3 isolation guard.

oneuniverse.simulation must NOT import from oneuniverse.data or
oneuniverse.combine (Rule 1: minimal cross-pillar coupling). This test
scans every source file under oneuniverse/simulation/ via the AST and
fails if a forbidden import appears.
"""
import ast
from pathlib import Path

import oneuniverse.simulation as sim_pkg

_FORBIDDEN_ROOTS = ("oneuniverse.data", "oneuniverse.combine")


def _sim_source_files():
    root = Path(sim_pkg.__file__).parent
    return sorted(root.rglob("*.py"))


def _forbidden_imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == r or alias.name.startswith(r + ".")
                       for r in _FORBIDDEN_ROOTS):
                    bad.append((path.name, alias.name))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod == r or mod.startswith(r + ".")
                   for r in _FORBIDDEN_ROOTS):
                bad.append((path.name, mod))
    return bad


def test_no_pillar1_imports_anywhere_in_sim():
    offenders = []
    files = _sim_source_files()
    assert files, "no source files found under oneuniverse/simulation/"
    for path in files:
        offenders.extend(_forbidden_imports(path))
    assert offenders == [], (
        "oneuniverse.simulation must not import oneuniverse.data / "
        f"oneuniverse.combine; offenders: {offenders}"
    )
