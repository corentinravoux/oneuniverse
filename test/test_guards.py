"""Architectural guards for the oneuniverse 1.0 rewrite (AST based).

1. Import boundaries: which subpackage may import which.
2. Cosmology guard: no cosmology identifiers, imports or string keys in the
   data / combine / measure layers.
"""
import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "oneuniverse"

BOUNDARY_ROWS = [
    ("_core", "_core",
     {"oneuniverse.data", "oneuniverse.combine", "oneuniverse.measure",
      "oneuniverse.simulation", "oneuniverse.twin", "oneuniverse.reference"}),
    ("simulation", "simulation",
     {"oneuniverse.data", "oneuniverse.combine", "oneuniverse.measure",
      "oneuniverse.twin", "oneuniverse.reference"}),
    ("measure", "measure",
     {"oneuniverse.simulation", "oneuniverse.reference", "oneuniverse.twin"}),
    ("twin/engine.py", "twin/engine.py", {"oneuniverse.reference"}),
]

COSMO_TREES = ["data", "combine", "measure"]
COSMO_IDENTIFIERS = {"H0", "Om0", "omega_m", "Omega_m", "sigma8", "sigma_8",
                     "w0", "wa"}
COSMO_IMPORTS = {"astropy.cosmology", "camb", "classy"}
COSMO_STRINGS = {"comoving_distance", "r_comoving", "dist_mpc"}

# Old-tree violations, keyed by path relative to oneuniverse/. Value is the
# set of allowed identifiers, or {"*"} for everything in the file.
# removed in R1 Task 1 when the old package is deleted
ALLOW = {
    "measure/measurement_set.py": {"comoving_distance", "r_comoving"},
}


def _matches(module, prefixes):
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _package_of(path, pkg_root):
    """Dotted package that relative imports in `path` resolve against."""
    rel = path.relative_to(pkg_root.parent).with_suffix("")
    parts = list(rel.parts)
    return parts[:-1]  # drop module name (also correct for __init__)


def _py_files(root):
    return [root] if root.is_file() else sorted(root.rglob("*.py"))


def _imported_modules(node, package):
    """Absolute dotted names imported by an Import/ImportFrom node."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if node.level:
        base_parts = package[: len(package) - (node.level - 1)] \
            if node.level - 1 <= len(package) else []
        base = ".".join(base_parts + ([node.module] if node.module else []))
    else:
        base = node.module or ""
    return [base] + [f"{base}.{a.name}" for a in node.names if a.name != "*"]


def violations(tree_root, forbidden_prefixes):
    tree_root = Path(tree_root)
    pkg_root = _pkg_root(tree_root)
    out = []
    for f in _py_files(tree_root):
        package = _package_of(f, pkg_root)
        mod = ast.parse(f.read_text(), filename=str(f))
        for node in ast.walk(mod):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for m in _imported_modules(node, package):
                    if _matches(m, forbidden_prefixes):
                        out.append(f"{f.relative_to(pkg_root)}:{node.lineno}: "
                                   f"imports {m}")
                        break
    return out


def _pkg_root(tree_root):
    for p in [tree_root, *tree_root.parents]:
        if p.name == "oneuniverse":
            return p
    raise ValueError(f"{tree_root} is not inside an 'oneuniverse' package")


def _docstring_nodes(mod):
    ids = set()
    for n in ast.walk(mod):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                          ast.AsyncFunctionDef)):
            body = n.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ids.add(id(body[0].value))
    return ids


def _cosmo_hits(path):
    """Yield (lineno, identifier) cosmology hits in one file."""
    mod = ast.parse(path.read_text(), filename=str(path))
    docs = _docstring_nodes(mod)
    package = _package_of(path, _pkg_root(path))
    for n in ast.walk(mod):
        if isinstance(n, ast.Name) and n.id in COSMO_IDENTIFIERS:
            yield n.lineno, n.id
        elif isinstance(n, ast.Attribute) and n.attr in COSMO_IDENTIFIERS:
            yield n.lineno, n.attr
        elif isinstance(n, ast.arg) and n.arg in COSMO_IDENTIFIERS:
            yield n.lineno, n.arg
        elif isinstance(n, ast.keyword) and n.arg in COSMO_IDENTIFIERS:
            yield n.value.lineno, n.arg
        elif (isinstance(n, ast.Constant) and isinstance(n.value, str)
              and n.value in COSMO_STRINGS and id(n) not in docs):
            yield n.lineno, n.value
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for m in _imported_modules(n, package):
                if _matches(m, COSMO_IMPORTS):
                    yield n.lineno, m
                    break


def cosmology_violations(tree_root, allow=None):
    allow = ALLOW if allow is None else allow
    tree_root = Path(tree_root)
    pkg_root = _pkg_root(tree_root)
    out = []
    for f in _py_files(tree_root):
        rel = f.relative_to(pkg_root).as_posix()
        allowed = allow.get(rel, set())
        if "*" in allowed:
            continue
        for lineno, ident in _cosmo_hits(f):
            if ident not in allowed:
                out.append(f"{rel}:{lineno}: cosmology identifier {ident!r}")
    return out


@pytest.mark.parametrize("label,subpath,forbidden", BOUNDARY_ROWS,
                         ids=[r[0] for r in BOUNDARY_ROWS])
def test_import_boundary(label, subpath, forbidden):
    root = PKG / subpath
    if not root.exists():
        pytest.skip(f"{subpath} does not exist yet")
    found = violations(root, forbidden)
    assert not found, "forbidden imports:\n" + "\n".join(found)


@pytest.mark.parametrize("tree", COSMO_TREES)
def test_no_cosmology_in_layer(tree):
    root = PKG / tree
    if not root.exists():
        pytest.skip(f"{tree} does not exist yet")
    found = cosmology_violations(root)
    assert not found, "cosmology in a cosmology-free layer:\n" + "\n".join(found)


def test_guard_helpers_detect_violations(tmp_path):
    pkg = tmp_path / "oneuniverse"
    sub = pkg / "layer"
    sub.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (sub / "__init__.py").write_text("")
    (sub / "bad.py").write_text(
        '"""Docstring mentions "comoving_distance" and H0."""\n'
        "import oneuniverse.data.x\n"
        "from .. import data\n"
        "from ..data import y\n"
        "import oneuniverse.database_x\n"
        "def f(a):\n"
        '    """comoving_distance"""\n'
        "    return H0 + a\n"
        'KEY = "comoving_distance"\n'
        "g = lambda sigma8: sigma8\n"
        "h(wa=1)\n"
        "import camb\n"
    )
    v = violations(sub, {"oneuniverse.data"})
    lines = sorted(int(s.split(":")[1]) for s in v)
    assert lines == [2, 3, 4], v  # database_x must not match
    c = cosmology_violations(sub, allow={})
    got = sorted((int(s.split(":")[1]), s.split("'")[1]) for s in c)
    assert got == [(8, "H0"), (9, "comoving_distance"), (10, "sigma8"),
                   (10, "sigma8"), (11, "wa"), (12, "camb")], c
    assert cosmology_violations(
        sub, allow={"layer/bad.py": {"*"}}) == []
    assert len(cosmology_violations(
        sub, allow={"layer/bad.py": {"H0"}})) == 5
