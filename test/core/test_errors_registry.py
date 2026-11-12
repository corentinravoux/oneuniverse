import pytest


def test_error_hierarchy():
    from oneuniverse._core.errors import OneuniverseError, FormatError, SchemaError, NotFoundError, ValidationError
    for cls in (FormatError, SchemaError, NotFoundError, ValidationError):
        assert issubclass(cls, OneuniverseError)


def test_registry_moved_and_shim_identical():
    from oneuniverse._core.registry import Registry
    import oneuniverse._registry as old
    assert old.Registry is Registry
    r = Registry("thing"); r.register(1, name="a")
    assert r.get("a") == 1 and "a" in r and r.names() == ["a"]


class _FakeEP:
    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


def test_registry_mapping_and_items_dict():
    from types import MappingProxyType
    from oneuniverse._core.registry import Registry
    r = Registry("thing"); r.register(1, name="a")
    assert isinstance(r.mapping, MappingProxyType) and dict(r.mapping) == {"a": 1}
    assert r.items_dict == {"a": 1}
    r.items_dict["b"] = 2  # live dict
    assert "b" in r


def test_registry_load_entry_points(monkeypatch):
    import oneuniverse._core.registry as rmod
    sentinel, builtin = object(), object()
    monkeypatch.setattr(
        rmod, "entry_points",
        lambda group=None: [_FakeEP("plug", sentinel), _FakeEP("dup", object())]
        if group == "grp" else [],
    )
    r = rmod.Registry("thing"); r.register(builtin, name="dup")
    assert r.load_entry_points("grp") == ["plug"]
    assert r.get("plug") is sentinel and r.get("dup") is builtin
    assert r.load_entry_points("other") == []


def test_dual_inheritance_keeps_old_except_clauses_working():
    from oneuniverse._core.errors import NotFoundError, ValidationError
    assert issubclass(ValidationError, ValueError)
    assert issubclass(NotFoundError, KeyError)


def test_registry_error_paths():
    from oneuniverse._core.errors import NotFoundError, ValidationError
    from oneuniverse._core.registry import Registry
    r = Registry("thing")
    with pytest.raises(ValidationError):
        r.register(1)  # no name, no key
    r.register(1, name="a")
    with pytest.raises(ValidationError):
        r.register(2, name="a")  # duplicate
    with pytest.raises(NotFoundError):
        r.get("zzz")
    with pytest.raises(KeyError):
        r.get("zzz")
