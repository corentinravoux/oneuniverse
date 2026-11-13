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


def _ep_registry(monkeypatch, eps):
    import oneuniverse._core.registry as rmod
    monkeypatch.setattr(rmod, "entry_points", lambda group=None: eps)
    return rmod.Registry("thing")


class _BadEP:
    def __init__(self, name, exc):
        self.name, self._exc = name, exc

    def load(self):
        raise self._exc


def test_load_entry_points_on_error_raise(monkeypatch):
    r = _ep_registry(monkeypatch, [_BadEP("x", ValueError("boom"))])
    with pytest.raises(ValueError):
        r.load_entry_points("g")


def test_load_entry_points_warn_records(monkeypatch):
    from oneuniverse._core.errors import ValidationError

    def validate(obj):
        if obj == "bad":
            raise ValidationError("nope")

    r = _ep_registry(monkeypatch, [_BadEP("x", ValueError("boom")), _FakeEP("y", "bad"),
                                   _FakeEP("z", "good")])
    with pytest.warns(RuntimeWarning, match="skipping plugin 'x'"):
        added = r.load_entry_points("g", on_error="warn", validate=validate)
    assert added == ["z"]
    assert r.plugin_errors == {"x": "ValueError: boom", "y": "ValidationError: nope"}


def test_load_entry_points_validate_raises_by_default(monkeypatch):
    r = _ep_registry(monkeypatch, [_FakeEP("y", "bad")])

    def validate(obj):
        raise ValueError("no")
    with pytest.raises(ValueError):
        r.load_entry_points("g", validate=validate)
