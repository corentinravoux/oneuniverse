import pytest
from oneuniverse._registry import Registry


def test_register_by_explicit_name_then_get():
    reg = Registry("widget")
    reg.register("ALPHA", name="a")
    assert reg.get("a") == "ALPHA"
    assert "a" in reg
    assert reg.names() == ["a"]


def test_register_with_key_function():
    reg = Registry("loader", key=lambda cls: cls.__name__.lower())

    class Foo: ...

    reg.register(Foo)
    assert reg.get("foo") is Foo


def test_duplicate_name_raises():
    reg = Registry("widget")
    reg.register(1, name="x")
    with pytest.raises(ValueError, match="already registered"):
        reg.register(2, name="x")


def test_unknown_get_raises_keyerror_with_known_list():
    reg = Registry("widget")
    reg.register(1, name="x")
    with pytest.raises(KeyError, match=r"known: \['x'\]"):
        reg.get("missing")


def test_mapping_is_read_only_view_of_live_dict():
    reg = Registry("widget")
    reg.register(1, name="x")
    m = reg.mapping
    assert dict(m) == {"x": 1}
    with pytest.raises(TypeError):
        m["y"] = 2  # MappingProxyType is read-only


def test_data_registry_still_exposes_live_dict_and_proxy():
    from oneuniverse.data import _registry as r
    # _REGISTRY is the live dict; REGISTRY is the read-only proxy over it.
    assert r.REGISTRY.keys() == r._REGISTRY.keys()
    assert isinstance(r.list_surveys(), dict)
