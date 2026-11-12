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
