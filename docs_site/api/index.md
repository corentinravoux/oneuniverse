# API reference

Generated from the NumPy-style docstrings with
[mkdocstrings](https://mkdocstrings.github.io/). Each page documents one
subpackage: the names in its `__all__`, or, for `reference` (which exports
nothing from its `__init__`), its modules.

| Page | Contents |
|---|---|
| [`data`](data.md) | OUF 3.0, loaders, `DatasetView`, `Database`, ONEUID, links, sidecars |
| [`combine`](combine.md) | `Weight` primitives, `WeightedCatalog` |
| [`measure`](measure.md) | `MeasurementSet`, products, `build_pointset`, presets, save and load |
| [`simulation`](simulation.md) | OUF-Sim 1.0 store, manifest and specs, converters, `SimDatabase` |
| [`twin`](twin.md) | Engine contracts, `observe_from_view`, `run_engine_to_store` |
| [`reference`](reference.md) | Toy reference physics, by module |
| [`core`](core.md) | Top-level package and `_core` (errors, geometry, registry, HEALPix, fs) |

The SQL functions `connect`, `cone_where` and `box_where` live in
`oneuniverse.data.sql` and `oneuniverse.simulation.sql`; see the
[SQL face](../sql.md).
