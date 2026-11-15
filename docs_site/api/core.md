# `oneuniverse` and `oneuniverse._core`

The top-level package re-exports `data`, `convert`, `DatasetView`, `Cone`,
`SkyPatch` and `ZRange`. The shared building blocks live in `oneuniverse._core`
and are documented here by module: errors, selection geometry, the registry
used by every plugin point, and HEALPix helpers. They are re-exported by the
subpackages that use them (for example `oneuniverse.data.Cone`).

::: oneuniverse._core.errors

::: oneuniverse._core.geometry

::: oneuniverse._core.registry

::: oneuniverse._core.healpix

::: oneuniverse._core.fs
