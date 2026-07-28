# Getting started

## Installation

```bash
git clone https://github.com/corentinravoux/oneuniverse.git
cd oneuniverse
pip install -e ".[all]"     # numpy/pandas + fitsio, pyarrow, healpy, matplotlib
# or the minimal core:
pip install -e .
pip install -e ".[dev]"     # + pytest and the full test dependencies
```

`oneuniverse` is pure Python (numpy / pandas / pyarrow / healpy). The optional
extras pull in FITS reading (`fitsio`), Parquet (`pyarrow`), HEALPix (`healpy`)
and plotting (`matplotlib`).

## Loading a catalog

```python
import oneuniverse as ou

ou.list_surveys()                     # what can I load?
ou.list_survey_types()                # galaxy / qso / pv / sn / shear / lya / map

df = ou.load_catalog(
    "sdss_mgs",
    selection=ou.Cone(ra=185, dec=15, radius=5),   # spatial / redshift filters
    columns=["ra", "dec", "z"],
)
```

Every loader returns a **standardised** pandas DataFrame with lowercase
oneuniverse column names, so the same downstream code works across surveys.
Selections (`Cone`, `Shell`, `SkyPatch`) combine with AND logic.

## Combining surveys and weights

```python
from oneuniverse.combine import default_weight_for, combine_weights, FKPWeight

weight = default_weight_for("galaxy", z_type="spec")
```

The [`combine`](api/combine.md) subpackage holds all weighting (FKP, inverse
variance, quality masks, PIP bitweights, shear weights) and the cross-survey
combination logic — no cosmology enters here.

## Building a MeasurementSet (P1 → P2)

```python
from oneuniverse.measure import build_galaxy_clustering

ms = build_galaxy_clustering(df, randoms=..., weights=...)
# ms is a cosmology-free MeasurementSet / Universal DataProduct
# ready for any Pillar-2 estimator (flip, p1desi, lyavoid, ...)
```

See [`measure`](api/measure.md) for the full set of builders
(`build_cosmic_shear`, `build_3x2pt`, `build_peculiar_velocity`,
`build_sn_hubble`, `build_lya`, `build_map_cross`) and the
`MeasurementSet` / `PointSet` / `Sightline` / `FieldMap` data products.

## Simulation and the digital twin (Pillar 3)

```python
from oneuniverse.simulation import Cube, CosmologySpec, ExecutionPlan
from oneuniverse.twin import wiener_reconstruct, run_mock_challenge
```

[`simulation`](api/simulation.md) is the OUF-Sim storage + orchestration
substrate; [`twin`](api/twin.md) is the data ↔ simulation coupling layer
(constrained realizations, mock observation, recovery metrics).

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q                   # ~780 tests
```
