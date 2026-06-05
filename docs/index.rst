oneuniverse
===========

**One sky, every survey, one queryable Universe.**

``oneuniverse`` ingests and standardises astronomical survey catalogs, combines
them, builds analysis-ready **measurements** for cosmology estimators, and
provides a storage + simulation substrate for a constrained digital twin of the
cosmic web. It is cosmology-free where it counts: H₀ / Ωₘ / distance models
never enter the data — only the estimator/inference call.

Three pillars
-------------

============  ===========================================================  ==============================
Pillar        What it does                                                 Package
============  ===========================================================  ==============================
1 — Data      ingest → standardise (OUF 2.5) → cross-match → weight        ``data``, ``combine``
2 — Measure   build the ``MeasurementSet`` (general P1→P2 output)          ``measure``
3 — Sim       OUF-Sim storage + fast-PM + resimulation + data↔sim twin     ``simulation``, ``twin``
============  ===========================================================  ==============================

Estimators (P(k), ξ, C_ℓ, f σ₈ — ``flip``, ``pycorr``, ``picca``) are external,
downstream tools that consume the ``MeasurementSet``; their adapters live in a
separate package.

Quickstart
----------

.. code-block:: python

   from oneuniverse.data import load_catalog, DatasetView
   from oneuniverse.measure import build_galaxy_clustering

   view = DatasetView.from_path(survey_path)          # partial-access OUF reader
   ms = build_galaxy_clustering(view, z_range=(0.8, 2.2),
                                weights=[...], nz_edges=..., randoms="generate")
   ms.summary()                                       # cosmology-free description

Pillar 1 — data
~~~~~~~~~~~~~~~~
OUF 2.5 on disk (``manifest.json`` + HEALPix-NSIDE32-NEST parquet). Geometries:
``POINT``, ``SIGHTLINE``, ``HEALPIX``/``GW_SKYMAP``, ``CUBE``, ``LIGHTCURVE``.
ONEUID bitemporal cross-match + sub-object links; weights in ``combine`` (FKP,
completeness, systematics, shear, PIP). Real loaders: ``eboss_qso``,
``desi_qso``, ``dummy`` (others are scaffolds — see ``REVIEW.md``).

Pillar 2 — measure
~~~~~~~~~~~~~~~~~~~
One Universal ``DataProduct`` (``PointSet`` / ``Sightline`` / ``FieldMap``)
carries every probe's atoms; per-probe builders emit a cosmology-free
``MeasurementSet`` (data + randoms + n(z) + window + weights + region map +
provenance). It builds and validates the handoff; it does **not** compute the
estimator. Validated end-to-end on real eBOSS DR16Q + DESI DR1 QSO.

Pillar 3 — simulation + twin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OUF-Sim multi-backend storage with index-only wrap-in-place + partial access;
fast-PM + TreePM-split resimulation; data↔sim twin (mock-challenge → Wiener →
data-driven resim). Dummy/toy physics; real storage/orchestration substrate.

API reference
-------------

.. toctree::
   :maxdepth: 2

   api

See also
--------

- ``REVIEW.md`` — honest external review + known issues.
- ``plans/README.md`` — full roadmap index.
- ``plans/2026-06-05-pillar2-definition.md`` — Pillar-2 / DataProduct definition.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
