# Weight Normalization for Cross-Survey Combination in `oneuniverse`

## Problem statement

When the same astrophysical object (galaxy, QSO, SN Ia) is observed by multiple
surveys, each measurement arrives with a raw weight `w_{s,i}` that mixes several
effects:

- **Inverse-variance** (`1/sigma^2`) on different observables: redshift `sigma_z`,
  velocity `sigma_v`, distance modulus `sigma_mu`, log-distance `sigma_eta`.
  Units differ by many orders of magnitude.
- **FKP weights** `w_FKP = 1 / (1 + nbar(z) * P0)`, scale with mean density and
  a fiducial power `P0`.
- **Systematics / completeness weights** (e.g. eBOSS `WEIGHT_SYSTOT`,
  `WEIGHT_CP`, `WEIGHT_NOZ`), which are O(1) multiplicative corrections for
  imaging systematics, fibre collisions, and redshift failures.
- **Subjective priorities** reflecting the user's trust in a survey
  (e.g. "DESI measurements should dominate over eBOSS for the same object").

These cannot be naively summed. This document reviews prior art and
recommends a default strategy for `oneuniverse.weight`.

## Prior art

### eBOSS / BOSS LSS catalogues (Ross et al. 2020, Ata et al. 2018)

The canonical eBOSS LSS weight is a **product of independent multiplicative
factors** times an FKP weight:

```
w_tot = w_systot * w_cp * w_noz * w_FKP
```

All `w_*` factors are O(1), and the FKP weight sets the absolute scale. No
normalization is applied: weights go directly into pair counts and the Landy-
Szalay estimator. Because FKP is a per-tracer optimal weighting under the
Feldman-Kaiser-Peacock (1994) formalism, multiplying by a global constant has
no effect on the estimator (it cancels in DD/RR), but it does affect a
likelihood where `C^{-1}` appears.

### DESI LSS pipeline (`LSS/py/LSS/`, Ross et al. 2024)

DESI follows the same factorised form and adds `WEIGHT_ZFAIL` and
`WEIGHT_IMLIN` (linear imaging regression). Importantly, DESI **never
cross-matches objects between tracers at the weight level**: BGS, LRG, ELG,
QSO are kept as separate tracers and combined at the power-spectrum /
likelihood level with multi-tracer estimators.

### `nbodykit`, `pypower`, `pycorr`

These libraries treat the weight column as opaque: they multiply it into pair
counts and delegate normalization to the user. `pypower.CatalogFFTPower`
normalizes the shot-noise term by `sum(w^2) / sum(w)^2 * V`, which means
**relative magnitudes across catalogues matter** and naive concatenation of
two surveys with different weight scales biases the shot-noise estimate. The
recommended practice (docstrings, Alam et al. 2021 appendix) is to rescale
each catalogue so `mean(w_data) / mean(w_random) = 1` per survey before
stacking.

### Multi-tracer literature

- **Percival & White 2009; McDonald & Seljak 2009**: multi-tracer analyses
  keep tracers *separate* and combine at the estimator / likelihood level.
  Each tracer retains its own `nbar`, bias, and FKP weight.
- **Abramo & Leonard 2013; Abramo et al. 2016**: the multi-tracer Fisher
  formalism derives optimal weights from the covariance between tracers; no
  ad-hoc normalization is applied.
- **Hamaus et al. 2010**: inverse-variance weighting across tracers assumes
  knowledge of the full cross-tracer covariance.

**Consensus**: the literature *does not* combine the same object across
surveys via weighted averaging. Duplicates are typically resolved by
**picking one catalogue** (best SNR or latest survey) and discarding the
rest. Cross-survey information enters through independent tracers, not
through per-object averaging.

### Subjective priorities

No mainstream LSS analysis injects subjective survey priorities into
weights. The closest analogue is the **hyperparameter technique of Lahav
et al. 2000 / Hobson, Bridle & Lahav 2002**, where each dataset gets a
multiplicative factor `alpha_s` in `-2 ln L = sum_s alpha_s chi^2_s`, with
`alpha_s` either fixed by the user or marginalised over a Jeffreys prior.
This is statistically clean: `alpha_s = 2` means "treat DESI as if it had
twice the effective number of data points". It does *not* change the
inverse-variance interpretation within a survey.

## Recommendations

### Default: "best-only" with conflict logging

For duplicates (same sky position, z, object type across surveys), keep the
measurement with the highest effective SNR (lowest variance on the common
observable, converted to a velocity-equivalent `sigma_v` via the flip
converters). Discard the others but log the conflict.

Rationale: matches DESI/eBOSS practice, preserves the inverse-variance
interpretation, and feeds `flip`'s Gaussian likelihood without rescaling
`C^{-1}`.

### Alternative 1: Per-object inverse-variance average

When the observables are homogenised to a common quantity (e.g. all
converted to `v_pec` in km/s via `flip.data_vector.vector_utils`), combine
with:

```
v_comb    = sum_s (w_s * v_s) / sum_s w_s
sigma^2   = 1 / sum_s w_s                 with  w_s = 1 / sigma_s^2
```

This is the BLUE estimator and is valid only if the measurements are
independent. The output `sigma^2` is what `flip` needs; no further rescaling.

### Alternative 2: Hyperparameter rescaling (Lahav 2000)

Expose a per-survey multiplicative `alpha_s` (default 1.0). Apply as

```
w_{s,i} -> alpha_s * w_{s,i}        (before inverse-variance combination)
```

For the likelihood in `flip`, this is equivalent to dividing the survey's
block of `C` by `alpha_s`. Document it as "effective relative sample size",
not as "trust". Users who want "DESI 2x preferred over eBOSS" set
`alpha_DESI / alpha_eBOSS = 2`.

### Alternative 3: Survey-level rescaling to unit mean

```
w_{s,i} -> w_{s,i} / <w_s>
```

Used by `pypower` recipes to make shot-noise comparable across catalogues.
Appropriate for clustering estimators but **not** for a Gaussian
likelihood on peculiar velocities, because it destroys the inverse-variance
scale. Provide it but gate it behind an explicit opt-in for clustering use.

### API sketch

```python
from oneuniverse.weight import combine_weights

combined = combine_weights(
    catalogs,                         # dict[survey_name] -> DataFrame
    strategy="best_only",             # "best_only" | "ivar_average" |
                                      # "hyperparameter" | "unit_mean"
    observable="v_pec",               # common quantity to combine on
    survey_alpha={"DESI": 2.0, "eBOSS": 1.0},   # only if hyperparameter
    match_tol={"sky_arcsec": 1.0, "dz": 1e-4},
    log_conflicts=True,
)
```

### Impact on `flip` likelihoods

- `best_only` and `ivar_average` preserve the inverse-variance interpretation
  of `data_variance`, so `flip.CovMatrix.compute_covariance_sum` consumes
  them unchanged.
- `hyperparameter` multiplies the data-variance block by `1/alpha_s`, which
  is exactly what the Lahav prescription prescribes. Apply it inside
  `oneuniverse.weight` before handing to `flip`; do not modify `flip`.
- `unit_mean` must not be used with `flip`'s Gaussian likelihood. Raise a
  warning when the user selects it together with a downstream `flip` fit.

## Summary table

| Strategy | Preserves `1/sigma^2` | Uses all data | Safe for `flip` GLL | Notes |
|---|---|---|---|---|
| best_only (default) | yes | no | yes | Matches DESI/eBOSS |
| ivar_average | yes | yes | yes | Requires common observable |
| hyperparameter | rescaled | yes | yes | Clean way to encode priorities |
| unit_mean | no | yes | no | Clustering only |

## References

- Feldman, Kaiser, Peacock 1994, ApJ 426, 23
- Lahav et al. 2000, MNRAS 315, L45; Hobson, Bridle, Lahav 2002, MNRAS 335, 377
- Percival & White 2009, MNRAS 393, 297
- McDonald & Seljak 2009, JCAP 10, 007
- Hamaus et al. 2010, PRD 82, 043515
- Abramo & Leonard 2013, MNRAS 432, 318
- Ross et al. 2020 (eBOSS DR16 LSS), MNRAS 498, 2354
- Alam et al. 2021 (eBOSS cosmology), PRD 103, 083533
- Ross et al. 2024 (DESI Y1 LSS catalogues)
