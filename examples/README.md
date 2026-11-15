# Examples

Diagnostic figure scripts, one PNG each. Run from the repository root:

```bash
python examples/diagnostics/<name>.py [--out examples/output] [--data-root DB --dataset NAME]
```

| Script | Figure |
|---|---|
| `sky_coverage.py` | Footprint of a dataset: scatter coloured by z, 2D density, n(z) and a HEALPix map with the stored partition cells. |
| `partition_pruning.py` | Cone query: cells read versus total (highlighted), the selected objects, and pruning versus cone radius. |
| `crossmatch_separations.py` | Separation and delta-z histograms of the pairs of a two-survey ONEUID build, and members per ONEUID. |
| `measurement_set_overview.py` | A clustering `MeasurementSet`: catalog, randoms, n(z), jackknife regions, counts per region and window. |
| `resim_buffer_convergence.py` | `reference.resim.buffer_convergence` (`cell_corr`, `r_lowk` against buffer, buffered versus treepm) and inner-cube slices. |
| `mock_challenge.py` | `reference.mock.run_mock_challenge`: truth, tracer and Wiener slices, plus r(k) and T(k). |

Without `--data-root` every script generates small synthetic inputs through the public API
(`data.write_dataset` on random columns in a temporary directory, `reference.linear.gaussian_field`,
`reference.mock`); nothing is read from `test/`. With `--data-root` (a `Database` root of real OUF
datasets) the first dataset(s), or those named with `--dataset`, are used. The scripts are read-only: nothing is
written under `--data-root` (the cross-match uses `match_pair`, not `build_oneuid`). `resim_buffer_convergence.py` and
`mock_challenge.py` are always synthetic. Each runs in a few seconds.

Output goes to `examples/output/` (git-ignored; no figure is committed). The path of the PNG is printed.
If the environment variable `PLT_STYLE` is set, it is applied with `plt.style.use`.

The command-line tools live in `scripts/`: `convert_surveys.py` (raw survey files to OUF datasets)
and `build_reference_store.py` (linear-theory reference simulation to an OUF-Sim store).
