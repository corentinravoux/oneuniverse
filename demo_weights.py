"""Visual demo of oneuniverse.weight — three synthetic surveys observing an
overlapping population of galaxies, cross-matched and combined with several
weighting strategies.

Produces: demo_weights.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oneuniverse.weight import (
    InverseVarianceWeight,
    QualityMaskWeight,
    WeightedCatalog,
    combine_weights,
)


# ── Build three synthetic surveys ──────────────────────────────────────────

rng = np.random.default_rng(0)

# 200 "true" universal objects on a small sky patch, redshift 0.1–0.6.
N_TRUE = 200
true_ra = rng.uniform(149.0, 151.0, N_TRUE)
true_dec = rng.uniform(1.0, 3.0, N_TRUE)
true_z = rng.uniform(0.10, 0.60, N_TRUE)
true_v = 299_792.458 * true_z  # toy: z → v


def sample_survey(name, selected_idx, sigma_v, quality_fail_rate=0.0,
                  sky_jitter_arcsec=0.3, dz_jitter=3e-5):
    """Return a DataFrame mimicking a survey that observed a subset."""
    n = len(selected_idx)
    jitter_deg = sky_jitter_arcsec / 3600.0
    df = pd.DataFrame({
        "ra":  true_ra[selected_idx] + rng.normal(0, jitter_deg, n),
        "dec": true_dec[selected_idx] + rng.normal(0, jitter_deg, n),
        "z":   true_z[selected_idx] + rng.normal(0, dz_jitter, n),
        "v":   true_v[selected_idx] + rng.normal(0, sigma_v, n),
        "v_err": np.full(n, sigma_v, dtype=float),
        "zwarn": (rng.uniform(0, 1, n) < quality_fail_rate).astype(int) * 4,
    })
    df["survey"] = name
    df["true_id"] = selected_idx
    return df


# eBOSS: large but noisier; DESI: newer, much better; 6dF: only bright, few.
idx_eboss = rng.choice(N_TRUE, 140, replace=False)
idx_desi  = rng.choice(N_TRUE, 120, replace=False)
idx_6df   = rng.choice(N_TRUE, 40,  replace=False)

eboss = sample_survey("eBOSS", idx_eboss, sigma_v=60.0, quality_fail_rate=0.05)
desi  = sample_survey("DESI",  idx_desi,  sigma_v=20.0, quality_fail_rate=0.01)
sixdf = sample_survey("6dFGS", idx_6df,   sigma_v=150.0, quality_fail_rate=0.10)

# ── Build WeightedCatalog and register per-survey weights ──────────────────

wc = WeightedCatalog({"eBOSS": eboss, "DESI": desi, "6dFGS": sixdf})

ivar = lambda: InverseVarianceWeight("v_err", floor=250.0)  # σ_* = 250 km/s
mask = QualityMaskWeight("zwarn", "==", 0)

wc.add_weight("eBOSS", ivar()).add_weight("eBOSS", mask)
wc.add_weight("DESI",  ivar()).add_weight("DESI",  mask)
wc.add_weight("6dFGS", ivar()).add_weight("6dFGS", mask)

match = wc.crossmatch(sky_tol_arcsec=2.0, dz_tol=2e-3)
print(match)

# Add the v_var column the combiner needs.
wc._weighted_long["v_var"] = wc._weighted_long["v_err"] ** 2

combined_best = wc.combine("v", "v_var", strategy="best_only")
combined_blue = wc.combine("v", "v_var", strategy="ivar_average")
combined_hyp  = wc.combine(
    "v", "v_var", strategy="hyperparameter",
    survey_alpha={"DESI": 3.0, "eBOSS": 1.0, "6dFGS": 0.5},
)

print(f"universal objects:       {wc.n_universal()}")
print(f"multi-survey concurrences: {wc.n_multi_survey()}")


# ── Figure ─────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

colors = {"eBOSS": "#e07a2b", "DESI": "#2b6fe0", "6dFGS": "#6ca84f"}
markers = {"eBOSS": "o", "DESI": "s", "6dFGS": "^"}

# 1) sky map of all three surveys
ax = fig.add_subplot(gs[0, 0])
for name, df in wc.catalogs.items():
    ax.scatter(df["ra"], df["dec"], s=18, alpha=0.7,
               c=colors[name], marker=markers[name], label=name,
               edgecolors="none")
ax.set_xlabel("RA [deg]")
ax.set_ylabel("Dec [deg]")
ax.set_title("(a) Sky positions")
ax.legend(loc="upper right", framealpha=0.9)
ax.set_aspect("equal")

# 2) multiplicity: how many surveys see each universal object
ax = fig.add_subplot(gs[0, 1])
ms = match.table.groupby("universal_id")["survey"].nunique()
counts = ms.value_counts().sort_index()
bars = ax.bar(counts.index, counts.values,
              color=["#bbbbbb", "#87c1ff", "#3a7bd5"], edgecolor="black")
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, str(int(v)),
            ha="center", fontsize=10)
ax.set_xlabel("# surveys observing the object")
ax.set_ylabel("# universal objects")
ax.set_title("(b) Cross-survey multiplicity")
ax.set_xticks(counts.index)

# 3) per-survey weight distributions (log scale)
ax = fig.add_subplot(gs[0, 2])
for name in ("eBOSS", "DESI", "6dFGS"):
    w = wc.total_weight(name)
    w = w[w > 0]
    ax.hist(np.log10(w), bins=25, alpha=0.55,
            color=colors[name], label=name, edgecolor="black", linewidth=0.4)
ax.set_xlabel(r"$\log_{10}\ w_i$")
ax.set_ylabel("count")
ax.set_title("(c) IVW × quality-mask weight")
ax.legend()

# 4) example of ONE universal object — all concurrences
ax = fig.add_subplot(gs[1, :])
ms_ids = match.multi_survey()["universal_id"].unique()
# pick a triple match if available
triples = [uid for uid in ms_ids
           if wc.concurrences(int(uid))["survey"].nunique() == 3]
example_uid = int(triples[0]) if triples else int(ms_ids[0])
cc = wc.concurrences(example_uid).sort_values("survey")
ax.errorbar(
    cc["survey"], cc["v"], yerr=cc["v_err"],
    fmt="o", capsize=5, lw=1.5, ms=9,
    color="black", zorder=3,
)
for _, row in cc.iterrows():
    ax.scatter(row["survey"], row["v"], s=230,
               color=colors[row["survey"]], zorder=2, edgecolor="black")
# Overlay combiners
for label, res, sty in [
    ("best_only",        combined_best, dict(ls="--", color="C3")),
    ("ivar_average",     combined_blue, dict(ls="-",  color="C0")),
    ("hyperparameter α_DESI=3", combined_hyp, dict(ls=":",  color="C2")),
]:
    row = res.table[res.table["universal_id"] == example_uid]
    if not row.empty:
        v = float(row["value"].iloc[0])
        ax.axhline(v, lw=2, label=label, **sty)
ax.set_title(f"(d) Universal object #{example_uid}: concurrences and combined values")
ax.set_ylabel("v [km/s]")
ax.legend(loc="best", framealpha=0.9)

# 5) best_only — where does the chosen measurement come from?
ax = fig.add_subplot(gs[2, 0])
picked_surveys = []
for uid in combined_best.table["universal_id"]:
    rows = wc.concurrences(int(uid))
    imin = rows["v_err"].idxmin()
    picked_surveys.append(rows.loc[imin, "survey"])
from collections import Counter
c = Counter(picked_surveys)
labels = ["eBOSS", "DESI", "6dFGS"]
vals = [c.get(k, 0) for k in labels]
ax.bar(labels, vals, color=[colors[k] for k in labels], edgecolor="black")
for i, v in enumerate(vals):
    ax.text(i, v + 1, str(v), ha="center")
ax.set_ylabel("# universal objects")
ax.set_title("(e) best_only: winner per object")

# 6) residuals of ivar_average vs true velocity
ax = fig.add_subplot(gs[2, 1])
# Recover true velocity by joining via nearest RA
long = wc._weighted_long.copy()
true_map = dict(zip(range(N_TRUE), true_v))
# Per universal_id, map the first true_id we can find through row_index
uid_to_true = {}
for uid, grp in long.groupby("universal_id"):
    # use the 'true_id' we stored in each survey DataFrame — rejoin:
    tids = []
    for _, row in grp.iterrows():
        src = wc.catalogs[row["survey"]].iloc[int(
            match.table[(match.table["universal_id"] == uid)
                        & (match.table["survey"] == row["survey"])]["row_index"].iloc[0]
        )]
        tids.append(int(src["true_id"]))
    uid_to_true[int(uid)] = true_map[tids[0]]

for name, res in [("best_only", combined_best),
                  ("ivar_average", combined_blue),
                  ("hyperparameter", combined_hyp)]:
    t = res.table.copy()
    t["true"] = t["universal_id"].map(uid_to_true)
    resid = t["value"] - t["true"]
    ax.hist(resid, bins=25, alpha=0.55, label=name, edgecolor="black", linewidth=0.4)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("v_combined − v_true [km/s]")
ax.set_ylabel("count")
ax.set_title("(f) Residuals of combiners")
ax.legend()

# 7) error bar shrinkage: BLUE σ vs single-survey best σ
ax = fig.add_subplot(gs[2, 2])
best_sigma = np.sqrt(combined_best.table.set_index("universal_id")["variance"])
blue_sigma = np.sqrt(combined_blue.table.set_index("universal_id")["variance"])
common = best_sigma.index.intersection(blue_sigma.index)
ax.scatter(best_sigma.loc[common], blue_sigma.loc[common],
           alpha=0.6, edgecolors="none")
lim = [0, float(best_sigma.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1, label="y = x")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"$\sigma_{\mathrm{best\_only}}$  [km/s]")
ax.set_ylabel(r"$\sigma_{\mathrm{ivar\_average}}$  [km/s]")
ax.set_title("(g) BLUE shrinks the error bars")
ax.set_aspect("equal")
ax.legend()

fig.suptitle(
    "oneuniverse.weight — cross-survey weighting demo "
    f"({wc.n_universal()} universal objects, {wc.n_multi_survey()} seen by ≥2 surveys)",
    fontsize=14, y=0.995,
)

out = Path(__file__).parent / "demo_weights.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"→ wrote {out}")
