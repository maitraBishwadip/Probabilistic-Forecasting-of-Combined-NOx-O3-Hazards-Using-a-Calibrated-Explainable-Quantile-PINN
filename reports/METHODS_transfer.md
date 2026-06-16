# Methods — Spatial Transfer (LOSO) & Bootstrap Uncertainty (E7)

**Purpose.** Paper-ready write-up of the spatial-transferability evaluation and the confidence
intervals that make the results table publication-grade. Method only; numbers go to `RESULTS_LOG.md`
(`CLAUDE.md §1`). Companion to `METHODS_calibration.md` (E6) and `EXPERIMENT_DESIGN.md`.

---

## 1. Why spatial transfer matters (IGARSS framing)

Ground monitors are sparse: Bangladesh's network is 9 DoE stations for ~170 M people. A forecast that
only works *at* a station it was trained on is operationally weak; the IGARSS contribution is a model
that **transfers to locations without local training history**, leaning on meteorology/reanalysis
dynamics and NOx–O3 photochemistry rather than memorised station identity. E7 quantifies this.

## 2. Leave-One-Station-Out (LOSO) cross-validation

We use **leave-one-station-out** spatial CV: for each of the 9 stations $k$, train on the other 8 and
evaluate on the held-out station $k$. This is the spatially-blocked analogue of cross-validation
recommended to avoid the optimism of random splits under spatial autocorrelation (Roberts et al. 2017;
Meyer et al. 2019; Milà et al. 2022). Because the 9 DoE stations are **distinct, well-separated
cities**, leaving an entire station out is a clean spatial hold-out and no exclusion buffer is needed.
(We note the methodological debate — Wadoux et al. (2021) caution that spatial CV is not universally
"more correct"; here LOSO directly matches the deployment question "predict at an unmonitored city",
which is the quantity of interest.)

**Protocol (per fold $k$):**
1. **Train** = all anchors from the 8 non-$k$ stations with target window in 2014–15.
2. **Leakage control (critical).** The rank-index transforms $F_{\mathrm{NOx}},F_{\mathrm{O_3}}$, the
   threshold $\mathrm{thr}=Q_{30}$, and all feature scalers are **fit on the 8 training stations'
   2014–15 data only** — the held-out station's climatology is never seen.
3. **Evaluate** on station $k$'s **2016** anchors (out-of-sample in space *and* time).
4. Report pinball, CRPS, $\mathrm{PICP}_{80/90}$, Brier(exceedance) for fold $k$.

**Station-agnostic model.** The in-distribution model used a learned **station embedding**, which is
undefined for an unseen station. For LOSO we therefore use a **station-agnostic** QR-PINN (embedding
removed; identical otherwise — LSTM encoder, monotone quantile head, coupled NOx–O3 physics residual).
This tests pure *dynamical* transfer. We also train a station-agnostic **in-distribution reference**
(all 9 stations, train 2014–15 → test 2016) so the transfer gap is measured against a like-for-like
architecture, not the embedding model.

**Reporting.** Per-station table + **mean ± standard deviation across the 9 folds** (the natural
measure of transfer variability), and the transfer gap = LOSO mean − in-distribution reference.

## 3. Bootstrap confidence intervals

Point metrics on one test year are uncertain; we attach **95 % CIs**. Because hourly forecast errors
are **serially correlated**, an i.i.d. bootstrap understates variance, so we use a **moving-block /
stationary bootstrap** (Künsch 1989; Politis & Romano 1994; general bootstrap, Efron & Tibshirani
1993):

- Reduce each test anchor $i$ to a scalar score $s_i$ (its mean pinball, and separately mean CRPS,
  over that anchor's valid horizons and quantiles).
- Order anchors by (station, time); resample **contiguous blocks of length $b=24$ h** with replacement
  to length $N$; recompute the mean. Repeat $B=1000$ times → 2.5/97.5 percentile CI.

We report CIs for (i) the in-distribution reference and (ii) the pooled LOSO predictions, plus the
across-fold mean ± std. Block length $b=24$ h spans the diurnal cycle (the dominant short-range
dependence); sensitivity to $b$ is a minor robustness check.

## 4. Metrics & tables (schema — filled only from `RESULTS_LOG.md`)

**T3 — Spatial transfer.**

| Setting | Pinball [95% CI] | CRPS [95% CI] | PICP80 | PICP90 | Brier(exc) |
|---|---|---|---|---|---|
| In-distribution (all-9, station-agnostic) | … | … | … | … | … |
| LOSO mean ± std (9 folds) | … | … | … | … | … |

**T4 — Per-station LOSO** (one row per held-out station: pinball, CRPS, PICP90, Brier) → shows which
cities transfer well/poorly and why (e.g., NOx-heavy traffic stations vs cleaner sites).

Figures: per-station LOSO skill bars (with the in-distribution reference line); transfer-gap summary.

## 5. Caveats (state in the paper)
- Only **9 stations** → the across-fold std is a coarse uncertainty estimate; the block-bootstrap CI on
  pooled predictions complements it.
- LOSO uses a **station-agnostic** model, so its in-distribution number differs slightly from the
  embedding model in E5/E6 — we compare LOSO only to the station-agnostic reference.
- Rank-index metrics are on $[0,1]$ and are **not** comparable across different index definitions.

## 6. Reproducibility
Fixed seed per fold; per-fold rank transforms/threshold/scalers persisted; block-bootstrap seed logged.
No imputation of targets; observed $H$ only.

---

## References (full list in `LITERATURE_REVIEW.md`)
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929.
- Meyer, H., et al. (2019). Importance of spatial predictor variable selection in machine learning applications. *Environmental Modelling & Software*, 101, 1–9.
- Milà, C., et al. (2022). Nearest neighbour distance matching Leave-One-Out CV for map validation. *Methods in Ecology and Evolution*, 13, 1304–1316.
- Wadoux, A. M. J.-C., et al. (2021). Spatial cross-validation is not the right way to evaluate map accuracy. (critique). *Machine Learning* / *Ecological Modelling*.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*, 17, 1217–1241.
- Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *JASA*, 89(428), 1303–1313.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Gneiting & Raftery (2007) *JASA* — proper scoring (CRPS).
