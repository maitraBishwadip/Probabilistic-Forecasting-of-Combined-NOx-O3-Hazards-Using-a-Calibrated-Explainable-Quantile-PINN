# Methods — Explaining the QR-PINN Predictions (E9)

**Purpose.** Paper-ready methods write-up for the **explainability** stage of the QR-PINN combined
NOx–O3 study. It states *exactly* what is done and *why*, with equations and citations, so the
manuscript's "Model explainability / physics faithfulness" subsection can be assembled directly from
here. All numbers go to `RESULTS_LOG.md`; this file is **method, not results** (`CLAUDE.md §1`).

---

## 1. Goal and what is being explained

We forecast the rank/quantile-uniform combined hazard index
$H_t=\tfrac12 F_{\mathrm{NOx}}(\mathrm{NOx}_t)+\tfrac12 F_{\mathrm{O_3}}(\mathrm{O_3}_t)\in[0,1]$
(E5; `EXPERIMENT_DESIGN.md §2`). The **subject of explanation** is the *frozen, already-trained
calibrated reference model* `artefacts/qrpinn_full_rank_ext.pt` (rank index, physics on, extended
13-τ grid; trained in E6). **Nothing is retrained**; explainability is strictly post-hoc, so it can
introduce no leakage and changes no locked decision.

A QR-PINN differs from a point forecaster in two ways that shape the analysis:

1. It emits the **entire predictive distribution** of $H_{t+h}$ (a dense set of non-crossing
   conditional quantiles). We therefore explain **three different functionals** of that distribution,
   not one number: the **median** (central forecast), the **upper tail** $Q_{0.95}$ (the
   severe/hazardous region), and the **interval width** $Q_{0.95}-Q_{0.05}$ (predictive
   *uncertainty*).
2. It carries **embedded NOx–O3 photochemistry** (`§3`). So explanation doubles as a
   **physics-faithfulness audit**: does the learned input→output map rely on the drivers the
   chemistry says it should (photolysis/temperature for O₃ production, NOx for titration,
   ventilation/BLH for dispersion, precipitation for wet removal), and does the balance flip
   **day vs night** as the diurnal NOx–O3 anti-phasing predicts?

The model input is the length-$L{=}48$ h window $x\in\mathbb R^{L\times F}$ of $F{=}26$ named
features (`qrpinn_meta.json:feat_names`) plus an integer **station id** (an 8-D embedding). The
network is $g_{\tau,h}(x,\mathrm{stn})=\hat q_\tau(H_{t+h}\mid x,\mathrm{stn})$.

---

## 2. Method 1 — Integrated Gradients (axiomatic attribution)

**Integrated Gradients (IG)** (Sundararajan, Taly & Yan 2017) attributes a scalar network output to
its inputs by integrating gradients along the straight line from a **baseline** $x'$ to the input
$x$. For input cell $j=(\ell,f)$ (lag $\ell$, feature $f$) and target scalar $g$,

$$\mathrm{IG}_j(x)=(x_j-x'_j)\int_{0}^{1}
\frac{\partial g\big(x'+\alpha(x-x')\big)}{\partial x_j}\,d\alpha
\;\approx\;(x_j-x'_j)\,\frac1M\sum_{m=1}^{M}
\frac{\partial g\big(x'+\tfrac{m-1/2}{M}(x-x')\big)}{\partial x_j},$$

a midpoint Riemann sum with $M{=}32$ steps. IG satisfies **Sensitivity** and **Implementation
Invariance** and, crucially, **Completeness**: $\sum_j \mathrm{IG}_j(x)=g(x)-g(x')$. We report the
mean relative completeness gap $\frac{|\sum_j \mathrm{IG}_j-(g(x)-g(x'))|}{|g(x)-g(x')|}$ as an
internal validity check (small ⇒ the attribution is faithful and $M$ is large enough). The station
embedding is held at the sample's true station along the path — IG attributes the **26 time-varying
features**; the station's role is quantified by occlusion (§3).

**Baseline = "missing / no information".** $x'$ sets every value channel to $0$ (the scaler median
for robust-scaled pollutants/AOD, the mean for standardized meteorology) and every `*_mask` channel
to $0$ (= *missing*). This is precisely the input the network sees for an unobserved hour, so the
baseline is the model's own "I know nothing" state and IG measures how the *actual* 48 h of evidence
moves each functional away from that state. (A zero baseline is also the conventional, axiom-friendly
choice.)

**Attributed functionals** (target $g$), each at the stated lead $h$:

| name | $g$ | lead | what it explains |
|---|---|---|---|
| `median_h1`  | $Q_{0.5}(H_{t+1})$              | 1  | central next-hour hazard |
| `median_h24` | $Q_{0.5}(H_{t+24})$            | 24 | central day-ahead hazard (driver shift with lead) |
| `tail_h1`    | $Q_{0.95}(H_{t+1})$            | 1  | the **severe/hazardous** upper tail |
| `width_h1`   | $Q_{0.95}(H_{t+1})-Q_{0.05}(H_{t+1})$ | 1 | **predictive uncertainty** (spread) |

**Aggregations.** From per-cell $\mathrm{IG}_{\ell,f}$ over a sample of anchors we report:
- **Per-feature importance** $\;\phi_f=\frac1N\sum_n\sum_\ell |\mathrm{IG}^{(n)}_{\ell,f}|$ (magnitude)
  and the **signed** mean $\bar\phi_f=\frac1N\sum_n\sum_\ell \mathrm{IG}^{(n)}_{\ell,f}$ (direction),
  normalized to fractions of total $|\mathrm{IG}|$.
- **Per-lag (temporal) saliency** $\;\psi_\ell=\frac1N\sum_n\sum_f |\mathrm{IG}^{(n)}_{\ell,f}|$ over
  $\ell=-47..0$ h — reveals reliance on recent hours (persistence) vs the $\sim$24 h-ago diurnal lag.
- **Per physics-group importance** — $\phi_f$ summed within the §1.3 groups below.

**Sampling.** IG needs a backward pass per path step, so it is computed on a **stratified sample**
of test-2016 anchors (target $N\approx2500$), balanced across {hazard $H_{t+1}\!\ge\!\mathrm{thr}$ vs
not} × {day vs night}, with a fixed seed and the chosen indices logged. Occlusion (§3) uses the full
test set.

---

## 3. Method 2 — Permutation / occlusion importance (model-agnostic, loss currency)

Gradient attributions are local and can be unstable on sequences (§5); we therefore cross-check with a
global, model-agnostic measure in the model's **own scoring currency**, in the spirit of
permutation variable importance (Breiman 2001; formalized by Fisher, Rudin & Dominici 2019). For
feature (or group) $c$, we replace its window values across **all test-2016 anchors** with the
**train-period mean** of that channel $\mu_c$ (mean-occlusion — a neutral, in-distribution value) and
recompute the headline scores:

$$\Delta\mathrm{Pinball}_c=\mathrm{Pinball}(\text{occlude }c)-\mathrm{Pinball}(\text{intact}),$$

and likewise $\Delta\mathrm{CRPS}_c$ and $\Delta\mathrm{Brier}_c$ on the exceedance event
$\{H_{t+h}\ge\mathrm{thr}\}$. A **positive** $\Delta$ means the feature was *helpful* (removing it
hurt the forecast). We report per-feature and per-**group** occlusion (groups occlude all member
channels jointly, capturing correlated inputs such as NO/NO₂/NOx), and a **station** permutation
(replace the station id with a fixed reference id) to size the embedding's contribution. The intact
scores reproduce the E5/E6 raw metrics as a sanity check that the checkpoint loaded correctly.

**Physics groups** (the §3 chemistry blocks; `feat_names` in brackets):

| group | features |
|---|---|
| `NOx_side`       | NO, NO_mask, NO2, NO2_mask, NOx, NOx_mask |
| `O3_side`        | O3, O3_mask |
| `photochem`      | solar_rad_Wm2, photochemical_activitiy_index, temp_C |
| `dispersion`     | boundary_layer_height_m, ventilation_coefficient, wind_speed_ms, wind_speed_u_ms, wind_speed_v_ms |
| `wet_removal`    | precip_mm, RH_pct |
| `pressure`       | surface_pressure_hPa |
| `satellite`      | modis_aod_550nm, modis_aod_550nm_mask |
| `calendar`       | hour_sin, hour_cos, doy_sin, doy_cos, weekday |
| `station`        | station embedding (occlusion only) |

---

## 4. Physics-faithfulness audit & day/night contrast

Combining §2–§3 we test the §3 narrative explicitly:

1. **Driver plausibility.** The NOx side of $H$ should lean on NOx-side lags; the O₃/tail side should
   lean on **photochemistry** (solar, photochemical index, temp) and **dispersion** (BLH, ventilation),
   with **wet removal** active in the monsoon. Reported as group rankings of $\phi$ and $\Delta$Pinball
   for the four functionals.
2. **Tail vs centre vs spread.** Compare group importance for `median_h1`, `tail_h1`, `width_h1`:
   does the **tail** (severe O₃-driven oxidant) load more on photochemistry, and does **uncertainty**
   (`width`) rise with missingness (mask channels) and meteorological volatility?
3. **Lead-time shift.** `median_h1` vs `median_h24`: short lead should be **persistence-dominated**
   (recent pollutant lags), long lead should shift toward **meteorology/calendar** (diurnal climatology).
4. **Day vs night.** Split the IG anchors by anchor local hour (day = solar-on hours; night otherwise)
   and compare group $\phi$: photochemistry/O₃ formation should dominate **by day**, NOx titration
   **at night** — the diurnal anti-phasing the PINN is meant to encode.

If the attributions match this chemistry, the explainability is positive evidence that the PINN learned
physically faithful structure (not just curve-fitting); where they disagree, that is reported honestly
as a finding, not hidden.

---

## 5. Validity caveats (stated honestly in the paper)

- **Saliency on time series is fragile.** Ismail et al. (2020) show many saliency methods degrade on
  time-series models (feature *and* time importance entangled). Mitigation: (i) IG is **axiomatic** and
  we report its **completeness gap**; (ii) we **cross-validate** IG against model-agnostic occlusion in
  loss currency and treat **agreement in sign and ranking** as the robustness claim; single-method
  conclusions are avoided.
- **Correlated inputs.** NO/NO₂/NOx and the wind components are correlated, so single-feature occlusion
  under-states grouped importance; we therefore report **group** occlusion as the primary global measure.
- **Baseline dependence.** IG attributions are relative to the missing-input baseline; we state this and
  use a baseline that is meaningful for this model (the gap state) rather than an arbitrary one.
- **Locality.** IG explains the model around each anchor; we summarize over a stratified sample and
  report dispersion, not just means.

---

## 6. Results tables to fill (schema — populated only from `RESULTS_LOG.md`)

**T1 — Per-group importance (frozen rank-index QR-PINN, test-2016).**

| Group | IG `median_h1` | IG `tail_h1` | IG `width_h1` | ΔPinball (occlusion) | ΔBrier(exc) |
|---|---|---|---|---|---|
| NOx_side / O3_side / photochem / dispersion / wet_removal / pressure / satellite / calendar / station | … | … | … | … | … |

**T2 — Top-k features** by IG (median) and by occlusion ΔPinball (agreement check); report the
**IG-vs-occlusion group-level Pearson $r$** as a single robustness number.
**T3 — Day vs night** group importance (photochem/O₃ vs NOx).

**Figures (publication-grade — 300-dpi PNG + vector PDF, Okabe–Ito colour-blind-safe palette, fixed
per-group colours, trimmed spines, panel letters):**
`e9_group_importance` — (a) per-group IG bars, (b) IG-vs-occlusion agreement scatter with $r$;
`e9_distributional` — per-group importance for centre / severe tail $Q_{0.95}$ / interval width;
`e9_temporal_heatmap` — group × lag (−47…0 h) IG saliency with the −24 h diurnal marker;
`e9_daynight` — day↔night dumbbell per group; `e9_leadtime` — per-group IG vs lead $h\in\{1,6,12,24\}$;
`e9_feature_importance` — supplementary top individual features (IG ∥ occlusion).

---

## 7. Reproducibility
Fixed seeds (numpy/torch); frozen checkpoint `artefacts/qrpinn_full_rank_ext.pt`; baseline definition,
$M{=}32$ IG steps, sampled-anchor indices, and occlusion fills ($\mu_c$) logged; no imputation of
targets; observed $H$ only; the run block records environment/package versions.

---

## References (this stage; full list in `LITERATURE_REVIEW.md`)
- Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks.* ICML, PMLR 70:3319–3328. (Integrated Gradients.) https://proceedings.mlr.press/v70/sundararajan17a.html
- Breiman, L. (2001). *Random Forests.* Machine Learning 45:5–32. (Permutation importance.)
- Fisher, A., Rudin, C., & Dominici, F. (2019). *All Models are Wrong, but Many are Useful: …Model Reliance.* JMLR 20(177):1–81. https://jmlr.org/papers/v20/18-760.html
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS 30.
- Ismail, A. A., Gunady, M., Corrada Bravo, H., & Feizi, S. (2020). *Benchmarking Deep Learning Interpretability in Time Series Predictions.* NeurIPS 33:6441–6452. https://arxiv.org/abs/2010.13924
- Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.). https://christophm.github.io/interpretable-ml-book/
- Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules…* JASA 102:359–378. (ΔPinball/ΔCRPS are proper-score deltas.)
