# Research Plan — Regime-Conditioned 24-h Forecasting of Compound PM2.5–O3 Extremes over Bangladesh

**Author:** Bishwadip Maitra · UW–BUET Air Quality Project
**Target venue:** IEEE IGARSS (remote-sensing-forward framing)
**Dataset:** `BD_DOE_2014-16.csv` — 9 DoE stations, hourly, 2014-01-01 → 2016-12-31

---

## 0. One-line objective

Forecast, at each issue time, the **probability that a *compound* PM2.5–O3 extreme occurs within the next 24 h** (extended to 48/72 h), by learning how meteorological **regimes** modulate the joint tail behaviour of two pollutants — and show this beats single-pollutant and regime-agnostic baselines, robustly **across regimes and across stations**.

---

## 1. What the data actually says (profiling results — already run)

| Aspect | Finding | Consequence for the plan |
|---|---|---|
| Coverage | 9 stations × hourly × 3 yr (~24.9k rows/station, 223,776 total) | Enough for DL; pool stations for positives |
| Meteorology + reanalysis | **100% populated** (temp, RH, wind, solar, precip, MSLP, **BLH, ventilation_coefficient, photochemical_activity_index**, geopotential heights) | Regime layer is fully usable; it is the backbone |
| Pollutants | O3 74.7%, CO 71.5%, PM2.5 68.1%, PM10 64.5%, NOx 62.6%, SO2 59.6%, NO2 57.1% | Need QC + gap-fill; models must tolerate missingness |
| MODIS (AOD/NDVI/fire) | ~4.2% (daily overpass) | Use at **daily** resolution as merged features, not hourly |
| OMI / VIIRS columns | ~0.1% | **Too sparse for hourly modelling** — drop or re-derive daily from source |
| PM2.5–O3 linear corr | r = **−0.10** overall, **−0.26 winter** | They *anti*-correlate linearly → compound extreme is a **tail-co-occurrence** problem, not a regression-correlation one |
| Compound-day rate (seasonal 95th joint) | ~**0.2%** of station-days; peaks **pre-monsoon (0.48%)**, lowest winter (0.05%) | Severe class imbalance → imbalance-aware design + relaxed/tunable thresholds |

> **Reframing insight:** the highest PM2.5 occurs in the *Winter Stagnant* regime, but the highest O3 and the highest *compound* rate occur in the *Dry-Sunny-Photochemical* regime (pre-monsoon). Compound extremes are a **photochemistry-meets-residual-aerosol** phenomenon, not a deep-winter one. This is a headline result, not just a method.

---

## 2. Pollutant-pair selection (the "at most 2 pollutants")

**Chosen pair: PM2.5 (primary) and O3 (secondary).**

Rationale, grounded in the data and the venue:
- It is the **canonical compound-pollution pair** in the literature (Lyu et al. 2024) and the only pair with a clear **secondary-vs-primary, meteorology-mediated** mechanism — exactly what regimes explain.
- The negative linear correlation is **the scientific hook**, not a problem: PM2.5 and O3 *rarely* peak together, so the days they *do* are regime-selected and high-value to forecast.
- **Remote-sensing alignment (IGARSS):** PM2.5 ↔ **MODIS AOD 550 nm** (column aerosol proxy); O3 photochemistry ↔ **photochemical_activity_index** + solar radiation (and OMI HCHO/NO2 where re-derivable). Both targets have a satellite-observable footprint.

*Robustness alternative (for an appendix/ablation only):* PM2.5–NO2 (r=+0.24) or PM2.5–CO (r=+0.32) as a positively-correlated combustion pair, to show the framework is pollutant-agnostic.

---

## 3. Phase 1 — QC, gap-filling, feature engineering

**3.1 Quality control (per station, per pollutant)**
- Physical-range clip (e.g., PM2.5 ∈ [0, 1000], O3 ∈ [0, 300] ppb, RH ∈ [0,100]).
- Flat-line / stuck-sensor detection (N identical consecutive values).
- Spike test (rolling-MAD / Hampel filter), cross-pollutant sanity (e.g., PM2.5 ≤ PM10).
- Flag, don't silently overwrite; keep a QC bitmask column.

**3.2 Gap-filling protocol (documented, defensible)**
- Short gaps (≤3 h): time interpolation.
- Medium gaps (≤24 h): diurnal-climatology + interpolation blend.
- Long gaps: **leave as NaN and flag**; never impute the *target*.
- DL path uses **masking**; GBT path uses **native NaN handling** — so heavy imputation is avoided.

**3.3 Feature engineering**
- **Lags** of PM2.5, O3, and key met: 1, 3, 6, 12, 24 h; **rolling** mean/max/std (6/24 h); **rate-of-change**.
- **Wind**: u, v already present; add sin/cos(wind_dir).
- **Calendar/diurnal**: hour, day-of-year (sin/cos), weekday, season, Bangladeshi holidays.
- **Regime descriptors** (Sec. 5) as features.
- **Daily satellite merge**: MODIS AOD (forward-fill within day + daily-mean feature), NDVI, FIRMS fire count/FRP.
- **Persistence helpers**: most-recent observed PM2.5/O3 and their current exceedance state.

---

## 4. Phase 2 — Quantifying a pollutant extreme & the compound label

**4.1 Per-pollutant extreme threshold (seasonal + station-aware)**
- Define seasons (Bangladesh): Winter (Dec–Feb), Pre-monsoon (Mar–May), Monsoon (Jun–Sep), Post-monsoon (Oct–Nov).
- Threshold = **seasonal percentile** computed *within each station × season* so winter PM2.5 (median ≈114, 95th ≈295 µg/m³) does not swamp other seasons and O3's seasonal cycle (95th ≈34.5 ppb) is respected.
- Health-relevant aggregation: **daily-mean PM2.5** and **daily max-8 h O3** (use `O38hr_ppb`/`CO8hr_ppm` where present, else compute rolling 8-h).
- Report results at **both 90th and 95th** thresholds (sensitivity), because at 95th-joint the positive class is only ~0.2% of station-days.

**4.2 Compound-extreme label**
- `compound = (PM2.5 ≥ thr_PM) AND (O3 ≥ thr_O3)` at the same day (and an hourly variant).
- Quantify co-occurrence beyond a binary flag:
  - Conditional probabilities `P(O3 ext | PM2.5 ext)`, `P(PM2.5 ext | O3 ext)` by season/hour/regime.
  - **Tail-dependence coefficient** χ (and a fitted copula) — the correct statistic when linear r is negative but the tails still co-occur.

**4.3 Forecast target (this is what the model predicts)**
- For each issue time *t*: binary `y = 1` if a compound extreme occurs anywhere in `(t, t+24h]` (sliding-window "any-occurrence"). This multiplies positives vs. a single-day label and matches "next 24 h" operationally.
- Train separate heads / horizons for **24 / 48 / 72 h**.
- Class imbalance mitigations: class weights / focal loss, threshold tuning on validation, and **report PR-curve-based metrics** (Sec. 7), never bare accuracy.

---

## 5. Phase 3 — Defining & quantifying weather regimes

Two complementary, cross-validating definitions. **The user's mental model ("humid + airflow + sunlight", "humid + rain", …) is captured directly by the regime feature vector below.**

**5.1 Regime feature vector (standardised, daily; hourly variant for diurnal regimes)**
`[temp, RH, wind_speed, solar_rad, precip, BLH, ventilation_coefficient, photochemical_activity_index, surface_pressure, (geopotential 850/500), stagnation_index]`
— i.e. the four axes the user named: **humidity (RH), airflow (wind/ventilation/BLH), sunlight (solar/photochemical index), and rain (precip)**, plus stability.

**5.2 Data-driven regimes (primary)**
- **K-means** on the standardised vector; choose *k* by **silhouette + elbow**; **cross-check with a Self-Organising Map** and with **Gaussian Mixture** (GMM gives *soft* membership probabilities used as model inputs).
- Already validated on this dataset at *k = 5* — recovered, with interpretable centroids:

| Regime | Signature (centroid) | PM2.5 | O3max | Compound rate | "User-language" |
|---|---|---|---|---|---|
| **R1 Winter Stagnant–Trapping** | cool 21°C, wind 6.2, BLH 346, **ventilation 618 (low)**, dry, MSLP 1013 | **126** | 22 | 0.1% | low airflow + low sun |
| **R2 Dry-Sunny Photochemical** | 28°C, **RH 65 (low)**, solar 281, **photochem 8927 (high)**, dry | 83 | **26.5** | **1.0%** | dry + strong sunlight + moderate airflow |
| **R3 Humid Transition** | 28°C, RH 83, wind 7, solar 212 | 43 | 16 | 0.1% | humid + sun |
| **R4 Monsoon Wet–Windy** | 26°C, **RH 91**, wind 10.4, **precip 28 mm**, ventilation 1399 | 28 | 13 | 0.2% | humid + rain + airflow |
| **R5 Ventilated–Stormy** | **wind 16.2**, **BLH 716**, **ventilation 3376 (high)** | 27 | 13 | 0.0% | strong airflow (flush-out) |

**5.3 Rule-based regime index (interpretable + transferable — IGARSS value)**
Parallel, threshold-based labels for explainability and transfer to stations without 3 yr of history:
- **Stagnation flag** = (wind < p25) ∧ (precip ≈ 0) ∧ (BLH < p25).
- **Ventilation class** = tertiles of `ventilation_coefficient (= BLH × transport wind)`.
- **Photochemical class** = tertiles of `photochemical_activity_index` (or solar × temperature).
- **Wet class** = precip > 1 mm.
Regime = the cross of these → maps 1:1 to the clusters above and is auditable.

**5.4 How a regime is "quantified" (deliverables of this phase)**
- Each timestamp → (a) hard cluster label, (b) **soft membership vector** (GMM), (c) continuous descriptors (ventilation, photochemical, stagnation indices). All three feed the forecaster (Sec. 6).
- Each regime → frequency, seasonal calendar, mean pollutant load, **compound-extreme rate**, and persistence/transition matrix (how long regimes last, what they transition into — relevant for lead time).

---

## 6. Phase 4 — Modelling the interaction & forecasting

**6.1 Baselines (must beat all of these)**
1. **Persistence** — current exceedance state carried forward.
2. **Seasonal climatology** — base rate by season × hour × regime.
3. **Single-pollutant** GBT predicting PM2.5-only and O3-only exceedance (shows the *compound* value-add).
4. **Regime-agnostic** version of the main model (ablation of the whole contribution).

**6.2 Primary models**
- **Gradient-boosted trees (LightGBM / XGBoost)** on the engineered tabular features — native NaN handling, fast, strong on tabular, and **SHAP-explainable per regime** (key for the science story). This is the workhorse.
- **Sequence DL** for temporal dynamics: a **Temporal Convolutional Network (TCN)** or **LSTM** (lightweight **Transformer** as a stretch) over a 72-h multivariate lookback window with a masking layer for gaps. Multi-horizon (24/48/72 h) heads.

**6.3 Injecting the regime (the core idea — three options, evaluated)**
- (a) **Concatenation**: regime soft-membership + descriptors as extra inputs (simple, strong).
- (b) **Mixture-of-experts / per-regime heads**: a gating on regime membership; lets the model use different physics in the stagnant vs photochemical regime.
- (c) **Regime-stratified training** with shared backbone + regime-specific calibration.

**6.4 Physics-guided option (differentiator, optional second paper-section)**
Soft penalty encouraging consistency with a boundary-layer mass-balance / **ventilation prior** (concentration ↓ as `BLH × wind` ↑ at fixed emission) and the **photostationary O3–NO–NO2** relation. Implemented as an added loss term on the DL model (a light **physics-informed** regulariser). Ties the work to the satellite/reanalysis physical fields and strengthens generalisation where data is sparse.

**6.5 Probabilistic output & calibration**
- Output calibrated probabilities; apply **isotonic / Platt** calibration on validation; report **reliability diagrams**. Operationally, expose tunable warning thresholds (high-recall vs high-precision modes).

---

## 7. Phase 5 — Rigorous, regime-aware evaluation

**7.1 Splits (no leakage)**
- **Temporal:** train 2014–2015, **test 2016** (held-out year). Hyper-tune with **blocked / expanding-window time-series CV** inside the training years.
- **Spatial:** **Leave-One-Station-Out (LOSO)** cross-validation — train on 8 stations, test on the 9th → the **transferability** claim IGARSS reviewers want.

**7.2 Metrics (imbalance-aware — accuracy is banned)**
- **PR-AUC (primary)**, ROC-AUC, **F1 / recall at fixed precision**, **Brier score**, reliability diagram, and **Critical Success Index / hit-rate–false-alarm** (forecasting-standard).
- Lead-time degradation curves (24 vs 48 vs 72 h).

**7.3 The headline test — per-regime skill**
Stratify every metric **by regime**. The model is only useful if it catches the **R2 Dry-Sunny-Photochemical** compound events (the 1.0% regime). Report a per-regime PR-AUC table and confusion of *which regime* the missed events fell in.

**7.4 Ablations (isolate each contribution)**
- − regime conditioning · − satellite features · − sequence model (GBT only) · single- vs compound-target · 90th vs 95th threshold · − physics term.

**7.5 Statistical rigour**
- **Bootstrap 95% CIs** on all metrics; **DeLong test** for AUC differences vs baselines; **permutation test** for skill > climatology. Report mean ± CI, not point estimates.

**7.6 Interpretability**
- **SHAP** globally and **within each regime** → show *which* met drivers flip a compound extreme on (e.g., photochemical index + residual AOD in R2 vs ventilation collapse in R1). This is the scientific payload.

---

## 8. Remote-sensing / IGARSS framing (make the satellite angle central)

- **PM2.5 ↔ MODIS AOD 550 nm**: use AOD as a daily PM proxy feature and to **spatially extend** point regimes; discuss AOD–PM2.5 humidity dependence (RH from the met data corrects hygroscopic growth).
- **O3 photochemistry ↔ photochemical_activity_index + solar**, with **OMI HCHO/NO2** (re-derived at daily resolution) for the VOC-/NOx-limited regime where feasible.
- **FIRMS fire / FRP** for biomass-burning episodes; **ERA5 BLH + winds** as the reanalysis backbone of the regime definition.
- Narrative: *ground sensors are sparse and gappy; satellite + reanalysis make the regime layer spatially continuous and transferable* — the IGARSS contribution.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Compound positives very rare at 95th | Use 90th joint + hourly "any-in-24h" target + pool 9 stations; report threshold sensitivity |
| Satellite columns (OMI/VIIRS) ~0.1% | Drop from hourly model; re-derive at daily resolution from source for context only |
| Pollutant gaps 25–43% | Masking (DL) + native NaN (GBT); never impute target; QC bitmask |
| Station heterogeneity | Station embedding + LOSO evaluation; rule-based regimes for transfer |
| Overfitting rare class | Class weights/focal loss, calibration, bootstrap CIs, held-out year |

---

## 10. Deliverables & timeline (toward IGARSS)

1. **Reproducible pipeline**: `01_qc.py → 02_features.py → 03_regimes.py → 04_labels.py → 05_models.py → 06_eval.py`.
2. **Regime atlas**: centroids, seasonal calendars, transition matrices, compound rates (Sec. 5).
3. **Forecast benchmark table**: models × horizons × regimes, with CIs.
4. **Figures**: regime map/centroids, per-regime PR curves, reliability diagrams, SHAP-by-regime, lead-time curves.
5. **Paper (IGARSS, ~4 pp)**: headline = first regime-conditioned compound PM2.5–O3 forecast for Bangladesh + the pre-monsoon photochemical finding.

**Suggested order:** QC/features (wk1–2) → regimes + labels + EDA (wk3) → baselines + GBT (wk4) → sequence/regime-conditioned + physics option (wk5–6) → evaluation + ablations + writing (wk7–8).

---

### Reproducibility notes
- Fix seeds; log configs; version the QC/threshold choices.
- Save the regime model and thresholds as artefacts so labels are deterministic.
- Keep all percentile thresholds **fit on training years only** and applied to test (no leakage).
