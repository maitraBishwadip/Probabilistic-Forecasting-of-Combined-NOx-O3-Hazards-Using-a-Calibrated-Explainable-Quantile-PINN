# Experiment Design & Architecture

**Project:** Regime-conditioned 24–72 h forecasting of compound PM2.5–O3 extremes over Bangladesh
**Data:** `BD_DOE_2014-16.csv` — 9 DoE stations, hourly, 2014–2016 (223,776 rows)
**Code:** `pipeline/01_build_dataset.py`, `02_train_eval.py`, `03_pinn.py`
**Artefacts:** `modelling_daily.csv`, `artefacts/artefacts.joblib`, `results_gbt.json`, `results_pinn.json`, `figs/`

---

## 1. Problem formulation

A **compound extreme** is a station-day on which **both** the daily-mean PM2.5 and the daily-max 8-h O3 exceed their **season- and station-specific 75th percentile**. Given everything observable up to day *t*, predict

> `P(compound extreme on day t+h)`, for **h = 1 (24 h, primary), 2 (48 h), 3 (72 h)**.

This is a **probabilistic binary forecast** under heavy class imbalance (positive rate ≈ 8.8% overall).

### Why daily, not hourly
At hourly resolution PM2.5 and O3 are **diurnally anti-phased** (PM peaks at night, O3 at midday), so the same-hour co-occurrence lift is < 1 in every regime — an artefact of the diurnal cycle, not of synoptic coupling. Aggregating to **daily-mean PM2.5 + daily-max-8-h O3** removes the anti-phasing; at that resolution the pooled association is weakly positive (lift ≈ 1.2, Pearson r ≈ +0.09) and becomes strongly **regime-dependent** — which is the scientific object of study.

### Why 75th percentile
At the 90th-joint hourly threshold the positive class collapses to ~0.35%; at the **75th seasonal/daily** threshold it is ~8.8% (≈434 compound days), which is enough to train and to evaluate per-regime. Thresholds are a **modelling choice reported with sensitivity**, not a physical constant.

---

## 2. Data pipeline (`01_build_dataset.py`)

1. **Parse & type-coerce** all pollutant/met/satellite columns (whitespace → NaN).
2. **QC — physical range:** per-variable plausibility clip (e.g. PM2.5∈[0,1500], O3∈[0,300] ppb, RH∈[0,100]); out-of-range → NaN (counts logged in `qc_report.json`).
3. **QC — flat-line:** ≥24 identical consecutive positive values (stuck sensor) → NaN, per station.
4. **8-h O3** rolling mean (min 6 valid h) → later daily max.
5. **Daily aggregation** per station: met = mean (precip = sum); PM2.5 = mean; O3 = max-8h; gases = mean; satellite = mean.
6. **Completeness gate:** daily PM2.5 needs ≥18 valid hours; daily O3 needs ≥12 — else the day's value is NaN (no fabricated targets).
7. **Calendar reindex:** every station reindexed to a continuous daily calendar so lags/leads are true **calendar** offsets.
8. **Weather regimes (fit on TRAIN years 2014–15 only):** standardise the 9-D regime vector → **K-means (k = 5)**; assign all years by `predict`; name clusters from centroids (Stagnant-Trapping, Dry-Sunny-Photochemical, Humid-Transition, Monsoon-Wet-Windy, Ventilated-Stormy).
9. **Compound labels (TRAIN-only thresholds):** season×station 75th percentiles fit on 2014–15, applied to all years → `pm_ex`, `o3_ex`, `compound`. **No test-set leakage.**
10. **Feature engineering** (see §4) and **leads** `y24/y48/y72`, plus next-day concentrations for the PINN.

**Regime feature vector:** temp, RH, wind speed, solar radiation, precipitation, boundary-layer height, ventilation coefficient, photochemical-activity index, surface pressure.

---

## 3. Splits & leakage control

| Axis | Train | Test | Purpose |
|---|---|---|---|
| **Temporal (primary)** | 2014–2015 | **2016** (held-out year) | realistic operational forecast skill |
| **Spatial (LOSO)** | 8 stations | the 9th (all years) | spatial transfer / generalisation |

- Percentile thresholds, the StandardScaler, and the K-means regime model are **all fit on training data only**.
- Feature standardisation (PINN) uses **train** mean/std.
- The F1 decision threshold is tuned on **train** predictions and applied unchanged to test.
- Note a genuine **distribution shift**: 2016 has a higher compound rate (≈13.8%) than 2014–15 (≈7.5%). PR-AUC is base-rate dependent, so models are compared **within the same test set**.

---

## 4. Features (known at forecast time *t*)

- **Today's** pollutants (PM2.5, O3-8h, NO2, NOx, CO, SO2, PM10) and meteorology/reanalysis (temp, RH, wind, solar, precip, BLH, ventilation coeff, photochemical index, pressure, u/v, geopotential heights), plus daily MODIS AOD (+ a missingness flag).
- **Lags** (t-1, t-2, t-3) of PM2.5, O3 and key met; **rolling** 3/7-day means of PM2.5/O3; 3-day precip sum.
- **Calendar:** day-of-year sin/cos, weekday, season (one-hot).
- **Regime:** one-hot of today's regime (and the continuous ventilation/photochemical indices it derives from).
- **Persistence states:** today's `pm_ex`, `o3_ex`, `compound`.
- **Station:** one-hot (GBT) / location lat-lon + one-hot (PINN).

---

## 5. Models

### 5.1 Baselines (the bar to beat)
- **Persistence** — today's compound state carried to t+1.
- **Climatology** — train base rate by (season × regime).
- **Independent-marginals (GBT × GBT)** — one HistGBM predicts `P(PM2.5 extreme_{t+1})`, another `P(O3 extreme_{t+1})`; compound score = product. A **strong** baseline (treats pollutants one at a time, assumes conditional independence).

### 5.2 Gradient boosting (`HistGradientBoostingClassifier`)
NaN-native, `class_weight="balanced"`, depth 4, 400 trees, lr 0.05, L2 = 1, early stopping. Two variants:
- **GBT joint + regime** (full feature set, direct compound target).
- **GBT no-regime** (regime one-hots removed) — ablation isolating the regime label's marginal value.

### 5.3 PINN (`03_pinn.py`, PyTorch)
**Multitask, physics-informed network.**

```
x ──► trunk MLP [d→96→64, ReLU, dropout]
          ├─► concentration head ─softplus─► [ĉ_PM(t+1), ĉ_O3(t+1)]   (>=0)
          └─► classification head ─────────► logit  P(compound_{t+1})
```

**Embedded physics — daily box mass-balance (regime-indexed, learnable):**
```
ĉ_phys,PM = c_PM(t)·exp(−(a_r·VC + b_r·P + d_r)) + q_r
ĉ_phys,O3 = c_O3(t)·exp(−(a_r·VC + d_r))         + p_r·photochem
        with  a_r,b_r,d_r,p_r,q_r = softplus(θ_r) ≥ 0,  per regime r
```
where `VC` = ventilation coefficient (dilution), `P` = precipitation (wet removal), `photochem` = photochemical-activity index (O3 production). Concentrations are scaled to O(1).

**Composite loss** (physics weight annealed 0→0.5 over training):
```
L = focal_BCE(logit, compound)        # imbalance-aware classification
  + 0.3 · MSE(ĉ, c_obs(t+1))          # data fit on concentrations (where observed)
  + λ_phys · MSE(ĉ, ĉ_phys)           # physics consistency (box-model residual)
```
The auxiliary concentration task + physics residual regularise the shared trunk; the **learned per-regime parameters** `{a,b,d,p,q}` are an interpretable output (deposition/dilution vs O3 production by regime).

**Ablations:** `PINN_full` vs `PINN_no_physics` (λ_phys=0) vs `PINN_no_regime` (global, non-regime physics params).

---

## 6. Evaluation protocol

- **Primary metric: PR-AUC (Average Precision)** — appropriate under imbalance; reported with **1,000-sample bootstrap 95% CI**.
- **Secondary:** ROC-AUC, Brier (raw + isotonic-calibrated), F1 at train-tuned threshold, **recall at precision ≥ 0.3**, reliability diagram.
- **Per-regime PR-AUC** on the test year — does the model fire in the high-burden Dry-Sunny-Photochemical regime?
- **Lead-time degradation:** 24 → 48 → 72 h.
- **LOSO PR-AUC** (mean ± std over 9 station folds) for spatial transfer.
- **Ablations:** ± regime (GBT), ± physics / ± regime (PINN), joint vs independent-marginals, 75th vs (reported) threshold sensitivity.

**Figures:** `figs/pr_curves_gbt.png`, `figs/pr_curves_pinn.png`, `figs/reliability_gbt.png`.

---

## 7. Reproducibility
Fixed seeds (numpy / torch / sklearn). Thresholds, scaler and K-means persisted in `artefacts/artefacts.joblib`. Each stage writes a JSON (`qc_report.json`, `results_gbt.json`, `results_pinn.json`) consumed verbatim by `results.md`. Run order: `01 → 02 → 03`.
