# Concrete Research Plan — Regime-Conditioned PINN Forecasting of Compound PM2.5–O3 Extremes & Their Health Burden over Bangladesh

**Author:** Bishwadip Maitra · UW–BUET Air Quality Project
**Venue:** IEEE IGARSS (remote-sensing-for-health framing)
**Data:** `BD_DOE_2014-16.csv` — 9 DoE stations, hourly, 2014–2016 (223,776 rows)
**Supersedes the scope/threshold choices in `research_plan.md`; that file keeps the detailed pipeline & evaluation protocol.**

---

## 0. Thesis (one paragraph)

PM2.5 and ground-level O3 are, in general, **anti-correlated** (this dataset: hourly r ≈ −0.10, winter −0.26; co-occurrence *lift* 0.58× below chance). Yet the days they spike **together** carry **synergistic — not additive — health harm**. The scientific question is therefore not "are they correlated" but **"under which meteorological *regime* does their joint tail dependence flip positive, and can we forecast and health-attribute those compound days?"** We answer this with a **regime-conditioned, physics-informed neural network (PINN)** that (i) embeds the dataset's own physical fields (boundary-layer height, ventilation coefficient, photochemical-activity index) as the terms of a coupled mass-balance, (ii) forecasts the probability of a compound PM2.5–O3 extreme at 24/48/72 h, and (iii) translates forecasts into **satellite-informed, regime-specific health-exposure burden** — the remote-sensing-for-health contribution.

---

## 1. Decisions locked for this study (and the data that backs them)

| Decision | Value | Evidence from `BD_DOE_2014-16.csv` |
|---|---|---|
| Pollutant pair | **PM2.5 + O3** | canonical compound pair; anti-correlated → tail-dependence problem |
| Extreme threshold | **75th percentile**, seasonal & per-station | gives a workable positive class (below) |
| Label resolution | **Daily** (daily-mean PM2.5, daily-max 8-h O3) | hourly co-occurrence is suppressed by **diurnal anti-phasing** (every regime lift < 1 hourly); daily removes it |
| Compound-day rate | **6.5%** (441 / 6,768 station-days) | enough positives for ML at 75th (vs 0.35% at 90th hourly) |
| Headline finding | Joint-extreme **lift > 1 only in the Dry-Sunny-Photochemical regime (1.10)** and pre-monsoon season (9.7%) | overall lift ≈ chance; regime-conditioning is what reveals dependence |

**Regime-conditioned joint behaviour (daily, 75th):**

| Regime | RH | solar | photochem idx | ventilation | compound-day % | **lift** |
|---|---|---|---|---|---|---|
| **Dry-Sunny Photochemical** | 65 | 281 | 8927 | 1564 | **17.6** | **1.10** |
| Humid Transition | 83 | 212 | 6370 | 900 | 5.8 | 1.03 |
| Monsoon Wet–Windy | 91 | 101 | 2816 | 1399 | 2.7 | 0.91 |
| Winter Stagnant (PM-rich) | 74 | 189 | 4705 | 618 | 6.7 | 0.84 |
| Ventilated–Stormy | 82 | 204 | 6261 | 3376 | 1.0 | 0.78 |

> **The contribution in one line:** *two pollutants that suppress each other on average become a compound hazard in a specific, forecastable, physically-interpretable regime — and that regime, not winter stagnation, is where Bangladesh's combined air-pollution health risk concentrates.*

---

## 2. Literature review (proper, thematic)

### 2.1 Compound PM2.5–O3 extremes & their synoptic drivers
The compound-extreme field is young and **geographically skewed to China and the US**.
- **Lyu et al. (2024, GRL)** — southern-China summer; >50% co-occurrence frequency; Random Forest links co-occurrence to typhoon-periphery and West-Pacific-Subtropical-High circulation and shared precursors (VOCs). The methodological anchor for ML co-occurrence prediction. [link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL106527)
- **Schnell & Prather (2017, PNAS)** — first to show **co-occurrence of O3, PM2.5 and temperature extremes** over eastern North America. [link](https://www.pnas.org/doi/10.1073/pnas.1614453114)
- **Kalashnikov et al. (2022, western US)** — co-occurring PM2.5–O3 extremes **rising ~25 million person-days/yr (2001–2020)**, driven by atmospheric **ridging**; explicit weather-pattern clustering. [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC8730618/)
- **YRD warm-season study (2024, ACS ES&T Air)** — Random Forest + **SHAP**; solar-radiation variables dominate during co-occurrence; **76.4%** of co-occurrences coincide with stagnation/heat extremes. [link](https://pubs.acs.org/doi/10.1021/acsestair.4c00314)
- **Large-scale synoptic drivers, eastern China** (Science China 2024; ACP 2021). [link](https://acp.copernicus.org/articles/21/9105/2021/)

**Gap:** no compound PM2.5–O3 study for **Bangladesh / the eastern IGP**; existing work is summer China/US, single-resolution, and does **not quantify regime-conditioned joint *lift*** the way this study does.

### 2.2 Weather-regime / circulation clustering for air pollution
- **K-means & Self-Organising-Map clustering** of geopotential/met anomalies into circulation regimes is the standard objective tool (western US ridging clusters; South-Korea synoptic clusters). [SOM case study](https://doi.org/10.3390/atmos17010044)
- **Zhou et al. (2024, Nat. Commun.)** — region-specific atmospheric **stagnation index** for the Indo-Gangetic Plain (blueprint to localise to the Bengal delta).
- **Kim et al. (2024, AAQR)** — K-means circulation patterns yielding an explicit **stagnation cluster** controlling high-PM2.5 days.
- **Chen et al. (2022)** — weather-pattern clustering for PM2.5/O3 co-occurrence.

**Gap:** regimes are usually defined on upper-air fields alone; here we cluster on **surface + reanalysis physics already in the dataset** (BLH, ventilation coefficient, photochemical index) and **profile each regime by joint-extreme lift**.

### 2.3 Physics-informed / physics-guided ML for air quality (the PINN basis)
A fast-growing line embeds the **advection–diffusion–reaction (ADR)** equation into neural nets:
- **AirPhyNet (2024, ICLR)** — RNN encoder + **GNN-based differential-equation network** representing advection & diffusion + decoder; learns transport physics for PM2.5. [link](https://arxiv.org/pdf/2402.03784)
- **CTENet** — ADR embedded in a PINN with an **Eulerian** representation; RMSE −45.8% (USA), −21% (China).
- **TransNet (2026, npj Clean Air)** — transport-informed GNN solving the **full ADR system** on a station graph; +72 h forecasts. [link](https://www.nature.com/articles/s44407-026-00052-x)
- **PINN for inverse advection–diffusion** source localisation under weak/strong wind (2025). [link](https://arxiv.org/pdf/2503.18849)
- **PINN improving WRF-CHEM with satellite remote-sensing data** (Sci. Total Environ. 2023) — *directly* the RS + physics combination this proposal uses. [link](https://www.sciencedirect.com/science/article/abs/pii/S1352231023004570)
- Dual neural-ODE open-system AQ (2024); NeuroDDAF evidential diffusion–advection (2026).

**Gap:** PINNs target **single-pollutant point/field prediction**. **No PINN forecasts a *compound* (joint two-pollutant) *extreme probability*, conditions on weather *regime*, or learns *regime-dependent* physical parameters.** That is this study's modelling novelty.

### 2.4 Remote sensing of PM2.5 & remote-sensing-for-health
- **AOD → surface PM2.5** is the established satellite exposure proxy and the backbone of **global mortality / GBD** estimates (van Donkelaar-type products); reduces exposure misclassification. [review](https://link.springer.com/article/10.5572/ajae.2020.14.4.319)
- **NASA — satellite monitoring for air quality & health** (Holloway review). [link](https://ntrs.nasa.gov/api/citations/20230000905/downloads/Holloway_review_satellite%20monitoring%20for%20air%20quality%20and%20health.pdf)
- AOD–PM2.5 relationship needs **RH (hygroscopic-growth) correction** — available in our met data.

**Gap:** satellite exposure work is **PM2.5-only**; a **combined PM2.5+O3 compound exposure** product for health does not exist for Bangladesh.

### 2.5 Why compound matters — PM2.5–O3 synergistic toxicity (health core)
- Multi-city (372 cities): combined exposure gives a **synergy index of 1.93** on total mortality (joint > sum of parts); **preterm-birth RR 3.63** for co-exposure vs 0.99 (PM) / 1.34 (O3) alone. [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC12030836/)
- Mechanisms: O3 **raises PM oxidative potential & hygroscopicity**; amplifies PM-induced inflammation (IL-6, IL-1β, TNF-α); elevated ROS / mitochondrial damage; barrier disruption.
- Bangladesh burden context: **World Bank (2022) "Breathing Heavy"**; Begum & Hopke source apportionment (brick kilns, biomass, traffic).

### 2.6 Bangladesh air-quality context
- PM2.5 peaks in **winter (up to ~284 µg/m³)**, ~5–6× monsoon, from inversions + brick kilns + reduced washout; **meteorology explains >77%** of PM variability; monsoon rain/wind flush pollutants. [link](https://link.springer.com/article/10.1007/s44274-026-00560-3)
- O3 is the **exception** — it does *not* peak in winter (low photochemistry) — consistent with our finding that **compound extremes move to pre-monsoon**.

---

## 3. The PINN — structure & equations

### 3.1 Physics already in the dataset (what makes a PINN feasible here)
The CSV ships the exact fields a coupled box/ADR model needs, all **100% populated**: `boundary_layer_height_m` (H), `ventilation_coefficient` (= H × transport wind ≡ dilution rate), `wind_speed_u/v` (advection), `solar_rad_Wm2` & `photochemical_activity_index` (photolysis driver J), `precip_mm` (wet scavenging), `temp_C`,`RH_pct` (reaction/hygroscopic), `surface_pressure`, geopotential heights (synoptic). **We do not estimate these — we plug them into the residual.**

### 3.2 Governing equations (coupled, two-pollutant box / ADR per station, graph-coupled across stations)
For pollutant concentration `C_k`, k ∈ {PM2.5, O3}, at station i:

```
∂C_k/∂t  =  E_k/H                      (emission into mixing layer of depth H)
          − (V_k/H)·C_k                (dilution/ventilation;  V = ventilation_coefficient/H ≈ transport wind)
          − (v_d,k/H)·C_k              (dry deposition, velocity v_d)
          − Λ_k·P·C_k                  (wet scavenging; P = precip)
          − (u·∇)C_k + ∇·(K ∇C_k)      (advection + turbulent diffusion; u from wind_u/v, K eddy diffusivity)
          + R_k(C, J, T, RH)           (chemistry / secondary formation)
```

Chemistry terms (the coupling that makes them a *joint* system):
```
R_O3   =  P(J, NOx, VOC) − k_titr·[NO]·C_O3 − loss        ;  J ∝ photochemical_activity_index / solar_rad
R_PM2.5 = f_sec(SO2, NOx, NH3, RH) + f_photo(J)·(secondary organic/inorganic) − loss
```
PM2.5 and O3 are coupled through **(a) shared meteorology (H, u, J)** and **(b) shared precursors / photochemistry (J, NOx, VOC)** — so a single regime simultaneously sets both, which is *why* a regime can flip the joint lift > 1.

### 3.3 Network architecture
```
 inputs (per station, sliding 72-h window):
   [ pollutant lags (PM2.5,O3,NOx,CO,SO2), met, reanalysis-physics fields,
     regime soft-membership g(x), daily MODIS AOD, calendar ]
        │
   ENCODER  : Temporal CNN / LSTM  →  latent state z_t
        │
   PHYSICS CORE : two coupled state heads  Ĉ_PM(t..t+H), Ĉ_O3(t..t+H)
                  advanced by the ADR/box residual above (Neural-ODE / discretised)
                  with regime-dependent learnable params θ_r = {v_d, K, k_rate, P_prod}
        │
   HEADS   :  (a) regression  Ĉ_PM, Ĉ_O3  trajectories
              (b) compound-extreme probability  p_comp = σ( joint-threshold layer )
                  at lead 24 / 48 / 72 h
```
- **Graph coupling (optional, IGARSS-friendly):** the 9 stations form a graph; advection `(u·∇)C` is realised as message passing along edges (à la AirPhyNet/TransNet).
- **Learnable physical parameters are regime-indexed** (`θ_r`) — so the model can report *different deposition/diffusion/production in the photochemical vs stagnant regime*: an interpretable scientific output.

### 3.4 Loss function
```
L  =  L_data            (BCE/focal on compound label  +  MSE on Ĉ_PM, Ĉ_O3)
   +  λ_phys · L_ADR     (mean-squared residual of the two governing PDE/ODEs)
   +  λ_couple · L_chem  (photostationary O3–NO–NO2 + precursor consistency)
   +  λ_ic/bc · L_constraints  (C ≥ 0 ;  initial/boundary conditions)
   +  λ_obs · L_AOD      (Ĉ_PM consistent with RH-corrected satellite AOD — the RS constraint)
   +  λ_reg · L_regime   (predictions consistent with regime soft-membership)
```
- **Focal loss** handles the 6.5% positive class.
- **λ_phys** is annealed (data-fit first, then tighten physics) — standard PINN training practice.
- `L_AOD` is the **remote-sensing physics constraint** (mirrors the WRF-CHEM+PINN+RS paper): satellite AOD pins the aerosol column where ground PM2.5 is missing (recall PM2.5 is only 68% populated).

### 3.5 Why a PINN (not a plain DL model) is justified *here*
1. **Gappy pollutants (57–75%):** the physics residual propagates information through gaps; AOD constrains PM where the sensor is down.
2. **Rare tail (6.5%):** physics constrains extrapolation into the joint tail where data is thin — exactly the compound-extreme regime.
3. **Interpretability / science payload:** learned **regime-dependent** deposition, diffusion and O3-production rates are publishable physical findings, not just a score.
4. **Transferability (LOSO / new stations):** physics + RS generalise where a pure data model overfits station idiosyncrasies — the IGARSS spatial-transfer claim.

---

## 4. Health & remote-sensing-for-health module (the "so what")

1. **Compound forecast → combined health risk.** Map predicted compound days to excess risk using the **synergistic** concentration–response (synergy index ≈ 1.93; combined > additive), reported **per regime** — so the warning carries a health weight, not just a concentration.
2. **Satellite exposure surface.** Fuse **AOD-derived PM2.5** (RH-corrected) with interpolated station O3 to build a **gridded compound-exposure map** for Dhaka and beyond, where ground stations are sparse.
3. **Population-weighted, regime-stratified burden.** Combine the exposure surface with population (and night-lights as a proxy) to estimate **person-days of compound exposure by regime** — the metric Kalashnikov et al. used for the western US, computed for the first time for Bangladesh.
4. **Operational output:** a 24–72 h **compound-risk early warning** with a health-tier (vulnerable-group advisory) keyed to the active regime.

---

## 5. Evaluation (summary — full protocol in `research_plan.md`)
- **Splits:** temporal hold-out (train 2014–15, test 2016) + **Leave-One-Station-Out** for transfer.
- **Metrics:** PR-AUC (primary, imbalanced), Brier, reliability, recall@precision, CSI; **stratified by regime** (must catch the Dry-Sunny-Photochemical 17.6% regime).
- **Decisive ablations:** PINN vs data-only NN (does physics help the tail/gaps/transfer?); ±regime conditioning; ±AOD constraint; single- vs compound-target; 75th vs 90th threshold.
- **Rigour:** bootstrap 95% CIs, DeLong AUC tests, permutation vs climatology.
- **Interpretability:** SHAP + the **learned regime-dependent physical parameters** θ_r.

---

## 6. Bridge to modelling (next step — what we build)
```
01_qc.py        physical-range/flatline/spike QC + bitmask
02_features.py  lags, rolling, u/v, sin/cos, calendar, daily AOD merge
03_regimes.py   standardise → KMeans(k via silhouette)+GMM soft membership (+SOM check); save model
04_labels.py    seasonal-75th thresholds (train-only) → daily compound label + lift tables
05a_baselines.py persistence, climatology, single-pollutant GBT, LightGBM compound
05b_pinn.py     encoder + coupled ADR core + heads + composite loss
06_eval.py      temporal + LOSO, per-regime metrics, ablations, CIs, SHAP
07_health.py    synergy CRF + AOD exposure surface + person-day burden
```
**Build order:** 01–04 (data + regimes + labels, reproduce the lift table) → 05a baselines → 05b PINN → 06 evaluation → 07 health/RS. We start at modelling once you approve this plan.

---

## References (URLs)
- Lyu et al. 2024, GRL — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL106527
- Schnell & Prather 2017, PNAS — https://www.pnas.org/doi/10.1073/pnas.1614453114
- Kalashnikov et al. 2022 (western US) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8730618/
- YRD co-occurrence 2024, ACS ES&T Air — https://pubs.acs.org/doi/10.1021/acsestair.4c00314
- Synoptic drivers, eastern China, ACP 2021 — https://acp.copernicus.org/articles/21/9105/2021/
- SOM clustering case study — https://doi.org/10.3390/atmos17010044
- AirPhyNet 2024 — https://arxiv.org/pdf/2402.03784
- TransNet 2026, npj Clean Air — https://www.nature.com/articles/s44407-026-00052-x
- PINN inverse advection–diffusion 2025 — https://arxiv.org/pdf/2503.18849
- PINN + WRF-CHEM + remote sensing 2023 — https://www.sciencedirect.com/science/article/abs/pii/S1352231023004570
- Satellite PM2.5 exposure review — https://link.springer.com/article/10.5572/ajae.2020.14.4.319
- NASA satellite monitoring for AQ & health — https://ntrs.nasa.gov/api/citations/20230000905/downloads/Holloway_review_satellite%20monitoring%20for%20air%20quality%20and%20health.pdf
- PM2.5–O3 synergistic toxicity — https://pmc.ncbi.nlm.nih.gov/articles/PMC12030836/
- Dhaka PM2.5 meteorology, seasonal — https://link.springer.com/article/10.1007/s44274-026-00560-3
