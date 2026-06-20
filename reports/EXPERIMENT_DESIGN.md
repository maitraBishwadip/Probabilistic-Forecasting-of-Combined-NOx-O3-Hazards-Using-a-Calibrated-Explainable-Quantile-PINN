# Experiment Design & Architecture — QR-PINN for Combined NOx–O3 Extremes

**Method (only model under study):** Quantile-Regressive Physics-Informed Neural Network (QR-PINN).
**Goal:** forecast the **full predictive probability distribution** of a **combined NOx–O3 hazard
index** at **hourly, multi-step** lead times, and from it the probability of the **combined extreme**
(the ≥70 % hazardous class).
**Data:** `BD_DOE_2014-16.csv` — 9 DoE stations, hourly, 2014–2016.

> **Governance.** This is the *architecture of record*. Every experiment in §7 must be **explained
> and approved before it is run** (`CLAUDE.md §0`). Results go only to `RESULTS_LOG.md`; the `.tex`
> reads only from there. This document holds *plans and hypotheses*, never claimed results.

---

## 1. Locked decisions

| Item | Choice |
|---|---|
| Targets | **NOx and O3**, whole dataset, **no regime conditioning** |
| Combined target | a **single combined hazard index** `H` (§2) |
| Output | the **entire predictive distribution** of `H` — a dense set of non-crossing conditional quantiles → CDF/PDF, per horizon |
| Extreme | **cutoff = 70th percentile of `H` (train-fit)** ⇒ the **most-polluted top 30 % are the hazardous class**; **Q90 = secondary "severe" tier** (corrects an earlier inverted Q30 definition — see `RESULTS_LOG.md` THRCORR-Q70) |
| Resolution / timing | **hourly, multi-step**: predict `H_{t+1..t+24}` (next-24 h trajectory distribution) |
| Splits | temporal (train 2014–15, test 2016) + Leave-One-Station-Out (LOSO) |

---

## 2. The combined NOx–O3 hazard index `H`

`H` must be (i) a graded hazard usable for a continuous quantile target and (ii) physically and
health-meaningful. We fix scalers/weights/thresholds on **training data only**.

- **ADOPTED definition (E5) — rank / quantile-uniform combination:**
  `H = ½·F_NOx(NOx) + ½·F_O3(O3)`, where `F_·` is the **train empirical CDF** (pooled) of each
  pollutant, so each marginal is **uniform[0,1]** and neither can dominate by scale or tail shape.
  Interpretable as "both pollutants high relative to their own climatology." `thr = Q₇₀(H_train)`
  (the most-polluted top 30 % are the extreme/hazardous class; Q90 = severe tier).
- **DEPRECATED original — equal-weight robust-scaled combination:**
  `H = ½·s(NOx) + ½·s(O3)` with a **robust scaler** (median / IQR). E3 showed this lets **O3
  overpower** the index (Var-share O3 73% vs NOx 44%; prediction corr O3 0.90 vs NOx 0.12). The
  rank index (above) fixes it: Var-share 61%/63%, prediction corr NOx 0.35 / O3 0.63 (E5). Kept
  only as a sensitivity reference.
- **Physics-aligned variant (secondary target & sensitivity):** **total oxidant `Oₓ = O₃ + NO₂`**
  (NO₂ ⊂ NOx) — the chemically conserved combined oxidant (§3), in native ppb. Used both as a target
  for sensitivity and as the basis of the conservation residual.
- **Health-weighted variant (sensitivity only):** weights from concentration–response evidence
  (`LITERATURE_REVIEW.md §7`).

**Extreme label / hazard region.** `thr = Q₇₀(H)` on the training set; an observation/forecast hour
is "combined-extreme/hazardous" if `H ≥ thr` (the most-polluted **top 30 %** by construction). The model
never trains on this binary label directly — it predicts the *distribution* of `H`; the exceedance
probability `P(H_{t+h} ≥ thr)` is read off the predicted CDF. Report sensitivity at **Q70/Q80/Q90** so
the "extreme" (top-30 %) and the "severe" (top-10 %) regions are both covered. (Corrects an earlier
*inverted* Q30/≥70 % definition; the re-score is recorded as `RESULTS_LOG.md` THRCORR-Q70.)

---

## 3. Embedded physics (the PI in QR-PINN) — no regimes, global learnable parameters

NOx and O₃ are coupled by the photostationary cycle, so the physics is a **coupled box / ADR system**
for NO, NO₂, O₃ at each station, using fields already in the data (all reanalysis fields are fully
populated):

**Per-species hourly box mass-balance** (discretized, station *i*):
```
ΔC_k/Δt = E_k/H_mix            emission into mixing layer (H_mix = boundary_layer_height_m)
        − (VC/H_mix)·C_k       dilution / ventilation (VC = ventilation_coefficient)
        − (v_d,k/H_mix)·C_k    dry deposition
        − Λ_k·P·C_k            wet scavenging (P = precip_mm)
        − adv_i(C_k)           advection from neighbour stations (wind_u/v) [optional graph term]
        + R_k(·)               chemistry
```
**Chemistry coupling (the joint constraint):**
```
Leighton photostationary state:  C_O3 · C_NO ≈ (J / k(T)) · C_NO2
   J ∝ photochemical_activitiy_index / solar_rad_Wm2 ;  k(T) via Arrhenius in temp_C
Total-oxidant quasi-conservation: Ox = O3 + NO2 evolves slowly (production − deposition − dilution),
   invariant to the fast NO↔NO2↔O3 titration → residual penalty d(Ox)/dt ≈ slow terms only
```
- Learnable physical parameters `{v_d, Λ, k-scaling, production}` are **global** (optionally smooth
  functions of meteorology) — **not** clustered into regimes (locked decision). Their learned values
  are an interpretable scientific output.
- The graph/advection term is **optional** (IGARSS-friendly) and introduced only as an ablation.

---

## 4. Data pipeline (refactor `pipeline/01_build_dataset.py`; data-manip stays outside `archive/`)

The existing `01` does PM2.5–O3 daily + K-means regimes; it will be **refactored** to:
1. **QC** (keep): physical-range clip + flat-line removal for `NOx_ppb, NO_ppb, NO2_ppb, O3_ppb`
   (and O₃-8 h), logged to `qc_report.json`.
2. **Hourly, per station** (no daily aggregation): continuous hourly calendar reindex so lags/leads
   are true hourly offsets; **missingness masks** kept per channel.
3. **Targets:** build `Oₓ = O₃ + NO₂`; fit robust scalers (train-only) → compute `H` (§2) and
   `thr = Q₃₀(H)` (train-only). Do **not** impute the target `H`; an hour with NOx **or** O₃ missing
   has `H = NaN` (excluded from the data term, still usable for the physics term).
4. **Features at forecast time *t*** (sliding window, length `L` ∈ {48,72} h to tune): lagged
   NO/NO₂/NOx/O₃/O₃-8 h; meteorology + reanalysis (temp, RH, wind speed + u/v, solar, precip, BLH,
   ventilation coeff, photochemical index, pressure, geopotential heights); daily MODIS AOD/NDVI/fire
   forward-filled within day (+ missing flag); calendar (hour sin/cos, day-of-year sin/cos, weekday);
   station id / lat-lon.
5. **Leads:** `H_{t+1..t+24}` and the component concentrations for the physics/auxiliary head.
6. Persist scalers, `thr`, and the `H`-definition as artefacts (deterministic, leakage-free).

**Known missingness (from prior profiling — re-verify for this target set):** O₃ ≈ 74.7 %,
NOx ≈ 62.6 % populated → masking + the physics residual (which propagates information through gaps)
are core, not optional.

---

## 5. QR-PINN architecture

```
 inputs: sliding window (L h) of [pollutant lags, met, reanalysis-physics, AOD, calendar, station, masks]
        │
   ENCODER : Temporal CNN (TCN) or LSTM  →  latent z_t                (sequence model over the window)
        │
        ├─► QUANTILE HEAD  (the model's product)
        │      MCQRNN-style: τ is an input; output Q_τ(H_{t+h}) for h=1..24, τ in a dense grid,
        │      MONOTONE in τ by construction (non-negative increments) ⇒ no quantile crossing
        │
        └─► PHYSICS/AUX HEAD  (regulariser; not the deliverable)
               predict ĉ_NO, ĉ_NO2, ĉ_O3 (⇒ NOx, Ox) trajectories advanced by the §3 box/ADR system
               with global learnable physical parameters
```

**Quantile grid:** dense (e.g., τ ∈ {0.05,0.10,…,0.95} for training; finer {0.01…0.99} for the
final PDF reconstruction). Non-crossing guaranteed by the monotone-in-τ construction (Cannon-style),
with a soft non-crossing penalty as backup.

**Composite loss** (physics weight `λ_phys` annealed 0 → target over training):
```
L =        Σ_{τ,h} pinball_τ( Q_τ(H_{t+h}), H_{t+h} )          # primary: distributional fit on H
  + λ_phys · MSE( box/ADR residual of ĉ )                       # §3 physics consistency
  + λ_chem · ( Leighton residual + Ox-conservation residual )   # the NOx–O3 coupling
  + λ_data · MSE( ĉ , c_obs )      where observed                # auxiliary concentration fit
  + λ_pos  · positivity(ĉ ≥ 0)  + λ_nc · non-crossing(Q)        # constraints
```

**Output → full PDF.** The dense, monotone quantiles define the predictive **CDF** per horizon
`h`; differentiate / kernel-smooth for the **PDF**; integrate the tail for the exceedance probability
`P(H_{t+h} ≥ thr)`. These per-horizon distributions are *the* deliverable.

---

## 6. Evaluation protocol (proper scoring + calibration + tail focus)

- **Distributional (primary):** average **pinball loss** over (τ, h); **CRPS** from the quantiles.
- **Calibration:** prediction-interval coverage **PICP** vs nominal + mean interval width; **PIT
  histogram** / reliability; optional PIT-recalibration & conformal coverage (`LIT §5,§9`).
- **Tail / hazard region:** treat `P(H_{t+h} ≥ thr)` as a probabilistic classifier of the
  **top-30 % hazardous class (`H ≥ Q70`)** → **Brier score + reliability**; also evaluate the
  **severe upper tail** (high-τ pinball at τ=0.9/0.95; exceedance skill at Q90) so the extreme
  *and* the severe spikes are both scored.
- **Lead-time degradation:** all metrics across h = 1 → 24 h.
- **Spatial transfer:** **LOSO** (train 8 stations, test the 9th), metrics mean ± std over folds.
- **Rigour:** bootstrap 95 % CIs on all headline metrics.

---

## 7. Experiment plan (each is EXPLAIN → APPROVE → RUN)

| ID | Experiment | Status / run | Produces |
|---|---|---|---|
| **E0** | **Data prep refactor** — hourly NOx–O3 cleaning, `H` index + `thr`, windowed tensors + masks, train-only scalers. | **done** (`02_qrpinn_dataprep.py`) | clean hourly set, `qrpinn_meta.json` |
| **E1/E2** | **Baselines + QR-PINN (full)** — climatology, QRNN, QR-PINN; full PDF output. | **done** (`03_qrpinn_model.py`) | `results_qrpinn.json` |
| **E3** | **Physics ablation + pollutant-dominance** — QRNN/data-only/full; index & prediction dominance. | **done** (`04_qrpinn_e3.py`) | `results_e3.json` (O3 overpowered) |
| **E4** | **Calibration (PIT)** — PIT recalibration, reliability. | **done** (`05_qrpinn_e4.py`) | `results_e4.json` (80% fixed; 90% tail-limited) |
| **E5** | **Index rebalance** — rank/quantile-uniform `H` (added after E3 found O3 dominance). | **done** (`06_qrpinn_e5.py`) | `results_e5.json` (balanced; NOx now used) |
| **E6** | **Tail calibration** — extended τ-grid (0.975/0.99) + **CQR** (Romano 2019) for guaranteed intervals on the rank model; PIT for full dist. See `METHODS_calibration.md`. | **done** (`07_qrpinn_e6.py`) | `results_e6.json`, reliability + coverage figs |
| **E7** | **Spatial transfer + bootstrap CIs** — Leave-One-Station-Out (station-agnostic model) + in-distribution reference; moving-block bootstrap 95% CIs. See `METHODS_transfer.md`. | **done** (`08_qrpinn_e7.py`) | `results_e7.json`, per-station LOSO figs |
| **E8** | **Robustness** — threshold sensitivity (Q25/30/35/90), λ-physics sweep, seed. | **done** (`09_qrpinn_e8.py`) | `results_e8.json`, sensitivity tables |
| **E9** | **Explainability** — post-hoc attribution of the frozen calibrated model: Integrated Gradients (median / tail-Q95 / interval-width / per-lead) + model-agnostic occlusion (ΔPinball/ΔCRPS/ΔBrier) + physics-group audit (day/night). See `METHODS_explainability.md` (§9). | **done** (`10_qrpinn_e9_xai.py`) | `results_e9.json`, attribution + temporal + group + day/night figs |
| **E10** | **Physics-formulation fix** — diagnose why the embedded physics was ineffective (`PHYSICS_DIAGNOSIS.md`) and re-engineer it: physics-guided **hybrid** (NN-residual forecast + interpretable, semi-implicit NO/NO₂/O₃ box-ODE with Leighton + structural Oₓ conservation, per-station emission). Decisive FREE vs HYBRID vs PHYS-ONLY test, seeds 0/1. | **done** (`11_qrpinn_e10.py`) | `results_e10.json`, `figs/e10_physics.png` |
| **E12** | **Regime-stratified physics value** — the pooled E10 verdict (physics ≠ accuracy) tested *per regime*, since PINN gains concentrate in sparse/gappy/tail/long-lead conditions (AirPhyNet ICLR'24; Krishnapriyan NeurIPS'21). **No new model design**: deterministically reproduce E10's seed-0 FREE/HYBRID/PHYS-ONLY (validity check vs `results_e10.json`), then stratify the FREE−HYBRID pinball gap by **(1) upper tail** (per-τ pinball τ=0.90/0.95, Brier@Q90, extreme-outcome subset H≥Q90), **(2) lead time** (h=1…24), **(3) input-window missingness** terciles, **(4) per-station** (in-distribution). Verdict: is there ANY regime where physics is ≤ FREE+0.003? LOSO FREE-vs-HYBRID is **deferred to E13** (needs 9-fold retraining). | **done** (`12_qrpinn_e12.py`) | `results_e12.json`, `e12_preds.npz`, `figs/e12_strata.png` |

Nothing runs without explicit approval of its plan. (Numbering reflects actual run order; E5 was
inserted in response to the E3 dominance finding.)

> **E10 finding (recorded fact, not a plan).** The physics *implementation* failures were fixed —
> the corrected ODE is stable and the hybrid (pinball 0.25) and physics-only (0.32) variants vastly
> outperform the earlier broken "physics-forced" trial (0.35; `RESULTS_LOG.md` PHYSFORCE). **But the
> scientific verdict is unchanged and reported honestly: embedding the NOx–O₃ box chemistry does not
> improve probabilistic accuracy of `H`.** The data-only forecast remains best (FREE pinball
> 0.195–0.210 vs HYBRID 0.252–0.260, stable across seeds), even though the physics variants are handed
> future met as forcing. Consistent with the O₃/met-dominated predictability structure (`PHYSICS_DIAGNOSIS.md`
> R6). The PINN's physics is therefore positioned as **interpretable / physics-guided** (learned,
> physically-ordered rates incl. per-station emission scales; forecast follows the rollout, corr≈0.75),
> **not** as an accuracy mechanism — `CLAUDE.md §1` integrity rules + the diagnosis's honesty fallback.

> **E12 finding (recorded fact, not a plan).** The pooled E10 verdict was tested *per regime* (upper tail,
> lead time, input-window missingness, per-station), since PINN gains typically concentrate in sparse/tail/
> long-lead conditions. **No robust regime was found where the embedded physics improves accuracy of `H`.**
> HYBRID is worse than FREE at every lead (no long-lead crossover) and in every missingness tercile (though
> the gap *narrows* in the theoretically-expected direction, +0.046→+0.040, never crossing), and worse in the
> tail by both proper metrics (τ=0.95 pinball, Brier@Q90). The only HYBRID≤FREE slices are 2/9 stations
> (GAZIPUR tie −0.0005, RAJSHAHI −0.005), both inside seed noise; the extreme-outcome-subset "win" for
> PHYS-ONLY is a selection artifact of its positive bias. This **confirms E10 at the regime level** and is
> consistent with the structural fact that the implemented Leighton triad is a closed null cycle (conserves
> Oₓ, zero net O₃ production) — it cannot generate the O₃ extremes it is scored on. **Next lever (E13):**
> add HCHO (`omi_column_hcho`) as a VOC/RO₂ proxy + the HCHO/NO₂ regime ratio so the box-ODE gains a net-O₃
> production pathway. (E12 = stratified re-scoring of the reproduced E10 models; no §1/§3 decision changed.)

---

## 8. Reproducibility & risks

**Reproducibility.** Fixed seeds (numpy/torch); persisted scalers, `thr`, `H`-definition, and model
checkpoints; train-only fitting of every transform (no leakage); each stage writes JSON consumed
verbatim by the report; environment/package versions recorded with each run in `RESULTS_LOG.md`.

| Risk | Mitigation |
|---|---|
| NOx/O₃ gaps (≈25–37 % missing) | masking + physics residual propagate through gaps; never impute target `H` |
| Diurnal NOx–O₃ anti-phasing at hourly scale | combined index `H` + Oₓ conservation absorb the fast titration; physics encodes it explicitly |
| Quantile crossing | monotone-in-τ construction + non-crossing penalty |
| corrected extreme = top-30 % (minority, base-rate ≈0.26) class | score with proper rules (pinball/CRPS) + calibration; evaluate exceedance at Q70 (extreme) and Q90 (severe), not a single cut |
| PINN training instability | anneal `λ_phys`, scale concentrations to O(1), curriculum on horizon |
| Station heterogeneity / transfer | station embedding + LOSO; global (not regime) physics for transferability |

---

## 9. Explainability of the predictions (E9 — architecture of record)

**Why.** Two properties make the QR-PINN worth explaining differently from a point forecaster: it emits
the **whole predictive distribution** of `H` and it carries **embedded NOx–O3 photochemistry** (§3). So
the explainability is designed to (i) attribute the **distribution**, not just a point, and (ii) **audit
whether the learned input–output map is faithful to the encoded chemistry** — the credibility argument for
a PINN, and the IGARSS framing. Full method, equations and citations: `METHODS_explainability.md`.

**Subject.** Post-hoc analysis of the **frozen, already-approved calibrated reference model**
`artefacts/qrpinn_full_rank_ext.pt` (rank index, physics on, 13-τ grid; E6). **No retraining; no change to
any §1/§3 locked decision.** Inputs are the 26 named features over the `L=48 h` window (`qrpinn_meta.json
feat_names`) plus the station id (embedding).

**Two complementary, cross-validating methods.**
1. **Integrated Gradients** (Sundararajan, Taly & Yan 2017) — axiomatic, completeness-checked gradient
   attribution along a straight path from a **"missing/no-information" baseline** (value channels = 0
   = scaler median/mean; mask channels = 0 = missing — exactly how the net sees a gap) to the real
   window. Attributed scalars expose the **distributional** angle, each at chosen lead(s):
   - **median** `Q₀.₅(H_{t+h})` (central forecast), at `h=1` and `h=24` (driver shift with lead);
   - **upper tail** `Q₀.₉₅(H_{t+h})` at `h=1` (the severe/hazardous region);
   - **interval width** `Q₀.₉₅−Q₀.₀₅` at `h=1` (what drives *predictive uncertainty*).
   Aggregated **per feature**, **per lag** (temporal saliency over the 48 h window), and **per physics
   group**. Completeness gap `|Σ attr − (g(x)−g(x'))|` reported as a validity check.
2. **Permutation / occlusion importance** (Fisher, Rudin & Dominici 2019) — model-agnostic, in the
   model's own **loss currency**: mean-occlude each feature / group across the window on the **test-2016**
   anchors and measure **ΔPinball / ΔCRPS / ΔBrier(exceedance)**; plus a station-embedding permutation.

**Physics-group audit.** Features are bucketed into the §3 chemistry blocks — NOx-side lags
(NO/NO₂/NOx + masks), O₃-side (O₃ + mask), **photochemistry** (solar_rad, photochemical index, temp),
**dispersion/transport** (BLH, ventilation coeff, wind speed/u/v), **wet removal** (precip, RH), pressure,
**satellite** (AOD + mask), **calendar** (hour/doy/weekday), **station**. The test: do photochemistry +
dispersion dominate the **O₃/tail** side and NOx lags the **NOx side**, and does the balance flip
**day vs night** (O₃ formation by day ↔ NOx titration at night, §3)?

**Honesty.** Saliency on time series can be unreliable (Ismail et al. 2020); the defensible position is to
report **IG and occlusion together** and treat their **agreement** (sign + ranking) as the robustness
claim, with the completeness gap as an internal IG check. This is interpretation of a frozen model, never a
source of new tuning.

**Outputs.** `results_e9.json`; **publication-grade figures (300-dpi PNG + vector PDF, colour-blind-safe,
consistent per-group palette)**: `e9_group_importance` (group bars + IG-vs-occlusion agreement scatter),
`e9_distributional` (centre vs tail vs uncertainty), `e9_temporal_heatmap` (group × lag saliency),
`e9_daynight` (day↔night dumbbell), `e9_leadtime` (driver shift h=1→24), `e9_feature_importance`
(supplementary top features); a run block in `RESULTS_LOG.md`. Seeds fixed; sampled-anchor indices,
baseline, and the IG-vs-occlusion agreement `r` logged for reproducibility.
