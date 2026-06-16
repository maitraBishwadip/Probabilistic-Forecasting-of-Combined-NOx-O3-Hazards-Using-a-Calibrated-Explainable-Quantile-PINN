# RESULTS & REPORT — QR-PINN for Combined NOx–O₃ Extremes over Bangladesh

**Researcher:** Bishwadip Maitra · BUET Air-Quality Project · target venue **IEEE IGARSS**
**Method under study:** Quantile-Regressive Physics-Informed Neural Network (**QR-PINN**)
**Compiled:** 2026-06-16

> **Status of this document.** This is a *human-readable synthesis* of every run recorded in
> `RESULTS_LOG.md`, which remains the **single source of truth** (`CLAUDE.md §1–2`). Every number
> below carries its originating **run ID** and is copied verbatim from the log — nothing here is a
> new measurement, a placeholder, or a hoped-for value. Where a result is unfavourable, it is stated
> plainly. Plans/hypotheses live in `EXPERIMENT_DESIGN.md`; this file reports only outcomes.

---

## 1. Study definition (locked) and data

| Item | Value |
|---|---|
| Targets | **NOx and O₃** combined, whole dataset, **no regime conditioning** |
| Combined target | a single **hazard index `H`** (see §3) |
| Model output | the **full predictive distribution** of `H` (dense non-crossing conditional quantiles → CDF/PDF), per horizon |
| Extreme definition | cutoff = **30th percentile of `H`** (train-fit) ⇒ **≥70 % of hours are the "hazardous" class** |
| Resolution / timing | **hourly, multi-step** — forecast `H_{t+1..t+24}` (next-24 h trajectory distribution) |
| Splits | temporal (train **2014–15**, test **2016**) + **Leave-One-Station-Out (LOSO)** for transfer |

**Data.** `BD_DOE_2014-16.csv` — 9 DoE stations, hourly, 2014-01-01 → 2016-12-31 (223,776 raw rows).
After QC + hourly windowing (`pipeline/02_qrpinn_dataprep.py`), the modelling tensor is
**9 stations × 26,304 hours × 26 features**, yielding **86,037 train / 31,067 test** forecast anchors
(an anchor needs a full 48 h input window, a valid current `H`, and ≥1 valid future `H`).
Pollutant missingness is heavy (design estimate O₃ ≈ 74.7 %, NOx ≈ 62.6 % *populated*), handled by
per-channel **masking** and by **never imputing the target** (`H = NaN` when NOx **or** O₃ is missing;
those hours are dropped from the data term only).

---

## 2. Method summary (architecture of record)

- **Encoder:** LSTM (hidden 64) over the `L = 48 h` window + 9-station embedding (dim 8).
- **Quantile head (the deliverable):** monotone-in-τ construction (non-negative increments) ⇒ **no
  quantile crossing**; trained on a 9-point τ grid `{0.05,…,0.95}`, extended to 13 points
  `{0.01,…,0.99}` for tail calibration (E6).
- **Physics/aux head (regulariser):** a coupled NOx–O₃ box model; its design and the *honest finding
  that it does not improve accuracy* are the subject of §6 and `PHYSICS_DIAGNOSIS.md`.
- **Loss:** pinball (primary, on `H`) + concentration data-fit + physics/chemistry residuals + spread
  constraints. Physics weight `λ_phys` annealed 0→target.
- **Scoring:** pinball + CRPS (distribution); PICP / interval width + PIT + CQR (calibration); Brier on
  the exceedance probability `P(H≥thr)` (the ≥70 % hazard class); all across leads h = 1→24.

---

## 3. The hazard index `H` — and the mid-study correction (critical caveat)

`H` combines the two pollutants' own-climatology percentiles. **The index definition changed once,
mid-study**, and this is the single most important caveat for reading absolute numbers:

- **ROBUST index (deprecated, runs: baseline, E3, E4):** `H = ½·s(NOx) + ½·s(O3)` with a median/IQR
  robust scaler. **E3 showed O₃ overpowers it** (variance-share O₃ 72.8 % vs NOx 44.1 %; prediction
  correlation O₃ **0.898** vs NOx **0.116**) — the "combined" index was effectively an O₃ index.
- **RANK / quantile-uniform index (ADOPTED, runs: E5–E10):** `H = ½·F_NOx(NOx) + ½·F_O3(O3)` with the
  **train empirical CDF**, so each marginal is uniform[0,1] and neither dominates by scale/tail. This
  **rebalanced** the index (E5: variance-share 61.0 %/63.3 %; prediction correlation NOx 0.348 / O₃
  0.629). `thr = Q₃₀(H_train) = 0.3948`.

> ⚠️ **Comparability rule.** ROBUST-index metrics (pinball ≈ 0.93–1.05) and RANK-index metrics
> (≈ 0.20–0.32, on [0,1]) are on **different scales and are NOT comparable**. Compare only *within*
> an index family. All headline conclusions use the adopted RANK index (E5–E10).

---

## 4. Results by experiment (verbatim from `RESULTS_LOG.md`)

### E0 — Data prep (`02_qrpinn_dataprep.py`)
Hourly NOx–O₃ QC, rank-index `H` + `thr`, windowed tensors + masks, train-only scalers. Artefacts:
`qrpinn_meta.json`, `qrpinn_data.npz`. (No metric; enabling stage.)

### Baseline run — `QRPINN-20260615-212045` · ROBUST index · seed 0 · 12 epochs
| model | pinball | CRPS | PICP80 | PICP90 | width80 | Brier(exc) | Brier@h1 |
|---|---|---|---|---|---|---|---|
| climatology | 1.9256 | 0.4279 | 0.693 | 0.764 | 1.749 | 0.2089 | 0.2063 |
| QRNN | 1.0571 | 0.2349 | 0.663 | 0.804 | 0.702 | 0.0989 | 0.0579 |
| QR-PINN | 1.0527 | 0.2339 | 0.667 | 0.808 | 0.721 | 0.0997 | 0.0568 |

*Both learned models crush climatology (~45 % lower pinball). QR-PINN ≈ QRNN.* (ROBUST index.)

### E3 — Physics ablation + pollutant dominance — `E3-20260615-222715` · ROBUST · seed 0
| model | pinball | pin_mean | CRPS | PICP80 | PICP90 | width80 | Brier | Brier@1 |
|---|---|---|---|---|---|---|---|---|
| QRNN | 1.0571 | 0.1175 | 0.2349 | 0.663 | 0.804 | 0.702 | 0.0989 | 0.0579 |
| DATA-ONLY | 1.0662 | 0.1185 | 0.2369 | 0.676 | 0.816 | 0.736 | 0.0983 | 0.0559 |
| FULL | 1.0527 | 0.1170 | 0.2339 | 0.667 | 0.808 | 0.721 | 0.0997 | 0.0568 |

- **Physics value (FULL − DATA-ONLY): −0.0135 pinball** (FULL better by ~1.3 %) — *small, and on the
  robust index only*. **This did not reproduce on the adopted rank index** (see E8).
- **Dominance:** O₃ overpowers the robust index (details in §3). → motivated E5.

### E4 — PIT recalibration of FULL — `E4-20260615-225005` · ROBUST · calib early-2016 / eval late-2016
| state | pinball | CRPS | PICP80 | PICP90 | width80 | Brier(exc) |
|---|---|---|---|---|---|---|
| before | 0.9342 | 0.2076 | 0.678 | 0.821 | 0.679 | 0.1183 |
| after | 0.9389 | 0.2086 | **0.765** | 0.821 | 0.899 | 0.1191 |

*PIT recalibration (Kuleshov 2018) **fixed the 80 % interval** (0.678→0.765) at a small sharpness cost;
the **90 % interval stayed at 0.821 — tail-limited**, motivating the E6 tail work.*

### E5 — Rank-index rebalance — `E5-20260615-230704` · RANK · seed 0
| index | Var-share NOx | Var-share O₃ | corr(H,NOx) | corr(H,O₃) |
|---|---|---|---|---|
| robust (old) | 44.1 % | 72.8 % | 0.536 | 0.754 |
| **rank (new)** | **61.0 %** | **63.3 %** | 0.625 | 0.643 |

| prediction-level | corr(pred H, NOx) | corr(pred H, O₃) |
|---|---|---|
| robust (E3) | 0.116 | 0.898 |
| **rank (E5)** | **0.348** | 0.629 |

- FULL-rank metrics (**new scale, not comparable to robust runs**): pinball **0.197**, CRPS 0.0438,
  PICP80 0.708 / width 0.171, PICP90 0.843 / width 0.24, Brier(exc) 0.0972;
  **test base-rate extreme = 0.834**.

### E6 — Tail calibration (extended grid + PIT + CQR) — `E6-20260615-233855` · RANK · 13-τ
- raw: pinball **0.2197**, CRPS 0.0338, Brier(exc) 0.1204; tail pinball .9/.95/.99 = 0.0158/0.0099/0.0034.

| level | nominal | PICP raw | PICP PIT | PICP CQR | width raw | width PIT | width CQR |
|---|---|---|---|---|---|---|---|
| 80 % | 0.80 | 0.725 | 0.760 | **0.781** | 0.184 | 0.200 | 0.208 |
| 90 % | 0.90 | 0.839 | 0.875 | **0.892** | 0.241 | 0.273 | 0.282 |
| 95 % | 0.95 | 0.923 | 0.936 | **0.951** | 0.321 | 0.348 | 0.365 |

- **CQR (Romano 2019)** gives the best coverage (95 %→0.951) at modestly wider intervals; CQR PICP90 by
  lead h=1/6/12/24 = 0.893/0.887/0.902/0.900 (**coverage holds across lead time**).
- ⚠️ Split-CQR coverage is **approximate under time-series non-exchangeability** (`METHODS_calibration.md`);
  ACI/EnbPI flagged as future work.

### E7 — LOSO spatial transfer + block-bootstrap CIs — `E7-20260616-033822` · RANK · station-agnostic · 8 epochs
Moving-block bootstrap (block = 24 h, B = 1000), 95 % CI.

| setting | pinball [95 % CI] | CRPS [95 % CI] | PICP80 | PICP90 | Brier(exc) |
|---|---|---|---|---|---|
| in-distribution (all-9) | 0.2024 [0.2025, 0.2140] | 0.0450 [0.0450, 0.0475] | 0.718 | 0.875 | 0.0979 |
| LOSO mean ± std (9 folds) | **0.2277 ± 0.0655** | 0.0506 ± 0.0146 | 0.700 | 0.859 | 0.0985 |
| LOSO pooled (CI) | [0.2025, 0.2132] | [0.0450, 0.0474] | | | |

| held-out station | pinball | CRPS | PICP90 | Brier(exc) | n |
|---|---|---|---|---|---|
| AGRABAD | 0.2328 | 0.0517 | 0.832 | 0.1257 | 93,415 |
| BARC | 0.3471 | 0.0771 | 0.814 | 0.0561 | 1,898 |
| BARISHAL | 0.1442 | 0.0320 | 0.971 | 0.1112 | 16,020 |
| DARUS SALAM | 0.2065 | 0.0459 | 0.880 | 0.1045 | 123,584 |
| GAZIPUR | 0.3167 | 0.0704 | 0.713 | 0.1529 | 7,244 |
| KHULNA | 0.1813 | 0.0403 | 0.877 | 0.0093 | 68,520 |
| NARAYANGANJ | 0.1475 | 0.0328 | 0.958 | 0.0770 | 131,176 |
| RAJSHAHI | 0.2482 | 0.0551 | 0.833 | 0.1517 | 76,134 |
| SYLHET | 0.2245 | 0.0499 | 0.850 | 0.0978 | 106,697 |

- **Transfer gap (LOSO mean − in-dist) = +0.0253 pinball** — the model generalises to unseen stations
  with a modest, quantified penalty.
- ⚠️ Per-station `n` varies by ~70× (BARC 1,898 vs NARAYANGANJ 131,176); the worst folds (BARC,
  GAZIPUR) are the smallest stations, the best (BARISHAL, NARAYANGANJ) the cleaner/easier ones. Spatial
  cross-validation caveats in `METHODS_transfer.md §5`.
- ⚠️ Bootstrap artefact: the in-dist point estimate (0.2024) sits just **below** its CI lower bound
  (0.2025) — a known quirk of the moving-block resample on autocorrelated residuals; treat the CI as
  indicative, not exact.

### E8 — Robustness (physics-weight sweep, seed, threshold) — `E8-20260616-040817` · RANK · 8 epochs
**(A) Physics-weight sweep (seed 0)**
| λ_phys | pinball | CRPS | PICP90 | Brier(Q30) |
|---|---|---|---|---|
| 0.0 | **0.2054** | 0.0456 | 0.858 | 0.1023 |
| 0.2 | 0.2063 | 0.0458 | 0.860 | 0.1014 |
| 0.5 | 0.2188 | 0.0486 | 0.850 | 0.1052 |

**(B) Seed stability (λ = 0.2)**
| seed | pinball | CRPS | PICP90 | Brier(Q30) |
|---|---|---|---|---|
| 0 | 0.2063 | 0.0458 | 0.860 | 0.1014 |
| 1 | 0.2060 | 0.0458 | 0.854 | 0.0994 |

**(C) Threshold sensitivity (no retrain; cutoff enters only the exceedance metric)**
| cutoff | thr(H) | test base-rate | Brier(exc) |
|---|---|---|---|
| Q25 | 0.361 | 0.887 | 0.0761 |
| Q30 | 0.395 | 0.834 | 0.1014 |
| Q35 | 0.426 | 0.766 | 0.1260 |

- **Key finding:** on the rank index, **physics does not help and *hurts* when up-weighted**
  (λ 0→0.2 essentially flat; λ=0.5 worse). The tiny E3 "physics helps" effect **did not reproduce**.
- Headline numbers are **seed-stable** (0.2063 vs 0.2060). Brier rises with a stricter (lower-base-rate)
  cutoff, as expected.

### E9 — Explainability (Integrated Gradients + occlusion) — `E9-20260616-085543` · frozen `qrpinn_full_rank_ext.pt`
Post-hoc on the frozen calibrated model; **no retraining**. IG baseline = "missing" (values 0, masks 0),
M = 32, N = 1280 stratified anchors (640 day / 640 night); occlusion on all 31,067 test anchors
(intact pinball 0.2165 / CRPS 0.0333 / Brier 0.1012).

- **IG vs occlusion agreement: Pearson r = 0.917** (two independent methods corroborate).
- Top-5 IG (median, h1): O3_ppb 0.207, NOx_ppb_mask 0.109, O3_ppb_mask 0.092, NO2_ppb_mask 0.068,
  NO_ppb_mask 0.063. Top-5 occlusion ΔPinball: O3_ppb +0.0928, NOx_ppb +0.0563, NO2_ppb_mask +0.0441,
  NO_ppb +0.0311, NO2_ppb +0.0150.

| group | IG med_h1 | IG tail_h1 | IG width_h1 | IG med_h24 | occ ΔPinball | occ ΔBrier |
|---|---|---|---|---|---|---|
| NOx_side | 0.375 | 0.343 | 0.292 | 0.292 | +0.2225 | +0.0372 |
| O3_side | 0.298 | 0.308 | 0.376 | 0.356 | +0.0924 | +0.0420 |
| photochem | 0.064 | 0.069 | 0.051 | 0.074 | +0.0027 | +0.0017 |
| dispersion | 0.076 | 0.078 | 0.078 | 0.073 | +0.0004 | −0.0010 |
| wet_removal | 0.013 | 0.013 | 0.027 | 0.015 | +0.0008 | +0.0015 |
| pressure | 0.018 | 0.020 | 0.016 | 0.023 | +0.0031 | +0.0035 |
| satellite | 0.003 | 0.003 | 0.004 | 0.004 | −0.0000 | +0.0003 |
| calendar | 0.153 | 0.164 | 0.156 | 0.163 | +0.0175 | +0.0098 |

- Drivers are **pollutant lags + masks + calendar**; meteorology/photochemistry contribute little —
  consistent with the physics being inessential to skill (§6). Station permutation is negligible
  (ΔPinball +0.0013). Day/night IG fractions barely shift (NOx_side day 0.364 / night 0.387).
- ⚠️ Saliency on time series can be unreliable (Ismail et al. 2020); the defensible claim is the
  **IG↔occlusion agreement (r = 0.917)**, not any single attribution. Interpretation only, frozen model.

### PHYSFORCE — physics-forcing trial — `PHYSFORCE-20260616-093019` · RANK · 12 epochs · seed 0
Forecast median = sunlight-driven box-ODE rollout; spread head for quantiles.
| variant | pinball | CRPS | PICP80 | PICP90 | Brier(exc) |
|---|---|---|---|---|---|
| physics-forced | 0.3479 | 0.0773 | 0.790 | 0.862 | 0.2123 |
| no-physics (free) | 0.2104 | 0.0467 | 0.662 | 0.775 | 0.1026 |

- **Δ (physics − free): +0.1375 pinball — physics WORSE.** Making the box-ODE the *sole* forecaster
  through 7 global rates collapsed forecast capacity. (This was the first attempt at the
  `PHYSICS_DIAGNOSIS.md §4` fix; it over-corrected.) **Mislabeled "E9" at run time → relabeled PHYSFORCE.**

### E10 — Physics-GUIDED HYBRID (the corrected physics fix) — `E10-20260616-104625` · RANK · 10 epochs · seeds 0(+1)
`H_med = H_phys (semi-implicit NO/NO₂/O₃ box-ODE, ppb) + bounded NN residual`; real chemistry
(NO₂+hν→NO+O₃ photolysis; O₃+NO→NO₂ titration), explicit Leighton residual + structural Oₓ=O₃+NO₂
conservation, per-station emission; `lam_c=0.3, lam_leighton=0.05`.

**Test-2016, seed 0**
| variant | pinball | CRPS | PICP80 | PICP90 | Brier(exc) |
|---|---|---|---|---|---|
| **FREE (no physics)** | **0.2099** | **0.0466** | 0.679 | 0.788 | **0.1005** |
| HYBRID (E10) | 0.2525 | 0.0561 | 0.711 | 0.826 | 0.1197 |
| PHYS-ONLY | 0.3174 | 0.0705 | 0.670 | 0.772 | 0.1370 |

**Seed stability (seed 1)**
| variant | pinball | CRPS | PICP90 | Brier(exc) |
|---|---|---|---|---|
| FREE | 0.1951 | 0.0434 | 0.811 | 0.0960 |
| HYBRID | 0.2598 | 0.0577 | 0.795 | 0.1200 |

- **Δ (HYBRID − FREE): +0.0426 pinball, +0.0095 CRPS, +0.0192 Brier** (seed 0; +0.065 pinball seed 1).
- **Δ (PHYS-ONLY − FREE): +0.1075 pinball.** corr(hybrid median, physics rollout) = **0.749**.
- **Engineering fixed:** HYBRID (0.25) and PHYS-ONLY (0.32) **vastly beat the broken PHYSFORCE (0.35)**;
  the ODE no longer collapses. **Scientific verdict unchanged: physics does NOT improve accuracy.**
- **Learned physical rates (interpretable, physically ordered):** photolysis 0.3524 ≫ titration 0.0570;
  Arrhenius Ea 0.0938 (positive ⇒ titration faster when warm); deposition dep_NO 0.0337 > dep_NO₂ 0.0273
  > dep_O₃ 0.0205; dilution 0.0186, wet 0.0111; Oₓ-production 0.5218, emission-base 0.5323;
  learned residual cap 0.3406.
- **Per-station emission scale (recovers real city differences):** NARAYANGANJ 3.26, GAZIPUR 1.10 high;
  DARUS SALAM 0.71, KHULNA 0.78, RAJSHAHI 0.60, SYLHET 0.52; AGRABAD 0.37, BARISHAL 0.28 low.
  ⚠️ **BARC = 8.40 is implausible** — almost certainly a sparse-data artefact (BARC n ≈ 1,898; cf. E7);
  do not over-interpret.

---

## 5. Headline numbers (adopted RANK index, test 2016)

| Quantity | Value | Source |
|---|---|---|
| Best probabilistic model (data-only) pinball | **≈ 0.195–0.210** | E8 (λ=0) 0.2054 · E10 FREE 0.195–0.210 |
| CRPS | ≈ 0.044–0.047 | E10 FREE |
| Brier (exceedance, Q30) | ≈ 0.096–0.102 | E8 / E10 FREE |
| Calibrated coverage (CQR) | 80→0.781, 90→0.892, 95→0.951 | E6 |
| Spatial transfer gap (LOSO) | +0.0253 pinball | E7 |
| Test base-rate of the ≥70 % class | 0.834 (Q30) | E5 / E8 |
| Physics effect on accuracy | **neutral-to-negative** (best at λ=0; HYBRID +0.043) | E8, PHYSFORCE, E10 |
| Explainability cross-method agreement | Pearson r = 0.917 | E9 |

---

## 6. Cross-cutting findings & explanations

**(a) The combined index is genuinely combined now, but skill is O₃-led.** The rank index (E5) fixed
the O₃ dominance of the robust index, raising NOx's prediction correlation 0.116→0.348. Yet the
remaining skill is still carried by the pollutant lags themselves (E9: O3_ppb and the NOx-side
lags/masks dominate attribution), because O₃ has a strong, learnable diurnal/photochemical signature
while NOx is noisy and traffic-driven.

**(b) Physics does not improve forecast accuracy — across three independent tests.** (i) E8 λ-sweep:
flat 0→0.2, worse at 0.5. (ii) PHYSFORCE: forcing the ODE as the sole forecaster was much worse
(+0.137). (iii) E10 hybrid: even a stable, well-conditioned, capacity-preserving hybrid is worse
(+0.043), *despite being handed future meteorology as forcing*. The root cause (`PHYSICS_DIAGNOSIS.md`
R6) is structural: predictability lives in O₃/met that the network already learns directly, while the
physics coupling mostly concerns the least-informative (NOx) component. **This is reported as the
result**, not hidden. The QR-PINN's physics is therefore positioned as **interpretable / physics-guided**
(physically-ordered learned rates; per-station emissions; forecast follows the rollout at corr ≈ 0.75),
**not** as an accuracy mechanism — consistent with `CLAUDE.md §1` and the diagnosis's honesty fallback.

**(c) Calibration is the model's strong, defensible property.** Raw quantiles under-cover (E6 raw
PICP90 0.839); **CQR restores near-nominal coverage at all levels and holds across lead time** (E6).
PIT recalibration alone fixed the 80 % band but not the 90 % tail (E4) — which is exactly why the
extended-grid + CQR work (E6) was done.

**(d) The model transfers spatially** with a small, quantified penalty (LOSO gap +0.025, E7), supporting
the "global, not regime-clustered, physics for transferability" design choice. Per-station performance
tracks data volume and cleanliness.

---

## 7. Caveats & limitations (consolidated)

1. **Index change mid-study.** Robust-index runs (baseline, E3, E4) are **not comparable** to
   rank-index runs (E5–E10); different scales. All conclusions use the rank index. (§3)
2. **Rank-index metrics are on [0,1]** and are for **internal comparison only** — not comparable across
   index definitions or to other papers' absolute numbers.
3. **High base rate (≥70 % "extreme").** By construction the positive class is the majority; Brier and
   exceedance skill must be read against the base rate (0.887/0.834/0.766 at Q25/Q30/Q35, E8-C), and the
   upper tail (Q90) is scored separately so "moderate-but-deadly" and "severe" are both covered.
4. **Physics is neutral-to-negative for accuracy** (E8/PHYSFORCE/E10). Any physics claim in the paper
   must be about **interpretability/consistency**, not predictive gain.
5. **Future-met forcing.** The physics variants (PHYSFORCE, E10) use *observed future* meteorology as
   ODE forcing — a perfect-forecast assumption. It advantages physics (and still loses), but a real
   deployment would need forecast met; the physics numbers are therefore optimistic for operations.
6. **Conformal coverage is approximate.** Split-CQR/PIT assume exchangeability that hourly,
   autocorrelated data violate; coverage is empirically good but not guaranteed (E6; ACI/EnbPI = future
   work).
7. **LOSO heterogeneity & bootstrap quirk.** Per-station `n` varies ~70×; small stations (BARC,
   GAZIPUR) dominate the LOSO variance (±0.066). The in-dist point estimate sits marginally below its
   bootstrap CI lower bound — treat CIs as indicative (E7).
8. **Explainability is interpretation, not tuning.** Saliency-on-time-series is known to be unreliable;
   the robustness claim rests on IG↔occlusion agreement (r = 0.917), on a frozen model (E9).
9. **BARC emission scale (8.40)** is an implausible sparse-data artefact (E10); exclude or caveat if
   reported.
10. **Variable epochs (8/10/12)** across runs (robustness scans used 8). Within-run comparisons are
    valid; cross-run *absolute* values are not fully converged-comparable. Loss curves are
    monotone-improving where physics-loss annealing is not active.
11. **Missingness.** O₃/NOx are heavily missing; training is on observed `H` only, with masking. Results
    are conditioned on hours where the target exists.

---

## 8. Reproducibility

Seeds fixed (numpy/torch); train-only scalers, percentile threshold (`thr = Q₃₀ = 0.3948`), and the
`H`-definition persisted as artefacts (`artefacts/qrpinn_meta.json`, `qrpinn_data.npz`); no leakage
(every transform fit on 2014–15 only). Each stage writes machine-readable JSON
(`results_e*.json`) consumed verbatim by the report. Code: `pipeline/02–11_*.py`. Source-of-truth log:
`RESULTS_LOG.md` (run IDs cited throughout this document).

---

## 9. What is defensible to claim (for the IGARSS manuscript)

- ✅ A QR-PINN that emits the **full predictive distribution** of a balanced combined NOx–O₃ hazard
  index at hourly, 24 h-ahead resolution, **strongly beating climatology** and matching a pure QRNN.
- ✅ **Well-calibrated** probabilistic forecasts via CQR (near-nominal at 80/90/95 %, stable across
  leads), with proper-scoring evaluation.
- ✅ **Spatial transferability** quantified by LOSO with bootstrap CIs.
- ✅ **Explainability** corroborated by two independent methods (IG + occlusion, r = 0.917).
- ✅ **Interpretable, physically-ordered learned chemistry** (photolysis ≫ titration; warm-biased
  Arrhenius; per-station emission scales) from the E10 hybrid.
- ❌ **Do not claim physics improves forecast accuracy** — it does not, on this O₃-dominated index
  (E8/PHYSFORCE/E10). Frame physics as interpretability/consistency, with the negative accuracy result
  reported honestly.
