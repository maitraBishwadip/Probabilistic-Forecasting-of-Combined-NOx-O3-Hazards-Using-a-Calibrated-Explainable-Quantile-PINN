# RESULTS LOG — source of truth for the paper (see CLAUDE.md)

> 🛠 **EXTREME-DEFINITION CORRECTION (2026-06-20 — see run `THRCORR-Q70` at the bottom).** The
> "extreme/hazardous" class was originally defined as `H ≥ Q30` (which wrongly labels ~70–83 % of
> hours "extreme"). It is **corrected to the most-polluted top 30 % = `H ≥ Q70(H_train) = 0.6087`**,
> with `Q90 = 0.7611` as a "severe" tier. This affects **only** the exceedance probability,
> **Brier(exceedance)**, and the **base rate**; pinball / CRPS / PICP / CQR / LOSO are
> threshold-independent and unchanged. In every run block **below**, any `Q30` Brier/base-rate is a
> historical record at the inverted cut and is **superseded** by the `THRCORR-Q70` re-score.

## Run QRPINN-20260615-212045
- date: 2026-06-15T21:20:45
- code: pipeline/02_qrpinn_dataprep.py + pipeline/03_qrpinn_model.py
- config: L=48, HZ=24, taus=[0.05000000074505806, 0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.5, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 0.949999988079071], epochs=12, batch=512, lr=0.001, lam_data=0.3, lam_phys_max=0.2, thr_Q30=-0.0469
- split: train 2014-15 (86037 anchors) / test 2016 (31067 anchors); seed=0
- target: combined index H=0.5 rs(NOx)+0.5 rs(O3); extreme = H>=Q30(train) (~70% class)

| model | pinball | CRPS | PICP80 | PICP90 | width80 | Brier(exceed) | Brier@h1 |
|---|---|---|---|---|---|---|---|
| climatology | 1.9256 | 0.4279 | 0.693 | 0.764 | 1.749 | 0.2089 | 0.2063 |
| QRNN | 1.0571 | 0.2349 | 0.663 | 0.804 | 0.702 | 0.0989 | 0.0579 |
| QR-PINN | 1.0527 | 0.2339 | 0.667 | 0.808 | 0.721 | 0.0997 | 0.0568 |

- QR-PINN learned physics params (softplus>0): P_o=0.436, L_titr=0.175, D_o=0.169, W_o=0.169, E_n=0.170, D_n=0.156, W_n=0.156
- nominal interval coverage targets: PICP80=0.80, PICP90=0.90
- NOTE: first-version box model (coupled NOx-O3, within-window, leakage-free); full Leighton/Ox residual + LOSO are later runs.

## Run E3-20260615-222715  (E3: physics ablation + dominance)
- date: 2026-06-15T22:27:15 | code: pipeline/04_qrpinn_e3.py
- config: L=48, HZ=24, epochs=12, lam_phys_max=0.2, thr=-0.0469, seed=0
- split: train 2014-15 (86037) / test 2016 (31067)

| model | pinball | pin_mean | CRPS | PICP80 | PICP90 | width80 | Brier | Brier@1 |
|---|---|---|---|---|---|---|---|---|
| QRNN | 1.0571 | 0.1175 | 0.2349 | 0.663 | 0.804 | 0.702 | 0.0989 | 0.0579 |
| DATA-ONLY | 1.0662 | 0.1185 | 0.2369 | 0.676 | 0.816 | 0.736 | 0.0983 | 0.0559 |
| FULL | 1.0527 | 0.1170 | 0.2339 | 0.667 | 0.808 | 0.721 | 0.0997 | 0.0568 |

- physics value = FULL - DATA-ONLY: pinball +0.0135, CRPS +0.0030
- FULL learned physics params: P_o=0.436, L_titr=0.175, D_o=0.169, W_o=0.169, E_n=0.170, D_n=0.156, W_n=0.156

- DOMINANCE: Var(H) share NOx=44.1% / O3=72.8% / cov=-16.9%; corr(H,NOx)=0.536 corr(H,O3)=0.754; pred corr NOx=0.116/O3=0.898; **dominant = O3** (overpowers the combined index).
- combined predictive distribution saved: artefacts/e3_full_test_quantiles.npz; fig: figs/e3_combined_pdf_dominance.png

## Run E4-20260615-225005  (E4: PIT recalibration of FULL)
- date: 2026-06-15T22:50:05 | code: pipeline/05_qrpinn_e4.py
- split: train 2014-15 (86037); calib=early-2016 (12426); eval=late-2016 (18641); seed=0
- method: PIT distribution recalibration (Kuleshov et al. 2018) on the FULL QR-PINN

| state | pinball | CRPS | PICP80 | PICP90 | width80 | Brier(exceed) |
|---|---|---|---|---|---|---|
| before | 0.9342 | 0.2076 | 0.678 | 0.821 | 0.679 | 0.1183 |
| after  | 0.9389 | 0.2086 | 0.765 | 0.821 | 0.899 | 0.1191 |

- recalibration map (nominal->p*): 0.05->0.00, 0.10->0.09, 0.20->0.19, 0.30->0.33, 0.50->0.58, 0.70->0.83, 0.80->0.92, 0.90->1.00, 0.95->1.00
- checkpoint: artefacts/qrpinn_full.pt; reliability fig: figs/e4_reliability.png
- NOTE: nominal coverage targets PICP80=0.80, PICP90=0.90; eval is the held-out late-2016 half.

## Run E5-20260615-230704  (E5: rank/quantile-uniform combined index)
- date: 2026-06-15T23:07:04 | code: pipeline/06_qrpinn_e5.py
- new target: H_rank = 0.5 F_NOx(NOx) + 0.5 F_O3(O3) (train empirical CDF); thr=Q30=0.3948
- split: train 2014-15 (86037) / test 2016 (31067); seed=0

| index | Var-share NOx | Var-share O3 | corr(H,NOx) | corr(H,O3) |
|---|---|---|---|---|
| robust (old) | 44.1% | 72.8% | 0.536 | 0.754 |
| rank (new) | 61.0% | 63.3% | 0.625 | 0.643 |

| prediction-level | corr(pred H, NOx) | corr(pred H, O3) |
|---|---|---|
| robust (E3) | 0.116 | 0.898 |
| rank (E5) | 0.348 | 0.629 |

- FULL-rank metrics (rank target, NOT comparable to robust-target runs): pinball=0.197, crps=0.0438, picp80=0.7083, width80=0.1711, picp90=0.8431, width90=0.24, brier_exceed=0.0972, test_base_rate_extreme=0.8343
- checkpoint artefacts/qrpinn_full_rank.pt ; fig figs/e5_dominance.png

## Run E6-20260615-233855  (E6: tail calibration — extended grid + PIT + CQR, rank index)
- date: 2026-06-15T23:38:55 | code: pipeline/07_qrpinn_e6.py | method: METHODS_calibration.md
- tau grid (13): [0.009999999776482582, 0.02500000037252903, 0.05000000074505806, 0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.5, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 0.949999988079071, 0.9750000238418579, 0.9900000095367432]
- split: train 2014-15 (86037); calib early-2016 (12426); eval late-2016 (18641); seed=0
- raw: pinball=0.2197 CRPS=0.0338 Brier(exc)=0.1204; tail pinball .9/.95/.99 = 0.0158/0.0099/0.0034

| level | nominal | PICP raw | PICP PIT | PICP CQR | width raw | width PIT | width CQR |
|---|---|---|---|---|---|---|---|
| 80% | 0.80 | 0.725 | 0.760 | 0.781 | 0.184 | 0.200 | 0.208 |
| 90% | 0.90 | 0.839 | 0.875 | 0.892 | 0.241 | 0.273 | 0.282 |
| 95% | 0.95 | 0.923 | 0.936 | 0.951 | 0.321 | 0.348 | 0.365 |

- CQR PICP90 by lead h=1/6/12/24: 0.893/0.887/0.902/0.900
- checkpoint artefacts/qrpinn_full_rank_ext.pt ; figs e6_reliability.png, e6_coverage.png
- NOTE: split-CQR coverage is approximate under time-series non-exchangeability (METHODS_calibration.md sec.3); ACI/EnbPI = future work.

## Run E7-20260616-033822  (E7: LOSO spatial transfer + block-bootstrap CIs)
- date: 2026-06-16T03:38:22 | code: pipeline/08_qrpinn_e7.py | method: METHODS_transfer.md
- model: STATION-AGNOSTIC rank-index QR-PINN (physics on); epochs=8; seed=0; rank index H
- bootstrap: moving-block (block=24h, B=1000); 95% CI

| setting | pinball [95% CI] | CRPS [95% CI] | PICP80 | PICP90 | Brier(exc) |
|---|---|---|---|---|---|
| in-distribution (all-9) | 0.2024 [0.2025,0.2140] | 0.0450 [0.0450,0.0475] | 0.718 | 0.875 | 0.0979 |
| LOSO mean±std (9 folds) | 0.2277 ± 0.0655 | 0.0506 ± 0.0146 | 0.700 | 0.859 | 0.0985 |
| LOSO pooled (CI) | [0.2025,0.2132] | [0.0450,0.0474] | | | |

| held-out station | pinball | CRPS | PICP90 | Brier(exc) | n |
|---|---|---|---|---|---|
| AGRABAD | 0.2328 | 0.0517 | 0.832 | 0.1257 | 93415 |
| BARC | 0.3471 | 0.0771 | 0.814 | 0.0561 | 1898 |
| BARISHAL | 0.1442 | 0.0320 | 0.971 | 0.1112 | 16020 |
| DARUS SALAM | 0.2065 | 0.0459 | 0.880 | 0.1045 | 123584 |
| GAZIPUR | 0.3167 | 0.0704 | 0.713 | 0.1529 | 7244 |
| KHULNA | 0.1813 | 0.0403 | 0.877 | 0.0093 | 68520 |
| NARAYANGANJ | 0.1475 | 0.0328 | 0.958 | 0.0770 | 131176 |
| RAJSHAHI | 0.2482 | 0.0551 | 0.833 | 0.1517 | 76134 |
| SYLHET | 0.2245 | 0.0499 | 0.850 | 0.0978 | 106697 |

- transfer gap (LOSO mean − in-dist) pinball = +0.0253
- fig figs/e7_loso.png ; NOTE rank-index metrics on [0,1], not comparable across index defs; spatial-CV caveats in METHODS_transfer.md sec.5.

## Run E8-20260616-040817  (E8: robustness — physics-weight sweep, seed, threshold)
- date: 2026-06-16T04:08:17 | code: pipeline/09_qrpinn_e8.py | rank index; epochs=8

**(A) Physics-weight sweep (seed 0)**
| lam_phys | pinball | CRPS | PICP90 | Brier(Q30) |
|---|---|---|---|---|
| 0.0 | 0.2054 | 0.0456 | 0.858 | 0.1023 |
| 0.2 | 0.2063 | 0.0458 | 0.860 | 0.1014 |
| 0.5 | 0.2188 | 0.0486 | 0.850 | 0.1052 |

**(B) Seed stability (lam=0.2)**
| seed | pinball | CRPS | PICP90 | Brier(Q30) |
|---|---|---|---|---|
| 0 | 0.2063 | 0.0458 | 0.860 | 0.1014 |
| 1 | 0.2060 | 0.0458 | 0.854 | 0.0994 |

**(C) Threshold sensitivity (lam=0.2 seed0; exceedance only, no retrain)**
| cutoff | thr(H) | test base-rate | Brier(exceed) |
|---|---|---|---|
| Q25 | 0.361 | 0.887 | 0.0761 |
| Q30 | 0.395 | 0.834 | 0.1014 |
| Q35 | 0.426 | 0.766 | 0.1260 |

- NOTE: rank-index metrics on [0,1]; internal comparisons only. 8 epochs (robustness scan).

## Run E9-20260616-085543  (E9: explainability — Integrated Gradients + occlusion, frozen ext model)
- date: 2026-06-16T08:55:43 | code: pipeline/10_qrpinn_e9_xai.py | method: METHODS_explainability.md
- subject: FROZEN artefacts/qrpinn_full_rank_ext.pt (rank index, physics on, 13-tau); POST-HOC, no retrain; seed=0
- IG: baseline=missing(values=0,masks=0), M=32, N=1280 stratified test-2016 anchors (640 day/640 night); completeness gap(med h1)=0.0293
- occlusion: mean-occlude on all 31067 test-2016 anchors; intact pinball=0.2165 crps=0.0333 brier(exc)=0.1012
- **IG vs occlusion agreement across groups: Pearson r = 0.917** (independent corroboration)

- top-5 IG (median h=1): O3_ppb 0.207, NOx_ppb_mask 0.109, O3_ppb_mask 0.092, NO2_ppb_mask 0.068, NO_ppb_mask 0.063
- top-5 occlusion dPinball: O3_ppb +0.0928, NOx_ppb +0.0563, NO2_ppb_mask +0.0441, NO_ppb +0.0311, NO2_ppb +0.0150

| group | IG median_h1 | IG tail_h1 | IG width_h1 | IG median_h24 | occ dPinball | occ dBrier |
|---|---|---|---|---|---|---|
| NOx_side | 0.375 | 0.343 | 0.292 | 0.292 | +0.2225 | +0.0372 |
| O3_side | 0.298 | 0.308 | 0.376 | 0.356 | +0.0924 | +0.0420 |
| photochem | 0.064 | 0.069 | 0.051 | 0.074 | +0.0027 | +0.0017 |
| dispersion | 0.076 | 0.078 | 0.078 | 0.073 | +0.0004 | -0.0010 |
| wet_removal | 0.013 | 0.013 | 0.027 | 0.015 | +0.0008 | +0.0015 |
| pressure | 0.018 | 0.020 | 0.016 | 0.023 | +0.0031 | +0.0035 |
| satellite | 0.003 | 0.003 | 0.004 | 0.004 | -0.0000 | +0.0003 |
| calendar | 0.153 | 0.164 | 0.156 | 0.163 | +0.0175 | +0.0098 |

- station permutation (->id0): dPinball=+0.0013, dBrier=+0.0004
- day/night (median h1, IG frac): photochem day=0.067/night=0.061; NOx_side day=0.364/night=0.387; O3_side day=0.307/night=0.290
- figs (png+pdf, 300dpi): e9_group_importance, e9_distributional, e9_temporal_heatmap, e9_daynight, e9_leadtime, e9_feature_importance
- NOTE: saliency-on-time-series caveat (Ismail et al. 2020) — IG (axiomatic, completeness-checked) cross-validated with occlusion in loss currency; agreement (r above) = robustness. Interpretation only, frozen model.

## Run PHYSFORCE-20260616-093019  (physics-forcing trial — SEPARATE from the planned E9-explainability)
- date: 2026-06-16T09:30:19 | code: pipeline/10b_physforce_trial.py | results_physforce.json | rank index; epochs=12; seed=0
- design: forecast median = sunlight-driven box-ODE rollout (J=photochemical index); spread head for quantiles; lam_c=0.3

| variant | pinball | CRPS | PICP80 | PICP90 | Brier(exc) |
|---|---|---|---|---|---|
| physics-forced | 0.3479 | 0.0773 | 0.790 | 0.862 | 0.2123 |
| no-physics (free) | 0.2104 | 0.0467 | 0.662 | 0.775 | 0.1026 |

- delta (physics − free): pinball +0.1375, CRPS +0.0306, Brier +0.1097
- learned rates (softplus>0): prod=2.103, titr=0.362, depO=0.371, dil=0.127, wet=0.088, emis=2.237, conv=0.075
- **VERDICT: physics WORSE** (threshold ~0.003 pinball / 0.0015 CRPS vs E8 noise)
- fig figs/physforce.png ; NOTE rank-index [0,1] metrics; future met used as forcing (forecast-met assumption).
- CONTEXT: this trial answered "does forcing physics help?" (no — physics WORSE). It is NOT the E9 in EXPERIMENT_DESIGN (=explainability, run E9-20260616-085543 above). Mislabeled "E9" at run time; relabeled here.

## Run E10-20260616-104625  (E10: physics-GUIDED HYBRID — NN-residual forecast + interpretable NO/NO2/O3 box-ODE)
- date: 2026-06-16T10:46:25 | code: pipeline/11_qrpinn_e10.py | rank index; epochs=10; seed=0(+1)
- design: H_med = H_phys(box-ODE rollout, ppb, semi-implicit) + bounded NN residual; chemistry NO2+hv->NO+O3, O3+NO->NO2; Leighton residual + structural Ox=O3+NO2 conservation; lam_c=0.3, lam_leighton=0.05

**Test-2016 (seed 0)**
| variant | pinball | CRPS | PICP80 | PICP90 | Brier(exc) |
|---|---|---|---|---|---|
| FREE (no physics) | 0.2099 | 0.0466 | 0.679 | 0.788 | 0.1005 |
| HYBRID (E10) | 0.2525 | 0.0561 | 0.711 | 0.826 | 0.1197 |
| PHYS-ONLY | 0.3174 | 0.0705 | 0.670 | 0.772 | 0.1370 |

**Seed stability (seed 1)**
| variant | pinball | CRPS | PICP90 | Brier(exc) |
|---|---|---|---|---|
| FREE | 0.1951 | 0.0434 | 0.811 | 0.0960 |
| HYBRID | 0.2598 | 0.0577 | 0.795 | 0.1200 |

- delta (HYBRID - FREE): pinball +0.0426, CRPS +0.0095, Brier +0.0192
- delta (PHYS-ONLY - FREE): pinball +0.1075, CRPS +0.0239, Brier +0.0365
- corr(hybrid median, physics rollout) = 0.749 (how much the forecast follows physics)
- learned global rates (softplus>0): photolysis_kj=0.3524, titration_ktit0=0.0570, arrhenius_Ea=0.0938, dep_NO=0.0337, dep_NO2=0.0273, dep_O3=0.0205, dilution_VC=0.0186, wet_scav=0.0111, Ox_production=0.5218, emission_base=0.5323, resid_scale_cap=0.3406
- per-station emission scale: AGRABAD=0.37, BARC=8.40, BARISHAL=0.28, DARUS SALAM=0.71, GAZIPUR=1.10, KHULNA=0.78, NARAYANGANJ=3.26, RAJSHAHI=0.60, SYLHET=0.52
- **VERDICT: physics WORSE** (threshold ~0.003 pinball / 0.0015 CRPS vs E8 noise)
- fig figs/e10_physics.png ; NOTE rank-index [0,1] metrics; future met used as forcing (forecast-met assumption).

## Run E12-20260618-005147  (E12: regime-stratified physics value — is there ANY regime where physics helps?)
- date: 2026-06-18T00:51:47 | code: pipeline/12_qrpinn_e12.py | rank index; seed=0 | stratified RE-SCORING of deterministically reproduced E10 models (no new design; predictions cached to artefacts/e12_preds.npz; LOSO deferred to E13)
- motivation: pooled E10 says physics ≠ accuracy, but PINN gains are known to concentrate in specific regimes (sparse/gappy, upper tail, long lead — AirPhyNet ICLR'24; Krishnapriyan NeurIPS'21). Test each regime separately.
- reproduction check (pinball Στ vs E10 seed0): free 0.2099 (E10 0.2099), physonly 0.3174 (E10 0.3174), hybrid 0.2525 (E10 0.2525) — EXACT (d=-0.0000), models identical to E10.

**Overall (test 2016, seed 0)** — pinball = SUM over 9 τ (E10 convention); τ=0.90/0.95 are per-quantile
| variant | pinball (Στ) | τ=0.90 | τ=0.95 | Brier@Q30 | Brier@Q90 |
|---|---|---|---|---|---|
| FREE | 0.2099 | 0.0155 | 0.0096 | 0.1005 | 0.0338 |
| HYBRID | 0.2525 | 0.0195 | 0.0120 | 0.1197 | 0.0470 |
| PHYS-ONLY | 0.3174 | 0.0210 | 0.0126 | 0.1370 | 0.0648 |

**Regime slices, HYBRID − FREE pinball (Στ); band ±0.003; negative = physics helps**
- upper tail (proper): τ=0.95 +0.0024 (worse-ish), Brier@Q90 FREE 0.034 < HYBRID 0.047 < PHYS 0.065 → physics worse in the tail.
- extreme-outcome subset (H≥Q90): FREE 0.354, HYBRID 0.415, PHYS-ONLY 0.333. CAVEAT — conditioning on the outcome rewards PHYS-ONLY's positive bias; NOT physics skill (proper tail metrics above favor FREE).
- lead time h=1→24: gap +0.030 (best, h≈11) → +0.058 (h=24); worse at EVERY lead, no long-lead crossover.
- by-lead pinball (FREE), absolute: h=1 0.141 → h=24 0.217 (from results_e12.json `by_lead.free`; rises with lead). Used in paper Table `tab:robust`.
- input-window missingness low/mid/high: +0.046 / +0.041 / +0.040 — gap narrows with gaps (theoretically-expected direction) but never reaches parity.
- per-station (9): 7 worse; GAZIPUR −0.0005 (tie), RAJSHAHI −0.005 (marginal). Both inside the E10 seed0↔seed1 noise (~0.01 Στ).

- **VERDICT: NO robust regime where physics improves accuracy.** Auto-flag found only station=GAZIPUR (tie) and station=RAJSHAHI (−0.005, within seed noise). Confirms E10 at the regime level; AirPhyNet's sparse/long-lead advantage does NOT replicate here. Consistent with PHYSICS_DIAGNOSIS R6 (O3/met-dominated predictability) + the structural fact that the implemented Leighton triad is a closed null cycle (conserves Oₓ, zero net O3) so it cannot generate the O3 extremes being scored.
- **Implication:** physics stays positioned as interpretable/physics-guided (per CLAUDE.md integrity fallback), NOT an accuracy mechanism. Next lever = add HCHO (VOC/RO₂ proxy + HCHO/NO₂ regime ratio) so the box-ODE can PRODUCE O3 → planned E13.
- fig figs/e12_strata.png ; artefacts/results_e12.json + artefacts/e12_preds.npz ; future met used as forcing (forecast-met assumption); rank-index [0,1] metrics.

## Run THRCORR-Q70-20260620-032931  (EXTREME-DEFINITION CORRECTION: top-30% = H>=Q70; supersedes the inverted H>=Q30)
- date: 2026-06-20T03:29:31 | code: pipeline/12b_threshold_q70.py | rank index; re-score of cached E10/E12 predictions (artefacts/e12_preds.npz); NO retraining; NO synthetic data
- CORRECTION: 'extreme/hazardous' was wrongly defined as H>=Q30 (~70-83% of hours). Corrected to the
  most-polluted top 30% = H>=Q70; Q90 kept as a secondary 'severe' tier. ALL prior Brier(exceedance)/
  base-rate numbers in earlier run blocks were computed at the inverted Q30 cut and are SUPERSEDED here.
- THRESHOLD-INDEPENDENT metrics (pinball, CRPS, PICP, CQR coverage, LOSO transfer) are UNCHANGED -- they
  never used the threshold (model trains on the continuous H, not the binary label).

- thresholds on train H: Q30(old)=0.3948  **Q70=0.6087**  Q80=0.6728  Q90=0.7611
- test base rate (observed): Q30(OLD)=0.834, Q70=0.263, Q80=0.134, Q90=0.046

**Brier(exceedance), test 2016, seed 0 -- re-scored at corrected cuts**
| cut | base rate | FREE | HYBRID | PHYS-ONLY |
|---|---|---|---|---|
| Q30(OLD) | 0.834 | 0.1005 | 0.1197 | 0.1370 |
| Q70 | 0.263 | 0.1101 | 0.1376 | 0.1940 |
| Q80 | 0.134 | 0.0747 | 0.0976 | 0.1445 |
| Q90 | 0.046 | 0.0338 | 0.0470 | 0.0648 |

- **VERDICT unchanged & threshold-robust:** FREE < HYBRID < PHYS-ONLY at every cut (Q70 delta HYBRID-FREE = +0.0275, PHYS-ONLY-FREE = +0.0839). Correcting the extreme cut does not rescue the physics for accuracy.
- NOTE: rank-index [0,1] metrics; thresholds fit on TRAIN only (leakage-free).
