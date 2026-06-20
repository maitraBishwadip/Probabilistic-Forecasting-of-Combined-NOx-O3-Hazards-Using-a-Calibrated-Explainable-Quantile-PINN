# WRITEUP_PLAN.md — InGARSS 2026 paper plan (QR-PINN, combined NOx–O₃ extremes)

**Purpose.** Single entry-point for drafting the manuscript. It fixes the *venue rules, the story,
the hypotheses, the section/equation/table/figure layout, the writing style, and exactly which file
+ run-ID to pull every number from.* Write the `.tex` against this file only; it points to the
others when a specific number is needed.

> **Integrity (from `CLAUDE.md §1`).** Every number/table/figure in the `.tex` must come from
> `RESULTS_LOG.md` with its run ID. Every literature claim must point to `LITERATURE_REVIEW.md`.
> No placeholder/"expected" values. The negative physics result is reported honestly, not hidden.
> Generating any *new* number or re-running a model is an experiment → **EXPLAIN → APPROVE → RUN**
> (`CLAUDE.md §0`). Re-plotting figures from existing `results_e*.json` is reproduction, not a new
> experiment, but see the open gaps in §12.

---

## 1. Venue & Track 06 (verified 2026-06-16)

| Item | Detail |
|---|---|
| Symposium | **IEEE India Geoscience and Remote Sensing Symposium — InGARSS 2026** (GRSS-IEEE) |
| Location / dates | **Hyderabad, India · 1–4 December 2026** |
| Theme | *"Digital Earth — Modeling, Mapping, and Monitoring"* |
| Submission system | **EDAS**; PDF; **standard IEEE Conference Template** (two-column) |
| Paper deadline | **30 June 2026** (today is 2026-06-16 → ~2 weeks) |
| AI-content rule | Must disclose AI-generated content (**<25 %**) + originality/plagiarism check |

**Track 06 — Machine Learning & AI for Digital Earth.** *AI/ML for modelling, detection,
prediction, and mapping. Topics: deep learning for image classification and segmentation,
**physics-informed machine learning**, AI for SAR and hyperspectral analysis, self-supervised and
transfer learning, **explainable AI for geospatial applications**.*

**Why we fit this track (lead with these three keywords):**
1. **Physics-informed ML** — QR-PINN embeds NOx–O₃ photochemistry (Leighton + Oₓ conservation).
2. **Explainable AI** — Integrated Gradients + occlusion audit of the distributional forecast (E9).
3. **Prediction + transfer learning** — probabilistic 24-h forecast + Leave-One-Station-Out spatial
   transfer (E7). Remote-sensing framing: reanalysis met + MODIS AOD/NDVI/fire as physical drivers.

> ⚠️ **Page-limit discrepancy — decide before drafting.** InGARSS 2026 officially allows **up to 7
> pages (6 content + 1 references)**, overlength charges on pages 6–7. You instructed **5 pages incl.
> citations**. This plan is budgeted to **5 pages** (conservative, cheaper). You have ~1.5 pages of
> headroom if you later choose to use it; expansion points are flagged **[+EXPAND]** below.

Sources: [InGARSS 2026 CfP (GRSS-IEEE)](https://www.grss-ieee.org/resources/news/call-for-papers-ingarss-2026/) ·
[InGARSS 2026 submit page](https://ingarss2026.co.in/submit)

---

## 2. The story (the spine of the paper)

**One-line thesis.** *We build the first probabilistic, physics-informed forecast of a **combined
NOx–O₃ hazard** over Bangladesh — emitting the full 24-h predictive distribution, well-calibrated by
conformal QR, spatially transferable, and explainable — and we report the honest, instructive finding
that embedded photochemistry aids **interpretability, not accuracy**, with XAI explaining why.*

**Narrative arc (5 beats — keep the paper on this line):**
1. **Hook / gap.** Bangladesh is among the most polluted countries; co-exposure to NOx + O₃ (oxidant
   Oₓ) is *synergistically* harmful, yet no forecast targets the **combined** hazard, none is
   **probabilistic + physics-informed**, and none is for Bangladesh. (LIT §2,§6,§7,§8,§10)
2. **Idea.** A QR-PINN that outputs the **whole predictive distribution** of a balanced combined
   hazard index `H`, with NOx–O₃ chemistry as a soft physics constraint. (DESIGN §2,§3,§5)
3. **It works as a forecaster.** Beats climatology, non-crossing quantiles, **calibrated** (CQR
   near-nominal at 80/90/95, stable across lead), **transfers** across unseen stations (LOSO gap
   +0.025). (E6, E7)
4. **The twist (honest result).** We *hypothesised* physics would sharpen accuracy; across **three
   independent tests** (λ-sweep, physics-forcing, hybrid) it is **neutral-to-negative**. (E8, PHYSFORCE,
   E10)
5. **Why — and the reframe.** XAI (IG + occlusion, agreement **r = 0.917**) shows skill is carried by
   **pollutant lags + calendar**, not meteorology, on an **O₃-led** index — so physics can't add
   accuracy here. We therefore reposition physics as **interpretability**: physically-ordered learned
   rates (photolysis ≫ titration) and recovered per-station emission scales. (E9, E10)

**Tone:** a *rigorous, honest* probabilistic-forecasting + XAI paper. The negative physics result is a
**feature** (a clean, reproducible scientific finding aligned to Track 06's XAI + physics-ML themes),
not a weakness to bury. Do **not** claim physics improves accuracy (`RESULTS_AND_REPORT.md §9`).

---

## 3. Hypotheses (state explicitly; map each to evidence)

| # | Hypothesis | Verdict | Evidence (run) |
|---|---|---|---|
| **H1** | A QR-PINN emits a valid (non-crossing), **calibrated** full predictive distribution of `H` at hourly/24-h lead, beating climatology. | **Supported** | baseline, E6 |
| **H2** | Embedding NOx–O₃ photochemistry **improves probabilistic accuracy**. | **Refuted (honest negative)** | E8, PHYSFORCE, E10 |
| **H3** | A rank/quantile-uniform `H` makes the index **genuinely combined** (neither pollutant dominates). | **Supported** | E5 |
| **H4** | The model **transfers** to unseen stations with a bounded penalty. | **Supported (+0.025 pinball)** | E7 |
| **H5** | Two independent XAI methods **agree**, and skill is driven by pollutant lags + calendar (not met) — explaining H2. | **Supported (r = 0.917)** | E9 |

H2 is the deliberate, falsifiable centerpiece; H5 closes the loop on *why* H2 failed.

---

## 4. Section layout & page budget (5 pages, IEEE two-column)

| § | Section | Budget | Core content | Pull from |
|---|---|---|---|---|
| — | Title, authors, **Abstract** (~150 w), **Index Terms** | 0.25 p | thesis + 1 headline number + honest physics note | §2 here |
| I | **Introduction** | 0.75 p | motivation (health/Bangladesh), gap (4 gaps), contributions (bullet list of 4) | LIT §2,6,7,8,10 |
| II | **Data & Study Area** | 0.4 p | 9 DoE stations, hourly 2014–16, 26 features, missingness + masking, splits | REPORT §1; DESIGN §4 |
| III | **Method: QR-PINN** | 1.25 p | index `H`; encoder; monotone quantile head; physics head + residuals; composite loss | DESIGN §2,3,5 |
| IV | **Evaluation protocol** | 0.35 p | pinball/CRPS/PICP/Brier; PIT + **CQR**; LOSO + bootstrap; XAI (IG+occlusion) | DESIGN §6,9; METHODS_* |
| V | **Results & Discussion** | 1.4 p | T-I skill, T-II calibration, transfer, **physics negative result**, XAI | RESULTS_LOG (E5–E10) |
| VI | **Conclusion** (+ limits/future) | 0.25 p | what's defensible; CQR/ACI + forecast-met as future work | REPORT §7,§9 |
| — | **References** | ~0.75 p | IEEE numeric, ~22–26 refs | LITERATURE_REVIEW |

**[+EXPAND] if 6+1 pages allowed:** promote the XAI to its own subsection with the day/night physics
audit; add the LOSO per-station table (T-III) and the e10 learned-rates panel; widen Related Work.

---

## 5. Section-by-section content (what to actually write)

**Abstract.** 1 sentence gap → 1 method → 1 results headline (calibrated, transfers) → **1 honest
sentence** (physics aids interpretability not accuracy) → 1 significance. Put **one** concrete number
(e.g., CQR 90 %→0.892, or LOSO gap +0.025). *Index Terms:* probabilistic forecasting, physics-informed
neural networks, quantile regression, explainable AI, air quality, Bangladesh.

**I. Introduction.** (a) Bangladesh + combined-oxidant health hook (LIT §7,§8). (b) Why *probabilistic*
+ *combined* + *physics-informed* + *Bangladesh* is unaddressed — the 4 gaps from LIT §10. (c)
**Contributions** as a 4-item bullet list mirroring §9 "defensible claims" in REPORT: (1) first
probabilistic combined NOx–O₃ hazard forecast w/ full distribution; (2) calibrated via CQR, transfers
(LOSO); (3) XAI audit (IG+occlusion); (4) **honest physics finding** + interpretable learned chemistry.

**II. Data & Study Area.** `BD_DOE_2014-16.csv`; 9 stations; hourly; 223,776 raw rows → 9×26,304×26
tensor → **86,037 train / 31,067 test** anchors; 48-h window. Missingness (O₃≈74.7 %, NOx≈62.6 %
populated) → **masking + never impute target**. Splits: temporal (train 2014–15 / test 2016) + LOSO.
**→ Fig 1 (study-area map).**

**III. Method.** Define `H` (Eq 1) and exceedance/threshold (Eq 2). Encoder = LSTM(64) + station
embedding(8). Monotone non-crossing quantile head (Eq 3) on τ-grid {0.05…0.95}→{0.01…0.99}. Physics
head: box mass-balance + **Leighton PSS** + **Oₓ=O₃+NO₂ conservation** residuals (Eq 4–5). Composite
loss with annealed `λ_phys` (Eq 6). State clearly: physics is a **soft regulariser / interpretability
device**, quantile head is the deliverable. **→ Fig 2 (architecture schematic).**

**IV. Evaluation.** Proper scores: pinball, CRPS (from quantile grid), Brier on exceedance
`P(H≥thr)`; calibration: PICP/width, PIT (Kuleshov 2018), **CQR (Romano 2019)**; transfer: LOSO +
moving-block bootstrap 95 % CI; XAI: Integrated Gradients (completeness-checked) + occlusion in
loss-currency, agreement = robustness (Ismail 2020 caveat).

**V. Results & Discussion.**
- *Distributional skill* → **Table I**: climatology vs QRNN(data-only/FREE) vs QR-PINN(hybrid),
  rank index, test 2016. State "beats climatology, matches QRNN." ⚠️ see §12 climatology gap.
- *Calibration* → **Table II** + **Fig 3**: raw under-covers; **CQR restores 80/90/95** and holds
  across lead (h=1/6/12/24 ≈ 0.90).
- *Spatial transfer* (E7): LOSO mean 0.2277 ± 0.0655 vs in-dist 0.2024; **gap +0.0253**; performance
  tracks station data volume; bootstrap-CI quirk caveat. (T-III if [+EXPAND].)
- *Physics — the honest result* (E8/PHYSFORCE/E10): λ-sweep flat→worse; forcing +0.137; hybrid +0.043
  *despite future-met forcing*. One crisp sentence + one number. Root cause = O₃/met-led predictability
  the net already learns (PHYSICS_DIAGNOSIS R6).
- *Why → XAI* → **Fig 4**: group importance + **IG↔occlusion r = 0.917**; drivers = pollutant lags +
  calendar, met/photochem small → consistent with physics being inessential to *skill*.
- *Interpretability payoff* (E10): physically-ordered learned rates (photolysis 0.352 ≫ titration
  0.057; dep_NO>dep_NO₂>dep_O₃) + per-station emission scales recover real city differences
  (NARAYANGANJ high; ⚠️ exclude/caveat BARC=8.40 artefact).

**VI. Conclusion.** Restate the reframe (calibrated, transferable, explainable probabilistic forecast;
physics = interpretability). Future work: ACI/EnbPI for non-exchangeable coverage; forecast-met (drop
perfect-met assumption); richer emissions. (REPORT §7,§9)

---

## 6. Equations to include (numbered; LaTeX-ready)

Keep **6 numbered equations** for a 5-page paper (mark *inline* ones to compress if tight).

1. **Hazard index** — `H = ½ F_NOx(NOx) + ½ F_O3(O3)`, `F.` = train empirical CDF (rank/quantile-uniform). *(DESIGN §2; E5)*
2. **Extreme / exceedance** — `thr = Q_{70}(H_train)=0.6087` (corrected: most-polluted top 30 %; Q90=0.7611 severe); `p_{t+h}=P(H_{t+h}≥thr)=1−\hat F_{t+h}(thr)`. *(DESIGN §2; corrected — RESULTS_LOG THRCORR-Q70)*
3. **Pinball + non-crossing head** — `ρ_τ(u)=u(τ−1{u<0})`; `Q_{τ_k}=Q_{τ_1}+Σ_{j≤k}softplus(δ_j)` (monotone). *(LIT §1; DESIGN §5)*
4. **Box mass-balance + Leighton** — `dC_k/dt = E_k/H_mix −(VC/H_mix)C_k−(v_{d,k}/H_mix)C_k−Λ_kP C_k+R_k`; PSS residual `r_L = C_{O3}C_{NO}−(J/k(T))C_{NO2}`. *(DESIGN §3)*
5. **Oₓ conservation** — `Ox=O3+NO2`; residual penalises fast `d(Ox)/dt` (slow terms only). *(DESIGN §3; LIT §6)* — *inline if tight*
6. **Composite loss** — `L = L_pin + λ_phys·MSE(box) + λ_chem(r_L + r_Ox) + λ_data·MSE(ĉ,c_obs) + λ_pos + λ_nc`. *(DESIGN §5)*

*Supporting (define in text, not numbered):* CRPS from quantile grid; Brier `=mean(p−y)²`; **CQR**
interval `C(x)=[\hat Q_{α/2}−η,\ \hat Q_{1−α/2}+η]`, `η`=conformity quantile; PICP. *(LIT §9; METHODS_calibration)*

---

## 7. Tables (exact sources — copy verbatim from RESULTS_LOG.md)

**Table I — Probabilistic forecast skill (rank index `H`, test 2016).** Columns: Pinball ↓, CRPS ↓,
Brier(exc) ↓, PICP₉₀.
- QR-PINN (HYBRID): pinball 0.2525 / CRPS 0.0561 / Brier 0.1197 / PICP90 0.826 — **E10 seed0**.
- QRNN ≡ FREE (data-only): pinball 0.2099 / CRPS 0.0466 / Brier 0.1005 / PICP90 0.788 — **E10 seed0**
  (or E8 λ=0: 0.2054/0.0456/0.1023/0.858).
- Climatology: ⚠️ **only logged on the *robust* index** (1.9256 pinball) — **NOT comparable** to rank.
  **See §12 open-gap C1** (need a rank-index climatology, or present within-rank comparison only).

**Table II — Calibration: coverage vs nominal (E6, rank, 13-τ).** `E6-20260615-233855`.
| level | nominal | PICP raw | PICP PIT | **PICP CQR** | width raw | width CQR |
|---|---|---|---|---|---|---|
| 80 % | 0.80 | 0.725 | 0.760 | **0.781** | 0.184 | 0.208 |
| 90 % | 0.90 | 0.839 | 0.875 | **0.892** | 0.241 | 0.282 |
| 95 % | 0.95 | 0.923 | 0.936 | **0.951** | 0.321 | 0.365 |
Footnote: CQR PICP90 by lead 1/6/12/24 = 0.893/0.887/0.902/0.900 (holds across lead).

**Table III — LOSO transfer [+EXPAND only].** `E7-20260616-033822`: in-dist 0.2024 [0.2025,0.2140] vs
LOSO 0.2277 ± 0.0655; gap +0.0253. Per-station rows available if a full page is used.

---

## 8. Figures (core 3 for 5 pages; status flagged)

> ⚠️ **Only `figs/e10_physics.png` exists on disk.** All E3–E9 figures referenced in the log were
> **not saved** (or were cleaned). Their underlying data **is** in `results_e*.json` + `artefacts/*.npz`,
> so they can be **regenerated** — this is reproduction, not a new experiment (see §12-C2).

| Fig | Content | Status | Source |
|---|---|---|---|
| **Fig 1** | **Study-area map** — 9 DoE stations on Bangladesh + data-availability glyph | **NEW (make)** | `bd.json`, `gis_station_summary.py`, `gis_station_pollutant_regimes.json` |
| **Fig 2** | **QR-PINN schematic** — window→LSTM→{quantile head (PDF), physics head (box/Leighton/Oₓ)} + loss | **NEW (draw)** | DESIGN §5 (conceptual) |
| **Fig 3** | **Calibration** — reliability/PIT + CQR coverage bars (one example predictive PDF inset) | **regenerate** | `results_e6.json` (e6_reliability/e6_coverage) |
| **Fig 4** | **XAI** — physics-group importance bars + IG↔occlusion agreement scatter (r=0.917) | **regenerate** | `results_e9.json`, `artefacts/e9_ig_sample.npz` |
| *Fig 5* | *Physics interpretability — learned rates + per-station emissions* **[+EXPAND]** | **exists / extend** | `figs/e10_physics.png`, `results_e10.json` |

**Default for 5 pages:** Fig 1 + Fig 2 + (Fig 3 **or** Fig 4). Recommend **Fig 1, Fig 2, Fig 4** (XAI is
the Track-06 differentiator) and fold calibration into Table II. Style: 300-dpi PNG + vector PDF,
colour-blind-safe palette, consistent per-group colours, single-column width where possible.

---

## 9. Writing style guide

- **Template:** IEEE two-column conference (`IEEEtran`, `conference` option). Numeric citations `[n]`.
- **Tense/voice:** present tense for findings ("the model attains"), past for what was done ("we
  trained"). Concise, claim-first sentences. Avoid hedging on solid results; **be explicit on the
  negative physics result** ("physics does not improve accuracy on this O₃-led index").
- **Length discipline (5 pp):** ≤6 equations, **2 tables**, **3 figures**, ~22–26 refs. Cut Related Work
  to one tight paragraph (cluster cites). No method appendix; push detail to the cited `METHODS_*.md`
  only conceptually (those are internal, not citable in the paper).
- **Numbers:** every value traceable to a run ID (keep a margin note `% src: E6-...` in the `.tex`).
- **Caveats:** one honest "Limitations" mini-paragraph in VI (index-change comparability, high base
  rate, future-met assumption, approximate conformal coverage, BARC artefact) — compressed from REPORT §7.
- **AI disclosure:** prepare the InGARSS <25 % AI-content statement.

---

## 10. Source map — *the only files you need while writing*

| Need | File | What it gives |
|---|---|---|
| Every number/table/figure value | **`RESULTS_LOG.md`** | source of truth; run IDs E0–E10, PHYSFORCE |
| Human synthesis + "what's defensible" | **`RESULTS_AND_REPORT.md`** | per-experiment prose, headline table (§5), claims (§9), caveats (§7) |
| Method/architecture/equations | **`EXPERIMENT_DESIGN.md`** | §2 index, §3 physics, §5 model+loss, §6 eval, §9 XAI |
| Citations (all refs) | **`LITERATURE_REVIEW.md`** | §1–11 + numbered References; map claims→[n] |
| Calibration method text | `METHODS_calibration.md` | PIT/CQR detail, exchangeability caveat |
| Transfer method text | `METHODS_transfer.md` | LOSO + bootstrap, spatial-CV caveats |
| XAI method text | `METHODS_explainability.md` | IG + occlusion, completeness, day/night |
| Physics negative-result rationale | `PHYSICS_DIAGNOSIS.md` | R6 root cause for the honest finding |
| Figure data | `results_e{3,5,6,7,8,9,10}.json`, `artefacts/*.npz`, `figs/e10_physics.png` | regenerate plots |
| Map data | `bd.json`, `gis_station_*.{py,json}` | study-area figure |
| Governance | `CLAUDE.md` | integrity + EXPLAIN→APPROVE→RUN |

---

## 11. Citation shortlist (from LITERATURE_REVIEW; ~22–26 for 5 pp)

Must-cite: QR/QRNN — Koenker&Bassett'78, Taylor'00, Cannon'18 (non-crossing) [1,3,5]. Prob. AQ —
Vaughan'21, Mukkavilli'21 [10,11]. PINN — Raissi'19, Karniadakis'21, Yang(B-PINN)'21 [13,14,15]. AQ-PINN
— AirPhyNet'24, WRF-CHEM+PINN+RS'23 [16,18]. QR∩PINN — Pereira'25 [19]. Chemistry — Seinfeld&Pandis'16,
Oₓ review'24 [23,24]. Health — Beijing O₃×NO₂'22, oxidant×heat'25 [26,27]. Bangladesh — World Bank'22,
Dhaka PM₂.₅'26 [31,32]. Scoring/calib — Gneiting&Raftery'07, Kuleshov'18, Romano(CQR)'19, Gibbs(ACI)'21
[29,30,33,36]. XAI — Sundararajan(IG)'17, Fisher(perm)'19, Ismail'20 [46,48,50]. Trim transfer/conformal
extras if over budget. ⚠️ verify foundational pagination (LIT verification note).

---

## 12. Open gaps to resolve BEFORE drafting (decisions/approvals needed)

- **C0 — Page limit.** Confirm **5 pp** (this plan) vs use the official **6+1 pp** headroom. Affects
  whether Fig 5 / Table III / a dedicated XAI subsection go in. *(your call)*
- **C1 — Climatology on the rank index (Table I).** Only robust-index climatology is logged (pinball
  1.93); it is **not comparable** to rank-index numbers. Options: (a) present *within-rank* comparison
  only (QR-PINN vs QRNN/FREE, drop climatology from T-I) — **no new run, recommended**; or (b) compute a
  rank-index climatology baseline — **that is a new number → EXPLAIN→APPROVE→RUN** (`CLAUDE.md §0`).
- **C2 — Figure regeneration.** E3–E9 PNG/PDFs are not on disk; regenerate Fig 3/Fig 4 from
  `results_e6.json` / `results_e9.json` + `artefacts/e9_ig_sample.npz` (reproduction). Confirm OK to run
  the small plotting scripts; outputs must match logged numbers exactly.
- **C3 — Reference verification.** Confirm pagination/years for foundational refs (LIT verification
  note: items 6,7,8,23,29,30) before they enter the `.tex`.
- **C4 — Author/affiliation block, funding/acknowledgements, AI-disclosure statement.**

---

## 13. Pre-writing checklist / next actions

1. ☐ Confirm **C0 page limit** + figure set (recommend Fig 1,2,4 + Table I,II).
2. ☐ Resolve **C1** (recommend within-rank Table I; else approve a climatology run).
3. ☐ Approve **C2** figure regeneration; produce Fig 1 (map), Fig 2 (schematic), Fig 3/4.
4. ☐ Lock the citation list (§11) and verify C3.
5. ☐ Draft in order: Method (III) → Results (V) → Intro (I) → Abstract → Conclusion → trim to 5 pp.
6. ☐ Final pass: every number carries a `% src: <run-id>` margin note; honest physics framing intact;
   AI-disclosure ready.
