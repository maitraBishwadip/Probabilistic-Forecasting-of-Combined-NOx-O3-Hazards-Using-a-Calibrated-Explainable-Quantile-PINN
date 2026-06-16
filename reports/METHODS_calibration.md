# Methods — Calibrating the QR-PINN Predictive Distribution (E6)

**Purpose.** Paper-ready methods write-up for the calibration stage of the QR-PINN combined NOx–O3
study. It states *exactly* what is done and *why*, with equations and citations, so the manuscript's
"Probabilistic calibration" subsection can be assembled directly from here. All numbers go to
`RESULTS_LOG.md`; this file is method, not results (`CLAUDE.md §1`).

---

## 1. Setup and notation

We forecast the **rank/quantile-uniform combined hazard index** (adopted in E5,
`EXPERIMENT_DESIGN.md §2`)

$$H_t \;=\; \tfrac12 F_{\mathrm{NOx}}(\mathrm{NOx}_t) \;+\; \tfrac12 F_{\mathrm{O_3}}(\mathrm{O_3}_t)\in[0,1],$$

where $F_\bullet$ is the train empirical CDF of each pollutant. For each issue time $t$ and lead
$h\in\{1,\dots,24\}$ the QR-PINN outputs a set of conditional quantiles
$\{\hat q_{\tau}(x_t,h)\}_{\tau\in\mathcal T}$, monotone non-crossing in $\tau$ by the
Cannon (2018) MCQRNN-style construction, which together approximate the **predictive CDF**
$\hat F(\cdot\mid x_t,h)$ and hence the full predictive distribution of $H_{t+h}$. Quantiles are
trained with the pinball loss $\rho_\tau(u)=\max(\tau u,(\tau-1)u)$ (Koenker & Bassett 1978).

**Calibration goal.** A predictive CDF is *calibrated* if its prediction intervals attain their
nominal coverage. For a nominal level $1-\alpha$ define the **Prediction Interval Coverage
Probability** and **Mean Prediction Interval Width**

$$\mathrm{PICP}_{1-\alpha}=\frac1N\sum_i \mathbf 1\{y_i\in C_{1-\alpha}(x_i)\},\qquad
\mathrm{MPIW}=\frac1N\sum_i \mathrm{len}\,C_{1-\alpha}(x_i).$$

E1–E4 found the QR-PINN **under-disperses**: $\mathrm{PICP}_{80}\approx0.66$–$0.71$ and
$\mathrm{PICP}_{90}\approx0.82$ against nominal $0.80/0.90$, and PIT recalibration (E4) could not lift
$\mathrm{PICP}_{90}$ because **>10 % of targets fell above the model's highest predicted quantile**
($\tau=0.95$). E6 fixes both the *tail reach* and the *coverage*.

---

## 2. Two complementary calibration mechanisms

We deliberately report two methods that answer different questions; together they make the calibration
claim robust.

### 2.1 Extended quantile grid (deeper tail reach)
We retrain the QR-PINN with an **extended grid**
$\mathcal T=\{0.01,0.025,0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.9,0.95,0.975,0.99\}$ so the network is
explicitly trained (pinball loss) to predict the **0.975/0.99 upper tail** — the hazardous region for
a combined-oxidant extreme. Non-crossing is preserved by the monotone construction (Cannon 2018), so
adding tail levels cannot create invalid (crossing) quantiles.

### 2.2 PIT distribution recalibration — full distribution (Kuleshov et al. 2018)
On a held-out calibration set we compute the **Probability Integral Transform (PIT)**
$u_i=\hat F(y_i\mid x_i)$ and the empirical recalibration map $R(p)=\frac1n\sum_i\mathbf 1\{u_i\le p\}$.
The calibrated quantile at nominal level $\tau$ is read at the adjusted level
$R^{-1}(\tau)=\mathrm{Quantile}_\tau(\{u_i\})$:

$$\hat q^{\,\mathrm{cal}}_{\tau}(x)=\hat q_{\,R^{-1}(\tau)}(x).$$

This recalibrates the **entire** predictive CDF (every $\tau$) and is what E4 used; here it is applied
to the extended-grid model. Limitation (made explicit in the paper): $R^{-1}$ can only *redistribute*
mass within the predicted quantile range — hence § 2.1 and § 2.3.

### 2.3 Conformalized Quantile Regression — guaranteed intervals (Romano, Patterson & Candès 2019)
CQR wraps the quantile model with split-conformal calibration to obtain **finite-sample, distribution-
free** coverage and, crucially, *can widen intervals beyond the predicted quantiles*. For a target
$1-\alpha$ interval with lower/upper quantile predictions $\hat q_{\alpha/2},\hat q_{1-\alpha/2}$,
compute on the calibration set the conformity scores

$$E_i=\max\!\big\{\hat q_{\alpha/2}(x_i)-y_i,\; y_i-\hat q_{1-\alpha/2}(x_i)\big\},$$

take $Q_{1-\alpha}=$ the $\big\lceil (n+1)(1-\alpha)\big\rceil$-th smallest $E_i$, and output

$$C_{1-\alpha}(x)=\big[\,\hat q_{\alpha/2}(x)-Q_{1-\alpha},\;\; \hat q_{1-\alpha/2}(x)+Q_{1-\alpha}\,\big].$$

Under exchangeability this guarantees $\mathbb P\{y\in C_{1-\alpha}(x)\}\ge 1-\alpha$
(Romano et al. 2019; building on split conformal, Vovk et al. 2005; Lei et al. 2018). $E_i<0$ when the
raw interval already over-covers (CQR then *tightens* it), $E_i>0$ widens it — the adaptivity that PIT
recalibration lacks at the tail. We compute $Q_{1-\alpha}$ **per lead time $h$** (coverage degrades
with horizon), and report two-sided intervals at $1-\alpha\in\{0.80,0.90,0.95\}$.

---

## 3. Time-series validity caveat (stated honestly in the paper)

Split conformal / CQR assume **exchangeability**, which hourly air-quality data violate
(autocorrelation, seasonal/inter-annual drift; recall 2016 has a higher extreme rate than 2014–15).
Consequently the finite-sample guarantee is **approximate** here, and we therefore *measure* empirical
coverage on a held-out block rather than asserting it. To respect temporal order we use a **temporal
calibration block** (the earliest 40 % of 2016) and evaluate on the later 60 %. Fully online,
shift-robust variants — **Adaptive Conformal Inference** (Gibbs & Candès 2021) and **EnbPI / conformal
prediction for time series** (Xu & Xie 2021) — are noted as the rigorous extension and left as future
work. This framing is the honest, defensible position for the manuscript.

---

## 4. Experimental protocol (E6)

1. **Model.** Retrain the FULL QR-PINN (rank index, physics on; seed 0; otherwise the
   `EXPERIMENT_DESIGN.md §5` architecture) with the extended grid (§ 2.1). Save checkpoint.
2. **Splits.** Train 2014–15 → fit model. **Calibration** = earliest 40 % of 2016 anchors.
   **Evaluation** = latest 60 % of 2016 anchors. (Calibration disjoint from both training and eval.)
3. **Calibrate.** Produce three predictive objects on eval: **(a) raw**, **(b) PIT-recalibrated**
   (§ 2.2), **(c) CQR intervals** at 80/90/95 % (§ 2.3, per-horizon).
4. **Score (proper rules + coverage).** Report, pooled and per-horizon:
   - Distributional: **pinball** (per-$\tau$ and mean), **CRPS** (Gneiting & Raftery 2007).
   - Calibration: $\mathrm{PICP}_{80/90/95}$ and **MPIW** for raw / PIT / CQR; **PIT reliability
     diagram** (empirical vs nominal) before/after.
   - Tail focus: pinball at $\tau\in\{0.9,0.95,0.99\}$; coverage of the upper hazardous tail.
   - Hazard: **Brier score** on the exceedance event $\{H_{t+h}\ge\mathrm{thr}\}$, $\mathrm{thr}=Q_{30}$.
5. **Decision rule for the paper.** The reference model is the **calibrated** combined-hazard
   forecaster: PIT recalibration for the *full distribution* (CRPS/PDF figures) and CQR for *interval
   guarantees* on the hazardous tail. Whichever attains nominal coverage with smallest MPIW is the
   headline interval method.

---

## 5. Results tables to fill (schema — populated only from `RESULTS_LOG.md`)

**T1 — Calibration on held-out late-2016 (pooled over $h=1..24$).**

| Method | Pinball | CRPS | PICP80 | MPIW80 | PICP90 | MPIW90 | PICP95 | MPIW95 | Brier(exc) |
|---|---|---|---|---|---|---|---|---|---|
| Raw QR-PINN | … | … | … | … | … | … | … | … | … |
| + PIT recal | … | … | … | … | … | … | … | … | … |
| + CQR | — | — | … | … | … | … | … | … | — |

**T2 — Coverage vs lead time** (PICP90 at $h=1,6,12,24$ for raw / PIT / CQR) → lead-time degradation.

Figures: reliability diagram (before/after), example calibrated predictive PDFs of $H$ with the
hazardous threshold marked, coverage-vs-horizon curve.

---

## 6. Reproducibility
Fixed seeds; calibration block and τ-grid logged; CQR $Q_{1-\alpha}^{(h)}$ persisted; checkpoint
saved (`artefacts/qrpinn_full_rank_ext.pt`). No imputation of targets; observed $H$ only.

---

## References (this stage; full list in `LITERATURE_REVIEW.md`)
- Koenker & Bassett (1978) *Econometrica* — quantile regression / pinball loss.
- Cannon (2018) *SERRA* — monotone composite QRNN (non-crossing quantiles).
- Kuleshov, Fenner & Ermon (2018) *ICML* — calibrated regression (PIT recalibration).
- Romano, Patterson & Candès (2019) *NeurIPS* 3538–3548; arXiv:1905.03222 — Conformalized Quantile Regression.
- Vovk, Gammerman & Shafer (2005) *Algorithmic Learning in a Random World*; Lei et al. (2018) *JASA* — (split) conformal prediction.
- Gibbs & Candès (2021) *NeurIPS*; arXiv:2106.00170 — Adaptive Conformal Inference under distribution shift.
- Xu & Xie (2021); arXiv:2010.09107 — conformal prediction intervals for time series (EnbPI).
- Angelopoulos & Bates (2021) arXiv:2107.07511 — gentle introduction to conformal prediction.
- Gneiting & Raftery (2007) *JASA* — proper scoring rules (CRPS).
