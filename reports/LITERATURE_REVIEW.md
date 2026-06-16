# Literature Review — Quantile-Regressive Physics-Informed Neural Networks for Probabilistic Forecasting of Combined NOx–O3 Extremes

**Scope.** This review supports a study whose method is a **Quantile-Regressive PINN (QR-PINN)**
that outputs the **entire predictive distribution** of a **combined NOx–O3 hazard index** at
hourly, multi-step lead times over Bangladesh. The review is organised around the five pillars the
method stands on — (1) quantile regression and quantile neural networks, (2) probabilistic /
distributional air-quality forecasting, (3) physics-informed neural networks, (4) physics-informed
ML for atmospheric chemistry and air quality, and (5) the QR ∩ PINN intersection — followed by the
domain science that justifies a *combined* NOx–O3 target: (6) NOx–O3 photochemistry and the total
oxidant Oₓ, (7) the synergistic health burden of co-exposure, and (8) the Bangladesh context. It
closes with the research gap and the evaluation theory (proper scoring rules and calibration).

> **Citation discipline (see `CLAUDE.md`):** every reference below is a real, locatable source
> (URLs in § References). No claim in the manuscript may be made without a pointer to this file.

---

## 1. Quantile regression and quantile-regression neural networks

Quantile regression (QR) estimates conditional quantiles of a response rather than its conditional
mean, giving a distribution-free picture of the whole conditional distribution and being robust to
heavy tails — exactly the setting of pollutant extremes. The framework was introduced by **Koenker
& Bassett (1978)** via the asymmetric "pinball" (check) loss, and consolidated in **Koenker
(2005)**.

The neural extension — the **Quantile Regression Neural Network (QRNN)** — was proposed by
**Taylor (2000)**, who used a network trained with the pinball loss to estimate the conditional
density of multi-period returns; **Cannon (2011)** gave a widely used implementation (the `qrnn`
R package) and applied it to precipitation downscaling. A central pathology of estimating several
quantiles independently is **quantile crossing** (a higher nominal quantile predicted below a
lower one), which is physically nonsensical for return levels. **Cannon (2018)** solved this with
the **Monotone Composite QRNN (MCQRNN)**, which simultaneously estimates multiple **non-crossing**
quantiles and supports monotonicity/positivity constraints — demonstrated on rainfall extremes,
and directly relevant to estimating non-crossing pollutant quantiles. Related strategies to learn
many quantiles jointly include the **simultaneous quantile regression (SQR)** of **Tagasovska &
Lopez-Paz (2019)** (a single network conditioned on the quantile level τ, yielding model-level
uncertainty), the **implicit quantile networks (IQN)** of **Dabney et al. (2018)** (reparameterised
sampling of τ to represent a full return distribution), and the joint multi-quantile training
analysed in **"Learning Multiple Quantiles With Neural Networks" (2021)**. Non-crossing QRNNs have
also been used to calibrate ensemble weather forecasts (**Adv. Atmos. Sci. 2023**).

**Takeaway for this study.** The QR-PINN inherits the pinball loss over a dense τ-grid and the
non-crossing machinery (MCQRNN-style monotone construction) so that the full predictive CDF/PDF of
the combined index is valid by construction.

---

## 2. Probabilistic / distributional air-quality forecasting

Operational air-quality forecasting is shifting from point predictions to **predictive
distributions**, because decisions (health warnings, exceedance thresholds) are inherently
probabilistic. **Vaughan et al. (2021, Sci. Rep.)** systematically compared ten QR methods for
probabilistic forecasting of urban **NO₂** out to 60 h and found quantile gradient-boosted trees
strongest for both the point value and the full distribution — establishing QR as a reference
methodology for gaseous-pollutant distributions and motivating a strong **quantile-GBT baseline**
here. **Murray et al. / Mukkavilli et al. (2021, arXiv:2112.02622)** survey deep probabilistic
approaches (quantile regression, Bayesian inference, MC-dropout, deep ensembles) for AQ uncertainty,
and **Zhang et al. (2022, GeoInformatica)** build a probabilistic spatio-temporal deep network for
AQ. These works output marginal predictive intervals; **none couples the predictive distribution to
the governing physics**, which is the contribution of a QR-PINN.

---

## 3. Physics-informed neural networks (PINNs)

PINNs embed governing differential equations into the training objective as a soft penalty on the
PDE/ODE residual evaluated by automatic differentiation, so the network honours physics where data
are sparse or noisy. The framework is due to **Raissi, Perdikaris & Karniadakis (2019)**; the
broader programme of physics-informed ML is reviewed by **Karniadakis et al. (2021, Nat. Rev.
Phys.)**. Because deterministic PINNs give point solutions, an uncertainty-aware variant —
**Bayesian PINNs (B-PINNs)**, **Yang, Meng & Karniadakis (2021)** — places priors over weights to
return posterior predictive uncertainty for forward/inverse PDE problems with noisy data. B-PINNs
are powerful but computationally heavy; quantile-based UQ (§ 5) is a lighter, distribution-free
alternative well suited to large hourly environmental datasets.

---

## 4. Physics-informed ML for atmospheric chemistry and air quality

A fast-growing line embeds the **advection–diffusion–reaction (ADR)** transport equation into
neural networks for pollutants:

- **AirPhyNet (Hettige et al., ICLR 2024)** couples an RNN encoder with a graph-based
  differential-equation network representing advection and diffusion, learning transport physics
  for PM₂.₅ across a station graph.
- **TransNet (npj Clean Air, 2026)** solves the full ADR system on a station graph for
  multi-day (+72 h) forecasts.
- **PINN-improved WRF-CHEM with satellite remote sensing (Sci. Total Environ., 2023)** fuses a
  chemistry-transport model, a PINN, and remote-sensing constraints — the closest precedent to a
  physics + remote-sensing air-quality model and an anchor for the IGARSS framing.

**Takeaway.** Existing air-quality PINNs target **single-pollutant, deterministic** point/field
prediction. None of them (a) outputs a **full predictive distribution**, (b) targets a **combined
two-pollutant** hazard, or (c) encodes **NOx–O3 photochemical coupling** as the physics residual.

---

## 5. The intersection: quantile regression ∩ PINN (the niche of this study)

Combining quantile regression with PINNs is recent and sparse, which is precisely why it is a
contribution:

- **Pereira et al. (2025, Bull. Math. Biol.)** — *"A Framework for Parameter Estimation and
  Uncertainty Quantification in Systems Biology Using Quantile Regression and Physics-Informed
  Neural Networks"* — a PINN with **multiple parallel quantile outputs** trained under physics
  constraints; the clearest demonstration that a PINN can carry a quantile-regression head for UQ.
- **Physics-informed deep Monte-Carlo quantile regression (Appl. Math. Model., 2023)** — a
  physics-informed CNN surrogate with Monte-Carlo quantile regression to quantify data uncertainty
  from sensor noise.
- **Conformal prediction for UQ in PINNs (arXiv:2509.13717, 2025)** and **conformal quantile
  regression for neural constitutive modelling (arXiv:2601.17437)** — distribution-free,
  finite-sample-valid prediction intervals layered onto PINN/quantile models; a calibration option
  for this study's evaluation.

**Gap.** These works are in systems biology, materials, and reliability — **not air quality**, and
they do not address a **combined two-pollutant extreme** or **hourly multi-step forecasting**. A
QR-PINN that (i) emits the full conditional distribution of a **combined NOx–O3 index**, (ii) is
constrained by **NOx–O3 photochemistry**, and (iii) is evaluated on the **deadly upper-70 % region**,
appears to be unaddressed.

---

## 6. NOx–O3 photochemistry and the total oxidant Oₓ (why a *combined* target is physical)

NOx (= NO + NO₂) and O₃ are not independent species: they are locked in the fast photochemical
cycle NO₂ + hν → NO + O(³P), O + O₂ → O₃, and O₃ + NO → NO₂ + O₂, summarised by the **Leighton
photostationary-state relation** [O₃] ≈ J(NO₂)·[NO₂] / (k·[NO]) (**Seinfeld & Pandis, 2016**). A
direct consequence is that the quantity **Oₓ = O₃ + NO₂ (total oxidant)** is approximately
conserved under the fast NO↔NO₂↔O₃ titration, so it separates a **regional (Oₓ)** component from a
local NOx-titration component — a standard diagnostic in monitoring studies (**Int. J. Environ.
Sci. Technol. review, 2024**; **Tehran long-term Oₓ analysis, Sci. Rep. 2024**; SW-Iberian
NO/NO₂/NOx/O₃/Oₓ analysis). This chemistry is **the physical backbone of the QR-PINN**: it explains
why NOx and O₃ are diurnally anti-phased (NOx titrates O₃ at night/rush-hour, O₃ forms at midday)
yet jointly governed by the same meteorology and photolysis, and it provides the **conservation /
photostationary residual** that turns a generic neural net into a physics-informed one.

**Relevance to the combined index.** Because Oₓ = O₃ + NO₂ is the chemically meaningful "combined
oxidant", the study's combined NOx–O3 hazard index is both a **health construct** and a
**physically grounded state variable**; the Leighton relation and Oₓ quasi-conservation are the
governing constraints the PINN enforces (`EXPERIMENT_DESIGN.md §3`).

---

## 7. Synergistic health burden of combined NOx–O3 exposure (why the extreme is "deadly")

The motivation for flagging a large (≥70 %) hazardous class is that combined oxidant exposure is
disproportionately harmful:

- **O₃ × NO₂ interaction, Beijing time-series (PMC, 2022)** — short-term O₃, NO₂ and Oₓ are
  positively associated with ER visits for respiratory disease, with interaction effects.
- **Oxidative pollutants × heat, circulatory mortality (PMC, 2025)** — O₃, NO₂ and Oₓ show
  **synergistic** effects with temperature on circulatory mortality, with nonlinear
  exposure–response.
- **Long-term NO₂, O₃ and oxidative potential & adolescent mental health (Environ. Int., 2024)** —
  each IQR rise in Oₓ is linked to adverse outcomes.

These establish that the *joint* upper range of NOx and O₃ — not only the rare top percentile — is
the public-health-relevant region, justifying the 30th-percentile cutoff (≥70 % hazardous class).

---

## 8. Bangladesh / South-Asia air-quality context

Bangladesh is among the most polluted countries globally (World Bank, *Breathing Heavy*, 2022).
Particulates peak in winter under inversions, brick-kiln and biomass emissions, and reduced
washout, with **meteorology explaining the majority of variability** (**Dhaka seasonal PM₂.₅
study, 2026**); O₃, being photochemical, does **not** peak in winter. No published study forecasts a
**combined NOx–O3 extreme** for Bangladesh, and none does so **probabilistically with embedded
photochemistry** — the regional gap this work fills, with an IGARSS-aligned remote-sensing framing
(reanalysis meteorology + satellite columns as physical drivers).

---

## 9. Evaluation theory — proper scoring and calibration

Probabilistic forecasts must be scored with **strictly proper scoring rules** so that honesty is
optimal (**Gneiting & Raftery, 2007**): the **pinball/quantile loss** (training and per-quantile
evaluation), the **Continuous Ranked Probability Score (CRPS)** for the whole distribution, and
**reliability/calibration** diagnostics. **Kuleshov, Fenner & Ermon (2018)** formalise calibrated
regression for deep nets (PIT-based recalibration). For *guaranteed* intervals, **conformalized
quantile regression (CQR; Romano, Patterson & Candès 2019)** wraps a quantile model with
split-conformal calibration (Vovk et al. 2005; Lei et al. 2018) for finite-sample, distribution-free,
heteroscedasticity-adaptive coverage; for non-exchangeable hourly data the principled variants are
**Adaptive Conformal Inference** (Gibbs & Candès 2021) and **conformal prediction for time series /
EnbPI** (Xu & Xie 2021). These define the evaluation protocol in `EXPERIMENT_DESIGN.md §6` and the
calibration methods in `METHODS_calibration.md`: pinball loss, CRPS, prediction-interval coverage
(PICP) and width, PIT histograms, CQR-guaranteed intervals, plus tail-focused exceedance metrics on
the ≥70 % hazardous region.

---

## 10. Synthesis — the gap this study fills

1. **Methodological:** no PINN emits a **full predictive distribution** via quantile regression for
   air quality; QR-PINN work exists only outside the atmospheric domain (§ 5).
2. **Target:** prior air-quality QR/PINN work is **single-pollutant**; this study forecasts a
   **combined NOx–O3 oxidant hazard** grounded in Oₓ chemistry (§ 6).
3. **Risk framing:** existing extreme studies chase the rare top percentile; the health evidence
   (§ 7) motivates a **broad ≥70 % hazardous class**, a deliberate departure.
4. **Region:** **first** probabilistic, physics-informed combined-pollutant forecast for
   **Bangladesh** (§ 8).
5. **Explainability:** the forecast is not a black box — its attributions are used to **audit physics
   faithfulness** and explain the *tail* and the *uncertainty*, not only a point (§ 11).

---

## 11. Explainability / attribution for a distributional, physics-informed forecast

Embedding physics is only half of trustworthiness; the other half is showing the *learned* input→output
map is faithful to that physics. Post-hoc attribution supplies that audit. Two complementary families
are used here. **Gradient attribution** — **Integrated Gradients (Sundararajan, Taly & Yan 2017)** —
integrates the model gradient along a path from a baseline input and is *axiomatic* (Sensitivity,
Implementation Invariance) with an exact **Completeness** identity (attributions sum to the
output–baseline gap), making it a principled, dependency-free choice for a differentiable LSTM-based
QR-PINN; the **SHAP** unification (**Lundberg & Lee 2017**) and **Molnar (2022)** give the broader
interpretable-ML framing. **Permutation / occlusion importance** — introduced for random forests by
**Breiman (2001)** and rigorously characterized (single-model and across a model class) by **Fisher,
Rudin & Dominici (2019)** — measures importance model-agnostically in the predictor's own loss
currency, a robust cross-check on gradient saliency. Crucially, **Ismail et al. (2020, NeurIPS)** show
that saliency methods can degrade sharply on **time-series** models (feature and time importance become
entangled), which is the explicit justification for (i) reporting IG *with* its completeness gap and
(ii) corroborating it with occlusion in proper-score currency (pinball/CRPS/Brier, §9), rather than
trusting any single saliency map.

**Gap / contribution.** Air-quality XAI to date overwhelmingly explains **point** predictions of a
**single** pollutant (typically tree-SHAP on PM₂.₅/O₃ regressors). Here attribution is applied to (a) a
**combined NOx–O3** hazard, (b) the **distribution** — separately explaining the *median*, the *severe
upper tail* $Q_{0.95}$, and the *predictive interval width* (uncertainty) — and (c) as a **faithfulness
audit of embedded photochemistry**, including a **day/night** test of the diurnal NOx–O3 anti-phasing
(§ 6). This distributional, physics-auditing use of XAI for a combined-pollutant PINN appears
unaddressed. Methods detailed in `METHODS_explainability.md`; evaluation in `EXPERIMENT_DESIGN.md §9`.

---

## References

*Quantile regression & quantile neural networks*
1. Koenker, R., & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33–50.
2. Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
3. Taylor, J. W. (2000). A quantile regression neural network approach to estimating the conditional density of multiperiod returns. *Journal of Forecasting*, 19(4), 299–311.
4. Cannon, A. J. (2011). Quantile regression neural networks: Implementation in R and application to precipitation downscaling. *Computers & Geosciences*, 37(9), 1277–1284. https://www.sciencedirect.com/science/article/abs/pii/S009830041000292X
5. Cannon, A. J. (2018). Non-crossing nonlinear regression quantiles by monotone composite quantile regression neural network, with application to rainfall extremes. *Stochastic Environmental Research and Risk Assessment*, 32, 3207–3225. https://link.springer.com/article/10.1007/s00477-018-1573-6
6. Tagasovska, N., & Lopez-Paz, D. (2019). Single-Model Uncertainties for Deep Learning. *NeurIPS*. (Simultaneous Quantile Regression.)
7. Dabney, W., et al. (2018). Implicit Quantile Networks for Distributional Reinforcement Learning. *ICML*.
8. Learning Multiple Quantiles With Neural Networks (2021). *Journal of Computational and Graphical Statistics*. https://www.tandfonline.com/doi/full/10.1080/10618600.2021.1909601
9. Non-crossing QRNN as a calibration tool for ensemble weather forecasts (2023). *Advances in Atmospheric Sciences*. https://link.springer.com/article/10.1007/s00376-023-3184-5

*Probabilistic air-quality forecasting*
10. Comparing quantile regression methods for probabilistic forecasting of NO₂ pollution levels (2021). *Scientific Reports*, 11. https://www.nature.com/articles/s41598-021-90063-3
11. Probabilistic Deep Learning to Quantify Uncertainty in Air Quality Forecasting (2021). arXiv:2112.02622. https://arxiv.org/pdf/2112.02622
12. Probabilistic air quality forecasting using deep learning spatial–temporal neural network (2022). *GeoInformatica*. https://link.springer.com/article/10.1007/s10707-022-00479-w

*Physics-informed neural networks*
13. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
14. Karniadakis, G. E., et al. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3, 422–440.
15. Yang, L., Meng, X., & Karniadakis, G. E. (2021). B-PINNs: Bayesian physics-informed neural networks. *Journal of Computational Physics*, 425, 109913.

*Physics-informed ML for air quality*
16. Hettige, K. H., et al. (2024). AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction. *ICLR*. https://arxiv.org/pdf/2402.03784
17. TransNet: transport-informed GNN solving the ADR system (2026). *npj Clean Air*. https://www.nature.com/articles/s44407-026-00052-x
18. PINN improving WRF-CHEM with satellite remote-sensing data (2023). *Science of the Total Environment*. https://www.sciencedirect.com/science/article/abs/pii/S1352231023004570

*Quantile regression ∩ PINN*
19. A Framework for Parameter Estimation and Uncertainty Quantification in Systems Biology Using Quantile Regression and PINNs (2025). *Bulletin of Mathematical Biology*. https://link.springer.com/article/10.1007/s11538-025-01439-9
20. Physics-informed deep Monte Carlo quantile regression method (2023). *Applied Mathematical Modelling*. https://www.sciencedirect.com/science/article/abs/pii/S0307904X23002792
21. A Conformal Prediction Framework for Uncertainty Quantification in PINNs (2025). arXiv:2509.13717. https://arxiv.org/abs/2509.13717
22. Conformal Quantile Regression for Neural Probabilistic Constitutive Modeling (2026). arXiv:2601.17437. https://arxiv.org/html/2601.17437

*NOx–O3 photochemistry & total oxidant Oₓ*
23. Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and Physics* (3rd ed.). Wiley. (Leighton photostationary state; Oₓ.)
24. Analysis of local and regional contributions of oxidant (Oₓ = O₃ + NO₂) levels — a review (2024). *Int. J. Environ. Sci. Technol.* https://link.springer.com/article/10.1007/s13762-024-05563-2
25. A long-term analysis of oxidant (Oₓ = O₃ + NO₂) in Tehran (2024). *Scientific Reports*. https://www.nature.com/articles/s41598-024-82709-9

*Combined-exposure health*
26. Association and interaction of O₃ and NO₂ with ER visits for respiratory diseases, Beijing (2022). *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9721066/
27. Interactive effects of atmospheric oxidative pollutants and heat on circulatory disease mortality (2025). *PMC*. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12339441/
28. Long-term exposure to NO₂, O₃ and their oxidative potential and adolescents' mental health (2024). *Environment International*. https://www.sciencedirect.com/science/article/pii/S0160412024007992

*Evaluation theory*
29. Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*, 102(477), 359–378.
30. Kuleshov, V., Fenner, N., & Ermon, S. (2018). Accurate Uncertainties for Deep Learning Using Calibrated Regression. *ICML*.

*Conformal prediction (interval calibration)*
33. Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized Quantile Regression. *NeurIPS* 32, 3538–3548. https://arxiv.org/abs/1905.03222
34. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
35. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-Free Predictive Inference for Regression. *JASA*, 113(523), 1094–1111.
36. Gibbs, I., & Candès, E. J. (2021). Adaptive Conformal Inference Under Distribution Shift. *NeurIPS*. https://arxiv.org/abs/2106.00170
37. Xu, C., & Xie, Y. (2021). Conformal Prediction Interval for Dynamic Time-Series. *ICML* (EnbPI). https://arxiv.org/pdf/2010.09107
38. Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.

*Spatial cross-validation (transfer) & bootstrap inference*
39. Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
40. Meyer, H., et al. (2019). Importance of spatial predictor variable selection in machine learning applications. *Environmental Modelling & Software*, 101, 1–9. https://www.sciencedirect.com/science/article/abs/pii/S0304380019303230
41. Milà, C., Mateu, J., Pebesma, E., & Meyer, H. (2022). Nearest neighbour distance matching Leave-One-Out CV for map validation. *Methods in Ecology and Evolution*, 13(6), 1304–1316.
42. Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S., & Brus, D. J. (2021). Spatial cross-validation is not the right way to evaluate map accuracy. *Ecological Modelling* / *Machine Learning* (critique). https://link.springer.com/article/10.1007/s10994-021-05972-1
43. Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*, 17(3), 1217–1241.
44. Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *JASA*, 89(428), 1303–1313.
45. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

*Bangladesh context*
31. World Bank (2022). *Breathing Heavy: New Evidence on Air Pollution and Health in Bangladesh.*
32. Seasonal meteorological control of Dhaka PM₂.₅ (2026). *Discover Environment / Springer*. https://link.springer.com/article/10.1007/s44274-026-00560-3

*Explainability / attribution*
46. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML*, PMLR 70, 3319–3328. (Integrated Gradients.) https://proceedings.mlr.press/v70/sundararajan17a.html
47. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. (Permutation variable importance.)
48. Fisher, A., Rudin, C., & Dominici, F. (2019). All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously. *JMLR*, 20(177), 1–81. https://jmlr.org/papers/v20/18-760.html
49. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS* 30, 4765–4774.
50. Ismail, A. A., Gunady, M., Corrada Bravo, H., & Feizi, S. (2020). Benchmarking Deep Learning Interpretability in Time Series Predictions. *NeurIPS* 33, 6441–6452. https://arxiv.org/abs/2010.13924
51. Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.). https://christophm.github.io/interpretable-ml-book/

> **Verification note.** Reference details (years, authors, volumes) carried over from the prior
> project plan or inferred from search snippets are marked where uncertain; confirm each on the
> publisher page before it enters the `.tex`. Items 6, 7, 8, 23, 29, 30 are foundational works cited
> from established knowledge — verify exact pagination at write-up time.
