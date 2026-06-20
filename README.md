# QR‑PINN for Combined NOₓ–O₃ Extremes over Bangladesh

A **Quantile‑Regressive Physics‑Informed Neural Network (QR‑PINN)** that forecasts the **full
predictive probability distribution** of a *combined* NOₓ–O₃ hazard over Bangladesh — hourly, 24 h
ahead — and is **calibrated**, **spatially transferable**, and **explainable**.

> **Researcher:** Bishwadip Maitra · BUET Air‑Quality Project  
> **Target venue:** IEEE **InGARSS 2026** (Hyderabad, Track 06 — *Machine Learning & AI for Digital Earth*)  
> **Data:** `BD_DOE_2014-16.csv` — 9 Department‑of‑Environment stations, hourly, 2014‑01‑01 → 2016‑12‑31 (223,776 rows)

---

## Overview

Bangladesh is among the most polluted countries on Earth, and the **joint** upper range of nitrogen
oxides (NOₓ) and ozone (O₃) — chemically summarised by the total oxidant **Oₓ = O₃ + NO₂** — is
disproportionately harmful to health. Existing forecasts are point‑valued, single‑pollutant, and
physics‑free. This project asks a sharp question:

> *Does embedding NOₓ–O₃ photochemistry into a probabilistic forecaster improve prediction of the
> combined hazard?*

We build a QR‑PINN that emits the entire predictive distribution of a balanced, climatology‑referenced
combined hazard index `H`, and we test the physics hypothesis rigorously. The headline scientific
result is reported **honestly**: embedded chemistry **does not** improve accuracy on this O₃‑led
target — and explainable‑AI attribution shows *why*. The physics is therefore positioned as a source
of **interpretability**, not accuracy.

### Contributions
1. **First probabilistic, physics‑informed, combined‑pollutant forecast for Bangladesh** — the full
   24 h predictive CDF/PDF of `H` from a monotone (non‑crossing) quantile head.
2. **Calibrated** forecasts — conformalized quantile regression (CQR) restores near‑nominal coverage
   that holds across lead time; quantified **spatial transfer** via Leave‑One‑Station‑Out + bootstrap.
3. **An honest negative result** — across three independent experiments, embedded physics is
   neutral‑to‑negative for accuracy, with **explainable AI** (Integrated Gradients + occlusion,
   agreement *r* = 0.917) explaining the cause.
4. **Interpretable, physically‑ordered learned chemistry** (photolysis ≫ titration; recovered
   per‑station emission scales) from the physics‑guided hybrid.

---

## Headline results (adopted rank index, test 2016)

All numbers are transcribed from [`reports/RESULTS_LOG.md`](reports/RESULTS_LOG.md), the single source
of truth.

| Quantity | Value | Source run |
|---|---|---|
| Best probabilistic model (data‑only) pinball | ≈ 0.195–0.210 | E8 / E10 |
| Calibrated coverage (CQR @ 80/90/95 %) | 0.781 / 0.892 / 0.951 | E6 |
| CQR 90 % coverage across lead (h = 1/6/12/24) | 0.893 / 0.887 / 0.902 / 0.900 | E6 |
| Spatial transfer gap (LOSO − in‑dist, pinball) | **+0.025** | E7 |
| Explainability cross‑method agreement | Pearson *r* = **0.917** | E9 |
| Effect of embedded physics on accuracy | **neutral‑to‑negative** | E8 / PHYSFORCE / E10 |

> ⚠️ Rank‑index metrics live on `[0,1]` and are for *within‑study* comparison only. The mid‑study
> index change (robust → rank) means absolute values are **not** comparable across index families;
> see [`reports/RESULTS_AND_REPORT.md`](reports/RESULTS_AND_REPORT.md) §3.

---

## Repository structure

```
.
├── .claude/CLAUDE.md            # project governance & working agreement (read first)
├── data/                        # datasets (BD_DOE_2014-16.csv, derived CSVs)
├── pipeline/                    # all code
│   ├── 01_build_dataset.py      # QC + feature build
│   ├── 02_qrpinn_dataprep.py    # hourly NOx–O3 prep, index H + threshold, windowed tensors
│   ├── 03..11_qrpinn_*.py       # QR-PINN model + experiments E1–E10
│   ├── 10b_physforce_trial.py   # physics-forcing ablation
│   └── make_paper_figs.py       # regenerates the manuscript figures
├── results/                     # results_e3..e9.json + qc_report.json  (E10/physforce/qrpinn at root)
├── reports/                     # all written documents (see below)
├── artefacts/                   # model checkpoints (.pt), tensors (.npz), qrpinn_meta.json
├── figs/                        # misc figure outputs
├── paper/                       # the InGARSS 2026 manuscript
│   ├── main.tex                 # IEEEtran 5-page paper
│   ├── figures/                 # publication figures (PNG + PDF)
│   └── README_paper.md          # how to build the paper + figure/citation provenance
├── archive/                     # prior PM2.5–O3 regime study (read-only reference)
├── bd.json                      # Bangladesh geometry (GeoJSON, EPSG:4326)
└── gis_station_pollutant_regimes.{json,geojson}  # per-station descriptive EDA
```

### Key documents (`reports/`)
| File | Purpose |
|---|---|
| [`EXPERIMENT_DESIGN.md`](reports/EXPERIMENT_DESIGN.md) | Architecture of record: model, loss, physics, evaluation, experiment plan |
| [`RESULTS_LOG.md`](reports/RESULTS_LOG.md) | **Single source of truth** — append‑only run log (E0–E10) |
| [`RESULTS_AND_REPORT.md`](reports/RESULTS_AND_REPORT.md) | Human‑readable synthesis of every run + caveats |
| [`LITERATURE_REVIEW.md`](reports/LITERATURE_REVIEW.md) | Cited, thematic review (QR + PINN + NOₓ–O₃ + XAI) |
| [`METHODS_calibration.md`](reports/METHODS_calibration.md) · [`METHODS_transfer.md`](reports/METHODS_transfer.md) · [`METHODS_explainability.md`](reports/METHODS_explainability.md) | Method details for CQR, LOSO, and IG/occlusion |
| [`PHYSICS_DIAGNOSIS.md`](reports/PHYSICS_DIAGNOSIS.md) | Why the embedded physics is inessential to skill |
| [`WRITEUP_PLAN.md`](reports/WRITEUP_PLAN.md) | Manuscript plan (sections, tables, figures, source map) |

---

## Method (one paragraph)

An LSTM encoder (hidden 64) + 9‑station embedding ingests a 48 h window of 26 features (pollutant
lags, reanalysis meteorology, MODIS satellite fields, calendar, missingness masks). Two heads branch:
a **monotone quantile head** (non‑crossing `Q_τ(H_{t+h})` by construction → the predictive CDF/PDF and
exceedance probability `P(H ≥ thr)`) and a **physics head** (a coupled NO–NO₂–O₃ box‑ODE with the
**Leighton** photostationary‑state and **Oₓ‑conservation** residuals, global learnable rates). The
composite loss is pinball (primary) + annealed physics/chemistry residuals. The combined hazard index
is rank/quantile‑uniform, `H = ½·F_NOₓ(NOₓ) + ½·F_O₃(O₃)` (train empirical CDFs), with the
combined‑extreme cutoff at the **70th percentile** (the most‑polluted **top 30 %** of hours are the
extreme/hazardous class; Q90 marks a secondary "severe" tier).

---

## Reproducing

**Environment** (Python 3.10+): `numpy`, `pandas`, `scipy`, `matplotlib`, `shapely`, `torch`.

```bash
pip install numpy pandas scipy matplotlib shapely torch
```

**Pipeline** — every modelling step follows *EXPLAIN → APPROVE → RUN* (`.claude/CLAUDE.md` §0). Run in
order; each stage writes a machine‑readable JSON consumed verbatim by the report:

```bash
python pipeline/02_qrpinn_dataprep.py     # build H, threshold, windowed tensors + masks
python pipeline/03_qrpinn_model.py        # baselines + QR-PINN
# ... experiments E3–E10: pipeline/04..11_qrpinn_*.py
python pipeline/make_paper_figs.py         # regenerate manuscript figures
```

> Figures are already rendered in `paper/figures/`. The generator reads the raw assets
> (`bd.json`, station JSON, the DoE CSV, and `results_e6/e9`) — adjust the paths at the top of
> `pipeline/make_paper_figs.py` if you have reorganised the data/results folders.

---

## The paper

The InGARSS 2026 manuscript lives in [`paper/`](paper/). Build it on **Overleaf** (IEEEtran is
preinstalled) or locally:

```bash
cd paper && pdflatex main && pdflatex main   # run twice; uses thebibliography (no biber)
```

All citations carry verified DOIs; build/provenance notes are in
[`paper/README_paper.md`](paper/README_paper.md).

---

## Research integrity

This repository follows a strict working agreement ([`.claude/CLAUDE.md`](.claude/CLAUDE.md)):

- **Nothing untested enters the paper.** Every number/table/figure in `main.tex` traces to a run in
  `reports/RESULTS_LOG.md` with a matching run ID, config, and seed.
- **Failures are reported honestly** — the physics‑does‑not‑improve‑accuracy finding is a result, not
  hidden.
- **Citations are real and verified**; no fabricated DOIs.
- **No leakage** — all scalers, percentile cutoffs, and the index definition are fit on training data
  (2014–15) only and applied unchanged to test (2016).

---

## Citation

The associated paper is in preparation for IEEE InGARSS 2026. Until publication, please cite as:

```bibtex
@misc{maitra2026qrpinn,
  author = {Bishwadip Maitra},
  title  = {A Calibrated, Explainable Quantile Physics-Informed Neural Network for
            Probabilistic Forecasting of the Combined NOx--O3 Hazard over Bangladesh},
  year   = {2026},
  note   = {Manuscript, IEEE InGARSS 2026 (under preparation)}
}
```

## Data & license

Air‑quality observations are from the **Bangladesh Department of Environment (DoE)** continuous
monitoring network; reanalysis and satellite fields are from public sources. Code in this repository
is for academic research — add a license file before public release.
# Probabilistic-Forecasting-of-Combined-NOx-O3-Hazards-Using-a-Calibrated-Explainable-Quantile-PINN
