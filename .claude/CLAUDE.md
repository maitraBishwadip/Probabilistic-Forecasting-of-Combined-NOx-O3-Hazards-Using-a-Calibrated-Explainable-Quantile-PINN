# CLAUDE.md — Project Governance & Working Agreement

**Project:** Quantile-Regressive Physics-Informed Neural Network (QR-PINN) for probabilistic
forecasting of the **combined NOx–O3 extreme** over Bangladesh.
**Researcher:** Bishwadip Maitra · BUET Air-Quality Project · target venue **IEEE IGARSS**.
**Data:** `BD_DOE_2014-16.csv` — 9 DoE stations, hourly, 2014-01-01 → 2016-12-31 (223,776 rows).

This file is the **standing instruction set** for any AI assistant working in this repository.
It overrides default behaviour. Read it before doing anything.

---

## 0. The single most important rule — experiment workflow

> **For EVERY experiment, follow this three-step protocol — no exceptions:**
> 1. **EXPLAIN** — state *what* you are going to do and *how* you will do it (data, model
>    change, loss, metric, expected artefacts). Write it down (in chat and/or in
>    `EXPERIMENT_DESIGN.md`).
> 2. **WAIT FOR APPROVAL** — do not run anything until the researcher explicitly approves.
> 3. **EXECUTE** — only then run the experiment, then report the actual results.

"Experiment" = anything that trains, fits, evaluates, tunes, or produces a number that could
end up in the paper. Data inspection / profiling / reading code is *not* an experiment and may
proceed freely. When in doubt, treat it as an experiment and ask first.

---

## 1. Research-integrity rules (the paper / `.tex`)

1. **Nothing untested goes into the `.tex`.** Every number, table, figure, and factual claim
   in the manuscript MUST come from an experiment that was actually run in this repository and
   whose output is recorded in `RESULTS_LOG.md`. No placeholder numbers, no "expected"
   performance written as if achieved, no values copied from other papers presented as ours.
2. **`RESULTS_LOG.md` is the single source of truth.** The `.tex` may only cite results that
   exist there with a matching run ID, config, seed, and metric value. If it is not in the log,
   it does not go in the paper.
3. **Literature claims are cited, not invented.** Any statement attributed to prior work must
   point to a real reference in `LITERATURE_REVIEW.md`. Do not fabricate citations or DOIs.
4. **Separate fact from plan.** `EXPERIMENT_DESIGN.md` contains intentions and hypotheses;
   `RESULTS_LOG.md` contains outcomes. Never let a planned/hoped-for result migrate into the
   manuscript as a finding.
5. **Report failures honestly.** If a model underperforms a baseline, that is the result —
   record it. Do not quietly drop unfavourable runs.

---

## 2. Remember the research architecture & results

The assistant must keep an accurate, persistent memory of the study's architecture and findings:

- **Architecture of record:** `EXPERIMENT_DESIGN.md` (§ model, loss, physics, data pipeline).
  Keep it in sync with the code. If the architecture changes, update this doc in the same change.
- **Results of record:** `RESULTS_LOG.md` (create on the first run). One append-only entry per
  run: `run_id`, date, code commit/version, config (seed, quantile grid, horizons, loss
  weights), split, and every metric value. The paper reads only from here.
- **Decisions of record:** locked choices live in § 3 below and in `EXPERIMENT_DESIGN.md §1`.
  Do not silently re-litigate them.

---

## 3. Locked study definition (decided with the researcher)

| Decision | Value |
|---|---|
| **Method (the only model under study)** | **Quantile-Regressive PINN (QR-PINN)** |
| **Target pollutants** | **NOx and O3** (combined) — whole dataset, **no regime conditioning** |
| **Combined target** | a **single combined NOx–O3 hazard index** `H` (see `EXPERIMENT_DESIGN.md §2`) |
| **Model output** | the **entire predictive probability distribution** of `H` (full set of conditional quantiles → CDF/PDF), per horizon |
| **Extreme definition** | **cutoff at the 30th percentile of `H`** ⇒ **≥70 % of observations are the "extreme/hazardous" class** ("combined impact can be deadly") |
| **Resolution / timing** | **hourly, multi-step**: forecast the next-24 h trajectory distribution |
| **Splits** | temporal (train 2014–15, test 2016) + Leave-One-Station-Out for transfer |

Anything not in this table is open and must be designed → approved → run per § 0.

---

## 4. Scope discipline

- **Experiments are QR-PINN only.** Baselines exist solely to benchmark the QR-PINN and are
  defined in `EXPERIMENT_DESIGN.md`. Do not start new lines of work (GBT-as-product, LSTM-only,
  regime clustering, PM2.5 compound study, etc.) unless explicitly asked.
- **Prior work is archived** in `archive/` (the previous PM2.5–O3 regime-conditioned study and
  all its models, results, figures, logs, and plans). Treat it as read-only reference; mine it
  for citations, do not rebuild it.

---

## 5. Repository map

```
ROOT/
├── CLAUDE.md                      # this file — governance
├── LITERATURE_REVIEW.md          # cited, thematic review (QR-PINN + NOx–O3)
├── EXPERIMENT_DESIGN.md          # architecture + experiment plan (approve before running)
├── RESULTS_LOG.md                # created on first run — source of truth for the paper
│
├── BD_DOE_2014-16.csv            # raw data (DO NOT EDIT)
├── bd.json                       # Bangladesh geometry (maps)
├── Probabilistic Extrame prediction.pdf   # reference
│
├── pipeline/                     # DATA MANIPULATION — kept outside archive, we work on these
│   ├── 01_build_dataset.py       # QC + feature build (to be refactored for NOx–O3, hourly, no regime)
│   └── eda_tables.py             # descriptive EDA
│
├── modelling_daily.csv, qc_report.json, *_mean_*.csv   # current data-manip outputs (regenerated)
│
└── archive/                      # PRIOR WORK — read-only
    ├── pipeline/ (02–06 models)  ├── figs/  ├── artefacts/
    ├── results_*.json            └── research_plan.md, Concrete_Plan_LitReview_PINN.md, experiment.md
```

**Data manipulation stays outside `archive/`** (we actively edit it). **Modelling = QR-PINN only.**

---

## 6. Reproducibility norms

- Fix and log all seeds (numpy / torch). Persist scalers, percentile thresholds, and the
  combined-index definition as artefacts so labels/targets are deterministic.
- **No leakage:** all scaling, percentile cutoffs, and the index definition are fit on the
  **training data only** and applied unchanged to validation/test.
- Each stage writes a machine-readable output (JSON) consumed verbatim by the report; the paper
  never hand-copies numbers.
- Record environment (key package versions) alongside results.

---

## 7. Interaction style expected here

- Explain before executing (§ 0). Recommend a default, don't dump every option.
- Keep `EXPERIMENT_DESIGN.md` and `RESULTS_LOG.md` current as work proceeds.
- Be precise about what is *measured* vs *planned*. The credibility of the paper depends on it.
