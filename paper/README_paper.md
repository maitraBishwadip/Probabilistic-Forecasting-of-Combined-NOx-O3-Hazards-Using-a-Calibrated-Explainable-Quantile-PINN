# Paper build & provenance — InGARSS 2026 (Track 06)

**File:** `main.tex` · IEEEtran `conference` class · target **≤ 5 pages** (free; pp. 6–7 incur
InGARSS overlength charges). Figures in `figures/` (PNG @300 dpi + vector PDF).

## How to compile
No LaTeX is installed on this machine, so the `.tex` was **structure-linted** (balanced
environments/math/braces; all 22 `\cite` keys ↔ `\bibitem`; all `\ref`↔`\label`) but **not
compiled** here. To build:

- **Overleaf (easiest):** new project → upload `main.tex` + `figures/` → IEEEtran is preinstalled →
  Recompile. Uses `thebibliography` (no `.bib`/biber needed).
- **Local (MiKTeX/TeX Live):** `pdflatex main` × 2 (run twice so refs/citations resolve).

## Figures (regenerate with `python ../pipeline/make_paper_figs.py`)
| Fig # (`label`) | Column span | File | Content |
|---|---|---|---|
| 1 (`fig:overview`) | **single-column** (half-col footprint) | `fig1_composite.png` | one-row composite: (a) BD map + 9 **numbered** stations, (b) pooled NOx/O₃ distributions, (c) NO₂ + (d) O₃ kriged surfaces |
| 2 (`fig:laboni`) | single-column | — none — | **reserved framebox placeholder** + placeholder ("lorem ipsum") caption; author inserts manually |
| 3 (`fig:arch`) | **single-column** | `fig2_architecture.png` | QR-PINN block diagram (**vertical layout**): input→LSTM→{quantile head ∥ physics head}→predictive dist.+loss |
| 4 (`fig:results`) | **double** (`figure*`) | `fig3_reliability.png` + `fig4_importance.png` | (a) reliability + calibration-gain arrow + per-lead inset; (b) emphasis-coloured IG importance + IG-vs-occlusion scatter (r=0.917) |

**Float layout (fixes the "figures pile at the end / out of bounds" issue):** only the results
figure uses `figure*`; Figs 1–3 are single-column. Preamble loads `stfloats` + `dblfloatfix` so the
wide float places in order instead of deferring to the end. The two result panels share one
`figure*` via `minipage`s (no `subcaption` package needed). The architecture (Fig 3) was redrawn in
a **vertical** layout so it stays readable at single-column width.
Stations are numbered on the map (1 Agrabad … 9 Gazipur); legend is in the Fig 1 caption.
**Note (image filenames vs. figure numbers):** files are still `fig1..fig4_*.png`, but in the
manuscript the architecture is **Fig 3** and the results are **Fig 4** (laboni took the Fig 2 slot).
Replace the `fig:laboni` framebox in `main.tex` with `\includegraphics{...}` and a real caption when
the image is ready.

## Integrity (CLAUDE.md §1)
Every number in `main.tex` is transcribed verbatim from **`RESULTS_LOG.md`**; each table/figure
carries a `% src: run E#` tag. Climatology is reported only on the deprecated robust index
(not comparable to rank-index numbers) — Table III is within-rank only, per WRITEUP_PLAN §12-C1.
No new experiment was run to make the paper.

## Citations — all DOIs verified (Crossref / publisher), 2026-06-16
**Author corrections caught vs. LITERATURE_REVIEW.md (fix the review doc too):**
- NO₂ probabilistic QR paper → **Pérez Vasseur & Aznarte (2021)**, *Sci. Rep.* 11:11592 —
  `10.1038/s41598-021-90063-3` (review said "Vaughan").
- QR∩PINN systems-biology → **Hu, Cheng, Guo et al. (2025)**, *Bull. Math. Biol.* —
  `10.1007/s11538-025-01439-9` (review said "Pereira").
- WRF-CHEM+RS PINN → **Bo Li et al. (2023)**, *Atmos. Environ.* 311:120031 —
  `10.1016/j.atmosenv.2023.120031` (review said "Sci. Total Environ.").
- Beijing health → **Fu et al. (2022)**, *BMC Public Health* 22:2265 —
  `10.1186/s12889-022-14473-2` (review said only "PMC").
- Dhaka PM₂.₅ → **Billah et al. (2026)**, *Discover Environment* — `10.1007/s44274-026-00560-3`.
- Oxidant review → **Taheri, Khorsandi, Alavi Moghaddam (2024)** — `10.1007/s13762-024-05563-2`.

**Conference papers (NeurIPS/ICML/ICLR) have no journal DOI** — cited by venue + stable arXiv id,
which is the correct form: Hettige (AirPhyNet, arXiv:2402.03784), Kuleshov (1807.00263),
Romano CQR (1905.03222), Gibbs ACI (2106.00170), Sundararajan IG (1703.01365), Ismail (2010.13924).
Fisher et al. (JMLR) and Seinfeld & Pandis (book) / World Bank (report) have no DOI by nature.

## If it overflows 5 pages — trim levers (in order)
1. Drop the AI-assistance section into a one-line footnote on the title.
2. Move Eq. (3) (non-crossing) inline; merge Eq. (5) into the text.
3. Cut §V-F (interpretability) to 3 sentences; fold learned-rate detail into Fig. 4 caption.
4. Compress Table I (hypotheses) to a 3-row version (H1/H2/H5 only).

## TODO before submission
- Replace placeholder author email in `main.tex`.
- Fix the six author names above in `LITERATURE_REVIEW.md`.
- Confirm `\IEEEoverridecommandlockouts` is acceptable (only needed for the long author block).
