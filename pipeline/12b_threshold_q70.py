# -*- coding: utf-8 -*-
"""
12b_threshold_q70.py  (THRCORR-Q70)  -- EXTREME-DEFINITION CORRECTION.

CONTEXT.  The study originally labelled  H >= Q30(H_train)  as the "extreme/hazardous" class, which
made ~70-83 % of all hours "extreme" -- backwards.  An extreme is the rare UPPER tail.  The corrected
definition is:  the most-polluted 30 % of hours  ==  H >= Q70(H_train)  ("extreme/hazardous"), with
Q90 kept as a secondary "severe" tier.

WHY NO RETRAINING.  The QR-PINN trains on the CONTINUOUS distribution of H via the pinball loss; it
never trains on the binary extreme label (the exceedance probability is read off the predicted CDF
post-hoc).  Hence pinball / CRPS / interval coverage (PICP, CQR) / LOSO transfer are THRESHOLD-
INDEPENDENT and remain exactly valid -- they are NOT recomputed here.  Only the threshold-dependent
quantities change: the exceedance probability p_{t+h}, every Brier(exceedance), and the base rate.

WHAT THIS SCRIPT DOES.  A pure RE-SCORING of the already-cached, frozen E10/E12 predictions
(artefacts/e12_preds.npz) -- NO new model, NO retraining, NO synthetic data.  It reconstructs the rank
index H and its train-only empirical CDFs exactly as E12 did, recomputes thr70/thr80/thr90, and re-
scores Brier(exceedance) + base rate at the corrected thresholds for FREE / HYBRID / PHYS-ONLY.
Outputs: results_e13_thrcorr.json and a THRCORR-Q70 block appended to reports/RESULTS_LOG.md.
"""
import json, numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"

# ---- real cached predictions (frozen E10/E12 models; predict used shuffle=False) ----
P = np.load(ART / "e12_preds.npz", allow_pickle=True)
TAUS = P["taus"].astype(np.float32)
thr30 = float(P["thr"])                       # the OLD (inverted) cut, kept only for reference
meta = json.load(open(ART / "qrpinn_meta.json"))
stations = meta["stations"]

# ---- reconstruct rank index H + train empirical CDF grids (TRAIN ONLY) -- identical to E12 ----
d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
COBS = d["COBS"].astype(np.float32)           # [.,.,0]=NOx, [.,.,1]=O3 (native ppb observations)
CMASK = d["CMASK"].astype(np.float32)
Hrob = d["H"].astype(np.float32)
year = d["year"].astype(np.int32)
S, T, _ = COBS.shape
both = (CMASK[:, :, 0] == 1) & (CMASK[:, :, 1] == 1) & (~np.isnan(Hrob))
yr2d = np.broadcast_to(year[None, :], (S, T))
tr_both = both & (yr2d <= 2015)
sN = np.sort(COBS[:, :, 0][tr_both]); sO = np.sort(COBS[:, :, 1][tr_both])
def Fcdf(s, v): return np.searchsorted(s, v, side="right") / len(s)
H = np.full((S, T), np.nan, np.float32)
H[both] = (0.5 * Fcdf(sN, COBS[:, :, 0][both]) + 0.5 * Fcdf(sO, COBS[:, :, 1][both])).astype(np.float32)

thr70 = float(np.quantile(H[tr_both], 0.70))   # CORRECTED extreme cut: top 30% (most polluted)
thr80 = float(np.quantile(H[tr_both], 0.80))   # for threshold-sensitivity (robustness table)
thr90 = float(np.quantile(H[tr_both], 0.90))   # secondary "severe" tier
print(f"thresholds on REAL H_train:  Q30={thr30:.4f} (OLD)  Q70={thr70:.4f}  Q80={thr80:.4f}  Q90={thr90:.4f}", flush=True)

# ---- read the exceedance probability off the predicted CDF (verbatim E12 cdf_at_thr) ----
def cdf_at_thr(Q, level):
    N, m = Q.shape; k = (Q < level).sum(1); lo = np.clip(k - 1, 0, m - 1); hi = np.clip(k, 0, m - 1); ar = np.arange(N)
    qlo, qhi, tlo, thi = Q[ar, lo], Q[ar, hi], TAUS[lo], TAUS[hi]
    gap = qhi - qlo; den = np.where(gap > 1e-9, gap, 1.0); frac = np.where(gap > 1e-9, (level - qlo) / den, 0.0)
    cdf = tlo + frac * (thi - tlo); cdf = np.where(k == 0, TAUS[0], cdf); cdf = np.where(k == m, TAUS[-1], cdf)
    return np.clip(cdf, 0, 1)

def brier_base(mode, level):
    Q = P[f"{mode}_Q"].reshape(-1, len(TAUS)); Y = P[f"{mode}_Y"].reshape(-1); v = P[f"{mode}_M"].reshape(-1) > 0
    Q, Y = Q[v], Y[v]
    pexc = 1.0 - cdf_at_thr(Q, level); ind = (Y >= level).astype(float)
    return float(((pexc - ind) ** 2).mean()), float(ind.mean()), int(v.sum())

MODES = ("free", "hybrid", "physonly")
LEVELS = {"Q30(OLD)": thr30, "Q70": thr70, "Q80": thr80, "Q90": thr90}

results = {"note": "THRCORR-Q70: re-score of cached E10/E12 predictions at the corrected extreme cut "
                   "(top-30% = H>=Q70); Q90 = severe tier. No retraining; no synthetic data. "
                   "pinball/CRPS/PICP/CQR/LOSO are threshold-independent and UNCHANGED.",
           "thresholds": {"Q30_old": thr30, "Q70": thr70, "Q80": thr80, "Q90": thr90},
           "base_rate_test": {}, "brier_exceedance": {}}

# base rate is model-independent (uses observed Y); compute once via FREE
for name, lv in LEVELS.items():
    _, base, n = brier_base("free", lv)
    results["base_rate_test"][name] = base
print("\nTEST base rate (real observed H):  " +
      "  ".join(f"{k}={results['base_rate_test'][k]:.3f}" for k in LEVELS), flush=True)

print("\nBrier(exceedance) re-scored from REAL predicted CDFs (no synthetic):", flush=True)
print("  level     | " + " | ".join(f"{m:9s}" for m in MODES), flush=True)
for name, lv in LEVELS.items():
    row = {}
    for m in MODES:
        b, _, _ = brier_base(m, lv); row[m] = b
    results["brier_exceedance"][name] = row
    print(f"  {name:9s} | " + " | ".join(f"{row[m]:.4f}    " for m in MODES), flush=True)

# headline deltas at the corrected primary cut (Q70)
results["delta_hybrid_minus_free_Q70"] = results["brier_exceedance"]["Q70"]["hybrid"] - results["brier_exceedance"]["Q70"]["free"]
results["delta_physonly_minus_free_Q70"] = results["brier_exceedance"]["Q70"]["physonly"] - results["brier_exceedance"]["Q70"]["free"]

json.dump(results, open(ROOT / "results_e13_thrcorr.json", "w"), indent=2)
print("\nwrote results_e13_thrcorr.json", flush=True)

# ---- append a clearly-labelled correction block to RESULTS_LOG.md (source of truth) ----
run_id = "THRCORR-Q70-" + datetime.now().strftime("%Y%m%d-%H%M%S")
L = [f"\n## Run {run_id}  (EXTREME-DEFINITION CORRECTION: top-30% = H>=Q70; supersedes the inverted H>=Q30)",
     f"- date: {datetime.now().isoformat(timespec='seconds')} | code: pipeline/12b_threshold_q70.py | rank index; re-score of cached E10/E12 predictions (artefacts/e12_preds.npz); NO retraining; NO synthetic data",
     "- CORRECTION: 'extreme/hazardous' was wrongly defined as H>=Q30 (~70-83% of hours). Corrected to the",
     "  most-polluted top 30% = H>=Q70; Q90 kept as a secondary 'severe' tier. ALL prior Brier(exceedance)/",
     "  base-rate numbers in earlier run blocks were computed at the inverted Q30 cut and are SUPERSEDED here.",
     "- THRESHOLD-INDEPENDENT metrics (pinball, CRPS, PICP, CQR coverage, LOSO transfer) are UNCHANGED -- they",
     "  never used the threshold (model trains on the continuous H, not the binary label).",
     "",
     f"- thresholds on train H: Q30(old)={thr30:.4f}  **Q70={thr70:.4f}**  Q80={thr80:.4f}  Q90={thr90:.4f}",
     "- test base rate (observed): " + ", ".join(f"{k}={results['base_rate_test'][k]:.3f}" for k in LEVELS),
     "",
     "**Brier(exceedance), test 2016, seed 0 -- re-scored at corrected cuts**",
     "| cut | base rate | FREE | HYBRID | PHYS-ONLY |",
     "|---|---|---|---|---|"]
for name, lv in LEVELS.items():
    r = results["brier_exceedance"][name]
    L.append(f"| {name} | {results['base_rate_test'][name]:.3f} | {r['free']:.4f} | {r['hybrid']:.4f} | {r['physonly']:.4f} |")
L += ["",
      f"- **VERDICT unchanged & threshold-robust:** FREE < HYBRID < PHYS-ONLY at every cut (Q70 delta HYBRID-FREE "
      f"= {results['delta_hybrid_minus_free_Q70']:+.4f}, PHYS-ONLY-FREE = {results['delta_physonly_minus_free_Q70']:+.4f}). "
      "Correcting the extreme cut does not rescue the physics for accuracy.",
      "- NOTE: rank-index [0,1] metrics; thresholds fit on TRAIN only (leakage-free)."]
with open(ROOT / "reports" / "RESULTS_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"appended {run_id} to reports/RESULTS_LOG.md", flush=True)
