# -*- coding: utf-8 -*-
"""
12_qrpinn_e12.py  (E12)  -- Regime-stratified physics value.

Question (from PHYSICS_DIAGNOSIS R6 + the PINN literature): the pooled E10 result says physics does
NOT help accuracy of H. But PINN gains are known to concentrate in *specific regimes* -- sparse/gappy
data, the upper tail, and long lead (AirPhyNet ICLR'24; Krishnapriyan NeurIPS'21; Karniadakis low-data).
E12 asks: is there ANY regime where the physics-guided forecast (HYBRID / PHYS-ONLY) is at least as good
as the data-only FREE forecast? If yes -> a genuine, honest "PINN earns its keep here" claim. If no ->
the physics is confirmed redundant for accuracy on this target (interpretability-only, per CLAUDE.md).

NO new model design. This RE-RUNS E10's seed-0 training deterministically (E10 saved only pooled metrics
and no checkpoints), VALIDATES that pooled pinball reproduces results_e10.json (FREE 0.2099 / HYBRID
0.2525 / PHYS-ONLY 0.3174), then STRATIFIES the per-anchor test predictions by:
  (1) upper tail   -- per-tau pinball (esp tau=0.90/0.95) + Brier at Q90 + extreme-outcome subset (H>=Q90)
  (2) lead time    -- pinball per horizon h=1..24 (does the FREE-HYBRID gap shrink/flip at long lead?)
  (3) missingness  -- terciles of input-window gap fraction (does physics help when inputs are gappy?)
  (4) per-station  -- in-distribution spatial view (LOSO is deferred to E13: needs 9-fold retraining).

Outputs: results_e12.json, figs/e12_strata.png, append RESULTS_LOG.md.  Observed H only; no synthetic data.
This is a stratified RE-SCORING of the already-approved E10 models -- no §1/§3 locked decision changes.
"""
import json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"; FIGS = ROOT / "figs"; FIGS.mkdir(exist_ok=True)

L, HZ = 48, 24
TAUS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
nT = len(TAUS); MED = int(np.where(np.isclose(TAUS, 0.5))[0][0])
EPOCHS, BATCH, LR = 10, 512, 1e-3
LAM_C, LAM_LEIGH, ANNEAL = 0.3, 0.05, 4

# expected pooled pinball from results_e10.json seed0 (reproduction validity check)
EXPECT = {"free": 0.20990778505802155, "physonly": 0.3174222707748413, "hybrid": 0.25248003005981445}

# ---- scalers (train-only, from meta) ----
meta = json.load(open(ART / "qrpinn_meta.json"))
rs = meta["robust_scaler"]
medNO, iqrNO = rs["NO_ppb"]["median"], rs["NO_ppb"]["iqr"]
medN2, iqrN2 = rs["NO2_ppb"]["median"], rs["NO2_ppb"]["iqr"]
medN,  iqrN  = rs["NOx_ppb"]["median"], rs["NOx_ppb"]["iqr"]
medO,  iqrO  = rs["O3_ppb"]["median"], rs["O3_ppb"]["iqr"]
ss = meta["standard_scaler"]
VCm, VCs   = ss["ventilation_coefficient"]["mean"], ss["ventilation_coefficient"]["std"]
BLm, BLs   = ss["boundary_layer_height_m"]["mean"], ss["boundary_layer_height_m"]["std"]
Jm,  Js_   = ss["photochemical_activitiy_index"]["mean"], ss["photochemical_activitiy_index"]["std"]
Pm,  Ps_   = ss["precip_mm"]["mean"], ss["precip_mm"]["std"]
Tm,  Ts_   = ss["temp_C"]["mean"], ss["temp_C"]["std"]

# ---- data ----
d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
X = d["X"].astype(np.float32); PHYS = d["PHYS"].astype(np.float32)
COBS = d["COBS"].astype(np.float32); CMASK = d["CMASK"].astype(np.float32)   # [.,.,0]=NOx, [.,.,1]=O3
Hrob = d["H"].astype(np.float32); year = d["year"].astype(np.int32); S, T, F = X.shape

# ---- rank index H + train empirical-CDF grids (train-only) — identical to E10 ----
both = (CMASK[:, :, 0] == 1) & (CMASK[:, :, 1] == 1) & (~np.isnan(Hrob))
yr2d = np.broadcast_to(year[None, :], (S, T)); tr_both = both & (yr2d <= 2015)
sN = np.sort(COBS[:, :, 0][tr_both]); sO = np.sort(COBS[:, :, 1][tr_both])
def Fcdf(s, v): return np.searchsorted(s, v, side="right") / len(s)
H = np.full((S, T), np.nan, np.float32)
H[both] = (0.5 * Fcdf(sN, COBS[:, :, 0][both]) + 0.5 * Fcdf(sO, COBS[:, :, 1][both])).astype(np.float32)
thr   = float(np.quantile(H[tr_both], 0.30))   # hazard cut (Q30)
thrhi = float(np.quantile(H[tr_both], 0.90))   # upper-tail level (Q90)
glev = np.linspace(0, 1, 256).astype(np.float32)
gzN = torch.tensor(np.quantile(sN, glev).astype(np.float32))
gzO = torch.tensor(np.quantile(sO, glev).astype(np.float32))
GLEV = torch.tensor(glev)
print(f"rank thr(Q30)={thr:.4f}  thrhi(Q90)={thrhi:.4f}  (S={S} T={T} F={F})", flush=True)

def build_index():
    tr, te = [], []
    for si in range(S):
        Hs = H[si]
        for t in range(L - 1, T - HZ):
            if np.isnan(Hs[t]): continue
            if not np.any(~np.isnan(Hs[t + 1:t + 1 + HZ])): continue
            if year[t + HZ] <= 2015: tr.append((si, t))
            elif year[t] == 2016: te.append((si, t))
    return np.array(tr, np.int32), np.array(te, np.int32)
idx_tr, idx_te = build_index()
print(f"train={len(idx_tr)} test={len(idx_te)}", flush=True)

# ---- per-test-anchor metadata for stratification (aligned to idx_te order; predict uses shuffle=False)
te_station = idx_te[:, 0].astype(np.int32)
te_miss = np.empty(len(idx_te), np.float32)          # input-window gap fraction over NOx & O3
for i in range(len(idx_te)):
    si, t = int(idx_te[i, 0]), int(idx_te[i, 1])
    w = CMASK[si, t - L + 1:t + 1, :2]                # 1=observed, 0=missing
    te_miss[i] = 1.0 - float(w.mean())
m_lo, m_hi = np.quantile(te_miss, [1/3, 2/3])
print(f"input-window missingness: min={te_miss.min():.2f} terciles=({m_lo:.2f},{m_hi:.2f}) max={te_miss.max():.2f}", flush=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        si, t = int(self.idx[i, 0]), int(self.idx[i, 1])
        fut = H[si, t + 1:t + 1 + HZ].copy(); tm = (~np.isnan(fut)).astype(np.float32); fut[np.isnan(fut)] = 0.0
        return (torch.from_numpy(X[si, t - L + 1:t + 1, :]), torch.tensor(si),
                torch.from_numpy(PHYS[si, t + 1:t + 1 + HZ, :]),
                torch.from_numpy(COBS[si, t + 1:t + 1 + HZ, :]),
                torch.from_numpy(CMASK[si, t + 1:t + 1 + HZ, :]),
                torch.from_numpy(fut), torch.from_numpy(tm))

def interp1d(x, xp, fp):
    xc = x.clamp(float(xp[0]), float(xp[-1]))
    idx = torch.searchsorted(xp, xc).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]; y0, y1 = fp[idx - 1], fp[idx]
    return (y0 + (xc - x0) / (x1 - x0 + 1e-9) * (y1 - y0)).clamp(0, 1)

class E10(nn.Module):
    """Verbatim E10 model (mode in {'hybrid','physonly','free'})."""
    def __init__(self, F, n_station, hidden=64, emb=8, mode="hybrid"):
        super().__init__(); self.mode = mode
        self.emb = nn.Embedding(n_station, emb); self.lstm = nn.LSTM(F, hidden, batch_first=True)
        din = hidden + emb
        self.inc = nn.Linear(din, HZ * (nT - 1))
        self.spread = nn.Parameter(torch.tensor(-2.0))
        if mode == "free":
            self.hmed = nn.Linear(din, HZ); return
        self.init_state = nn.Linear(din, 3)
        self.rate_raw = nn.Parameter(torch.tensor(
            [-1.0, -3.0, -2.0, -2.5, -2.5, -2.5, -2.5, -3.0, -1.5], dtype=torch.float32))
        self.emis_raw = nn.Parameter(torch.tensor(-1.0))
        self.emis_station = nn.Embedding(n_station, 1); self.emis_station.weight.data.zero_()
        if mode == "hybrid":
            self.resid = nn.Linear(din, HZ); self.resid.weight.data.mul_(0.1); self.resid.bias.data.zero_()
            self.res_scale = nn.Parameter(torch.tensor(-1.5))

    def forward(self, x, si, pf):
        out, (hn, cn) = self.lstm(x); z = torch.cat([hn[-1], self.emb(si)], 1)
        inc = torch.nn.functional.softplus(self.inc(z)).view(-1, HZ, nT - 1)
        cum = torch.cat([torch.zeros_like(inc[:, :, :1]), torch.cumsum(inc, -1)], -1)
        off = (cum - cum[:, :, MED:MED + 1]) * torch.nn.functional.softplus(self.spread)
        if self.mode == "free":
            Hmed = torch.sigmoid(self.hmed(z)); Q = (Hmed.unsqueeze(-1) + off).clamp(0, 1)
            return Q, Hmed, None, None
        s0 = torch.nn.functional.softplus(self.init_state(z))
        no  = s0[:, 0] * iqrNO + medNO; no2 = s0[:, 1] * iqrN2 + medN2; o = s0[:, 2] * iqrO + medO
        r = torch.nn.functional.softplus(self.rate_raw)
        kj, ktit0, Ea, dep_no, dep_no2, dep_o3, dil, wet, P_o3 = [r[k] for k in range(9)]
        E_nox = torch.nn.functional.softplus(self.emis_raw) * torch.exp(self.emis_station(si).squeeze(-1))
        Hphys, leigh, nox_tr, o_tr = [], [], [], []
        for s in range(HZ):
            J  = torch.relu(pf[:, s, 2] * Js_ + Jm) / Jm
            vc = torch.relu(pf[:, s, 0] * VCs + VCm) / VCm
            bl = torch.relu(pf[:, s, 1] * BLs + BLm) + 50.0
            pr = torch.relu(pf[:, s, 3] * Ps_ + Pm)
            tC = pf[:, s, 4] * Ts_ + Tm
            jph = kj * J; tit = ktit0 * torch.exp(Ea * (tC - 25.0) / 25.0)
            sinkb = dil * vc + wet * pr; emisNO = E_nox / (1.0 + bl / BLm); prodO3 = P_o3 * J
            no2_n = (no2 + tit * no * o)        / (1.0 + jph + dep_no2 + sinkb)
            no_n  = (no  + jph * no2 + emisNO)  / (1.0 + tit * o + dep_no + sinkb)
            o_n   = (o   + jph * no2 + prodO3)  / (1.0 + tit * no + dep_o3 + sinkb)
            no, no2, o = no_n, no2_n, o_n; nox = no + no2
            rN = interp1d((nox - medN) / iqrN, gzN, GLEV); rO = interp1d((o - medO) / iqrO, gzO, GLEV)
            Hphys.append(0.5 * (rN + rO)); leigh.append((jph * no2 - tit * no * o) / (no2 + 1.0))
            nox_tr.append((nox - medN) / iqrN); o_tr.append((o - medO) / iqrO)
        Hphys = torch.stack(Hphys, 1); leigh = torch.stack(leigh, 1)
        chat = torch.stack([torch.stack(nox_tr, 1), torch.stack(o_tr, 1)], -1)
        if self.mode == "physonly":
            Hmed = Hphys
        else:
            delta = torch.nn.functional.softplus(self.res_scale) * torch.tanh(self.resid(z))
            Hmed = (Hphys + delta).clamp(0, 1)
        Q = (Hmed.unsqueeze(-1) + off).clamp(0, 1)
        return Q, Hmed, chat, leigh

def pinball_loss(q, y, tm):
    taus = torch.tensor(TAUS).view(1, 1, nT); err = y.unsqueeze(-1) - q
    return (torch.maximum(taus * err, (taus - 1) * err).mean(-1) * tm).sum() / tm.sum().clamp(min=1)

def train(mode, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = E10(F, S, mode=mode); opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = torch.utils.data.DataLoader(DS(idx_tr), batch_size=BATCH, shuffle=True, num_workers=0)
    for ep in range(EPOCHS):
        model.train(); a = min(1.0, (ep + 1) / ANNEAL); t0 = time.time(); tot = nb = 0
        for x, si, pf, cobs, cmask, fut, tm in dl:
            opt.zero_grad(); Q, Hmed, chat, leigh = model(x, si, pf)
            loss = pinball_loss(Q, fut, tm)
            if chat is not None:
                loss = loss + a * LAM_C * (((chat - cobs) ** 2 * cmask).sum() / cmask.sum().clamp(min=1))
                loss = loss + a * LAM_LEIGH * (leigh ** 2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item(); nb += 1
        print(f"  [{mode:8s}] ep {ep+1}/{EPOCHS} loss={tot/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model

def predict(model):
    model.eval(); dl = torch.utils.data.DataLoader(DS(idx_te), batch_size=1024, shuffle=False, num_workers=0)
    Q, Y, M = [], [], []
    with torch.no_grad():
        for x, si, pf, cobs, cmask, fut, tm in dl:
            q, hmed, _, _ = model(x, si, pf)
            Q.append(q.numpy()); Y.append(fut.numpy()); M.append(tm.numpy())
    return np.concatenate(Q), np.concatenate(Y), np.concatenate(M)   # [N,HZ,nT],[N,HZ],[N,HZ]

def cdf_at_thr(Q, level):
    N, m = Q.shape; k = (Q < level).sum(1); lo = np.clip(k - 1, 0, m - 1); hi = np.clip(k, 0, m - 1); ar = np.arange(N)
    qlo, qhi, tlo, thi = Q[ar, lo], Q[ar, hi], TAUS[lo], TAUS[hi]
    gap = qhi - qlo; den = np.where(gap > 1e-9, gap, 1.0); frac = np.where(gap > 1e-9, (level - qlo) / den, 0.0)
    cdf = tlo + frac * (thi - tlo); cdf = np.where(k == 0, TAUS[0], cdf); cdf = np.where(k == m, TAUS[-1], cdf)
    return np.clip(cdf, 0, 1)

# ---- stratified scoring on flat (anchor*HZ, .) arrays ----
def score(Qf, Yf, selflat):
    """selflat: boolean over flattened (N*HZ). Returns pinball overall, per-tau, Brier@Q30, Brier@Q90, n."""
    if selflat.sum() == 0: return None
    Q, Y = Qf[selflat], Yf[selflat]
    err = Y[:, None] - Q
    pb = np.maximum(TAUS[None, :] * err, (TAUS[None, :] - 1) * err)   # [n,nT]
    per_tau = pb.mean(0)
    pin = float(per_tau.sum())   # E10 convention: pinball SUMMED over the 9-tau grid (so it matches
                                 # RESULTS_LOG headline numbers and the established 0.003 noise band)
    pexc30 = 1.0 - cdf_at_thr(Q, thr);  ind30 = (Y >= thr).astype(float)
    pexc90 = 1.0 - cdf_at_thr(Q, thrhi); ind90 = (Y >= thrhi).astype(float)
    return {"n": int(selflat.sum()), "pinball": pin,
            "pinball_per_tau": {f"{t:.2f}": float(v) for t, v in zip(TAUS, per_tau)},
            "pinball_tau90": float(per_tau[list(np.round(TAUS,2)).index(0.90)]),
            "pinball_tau95": float(per_tau[list(np.round(TAUS,2)).index(0.95)]),
            "brier_q30": float(((pexc30 - ind30) ** 2).mean()),
            "brier_q90": float(((pexc90 - ind90) ** 2).mean())}

# ===================== RUN =====================
print("\n=== E12: reproduce E10 seed-0, then stratify ===", flush=True)
PRED = {}
for mode in ("free", "physonly", "hybrid"):     # SAME order as E10 -> reproduces seed-0 RNG stream
    print(f"\n== {mode.upper()} ==", flush=True)
    m = train(mode, seed=0); Q, Y, M = predict(m); PRED[mode] = (Q, Y, M)

# persist predictions so E13 (and any re-slice) needs no retraining
np.savez_compressed(ART / "e12_preds.npz", taus=TAUS, thr=thr, thrhi=thrhi,
                    te_station=te_station, te_miss=te_miss,
                    **{f"{m}_{k}": PRED[m][i] for m in PRED for i, k in enumerate(("Q", "Y", "M"))})
print("saved artefacts/e12_preds.npz", flush=True)

# flatten + masks
Hflat_idx = np.repeat(np.arange(len(idx_te)), HZ)          # anchor index per flat row
lead_flat = np.tile(np.arange(HZ), len(idx_te))            # horizon (0..23) per flat row
miss_flat = te_miss[Hflat_idx]; stn_flat = te_station[Hflat_idx]
valid = {mode: (PRED[mode][2].reshape(-1) > 0) for mode in PRED}
Qf = {mode: PRED[mode][0].reshape(-1, nT) for mode in PRED}
Yf = {mode: PRED[mode][1].reshape(-1) for mode in PRED}

# ---- validity check: pooled pinball must reproduce results_e10.json ----
repro = {}
for mode in PRED:
    s = score(Qf[mode], Yf[mode], valid[mode]); repro[mode] = s["pinball"]
    d_ = s["pinball"] - EXPECT[mode]
    flag = "OK" if abs(d_) < 1e-3 else "MISMATCH"
    print(f"  reproduce {mode:8s}: pinball={s['pinball']:.4f}  (E10={EXPECT[mode]:.4f}, d={d_:+.4f}) [{flag}]", flush=True)

# ---- (0) overall ; (1) tail ; (2) lead ; (3) missingness ; (4) per-station ----
strata = {"overall": {}, "tail_extreme_outcome": {}, "by_lead": {}, "by_missingness": {}, "by_station": {}}
for mode in PRED:
    v = valid[mode]
    strata["overall"][mode] = score(Qf[mode], Yf[mode], v)
    # tail: realized outcome in upper decile (H >= Q90)
    strata["tail_extreme_outcome"][mode] = score(Qf[mode], Yf[mode], v & (Yf[mode] >= thrhi))
    # by lead
    strata["by_lead"][mode] = {}
    for h in range(HZ):
        s = score(Qf[mode], Yf[mode], v & (lead_flat == h))
        strata["by_lead"][mode][h + 1] = None if s is None else {"pinball": s["pinball"], "n": s["n"]}
    # by missingness tercile
    strata["by_missingness"][mode] = {}
    for name, sel in (("low", miss_flat <= m_lo), ("mid", (miss_flat > m_lo) & (miss_flat <= m_hi)),
                      ("high", miss_flat > m_hi)):
        strata["by_missingness"][mode][name] = score(Qf[mode], Yf[mode], v & sel)
    # by station
    strata["by_station"][mode] = {}
    for si in range(S):
        s = score(Qf[mode], Yf[mode], v & (stn_flat == si))
        strata["by_station"][mode][meta["stations"][si]] = None if s is None else {"pinball": s["pinball"], "n": s["n"]}

# ---- deltas (physics - free); negative = physics HELPS in that slice ----
def dlt(a, b): return None if (a is None or b is None) else float(a["pinball"] - b["pinball"])
deltas = {
    "overall_hybrid_minus_free": dlt(strata["overall"]["hybrid"], strata["overall"]["free"]),
    "overall_physonly_minus_free": dlt(strata["overall"]["physonly"], strata["overall"]["free"]),
    "tail_hybrid_minus_free": dlt(strata["tail_extreme_outcome"]["hybrid"], strata["tail_extreme_outcome"]["free"]),
    "tail_physonly_minus_free": dlt(strata["tail_extreme_outcome"]["physonly"], strata["tail_extreme_outcome"]["free"]),
    "tau95_hybrid_minus_free": float(strata["overall"]["hybrid"]["pinball_tau95"] - strata["overall"]["free"]["pinball_tau95"]),
    "tau95_physonly_minus_free": float(strata["overall"]["physonly"]["pinball_tau95"] - strata["overall"]["free"]["pinball_tau95"]),
    "by_lead_hybrid_minus_free": {h + 1: dlt(strata["by_lead"]["hybrid"][h + 1], strata["by_lead"]["free"][h + 1]) for h in range(HZ)},
    "by_missingness_hybrid_minus_free": {k: dlt(strata["by_missingness"]["hybrid"][k], strata["by_missingness"]["free"][k]) for k in ("low", "mid", "high")},
    "by_station_hybrid_minus_free": {meta["stations"][si]: dlt(strata["by_station"]["hybrid"][meta["stations"][si]], strata["by_station"]["free"][meta["stations"][si]]) for si in range(S)},
}

# Binary "physics competitive" flag uses ONLY aggregate (tau-summed) pinball slices, where the
# E8/E10 0.003 noise band is defined. tau=0.95 (per-quantile) and the extreme-outcome subset are
# NOT used as flags: the former is on a different (per-tau) scale, and the latter is selection-biased
# (conditioning on realized H>=Q90 rewards a positively-biased forecaster). Both are reported below.
TOLER = 0.003
wins = []
if deltas["overall_hybrid_minus_free"] <= TOLER: wins.append("overall")
for h, dv in deltas["by_lead_hybrid_minus_free"].items():
    if dv is not None and dv <= TOLER: wins.append(f"lead h={h}")
for k, dv in deltas["by_missingness_hybrid_minus_free"].items():
    if dv is not None and dv <= TOLER: wins.append(f"missingness={k}")
for st, dv in deltas["by_station_hybrid_minus_free"].items():
    if dv is not None and dv <= TOLER: wins.append(f"station={st}")
verdict = ("HYBRID competitive (<= FREE + 0.003 summed-pinball) only in: " + "; ".join(wins)) if wins else \
          "NO regime where HYBRID is competitive; physics confirmed redundant for accuracy on H"

R = {"config": {"epochs": EPOCHS, "thr_q30": thr, "thr_q90": thrhi, "taus": TAUS.tolist(),
                "miss_terciles": [float(m_lo), float(m_hi)], "noise_band": TOLER,
                "note": "stratified re-scoring of reproduced E10 seed-0 models; LOSO deferred to E13"},
     "reproduction_check": {m: {"e12": repro[m], "e10": EXPECT[m], "abs_diff": abs(repro[m] - EXPECT[m])} for m in repro},
     "strata": strata, "deltas": deltas, "verdict": verdict}
json.dump(R, open(ROOT / "results_e12.json", "w"), indent=2)
print("\nVERDICT:", verdict, flush=True)

# ---- figure ----
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    col = {"free": "tab:gray", "hybrid": "tab:green", "physonly": "tab:orange"}
    # (a) pinball per lead
    for mode in ("free", "hybrid", "physonly"):
        y = [strata["by_lead"][mode][h + 1]["pinball"] for h in range(HZ)]
        ax[0, 0].plot(range(1, HZ + 1), y, marker=".", color=col[mode], label=mode)
    ax[0, 0].set_title("(a) pinball vs lead time"); ax[0, 0].set_xlabel("horizon h"); ax[0, 0].set_ylabel("pinball"); ax[0, 0].legend()
    # (b) per-tau pinball (overall)
    x = np.arange(nT); w = 0.26
    for j, mode in enumerate(("free", "hybrid", "physonly")):
        y = [strata["overall"][mode]["pinball_per_tau"][f"{t:.2f}"] for t in TAUS]
        ax[0, 1].bar(x + (j - 1) * w, y, w, color=col[mode], label=mode)
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels([f"{t:.2f}" for t in TAUS], rotation=45)
    ax[0, 1].set_title("(b) pinball per quantile tau (overall)"); ax[0, 1].set_xlabel("tau"); ax[0, 1].legend()
    # (c) pinball by missingness tercile
    cats = ["low", "mid", "high"]; x = np.arange(len(cats))
    for j, mode in enumerate(("free", "hybrid", "physonly")):
        y = [strata["by_missingness"][mode][c]["pinball"] for c in cats]
        ax[1, 0].bar(x + (j - 1) * w, y, w, color=col[mode], label=mode)
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels([f"{c}\n(input gaps)" for c in cats])
    ax[1, 0].set_title("(c) pinball by input-window missingness"); ax[1, 0].legend()
    # (d) delta(HYBRID - FREE) by lead  (negative = physics helps)
    dl_ = [deltas["by_lead_hybrid_minus_free"][h + 1] for h in range(HZ)]
    ax[1, 1].axhline(0, color="k", lw=0.8); ax[1, 1].axhline(TOLER, color="r", ls="--", lw=0.8, label="+0.003 noise band")
    ax[1, 1].plot(range(1, HZ + 1), dl_, marker=".", color="tab:green")
    ax[1, 1].set_title("(d) HYBRID - FREE pinball by lead (<0 = physics helps)"); ax[1, 1].set_xlabel("horizon h"); ax[1, 1].legend()
    plt.suptitle("E12: regime-stratified physics value (test 2016, rank index, seed 0)"); plt.tight_layout()
    plt.savefig(FIGS / "e12_strata.png", dpi=140); plt.close(); print("saved figs/e12_strata.png", flush=True)
except Exception as e:
    print("figure skipped:", e, flush=True)

# ---- append RESULTS_LOG.md ----
run_id = "E12-" + datetime.now().strftime("%Y%m%d-%H%M%S")
ov = strata["overall"]; tl = strata["tail_extreme_outcome"]; mz = strata["by_missingness"]
lines = [f"\n## Run {run_id}  (E12: regime-stratified physics value — FREE vs HYBRID vs PHYS-ONLY)",
         f"- date: {datetime.now().isoformat(timespec='seconds')} | code: pipeline/12_qrpinn_e12.py | rank index; seed=0; stratified re-scoring of reproduced E10 models (no new design; LOSO deferred to E13)",
         f"- reproduction check (pinball vs E10 seed0): " + ", ".join(f"{m} {repro[m]:.4f} (E10 {EXPECT[m]:.4f})" for m in repro),
         "", "**Overall (test 2016, seed 0)**  — pinball = SUM over 9 taus (E10 convention); tau=0.90/0.95 are per-quantile",
         "| variant | pinball (Στ) | tau=0.90 | tau=0.95 | Brier@Q30 | Brier@Q90 |", "|---|---|---|---|---|---|"]
for mode in ("free", "hybrid", "physonly"):
    v = ov[mode]
    lines.append(f"| {mode} | {v['pinball']:.4f} | {v['pinball_tau90']:.4f} | {v['pinball_tau95']:.4f} | {v['brier_q30']:.4f} | {v['brier_q90']:.4f} |")
lines += ["", "**Extreme-outcome subset (realized H >= Q90)** — CAVEAT: scoring conditional on the outcome being extreme rewards a positively-biased forecaster (PHYS-ONLY over-predicts), so a lower value here is NOT evidence of physics skill; the proper tail metrics are tau=0.95 and Brier@Q90 above.",
          "| variant | n | pinball (Στ) |", "|---|---|---|"]
for mode in ("free", "hybrid", "physonly"):
    v = tl[mode]; lines.append(f"| {mode} | {v['n']} | {v['pinball']:.4f} |" if v else f"| {mode} | 0 | NA |")
lines += ["", "**By input-window missingness (pinball)**", "| variant | low | mid | high |", "|---|---|---|---|"]
for mode in ("free", "hybrid", "physonly"):
    lines.append(f"| {mode} | " + " | ".join(f"{mz[mode][c]['pinball']:.4f}" for c in ("low", "mid", "high")) + " |")
lines += ["",
          f"- delta(HYBRID-FREE): overall {deltas['overall_hybrid_minus_free']:+.4f}, tail {deltas['tail_hybrid_minus_free']:+.4f}, tau95 {deltas['tau95_hybrid_minus_free']:+.4f}",
          f"- delta(HYBRID-FREE) by missingness: " + ", ".join(f"{k} {deltas['by_missingness_hybrid_minus_free'][k]:+.4f}" for k in ("low", "mid", "high")),
          f"- best (most negative) HYBRID-FREE lead delta: h={min(deltas['by_lead_hybrid_minus_free'], key=lambda h: deltas['by_lead_hybrid_minus_free'][h])} "
          f"({min(deltas['by_lead_hybrid_minus_free'].values()):+.4f})",
          f"- **VERDICT: {verdict}**",
          f"- fig figs/e12_strata.png ; future met used as forcing (forecast-met assumption); rank-index [0,1] metrics."]
with open(ROOT / "RESULTS_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nWrote results_e12.json and appended {run_id} to RESULTS_LOG.md", flush=True)
