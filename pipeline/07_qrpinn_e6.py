# -*- coding: utf-8 -*-
"""
07_qrpinn_e6.py  (E6)  -- Tail calibration of the rank-index QR-PINN.
Method & citations: METHODS_calibration.md.

Two complementary calibrations on the ADOPTED rank index H = 0.5 F_NOx(NOx)+0.5 F_O3(O3):
  (1) EXTENDED tau-grid retrain -> the net predicts the 0.975/0.99 tail (the hazardous region).
  (2) PIT distribution recalibration (Kuleshov et al. 2018)  -> full predictive CDF.
  (3) Conformalized Quantile Regression (Romano, Patterson & Candes 2019), per-horizon
      -> finite-sample, distribution-free interval coverage that can widen beyond predicted quantiles.

Split: train 2014-15 ; calib = earliest 40% of 2016 ; eval = latest 60% (temporal block; the
time-series exchangeability caveat is discussed in METHODS_calibration.md sec.3).
Outputs: results_e6.json, artefacts/qrpinn_full_rank_ext.pt, figs/e6_reliability.png,
         figs/e6_coverage.png, append RESULTS_LOG.md.  No synthetic data; observed H only.
"""
import json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"; FIGS = ROOT / "figs"; FIGS.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

L, HZ = 48, 24
TAUS = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99], dtype=np.float32)
nT = len(TAUS)
def ti(tau): return int(np.argmin(np.abs(TAUS - tau)))
LEVELS = {"80": (ti(0.10), ti(0.90), 0.20), "90": (ti(0.05), ti(0.95), 0.10), "95": (ti(0.025), ti(0.975), 0.05)}
EPOCHS, BATCH, LR = 12, 512, 1e-3
LAM_DATA, LAM_PHYS_MAX, ANNEAL = 0.3, 0.2, 6

d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
X = d["X"].astype(np.float32); PHYS = d["PHYS"].astype(np.float32)
COBS = d["COBS"].astype(np.float32); CMASK = d["CMASK"].astype(np.float32)
Hrob = d["H"].astype(np.float32); year = d["year"].astype(np.int32); S, T, F = X.shape

# ---- rank index (same as E5) ----
both = (CMASK[:, :, 0] == 1) & (CMASK[:, :, 1] == 1) & (~np.isnan(Hrob))
yr2d = np.broadcast_to(year[None, :], (S, T)); tr_both = both & (yr2d <= 2015)
sN = np.sort(COBS[:, :, 0][tr_both]); sO = np.sort(COBS[:, :, 1][tr_both])
def Fcdf(s, v): return np.searchsorted(s, v, side="right") / len(s)
H = np.full((S, T), np.nan, np.float32)
H[both] = (0.5*Fcdf(sN, COBS[:, :, 0][both]) + 0.5*Fcdf(sO, COBS[:, :, 1][both])).astype(np.float32)
thr = float(np.quantile(H[tr_both], 0.30))
print(f"rank thr={thr:.4f}", flush=True)

def build_index():
    tr, te = [], []
    for si in range(S):
        Hs = H[si]
        for t in range(L-1, T-HZ):
            if np.isnan(Hs[t]): continue
            if not np.any(~np.isnan(Hs[t+1:t+1+HZ])): continue
            if year[t+HZ] <= 2015: tr.append((si, t))
            elif year[t] == 2016: te.append((si, t))
    return np.array(tr, np.int32), np.array(te, np.int32)
idx_tr, idx_te = build_index()
idx_te = idx_te[np.argsort(idx_te[:, 1], kind="stable")]
ncal = int(0.4*len(idx_te)); idx_cal, idx_eval = idx_te[:ncal], idx_te[ncal:]
print(f"train={len(idx_tr)} calib={len(idx_cal)} eval={len(idx_eval)}", flush=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        si, t = int(self.idx[i, 0]), int(self.idx[i, 1])
        fut = H[si, t+1:t+1+HZ].copy(); tm = (~np.isnan(fut)).astype(np.float32); fut[np.isnan(fut)] = 0.0
        return (torch.from_numpy(X[si, t-L+1:t+1, :]), torch.tensor(si),
                torch.from_numpy(PHYS[si, t-L+1:t+1, :]), torch.from_numpy(COBS[si, t-L+1:t+1, :]),
                torch.from_numpy(CMASK[si, t-L+1:t+1, :]), torch.from_numpy(fut), torch.from_numpy(tm))

class QRPINN(nn.Module):
    def __init__(self, F, n_station, hidden=64, emb=8):
        super().__init__()
        self.emb = nn.Embedding(n_station, emb); self.lstm = nn.LSTM(F, hidden, batch_first=True)
        self.base = nn.Linear(hidden+emb, HZ); self.inc = nn.Linear(hidden+emb, HZ*(nT-1))
        self.chead = nn.Linear(hidden, 2); self.praw = nn.Parameter(torch.zeros(7))
    def forward(self, x, si):
        out, (hn, cn) = self.lstm(x); trunk = torch.cat([hn[-1], self.emb(si)], 1)
        base = self.base(trunk); inc = torch.nn.functional.softplus(self.inc(trunk)).view(-1, HZ, nT-1)
        q = torch.cat([base.unsqueeze(-1), base.unsqueeze(-1)+torch.cumsum(inc, -1)], -1)
        return q, self.chead(out)
def pos(p): return torch.nn.functional.softplus(p)
def physics_residual(chat, pw, praw):
    n, o = chat[:, :, 0], chat[:, :, 1]
    gv, gp, gr = torch.sigmoid(pw[:, :, 0]), torch.sigmoid(pw[:, :, 2]), torch.sigmoid(pw[:, :, 3])
    P_o, L_titr, D_o, W_o, E_n, D_n, W_n = [pos(praw[i]) for i in range(7)]
    np_, op_ = n[:, :-1], o[:, :-1]; gv, gp, gr = gv[:, :-1], gp[:, :-1], gr[:, :-1]
    o_hat = op_ + (P_o*gp - L_titr*torch.relu(np_)*torch.sigmoid(op_) - D_o*gv*op_ - W_o*gr*op_)
    n_hat = np_ + (E_n - D_n*gv*np_ - W_n*gr*np_)
    return ((o[:, 1:]-o_hat)**2).mean() + ((n[:, 1:]-n_hat)**2).mean()
def pinball_loss(q, y, tm):
    taus = torch.tensor(TAUS).view(1, 1, nT); err = y.unsqueeze(-1)-q
    return (torch.maximum(taus*err, (taus-1)*err).mean(-1)*tm).sum()/tm.sum().clamp(min=1)

def train_full():
    model = QRPINN(F, S); opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = torch.utils.data.DataLoader(DS(idx_tr), batch_size=BATCH, shuffle=True, num_workers=0)
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = nb = 0; lam = LAM_PHYS_MAX*min(1.0, (ep+1)/ANNEAL)
        for x, si, pw, cobs, cmask, fut, tm in dl:
            opt.zero_grad(); q, chat = model(x, si)
            loss = pinball_loss(q, fut, tm) + LAM_DATA*(((chat-cobs)**2*cmask).sum()/cmask.sum().clamp(min=1))
            loss = loss + lam*physics_residual(chat, pw, model.praw)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [E6 ext] ep {ep+1}/{EPOCHS} loss={tot/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model
def predict(model, idx):
    model.eval(); dl = torch.utils.data.DataLoader(DS(idx), batch_size=1024, shuffle=False, num_workers=0)
    Q, Y, M = [], [], []
    with torch.no_grad():
        for x, si, pw, cobs, cmask, fut, tm in dl:
            q, _ = model(x, si); Q.append(q.numpy()); Y.append(fut.numpy()); M.append(tm.numpy())
    return np.concatenate(Q), np.concatenate(Y), np.concatenate(M)

# ---- CDF helpers ----
def cdf_at_value(Q, yv):
    N, m = Q.shape; k = (Q < yv[:, None]).sum(1); lo = np.clip(k-1, 0, m-1); hi = np.clip(k, 0, m-1); ar = np.arange(N)
    qlo, qhi, tlo, thi = Q[ar, lo], Q[ar, hi], TAUS[lo], TAUS[hi]
    gap = qhi-qlo; den = np.where(gap > 1e-9, gap, 1.0); frac = np.where(gap > 1e-9, (yv-qlo)/den, 0.0)
    cdf = tlo+frac*(thi-tlo); cdf = np.where(k == 0, 0.0, cdf); cdf = np.where(k == m, 1.0, cdf)
    return np.clip(cdf, 0, 1)
def interp_pred_cdf(Q, p):
    j = int(np.searchsorted(TAUS, p, side="left")); j = min(max(j, 1), nT-1)
    t0, t1 = TAUS[j-1], TAUS[j]; w = np.clip((p-t0)/(t1-t0), 0, 1)
    return (1-w)*Q[..., j-1] + w*Q[..., j]

def pinball_pooled(Q, Y, M):
    taus = TAUS.reshape(1, 1, nT); err = Y[..., None]-Q
    pb = np.maximum(taus*err, (taus-1)*err); m3 = M[..., None]
    pin = float((pb*m3).sum()/m3.sum()); crps = float(2*((pb*m3).sum((0, 1))/m3.sum((0, 1))).mean())
    return pin, crps
def picp_width(Q, Y, M, lo_i, hi_i):
    lo, hi = Q[:, :, lo_i], Q[:, :, hi_i]; ins = ((Y >= lo) & (Y <= hi)).astype(float)*M
    return float(ins.sum()/M.sum()), float(((hi-lo)*M).sum()/M.sum())
def tail_pinball(Q, Y, M, tau):
    j = ti(tau); err = Y-Q[:, :, j]; pb = np.maximum(tau*err, (tau-1)*err)
    return float((pb*M).sum()/M.sum())
def brier_exc(Q, Y, M):
    Qf, Yf, Mf = Q.reshape(-1, nT), Y.reshape(-1), M.reshape(-1)
    pexc = 1.0 - cdf_at_value(Qf, np.full(Qf.shape[0], thr)); ind = (Yf >= thr).astype(float); sel = Mf > 0
    return float((((pexc-ind)**2)[sel]).mean())

# ============ train ============
print("\n== train FULL (extended grid, rank index) ==", flush=True); model = train_full()
torch.save(model.state_dict(), ART / "qrpinn_full_rank_ext.pt"); print("saved checkpoint", flush=True)
Qc, Yc, Mc = predict(model, idx_cal)
Qe, Ye, Me = predict(model, idx_eval)

# ============ (2) PIT recalibration (full distribution) ============
selc = Mc.reshape(-1) > 0
u = cdf_at_value(Qc.reshape(-1, nT)[selc], Yc.reshape(-1)[selc])
pstar = np.array([np.quantile(u, t) for t in TAUS], np.float32)
Qe_pit = np.sort(np.stack([interp_pred_cdf(Qe, float(p)) for p in pstar], -1), axis=-1)

# ============ (3) CQR per-horizon, per level ============
def cqr_Q(E, alpha):                       # E: [n] conformity scores -> conformal quantile
    n = len(E); lvl = (1-alpha)*(n+1)/n
    return float(np.max(E)) if lvl >= 1 else float(np.quantile(E, lvl, method="higher"))
cqr = {}
for name, (lo_i, hi_i, alpha) in LEVELS.items():
    Qh = np.zeros(HZ)
    for h in range(HZ):
        m = Mc[:, h] > 0
        E = np.maximum(Qc[m, h, lo_i]-Yc[m, h], Yc[m, h]-Qc[m, h, hi_i])
        Qh[h] = cqr_Q(E, alpha) if m.sum() > 10 else 0.0
    lo = Qe[:, :, lo_i] - Qh[None, :]; hi = Qe[:, :, hi_i] + Qh[None, :]
    ins = ((Ye >= lo) & (Ye <= hi)).astype(float)*Me
    picp = float(ins.sum()/Me.sum()); width = float(((hi-lo)*Me).sum()/Me.sum())
    perh = [float((((Ye[:, h] >= lo[:, h]) & (Ye[:, h] <= hi[:, h])).astype(float)*Me[:, h]).sum()/Me[:, h].sum())
            for h in (0, 5, 11, 23)]
    cqr[name] = {"picp": picp, "width": width, "Qh_mean": float(Qh.mean()), "picp_h_1_6_12_24": perh}

# ============ assemble metrics ============
pin_raw, crps_raw = pinball_pooled(Qe, Ye, Me); pin_pit, crps_pit = pinball_pooled(Qe_pit, Ye, Me)
R = {"config": {"taus": TAUS.tolist(), "epochs": EPOCHS, "n_calib": int(len(idx_cal)),
                "n_eval": int(len(idx_eval)), "thr": thr},
     "raw": {"pinball": pin_raw, "crps": crps_raw, "brier_exceed": brier_exc(Qe, Ye, Me),
             "tail_pinball": {t: tail_pinball(Qe, Ye, Me, t) for t in (0.9, 0.95, 0.99)}},
     "pit": {"pinball": pin_pit, "crps": crps_pit, "brier_exceed": brier_exc(Qe_pit, Ye, Me)},
     "coverage": {}, "cqr": cqr, "pit_recal_map": {float(t): float(p) for t, p in zip(TAUS, pstar)}}
for name, (lo_i, hi_i, alpha) in LEVELS.items():
    pr, wr = picp_width(Qe, Ye, Me, lo_i, hi_i); pp, wp = picp_width(Qe_pit, Ye, Me, lo_i, hi_i)
    R["coverage"][name] = {"nominal": 1-alpha, "raw_picp": pr, "raw_width": wr,
                           "pit_picp": pp, "pit_width": wp,
                           "cqr_picp": cqr[name]["picp"], "cqr_width": cqr[name]["width"]}
json.dump(R, open(ROOT / "results_e6.json", "w"), indent=2)

# ============ figures ============
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def reliab(Q):
        s = Me.reshape(-1) > 0; uu = cdf_at_value(Q.reshape(-1, nT)[s], Ye.reshape(-1)[s])
        return [float(np.mean(uu <= t)) for t in TAUS]
    rb, ra = reliab(Qe), reliab(Qe_pit)
    plt.figure(figsize=(5, 5)); plt.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    plt.plot(TAUS, rb, "o-", label="raw"); plt.plot(TAUS, ra, "s-", label="PIT recal")
    plt.xlabel("nominal"); plt.ylabel("empirical coverage (eval late-2016)")
    plt.title("E6 reliability — rank QR-PINN"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGS / "e6_reliability.png", dpi=130); plt.close()
    names = list(LEVELS); x = np.arange(len(names)); nom = [1-LEVELS[n][2] for n in names]
    plt.figure(figsize=(6, 4))
    plt.bar(x-0.25, [R["coverage"][n]["raw_picp"] for n in names], 0.25, label="raw")
    plt.bar(x, [R["coverage"][n]["pit_picp"] for n in names], 0.25, label="PIT")
    plt.bar(x+0.25, [R["coverage"][n]["cqr_picp"] for n in names], 0.25, label="CQR")
    plt.plot(x, nom, "k_", ms=22, label="nominal")
    plt.xticks(x, [f"{int((1-LEVELS[n][2])*100)}%" for n in names]); plt.ylabel("PICP (eval)")
    plt.title("E6 interval coverage: raw vs PIT vs CQR"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGS / "e6_coverage.png", dpi=130); plt.close()
    print("saved figs/e6_reliability.png, figs/e6_coverage.png", flush=True)
except Exception as e:
    print("figure skipped:", e, flush=True)

# ============ console + log ============
print("\n=============== E6 (eval late-2016) ===============")
print(f"raw  pinball={pin_raw:.4f} crps={crps_raw:.4f} brier={R['raw']['brier_exceed']:.4f} "
      f"tailPB(.9/.95/.99)={[round(R['raw']['tail_pinball'][t],4) for t in (0.9,0.95,0.99)]}")
print(f"pit  pinball={pin_pit:.4f} crps={crps_pit:.4f} brier={R['pit']['brier_exceed']:.4f}")
print(f"{'level':6s} {'nominal':>7s} {'raw':>6s} {'PIT':>6s} {'CQR':>6s} | widths raw/PIT/CQR")
for n in LEVELS:
    c = R["coverage"][n]
    print(f"{n:6s} {c['nominal']:7.2f} {c['raw_picp']:6.3f} {c['pit_picp']:6.3f} {c['cqr_picp']:6.3f} | "
          f"{c['raw_width']:.3f}/{c['pit_width']:.3f}/{c['cqr_width']:.3f}")
print("CQR PICP90 by lead (h=1,6,12,24):", [round(v, 3) for v in cqr["90"]["picp_h_1_6_12_24"]])

run_id = "E6-" + datetime.now().strftime("%Y%m%d-%H%M%S")
lines = [f"\n## Run {run_id}  (E6: tail calibration — extended grid + PIT + CQR, rank index)",
         f"- date: {datetime.now().isoformat(timespec='seconds')} | code: pipeline/07_qrpinn_e6.py | method: METHODS_calibration.md",
         f"- tau grid (13): {TAUS.tolist()}",
         f"- split: train 2014-15 ({len(idx_tr)}); calib early-2016 ({len(idx_cal)}); eval late-2016 ({len(idx_eval)}); seed=0",
         f"- raw: pinball={pin_raw:.4f} CRPS={crps_raw:.4f} Brier(exc)={R['raw']['brier_exceed']:.4f}; "
         f"tail pinball .9/.95/.99 = " + "/".join(f"{R['raw']['tail_pinball'][t]:.4f}" for t in (0.9, 0.95, 0.99)),
         "", "| level | nominal | PICP raw | PICP PIT | PICP CQR | width raw | width PIT | width CQR |",
         "|---|---|---|---|---|---|---|---|"]
for n in LEVELS:
    c = R["coverage"][n]
    lines.append(f"| {n}% | {c['nominal']:.2f} | {c['raw_picp']:.3f} | {c['pit_picp']:.3f} | {c['cqr_picp']:.3f} | "
                 f"{c['raw_width']:.3f} | {c['pit_width']:.3f} | {c['cqr_width']:.3f} |")
lines += ["", f"- CQR PICP90 by lead h=1/6/12/24: " + "/".join(f"{v:.3f}" for v in cqr['90']['picp_h_1_6_12_24']),
          f"- checkpoint artefacts/qrpinn_full_rank_ext.pt ; figs e6_reliability.png, e6_coverage.png",
          f"- NOTE: split-CQR coverage is approximate under time-series non-exchangeability "
          f"(METHODS_calibration.md sec.3); ACI/EnbPI = future work."]
with open(ROOT / "RESULTS_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines)+"\n")
print(f"\nWrote results_e6.json and appended {run_id} to RESULTS_LOG.md", flush=True)
