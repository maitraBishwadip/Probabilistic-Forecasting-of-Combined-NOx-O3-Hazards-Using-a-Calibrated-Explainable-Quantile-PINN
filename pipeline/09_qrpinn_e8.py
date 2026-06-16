# -*- coding: utf-8 -*-
"""
09_qrpinn_e8.py  (E8)  -- Robustness checks for the adopted rank-index QR-PINN.

Three robustness questions (temporal split: train 2014-15 / test 2016; rank index H):
  (A) PHYSICS-WEIGHT SWEEP  lam_phys in {0.0, 0.2, 0.5}  (seed 0) -> is the physics gain stable?
  (B) SEED STABILITY        lam_phys=0.2, seed in {0, 1}          -> are headline numbers stable?
  (C) THRESHOLD SENSITIVITY exceedance Brier/base-rate at thr=Q25/Q30/Q35 of train H
                            (NO retraining: the model predicts the full H distribution;
                             the cutoff only enters the exceedance metric).

Model = E5 architecture (rank index, station embedding, coupled NOx-O3 physics). 8 epochs each.
Outputs: results_e8.json, append RESULTS_LOG.md.  Observed H only; no synthetic data.
"""
import json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"
L, HZ = 48, 24
TAUS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95], dtype=np.float32); nT = len(TAUS)
EPOCHS, BATCH, LR = 8, 512, 1e-3
LAM_DATA, ANNEAL = 0.3, 5

d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
X = d["X"].astype(np.float32); PHYS = d["PHYS"].astype(np.float32)
COBS = d["COBS"].astype(np.float32); CMASK = d["CMASK"].astype(np.float32)
Hrob = d["H"].astype(np.float32); year = d["year"].astype(np.int32); S, T, F = X.shape
both = (CMASK[:, :, 0] == 1) & (CMASK[:, :, 1] == 1) & (~np.isnan(Hrob))
yr2d = np.broadcast_to(year[None, :], (S, T)); tr_both = both & (yr2d <= 2015)
sN = np.sort(COBS[:, :, 0][tr_both]); sO = np.sort(COBS[:, :, 1][tr_both])
def Fcdf(s, v): return np.searchsorted(s, v, side="right")/len(s)
H = np.full((S, T), np.nan, np.float32)
H[both] = (0.5*Fcdf(sN, COBS[:, :, 0][both]) + 0.5*Fcdf(sO, COBS[:, :, 1][both])).astype(np.float32)
THR = {q: float(np.quantile(H[tr_both], q)) for q in (0.25, 0.30, 0.35)}
print("thresholds:", {k: round(v, 4) for k, v in THR.items()}, flush=True)

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
print(f"train={len(idx_tr)} test={len(idx_te)}", flush=True)

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

def train(lam_phys, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = QRPINN(F, S); opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = torch.utils.data.DataLoader(DS(idx_tr), batch_size=BATCH, shuffle=True, num_workers=0)
    for ep in range(EPOCHS):
        model.train(); lam = lam_phys*min(1.0, (ep+1)/ANNEAL)
        for x, si, pw, cobs, cmask, fut, tm in dl:
            opt.zero_grad(); q, chat = model(x, si)
            loss = pinball_loss(q, fut, tm) + LAM_DATA*(((chat-cobs)**2*cmask).sum()/cmask.sum().clamp(min=1))
            if lam_phys > 0: loss = loss + lam*physics_residual(chat, pw, model.praw)
            loss.backward(); opt.step()
    return model
def predict(model):
    model.eval(); dl = torch.utils.data.DataLoader(DS(idx_te), batch_size=1024, shuffle=False, num_workers=0)
    Q, Y, M = [], [], []
    with torch.no_grad():
        for x, si, pw, cobs, cmask, fut, tm in dl:
            q, _ = model(x, si); Q.append(q.numpy()); Y.append(fut.numpy()); M.append(tm.numpy())
    return np.concatenate(Q), np.concatenate(Y), np.concatenate(M)
def cdf_at_thr(Q, thr):
    N, m = Q.shape; k = (Q < thr).sum(1); lo = np.clip(k-1, 0, m-1); hi = np.clip(k, 0, m-1); ar = np.arange(N)
    qlo, qhi, tlo, thi = Q[ar, lo], Q[ar, hi], TAUS[lo], TAUS[hi]
    gap = qhi-qlo; den = np.where(gap > 1e-9, gap, 1.0); frac = np.where(gap > 1e-9, (thr-qlo)/den, 0.0)
    cdf = tlo+frac*(thi-tlo); cdf = np.where(k == 0, TAUS[0], cdf); cdf = np.where(k == m, TAUS[-1], cdf)
    return np.clip(cdf, 0, 1)
def metrics(Q, Y, M):
    taus = TAUS.reshape(1, 1, nT); err = Y[..., None]-Q; pb = np.maximum(taus*err, (taus-1)*err); m3 = M[..., None]
    pin = float((pb*m3).sum()/m3.sum()); crps = float(2*((pb*m3).sum((0, 1))/m3.sum((0, 1))).mean())
    def cov(a, b):
        lo, hi = Q[:, :, a], Q[:, :, b]; return float((((Y >= lo)&(Y <= hi)).astype(float)*M).sum()/M.sum())
    return {"pinball": pin, "crps": crps, "picp80": cov(1, 7), "picp90": cov(0, 8)}
def brier_at(Q, Y, M, thr):
    Qf, Yf, Mf = Q.reshape(-1, nT), Y.reshape(-1), M.reshape(-1)
    pexc = 1.0-cdf_at_thr(Qf, thr); ind = (Yf >= thr).astype(float); sel = Mf > 0
    return float((((pexc-ind)**2)[sel]).mean()), float(ind[sel].mean())

R = {"config": {"epochs": EPOCHS, "thresholds": THR}, "physics_sweep": {}, "seed_stability": {}, "threshold_sensitivity": {}}

# (A) physics-weight sweep (seed 0)
preds_ref = None
for lam in (0.0, 0.2, 0.5):
    t0 = time.time(); m = train(lam, 0); Q, Y, M = predict(m); met = metrics(Q, Y, M)
    met["brier_Q30"] = brier_at(Q, Y, M, THR[0.30])[0]
    R["physics_sweep"][f"lam_{lam}"] = met
    if lam == 0.2: preds_ref = (Q, Y, M)
    print(f"[A] lam={lam} {met} ({time.time()-t0:.0f}s)", flush=True)

# (B) seed stability (lam=0.2, seed 1) -- seed 0 already in sweep
t0 = time.time(); m1 = train(0.2, 1); Q1, Y1, M1 = predict(m1); met1 = metrics(Q1, Y1, M1)
met1["brier_Q30"] = brier_at(Q1, Y1, M1, THR[0.30])[0]
R["seed_stability"] = {"seed0": R["physics_sweep"]["lam_0.2"], "seed1": met1}
print(f"[B] seed1 {met1} ({time.time()-t0:.0f}s)", flush=True)

# (C) threshold sensitivity on the lam=0.2 seed0 model
Q, Y, M = preds_ref
for q in (0.25, 0.30, 0.35):
    b, base = brier_at(Q, Y, M, THR[q]); R["threshold_sensitivity"][f"Q{int(q*100)}"] = {"thr": THR[q], "brier": b, "test_base_rate": base}
json.dump(R, open(ROOT / "results_e8.json", "w"), indent=2)

# console + log
print("\n=============== E8 ROBUSTNESS (test 2016) ===============")
print("(A) physics-weight sweep:")
for lam in (0.0, 0.2, 0.5):
    v = R["physics_sweep"][f"lam_{lam}"]; print(f"   lam={lam}: pinball={v['pinball']:.4f} crps={v['crps']:.4f} picp90={v['picp90']:.3f} brierQ30={v['brier_Q30']:.4f}")
print("(B) seed stability (lam=0.2):")
for s in ("seed0", "seed1"):
    v = R["seed_stability"][s]; print(f"   {s}: pinball={v['pinball']:.4f} crps={v['crps']:.4f} picp90={v['picp90']:.3f} brierQ30={v['brier_Q30']:.4f}")
print("(C) threshold sensitivity (lam=0.2 seed0):")
for q in (25, 30, 35):
    v = R["threshold_sensitivity"][f"Q{q}"]; print(f"   Q{q}: thr={v['thr']:.3f} base_rate={v['test_base_rate']:.3f} brier={v['brier']:.4f}")

run_id = "E8-" + datetime.now().strftime("%Y%m%d-%H%M%S")
sw = R["physics_sweep"]; ss = R["seed_stability"]; tsv = R["threshold_sensitivity"]
lines = [f"\n## Run {run_id}  (E8: robustness — physics-weight sweep, seed, threshold)",
         f"- date: {datetime.now().isoformat(timespec='seconds')} | code: pipeline/09_qrpinn_e8.py | rank index; epochs={EPOCHS}",
         "", "**(A) Physics-weight sweep (seed 0)**", "| lam_phys | pinball | CRPS | PICP90 | Brier(Q30) |", "|---|---|---|---|---|"]
for lam in (0.0, 0.2, 0.5):
    v = sw[f"lam_{lam}"]; lines.append(f"| {lam} | {v['pinball']:.4f} | {v['crps']:.4f} | {v['picp90']:.3f} | {v['brier_Q30']:.4f} |")
lines += ["", "**(B) Seed stability (lam=0.2)**", "| seed | pinball | CRPS | PICP90 | Brier(Q30) |", "|---|---|---|---|---|",
          f"| 0 | {ss['seed0']['pinball']:.4f} | {ss['seed0']['crps']:.4f} | {ss['seed0']['picp90']:.3f} | {ss['seed0']['brier_Q30']:.4f} |",
          f"| 1 | {ss['seed1']['pinball']:.4f} | {ss['seed1']['crps']:.4f} | {ss['seed1']['picp90']:.3f} | {ss['seed1']['brier_Q30']:.4f} |",
          "", "**(C) Threshold sensitivity (lam=0.2 seed0; exceedance only, no retrain)**",
          "| cutoff | thr(H) | test base-rate | Brier(exceed) |", "|---|---|---|---|"]
for q in (25, 30, 35):
    v = tsv[f"Q{q}"]; lines.append(f"| Q{q} | {v['thr']:.3f} | {v['test_base_rate']:.3f} | {v['brier']:.4f} |")
lines += ["", "- NOTE: rank-index metrics on [0,1]; internal comparisons only. 8 epochs (robustness scan)."]
with open(ROOT / "RESULTS_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines)+"\n")
print(f"\nWrote results_e8.json and appended {run_id} to RESULTS_LOG.md", flush=True)
