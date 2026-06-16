# -*- coding: utf-8 -*-
"""
05_qrpinn_e4.py  (E4)
Calibration of the QR-PINN (FULL) predictive distribution.

Problem from E1-E3: intervals are under-dispersed (PICP < nominal).
Method: PIT-based distribution recalibration (Kuleshov, Fenner & Ermon 2018).
  * Retrain FULL (seed 0, same config), save checkpoint.
  * Split TEST year 2016 temporally: calibration = earliest 40% anchors, eval = latest 60%.
    (calibration set is held out from training AND from the final eval => honest.)
  * On calibration, compute PIT u = F_pred(y); recalibration shifts each nominal level tau to
    p* = empirical tau-quantile of u, and reads the predicted CDF at p*.
  * Report PICP/width/Brier/pinball BEFORE vs AFTER on the eval half.

Outputs: results_e4.json, artefacts/qrpinn_full.pt, figs/e4_reliability.png, append RESULTS_LOG.md
No synthetic data: observed targets only.
"""
import json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime

ROOT = Path(r"d:\BUET RESEARCH WORK\Multiple pollutant combine impact  study in Bangladesh")
ART = ROOT / "artefacts"; FIGS = ROOT / "figs"; FIGS.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

L, HZ = 48, 24
TAUS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95], dtype=np.float32)
nT = len(TAUS)
EPOCHS, BATCH, LR = 12, 512, 1e-3
LAM_DATA, LAM_PHYS_MAX, ANNEAL = 0.3, 0.2, 6

d = np.load(ART / "qrpinn_data.npz", allow_pickle=True)
X = d["X"].astype(np.float32); PHYS = d["PHYS"].astype(np.float32)
COBS = d["COBS"].astype(np.float32); CMASK = d["CMASK"].astype(np.float32)
H = d["H"].astype(np.float32); year = d["year"].astype(np.int32); thr = float(d["thr"][0])
S, T, F = X.shape
print(f"Loaded X{X.shape} thr={thr:.4f}", flush=True)

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
idx_te = idx_te[np.argsort(idx_te[:, 1], kind="stable")]      # chronological by anchor hour
ncal = int(0.4*len(idx_te)); idx_cal, idx_eval = idx_te[:ncal], idx_te[ncal:]
print(f"anchors train={len(idx_tr)} calib={len(idx_cal)} eval={len(idx_eval)}", flush=True)

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
        model.train(); t0 = time.time(); tot = nb = 0
        lam = LAM_PHYS_MAX*min(1.0, (ep+1)/ANNEAL)
        for x, si, pw, cobs, cmask, fut, tm in dl:
            opt.zero_grad(); q, chat = model(x, si)
            loss = pinball_loss(q, fut, tm) + LAM_DATA*(((chat-cobs)**2*cmask).sum()/cmask.sum().clamp(min=1))
            loss = loss + lam*physics_residual(chat, pw, model.praw)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [FULL] ep {ep+1}/{EPOCHS} loss={tot/nb:.4f} lam={lam:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return model

def predict(model, idx):
    model.eval(); dl = torch.utils.data.DataLoader(DS(idx), batch_size=1024, shuffle=False, num_workers=0)
    Q, Y, M = [], [], []
    with torch.no_grad():
        for x, si, pw, cobs, cmask, fut, tm in dl:
            q, _ = model(x, si); Q.append(q.numpy()); Y.append(fut.numpy()); M.append(tm.numpy())
    return np.concatenate(Q), np.concatenate(Y), np.concatenate(M)

# ---- CDF helpers ----
def cdf_at_value(Q, yv):                      # predicted CDF evaluated at per-sample value yv
    N, m = Q.shape; k = (Q < yv[:, None]).sum(1); lo = np.clip(k-1, 0, m-1); hi = np.clip(k, 0, m-1); ar = np.arange(N)
    qlo, qhi, tlo, thi = Q[ar, lo], Q[ar, hi], TAUS[lo], TAUS[hi]
    gap = qhi-qlo; den = np.where(gap > 1e-9, gap, 1.0); frac = np.where(gap > 1e-9, (yv-qlo)/den, 0.0)
    cdf = tlo+frac*(thi-tlo); cdf = np.where(k == 0, 0.0, cdf); cdf = np.where(k == m, 1.0, cdf)
    return np.clip(cdf, 0, 1)
def cdf_at_thr(Q, thr):                       # predicted CDF at a scalar threshold
    return cdf_at_value(Q, np.full(Q.shape[0], thr))
def interp_pred_cdf(Q, p):                    # predicted quantile value at scalar CDF level p
    j = int(np.searchsorted(TAUS, p, side="left")); j = min(max(j, 1), nT-1)
    t0, t1 = TAUS[j-1], TAUS[j]; w = np.clip((p-t0)/(t1-t0), 0, 1)
    return (1-w)*Q[..., j-1] + w*Q[..., j]

def metrics(Q, Y, M):
    taus = TAUS.reshape(1, 1, nT); err = Y[..., None]-Q
    pb = np.maximum(taus*err, (taus-1)*err); m3 = M[..., None]
    pin = float((pb*m3).sum()/m3.sum())
    def cov(a, b):
        lo, hi = Q[:, :, a], Q[:, :, b]; ins = ((Y >= lo)&(Y <= hi)).astype(float)*M
        return float(ins.sum()/M.sum()), float(((hi-lo)*M).sum()/M.sum())
    p80, w80 = cov(1, 7); p90, w90 = cov(0, 8)
    Qf, Yf, Mf = Q.reshape(-1, nT), Y.reshape(-1), M.reshape(-1)
    pexc = 1.0-cdf_at_thr(Qf, thr); ind = (Yf >= thr).astype(float); sel = Mf > 0
    brier = float((((pexc-ind)**2)[sel]).mean())
    return {"pinball": pin, "crps": float(2*((pb*m3).sum((0, 1))/m3.sum((0, 1))).mean()),
            "picp80": p80, "width80": w80, "picp90": p90, "width90": w90, "brier_exceed": brier}

# ============ run ============
print("\n== train FULL ==", flush=True); model = train_full()
torch.save(model.state_dict(), ART / "qrpinn_full.pt"); print("saved checkpoint", flush=True)
Qc, Yc, Mc = predict(model, idx_cal)
Qe, Ye, Me = predict(model, idx_eval)

# ---- fit recalibration map on calibration PIT ----
sel = Mc.reshape(-1) > 0
u = cdf_at_value(Qc.reshape(-1, nT)[sel], Yc.reshape(-1)[sel])     # PIT values
pstar = np.array([np.quantile(u, t) for t in TAUS], dtype=np.float32)  # nominal tau -> adjusted level
print("recalibration map (nominal -> p*):", {float(t): round(float(p), 3) for t, p in zip(TAUS, pstar)}, flush=True)

# ---- apply to eval: recalibrated quantiles ----
Qe_cal = np.stack([interp_pred_cdf(Qe, float(p)) for p in pstar], axis=-1)  # [N,HZ,nT]
Qe_cal = np.sort(Qe_cal, axis=-1)                                            # enforce monotone

before = metrics(Qe, Ye, Me); after = metrics(Qe_cal, Ye, Me)
R = {"config": {"epochs": EPOCHS, "calib_frac": 0.4, "n_calib": int(len(idx_cal)),
                "n_eval": int(len(idx_eval)), "thr": thr},
     "recal_map_nominal_to_pstar": {float(t): float(p) for t, p in zip(TAUS, pstar)},
     "before": before, "after": after}
json.dump(R, open(ROOT / "results_e4.json", "w"), indent=2)

# ---- reliability figure ----
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def reliab(Q, Y, M):
        s = M.reshape(-1) > 0; uu = cdf_at_value(Q.reshape(-1, nT)[s], Y.reshape(-1)[s])
        return [float(np.mean(uu <= t)) for t in TAUS]
    rb, ra = reliab(Qe, Ye, Me), reliab(Qe_cal, Ye, Me)
    plt.figure(figsize=(5, 5)); plt.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    plt.plot(TAUS, rb, "o-", label="before"); plt.plot(TAUS, ra, "s-", label="after recal")
    plt.xlabel("nominal coverage"); plt.ylabel("empirical coverage (eval 2016)")
    plt.title("QR-PINN reliability — PIT recalibration"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGS / "e4_reliability.png", dpi=130); plt.close(); print("saved figs/e4_reliability.png", flush=True)
except Exception as e:
    print("figure skipped:", e, flush=True)

# ---- console + log ----
print("\n=============== E4 (eval half of 2016) ===============")
print(f"{'':8s} {'pinball':>8s} {'CRPS':>7s} {'PICP80':>7s} {'PICP90':>7s} {'width80':>8s} {'Brier':>7s}")
for nm, v in [("before", before), ("after", after)]:
    print(f"{nm:8s} {v['pinball']:8.4f} {v['crps']:7.4f} {v['picp80']:7.3f} {v['picp90']:7.3f} {v['width80']:8.3f} {v['brier_exceed']:7.4f}")
print("(nominal coverage targets: 0.80 / 0.90)")

run_id = "E4-" + datetime.now().strftime("%Y%m%d-%H%M%S")
lines = [f"\n## Run {run_id}  (E4: PIT recalibration of FULL)",
         f"- date: {datetime.now().isoformat(timespec='seconds')} | code: pipeline/05_qrpinn_e4.py",
         f"- split: train 2014-15 ({len(idx_tr)}); calib=early-2016 ({len(idx_cal)}); eval=late-2016 ({len(idx_eval)}); seed=0",
         f"- method: PIT distribution recalibration (Kuleshov et al. 2018) on the FULL QR-PINN", "",
         "| state | pinball | CRPS | PICP80 | PICP90 | width80 | Brier(exceed) |",
         "|---|---|---|---|---|---|---|",
         f"| before | {before['pinball']:.4f} | {before['crps']:.4f} | {before['picp80']:.3f} | {before['picp90']:.3f} | {before['width80']:.3f} | {before['brier_exceed']:.4f} |",
         f"| after  | {after['pinball']:.4f} | {after['crps']:.4f} | {after['picp80']:.3f} | {after['picp90']:.3f} | {after['width80']:.3f} | {after['brier_exceed']:.4f} |",
         "", f"- recalibration map (nominal->p*): " + ", ".join(f"{float(t):.2f}->{float(p):.2f}" for t, p in zip(TAUS, pstar)),
         f"- checkpoint: artefacts/qrpinn_full.pt; reliability fig: figs/e4_reliability.png",
         f"- NOTE: nominal coverage targets PICP80=0.80, PICP90=0.90; eval is the held-out late-2016 half."]
with open(ROOT / "RESULTS_LOG.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines)+"\n")
print(f"\nWrote results_e4.json and appended {run_id} to RESULTS_LOG.md", flush=True)
